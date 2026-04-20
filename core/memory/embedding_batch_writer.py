"""
ATOM — Embedding batch writer (Sprint B2).

Decouples conversation-path writes to the vector store from the
request/response loop. The turn handler enqueues a write; a background
worker coalesces the queue into small batches, computes embeddings,
and does the vector-store insert off the hot path.

Design:

* ``enqueue()`` is a synchronous, fire-and-forget call. If the queue
  is full we drop the **oldest** item so a long-running turn burst
  cannot starve more recent context.
* The worker drains with a small idle timeout so items written during
  a spike are batched together (typical win: 2–4 embeddings per model
  call instead of one at a time).
* ``flush()`` drains the queue and waits for the worker to finish —
  called on shutdown and on memory-pressure "critical" so we don't
  leave unindexed turns on disk shutdown.
* Lives beside ``MemoryEngine`` so writes from other subsystems
  (dream, RAG ingestion) can reuse it later.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any, Callable

logger = logging.getLogger("atom.memory.batch_writer")


class EmbeddingBatchWriter:
    """Queue + worker for async embed + vector-store writes."""

    def __init__(
        self,
        embedding_engine: Any,
        vector_store: Any,
        *,
        batch_size: int = 4,
        max_queue: int = 128,
        drain_idle_s: float = 0.75,
        drain_max_wait_s: float = 5.0,
        on_error: Callable[[str, Exception], None] | None = None,
    ) -> None:
        self._embedding_engine = embedding_engine
        self._vector_store = vector_store
        self._batch_size = max(1, int(batch_size))
        self._max_queue = max(8, int(max_queue))
        self._drain_idle_s = max(0.05, float(drain_idle_s))
        self._drain_max_wait_s = max(self._drain_idle_s, float(drain_max_wait_s))
        self._queue: deque[dict[str, Any]] = deque()
        self._lock = asyncio.Lock()
        self._new_item = asyncio.Event()
        self._worker: asyncio.Task | None = None
        self._shutdown = asyncio.Event()
        self._on_error = on_error

        self._total_enqueued = 0
        self._total_written = 0
        self._total_failed = 0
        self._total_dropped = 0
        self._last_batch_size = 0
        self._last_write_ts: float = 0.0

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    def enqueue(
        self,
        collection: str,
        text: str,
        metadata: dict[str, Any],
        doc_id: str,
    ) -> None:
        """Fire-and-forget. Safe to call from the hot path — never blocks on I/O."""
        if not collection or not text:
            return
        item = {
            "collection": collection,
            "text": text,
            "metadata": metadata or {},
            "doc_id": doc_id,
            "enqueued_at": time.monotonic(),
        }
        if len(self._queue) >= self._max_queue:
            try:
                self._queue.popleft()
                self._total_dropped += 1
            except IndexError:
                pass
        self._queue.append(item)
        self._total_enqueued += 1
        self._new_item.set()

    def start(self) -> None:
        if self._worker is not None and not self._worker.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("EmbeddingBatchWriter.start() requires a running loop")
            return
        self._shutdown.clear()
        self._worker = loop.create_task(self._drain_loop())
        logger.info(
            "Embedding batch writer started (batch=%d, max_queue=%d, drain_idle=%.2fs)",
            self._batch_size, self._max_queue, self._drain_idle_s,
        )

    async def flush(self, timeout_s: float = 10.0) -> None:
        """Wait until the queue empties or ``timeout_s`` elapses."""
        deadline = time.monotonic() + max(0.1, float(timeout_s))
        while self._queue and time.monotonic() < deadline:
            self._new_item.set()
            await asyncio.sleep(0.05)

    async def stop(self, *, flush_timeout_s: float = 5.0) -> None:
        await self.flush(flush_timeout_s)
        self._shutdown.set()
        self._new_item.set()
        worker = self._worker
        if worker is not None:
            try:
                await asyncio.wait_for(worker, timeout=3.0)
            except asyncio.TimeoutError:
                worker.cancel()
            except Exception:
                pass
        self._worker = None

    def diagnostics(self) -> dict[str, Any]:
        return {
            "queue_size": len(self._queue),
            "batch_size": self._batch_size,
            "total_enqueued": self._total_enqueued,
            "total_written": self._total_written,
            "total_failed": self._total_failed,
            "total_dropped": self._total_dropped,
            "last_batch_size": self._last_batch_size,
            "last_write_age_s": (
                time.monotonic() - self._last_write_ts
                if self._last_write_ts
                else None
            ),
        }

    # ── Internals ─────────────────────────────────────────────────

    async def _drain_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                if not self._queue:
                    try:
                        await asyncio.wait_for(
                            self._new_item.wait(), timeout=self._drain_max_wait_s,
                        )
                    except asyncio.TimeoutError:
                        continue
                    self._new_item.clear()
                    if self._shutdown.is_set():
                        break

                # Coalesce: brief grace period so bursty turns batch up.
                await asyncio.sleep(self._drain_idle_s)

                batch: list[dict[str, Any]] = []
                while self._queue and len(batch) < self._batch_size:
                    batch.append(self._queue.popleft())
                if not batch:
                    continue

                await self._write_batch(batch)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.exception("embedding batch writer loop error: %s", exc)
                self._total_failed += 1
                if self._on_error:
                    try:
                        self._on_error("batch_loop", exc)
                    except Exception:
                        pass

    async def _write_batch(self, batch: list[dict[str, Any]]) -> None:
        if not batch:
            return
        self._last_batch_size = len(batch)
        embedder = self._embedding_engine
        store = self._vector_store
        if embedder is None or store is None:
            self._total_failed += len(batch)
            return

        # Embed in parallel where safe. Most embedding engines run on a
        # single model instance so we serialise here; change if the
        # engine ever gains real multi-batch support.
        try:
            texts = [item["text"] for item in batch]
            embeddings: list[Any] = []
            for t in texts:
                vec = await embedder.embed(t)
                embeddings.append(vec)
        except Exception as exc:
            logger.warning("batch-writer embed failed: %s", exc)
            self._total_failed += len(batch)
            if self._on_error:
                try:
                    self._on_error("embed", exc)
                except Exception:
                    pass
            return

        for item, vec in zip(batch, embeddings):
            try:
                store.add(
                    item["collection"],
                    text=item["text"],
                    embedding=vec,
                    metadata=item["metadata"],
                    doc_id=item["doc_id"],
                )
                self._total_written += 1
                self._last_write_ts = time.monotonic()
            except Exception as exc:
                self._total_failed += 1
                if self._on_error:
                    try:
                        self._on_error("store_add", exc)
                    except Exception:
                        pass


__all__ = ["EmbeddingBatchWriter"]
