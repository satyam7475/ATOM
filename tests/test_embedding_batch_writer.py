"""Focused tests for Sprint B2: embedding batch writer."""

from __future__ import annotations

import asyncio
import unittest
from typing import Any


class _FakeEmbedder:
    def __init__(self, delay_s: float = 0.0) -> None:
        self._delay = delay_s
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        return [float(len(text))]


class _FakeStore:
    def __init__(self) -> None:
        self.writes: list[dict[str, Any]] = []

    def add(self, collection: str, *, text: str, embedding: Any, metadata: dict, doc_id: str) -> None:
        self.writes.append({
            "collection": collection,
            "text": text,
            "embedding": embedding,
            "metadata": metadata,
            "doc_id": doc_id,
        })


class EmbeddingBatchWriterTests(unittest.TestCase):
    def test_drains_queue_and_writes(self) -> None:
        async def _run() -> None:
            from core.memory.embedding_batch_writer import EmbeddingBatchWriter

            embedder = _FakeEmbedder()
            store = _FakeStore()
            w = EmbeddingBatchWriter(
                embedder, store, batch_size=3, max_queue=32, drain_idle_s=0.05,
            )
            w.start()
            for i in range(5):
                w.enqueue("conversations", f"hello {i}", {"i": i}, f"doc_{i}")
            await w.flush(timeout_s=2.0)
            await w.stop()

            self.assertEqual(len(store.writes), 5)
            self.assertEqual(embedder.calls, 5)
            diag = w.diagnostics()
            self.assertEqual(diag["total_written"], 5)
            self.assertEqual(diag["total_failed"], 0)
            self.assertGreaterEqual(diag["last_batch_size"], 1)

        asyncio.run(_run())

    def test_drops_oldest_on_overflow(self) -> None:
        async def _run() -> None:
            from core.memory.embedding_batch_writer import EmbeddingBatchWriter

            embedder = _FakeEmbedder()
            store = _FakeStore()
            w = EmbeddingBatchWriter(
                embedder, store, batch_size=2, max_queue=10, drain_idle_s=0.05,
            )
            # Fill past capacity WITHOUT a worker, so overflow triggers.
            for i in range(20):
                w.enqueue("conversations", f"msg {i}", {"i": i}, f"id_{i}")
            diag = w.diagnostics()
            self.assertGreaterEqual(diag["total_dropped"], 10)

            w.start()
            await w.flush(timeout_s=2.0)
            await w.stop()
            # Oldest entries dropped; only the latest 10 should reach the store.
            ids = [wr["doc_id"] for wr in store.writes]
            for old in ("id_0", "id_5", "id_9"):
                self.assertNotIn(old, ids)
            for recent in ("id_19", "id_18"):
                self.assertIn(recent, ids)

        asyncio.run(_run())

    def test_embed_failure_counts_as_failed(self) -> None:
        async def _run() -> None:
            from core.memory.embedding_batch_writer import EmbeddingBatchWriter

            class _BadEmbedder:
                calls = 0

                async def embed(self, text: str) -> list[float]:
                    _BadEmbedder.calls += 1
                    raise RuntimeError("embed boom")

            store = _FakeStore()
            w = EmbeddingBatchWriter(_BadEmbedder(), store, batch_size=2, drain_idle_s=0.05)
            w.start()
            w.enqueue("conversations", "hello", {}, "doc_0")
            w.enqueue("conversations", "world", {}, "doc_1")
            await w.flush(timeout_s=1.5)
            await w.stop()

            diag = w.diagnostics()
            self.assertEqual(diag["total_written"], 0)
            self.assertGreaterEqual(diag["total_failed"], 2)
            self.assertEqual(store.writes, [])

        asyncio.run(_run())

    def test_stop_without_start_is_safe(self) -> None:
        async def _run() -> None:
            from core.memory.embedding_batch_writer import EmbeddingBatchWriter

            w = EmbeddingBatchWriter(_FakeEmbedder(), _FakeStore())
            await w.stop()

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
