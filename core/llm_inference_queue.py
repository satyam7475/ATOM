"""
ATOM — Single-slot LLM inference queue.

Runs one local LLM job at a time. Queue depth = 1: new submissions replace
the pending job (coalesce). Work runs in this asyncio task — not inside the
short (10s) AsyncEventBus handler path.

Integration: cursor_query handler only calls submit(); worker calls
LocalBrainController.on_query().
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.metrics import MetricsCollector

logger = logging.getLogger("atom.llm_queue")


class LLMInferenceQueue:
    """Serializes local LLM work; coalesces backlog to latest query."""

    __slots__ = (
        "_bus", "_metrics", "_brain", "_pending", "_lock", "_wake",
        "_shutdown", "_worker_task",
    )

    def __init__(
        self,
        bus: Any,
        metrics: "MetricsCollector | None" = None,
    ) -> None:
        self._bus = bus
        self._metrics = metrics
        self._brain = None
        self._pending: dict[str, Any] | None = None
        self._lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._shutdown = False
        self._worker_task: asyncio.Task | None = None

    def attach_brain(self, brain: Any) -> None:
        self._brain = brain

    def start(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._shutdown = False
            self._worker_task = asyncio.create_task(
                self._worker_loop(), name="atom_llm_queue_worker"
            )
            logger.info("LLM inference queue worker started")

    async def shutdown(self) -> None:
        self._shutdown = True
        self._wake.set()
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        self._worker_task = None
        logger.info("LLM inference queue stopped")

    def has_pending(self) -> bool:
        """Best-effort: True if a job is waiting for the worker (may race)."""
        return self._pending is not None

    async def clear_pending(self) -> None:
        """Drop the coalesced next job (e.g. user interrupted before it runs)."""
        async with self._lock:
            self._pending = None
        if self._metrics is not None:
            self._metrics.inc("llm_queue_cleared_interrupt")

    async def submit(
        self,
        text: str,
        *,
        memory_context: list[str] | None = None,
        context: dict[str, str] | None = None,
        history: list[tuple[str, str]] | None = None,
        query_plan: Any | None = None,
        repeat_hint: bool = False,
    ) -> None:
        """Enqueue (or replace) one job. Returns immediately.

        ``repeat_hint`` propagates from the router (set when Boss is asking
        the same thing again) so the prompt builder can inject a steer in
        the system layer telling the LLM to reformulate from a fresh angle.
        Without forwarding this flag the JARVIS-feel "I asked already" loop
        regresses into ATOM repeating the same wording verbatim.
        """
        async with self._lock:
            if self._pending is not None and self._metrics is not None:
                self._metrics.inc("llm_queue_coalesced")
            self._pending = {
                "text": text,
                "memory_context": memory_context,
                "context": context,
                "history": history or [],
                "query_plan": query_plan,
                "repeat_hint": repeat_hint,
            }
        self._wake.set()

    async def _worker_loop(self) -> None:
        while not self._shutdown:
            await self._wake.wait()
            self._wake.clear()
            while not self._shutdown:
                async with self._lock:
                    job = self._pending
                    self._pending = None
                if job is None:
                    break
                brain = self._brain
                if brain is None or not brain.available:
                    self._bus.emit_long(
                        "response_ready",
                        text="Local brain is not ready, Boss.",
                    )
                    continue
                try:
                    await brain.on_query(
                        job["text"],
                        memory_context=job.get("memory_context"),
                        context=job.get("context"),
                        history=job.get("history") or [],
                        query_plan=job.get("query_plan"),
                        repeat_hint=bool(job.get("repeat_hint", False)),
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("LLM queue worker: on_query failed")
                    self._bus.emit_long(
                        "response_ready",
                        text="Local brain hit an error, Boss. Check the log and try again.",
                    )
