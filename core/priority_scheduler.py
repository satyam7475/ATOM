"""
ATOM — Single-worker priority job queue (voice > LLM > background).

Voice and LLM work is submitted here so bus handlers return quickly and
user input is not stuck behind long jobs. LLM serialization remains in
LLMInferenceQueue.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from core.metrics import MetricsCollector

logger = logging.getLogger("atom.priority_sched")

# Log when pending jobs exceed this (backpressure signal).
_SCHED_QUEUE_WARN = 48
_SCHED_QUEUE_CRITICAL = 96

# Lower number = higher priority (asyncio.PriorityQueue orders smallest first).
PRIORITY_VOICE = 0
PRIORITY_LLM = 1
PRIORITY_BACKGROUND = 2

_WaitMetricByPrio: dict[int, str] = {
    PRIORITY_VOICE: "scheduler_wait_voice",
    PRIORITY_LLM: "scheduler_wait_llm",
    PRIORITY_BACKGROUND: "scheduler_wait_background",
}


@dataclass(order=True)
class _PJob:
    priority: int
    seq: int
    enqueued_at: float = field(compare=False)
    name: str = field(compare=False)
    coro_factory: object = field(compare=False)


class PriorityScheduler:
    """One asyncio worker; jobs run in priority order (0 first)."""

    __slots__ = ("_q", "_seq", "_worker", "_shutdown", "_metrics")

    def __init__(self, metrics: MetricsCollector | None = None) -> None:
        self._q: asyncio.PriorityQueue[_PJob] = asyncio.PriorityQueue()
        self._seq = 0
        self._worker: asyncio.Task | None = None
        self._shutdown = False
        self._metrics = metrics

    def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._shutdown = False
            self._worker = asyncio.create_task(
                self._run(), name="atom_priority_scheduler"
            )

    @property
    def queue_depth(self) -> int:
        """Approximate pending jobs (may race with worker)."""
        return self._q.qsize()

    async def shutdown(self) -> None:
        self._shutdown = True
        if self._worker and not self._worker.done():
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
        self._worker = None
        if self._metrics is not None:
            self._metrics.set_gauge("scheduler_queue_depth", 0)

    async def _run(self) -> None:
        while not self._shutdown:
            try:
                job = await self._q.get()
            except asyncio.CancelledError:
                break
            try:
                if self._metrics is not None:
                    wait_ms = (time.perf_counter() - job.enqueued_at) * 1000
                    mname = _WaitMetricByPrio.get(job.priority, "scheduler_wait_other")
                    self._metrics.record_latency(mname, wait_ms)
                    self._metrics.set_gauge("scheduler_queue_depth", self._q.qsize())
                coro = job.coro_factory()
                if asyncio.iscoroutine(coro):
                    await coro
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Priority job %s failed", job.name)
            finally:
                self._q.task_done()
                if self._metrics is not None:
                    self._metrics.set_gauge("scheduler_queue_depth", self._q.qsize())

    def submit(self, priority: int, name: str, coro_factory: Callable[[], Any]) -> None:
        """Queue work. *coro_factory* must return a coroutine when called (no args)."""
        if self._shutdown:
            logger.debug("Scheduler shut down — dropped job %s", name)
            return
        depth = self._q.qsize()
        if depth >= _SCHED_QUEUE_CRITICAL:
            logger.error(
                "Priority scheduler queue very deep (%d) — job %s may delay responsiveness",
                depth,
                name,
            )
        elif depth >= _SCHED_QUEUE_WARN:
            logger.warning(
                "Priority scheduler queue depth high (%d) — job %s",
                depth,
                name,
            )
        self._seq += 1
        t = time.perf_counter()
        self._q.put_nowait(
            _PJob(priority, self._seq, t, name, coro_factory)
        )
        if self._metrics is not None:
            self._metrics.set_gauge("scheduler_queue_depth", self._q.qsize())
            self._metrics.inc("scheduler_jobs_submitted")
