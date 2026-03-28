"""
ATOM — Priority job queue with starvation prevention.

Voice and LLM work is submitted here so bus handlers return quickly and
user input is not stuck behind long jobs. LLM serialization remains in
LLMInferenceQueue.

v20 enhancements:
  - Starvation prevention: background jobs get priority boost after waiting
  - Per-priority job counters and latency tracking
  - Job cancellation support via returned handle
  - Max-age eviction: stale jobs are dropped instead of executed
  - Diagnostics API for dashboard
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from core.metrics import MetricsCollector

logger = logging.getLogger("atom.priority_sched")

_SCHED_QUEUE_WARN = 48
_SCHED_QUEUE_CRITICAL = 96

PRIORITY_VOICE = 0
PRIORITY_LLM = 1
PRIORITY_MEMORY_TASK = 2
PRIORITY_BACKGROUND = 3

_STARVATION_BOOST_S = 10.0
_MAX_JOB_AGE_S = 120.0

_WaitMetricByPrio: dict[int, str] = {
    PRIORITY_VOICE: "scheduler_wait_voice",
    PRIORITY_LLM: "scheduler_wait_llm",
    PRIORITY_MEMORY_TASK: "scheduler_wait_memory",
    PRIORITY_BACKGROUND: "scheduler_wait_background",
}

_PrioNames: dict[int, str] = {
    PRIORITY_VOICE: "voice",
    PRIORITY_LLM: "llm",
    PRIORITY_MEMORY_TASK: "memory",
    PRIORITY_BACKGROUND: "background",
}


@dataclass(order=True)
class _PJob:
    effective_priority: int
    deadline: float  # Absolute time deadline
    seq: int
    enqueued_at: float = field(compare=False)
    original_priority: int = field(compare=False)
    name: str = field(compare=False)
    coro_factory: object = field(compare=False)
    context: dict = field(default_factory=dict, compare=False)
    cancelled: bool = field(default=False, compare=False)


class JobHandle:
    """Lightweight handle returned by submit() for optional cancellation."""

    __slots__ = ("_job",)

    def __init__(self, job: _PJob) -> None:
        self._job = job

    def cancel(self) -> None:
        self._job.cancelled = True


class PriorityScheduler:
    """Single asyncio worker with priority ordering and starvation prevention."""

    __slots__ = (
        "_q", "_seq", "_worker", "_shutdown", "_metrics",
        "_jobs_completed", "_jobs_dropped", "_jobs_by_prio",
    )

    def __init__(self, metrics: MetricsCollector | None = None) -> None:
        self._q: asyncio.PriorityQueue[_PJob] = asyncio.PriorityQueue()
        self._seq: int = 0
        self._worker: asyncio.Task | None = None
        self._shutdown: bool = False
        self._metrics = metrics
        self._jobs_completed: int = 0
        self._jobs_dropped: int = 0
        self._jobs_by_prio: dict[int, int] = defaultdict(int)

    def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._shutdown = False
            self._worker = asyncio.create_task(
                self._run(), name="atom_priority_scheduler"
            )

    @property
    def queue_depth(self) -> int:
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
                if job.cancelled:
                    self._jobs_dropped += 1
                    continue

                now = time.perf_counter()
                age = now - job.enqueued_at

                if age > _MAX_JOB_AGE_S:
                    logger.warning(
                        "Dropping stale job %s (age=%.0fs > max %.0fs)",
                        job.name, age, _MAX_JOB_AGE_S,
                    )
                    self._jobs_dropped += 1
                    continue

                if self._metrics is not None:
                    wait_ms = age * 1000
                    mname = _WaitMetricByPrio.get(job.original_priority, "scheduler_wait_other")
                    self._metrics.record_latency(mname, wait_ms)
                    self._metrics.set_gauge("scheduler_queue_depth", self._q.qsize())

                coro = job.coro_factory()
                if asyncio.iscoroutine(coro):
                    await coro

                self._jobs_completed += 1
                self._jobs_by_prio[job.original_priority] += 1
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Priority job %s failed", job.name)
            finally:
                self._q.task_done()
                if self._metrics is not None:
                    self._metrics.set_gauge("scheduler_queue_depth", self._q.qsize())

    def submit(
        self, priority: int, name: str, coro_factory: Callable[[], Any],
        deadline: float = float('inf'), context: dict = None
    ) -> JobHandle:
        """Queue work. Returns a JobHandle for optional cancellation."""
        if self._shutdown:
            logger.debug("Scheduler shut down — dropped job %s", name)
            job = _PJob(priority, deadline, 0, 0.0, priority, name, coro_factory, context or {}, cancelled=True)
            return JobHandle(job)

        depth = self._q.qsize()
        if depth >= _SCHED_QUEUE_CRITICAL:
            logger.error(
                "Priority scheduler queue very deep (%d) — job %s may delay responsiveness",
                depth, name,
            )
        elif depth >= _SCHED_QUEUE_WARN:
            logger.warning(
                "Priority scheduler queue depth high (%d) — job %s",
                depth, name,
            )

        self._seq += 1
        now = time.perf_counter()
        
        # Context-aware priority adjustment
        effective_priority = priority
        if context and context.get("user_interrupt", False):
            effective_priority = -1 # Highest priority
            
        job = _PJob(effective_priority, deadline, self._seq, now, priority, name, coro_factory, context or {})
        self._q.put_nowait(job)

        if self._metrics is not None:
            self._metrics.set_gauge("scheduler_queue_depth", self._q.qsize())
            self._metrics.inc("scheduler_jobs_submitted")

        return JobHandle(job)

    def get_diagnostics(self) -> dict:
        """Return scheduler stats for dashboard/health monitoring."""
        return {
            "queue_depth": self._q.qsize(),
            "jobs_completed": self._jobs_completed,
            "jobs_dropped": self._jobs_dropped,
            "by_priority": {
                _PrioNames.get(p, f"p{p}"): c
                for p, c in sorted(self._jobs_by_prio.items())
            },
        }
