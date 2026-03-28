"""
ATOM V7 — GPU-oriented scheduler facade over PriorityScheduler.

Maps P0–P3 to internal priorities with cooperative preemption hooks.

When ``v7_gpu.strict_control`` is enabled, heavy work should be submitted only
through this scheduler (or PriorityScheduler with matching tiers) so
GPUResourceManager grants and queue depth stay coherent.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from core.priority_scheduler import (
    PRIORITY_BACKGROUND,
    PRIORITY_LLM,
    PRIORITY_MEMORY_TASK,
    PRIORITY_VOICE,
    JobHandle,
    PriorityScheduler,
)

logger = logging.getLogger("atom.gpu_scheduler")

# V7 tiers (lower integer = higher urgency in PriorityQueue)
P0_VOICE_INTERRUPT = 0
P1_USER_COMMAND = 1
P2_MEMORY_REASONING = 2
P3_BACKGROUND = 3

_V7_MAP: dict[int, int] = {
    P0_VOICE_INTERRUPT: PRIORITY_VOICE,
    P1_USER_COMMAND: PRIORITY_LLM,
    P2_MEMORY_REASONING: PRIORITY_MEMORY_TASK,
    P3_BACKGROUND: PRIORITY_BACKGROUND,
}


class GPUScheduler:
    """Submit work with V7 priority labels; delegates to PriorityScheduler."""

    __slots__ = ("_sched", "_gpu_mgr")

    def __init__(
        self,
        scheduler: PriorityScheduler | None,
        gpu_resource_manager: Any = None,
    ) -> None:
        self._sched = scheduler
        self._gpu_mgr = gpu_resource_manager

    def submit_gpu_task(
        self,
        v7_priority: int,
        name: str,
        coro_factory: Callable[[], Any],
        deadline: float = float("inf"),
        context: dict | None = None,
        trace_id: str | None = None,
    ) -> JobHandle | None:
        """Queue coroutine factory; returns handle or None if scheduler disabled."""
        if self._sched is None:
            logger.debug("GPUScheduler: no PriorityScheduler — running would block; drop %s", name)
            return None

        mapped = _V7_MAP.get(v7_priority, PRIORITY_BACKGROUND)
        ctx = dict(context or {})
        if trace_id:
            ctx["trace_id"] = trace_id
        if self._gpu_mgr is not None:
            ctx["gpu_stream"] = self._gpu_mgr.allocate_task(mapped)

        try:
            from core.metrics import get_metrics
            get_metrics().set_gauge(
                "gpu_sched_queue_depth",
                float(self._sched.queue_depth + 1),
            )
        except Exception:
            pass

        return self._sched.submit(
            mapped, name, coro_factory, deadline=deadline, context=ctx,
        )

    def preempt_lower_queued(self, min_v7_priority: int) -> int:
        """Cooperative: cancel queued jobs strictly below min_v7_priority (best-effort)."""
        if self._sched is None:
            return 0
        # PriorityScheduler stores jobs internally; no iterator exposed.
        # Callers use JobHandle.cancel() when holding handles; this is a placeholder count.
        return 0


__all__ = [
    "GPUScheduler",
    "P0_VOICE_INTERRUPT",
    "P1_USER_COMMAND",
    "P2_MEMORY_REASONING",
    "P3_BACKGROUND",
]
