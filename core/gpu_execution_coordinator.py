"""
ATOM — Hardware-aware GPU execution coordinator (soft scheduling).

Serializes heavy GPU phases by default; uses NVML (and optional torch) for
admission control — defer / delay, not hard locks. Compatible with
LocalBrainController + MiniLLM._abort_generation preemption.

Does not replace MiniLLM or STTAsync executors; it schedules when their work runs.

STT integration: pass the same ``GPUTask`` into the closure and check
``task.cancelled`` immediately before ``run_in_executor``; Whisper cannot abort
mid-kernel, but the next utterance can skip scheduling. Pair with
``preempt_for_voice()`` which sets ``_current_stt_task.cancel()``.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus

logger = logging.getLogger("atom.gpu_execution_coordinator")


# ---------------------------------------------------------------------------
# NVML + optional torch memory probes
# ---------------------------------------------------------------------------

def _nvml_snapshot() -> tuple[float, float, float, float]:
    """Return (vram_used_mb, vram_total_mb, gpu_util_pct, mem_util_pct).

    Tolerant of missing NVML / non-NVIDIA hosts: returns zeros without raising.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            rates = pynvml.nvmlDeviceGetUtilizationRates(h)
            used = mem.used / (1024 * 1024)
            total = mem.total / (1024 * 1024)
            return used, total, float(rates.gpu), float(rates.memory)
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


def _torch_cuda_mem_mb() -> tuple[float | None, float | None]:
    try:
        import torch
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            return free_b / (1024 * 1024), total_b / (1024 * 1024)
    except Exception:
        pass
    return None, None


def estimate_fragmentation(
    vram_used_mb: float,
    vram_total_mb: float,
    vram_free_mb: float | None,
) -> float:
    """Heuristic in [0, 1]: higher = more pressure / likely fragmentation.

    NVML does not expose allocator fragmentation; we combine utilization of
    the memory pool with how small the free slab is relative to total.
    """
    if vram_total_mb <= 0:
        return 0.0
    used_ratio = min(1.0, max(0.0, vram_used_mb / vram_total_mb))
    if vram_free_mb is None:
        vram_free_mb = max(0.0, vram_total_mb - vram_used_mb)
    free_ratio = min(1.0, vram_free_mb / vram_total_mb)
    # Tight free slab under load suggests allocator pressure
    pressure = (1.0 - free_ratio) * 0.5 + used_ratio * 0.5
    return max(0.0, min(1.0, pressure))


# ---------------------------------------------------------------------------
# Task model
# ---------------------------------------------------------------------------

class Phase(Enum):
    IDLE = auto()
    STT = auto()
    LLM = auto()
    EMBED = auto()


class TaskIntent(str, Enum):
    USER_QUERY = "user_query"
    SYSTEM_THINKING = "system_thinking"
    MEMORY_UPDATE = "memory_update"
    PROACTIVE_ACTION = "proactive_action"
    VOICE_INPUT = "voice_input"
    BACKGROUND = "background"


# Align with priority_scheduler (lower number = higher priority)
PRIORITY_VOICE = 0
PRIORITY_LLM = 1
PRIORITY_MEMORY = 2
PRIORITY_BACKGROUND = 3


@dataclass
class GPUState:
    vram_used_mb: float = 0.0
    vram_total_mb: float = 0.0
    vram_free_mb: float = 0.0
    gpu_util_pct: float = 0.0
    mem_util_pct: float = 0.0
    fragmentation_estimate: float = 0.0
    torch_free_mb: float | None = None
    torch_total_mb: float | None = None
    updated_monotonic: float = field(default_factory=time.monotonic)

    def can_fit(self, required_mb: float, reserve_mb: float) -> bool:
        if self.vram_total_mb <= 0:
            return True
        need = required_mb + reserve_mb
        # Prefer torch free if available (same process allocator view)
        if self.torch_free_mb is not None:
            return self.torch_free_mb >= need * 0.9
        return self.vram_free_mb >= need * 0.9


@dataclass
class GPUTask:
    kind: str  # "stt" | "llm" | "embed"
    priority: int
    intent: TaskIntent
    name: str
    run: Callable[[], Awaitable[Any]]
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    trace_id: str | None = None
    vram_required_mb: float = 512.0
    estimated_duration_ms: float = 500.0
    allow_overlap: bool = False
    cancelled: bool = False
    enqueued_at: float = field(default_factory=time.perf_counter)
    seq: int = 0
    context: dict[str, Any] = field(default_factory=dict)
    defer_cycles: int = 0

    def cancel(self) -> None:
        self.cancelled = True


@dataclass(order=True)
class _PQItem:
    effective_pri: int
    seq: int
    task: GPUTask = field(compare=False)


@dataclass
class TaskFeedback:
    task_id: str
    intent: TaskIntent
    latency_ms: float
    success: bool
    preempted: bool
    queue_wait_ms: float
    error: str | None = None


class GPUExecutionCoordinator:
    """Hardware-aware soft scheduler for GPU-bound work."""

    def __init__(
        self,
        bus: Optional["AsyncEventBus"] = None,
        config: dict | None = None,
        metrics: Any = None,
    ) -> None:
        self._bus = bus
        self._config = config or {}
        self._metrics = metrics
        sched = self._config.get("gpu_execution", {})
        self._vram_reserve_mb = float(sched.get("vram_reserve_mb", 512))
        self._high_util_pct = float(sched.get("high_gpu_util_defer_background", 85))
        self._frag_defer_threshold = float(sched.get("fragmentation_defer_threshold", 0.92))
        self._defer_backoff_s = float(sched.get("defer_backoff_s", 0.05))
        self._max_defer_cycles = int(sched.get("max_defer_cycles", 40))
        self._embed_light_max_mb = float(sched.get("embed_light_max_mb", 256))
        self._overlap_util_max = float(sched.get("overlap_max_gpu_util", 55))

        self._phase = Phase.IDLE
        self._pq: asyncio.PriorityQueue[_PQItem] = asyncio.PriorityQueue()
        self._seq = 0
        self._shutdown = False
        self._worker: asyncio.Task | None = None
        self._gpu_state = GPUState()
        self._state_ttl_s = float(sched.get("gpu_state_ttl_s", 0.25))

        # Optional brain reference for voice preemption (LLM only)
        self._brain: Any = None

        # Feedback → dynamic priority bias (EWMA latency ms per intent)
        self._ewma_latency: dict[str, float] = {}
        self._ewma_alpha = float(sched.get("feedback_ewma_alpha", 0.15))
        self._preempt_counts: dict[str, int] = {}
        self._recent_logs: deque[dict[str, Any]] = deque(maxlen=int(sched.get("exec_log_max", 128)))

        # Current STT task handle for cooperative cancel
        self._current_stt_task: GPUTask | None = None

    def attach_brain(self, brain: Any) -> None:
        """LocalBrainController — used for request_preempt on voice."""
        self._brain = brain

    def refresh_gpu_state(self) -> GPUState:
        used, total, gpu_u, mem_u = _nvml_snapshot()
        free = max(0.0, total - used) if total > 0 else 0.0
        tf, tt = _torch_cuda_mem_mb()
        frag = estimate_fragmentation(used, total, free)
        self._gpu_state = GPUState(
            vram_used_mb=used,
            vram_total_mb=total,
            vram_free_mb=free,
            gpu_util_pct=gpu_u,
            mem_util_pct=mem_u,
            fragmentation_estimate=frag,
            torch_free_mb=tf,
            torch_total_mb=tt,
            updated_monotonic=time.monotonic(),
        )
        if self._metrics is not None:
            try:
                self._metrics.set_gauge("vram_used_mb", used)
                self._metrics.set_gauge("gpu_util_pct", gpu_u)
                self._metrics.set_gauge("gpu_frag_estimate", frag)
            except Exception:
                pass
        return self._gpu_state

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def gpu_state(self) -> GPUState:
        return self._gpu_state

    def preempt_for_voice(self, reason: str = "barge_in") -> None:
        if self._brain is not None:
            try:
                self._brain.request_preempt()
            except Exception:
                logger.debug("preempt brain", exc_info=True)
        if self._current_stt_task is not None:
            self._current_stt_task.cancel()
        if self._bus:
            self._bus.emit_fast("gpu_preempt", reason=reason, t=time.time())
        self._preempt_counts["voice"] = self._preempt_counts.get("voice", 0) + 1
        if self._metrics is not None:
            try:
                self._metrics.inc("llm_preempted")
            except Exception:
                pass

    def _effective_priority(self, task: GPUTask) -> int:
        """Lower = sooner. Bias from feedback EWMA (high latency → boost interactive)."""
        p = task.priority
        key = task.intent.value
        ew = self._ewma_latency.get(key)
        if ew is not None and ew > 2000 and task.intent in (
            TaskIntent.USER_QUERY,
            TaskIntent.VOICE_INPUT,
        ):
            p = max(PRIORITY_VOICE, p - 1)
        if task.context.get("user_interrupt"):
            return -1
        return p

    def record_feedback(self, fb: TaskFeedback) -> None:
        key = fb.intent.value
        prev = self._ewma_latency.get(key)
        if prev is None:
            self._ewma_latency[key] = fb.latency_ms
        else:
            a = self._ewma_alpha
            self._ewma_latency[key] = a * fb.latency_ms + (1 - a) * prev
        if fb.preempted:
            self._preempt_counts[key] = self._preempt_counts.get(key, 0) + 1
        entry = {
            "task_id": fb.task_id,
            "intent": key,
            "latency_ms": round(fb.latency_ms, 1),
            "queue_wait_ms": round(fb.queue_wait_ms, 1),
            "success": fb.success,
            "preempted": fb.preempted,
            "error": fb.error,
        }
        self._recent_logs.append(entry)
        if self._metrics is not None:
            try:
                self._metrics.record_latency(f"gpu_task_{key}", fb.latency_ms)
                self._metrics.record_latency("gpu_queue_wait_ms", fb.queue_wait_ms)
            except Exception:
                pass

    def post_task_analysis(
        self,
        task: GPUTask,
        *,
        latency_ms: float,
        queue_wait_ms: float,
        success: bool,
        preempted: bool = False,
        error: str | None = None,
    ) -> None:
        self.record_feedback(
            TaskFeedback(
                task_id=task.task_id,
                intent=task.intent,
                latency_ms=latency_ms,
                success=success,
                preempted=preempted,
                queue_wait_ms=queue_wait_ms,
                error=error,
            )
        )

    async def submit_task(
        self,
        kind: str,
        priority: int,
        intent: TaskIntent,
        name: str,
        run: Callable[[], Awaitable[Any]],
        *,
        trace_id: str | None = None,
        vram_required_mb: float = 512.0,
        estimated_duration_ms: float = 500.0,
        allow_overlap: bool = False,
        context: dict | None = None,
    ) -> GPUTask:
        self._seq += 1
        ctx = context or {}
        task = GPUTask(
            kind=kind,
            priority=priority,
            intent=intent,
            name=name,
            run=run,
            trace_id=trace_id,
            vram_required_mb=vram_required_mb,
            estimated_duration_ms=estimated_duration_ms,
            allow_overlap=allow_overlap,
            seq=self._seq,
            context=ctx,
        )
        ep = self._effective_priority(task)
        await self._pq.put(_PQItem(ep, self._seq, task))
        if self._bus:
            self._bus.emit_fast(
                "gpu_task_enqueued",
                task_id=task.task_id,
                kind=kind,
                intent=intent.value,
                priority=priority,
            )
        if self._metrics is not None:
            try:
                self._metrics.set_gauge("gpu_sched_queue_depth", float(self._pq.qsize()))
            except Exception:
                pass
        return task

    def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._shutdown = False
            self._worker = asyncio.create_task(
                self._run_loop(), name="atom_gpu_exec_coordinator"
            )

    async def shutdown(self) -> None:
        self._shutdown = True
        if self._worker and not self._worker.done():
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
        self._worker = None

    def get_observability(self) -> dict[str, Any]:
        return {
            "phase": self._phase.name,
            "queue_depth": self._pq.qsize(),
            "gpu_state": {
                "vram_used_mb": round(self._gpu_state.vram_used_mb, 1),
                "vram_total_mb": round(self._gpu_state.vram_total_mb, 1),
                "vram_free_mb": round(self._gpu_state.vram_free_mb, 1),
                "gpu_util_pct": round(self._gpu_state.gpu_util_pct, 1),
                "fragmentation_estimate": round(self._gpu_state.fragmentation_estimate, 3),
            },
            "ewma_latency_ms_by_intent": dict(self._ewma_latency),
            "preempt_counts": dict(self._preempt_counts),
            "recent_task_logs": list(self._recent_logs)[-32:],
        }

    async def _run_loop(self) -> None:
        while not self._shutdown:
            item = await self._get_next_admissible_task()
            if item is None:
                continue
            task = item.task
            if task.cancelled:
                continue

            phase_map = {"stt": Phase.STT, "llm": Phase.LLM, "embed": Phase.EMBED}
            self._phase = phase_map.get(task.kind, Phase.IDLE)
            if task.kind == "stt":
                self._current_stt_task = task
            if self._bus:
                self._bus.emit_fast("gpu_phase", phase=self._phase.name, task=task.name, task_id=task.task_id)

            t0 = time.perf_counter()
            qwait_ms = (t0 - task.enqueued_at) * 1000
            err: str | None = None
            success = False
            preempted = False
            try:
                await task.run()
                success = True
            except asyncio.CancelledError:
                raise
            except Exception as e:
                err = str(e)
                logger.exception("GPU task %s failed", task.name)
            finally:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                if task.kind == "stt":
                    self._current_stt_task = None
                self._phase = Phase.IDLE
                if self._bus:
                    self._bus.emit_fast("gpu_phase", phase="IDLE", task=None)
                self.post_task_analysis(
                    task,
                    latency_ms=elapsed_ms,
                    queue_wait_ms=qwait_ms,
                    success=success and err is None,
                    preempted=task.cancelled,
                    error=err,
                )
                if self._metrics is not None:
                    try:
                        self._metrics.set_gauge("gpu_sched_queue_depth", float(self._pq.qsize()))
                    except Exception:
                        pass

    async def _get_next_admissible_task(self) -> _PQItem | None:
        while not self._shutdown:
            item = await self._pq.get()
            task = item.task
            if task.cancelled:
                continue

            now = time.monotonic()
            if now - self._gpu_state.updated_monotonic > self._state_ttl_s:
                self.refresh_gpu_state()

            gs = self._gpu_state

            # High GPU utilization: delay background / system thinking
            if gs.gpu_util_pct >= self._high_util_pct:
                if task.intent in (TaskIntent.BACKGROUND, TaskIntent.SYSTEM_THINKING, TaskIntent.PROACTIVE_ACTION):
                    if task.defer_cycles < self._max_defer_cycles:
                        task.defer_cycles += 1
                        await asyncio.sleep(self._defer_backoff_s)
                        ep = self._effective_priority(task)
                        await self._pq.put(_PQItem(ep, self._seq, task))
                        continue

            # Fragmentation pressure: defer heavy non-voice work
            if gs.fragmentation_estimate >= self._frag_defer_threshold:
                if task.intent not in (TaskIntent.VOICE_INPUT, TaskIntent.USER_QUERY):
                    if task.defer_cycles < self._max_defer_cycles:
                        task.defer_cycles += 1
                        await asyncio.sleep(self._defer_backoff_s)
                        ep = self._effective_priority(task)
                        await self._pq.put(_PQItem(ep, self._seq, task))
                        continue

            # VRAM admission (soft)
            if not gs.can_fit(task.vram_required_mb, self._vram_reserve_mb):
                if task.defer_cycles < self._max_defer_cycles:
                    task.defer_cycles += 1
                    if self._bus:
                        self._bus.emit_fast(
                            "gpu_task_deferred",
                            task_id=task.task_id,
                            reason="vram",
                            required_mb=task.vram_required_mb,
                        )
                    await asyncio.sleep(self._defer_backoff_s * 2)
                    ep = self._effective_priority(task)
                    await self._pq.put(_PQItem(ep, self._seq, task))
                    continue
                logger.warning(
                    "Task %s exceeded max defer cycles; running anyway (may OOM)",
                    task.name,
                )

            # Soft overlap: defer light embed when GPU busy (sequential execution preserved)
            if task.allow_overlap and task.kind == "embed":
                if task.vram_required_mb > self._embed_light_max_mb:
                    task.allow_overlap = False
                elif gs.gpu_util_pct > self._overlap_util_max:
                    if task.defer_cycles < self._max_defer_cycles:
                        task.defer_cycles += 1
                        await asyncio.sleep(self._defer_backoff_s)
                        ep = self._effective_priority(task)
                        await self._pq.put(_PQItem(ep, self._seq, task))
                        continue

            return item
        return None


def get_gpu_execution_coordinator(
    bus: Optional["AsyncEventBus"] = None,
    config: dict | None = None,
    metrics: Any = None,
) -> GPUExecutionCoordinator:
    return GPUExecutionCoordinator(bus=bus, config=config, metrics=metrics)


__all__ = [
    "GPUExecutionCoordinator",
    "GPUState",
    "GPUTask",
    "TaskIntent",
    "TaskFeedback",
    "Phase",
    "PRIORITY_VOICE",
    "PRIORITY_LLM",
    "PRIORITY_MEMORY",
    "PRIORITY_BACKGROUND",
    "estimate_fragmentation",
    "get_gpu_execution_coordinator",
]
