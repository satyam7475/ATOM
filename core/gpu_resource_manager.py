"""
ATOM V7 — GPU Resource Manager.

VRAM budgets, named model slots (llm, stt, embeddings), load/evict policy,
and cross-process coordination via AsyncEventBus / ZmqEventBus events.

Does not replace GPUGovernor (telemetry/thermal); complements it with policy.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

logger = logging.getLogger("atom.gpu_resource_manager")

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus


class ModelSlot(str, Enum):
    LLM = "llm"
    STT = "stt"
    EMBEDDINGS = "embeddings"


class PowerMode(str, Enum):
    ACTIVE = "active"
    IDLE = "idle"
    SLEEP = "sleep"


@dataclass
class GPUResourceState:
    """Snapshot of what this process believes is loaded (controller view)."""
    loaded: dict[str, bool] = field(default_factory=lambda: {
        ModelSlot.LLM.value: False,
        ModelSlot.STT.value: False,
        ModelSlot.EMBEDDINGS.value: False,
    })
    vram_total_mb: float = 0.0
    vram_used_mb: float = 0.0
    power_mode: PowerMode = PowerMode.ACTIVE
    last_activity_monotonic: float = field(default_factory=time.monotonic)


# ZMQ / bus event names (contract for workers)
EVENT_GPU_REQUEST_LOAD = "v7_gpu_request_load"
EVENT_GPU_UNLOAD = "v7_gpu_unload"
EVENT_GPU_STATUS = "v7_gpu_status"
EVENT_GPU_ACK = "v7_gpu_ack"
EVENT_GPU_PRELOAD = "v7_gpu_preload"


def get_nvml_vram_mb() -> tuple[float, float]:
    """Return (used_mb, total_mb) or (0, 0) if unavailable."""
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        used = mem.used / (1024 * 1024)
        total = mem.total / (1024 * 1024)
        pynvml.nvmlShutdown()
        return used, total
    except Exception:
        return 0.0, 0.0


class GPUResourceManager:
    """Single-process policy + optional bus fan-out for distributed workers."""

    __slots__ = (
        "_bus", "_config", "_v7", "_state", "_eviction_order",
        "_slots_mb", "_reserve_mb", "_enabled", "_power_task",
        "_strict", "_grants",
    )

    def __init__(
        self,
        bus: "AsyncEventBus | None" = None,
        config: dict | None = None,
    ) -> None:
        self._bus = bus
        self._config = config or {}
        self._v7 = (self._config.get("v7_gpu") or {})
        self._enabled = bool(self._v7.get("enabled", True))
        self._reserve_mb = float(self._v7.get("vram_reserve_mb", 512))
        self._slots_mb = dict(self._v7.get("model_slots_mb") or {
            "llm": 8192,
            "stt": 3072,
            "embeddings": 512,
        })
        self._eviction_order: list[str] = list(
            self._v7.get("eviction_order") or ["embeddings", "stt", "llm"],
        )
        self._state = GPUResourceState()
        self._power_task: Any = None
        self._last_power_unload: str | None = None
        self._strict = bool(self._v7.get("strict_control", False))
        self._grants: dict[str, str] = {}

        if self._bus and self._enabled:
            self._bus.on("speech_final", self._on_activity)
            self._bus.on("intent_classified", self._on_activity)
            self._bus.on("cursor_query", self._on_activity)

    def _on_activity(self, **_kw: Any) -> None:
        self._state.last_activity_monotonic = time.monotonic()
        self._last_power_unload = None
        if self._state.power_mode != PowerMode.ACTIVE:
            self._state.power_mode = PowerMode.ACTIVE
            self._emit_power_event("active")

    def _emit_power_event(self, name: str) -> None:
        if self._bus:
            self._bus.emit_fast("v7_gpu_power", mode=name, t=time.time())

    def refresh_vram(self) -> None:
        used, total = get_nvml_vram_mb()
        self._state.vram_used_mb = used
        self._state.vram_total_mb = total

    def enough_vram_for(self, slot: str) -> bool:
        """Heuristic: free VRAM >= slot budget + reserve."""
        self.refresh_vram()
        need = float(self._slots_mb.get(slot, 0)) + self._reserve_mb
        free = max(0.0, self._state.vram_total_mb - self._state.vram_used_mb)
        if self._state.vram_total_mb <= 0:
            return True
        return free >= need * 0.85

    def mark_loaded(self, slot: str, loaded: bool) -> None:
        self._state.loaded[slot] = loaded

    def evict_low_priority(self) -> list[str]:
        """Return list of slots to unload (in order) until budget may fit."""
        evicted: list[str] = []
        for name in self._eviction_order:
            if self._state.loaded.get(name):
                evicted.append(name)
        return evicted

    @property
    def strict_control(self) -> bool:
        return self._strict

    def issue_load_grant(self, slot: str, priority: str = "normal") -> str | None:
        """Single authority: VRAM check + optional eviction list; returns grant token.

        When ``strict_control`` is False, returns ``permissive`` and emits load as before.
        When True, registers a token; workers must call ``complete_load`` with token.
        """
        if not self._enabled:
            return None
        if not self._strict:
            self._emit_load_request(slot, priority)
            return "permissive"

        if not self.enough_vram_for(slot):
            logger.warning(
                "GPUResourceManager: cannot grant %s — insufficient VRAM (evict manually)",
                slot,
            )
            return None
        token = uuid.uuid4().hex
        self._grants[slot] = token
        self._emit_load_request(slot, priority, grant_token=token)
        logger.debug("GPUResourceManager: issued grant for %s", slot)
        return token

    def validate_grant(self, slot: str, token: str | None) -> bool:
        if not self._strict:
            return True
        if not token or token == "permissive":
            return False
        return self._grants.get(slot) == token

    def complete_load(self, slot: str, token: str | None) -> bool:
        """Worker ack after load: validates token and marks slot loaded."""
        if not self._strict:
            self.mark_loaded(slot, True)
            return True
        if not self.validate_grant(slot, token):
            logger.warning("GPUResourceManager: complete_load denied for %s", slot)
            return False
        self._grants.pop(slot, None)
        self.mark_loaded(slot, True)
        return True

    def abort_grant(self, slot: str) -> None:
        self._grants.pop(slot, None)

    def request_load(self, slot: str, priority: str = "normal") -> None:
        """Emit cross-process load request (workers perform actual load).

        If ``strict_control`` is True, no-ops unless a grant exists (use ``issue_load_grant``).
        """
        if not self._enabled or not self._bus:
            return
        if self._strict and slot not in self._grants:
            logger.warning(
                "GPUResourceManager: blocked request_load(%s) — strict mode requires issue_load_grant",
                slot,
            )
            return
        self._emit_load_request(slot, priority)

    def _emit_load_request(
        self, slot: str, priority: str = "normal", grant_token: str | None = None,
    ) -> None:
        if not self._bus:
            return
        payload: dict[str, Any] = {
            "slot": slot,
            "priority": priority,
            "t": time.time(),
        }
        if grant_token:
            payload["grant_token"] = grant_token
        self._bus.emit_fast(EVENT_GPU_REQUEST_LOAD, **payload)

    def request_unload(self, slot: str, reason: str = "policy") -> None:
        if not self._enabled or not self._bus:
            return
        self._bus.emit_fast(
            EVENT_GPU_UNLOAD,
            slot=slot,
            reason=reason,
            t=time.time(),
        )

    def emit_status(self, extra: dict | None = None) -> None:
        self.refresh_vram()
        payload = {
            "loaded": dict(self._state.loaded),
            "vram_used_mb": round(self._state.vram_used_mb, 1),
            "vram_total_mb": round(self._state.vram_total_mb, 1),
            "power_mode": self._state.power_mode.value,
        }
        if extra:
            payload.update(extra)
        if self._bus:
            self._bus.emit_fast(EVENT_GPU_STATUS, **payload)

    def allocate_task(self, task_priority: int) -> str:
        """Map scheduler priority to a logical stream name (metrics)."""
        if task_priority <= 0:
            return "stream_voice"
        if task_priority == 1:
            return "stream_user"
        if task_priority == 2:
            return "stream_memory"
        return "stream_background"

    def idle_seconds(self) -> float:
        return time.monotonic() - self._state.last_activity_monotonic

    def apply_idle_power_policy(self) -> None:
        """Unload STT / LLM after configured idle thresholds (main process)."""
        if not self._enabled:
            return
        idle = self.idle_seconds()
        stt_s = float(self._v7.get("idle_unload_stt_s", 60))
        llm_s = float(self._v7.get("idle_unload_llm_s", 180))
        sleep_s = float(self._v7.get("idle_sleep_s", 600))

        if idle >= sleep_s:
            if self._state.power_mode != PowerMode.SLEEP:
                self._state.power_mode = PowerMode.SLEEP
                self._emit_power_event("sleep")
                self.request_unload("stt", "idle_sleep")
                self.request_unload("llm", "idle_sleep")
                self._last_power_unload = "sleep"
        elif idle >= llm_s:
            if self._state.power_mode != PowerMode.IDLE:
                self._state.power_mode = PowerMode.IDLE
                self._emit_power_event("idle")
            if self._last_power_unload not in ("llm", "sleep"):
                self.request_unload("llm", "idle")
                self._last_power_unload = "llm"
        elif idle >= stt_s:
            if self._last_power_unload not in ("stt", "llm", "sleep"):
                self.request_unload("stt", "idle")
                self._last_power_unload = "stt"

    def preload_models(self) -> None:
        """Signal workers to preload critical models (after wake)."""
        if not self._enabled or not self._bus:
            return
        self._bus.emit_fast(EVENT_GPU_PRELOAD, t=time.time())
        for slot in ("embeddings", "stt", "llm"):
            if self._strict:
                self.issue_load_grant(slot, priority="high")
            else:
                self._emit_load_request(slot, priority="high")

    def start_power_task(self) -> None:
        """Background: idle-based power policy (call from main asyncio loop)."""
        if not self._enabled or self._bus is None:
            return
        try:
            import asyncio
            loop = asyncio.get_running_loop()

            async def _loop() -> None:
                while True:
                    await asyncio.sleep(15.0)
                    try:
                        self.apply_idle_power_policy()
                    except Exception:
                        logger.debug("idle power policy", exc_info=True)

            self._power_task = asyncio.create_task(_loop(), name="v7_gpu_power")
        except RuntimeError:
            pass


def get_gpu_resource_manager(
    bus: "AsyncEventBus | None" = None,
    config: dict | None = None,
) -> GPUResourceManager:
    return GPUResourceManager(bus=bus, config=config)


__all__ = [
    "GPUResourceManager",
    "GPUResourceState",
    "ModelSlot",
    "PowerMode",
    "get_nvml_vram_mb",
    "EVENT_GPU_REQUEST_LOAD",
    "EVENT_GPU_UNLOAD",
    "EVENT_GPU_STATUS",
    "EVENT_GPU_ACK",
    "EVENT_GPU_PRELOAD",
    "get_gpu_resource_manager",
]
