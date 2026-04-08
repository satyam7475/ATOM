"""
ATOM — Inference Guard (Apple Silicon).

Lightweight model lifecycle manager for Unified Memory:
  - Model slot tracking (LLM / STT / Embeddings loaded state)
  - Memory pressure admission (system RAM = GPU RAM on Apple Silicon)
  - Idle unload policy (power-aware: unload after inactivity)
  - Bus events for cross-module coordination

Replaces GPUResourceManager + GPUExecutionCoordinator with a single
~120-line module that matches Apple Silicon reality: no VRAM budgets,
no slot allocations, no fragmentation heuristics.
"""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import TYPE_CHECKING, Any

from core.apple_silicon_monitor import get_apple_silicon_memory_mb

logger = logging.getLogger("atom.inference_guard")

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


EVENT_MODEL_LOAD = "v7_gpu_request_load"
EVENT_MODEL_UNLOAD = "v7_gpu_unload"
EVENT_MODEL_STATUS = "v7_gpu_status"
EVENT_MODEL_ACK = "v7_gpu_ack"
EVENT_MODEL_PRELOAD = "v7_gpu_preload"

_MEMORY_PRESSURE_THRESHOLD = 0.90


class InferenceGuard:
    """Model lifecycle + memory pressure guard for Apple Silicon."""

    def __init__(
        self,
        bus: "AsyncEventBus | None" = None,
        config: dict | None = None,
    ) -> None:
        self._bus = bus
        self._config = config or {}
        self._v7 = self._config.get("v7_gpu") or {}
        self._enabled = bool(self._v7.get("enabled", True))

        self._loaded: dict[str, bool] = {
            ModelSlot.LLM.value: False,
            ModelSlot.STT.value: False,
            ModelSlot.EMBEDDINGS.value: False,
        }
        self._power_mode = PowerMode.ACTIVE
        self._last_activity = time.monotonic()
        self._power_task: asyncio.Task | None = None

        if self._bus and self._enabled:
            self._bus.on("speech_final", self._on_activity)
            self._bus.on("intent_classified", self._on_activity)
            self._bus.on("cursor_query", self._on_activity)

        logger.info("InferenceGuard: Unified Memory mode (Apple Silicon)")

    def _on_activity(self, **_kw: Any) -> None:
        self._last_activity = time.monotonic()
        if self._power_mode != PowerMode.ACTIVE:
            self._power_mode = PowerMode.ACTIVE
            if self._bus:
                self._bus.emit_fast("v7_gpu_power", mode="active", t=time.time())

    def refresh_vram(self) -> None:
        """Refresh memory stats (compatibility shim for LocalBrainController)."""
        pass

    def memory_available(self) -> bool:
        """Check if system memory pressure allows loading another model."""
        used, total = get_apple_silicon_memory_mb()
        if total <= 0:
            return True
        return (used / total) < _MEMORY_PRESSURE_THRESHOLD

    def mark_loaded(self, slot: str, loaded: bool) -> None:
        self._loaded[slot] = loaded

    def request_load(self, slot: str, priority: str = "normal") -> None:
        if not self._enabled or not self._bus:
            return
        self._bus.emit_fast(EVENT_MODEL_LOAD, slot=slot, priority=priority, t=time.time())

    def request_unload(self, slot: str, reason: str = "policy") -> None:
        if not self._enabled or not self._bus:
            return
        self._bus.emit_fast(EVENT_MODEL_UNLOAD, slot=slot, reason=reason, t=time.time())

    def emit_status(self) -> None:
        if not self._bus:
            return
        used, total = get_apple_silicon_memory_mb()
        self._bus.emit_fast(
            EVENT_MODEL_STATUS,
            loaded=dict(self._loaded),
            memory_used_mb=round(used, 1),
            memory_total_mb=round(total, 1),
            power_mode=self._power_mode.value,
            unified_memory=True,
        )

    def idle_seconds(self) -> float:
        return time.monotonic() - self._last_activity

    def _apply_idle_policy(self) -> None:
        if not self._enabled:
            return
        idle = self.idle_seconds()
        stt_s = float(self._v7.get("idle_unload_stt_s", 60))
        llm_s = float(self._v7.get("idle_unload_llm_s", 180))
        sleep_s = float(self._v7.get("idle_sleep_s", 600))

        if idle >= sleep_s and self._power_mode != PowerMode.SLEEP:
            self._power_mode = PowerMode.SLEEP
            if self._bus:
                self._bus.emit_fast("v7_gpu_power", mode="sleep", t=time.time())
            self.request_unload("stt", "idle_sleep")
            self.request_unload("llm", "idle_sleep")
        elif idle >= llm_s and self._power_mode == PowerMode.ACTIVE:
            self._power_mode = PowerMode.IDLE
            if self._bus:
                self._bus.emit_fast("v7_gpu_power", mode="idle", t=time.time())
            self.request_unload("llm", "idle")
        elif idle >= stt_s and self._power_mode == PowerMode.ACTIVE:
            self.request_unload("stt", "idle")

    def preload_models(self) -> None:
        if not self._enabled or not self._bus:
            return
        self._bus.emit_fast(EVENT_MODEL_PRELOAD, t=time.time())
        for slot in ("embeddings", "stt", "llm"):
            self.request_load(slot, priority="high")

    def start_power_task(self) -> None:
        if not self._enabled or self._bus is None:
            return
        try:
            async def _loop() -> None:
                while True:
                    await asyncio.sleep(15.0)
                    try:
                        self._apply_idle_policy()
                    except Exception:
                        logger.debug("idle power policy error", exc_info=True)

            self._power_task = asyncio.create_task(_loop(), name="inference_guard_power")
        except RuntimeError:
            pass


__all__ = [
    "InferenceGuard",
    "ModelSlot",
    "PowerMode",
    "EVENT_MODEL_LOAD",
    "EVENT_MODEL_UNLOAD",
    "EVENT_MODEL_STATUS",
    "EVENT_MODEL_ACK",
    "EVENT_MODEL_PRELOAD",
]
