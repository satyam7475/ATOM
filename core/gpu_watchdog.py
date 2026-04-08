"""
ATOM — Inference stall watchdog.

Detects hung inference (timeout exceeded) and triggers degradation mode.
Platform-agnostic — works on any backend (Metal, CPU).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus

logger = logging.getLogger("atom.gpu_watchdog")


class GPUStallWatchdog:
    """Monitor inference duration; emit gpu_stall when threshold exceeded."""

    __slots__ = ("_bus", "_timeout_s", "_task", "_running", "_deadline")

    def __init__(
        self,
        bus: "AsyncEventBus | None",
        config: dict | None = None,
    ) -> None:
        self._bus = bus
        cfg = config or {}
        v7 = cfg.get("v7_gpu") or {}
        self._timeout_s = float(v7.get("gpu_stall_timeout_s", 120))
        self._task: asyncio.Task | None = None
        self._running = False
        self._deadline: float | None = None

    def start_inference_watch(self) -> None:
        self._deadline = time.monotonic() + self._timeout_s

    def clear_inference_watch(self) -> None:
        self._deadline = None

    async def _run(self) -> None:
        while self._running:
            await asyncio.sleep(2.0)
            if self._deadline is None:
                continue
            if time.monotonic() > self._deadline:
                logger.warning("Inference stall: exceeded %.0fs timeout", self._timeout_s)
                if self._bus:
                    self._bus.emit_fast(
                        "gpu_stall",
                        timeout_s=self._timeout_s,
                        t=time.time(),
                    )
                from core.runtime_config import DegradationMode, set_degradation_mode
                set_degradation_mode(DegradationMode.LIMITED)
                self._deadline = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name="inference_stall_watchdog")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
