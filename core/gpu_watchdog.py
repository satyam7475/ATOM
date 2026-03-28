"""
ATOM V7 — GPU stall watchdog: detect hung inference (timeout) and emit events.

CUDA reset is optional and dangerous on Windows; gated by v7_gpu.allow_cuda_reset.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus

logger = logging.getLogger("atom.gpu_watchdog")


class GPUStallWatchdog:
    """Monitor inference duration; emit gpu_stall when threshold exceeded."""

    __slots__ = ("_bus", "_timeout_s", "_allow_reset", "_task", "_running", "_deadline", "_config")

    def __init__(
        self,
        bus: "AsyncEventBus | None",
        config: dict | None = None,
    ) -> None:
        self._bus = bus
        self._config = config or {}
        v7 = self._config.get("v7_gpu") or {}
        self._timeout_s = float(v7.get("gpu_stall_timeout_s", 120))
        self._allow_reset = bool(v7.get("allow_cuda_reset", False))
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
                logger.warning("GPU stall watchdog: inference exceeded %.0fs", self._timeout_s)
                if self._bus:
                    self._bus.emit_fast(
                        "gpu_stall",
                        timeout_s=self._timeout_s,
                        t=time.time(),
                    )
                from core.runtime_config import DegradationMode, set_degradation_mode
                set_degradation_mode(DegradationMode.LIMITED)
                if self._allow_reset:
                    self._maybe_cuda_reset()
                self._deadline = None

    def _maybe_cuda_reset(self) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                logger.info("GPU watchdog: empty_cache() after stall")
        except Exception:
            logger.debug("cuda reset path failed", exc_info=True)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name="v7_gpu_stall_watchdog")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
