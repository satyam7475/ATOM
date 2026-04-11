"""
ATOM — Apple Silicon Governor.

Monitors Apple Silicon hardware and manages thermal/memory protection.
Replaces the multi-backend GPUGovernor with a single, clean Apple Silicon path.

Apple Silicon has Unified Memory — CPU, GPU, and Neural Engine share the same
memory pool. There is no VRAM, no discrete GPU, no NVIDIA. This module
embraces that architecture directly.

Data source: core.apple_silicon_monitor (psutil + pmset + system_profiler)

Emits events:
    silicon_stats_update:  periodic hardware telemetry
    silicon_thermal_warn:  thermal pressure exceeded threshold
    silicon_memory_warn:   memory usage exceeded threshold
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from core.apple_silicon_monitor import AppleSiliconMonitor, AppleSiliconStats

logger = logging.getLogger("atom.silicon_governor")

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus


class SiliconGovernor:
    """Apple Silicon hardware monitor and governor.

    Single-backend, zero-abstraction layer for M-series chips.
    Monitors unified memory pressure, thermal throttling, and battery state.
    """

    _THERMAL_THRESHOLD = 95
    _MEMORY_THRESHOLD_PCT = 85
    _POLL_INTERVAL_S = 30

    def __init__(
        self,
        bus: "AsyncEventBus | None" = None,
        config: dict | None = None,
    ) -> None:
        self._bus = bus
        self._config = (config or {}).get("gpu", {})
        self._enabled = self._config.get("enabled", True)
        self._thermal_threshold = self._config.get(
            "thermal_threshold", self._THERMAL_THRESHOLD,
        )
        self._memory_threshold = self._config.get(
            "memory_threshold_pct", self._MEMORY_THRESHOLD_PCT,
        )
        self._poll_interval = self._config.get(
            "poll_interval_s", self._POLL_INTERVAL_S,
        )
        self._monitor = AppleSiliconMonitor()
        self._last_stats = AppleSiliconStats()
        self._thermal_warned = False
        self._memory_warned = False
        self._running = False
        self._task: asyncio.Task | None = None

        if self._enabled:
            logger.info(
                "Silicon Governor: %s (%d GPU cores, Unified Memory)",
                self._monitor.gpu_name, self._monitor.gpu_cores,
            )

    @property
    def is_available(self) -> bool:
        return self._enabled

    @property
    def gpu_name(self) -> str:
        return self._monitor.gpu_name

    def get_stats(self) -> AppleSiliconStats:
        """Get current Apple Silicon hardware snapshot."""
        try:
            self._last_stats = self._monitor.get_stats()
        except Exception:
            logger.debug("Silicon stats query failed", exc_info=True)
        return self._last_stats

    def start(self) -> None:
        if not self._enabled:
            return
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("Silicon Governor monitoring started (interval=%ds)", self._poll_interval)

    def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            self._task = None

    def shutdown(self) -> None:
        self.stop()

    async def _monitor_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self._poll_interval)
                if not self._running:
                    break

                stats = self.get_stats()

                try:
                    from core.metrics import get_metrics
                    m = get_metrics()
                    m.set_gauge("memory_used_mb", stats.memory_used_mb)
                    m.set_gauge("memory_pct", stats.memory_pct)
                    m.set_gauge("cpu_pct", stats.cpu_pct)
                except Exception:
                    pass

                if self._bus is not None:
                    self._bus.emit_fast("silicon_stats_update", stats=stats.summary())
                    self._bus.emit_fast("gpu_stats_update", stats=stats.summary())

                thermal_triggered = stats.is_throttled
                thermal_cleared = stats.thermal_pressure in ("nominal", "moderate", "")

                if thermal_triggered and not self._thermal_warned:
                    self._thermal_warned = True
                    logger.warning(
                        "Thermal warning: pressure=%s, temp=~%.0f°C",
                        stats.thermal_pressure, stats.cpu_temp_c,
                    )
                    if self._bus is not None:
                        self._bus.emit("silicon_thermal_warn", pressure=stats.thermal_pressure)
                        self._bus.emit("gpu_thermal_warning", temp=stats.cpu_temp_c)
                elif self._thermal_warned and thermal_cleared:
                    self._thermal_warned = False

                if stats.memory_pct > self._memory_threshold:
                    if not self._memory_warned:
                        self._memory_warned = True
                        logger.warning(
                            "Memory pressure warning: %.0f%% (%.0f/%.0fMB)",
                            stats.memory_pct, stats.memory_used_mb, stats.memory_total_mb,
                        )
                        if self._bus is not None:
                            self._bus.emit("silicon_memory_warn", memory_pct=stats.memory_pct)
                            self._bus.emit("gpu_memory_warning", memory_pct=stats.memory_pct)
                elif self._memory_warned and stats.memory_pct < self._memory_threshold - 10:
                    self._memory_warned = False

            except asyncio.CancelledError:
                break
            except Exception:
                logger.debug("Silicon monitor error", exc_info=True)


__all__ = ["SiliconGovernor"]
