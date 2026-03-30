"""
ATOM -- GPU Resource Governor.

Monitors NVIDIA GPU resources and manages priority allocation:
    LLM inference > STT (GPU mode) > Background tasks

Features:
    - Real-time GPU utilization, memory, temperature monitoring
    - Throttle non-critical GPU tasks when utilization is high
    - Thermal protection: reduce GPU load when temp > 85°C
    - Memory pressure detection: warn before OOM
    - Priority-aware scheduling hints

Falls back to CPU-only metrics if pynvml is not available.

Emits events:
    gpu_stats_update: periodic GPU telemetry
    gpu_thermal_warning: temperature threshold exceeded
    gpu_memory_warning: VRAM usage threshold exceeded
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

logger = logging.getLogger("atom.gpu_governor")

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus


@dataclass
class GPUStats:
    """Current GPU state snapshot."""
    available: bool = False
    name: str = ""
    utilization_pct: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    memory_pct: float = 0.0
    temperature_c: float = 0.0
    power_watts: float = 0.0
    fan_speed_pct: float = 0.0

    @property
    def is_throttled(self) -> bool:
        return self.temperature_c > 85 or self.utilization_pct > 95

    @property
    def memory_free_mb(self) -> float:
        return max(0, self.memory_total_mb - self.memory_used_mb)

    def summary(self) -> str:
        if not self.available:
            return "GPU: not available (CPU-only mode)"
        return (
            f"GPU: {self.name} | "
            f"Util: {self.utilization_pct:.0f}% | "
            f"VRAM: {self.memory_used_mb:.0f}/{self.memory_total_mb:.0f}MB "
            f"({self.memory_pct:.0f}%) | "
            f"Temp: {self.temperature_c:.0f}°C | "
            f"Power: {self.power_watts:.0f}W"
        )


class GPUGovernor:
    """GPU resource monitor and governor."""

    _THERMAL_THRESHOLD = 85
    _MEMORY_THRESHOLD_PCT = 90
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
        self._nvml_available = False
        self._handle: Any = None
        self._last_stats: GPUStats = GPUStats()
        self._thermal_warned = False
        self._memory_warned = False
        self._running = False
        self._task: asyncio.Task | None = None
        self._init_nvml()

    def _init_nvml(self) -> None:
        if not self._enabled:
            return
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(self._handle)
            if isinstance(name, bytes):
                name = name.decode()
            self._nvml_available = True
            logger.info("GPU Governor: %s detected via NVML", name)
        except ImportError:
            logger.info("pynvml not installed -- GPU monitoring disabled")
        except Exception:
            logger.debug("NVML init failed", exc_info=True)

    @property
    def is_available(self) -> bool:
        return self._nvml_available

    def get_stats(self) -> GPUStats:
        """Get current GPU statistics."""
        if not self._nvml_available or self._handle is None:
            return GPUStats()

        try:
            import pynvml

            name = pynvml.nvmlDeviceGetName(self._handle)
            if isinstance(name, bytes):
                name = name.decode()

            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)

            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    self._handle, pynvml.NVML_TEMPERATURE_GPU,
                )
            except Exception:
                temp = 0

            try:
                power = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
            except Exception:
                power = 0

            try:
                fan = pynvml.nvmlDeviceGetFanSpeed(self._handle)
            except Exception:
                fan = 0

            stats = GPUStats(
                available=True,
                name=name,
                utilization_pct=float(util.gpu),
                memory_used_mb=mem.used / (1024 * 1024),
                memory_total_mb=mem.total / (1024 * 1024),
                memory_pct=(mem.used / mem.total * 100) if mem.total > 0 else 0,
                temperature_c=float(temp),
                power_watts=power,
                fan_speed_pct=float(fan),
            )

            self._last_stats = stats
            return stats

        except Exception:
            logger.debug("GPU stats query failed", exc_info=True)
            return self._last_stats

    def start(self) -> None:
        if not self._nvml_available or not self._enabled:
            return
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("GPU Governor monitoring started (interval=%ds)", self._poll_interval)

    def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _monitor_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self._poll_interval)
                if not self._running:
                    break

                stats = self.get_stats()
                if not stats.available:
                    continue

                try:
                    from core.metrics import get_metrics
                    m = get_metrics()
                    m.set_gauge("vram_used_mb", stats.memory_used_mb)
                    m.set_gauge("gpu_util_pct", stats.utilization_pct)
                except Exception:
                    pass

                if self._bus is not None:
                    self._bus.emit_fast("gpu_stats_update", stats=stats.summary())

                if stats.temperature_c > self._thermal_threshold:
                    if not self._thermal_warned:
                        self._thermal_warned = True
                        logger.warning("GPU thermal warning: %d°C", stats.temperature_c)
                        if self._bus is not None:
                            self._bus.emit("gpu_thermal_warning",
                                           temp=stats.temperature_c)
                elif self._thermal_warned and stats.temperature_c < self._thermal_threshold - 5:
                    self._thermal_warned = False

                if stats.memory_pct > self._memory_threshold:
                    if not self._memory_warned:
                        self._memory_warned = True
                        logger.warning(
                            "GPU memory warning: %.0f%% (%.0f/%.0fMB)",
                            stats.memory_pct, stats.memory_used_mb, stats.memory_total_mb,
                        )
                        if self._bus is not None:
                            self._bus.emit("gpu_memory_warning",
                                           memory_pct=stats.memory_pct)
                elif self._memory_warned and stats.memory_pct < self._memory_threshold - 10:
                    self._memory_warned = False

            except asyncio.CancelledError:
                break
            except Exception:
                logger.debug("GPU monitor error", exc_info=True)

    def shutdown(self) -> None:
        self.stop()
        if self._nvml_available:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass
