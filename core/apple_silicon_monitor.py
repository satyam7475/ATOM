"""
Apple Silicon hardware monitoring for ATOM.

Replaces pynvml on Apple Silicon Macs. Apple Silicon has no discrete GPU —
CPU, GPU, and Neural Engine share Unified Memory. This module provides
real-time monitoring using macOS-native commands (no sudo required).

Data sources:
    psutil            — Unified Memory, CPU utilization, battery
    system_profiler   — GPU name/core count (cached, one-time ~350ms)
    pmset -g therm    — Thermal throttling state (CPU_Speed_Limit)
    sysctl            — CPU thermal level (approximate temperature)

Emits nothing on its own — SiliconGovernor wraps this and emits events.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("atom.apple_silicon_monitor")

_GPU_NAME_CACHE: Optional[str] = None
_GPU_CORE_COUNT_CACHE: Optional[int] = None


def is_apple_silicon() -> bool:
    """Detect Apple Silicon (M1/M2/M3/M4/M5) via arm64 architecture."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def get_apple_silicon_memory_mb() -> tuple:
    """Return (used_mb, total_mb) for Unified Memory.

    Drop-in replacement for get_nvml_vram_mb() on Apple Silicon.
    CPU and GPU share the same memory pool, so system memory
    stats ARE the GPU memory stats.
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        used = (mem.total - mem.available) / (1024 * 1024)
        total = mem.total / (1024 * 1024)
        return used, total
    except Exception:
        return 0.0, 0.0


@dataclass
class AppleSiliconStats:
    """Apple Silicon hardware state snapshot."""

    available: bool = True
    gpu_name: str = "Apple Silicon GPU"
    gpu_cores: int = 0

    memory_total_mb: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    memory_pct: float = 0.0

    # "nominal" | "moderate" | "heavy" | "critical"
    thermal_pressure: str = "nominal"
    cpu_temp_c: float = 0.0

    power_watts: float = 0.0
    battery_pct: float = 100.0
    on_battery: bool = False

    cpu_pct: float = 0.0

    @property
    def is_throttled(self) -> bool:
        return self.thermal_pressure in ("heavy", "critical")

    @property
    def unified_memory(self) -> bool:
        return True

    def summary(self) -> str:
        throttle = " [THROTTLED]" if self.is_throttled else ""
        temp_str = f" | Temp: ~{self.cpu_temp_c:.0f}°C" if self.cpu_temp_c > 0 else ""
        power_str = ""
        if self.on_battery:
            power_str = f" | Battery: {self.battery_pct:.0f}%"
        return (
            f"GPU: {self.gpu_name} (Unified Memory) | "
            f"Mem: {self.memory_used_mb:.0f}/{self.memory_total_mb:.0f}MB "
            f"({self.memory_pct:.0f}%) | "
            f"CPU: {self.cpu_pct:.0f}% | "
            f"Thermal: {self.thermal_pressure}"
            f"{temp_str}{power_str}{throttle}"
        )


class AppleSiliconMonitor:
    """Non-sudo Apple Silicon hardware monitor.

    Collects thermal, memory, and power telemetry using macOS-native
    data sources. Designed as a backend for SiliconGovernor.
    """

    def __init__(self) -> None:
        self._gpu_name: str = "Apple Silicon GPU"
        self._gpu_cores: int = 0
        self._last_stats = AppleSiliconStats()
        self._last_poll: float = 0.0
        self._init_gpu_info()

    def _init_gpu_info(self) -> None:
        """Cache GPU name + core count from system_profiler (one-time, ~350ms)."""
        global _GPU_NAME_CACHE, _GPU_CORE_COUNT_CACHE

        if _GPU_NAME_CACHE is not None:
            self._gpu_name = _GPU_NAME_CACHE
            self._gpu_cores = _GPU_CORE_COUNT_CACHE or 0
            return

        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                displays = data.get("SPDisplaysDataType", [])
                if displays:
                    gpu = displays[0]
                    self._gpu_name = gpu.get("sppci_model", "Apple Silicon GPU")
                    cores_str = gpu.get("sppci_cores", "")
                    if cores_str:
                        try:
                            self._gpu_cores = int(cores_str)
                        except ValueError:
                            pass

                    _GPU_NAME_CACHE = self._gpu_name
                    _GPU_CORE_COUNT_CACHE = self._gpu_cores
                    logger.info(
                        "Apple Silicon: %s (%d GPU cores)",
                        self._gpu_name, self._gpu_cores,
                    )
        except Exception:
            logger.debug("system_profiler GPU query failed", exc_info=True)

    # ------------------------------------------------------------------
    # Individual sensor reads — each tolerant of failure
    # ------------------------------------------------------------------

    def _read_memory(self) -> dict:
        """Unified Memory stats via psutil."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                "total_mb": mem.total / (1024 * 1024),
                "used_mb": (mem.total - mem.available) / (1024 * 1024),
                "available_mb": mem.available / (1024 * 1024),
                "pct": mem.percent,
            }
        except Exception:
            return {"total_mb": 0, "used_mb": 0, "available_mb": 0, "pct": 0}

    def _read_thermal_pressure(self) -> str:
        """Read thermal throttling state from pmset (no sudo required).

        Parses CPU_Speed_Limit from ``pmset -g therm``:
            100   → nominal
            70-99 → moderate
            40-69 → heavy
            <40   → critical
        """
        try:
            result = subprocess.run(
                ["pmset", "-g", "therm"],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "CPU_Speed_Limit" in line:
                        parts = line.strip().split()
                        if parts:
                            try:
                                pct = int(parts[-1])
                                if pct >= 100:
                                    return "nominal"
                                if pct >= 70:
                                    return "moderate"
                                if pct >= 40:
                                    return "heavy"
                                return "critical"
                            except ValueError:
                                pass
        except Exception:
            pass
        return "nominal"

    def _read_cpu_temp(self) -> float:
        """Approximate SoC temperature from thermal level (no sudo required).

        ``sysctl machdep.xcpm.cpu_thermal_level`` returns a 0-100 value.
        Linearly mapped: 0 → 35 °C, 100 → 105 °C.
        Returns 0.0 if unavailable (sensor not exposed on this OS version).
        """
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.xcpm.cpu_thermal_level"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0 and result.stdout.strip():
                level = int(result.stdout.strip())
                return 35.0 + (level * 0.70)
        except Exception:
            pass
        return 0.0

    def _read_power(self) -> dict:
        """Battery status via psutil."""
        try:
            import psutil
            batt = psutil.sensors_battery()
            if batt is not None:
                return {
                    "battery_pct": batt.percent,
                    "on_battery": not batt.power_plugged,
                    "power_watts": 0.0,
                }
        except Exception:
            pass
        return {"battery_pct": 100.0, "on_battery": False, "power_watts": 0.0}

    def _read_cpu_percent(self) -> float:
        """CPU utilization as a proxy for chip load (GPU shares the SoC)."""
        try:
            import psutil
            return psutil.cpu_percent(interval=None)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_stats(self) -> AppleSiliconStats:
        """Collect a full hardware snapshot (~15-20ms total)."""
        mem = self._read_memory()
        power = self._read_power()

        stats = AppleSiliconStats(
            available=True,
            gpu_name=self._gpu_name,
            gpu_cores=self._gpu_cores,
            memory_total_mb=mem["total_mb"],
            memory_used_mb=mem["used_mb"],
            memory_available_mb=mem["available_mb"],
            memory_pct=mem["pct"],
            thermal_pressure=self._read_thermal_pressure(),
            cpu_temp_c=self._read_cpu_temp(),
            power_watts=power["power_watts"],
            battery_pct=power["battery_pct"],
            on_battery=power["on_battery"],
            cpu_pct=self._read_cpu_percent(),
        )

        self._last_stats = stats
        self._last_poll = time.monotonic()
        return stats

    @property
    def last_stats(self) -> AppleSiliconStats:
        return self._last_stats

    @property
    def gpu_name(self) -> str:
        return self._gpu_name

    @property
    def gpu_cores(self) -> int:
        return self._gpu_cores


__all__ = [
    "is_apple_silicon",
    "get_apple_silicon_memory_mb",
    "AppleSiliconStats",
    "AppleSiliconMonitor",
]
