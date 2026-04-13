"""
ATOM -- Hardware profile helpers for Apple Silicon-aware runtime decisions.
"""

from __future__ import annotations

import platform
import subprocess
from typing import Any

import psutil


def _chip_name() -> str:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        name = (result.stdout or "").strip()
        if name:
            return name
    except Exception:
        pass
    try:
        model = platform.uname().machine
        return f"Apple Silicon ({model})" if model else "Apple Silicon"
    except Exception:
        return "Apple Silicon"


def _ram_class(total_gb: float) -> str:
    if total_gb >= 64:
        return "ultra"
    if total_gb >= 32:
        return "high"
    if total_gb >= 16:
        return "balanced"
    return "compact"


def get_hardware_profile(*, silicon_stats: Any | None = None) -> dict[str, Any]:
    """Return a lightweight hardware profile for runtime state + mode reasoning."""
    mem = psutil.virtual_memory()
    total_gb = round(mem.total / (1024 ** 3), 1)
    battery = psutil.sensors_battery()
    battery_pct = float(getattr(battery, "percent", 100.0) or 100.0)
    charging = bool(getattr(battery, "power_plugged", False)) if battery is not None else False

    gpu_name = ""
    gpu_cores = 0
    thermal_pressure = "unknown"
    cpu_temp_c = 0.0
    on_battery = battery is not None and not charging
    power_watts = 0.0
    throttled = False
    memory_total_mb = float(mem.total / (1024 ** 2))
    memory_used_mb = float((mem.total - mem.available) / (1024 ** 2))
    memory_available_mb = float(mem.available / (1024 ** 2))

    if silicon_stats is not None:
        gpu_name = str(getattr(silicon_stats, "gpu_name", "") or "")
        gpu_cores = int(getattr(silicon_stats, "gpu_cores", 0) or 0)
        thermal_pressure = str(getattr(silicon_stats, "thermal_pressure", "unknown") or "unknown")
        cpu_temp_c = float(getattr(silicon_stats, "cpu_temp_c", 0.0) or 0.0)
        on_battery = bool(getattr(silicon_stats, "on_battery", on_battery))
        power_watts = float(getattr(silicon_stats, "power_watts", 0.0) or 0.0)
        throttled = bool(getattr(silicon_stats, "is_throttled", False))
        battery_pct = float(getattr(silicon_stats, "battery_pct", battery_pct) or battery_pct)
        memory_total_mb = float(getattr(silicon_stats, "memory_total_mb", memory_total_mb) or memory_total_mb)
        memory_used_mb = float(getattr(silicon_stats, "memory_used_mb", memory_used_mb) or memory_used_mb)
        memory_available_mb = float(
            getattr(silicon_stats, "memory_available_mb", memory_available_mb) or memory_available_mb
        )

    return {
        "chip": _chip_name(),
        "chip_class": "apple_silicon" if platform.system() == "Darwin" and platform.machine() == "arm64" else platform.machine(),
        "ram_class": _ram_class(total_gb),
        "ram_total_gb": total_gb,
        "battery_state": "battery" if on_battery else "plugged_in",
        "battery_pct": round(battery_pct, 1),
        "charging": charging,
        "thermal_headroom": thermal_pressure,
        "gpu_name": gpu_name,
        "gpu_cores": gpu_cores,
        "memory_total_mb": round(memory_total_mb, 1),
        "memory_used_mb": round(memory_used_mb, 1),
        "memory_available_mb": round(memory_available_mb, 1),
        "on_battery": on_battery,
        "power_watts": round(power_watts, 1),
        "cpu_temp_c": round(cpu_temp_c, 1),
        "is_throttled": throttled,
    }
