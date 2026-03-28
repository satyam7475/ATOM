"""
Lightweight system snapshot for adaptive modes and prompts.

Uses psutil when available; never blocks on network. Foreground window is
best-effort (Windows-focused with safe fallbacks).
"""

from __future__ import annotations

import logging
import sys
from typing import Any

logger = logging.getLogger("atom.system_monitor")


def _foreground_title() -> str:
    if sys.platform != "win32":
        return ""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return ""
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 2)
        user32.GetWindowTextW(hwnd, buf, length + 2)
        return (buf.value or "").strip()[:240]
    except Exception:
        return ""


def get_system_state() -> dict[str, Any]:
    """Return a single snapshot: CPU%, RAM%, top apps, foreground window."""
    cpu_pct = 0.0
    ram_pct = 0.0
    top_apps: list[dict[str, Any]] = []
    try:
        import psutil
        cpu_pct = float(psutil.cpu_percent(interval=0))
        ram_pct = float(psutil.virtual_memory().percent)
        by_cpu: list[tuple[float, str]] = []
        for p in psutil.process_iter(["name", "cpu_percent"]):
            try:
                nm = (p.info.get("name") or "")[:64]
                c = float(p.info.get("cpu_percent") or 0)
                if nm and c > 0.5:
                    by_cpu.append((c, nm))
            except Exception:
                continue
        by_cpu.sort(reverse=True)
        seen: set[str] = set()
        for c, nm in by_cpu[:24]:
            if nm.lower() in seen:
                continue
            seen.add(nm.lower())
            top_apps.append({"name": nm, "cpu_percent": round(c, 2)})
            if len(top_apps) >= 8:
                break
    except Exception:
        logger.debug("system_monitor psutil unavailable", exc_info=True)

    fg = _foreground_title()
    state = {
        "cpu_percent": round(cpu_pct, 2),
        "ram_percent": round(ram_pct, 2),
        "active_applications": top_apps,
        "foreground_window_title": fg,
        "ts": __import__("time").time(),
    }
    try:
        logger.info(
            "v7_system_state cpu=%.1f ram=%.1f fg=%s apps=%d",
            cpu_pct,
            ram_pct,
            (fg[:40] + "…") if len(fg) > 40 else fg,
            len(top_apps),
        )
    except Exception:
        pass
    return state


class SystemMonitor:
    """Injectable monitor; callers can cache snapshots to avoid repeated psutil."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    def get_system_state(self) -> dict[str, Any]:
        return get_system_state()
