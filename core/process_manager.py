"""ATOM -- Process Manager (AI OS Kernel Service).

Provides OS-level process management capabilities:
  - List top processes by CPU/memory
  - Kill processes by name (forceful, unlike close_app)
  - Comprehensive system resource snapshots
  - Resource trend analysis over time
  - Foreground app history tracking

Uses psutil (already a dependency).

Sprint P4.7 (Apr 26 2026): the Win32 ``EnumWindows`` path in
``get_open_windows`` was removed; on macOS we delegate to AppKit /
NSWorkspace via :py:mod:`context.context_darwin` for window listing
when needed. ``get_open_windows`` returns ``[]`` here and the
JARVIS-level "what's open" surface lives in
:py:mod:`core.platform_adapter` and :py:mod:`context.screen_reader`.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from datetime import datetime
import psutil

logger = logging.getLogger("atom.process_mgr")


class ProcessManager:
    """Lightweight process manager for ATOM AI OS."""

    def __init__(self, history_size: int = 50) -> None:
        self._app_history: deque[tuple[str, float]] = deque(maxlen=history_size)
        self._resource_snapshots: deque[dict] = deque(maxlen=60)

    def get_top_processes(self, n: int = 8, sort_by: str = "cpu") -> list[dict]:
        """Return top N processes sorted by CPU or memory usage."""
        try:
            procs = []
            for p in psutil.process_iter(
                ["pid", "name", "cpu_percent", "memory_percent", "status"]
            ):
                try:
                    info = p.info
                    if info.get("name") in ("System Idle Process", "Idle", ""):
                        continue
                    procs.append(info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            key = "cpu_percent" if sort_by == "cpu" else "memory_percent"
            procs.sort(key=lambda x: x.get(key, 0) or 0, reverse=True)
            return procs[:n]
        except Exception:
            logger.debug("Failed to get processes", exc_info=True)
            return []

    def kill_process(self, name: str) -> tuple[bool, str]:
        """Kill all processes matching name. Returns (success, message)."""
        try:
            killed = 0
            name_lower = name.lower().replace(".exe", "").strip()
            for p in psutil.process_iter(["pid", "name"]):
                try:
                    pname = (p.info.get("name") or "").lower().replace(".exe", "")
                    if name_lower == pname or name_lower in pname:
                        p.terminate()
                        killed += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            if killed > 0:
                msg = (f"Terminated {killed} "
                       f"process{'es' if killed > 1 else ''} "
                       f"matching '{name}'.")
                logger.info(msg)
                return True, msg
            return False, f"No running process found matching '{name}'."
        except Exception as e:
            return False, f"Failed to kill process: {e}"

    def get_resource_summary(self) -> dict:
        """Comprehensive system resource snapshot."""
        try:
            cpu = psutil.cpu_percent(interval=0.3)
            cpu_count = psutil.cpu_count()
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            net = psutil.net_io_counters()
            bat = psutil.sensors_battery()
            uptime_s = time.time() - psutil.boot_time()

            snapshot = {
                "cpu_percent": cpu,
                "cpu_cores": cpu_count,
                "ram_percent": mem.percent,
                "ram_used_gb": round(mem.used / (1024 ** 3), 1),
                "ram_total_gb": round(mem.total / (1024 ** 3), 1),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024 ** 3), 1),
                "net_sent_mb": round(net.bytes_sent / (1024 ** 2), 1),
                "net_recv_mb": round(net.bytes_recv / (1024 ** 2), 1),
                "battery_percent": bat.percent if bat else None,
                "battery_plugged": bat.power_plugged if bat else None,
                "uptime_hours": round(uptime_s / 3600, 1),
                "process_count": len(list(psutil.pids())),
                "ts": time.time(),
            }
            self._resource_snapshots.append(snapshot)
            return snapshot
        except Exception:
            logger.debug("Resource summary failed", exc_info=True)
            return {}

    def get_resource_trend(self) -> str:
        """Analyze resource trends from recent snapshots."""
        if len(self._resource_snapshots) < 3:
            return "Not enough data for trend analysis yet. I'll have more after a few minutes."

        recent = list(self._resource_snapshots)[-10:]
        cpu_avg = sum(s.get("cpu_percent", 0) for s in recent) / len(recent)
        ram_avg = sum(s.get("ram_percent", 0) for s in recent) / len(recent)
        cpu_trend = recent[-1].get("cpu_percent", 0) - recent[0].get("cpu_percent", 0)
        ram_trend = recent[-1].get("ram_percent", 0) - recent[0].get("ram_percent", 0)

        def _trend_word(delta: float) -> str:
            if delta > 5:
                return "rising"
            if delta < -5:
                return "falling"
            return "stable"

        parts = [
            f"CPU avg {cpu_avg:.0f}% ({_trend_word(cpu_trend)})",
            f"RAM avg {ram_avg:.0f}% ({_trend_word(ram_trend)})",
        ]
        if cpu_avg > 80:
            parts.append("WARNING: sustained high CPU load")
        if ram_avg > 85:
            parts.append("WARNING: memory pressure detected")
        return " | ".join(parts)

    def record_app_switch(self, app_name: str) -> None:
        self._app_history.append((app_name, time.time()))

    def get_app_history(self, n: int = 10) -> list[tuple[str, str]]:
        result = []
        for app, ts in list(self._app_history)[-n:]:
            time_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            result.append((app, time_str))
        return result

    def format_top_processes(self, n: int = 5, sort_by: str = "cpu") -> str:
        procs = self.get_top_processes(n, sort_by)
        if not procs:
            return "Couldn't retrieve process list."
        parts = []
        for p in procs:
            name = p.get("name", "?")
            cpu = p.get("cpu_percent", 0) or 0
            mem = p.get("memory_percent", 0) or 0
            parts.append(f"{name}: {cpu:.0f}% CPU, {mem:.1f}% RAM")
        return "Top processes: " + " | ".join(parts)

    def format_resource_summary(self) -> str:
        s = self.get_resource_summary()
        if not s:
            return "Couldn't get system resources."
        parts = [
            f"CPU {s['cpu_percent']:.0f}% ({s['cpu_cores']} cores)",
            f"RAM {s['ram_percent']:.0f}% ({s['ram_used_gb']}/{s['ram_total_gb']}GB)",
            f"Disk {s['disk_percent']:.0f}% used ({s['disk_free_gb']}GB free)",
            f"Uptime {s['uptime_hours']:.1f}h",
            f"{s['process_count']} processes",
        ]
        if s.get("battery_percent") is not None:
            plug = "charging" if s["battery_plugged"] else "on battery"
            parts.append(f"Battery {s['battery_percent']:.0f}% ({plug})")
        return " | ".join(parts)

    def format_app_history(self, n: int = 8) -> str:
        history = self.get_app_history(n)
        if not history:
            return "No app switch history yet."
        parts = [f"Recent app switches ({len(history)}):"]
        for app, ts in history:
            parts.append(f"  {ts} - {app}")
        return " ".join(parts)

    def get_open_windows(self) -> list[str]:
        """Return open window titles. macOS implementation uses
        :py:mod:`context.context_darwin` (Quartz / NSWorkspace) when
        available; on platforms where it isn't, we return ``[]``
        rather than fail loudly so callers can compose voice output.

        Sprint P4.7 (Apr 26 2026): replaces the dead Win32 EnumWindows
        branch.
        """
        try:
            from context.context_darwin import quartz_window_titles
            return list(quartz_window_titles() or [])
        except Exception:
            logger.debug("Window enumeration unavailable on this platform",
                         exc_info=True)
            return []

    def format_open_windows(self) -> str:
        """Format visible windows for voice output."""
        windows = self.get_open_windows()
        if not windows:
            return "Couldn't list open windows."
        unique = []
        seen: set[str] = set()
        for w in windows:
            short = w[:50]
            if short not in seen:
                seen.add(short)
                unique.append(short)
        parts = [f"You have {len(unique)} windows open:"]
        for w in unique[:8]:
            parts.append(f"  {w}")
        if len(unique) > 8:
            parts.append(f"  and {len(unique) - 8} more.")
        return " ".join(parts)

    def get_full_system_report(self) -> str:
        """Comprehensive system report for voice output.

        Combines resource summary, top processes, and open windows
        into a single spoken summary.
        """
        parts: list[str] = ["System report, Boss."]

        resource = self.format_resource_summary()
        if resource:
            parts.append(resource)

        top = self.get_top_processes(5, "cpu")
        if top:
            heavy = [p for p in top if (p.get("cpu_percent", 0) or 0) > 5]
            if heavy:
                names = [f"{p.get('name', '?')} at {p.get('cpu_percent', 0):.0f}%"
                         for p in heavy[:3]]
                parts.append("Heaviest apps: " + ", ".join(names))

        windows = self.get_open_windows()
        if windows:
            unique_apps: set[str] = set()
            for w in windows:
                app = w.split(" - ")[-1].strip() if " - " in w else w
                if len(app) > 3:
                    unique_apps.add(app[:30])
            if unique_apps:
                parts.append(f"Open apps: {', '.join(list(unique_apps)[:6])}")

        trend = self.get_resource_trend()
        if trend and "Not enough" not in trend:
            parts.append("Trend: " + trend)

        return " ".join(parts)
