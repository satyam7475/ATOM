"""
ATOM -- System Awareness Daemon (AI OS Kernel).

Lightweight background loop that monitors system-level changes
and emits events for proactive ATOM responses:
  - Active app switches
  - Network connectivity changes
  - Battery state transitions (plugged/unplugged, critical levels)
  - Bluetooth device connections/disconnections

Zero external dependencies beyond psutil (already in requirements).
Runs as an asyncio task, polls every 10 seconds.
"""

from __future__ import annotations

import asyncio
import logging
import socket
from typing import TYPE_CHECKING

logger = logging.getLogger("atom.watcher")

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus

_NETWORK_CHECK_HOSTS = [
    ("8.8.8.8", 53),
    ("1.1.1.1", 53),
]


def _get_foreground_app() -> str:
    try:
        import ctypes
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        hwnd = user32.GetForegroundWindow()
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value
        if " - " in title:
            return title.rsplit(" - ", 1)[-1].strip()
        return title.strip()
    except Exception:
        return ""


def _check_internet(timeout: float = 2.0) -> bool:
    """Check connectivity against multiple endpoints for reliability."""
    for host, port in _NETWORK_CHECK_HOSTS:
        try:
            socket.create_connection((host, port), timeout=timeout).close()
            return True
        except OSError:
            continue
    return False


class SystemWatcher:
    """Background daemon that detects system state changes.

    PyAudio is initialized once and reused across Bluetooth checks
    to avoid the overhead of repeated init/terminate cycles.
    """

    _BT_KEYWORDS = ("bluetooth", "buds", "airpods", "wireless", "bt ",
                     "headset", "earphone", "jbl", "sony", "oneplus",
                     "mivi", "realme", "jabra", "bose", "blaupunkt")

    def __init__(self, bus: AsyncEventBus, poll_interval: float = 10.0) -> None:
        self._bus = bus
        self._base_interval = poll_interval
        self._interval = poll_interval
        self._stop = False
        self._task: asyncio.Task | None = None

        self._last_app: str = ""
        self._last_online: bool | None = None
        self._last_plugged: bool | None = None
        self._last_battery_level: int | None = None
        self._last_bt_devices: set[str] = set()
        self._bt_check_cycle = 0
        self._net_check_cycle = 0
        self._resource_check_cycle = 0
        self._last_cpu_alert: float = 0
        self._last_ram_alert: float = 0

        self._pa = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop = False
            self._task = asyncio.ensure_future(self._run())
            logger.info("SystemWatcher started (poll=%.0fs)", self._interval)

        self._bus.on("governor_throttle", self._on_governor_throttle)
        self._bus.on("governor_normal", self._on_governor_normal)

    async def _on_governor_throttle(self, **_kw) -> None:
        self._interval = self._base_interval * 3.0
        logger.info("SystemWatcher: governor throttle -> interval %.0fs", self._interval)

    async def _on_governor_normal(self, **_kw) -> None:
        self._interval = self._base_interval
        logger.info("SystemWatcher: governor normal -> interval %.0fs", self._interval)

    def stop(self) -> None:
        self._stop = True
        if self._task and not self._task.done():
            self._task.cancel()
            logger.info("SystemWatcher stopped")
        self._cleanup_pyaudio()

    def _cleanup_pyaudio(self) -> None:
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

    def _get_pyaudio(self):
        """Lazy-init PyAudio and reuse the same instance."""
        if self._pa is None:
            try:
                import pyaudio
                self._pa = pyaudio.PyAudio()
            except Exception:
                return None
        return self._pa

    def _get_audio_devices(self) -> set[str]:
        """List audio input devices via cached PyAudio instance."""
        pa = self._get_pyaudio()
        if pa is None:
            return set()
        try:
            devices = set()
            for i in range(pa.get_device_count()):
                try:
                    info = pa.get_device_info_by_index(i)
                except Exception:
                    continue
                if info.get("maxInputChannels", 0) > 0:
                    name = info.get("name", "").lower()
                    if name:
                        devices.add(name)
            return devices
        except Exception:
            self._cleanup_pyaudio()
            return set()

    async def _run(self) -> None:
        await asyncio.sleep(15)
        try:
            while not self._stop:
                try:
                    self._check_app_change()
                    self._check_battery()

                    self._net_check_cycle += 1
                    if self._net_check_cycle % 3 == 0:
                        await asyncio.get_running_loop().run_in_executor(
                            None, self._check_network)

                    self._bt_check_cycle += 1
                    if self._bt_check_cycle % 6 == 0:
                        await asyncio.get_running_loop().run_in_executor(
                            None, self._check_bluetooth)

                    self._resource_check_cycle += 1
                    if self._resource_check_cycle % 6 == 0:
                        self._check_resource_alerts()

                except Exception:
                    logger.debug("SystemWatcher cycle error", exc_info=True)

                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            pass

    def _check_app_change(self) -> None:
        app = _get_foreground_app()
        if not app or app == self._last_app:
            return
        prev = self._last_app
        self._last_app = app
        if prev:
            self._bus.emit_fast("system_event",
                                kind="app_switch", app=app, prev=prev)
            logger.debug("App switch: %s -> %s", prev, app)

    def _check_network(self) -> None:
        online = _check_internet()
        if self._last_online is None:
            self._last_online = online
            return
        if online != self._last_online:
            self._last_online = online
            if online:
                self._bus.emit_fast("system_event",
                                    kind="network_restored")
                logger.info("Network restored")
            else:
                self._bus.emit_fast("system_event",
                                    kind="network_lost")
                logger.warning("Network connection lost")

    def _check_battery(self) -> None:
        try:
            import psutil
            bat = psutil.sensors_battery()
            if bat is None:
                return

            level = int(bat.percent)
            plugged = bat.power_plugged

            if self._last_plugged is not None and plugged != self._last_plugged:
                if plugged:
                    self._bus.emit_fast("system_event",
                                        kind="power_plugged", level=level)
                    logger.info("Power plugged in (%d%%)", level)
                else:
                    self._bus.emit_fast("system_event",
                                        kind="power_unplugged", level=level)
                    logger.info("Power unplugged (%d%%)", level)

            if self._last_battery_level is not None:
                for threshold in (10, 5):
                    if self._last_battery_level > threshold >= level and not plugged:
                        self._bus.emit_fast("system_event",
                                            kind="battery_critical",
                                            level=level)
                        logger.warning("Battery critical: %d%%", level)

            self._last_plugged = plugged
            self._last_battery_level = level
        except Exception:
            pass

    def _check_bluetooth(self) -> None:
        devices = self._get_audio_devices()
        if not self._last_bt_devices and not devices:
            return

        if self._last_bt_devices and not devices:
            return

        new_devices = devices - self._last_bt_devices
        removed_devices = self._last_bt_devices - devices

        for dev in new_devices:
            if any(kw in dev for kw in self._BT_KEYWORDS):
                self._bus.emit_fast("system_event",
                                    kind="bt_connected", device=dev)
                logger.info("Bluetooth device connected: %s", dev)

        for dev in removed_devices:
            if any(kw in dev for kw in self._BT_KEYWORDS):
                self._bus.emit_fast("system_event",
                                    kind="bt_disconnected", device=dev)
                logger.info("Bluetooth device disconnected: %s", dev)

        self._last_bt_devices = devices

    def _check_resource_alerts(self) -> None:
        """Proactive alerts when CPU or RAM exceed thresholds.

        Alerts are rate-limited to once per 5 minutes per metric
        to avoid flooding the user.
        """
        import time
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=0.5)
            mem = psutil.virtual_memory()
            now = time.time()

            if cpu > 85 and (now - self._last_cpu_alert) > 300:
                self._last_cpu_alert = now
                top = []
                for p in psutil.process_iter(["name", "cpu_percent"]):
                    try:
                        c = p.info.get("cpu_percent", 0) or 0
                        if c > 10:
                            top.append(f"{p.info['name']} at {c:.0f}%")
                    except Exception:
                        pass
                detail = ", ".join(top[:3]) if top else "multiple processes"
                msg = f"Boss, CPU is at {cpu:.0f}% -- {detail}."
                self._bus.emit_fast("system_event",
                                    kind="resource_alert", metric="cpu",
                                    value=cpu, message=msg)
                logger.warning("Resource alert: CPU %.0f%%", cpu)

            if mem.percent > 85 and (now - self._last_ram_alert) > 300:
                self._last_ram_alert = now
                used_gb = round(mem.used / (1024 ** 3), 1)
                total_gb = round(mem.total / (1024 ** 3), 1)
                msg = (f"Boss, RAM is at {mem.percent:.0f}% -- "
                       f"{used_gb}GB of {total_gb}GB used.")
                self._bus.emit_fast("system_event",
                                    kind="resource_alert", metric="ram",
                                    value=mem.percent, message=msg)
                logger.warning("Resource alert: RAM %.0f%%", mem.percent)

        except Exception:
            logger.debug("Resource alert check failed", exc_info=True)
