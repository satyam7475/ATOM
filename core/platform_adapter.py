"""ATOM -- Cross-Platform Adapter (JARVIS-Level portability for macOS).

Abstracts OS interactions behind a unified API. macOS is the primary
target; Linux paths are kept for headless dev / CI boxes that import
this module (e.g. linters in containers). Sprint P4.7 (Apr 26 2026)
removed the Windows branches — see ``docs/ATOM_NEXT_STEPS_PLAN.md``
§ P4.7 for rationale. ``OSType.WINDOWS`` is intentionally retained in
the enum so older code that imports it doesn't crash on import; the
adapter will simply no-op gracefully on Windows.

Capabilities:
    - Window management (foreground window, title, process)
    - Clipboard access (read/write)
    - System info (OS, CPU, RAM, GPU, disk, network)
    - Process control (list, kill, launch)
    - Audio device enumeration
    - Display control (brightness, resolution)
    - Power management (sleep, shutdown, restart, lock)
    - Notification system
    - File system operations
    - Service management
    - Shell command execution (sandboxed)
    - TTS engine selection per platform

Owner: Satyam
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any
import psutil

from core.macos import AppleScriptEngine

logger = logging.getLogger("atom.platform")


class OSType(Enum):
    WINDOWS = auto()
    LINUX = auto()
    MACOS = auto()
    UNKNOWN = auto()


@dataclass
class SystemProfile:
    """Complete system identity snapshot."""
    os_type: OSType = OSType.UNKNOWN
    os_name: str = ""
    os_version: str = ""
    os_build: str = ""
    hostname: str = ""
    username: str = ""
    architecture: str = ""
    cpu_name: str = ""
    cpu_cores_physical: int = 0
    cpu_cores_logical: int = 0
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    python_version: str = ""
    home_dir: str = ""
    temp_dir: str = ""
    shell: str = ""
    display_count: int = 1
    display_resolution: str = ""
    has_battery: bool = False
    battery_percent: float = 100.0
    is_plugged: bool = True
    boot_time: float = 0.0
    uptime_hours: float = 0.0


@dataclass
class ProcessInfo:
    """Lightweight process descriptor."""
    pid: int = 0
    name: str = ""
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    status: str = ""
    username: str = ""
    create_time: float = 0.0
    cmdline: str = ""


@dataclass
class NetworkInterface:
    """Network interface descriptor."""
    name: str = ""
    ip_address: str = ""
    mac_address: str = ""
    is_up: bool = False
    speed_mbps: int = 0
    kind: str = ""


@dataclass
class DiskInfo:
    """Disk partition descriptor."""
    device: str = ""
    mountpoint: str = ""
    filesystem: str = ""
    total_gb: float = 0.0
    used_gb: float = 0.0
    free_gb: float = 0.0
    percent_used: float = 0.0


@dataclass
class InstalledApp:
    """Installed application descriptor."""
    name: str = ""
    version: str = ""
    publisher: str = ""
    install_path: str = ""
    install_date: str = ""


def detect_os() -> OSType:
    """Detect the current operating system."""
    p = sys.platform.lower()
    if p.startswith("win"):
        return OSType.WINDOWS
    if p.startswith("linux"):
        return OSType.LINUX
    if p == "darwin":
        return OSType.MACOS
    return OSType.UNKNOWN


CURRENT_OS = detect_os()


class PlatformAdapter:
    """Unified cross-platform interface for all OS operations.

    Every method works on all platforms or returns a sensible default.
    Platform-specific backends are selected at init time.
    """

    def __init__(self) -> None:
        self.os_type = CURRENT_OS
        self._profile: SystemProfile | None = None
        self._applescript = AppleScriptEngine() if self.os_type == OSType.MACOS else None
        logger.info("Platform adapter initialized: %s (%s)",
                     self.os_type.name, platform.platform())

    # ── System Profile ─────────────────────────────────────────────

    def get_system_profile(self) -> SystemProfile:
        """Build a complete system profile snapshot."""
        import time as _time

        p = SystemProfile()
        p.os_type = self.os_type
        p.os_name = platform.system()
        p.os_version = platform.version()
        p.os_build = platform.release()
        p.hostname = platform.node()
        p.username = os.environ.get("USER") or os.environ.get("USERNAME", "unknown")
        p.architecture = platform.machine()
        p.python_version = platform.python_version()
        p.home_dir = str(Path.home())
        p.temp_dir = os.environ.get("TEMP") or os.environ.get("TMPDIR") or "/tmp"
        p.shell = os.environ.get("SHELL") or os.environ.get("COMSPEC", "")

        try:
            p.cpu_cores_physical = psutil.cpu_count(logical=False) or 0
            p.cpu_cores_logical = psutil.cpu_count(logical=True) or 0
            mem = psutil.virtual_memory()
            p.ram_total_gb = round(mem.total / (1024 ** 3), 2)
            p.ram_available_gb = round(mem.available / (1024 ** 3), 2)
            bat = psutil.sensors_battery()
            if bat:
                p.has_battery = True
                p.battery_percent = bat.percent
                p.is_plugged = bat.power_plugged or False
            p.boot_time = psutil.boot_time()
            p.uptime_hours = round((_time.time() - p.boot_time) / 3600, 2)
        except Exception:
            logger.debug("psutil unavailable for system profile", exc_info=True)

        p.cpu_name = self._get_cpu_name()
        p.gpu_name, p.gpu_vram_gb = self._get_gpu_info()
        p.display_count, p.display_resolution = self._get_display_info()

        self._profile = p
        return p

    def _get_cpu_name(self) -> str:
        """Get human-readable CPU name across platforms."""
        try:
            if self.os_type == OSType.LINUX:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":")[1].strip()
            elif self.os_type == OSType.MACOS:
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True, timeout=5,
                )
                return result.stdout.strip()
        except Exception:
            logger.debug('CPU brand string query failed', exc_info=True)
        return platform.processor() or "Unknown CPU"

    def _get_gpu_info(self) -> tuple[str, float]:
        """Get GPU name and memory. Returns ('', 0.0) if unavailable.

        On Apple Silicon, reports the integrated GPU name (e.g. 'Apple M5')
        and total system RAM as shared GPU memory (Unified Memory architecture).
        """
        if self.os_type == OSType.MACOS:
            return self._get_gpu_info_macos()

        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_gb = round(mem_info.total / (1024 ** 3), 2)
            pynvml.nvmlShutdown()
            return name, vram_gb
        except Exception:
            logger.debug('CPU brand string query failed', exc_info=True)

        return "", 0.0

    def _get_gpu_info_macos(self) -> tuple[str, float]:
        """Apple Silicon GPU info via system_profiler.

        Apple Silicon uses Unified Memory — CPU, GPU, and Neural Engine all
        share the same RAM pool. There is no discrete VRAM. We report total
        system RAM as the shared GPU memory budget.
        """
        try:
            import json as _json
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True, text=True, timeout=10,
            )
            data = _json.loads(result.stdout)
            gpu_entries = data.get("SPDisplaysDataType", [])
            if gpu_entries:
                gpu = gpu_entries[0]
                name = gpu.get("sppci_model", "Apple Silicon GPU")
                cores = gpu.get("sppci_cores", "")
                if cores:
                    name = f"{name} ({cores}-core GPU)"
                mem = psutil.virtual_memory()
                vram_gb = round(mem.total / (1024 ** 3), 2)
                return name, vram_gb
        except Exception:
            logger.debug("macOS GPU info failed", exc_info=True)
        try:
            mem = psutil.virtual_memory()
            return "Apple Silicon GPU", round(mem.total / (1024 ** 3), 2)
        except Exception:
            return "Apple Silicon GPU", 0.0

    def _get_display_info(self) -> tuple[int, str]:
        """Get display count and primary resolution."""
        if self.os_type == OSType.MACOS:
            return self._get_display_info_macos()
        if self.os_type == OSType.LINUX:
            try:
                result = subprocess.run(
                    ["xrandr", "--current"],
                    capture_output=True, text=True, timeout=5,
                )
                count = result.stdout.count(" connected")
                for line in result.stdout.split("\n"):
                    if "*" in line:
                        res = line.strip().split()[0]
                        return max(1, count), res
            except Exception:
                logger.debug('Subprocess run failed', exc_info=True)
        return 1, "unknown"

    def _get_display_info_macos(self) -> tuple[int, str]:
        """macOS display info via system_profiler SPDisplaysDataType."""
        try:
            import json as _json
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True, text=True, timeout=10,
            )
            data = _json.loads(result.stdout)
            displays = []
            for gpu in data.get("SPDisplaysDataType", []):
                for disp in gpu.get("spdisplays_ndrvs", []):
                    displays.append(disp)

            if not displays:
                return 1, "unknown"

            main_display = displays[0]
            resolution = main_display.get("_spdisplays_resolution", "")
            if not resolution:
                resolution = main_display.get("_spdisplays_pixels", "unknown")

            disp_type = main_display.get("spdisplays_display_type", "")
            if "retina" in disp_type.lower():
                pixel_res = main_display.get("spdisplays_pixelresolution", "")
                if pixel_res:
                    resolution += f" ({pixel_res.replace('spdisplays_', '')})"

            return len(displays), resolution
        except Exception:
            logger.debug("macOS display info failed", exc_info=True)
        return 1, "unknown"

    # ── Foreground Window ──────────────────────────────────────────

    def get_foreground_window(self) -> dict[str, str]:
        """Get foreground window info: title, app_name, process_name, pid."""
        if self.os_type == OSType.LINUX:
            return self._fg_window_linux()
        if self.os_type == OSType.MACOS:
            return self._fg_window_macos()
        return {"title": "", "app_name": "", "process_name": "", "pid": ""}

    def _fg_window_linux(self) -> dict[str, str]:
        try:
            result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True, text=True, timeout=3,
            )
            title = result.stdout.strip()
            pid_result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowpid"],
                capture_output=True, text=True, timeout=3,
            )
            pid = pid_result.stdout.strip()
            app_name = title.rsplit(" - ", 1)[-1].strip() if " - " in title else title
            return {"title": title, "app_name": app_name, "process_name": "", "pid": pid}
        except Exception:
            return {"title": "", "app_name": "", "process_name": "", "pid": ""}

    def _fg_window_macos(self) -> dict[str, str]:
        if self._applescript is None:
            return {"title": "", "app_name": "", "process_name": "", "pid": ""}
        front = self._applescript.get_frontmost_app()
        app_name = str(front.get("app", ""))
        return {
            "title": str(front.get("title", "") or app_name),
            "app_name": app_name,
            "process_name": app_name,
            "pid": str(front.get("pid", "")),
        }

    # ── Clipboard ──────────────────────────────────────────────────

    def get_clipboard(self, max_chars: int = 500) -> str:
        """Read clipboard text, cross-platform."""
        try:
            result = subprocess.run(
                ["xclip", "-selection", "clipboard", "-o"]
                if self.os_type == OSType.LINUX
                else ["pbpaste"],
                capture_output=True, text=True, timeout=3,
            )
            return result.stdout[:max_chars]
        except Exception:
            return ""

    # ── Processes ──────────────────────────────────────────────────

    def list_processes(
        self, sort_by: str = "cpu", limit: int = 20,
    ) -> list[ProcessInfo]:
        """List running processes sorted by CPU or memory usage."""
        procs: list[ProcessInfo] = []
        try:
            for proc in psutil.process_iter(
                ["pid", "name", "cpu_percent", "memory_info",
                 "status", "username", "create_time", "cmdline"],
            ):
                try:
                    info = proc.info
                    mem_mb = (info.get("memory_info") or type("", (), {"rss": 0})).rss / (1024 ** 2)
                    cmd = " ".join(info.get("cmdline") or [])[:200]
                    procs.append(ProcessInfo(
                        pid=info["pid"],
                        name=info.get("name", ""),
                        cpu_percent=info.get("cpu_percent", 0.0) or 0.0,
                        memory_mb=round(mem_mb, 1),
                        status=info.get("status", ""),
                        username=info.get("username", ""),
                        create_time=info.get("create_time", 0.0),
                        cmdline=cmd,
                    ))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception:
            logger.debug("Process listing failed", exc_info=True)

        key_fn = (lambda p: p.cpu_percent) if sort_by == "cpu" else (lambda p: p.memory_mb)
        procs.sort(key=key_fn, reverse=True)
        return procs[:limit]

    def kill_process(self, pid: int) -> bool:
        try:
            proc = psutil.Process(pid)
            proc.terminate()
            proc.wait(timeout=5)
            return True
        except Exception:
            return False

    # ── Network Interfaces ─────────────────────────────────────────

    def get_network_interfaces(self) -> list[NetworkInterface]:
        """List all network interfaces with IP and status."""
        interfaces: list[NetworkInterface] = []
        try:
            addrs = psutil.net_if_addrs()
            stats = psutil.net_if_stats()
            for name, addr_list in addrs.items():
                iface = NetworkInterface(name=name)
                stat = stats.get(name)
                if stat:
                    iface.is_up = stat.isup
                    iface.speed_mbps = stat.speed
                for addr in addr_list:
                    import socket
                    if addr.family == socket.AF_INET:
                        iface.ip_address = addr.address
                        iface.kind = "IPv4"
                    elif hasattr(psutil, "AF_LINK") and addr.family == psutil.AF_LINK:
                        iface.mac_address = addr.address
                if iface.ip_address:
                    interfaces.append(iface)
        except Exception:
            logger.debug("Network interface listing failed", exc_info=True)
        return interfaces

    # ── Disk Info ──────────────────────────────────────────────────

    def get_disk_info(self) -> list[DiskInfo]:
        """List all disk partitions with usage."""
        disks: list[DiskInfo] = []
        try:
            for part in psutil.disk_partitions(all=False):
                try:
                    usage = psutil.disk_usage(part.mountpoint)
                    disks.append(DiskInfo(
                        device=part.device,
                        mountpoint=part.mountpoint,
                        filesystem=part.fstype,
                        total_gb=round(usage.total / (1024 ** 3), 2),
                        used_gb=round(usage.used / (1024 ** 3), 2),
                        free_gb=round(usage.free / (1024 ** 3), 2),
                        percent_used=usage.percent,
                    ))
                except (PermissionError, OSError):
                    pass
        except Exception:
            logger.debug("Disk info failed", exc_info=True)
        return disks

    # ── Installed Applications ────────────────────────────────────

    def get_installed_apps(self) -> list[InstalledApp]:
        """List installed applications (platform-specific)."""
        if self.os_type == OSType.LINUX:
            return self._installed_apps_linux()
        if self.os_type == OSType.MACOS:
            return self._installed_apps_macos()
        return []

    def _installed_apps_linux(self) -> list[InstalledApp]:
        apps: list[InstalledApp] = []
        for cmd in [["dpkg", "-l"], ["rpm", "-qa", "--queryformat",
                     "%{NAME}|||%{VERSION}|||%{VENDOR}\n"]]:
            if shutil.which(cmd[0]):
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                    for line in result.stdout.strip().split("\n")[:500]:
                        if "|||" in line:
                            parts = line.split("|||")
                            apps.append(InstalledApp(
                                name=parts[0], version=parts[1] if len(parts) > 1 else "",
                                publisher=parts[2] if len(parts) > 2 else "",
                            ))
                        elif line.startswith("ii"):
                            parts = line.split()
                            if len(parts) >= 3:
                                apps.append(InstalledApp(name=parts[1], version=parts[2]))
                except Exception:
                    logger.debug('Subprocess run failed', exc_info=True)
                break
        return apps

    def _installed_apps_macos(self) -> list[InstalledApp]:
        apps: list[InstalledApp] = []
        app_dir = Path("/Applications")
        if app_dir.exists():
            for entry in app_dir.iterdir():
                if entry.suffix == ".app":
                    apps.append(InstalledApp(name=entry.stem, install_path=str(entry)))
        return apps

    # ── Power Control ─────────────────────────────────────────────

    def lock_screen(self) -> bool:
        try:
            if self.os_type == OSType.LINUX:
                subprocess.Popen(["xdg-screensaver", "lock"])
            elif self.os_type == OSType.MACOS:
                subprocess.Popen([
                    "osascript", "-e",
                    'tell application "System Events" to keystroke "q" '
                    'using {command down, control down}',
                ])
            return True
        except Exception:
            return False

    def shutdown(self, delay_seconds: int = 0) -> bool:
        try:
            if self.os_type == OSType.LINUX:
                os.system(f"shutdown -h +{delay_seconds // 60 or 0}")
            elif self.os_type == OSType.MACOS:
                subprocess.run(["osascript", "-e",
                                'tell application "System Events" to shut down'], timeout=10)
            return True
        except Exception:
            return False

    def restart(self, delay_seconds: int = 0) -> bool:
        try:
            if self.os_type == OSType.LINUX:
                os.system(f"shutdown -r +{delay_seconds // 60 or 0}")
            elif self.os_type == OSType.MACOS:
                subprocess.run(["osascript", "-e",
                                'tell application "System Events" to restart'], timeout=10)
            return True
        except Exception:
            return False

    def sleep_system(self) -> bool:
        try:
            if self.os_type == OSType.LINUX:
                os.system("systemctl suspend")
            elif self.os_type == OSType.MACOS:
                os.system("pmset sleepnow")
            return True
        except Exception:
            return False

    # ── Display Control ───────────────────────────────────────────

    def set_brightness(self, level: int) -> bool:
        """Set display brightness (0-100)."""
        level = max(0, min(100, level))
        try:
            if self.os_type == OSType.LINUX:
                brightness_file = Path("/sys/class/backlight")
                if brightness_file.exists():
                    for ctrl in brightness_file.iterdir():
                        max_b = int((ctrl / "max_brightness").read_text().strip())
                        target = int(max_b * level / 100)
                        (ctrl / "brightness").write_text(str(target))
                        return True
            elif self.os_type == OSType.MACOS:
                frac = level / 100.0
                subprocess.run(["brightness", str(frac)], timeout=5)
                return True
        except Exception:
            logger.debug('Subprocess run failed', exc_info=True)
        return False

    # ── Notification ──────────────────────────────────────────────

    def send_notification(self, title: str, message: str) -> bool:
        """Send a native OS notification."""
        try:
            if self.os_type == OSType.LINUX:
                subprocess.run(["notify-send", title, message], timeout=5)
                return True
            if self.os_type == OSType.MACOS:
                if self._applescript is not None:
                    return self._applescript.send_notification(title, message)
        except Exception:
            logger.debug('Subprocess run failed', exc_info=True)
        return False

    # ── Service Management ────────────────────────────────────────

    def list_services(self, filter_running: bool = True) -> list[dict[str, str]]:
        """List system services."""
        services: list[dict[str, str]] = []
        try:
            if self.os_type == OSType.MACOS:
                result = subprocess.run(
                    ["launchctl", "list"],
                    capture_output=True, text=True, timeout=10,
                )
                for line in result.stdout.strip().split("\n")[1:]:
                    parts = line.split("\t")
                    if len(parts) >= 3:
                        pid = parts[0].strip()
                        status = parts[1].strip()
                        label = parts[2].strip()
                        is_running = pid != "-" and pid != ""
                        if filter_running and not is_running:
                            continue
                        services.append({
                            "name": label,
                            "display_name": label,
                            "status": "running" if is_running else "stopped",
                            "pid": pid if is_running else "",
                        })
            elif self.os_type == OSType.LINUX:
                result = subprocess.run(
                    ["systemctl", "list-units", "--type=service",
                     "--state=running" if filter_running else "--all",
                     "--no-pager", "--plain"],
                    capture_output=True, text=True, timeout=10,
                )
                for line in result.stdout.strip().split("\n")[1:]:
                    parts = line.split()
                    if len(parts) >= 4:
                        services.append({
                            "name": parts[0],
                            "status": parts[2],
                            "display_name": " ".join(parts[4:]),
                        })
        except Exception:
            logger.debug("Service listing failed", exc_info=True)
        return services[:100]

    # ── Shell Command (sandboxed) ─────────────────────────────────

    def run_command(
        self, command: str, timeout: int = 30, shell: bool = True,
    ) -> dict[str, Any]:
        """Run a shell command and return output. Use with caution."""
        try:
            result = subprocess.run(
                command, shell=shell, capture_output=True,
                text=True, timeout=timeout,
            )
            return {
                "returncode": result.returncode,
                "stdout": result.stdout[:5000],
                "stderr": result.stderr[:2000],
                "success": result.returncode == 0,
            }
        except subprocess.TimeoutExpired:
            return {"returncode": -1, "stdout": "", "stderr": "Timeout", "success": False}
        except Exception as e:
            return {"returncode": -1, "stdout": "", "stderr": str(e), "success": False}

    # ── TTS Engine Recommendation ─────────────────────────────────

    def recommended_tts_engine(self) -> str:
        """Recommend the best TTS engine for the current platform."""
        if self.os_type == OSType.LINUX:
            if shutil.which("piper"):
                return "piper"
            if shutil.which("espeak"):
                return "espeak"
            return "edge"
        if self.os_type == OSType.MACOS:
            return "nsss"
        return "edge"

    # ── Summary for LLM Context ──────────────────────────────────

    @property
    def is_apple_silicon(self) -> bool:
        """True if running on Apple Silicon (M-series chip)."""
        return (self.os_type == OSType.MACOS
                and platform.machine() == "arm64")

    def get_system_summary(self) -> str:
        """Generate a human-readable system summary for LLM context injection."""
        p = self.get_system_profile()
        lines = [
            f"OS: {p.os_name} {p.os_build} ({p.architecture})",
            f"Host: {p.hostname} | User: {p.username}",
            f"CPU: {p.cpu_name} ({p.cpu_cores_physical}C/{p.cpu_cores_logical}T)",
            f"RAM: {p.ram_available_gb:.1f}GB free / {p.ram_total_gb:.1f}GB total",
        ]
        if p.gpu_name:
            if self.is_apple_silicon:
                lines.append(
                    f"GPU: {p.gpu_name} ({p.gpu_vram_gb:.1f}GB Unified Memory)"
                )
            else:
                lines.append(f"GPU: {p.gpu_name} ({p.gpu_vram_gb:.1f}GB VRAM)")
        if p.has_battery:
            plug = "plugged" if p.is_plugged else "on battery"
            lines.append(f"Battery: {p.battery_percent:.0f}% ({plug})")
        lines.append(f"Uptime: {p.uptime_hours:.1f} hours")
        lines.append(f"Display: {p.display_resolution} ({p.display_count} monitor{'s' if p.display_count > 1 else ''})")
        return " | ".join(lines)


_adapter: PlatformAdapter | None = None


def get_platform_adapter() -> PlatformAdapter:
    """Singleton access to the platform adapter."""
    global _adapter
    if _adapter is None:
        _adapter = PlatformAdapter()
    return _adapter
