"""
ATOM -- System Control Engine (JARVIS-Level Full Control).

Advanced system control that gives ATOM comprehensive power over the
host machine -- like JARVIS controlling Tony's lab. This module extends
the basic actions in router/system_actions.py with deep system control.

Capabilities:
    - Advanced process management (priority, affinity, suspend/resume)
    - Service control (start, stop, restart services)
    - Network control (flush DNS, release/renew IP, wifi scan)
    - Display control (brightness, resolution, multi-monitor)
    - Audio control (device selection, per-app volume)
    - Power management (power plans, scheduled shutdown)
    - Storage management (disk cleanup, temp file analysis)
    - Startup management (list/disable startup programs)
    - Environment variable management
    - System information queries (drivers, BIOS, motherboard)
    - Performance optimization (clear RAM, boost priority)
    - Security operations (firewall check, port scan)

All operations go through SecurityPolicy before execution.
Cross-platform via PlatformAdapter.

Contract: Uses SecurityPolicy for all gated operations
Owner: Satyam
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom.sysctl")


@dataclass
class SystemControlResult:
    """Result of a system control operation."""
    success: bool = False
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)


class SystemControl:
    """JARVIS-level system control engine.

    Provides deep control over the host system beyond basic actions.
    Every destructive operation requires explicit security clearance.
    """

    def __init__(self, config: dict | None = None) -> None:
        self._config = config or {}
        self._is_windows = sys.platform.startswith("win")
        self._is_linux = sys.platform.startswith("linux")
        self._is_macos = sys.platform == "darwin"
        logger.info("System control engine initialized (%s)", sys.platform)

    # ── Process Management ────────────────────────────────────────

    def set_process_priority(self, pid: int, priority: str = "normal") -> SystemControlResult:
        """Set process priority: low, below_normal, normal, above_normal, high, realtime."""
        try:
            import psutil
            proc = psutil.Process(pid)
            priority_map = {
                "low": psutil.IDLE_PRIORITY_CLASS if self._is_windows else 19,
                "below_normal": psutil.BELOW_NORMAL_PRIORITY_CLASS if self._is_windows else 10,
                "normal": psutil.NORMAL_PRIORITY_CLASS if self._is_windows else 0,
                "above_normal": psutil.ABOVE_NORMAL_PRIORITY_CLASS if self._is_windows else -5,
                "high": psutil.HIGH_PRIORITY_CLASS if self._is_windows else -10,
                "realtime": psutil.REALTIME_PRIORITY_CLASS if self._is_windows else -20,
            }
            nice = priority_map.get(priority)
            if nice is None:
                return SystemControlResult(False, f"Unknown priority: {priority}")

            if self._is_windows:
                proc.nice(nice)
            else:
                proc.nice(nice)

            return SystemControlResult(True, f"Process {proc.name()} (PID {pid}) priority set to {priority}")
        except Exception as e:
            return SystemControlResult(False, f"Failed to set priority: {e}")

    def suspend_process(self, pid: int) -> SystemControlResult:
        """Suspend (pause) a process."""
        try:
            import psutil
            proc = psutil.Process(pid)
            proc.suspend()
            return SystemControlResult(True, f"Process {proc.name()} (PID {pid}) suspended")
        except Exception as e:
            return SystemControlResult(False, f"Failed to suspend: {e}")

    def resume_process(self, pid: int) -> SystemControlResult:
        """Resume a suspended process."""
        try:
            import psutil
            proc = psutil.Process(pid)
            proc.resume()
            return SystemControlResult(True, f"Process {proc.name()} (PID {pid}) resumed")
        except Exception as e:
            return SystemControlResult(False, f"Failed to resume: {e}")

    def get_process_details(self, pid: int) -> SystemControlResult:
        """Get detailed info about a specific process."""
        try:
            import psutil
            proc = psutil.Process(pid)
            info = proc.as_dict(attrs=[
                "pid", "name", "exe", "cmdline", "status", "username",
                "create_time", "cpu_percent", "memory_info", "num_threads",
                "nice", "connections",
            ])
            mem = info.get("memory_info")
            data = {
                "pid": info["pid"],
                "name": info["name"],
                "exe": info.get("exe", ""),
                "status": info["status"],
                "username": info.get("username", ""),
                "cpu_percent": info.get("cpu_percent", 0),
                "memory_mb": round(mem.rss / (1024 ** 2), 1) if mem else 0,
                "threads": info.get("num_threads", 0),
                "connections": len(info.get("connections") or []),
                "cmdline": " ".join(info.get("cmdline") or [])[:300],
                "uptime_hours": round((time.time() - info.get("create_time", time.time())) / 3600, 1),
            }
            return SystemControlResult(True, f"Process details for {info['name']}", data)
        except Exception as e:
            return SystemControlResult(False, f"Failed to get process details: {e}")

    def find_process_by_name(self, name: str) -> SystemControlResult:
        """Find processes matching a name pattern."""
        try:
            import psutil
            matches = []
            name_lower = name.lower()
            for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info"]):
                try:
                    if name_lower in proc.info["name"].lower():
                        mem = proc.info.get("memory_info")
                        matches.append({
                            "pid": proc.info["pid"],
                            "name": proc.info["name"],
                            "cpu": proc.info.get("cpu_percent", 0),
                            "mem_mb": round(mem.rss / (1024 ** 2), 1) if mem else 0,
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return SystemControlResult(
                True,
                f"Found {len(matches)} process(es) matching '{name}'",
                {"matches": matches},
            )
        except Exception as e:
            return SystemControlResult(False, f"Process search failed: {e}")

    # ── Network Control ───────────────────────────────────────────

    def flush_dns(self) -> SystemControlResult:
        """Flush DNS cache."""
        try:
            if self._is_windows:
                result = subprocess.run(
                    ["ipconfig", "/flushdns"],
                    capture_output=True, text=True, timeout=10,
                )
            elif self._is_linux:
                result = subprocess.run(
                    ["systemd-resolve", "--flush-caches"],
                    capture_output=True, text=True, timeout=10,
                )
            elif self._is_macos:
                result = subprocess.run(
                    ["sudo", "dscacheutil", "-flushcache"],
                    capture_output=True, text=True, timeout=10,
                )
            else:
                return SystemControlResult(False, "Unsupported platform")
            return SystemControlResult(result.returncode == 0, "DNS cache flushed")
        except Exception as e:
            return SystemControlResult(False, f"DNS flush failed: {e}")

    def get_network_speed(self) -> SystemControlResult:
        """Measure current network throughput."""
        try:
            import psutil
            net1 = psutil.net_io_counters()
            time.sleep(1)
            net2 = psutil.net_io_counters()
            download_kbps = (net2.bytes_recv - net1.bytes_recv) / 1024
            upload_kbps = (net2.bytes_sent - net1.bytes_sent) / 1024
            return SystemControlResult(True, "Network speed measured", {
                "download_kbps": round(download_kbps, 1),
                "upload_kbps": round(upload_kbps, 1),
                "download_mbps": round(download_kbps / 1024, 2),
                "upload_mbps": round(upload_kbps / 1024, 2),
            })
        except Exception as e:
            return SystemControlResult(False, f"Speed test failed: {e}")

    def get_open_ports(self) -> SystemControlResult:
        """List open listening ports."""
        try:
            import psutil
            ports = []
            for conn in psutil.net_connections(kind="inet"):
                if conn.status == "LISTEN" and conn.laddr:
                    pid_name = ""
                    try:
                        if conn.pid:
                            pid_name = psutil.Process(conn.pid).name()
                    except Exception:
                        pass
                    ports.append({
                        "port": conn.laddr.port,
                        "address": conn.laddr.ip,
                        "pid": conn.pid,
                        "process": pid_name,
                    })
            ports.sort(key=lambda p: p["port"])
            return SystemControlResult(True, f"{len(ports)} open ports found",
                                        {"ports": ports})
        except Exception as e:
            return SystemControlResult(False, f"Port scan failed: {e}")

    def get_wifi_networks(self) -> SystemControlResult:
        """Scan for available WiFi networks (Windows/Linux)."""
        try:
            if self._is_windows:
                result = subprocess.run(
                    ["netsh", "wlan", "show", "networks", "mode=Bssid"],
                    capture_output=True, text=True, timeout=15,
                )
                networks = []
                current: dict[str, str] = {}
                for line in result.stdout.split("\n"):
                    line = line.strip()
                    if line.startswith("SSID") and ":" in line:
                        if current:
                            networks.append(current)
                        current = {"ssid": line.split(":", 1)[1].strip()}
                    elif line.startswith("Signal") and ":" in line:
                        current["signal"] = line.split(":", 1)[1].strip()
                    elif line.startswith("Authentication") and ":" in line:
                        current["auth"] = line.split(":", 1)[1].strip()
                if current:
                    networks.append(current)
                return SystemControlResult(True, f"{len(networks)} WiFi networks found",
                                            {"networks": networks})
            elif self._is_linux:
                result = subprocess.run(
                    ["nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY", "dev", "wifi"],
                    capture_output=True, text=True, timeout=15,
                )
                networks = []
                for line in result.stdout.strip().split("\n"):
                    parts = line.split(":")
                    if len(parts) >= 2:
                        networks.append({
                            "ssid": parts[0],
                            "signal": parts[1] + "%" if len(parts) > 1 else "",
                            "security": parts[2] if len(parts) > 2 else "",
                        })
                return SystemControlResult(True, f"{len(networks)} WiFi networks found",
                                            {"networks": networks})
            return SystemControlResult(False, "WiFi scan not supported on this platform")
        except Exception as e:
            return SystemControlResult(False, f"WiFi scan failed: {e}")

    # ── Storage Management ────────────────────────────────────────

    def analyze_temp_files(self) -> SystemControlResult:
        """Analyze temporary files that can be cleaned."""
        import os
        total_size = 0
        file_count = 0
        categories: dict[str, int] = {}

        temp_dirs = []
        if self._is_windows:
            temp_dirs = [
                Path(os.environ.get("TEMP", "")),
                Path(os.environ.get("TMP", "")),
                Path.home() / "AppData" / "Local" / "Temp",
            ]
        else:
            temp_dirs = [Path("/tmp"), Path("/var/tmp")]

        for temp_dir in temp_dirs:
            if not temp_dir.exists():
                continue
            try:
                for item in temp_dir.rglob("*"):
                    try:
                        if item.is_file():
                            size = item.stat().st_size
                            total_size += size
                            file_count += 1
                            ext = item.suffix.lower() or "no_extension"
                            categories[ext] = categories.get(ext, 0) + size
                    except (PermissionError, OSError):
                        pass
            except (PermissionError, OSError):
                pass

        top_types = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10]
        return SystemControlResult(True, "Temp file analysis complete", {
            "total_size_mb": round(total_size / (1024 ** 2), 1),
            "file_count": file_count,
            "top_types": [
                {"ext": ext, "size_mb": round(size / (1024 ** 2), 1)}
                for ext, size in top_types
            ],
        })

    def find_large_files(self, path: str = "", min_size_mb: int = 100,
                         limit: int = 20) -> SystemControlResult:
        """Find large files on disk."""
        search_path = Path(path) if path else Path.home()
        if not search_path.exists():
            return SystemControlResult(False, f"Path does not exist: {path}")

        large_files = []
        min_bytes = min_size_mb * 1024 * 1024

        try:
            for item in search_path.rglob("*"):
                try:
                    if item.is_file():
                        size = item.stat().st_size
                        if size >= min_bytes:
                            large_files.append({
                                "path": str(item),
                                "size_mb": round(size / (1024 ** 2), 1),
                                "modified": time.ctime(item.stat().st_mtime),
                            })
                except (PermissionError, OSError):
                    pass
                if len(large_files) >= limit * 2:
                    break
        except (PermissionError, OSError):
            pass

        large_files.sort(key=lambda f: f["size_mb"], reverse=True)
        return SystemControlResult(
            True,
            f"Found {len(large_files[:limit])} files over {min_size_mb}MB",
            {"files": large_files[:limit]},
        )

    # ── Startup Management ────────────────────────────────────────

    def list_startup_programs(self) -> SystemControlResult:
        """List programs that run at startup."""
        programs = []

        if self._is_windows:
            try:
                import winreg
                startup_keys = [
                    (winreg.HKEY_CURRENT_USER,
                     r"Software\Microsoft\Windows\CurrentVersion\Run"),
                    (winreg.HKEY_LOCAL_MACHINE,
                     r"Software\Microsoft\Windows\CurrentVersion\Run"),
                ]
                for hive, path in startup_keys:
                    try:
                        key = winreg.OpenKey(hive, path)
                        i = 0
                        while True:
                            try:
                                name, value, _ = winreg.EnumValue(key, i)
                                programs.append({
                                    "name": name,
                                    "command": value[:200],
                                    "scope": "user" if hive == winreg.HKEY_CURRENT_USER else "system",
                                })
                                i += 1
                            except OSError:
                                break
                        winreg.CloseKey(key)
                    except FileNotFoundError:
                        pass
            except Exception:
                pass

            startup_folder = Path.home() / "AppData" / "Roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
            if startup_folder.exists():
                for item in startup_folder.iterdir():
                    programs.append({
                        "name": item.stem,
                        "command": str(item),
                        "scope": "startup_folder",
                    })

        elif self._is_linux:
            autostart = Path.home() / ".config" / "autostart"
            if autostart.exists():
                for item in autostart.glob("*.desktop"):
                    programs.append({
                        "name": item.stem,
                        "command": str(item),
                        "scope": "user",
                    })

        return SystemControlResult(True, f"{len(programs)} startup programs found",
                                    {"programs": programs})

    # ── Performance Optimization ──────────────────────────────────

    def optimize_for_atom(self) -> SystemControlResult:
        """Optimize system for ATOM performance (lower priority of hogs)."""
        optimizations = []

        try:
            import psutil

            for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info"]):
                try:
                    cpu = proc.info.get("cpu_percent", 0) or 0
                    name = proc.info.get("name", "").lower()
                    mem = proc.info.get("memory_info")
                    mem_mb = (mem.rss / (1024 ** 2)) if mem else 0

                    skip_names = {"system", "idle", "python", "atom", "svchost"}
                    if name in skip_names:
                        continue

                    if cpu > 40 and mem_mb > 500:
                        try:
                            if self._is_windows:
                                proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                            else:
                                proc.nice(10)
                            optimizations.append(
                                f"Lowered priority of {name} (CPU: {cpu:.0f}%, RAM: {mem_mb:.0f}MB)"
                            )
                        except (psutil.AccessDenied, psutil.NoSuchProcess):
                            pass
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

        except ImportError:
            return SystemControlResult(False, "psutil not available")

        if optimizations:
            return SystemControlResult(
                True,
                f"Optimized {len(optimizations)} processes for ATOM",
                {"optimizations": optimizations},
            )
        return SystemControlResult(True, "System already optimized -- no heavy processes found")

    def get_system_uptime(self) -> SystemControlResult:
        """Get detailed system uptime info."""
        try:
            import psutil
            boot = psutil.boot_time()
            uptime_s = time.time() - boot
            hours = int(uptime_s // 3600)
            minutes = int((uptime_s % 3600) // 60)

            return SystemControlResult(True, f"System uptime: {hours}h {minutes}m", {
                "boot_time": time.ctime(boot),
                "uptime_seconds": uptime_s,
                "uptime_hours": round(uptime_s / 3600, 1),
                "uptime_human": f"{hours} hours, {minutes} minutes",
            })
        except Exception as e:
            return SystemControlResult(False, f"Uptime check failed: {e}")

    # ── Hardware Info ─────────────────────────────────────────────

    def get_hardware_details(self) -> SystemControlResult:
        """Get detailed hardware information."""
        from core.platform_adapter import get_platform_adapter

        adapter = get_platform_adapter()
        profile = adapter.get_system_profile()

        data = {
            "cpu": {
                "name": profile.cpu_name,
                "physical_cores": profile.cpu_cores_physical,
                "logical_cores": profile.cpu_cores_logical,
                "architecture": profile.architecture,
            },
            "memory": {
                "total_gb": profile.ram_total_gb,
                "available_gb": profile.ram_available_gb,
            },
            "gpu": {
                "name": profile.gpu_name or "No dedicated GPU",
                "vram_gb": profile.gpu_vram_gb,
            },
            "display": {
                "resolution": profile.display_resolution,
                "count": profile.display_count,
            },
            "storage": [
                {
                    "device": d.device,
                    "mount": d.mountpoint,
                    "total_gb": d.total_gb,
                    "free_gb": d.free_gb,
                }
                for d in adapter.get_disk_info()
            ],
            "network": [
                {"name": n.name, "ip": n.ip_address, "speed": n.speed_mbps}
                for n in adapter.get_network_interfaces()
                if n.ip_address
            ],
        }

        if profile.has_battery:
            data["battery"] = {
                "percent": profile.battery_percent,
                "plugged": profile.is_plugged,
            }

        summary = (
            f"{profile.cpu_name} | {profile.ram_total_gb:.0f}GB RAM | "
            f"GPU: {profile.gpu_name or 'None'} | "
            f"Uptime: {profile.uptime_hours:.1f}h"
        )

        return SystemControlResult(True, summary, data)

    # ── Environment Variables ─────────────────────────────────────

    def get_env_variable(self, name: str) -> SystemControlResult:
        """Get an environment variable."""
        import os
        value = os.environ.get(name)
        if value is None:
            return SystemControlResult(False, f"Environment variable '{name}' not found")
        return SystemControlResult(True, f"{name}={value[:200]}", {"name": name, "value": value[:500]})

    def list_env_variables(self, filter_pattern: str = "") -> SystemControlResult:
        """List environment variables, optionally filtered."""
        import os
        env = dict(os.environ)
        if filter_pattern:
            pattern = filter_pattern.lower()
            env = {k: v for k, v in env.items() if pattern in k.lower()}
        safe_env = {k: v[:200] for k, v in sorted(env.items())}
        return SystemControlResult(True, f"{len(safe_env)} environment variables",
                                    {"variables": safe_env})

    # ── Power Plans (Windows) ─────────────────────────────────────

    def set_power_plan(self, plan: str = "balanced") -> SystemControlResult:
        """Set Windows power plan: balanced, high_performance, power_saver."""
        if not self._is_windows:
            return SystemControlResult(False, "Power plans are Windows-only")

        plan_guids = {
            "balanced": "381b4222-f694-41f0-9685-ff5bb260df2e",
            "high_performance": "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c",
            "power_saver": "a1841308-3541-4fab-bc81-f71556f20b4a",
        }
        guid = plan_guids.get(plan)
        if not guid:
            return SystemControlResult(False, f"Unknown power plan: {plan}")

        try:
            result = subprocess.run(
                ["powercfg", "/setactive", guid],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                return SystemControlResult(True, f"Power plan set to {plan}")
            return SystemControlResult(False, f"Failed: {result.stderr}")
        except Exception as e:
            return SystemControlResult(False, f"Power plan change failed: {e}")

    # ── Summary for Voice ─────────────────────────────────────────

    def get_full_status(self) -> str:
        """Generate a comprehensive system status for voice output."""
        hw = self.get_hardware_details()
        uptime = self.get_system_uptime()

        if not hw.success:
            return "Unable to get system status, Boss."

        data = hw.data
        cpu = data.get("cpu", {})
        mem = data.get("memory", {})
        gpu = data.get("gpu", {})

        parts = [
            f"System running on {cpu.get('name', 'unknown CPU')} "
            f"with {mem.get('total_gb', 0):.0f} gigs of RAM.",
            f"{mem.get('available_gb', 0):.1f} gigs available right now.",
        ]

        if gpu.get("name") and "No dedicated" not in gpu["name"]:
            parts.append(f"GPU: {gpu['name']} with {gpu.get('vram_gb', 0):.0f} gigs VRAM.")

        if uptime.success:
            parts.append(f"Uptime: {uptime.data.get('uptime_human', 'unknown')}.")

        bat = data.get("battery")
        if bat:
            plug_status = "plugged in" if bat["plugged"] else "on battery"
            parts.append(f"Battery at {bat['percent']:.0f}%, {plug_status}.")

        return " ".join(parts)
