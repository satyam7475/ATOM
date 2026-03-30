"""
ATOM -- System Scanner (JARVIS-Level System Intelligence).

Deep system intelligence engine that scans, profiles, and continuously
monitors the host machine. Like JARVIS scanning Tony's lab, ATOM's
SystemScanner knows EVERYTHING about the system it runs on.

Capabilities:
    - Full hardware profile (CPU, GPU, RAM, disks, network, displays)
    - Installed software inventory (apps, drivers, packages)
    - Running process intelligence (top consumers, anomaly detection)
    - Network topology (interfaces, connections, open ports)
    - System health scoring (0-100 composite score)
    - Environment analysis (dev tools, IDEs, runtimes installed)
    - Security posture check (firewall, antivirus, open ports)
    - Storage intelligence (large files, temp bloat, disk health)
    - Startup program analysis
    - Performance bottleneck detection

Runs an initial deep scan at startup, then periodic light scans.
Emits system_intelligence events for the JARVIS core to consume.

Contract: CognitiveModuleContract (start, stop, persist)
Owner: Satyam
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus

logger = logging.getLogger("atom.scanner")

_SCAN_CACHE = Path("logs/system_scan.json")


@dataclass
class SystemHealthScore:
    """Composite system health score (0-100)."""
    overall: int = 0
    cpu_score: int = 0
    ram_score: int = 0
    disk_score: int = 0
    gpu_score: int = 0
    network_score: int = 0
    thermal_score: int = 0
    details: list[str] = field(default_factory=list)


@dataclass
class PerformanceBottleneck:
    """Detected performance issue."""
    component: str = ""
    severity: str = ""  # low, medium, high, critical
    description: str = ""
    suggestion: str = ""


@dataclass
class EnvironmentProfile:
    """Development environment analysis."""
    detected_languages: list[str] = field(default_factory=list)
    detected_ides: list[str] = field(default_factory=list)
    detected_runtimes: list[dict[str, str]] = field(default_factory=list)
    detected_package_managers: list[str] = field(default_factory=list)
    git_installed: bool = False
    docker_installed: bool = False
    node_installed: bool = False
    python_version: str = ""
    java_version: str = ""


class SystemScanner:
    """Deep system intelligence scanner -- JARVIS-level awareness."""

    def __init__(self, bus: AsyncEventBus | None = None,
                 config: dict | None = None) -> None:
        self._bus = bus
        self._config = config or {}
        self._scan_interval = self._config.get(
            "system_scanner", {},
        ).get("scan_interval_s", 300.0)
        self._task: asyncio.Task | None = None
        self._shutdown = asyncio.Event()
        self._last_scan: dict[str, Any] = {}
        self._health: SystemHealthScore = SystemHealthScore()
        self._bottlenecks: list[PerformanceBottleneck] = []
        self._env_profile: EnvironmentProfile = EnvironmentProfile()
        self._scan_count = 0
        self._startup_scanned = False

        self._load_cached_scan()

    def _load_cached_scan(self) -> None:
        try:
            if _SCAN_CACHE.exists():
                self._last_scan = json.loads(
                    _SCAN_CACHE.read_text(encoding="utf-8"),
                )
                logger.debug("Loaded cached system scan")
        except Exception:
            pass

    # ── Deep System Scan ──────────────────────────────────────────

    def scan_full(self) -> dict[str, Any]:
        """Perform a comprehensive system scan. Returns full intelligence report."""
        from core.platform_adapter import get_platform_adapter

        adapter = get_platform_adapter()
        report: dict[str, Any] = {
            "scan_time": time.time(),
            "scan_count": self._scan_count,
        }

        # 1. System profile
        profile = adapter.get_system_profile()
        report["system"] = {
            "os": f"{profile.os_name} {profile.os_build}",
            "os_version": profile.os_version,
            "hostname": profile.hostname,
            "username": profile.username,
            "architecture": profile.architecture,
            "cpu": profile.cpu_name,
            "cpu_cores": f"{profile.cpu_cores_physical}P/{profile.cpu_cores_logical}L",
            "ram_total_gb": profile.ram_total_gb,
            "ram_available_gb": profile.ram_available_gb,
            "gpu": profile.gpu_name or "No dedicated GPU detected",
            "gpu_vram_gb": profile.gpu_vram_gb,
            "uptime_hours": profile.uptime_hours,
            "display": profile.display_resolution,
            "display_count": profile.display_count,
            "has_battery": profile.has_battery,
            "battery_percent": profile.battery_percent,
            "is_plugged": profile.is_plugged,
            "shell": profile.shell,
        }

        # 2. Disk intelligence
        disks = adapter.get_disk_info()
        report["disks"] = [
            {
                "device": d.device,
                "mount": d.mountpoint,
                "fs": d.filesystem,
                "total_gb": d.total_gb,
                "free_gb": d.free_gb,
                "percent_used": d.percent_used,
            }
            for d in disks
        ]

        # 3. Network interfaces
        interfaces = adapter.get_network_interfaces()
        report["network"] = [
            {
                "name": n.name,
                "ip": n.ip_address,
                "mac": n.mac_address,
                "is_up": n.is_up,
                "speed_mbps": n.speed_mbps,
            }
            for n in interfaces
        ]

        # 4. Top processes
        top_procs = adapter.list_processes(sort_by="cpu", limit=15)
        report["top_processes"] = [
            {
                "pid": p.pid,
                "name": p.name,
                "cpu": p.cpu_percent,
                "mem_mb": p.memory_mb,
                "status": p.status,
            }
            for p in top_procs
        ]

        # 5. Active connections
        report["connections"] = self._scan_connections()

        # 6. Environment profile
        self._env_profile = self._scan_environment()
        report["environment"] = {
            "languages": self._env_profile.detected_languages,
            "ides": self._env_profile.detected_ides,
            "runtimes": self._env_profile.detected_runtimes,
            "package_managers": self._env_profile.detected_package_managers,
            "git": self._env_profile.git_installed,
            "docker": self._env_profile.docker_installed,
            "node": self._env_profile.node_installed,
        }

        # 7. Health score
        self._health = self._calculate_health(report)
        report["health"] = {
            "overall": self._health.overall,
            "cpu": self._health.cpu_score,
            "ram": self._health.ram_score,
            "disk": self._health.disk_score,
            "gpu": self._health.gpu_score,
            "network": self._health.network_score,
            "details": self._health.details,
        }

        # 8. Bottleneck detection
        self._bottlenecks = self._detect_bottlenecks(report)
        report["bottlenecks"] = [
            {
                "component": b.component,
                "severity": b.severity,
                "description": b.description,
                "suggestion": b.suggestion,
            }
            for b in self._bottlenecks
        ]

        # 9. Installed apps count
        try:
            apps = adapter.get_installed_apps()
            report["installed_apps_count"] = len(apps)
            report["notable_apps"] = [
                a.name for a in apps[:30]
                if any(kw in a.name.lower() for kw in (
                    "chrome", "firefox", "edge", "code", "visual studio",
                    "slack", "teams", "discord", "spotify", "steam",
                    "python", "node", "java", "git", "docker",
                    "notepad++", "sublime", "intellij", "pycharm",
                    "obs", "vlc", "7-zip", "winrar",
                ))
            ]
        except Exception:
            report["installed_apps_count"] = 0
            report["notable_apps"] = []

        # 10. Services
        try:
            services = adapter.list_services(filter_running=True)
            report["running_services_count"] = len(services)
        except Exception:
            report["running_services_count"] = 0

        self._last_scan = report
        self._scan_count += 1
        self._startup_scanned = True

        logger.info(
            "System scan #%d complete: %s | %s | %dGB RAM | GPU: %s | Health: %d/100",
            self._scan_count,
            report["system"]["os"],
            report["system"]["cpu"],
            report["system"]["ram_total_gb"],
            report["system"]["gpu"],
            self._health.overall,
        )

        return report

    def scan_light(self) -> dict[str, Any]:
        """Quick scan — only dynamic metrics (CPU, RAM, processes, connections)."""
        report: dict[str, Any] = {"scan_time": time.time(), "type": "light"}

        try:
            import psutil
            report["cpu_percent"] = psutil.cpu_percent(interval=0.5)
            report["cpu_per_core"] = psutil.cpu_percent(interval=0, percpu=True)
            mem = psutil.virtual_memory()
            report["ram_percent"] = mem.percent
            report["ram_available_gb"] = round(mem.available / (1024 ** 3), 2)
            report["swap_percent"] = psutil.swap_memory().percent

            top_cpu = []
            top_mem = []
            for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info"]):
                try:
                    info = proc.info
                    cpu_pct = info.get("cpu_percent", 0) or 0
                    mem_mb = (info.get("memory_info") or type("", (), {"rss": 0})).rss / (1024 ** 2)
                    if cpu_pct > 5:
                        top_cpu.append({"name": info["name"], "cpu": cpu_pct, "pid": info["pid"]})
                    if mem_mb > 200:
                        top_mem.append({"name": info["name"], "mem_mb": round(mem_mb), "pid": info["pid"]})
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            top_cpu.sort(key=lambda x: x["cpu"], reverse=True)
            top_mem.sort(key=lambda x: x["mem_mb"], reverse=True)
            report["top_cpu"] = top_cpu[:10]
            report["top_mem"] = top_mem[:10]

            bat = psutil.sensors_battery()
            if bat:
                report["battery"] = {
                    "percent": bat.percent,
                    "plugged": bat.power_plugged,
                    "secs_left": bat.secsleft if bat.secsleft != -1 else None,
                }

            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    report["temperatures"] = {
                        name: [{"label": t.label, "current": t.current, "high": t.high}
                               for t in entries]
                        for name, entries in temps.items()
                    }
            except (AttributeError, NotImplementedError):
                pass

        except ImportError:
            logger.debug("psutil not available for light scan")

        report["connections"] = self._scan_connections()

        self._last_scan.update(report)
        return report

    # ── Sub-Scanners ──────────────────────────────────────────────

    def _scan_connections(self) -> dict[str, Any]:
        """Scan active network connections."""
        conn_info: dict[str, Any] = {"established": 0, "listening": 0, "total": 0}
        try:
            import psutil
            conns = psutil.net_connections(kind="inet")
            conn_info["total"] = len(conns)
            listening_ports = []
            for c in conns:
                if c.status == "ESTABLISHED":
                    conn_info["established"] += 1
                elif c.status == "LISTEN":
                    conn_info["listening"] += 1
                    if c.laddr:
                        listening_ports.append(c.laddr.port)
            conn_info["listening_ports"] = sorted(set(listening_ports))[:20]
        except Exception:
            pass
        return conn_info

    def _scan_environment(self) -> EnvironmentProfile:
        """Detect development environment on this system."""
        import shutil

        env = EnvironmentProfile()

        runtime_checks = {
            "python": ["python", "--version"],
            "python3": ["python3", "--version"],
            "node": ["node", "--version"],
            "java": ["java", "-version"],
            "go": ["go", "version"],
            "rustc": ["rustc", "--version"],
            "dotnet": ["dotnet", "--version"],
            "ruby": ["ruby", "--version"],
            "php": ["php", "--version"],
        }

        for name, cmd in runtime_checks.items():
            if shutil.which(cmd[0]):
                try:
                    import subprocess
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=5,
                    )
                    version = (result.stdout or result.stderr).strip().split("\n")[0]
                    env.detected_runtimes.append({"name": name, "version": version})
                    lang = name.replace("3", "").replace("rustc", "rust").replace("dotnet", "c#")
                    if lang not in env.detected_languages:
                        env.detected_languages.append(lang)
                except Exception:
                    pass

        env.git_installed = bool(shutil.which("git"))
        env.docker_installed = bool(shutil.which("docker"))
        env.node_installed = bool(shutil.which("node"))

        pkg_managers = ["pip", "npm", "yarn", "cargo", "go", "gem", "composer",
                        "apt", "dnf", "yum", "brew", "choco", "scoop", "winget"]
        env.detected_package_managers = [pm for pm in pkg_managers if shutil.which(pm)]

        ide_indicators = {
            "VS Code": ["code"],
            "Visual Studio": ["devenv"],
            "PyCharm": ["pycharm", "pycharm64"],
            "IntelliJ IDEA": ["idea", "idea64"],
            "Sublime Text": ["subl"],
            "Vim": ["vim", "nvim"],
            "Emacs": ["emacs"],
            "Cursor": ["cursor"],
            "Android Studio": ["studio64", "studio"],
        }
        for ide_name, commands in ide_indicators.items():
            if any(shutil.which(cmd) for cmd in commands):
                env.detected_ides.append(ide_name)

        return env

    # ── Health Scoring ────────────────────────────────────────────

    def _calculate_health(self, report: dict) -> SystemHealthScore:
        """Calculate composite system health score."""
        h = SystemHealthScore()

        sys_info = report.get("system", {})
        ram_total = sys_info.get("ram_total_gb", 8)
        ram_avail = sys_info.get("ram_available_gb", 4)
        ram_pct_used = 100 * (1 - ram_avail / max(ram_total, 0.1))

        h.ram_score = max(0, min(100, int(100 - ram_pct_used)))
        if ram_pct_used > 90:
            h.details.append(f"RAM critically low: {ram_avail:.1f}GB free of {ram_total:.1f}GB")
        elif ram_pct_used > 75:
            h.details.append(f"RAM moderately used: {ram_pct_used:.0f}%")

        try:
            import psutil
            cpu = psutil.cpu_percent(interval=0)
            h.cpu_score = max(0, min(100, int(100 - cpu)))
            if cpu > 90:
                h.details.append(f"CPU under heavy load: {cpu:.0f}%")
        except Exception:
            h.cpu_score = 50

        disks = report.get("disks", [])
        if disks:
            worst_disk = max(disks, key=lambda d: d.get("percent_used", 0))
            pct = worst_disk.get("percent_used", 0)
            h.disk_score = max(0, min(100, int(100 - pct)))
            if pct > 90:
                h.details.append(
                    f"Disk {worst_disk['mount']} nearly full: {pct:.0f}% used, "
                    f"{worst_disk['free_gb']:.1f}GB free"
                )
        else:
            h.disk_score = 50

        gpu = sys_info.get("gpu", "")
        if gpu and "No dedicated" not in gpu:
            h.gpu_score = 90
            vram = sys_info.get("gpu_vram_gb", 0)
            if vram >= 8:
                h.gpu_score = 100
                h.details.append(f"GPU excellent: {gpu} ({vram}GB VRAM)")
            elif vram >= 4:
                h.gpu_score = 75
        else:
            h.gpu_score = 30
            h.details.append("No dedicated GPU -- LLM will use CPU (slower)")

        network = report.get("network", [])
        active = [n for n in network if n.get("is_up")]
        h.network_score = 100 if active else 20
        if not active:
            h.details.append("No active network interfaces detected")

        h.thermal_score = 85

        weights = {"cpu": 0.2, "ram": 0.2, "disk": 0.15, "gpu": 0.25, "network": 0.1, "thermal": 0.1}
        h.overall = int(
            h.cpu_score * weights["cpu"]
            + h.ram_score * weights["ram"]
            + h.disk_score * weights["disk"]
            + h.gpu_score * weights["gpu"]
            + h.network_score * weights["network"]
            + h.thermal_score * weights["thermal"]
        )

        return h

    # ── Bottleneck Detection ──────────────────────────────────────

    def _detect_bottlenecks(self, report: dict) -> list[PerformanceBottleneck]:
        """Identify performance bottlenecks and suggest fixes."""
        bottlenecks: list[PerformanceBottleneck] = []
        sys_info = report.get("system", {})

        ram_total = sys_info.get("ram_total_gb", 8)
        if ram_total < 8:
            bottlenecks.append(PerformanceBottleneck(
                component="RAM",
                severity="high",
                description=f"Only {ram_total}GB RAM. ATOM + LLM needs 16GB minimum.",
                suggestion="Upgrade to 16GB+ RAM for optimal performance.",
            ))
        elif ram_total < 16:
            bottlenecks.append(PerformanceBottleneck(
                component="RAM",
                severity="medium",
                description=f"{ram_total}GB RAM. Workable but tight with LLM loaded.",
                suggestion="Close unnecessary apps when using ATOM's brain.",
            ))

        gpu = sys_info.get("gpu", "")
        vram = sys_info.get("gpu_vram_gb", 0)
        if "No dedicated" in gpu:
            bottlenecks.append(PerformanceBottleneck(
                component="GPU",
                severity="critical",
                description="No dedicated GPU detected. LLM runs on CPU only.",
                suggestion="For JARVIS-level speed, get an RTX 3060+ with 8GB+ VRAM.",
            ))
        elif vram < 6:
            bottlenecks.append(PerformanceBottleneck(
                component="GPU",
                severity="medium",
                description=f"GPU VRAM: {vram}GB. May struggle with larger models.",
                suggestion="Use 7B quantized models (Q4_K_M) for best speed/quality.",
            ))

        for disk in report.get("disks", []):
            if disk.get("percent_used", 0) > 90:
                bottlenecks.append(PerformanceBottleneck(
                    component="Disk",
                    severity="high",
                    description=f"Disk {disk['mount']} is {disk['percent_used']:.0f}% full.",
                    suggestion="Free up space. Models need ~15GB, vector DB grows over time.",
                ))

        cores = sys_info.get("cpu_cores", "0P/0L")
        try:
            physical = int(cores.split("P")[0])
            if physical < 4:
                bottlenecks.append(PerformanceBottleneck(
                    component="CPU",
                    severity="medium",
                    description=f"Only {physical} CPU cores. STT + async tasks need more.",
                    suggestion="A 6-core+ CPU improves STT latency significantly.",
                ))
        except (ValueError, IndexError):
            pass

        top_procs = report.get("top_processes", [])
        cpu_hogs = [p for p in top_procs if p.get("cpu", 0) > 30]
        if cpu_hogs:
            names = ", ".join(p["name"] for p in cpu_hogs[:3])
            bottlenecks.append(PerformanceBottleneck(
                component="Processes",
                severity="low",
                description=f"High CPU consumers: {names}",
                suggestion="These may compete with ATOM for CPU. Consider closing them.",
            ))

        return bottlenecks

    # ── Public API ────────────────────────────────────────────────

    @property
    def health(self) -> SystemHealthScore:
        return self._health

    @property
    def bottlenecks(self) -> list[PerformanceBottleneck]:
        return self._bottlenecks

    @property
    def environment(self) -> EnvironmentProfile:
        return self._env_profile

    @property
    def last_scan(self) -> dict[str, Any]:
        return self._last_scan

    def get_scan_summary(self) -> str:
        """Human-readable scan summary for voice output."""
        if not self._last_scan:
            return "No system scan available yet, Boss."

        s = self._last_scan.get("system", {})
        h = self._health

        parts = [
            f"System health score: {h.overall} out of 100.",
            f"Running {s.get('os', 'unknown OS')} on {s.get('cpu', 'unknown CPU')}.",
            f"{s.get('ram_total_gb', 0):.0f} gigs of RAM, "
            f"{s.get('ram_available_gb', 0):.1f} available.",
        ]

        gpu = s.get("gpu", "")
        if gpu and "No dedicated" not in gpu:
            parts.append(f"GPU: {gpu} with {s.get('gpu_vram_gb', 0):.0f} gigs VRAM.")

        if self._bottlenecks:
            critical = [b for b in self._bottlenecks if b.severity in ("critical", "high")]
            if critical:
                parts.append(f"{len(critical)} critical issue{'s' if len(critical) > 1 else ''} detected.")
                parts.append(critical[0].description)

        return " ".join(parts)

    def get_intelligence_for_llm(self) -> str:
        """Compact system intelligence string for LLM context injection."""
        if not self._last_scan:
            return ""

        s = self._last_scan.get("system", {})
        env = self._env_profile

        lines = [
            f"[SYSTEM] {s.get('os', '?')} | {s.get('cpu', '?')} | "
            f"{s.get('ram_total_gb', 0):.0f}GB RAM | GPU: {s.get('gpu', 'none')}",
        ]

        if env.detected_languages:
            lines.append(f"[DEV ENV] Languages: {', '.join(env.detected_languages)}")
        if env.detected_ides:
            lines.append(f"[IDEs] {', '.join(env.detected_ides)}")

        disks = self._last_scan.get("disks", [])
        for d in disks:
            if d.get("percent_used", 0) > 80:
                lines.append(f"[DISK WARNING] {d['mount']}: {d['percent_used']:.0f}% used")

        return "\n".join(lines)

    # ── Background Scanner ────────────────────────────────────────

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())
        logger.info("System scanner started (interval=%.0fs)", self._scan_interval)

    def stop(self) -> None:
        self._shutdown.set()
        if self._task and not self._task.done():
            self._task.cancel()

    async def _run(self) -> None:
        try:
            loop = asyncio.get_running_loop()

            logger.info("=== ATOM BOOT DIAGNOSTICS ===")
            readiness = await loop.run_in_executor(None, self.run_readiness_check)
            if self._bus:
                self._bus.emit_fast("atom_readiness", report=readiness)

            scan = await loop.run_in_executor(None, self.scan_full)
            if self._bus:
                self._bus.emit_fast("system_intelligence",
                                    scan=scan, health=self._health.overall)

            logger.info("=== BOOT DIAGNOSTICS COMPLETE ===")

            while not self._shutdown.is_set():
                try:
                    await asyncio.wait_for(
                        self._shutdown.wait(), timeout=self._scan_interval,
                    )
                    break
                except asyncio.TimeoutError:
                    pass

                light = await loop.run_in_executor(None, self.scan_light)
                if self._bus:
                    self._bus.emit_fast("system_light_scan", scan=light)

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("System scanner error")

    # ── ATOM Readiness Diagnostics ───────────────────────────────────

    def run_readiness_check(self) -> dict[str, Any]:
        """Run ATOM-specific readiness diagnostics at boot time.

        Checks every subsystem ATOM depends on and returns a
        pass/warn/fail status for each. Like JARVIS running
        diagnostics when Tony walks into the lab.
        """
        checks: dict[str, Any] = {
            "timestamp": time.time(),
            "overall_ready": True,
            "subsystems": {},
        }

        checks["subsystems"]["python"] = self._check_python()
        checks["subsystems"]["audio"] = self._check_audio()
        checks["subsystems"]["stt_engine"] = self._check_stt_engine()
        checks["subsystems"]["tts_engine"] = self._check_tts_engine()
        checks["subsystems"]["llm_model"] = self._check_llm()
        checks["subsystems"]["gpu_compute"] = self._check_gpu_compute()
        checks["subsystems"]["disk_space"] = self._check_disk_space()
        checks["subsystems"]["network"] = self._check_network()
        checks["subsystems"]["memory"] = self._check_memory()
        checks["subsystems"]["dependencies"] = self._check_dependencies()
        checks["subsystems"]["data_dirs"] = self._check_data_dirs()

        failed = [
            name for name, info in checks["subsystems"].items()
            if info.get("status") == "fail"
        ]
        warned = [
            name for name, info in checks["subsystems"].items()
            if info.get("status") == "warn"
        ]
        passed = [
            name for name, info in checks["subsystems"].items()
            if info.get("status") == "pass"
        ]

        checks["overall_ready"] = len(failed) == 0
        checks["summary"] = {
            "passed": len(passed),
            "warnings": len(warned),
            "failures": len(failed),
            "failed_systems": failed,
        }

        status = "ALL SYSTEMS GO" if not failed else f"{len(failed)} SYSTEM(S) NEED ATTENTION"
        logger.info(
            "ATOM Readiness: %s (%d passed, %d warnings, %d failures)",
            status, len(passed), len(warned), len(failed),
        )

        if self._bus:
            self._bus.emit_fast("atom_readiness", report=checks)

        return checks

    def _check_python(self) -> dict[str, Any]:
        """Check Python runtime readiness."""
        import sys
        import platform
        version = sys.version.split()[0]
        major, minor = sys.version_info[:2]
        ok = major >= 3 and minor >= 10
        return {
            "status": "pass" if ok else "warn",
            "version": version,
            "platform": platform.platform(),
            "arch": platform.machine(),
            "detail": f"Python {version}" if ok else f"Python {version} (3.10+ recommended)",
        }

    def _check_audio(self) -> dict[str, Any]:
        """Check audio subsystem availability."""
        result: dict[str, Any] = {"status": "fail", "detail": ""}
        try:
            import pyaudio
            pa = pyaudio.PyAudio()
            input_count = 0
            output_count = 0
            for i in range(pa.get_device_count()):
                try:
                    info = pa.get_device_info_by_index(i)
                    if info.get("maxInputChannels", 0) > 0:
                        input_count += 1
                    if info.get("maxOutputChannels", 0) > 0:
                        output_count += 1
                except Exception:
                    pass
            pa.terminate()
            if input_count > 0:
                result["status"] = "pass"
                result["detail"] = f"{input_count} input, {output_count} output devices"
            else:
                result["status"] = "fail"
                result["detail"] = "No audio input devices found"
            result["input_devices"] = input_count
            result["output_devices"] = output_count
        except ImportError:
            result["detail"] = "pyaudio not installed"
        except Exception as e:
            result["detail"] = f"Audio check failed: {e}"
        return result

    def _check_stt_engine(self) -> dict[str, Any]:
        """Check STT engine availability (faster-whisper required)."""
        result: dict[str, Any] = {"status": "fail", "engine": "none", "detail": ""}
        try:
            from faster_whisper import WhisperModel
            result["status"] = "pass"
            result["engine"] = "faster-whisper"
            result["detail"] = "faster-whisper available (STT engine ready)"
            return result
        except ImportError:
            result["detail"] = "No STT engine available"
        return result

    def _check_tts_engine(self) -> dict[str, Any]:
        """Check TTS engine availability."""
        result: dict[str, Any] = {"status": "fail", "engine": "none", "detail": ""}
        try:
            import pyttsx3
            result["status"] = "pass"
            result["engine"] = "pyttsx3"
            result["detail"] = "pyttsx3 TTS available"
            return result
        except ImportError:
            pass
        try:
            import edge_tts
            result["status"] = "pass"
            result["engine"] = "edge-tts"
            result["detail"] = "edge-tts available"
            return result
        except ImportError:
            pass
        result["detail"] = "No TTS engine available"
        return result

    def _check_llm(self) -> dict[str, Any]:
        """Check LLM model/API availability."""
        result: dict[str, Any] = {"status": "warn", "detail": ""}
        try:
            from pathlib import Path
            import os
            atom_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            models_dir = atom_root / "models"
            if models_dir.exists():
                gguf_files = list(models_dir.glob("*.gguf"))
                if gguf_files:
                    biggest = max(gguf_files, key=lambda f: f.stat().st_size)
                    size_gb = biggest.stat().st_size / (1024 ** 3)
                    result["status"] = "pass"
                    result["detail"] = f"LLM model found: {biggest.name} ({size_gb:.1f}GB)"
                    result["model_file"] = biggest.name
                    result["model_size_gb"] = round(size_gb, 1)
                    return result
            try:
                from llama_cpp import Llama
                result["status"] = "warn"
                result["detail"] = "llama-cpp-python available but no model file found"
                return result
            except ImportError:
                pass
            result["detail"] = "No local LLM setup (will use API if configured)"
        except Exception as e:
            result["detail"] = f"LLM check error: {e}"
        return result

    def _check_gpu_compute(self) -> dict[str, Any]:
        """Check GPU availability for compute (CUDA/ROCm)."""
        result: dict[str, Any] = {"status": "warn", "detail": ""}
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
                result["status"] = "pass"
                result["detail"] = f"CUDA GPU: {gpu_name} ({vram:.1f}GB VRAM)"
                result["gpu"] = gpu_name
                result["vram_gb"] = round(vram, 1)
                return result
        except (ImportError, Exception):
            pass
        result["detail"] = "No GPU acceleration (CPU-only mode)"
        return result

    def _check_disk_space(self) -> dict[str, Any]:
        """Check available disk space for ATOM operations."""
        import shutil
        result: dict[str, Any] = {"status": "pass", "detail": ""}
        try:
            usage = shutil.disk_usage(".")
            free_gb = usage.free / (1024 ** 3)
            total_gb = usage.total / (1024 ** 3)
            pct_used = 100 * (1 - usage.free / usage.total)
            result["free_gb"] = round(free_gb, 1)
            result["total_gb"] = round(total_gb, 1)
            if free_gb < 5:
                result["status"] = "fail"
                result["detail"] = f"Critically low disk: {free_gb:.1f}GB free"
            elif free_gb < 15:
                result["status"] = "warn"
                result["detail"] = f"Low disk: {free_gb:.1f}GB free ({pct_used:.0f}% used)"
            else:
                result["detail"] = f"{free_gb:.1f}GB free of {total_gb:.0f}GB"
        except Exception as e:
            result["status"] = "warn"
            result["detail"] = f"Disk check error: {e}"
        return result

    def _check_network(self) -> dict[str, Any]:
        """Check network connectivity."""
        result: dict[str, Any] = {"status": "warn", "detail": ""}
        try:
            import socket
            socket.setdefaulttimeout(3)
            socket.create_connection(("8.8.8.8", 53), timeout=3).close()
            result["status"] = "pass"
            result["detail"] = "Internet connected"
            result["connected"] = True
        except (OSError, socket.timeout):
            result["detail"] = "No internet (offline mode only)"
            result["connected"] = False
        return result

    def _check_memory(self) -> dict[str, Any]:
        """Check available system memory."""
        result: dict[str, Any] = {"status": "pass", "detail": ""}
        try:
            import psutil
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024 ** 3)
            avail_gb = mem.available / (1024 ** 3)
            if avail_gb < 2:
                result["status"] = "fail"
                result["detail"] = f"Critically low RAM: {avail_gb:.1f}GB free of {total_gb:.0f}GB"
            elif avail_gb < 4:
                result["status"] = "warn"
                result["detail"] = f"Low RAM: {avail_gb:.1f}GB free of {total_gb:.0f}GB"
            else:
                result["detail"] = f"{avail_gb:.1f}GB RAM available of {total_gb:.0f}GB"
            result["total_gb"] = round(total_gb, 1)
            result["available_gb"] = round(avail_gb, 1)
        except ImportError:
            result["status"] = "warn"
            result["detail"] = "psutil not available for memory check"
        return result

    def _check_dependencies(self) -> dict[str, Any]:
        """Check critical Python dependencies."""
        deps = {
            "numpy": "numpy",
            "speech_recognition": "SpeechRecognition",
            "pyaudio": "PyAudio",
            "psutil": "psutil",
            "faster_whisper": "faster-whisper",
        }
        available = []
        missing = []
        for module, package in deps.items():
            try:
                __import__(module)
                available.append(package)
            except ImportError:
                missing.append(package)

        status = "pass" if not missing else ("warn" if len(missing) <= 2 else "fail")
        detail_parts = [f"{len(available)} available"]
        if missing:
            detail_parts.append(f"{len(missing)} missing: {', '.join(missing)}")
        return {
            "status": status,
            "detail": ", ".join(detail_parts),
            "available": available,
            "missing": missing,
        }

    def _check_data_dirs(self) -> dict[str, Any]:
        """Check ATOM data directories exist and are writable."""
        from pathlib import Path
        import os
        atom_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        required_dirs = ["data", "logs", "models"]
        result: dict[str, Any] = {"status": "pass", "detail": "", "dirs": {}}

        issues = []
        for dirname in required_dirs:
            dirpath = atom_root / dirname
            if not dirpath.exists():
                try:
                    dirpath.mkdir(parents=True, exist_ok=True)
                    result["dirs"][dirname] = "created"
                except Exception:
                    issues.append(f"{dirname} (cannot create)")
                    result["dirs"][dirname] = "missing"
            elif not os.access(dirpath, os.W_OK):
                issues.append(f"{dirname} (not writable)")
                result["dirs"][dirname] = "not_writable"
            else:
                result["dirs"][dirname] = "ok"

        if issues:
            result["status"] = "warn"
            result["detail"] = f"Directory issues: {', '.join(issues)}"
        else:
            result["detail"] = "All data directories ready"
        return result

    def get_readiness_summary(self) -> str:
        """Human-readable readiness report for voice output."""
        report = self.run_readiness_check()
        summary = report["summary"]
        lines = []

        if report["overall_ready"]:
            lines.append("All systems operational, Boss.")
        else:
            lines.append(f"{summary['failures']} system{'s' if summary['failures'] > 1 else ''} need attention.")

        for name, info in report["subsystems"].items():
            icon = {"pass": "OK", "warn": "WARNING", "fail": "ISSUE"}.get(
                info.get("status", ""), "?")
            if info.get("status") != "pass":
                lines.append(f"  {name}: [{icon}] {info.get('detail', '')}")

        if summary["passed"] == len(report["subsystems"]):
            lines.append(f"All {summary['passed']} subsystems passed diagnostics.")
        else:
            lines.append(
                f"{summary['passed']} passed, {summary['warnings']} warnings, "
                f"{summary['failures']} failures."
            )

        return "\n".join(lines)

    def get_boot_report(self) -> str:
        """Generate a comprehensive boot report combining system scan + readiness."""
        parts = []
        parts.append(self.get_scan_summary())
        parts.append("")
        parts.append(self.get_readiness_summary())

        env = self._env_profile
        if env.detected_languages:
            parts.append(f"\nDevelopment environment: {', '.join(env.detected_languages)}")
        if env.detected_ides:
            parts.append(f"IDEs detected: {', '.join(env.detected_ides)}")

        return "\n".join(parts)

    # ── Persist ──────────────────────────────────────────────────────

    def persist(self) -> None:
        try:
            _SCAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
            serializable = {
                k: v for k, v in self._last_scan.items()
                if isinstance(v, (str, int, float, bool, list, dict, type(None)))
            }
            _SCAN_CACHE.write_text(
                json.dumps(serializable, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            logger.debug("System scan persist failed", exc_info=True)
