"""
ATOM OS -- Background health monitor / watchdog with CPU governor.

Periodically checks the health of critical ATOM subsystems:
    - Event bus (pending task count, stuck detection)
    - State machine (stuck-state detection + auto-recovery)
    - System resources (CPU, RAM)
    - Mic stream (if available)
    - TTS (mixer initialized, consecutive failures)
    - STT (mic attached, not stuck)

CPU Governor:
    When system CPU exceeds the configured threshold, the watchdog
    automatically widens its own check interval and emits a
    ``governor_throttle`` event so other components can back off.
    When CPU drops below threshold, normal intervals resume.

Context Awareness:
    Tracks user idle time and emits ``idle_detected`` when idle
    exceeds configured threshold. Emits ``context_snapshot`` each
    cycle with time-of-day, CPU, RAM, idle, and active app info
    for the AutonomyEngine to consume.

Runs as an asyncio background task.  When a component is degraded it
logs a warning, optionally triggers auto-recovery, and notifies the
user after 3 consecutive warning cycles.

Check interval is configurable (default 60s).
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.state_manager import StateManager

logger = logging.getLogger("atom.health")

DEFAULT_CHECK_INTERVAL_S = 60.0
STUCK_STATE_THRESHOLD_DEFAULT_S = 75.0
ERROR_RECOVERY_TIMEOUT_S = 180.0
MAX_PENDING_TASKS = 50

_GOVERNOR_THROTTLE_MULTIPLIER = 2.5
_GOVERNOR_COOLDOWN_CHECKS = 3

_TIME_SLOTS = {
    range(5, 12): "morning",
    range(12, 17): "afternoon",
    range(17, 21): "evening",
}


def _time_of_day(hour: int) -> str:
    for rng, label in _TIME_SLOTS.items():
        if hour in rng:
            return label
    return "night"


class HealthMonitor:
    """Background health checker / watchdog for ATOM subsystems.

    When ``config["performance"]["cpu_governor"]`` is *True* (default),
    the governor monitors system CPU.  If it exceeds
    ``cpu_governor_threshold`` for two consecutive checks the monitor:
      1. Widens its own interval by 2.5x.
      2. Emits ``governor_throttle`` so SystemWatcher and other
         components can also back off.
      3. Logs at WARNING level so the user/dashboard sees it.
    When CPU falls back below threshold for 3 consecutive checks,
    normal intervals are restored and ``governor_normal`` is emitted.

    Additionally emits ``context_snapshot`` each cycle and
    ``idle_detected`` when user is idle beyond the configured timeout.
    """

    __slots__ = (
        "_bus", "_state", "_interval", "_base_interval",
        "_task", "_shutdown_event",
        "_last_state_change", "_last_state",
        "_stt", "_tts", "_consecutive_warnings",
        "_bt_check_counter", "_degraded_components",
        "_governor_enabled", "_governor_threshold",
        "_governor_throttled", "_governor_hot_count",
        "_governor_cool_count",
        "_last_user_activity", "_idle_timeout_min",
        "_idle_notified", "_last_cpu",
        "_stuck_state_threshold_s",
    )

    def __init__(
        self,
        bus: AsyncEventBus,
        state: StateManager,
        stt: Any = None,
        tts: Any = None,
        check_interval: float = DEFAULT_CHECK_INTERVAL_S,
        config: dict | None = None,
    ) -> None:
        self._bus = bus
        self._state = state
        self._stt = stt
        self._tts = tts
        self._base_interval = check_interval
        self._interval = check_interval
        self._task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()
        self._last_state_change: float = time.monotonic()
        self._last_state = state.current
        self._consecutive_warnings: int = 0
        self._bt_check_counter: int = 0
        self._degraded_components: set[str] = set()

        perf = (config or {}).get("performance", {})
        self._stuck_state_threshold_s: float = float(
            perf.get("stuck_state_threshold_s", STUCK_STATE_THRESHOLD_DEFAULT_S)
        )
        self._governor_enabled: bool = perf.get("cpu_governor", True)
        self._governor_threshold: int = perf.get("cpu_governor_threshold", 75)
        self._governor_throttled: bool = False
        self._governor_hot_count: int = 0
        self._governor_cool_count: int = 0

        auto_cfg = (config or {}).get("autonomy", {})
        self._idle_timeout_min: float = auto_cfg.get("idle_timeout_minutes", 10.0)
        self._last_user_activity: float = time.monotonic()
        self._idle_notified: bool = False
        self._last_cpu: float = 0.0

    def start(self) -> None:
        self._bus.on("state_changed", self._on_state_changed)
        self._bus.on("speech_final", self._on_user_active)
        self._bus.on("intent_classified", self._on_user_active)
        self._task = asyncio.create_task(self._run())
        logger.info("Health watchdog started (interval=%.0fs)", self._interval)

    async def stop(self) -> None:
        self._shutdown_event.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _on_state_changed(self, old=None, new=None, **_kw) -> None:
        self._last_state_change = time.monotonic()
        self._last_state = new

    async def _on_user_active(self, **_kw: Any) -> None:
        self._last_user_activity = time.monotonic()
        self._idle_notified = False

    async def _run(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self._interval,
                )
                break
            except asyncio.TimeoutError:
                pass

            issues = await self._check_all()

            self._emit_context_snapshot()
            self._check_idle()

            if issues:
                self._consecutive_warnings += 1
                for issue in issues:
                    logger.warning("HEALTH: %s", issue)
                if self._consecutive_warnings >= 3:
                    logger.error(
                        "HEALTH: %d consecutive warning cycles -- "
                        "system may be degraded",
                        self._consecutive_warnings,
                    )
                    degraded_list = ", ".join(self._degraded_components) or "unknown"
                    self._bus.emit(
                        "response_ready",
                        text=f"Boss, I'm having trouble with: {degraded_list}. "
                             "Check logs for details.",
                    )
                    self._consecutive_warnings = 0
            else:
                if self._consecutive_warnings > 0:
                    logger.info("HEALTH: All checks passed (recovered)")
                self._consecutive_warnings = 0
                self._degraded_components.clear()

    async def _check_all(self) -> list[str]:
        issues: list[str] = []
        self._degraded_components.clear()

        issues.extend(self._check_event_bus())
        issues.extend(await self._check_state_machine())
        issues.extend(self._check_system_resources())
        issues.extend(self._check_mic())
        issues.extend(self._check_tts())

        self._bt_check_counter += 1
        if self._bt_check_counter % 4 == 0:
            self._check_bluetooth_devices()

        return issues

    def _check_event_bus(self) -> list[str]:
        issues: list[str] = []
        pending = self._bus.pending_count
        if pending > MAX_PENDING_TASKS:
            issues.append(
                f"Event bus has {pending} pending tasks "
                f"(threshold={MAX_PENDING_TASKS})"
            )
            self._degraded_components.add("event_bus")
        return issues

    async def _check_state_machine(self) -> list[str]:
        from core.state_manager import AtomState

        issues: list[str] = []
        now = time.monotonic()
        age = now - self._last_state_change
        current = self._state.current

        stuck_states = {AtomState.THINKING, AtomState.SPEAKING}
        if current in stuck_states and age >= self._stuck_state_threshold_s:
            issues.append(
                f"State '{current.value}' unchanged for {age:.0f}s "
                f"(threshold={self._stuck_state_threshold_s:.0f}s) -- auto-recovering"
            )
            self._degraded_components.add("state_machine")
            try:
                if current is AtomState.SPEAKING and self._tts is not None:
                    try:
                        await self._tts.stop()
                    except Exception:
                        logger.debug("HEALTH: tts.stop during stuck recovery", exc_info=True)
                await self._state.transition(AtomState.LISTENING)
                self._bus.emit("restart_listening")
                logger.info("HEALTH: Auto-recovered from stuck '%s' -> LISTENING",
                            current.value)
            except Exception as exc:
                logger.error("HEALTH: Auto-recovery failed: %s", exc)

        if current is AtomState.ERROR_RECOVERY and age >= ERROR_RECOVERY_TIMEOUT_S:
            issues.append(
                f"ERROR_RECOVERY state for {age:.0f}s -- forcing recovery via IDLE"
            )
            self._degraded_components.add("state_machine")
            try:
                await self._state.transition(AtomState.IDLE)
                logger.info("HEALTH: Forced recovery from ERROR_RECOVERY -> IDLE")
            except Exception as exc:
                logger.error("HEALTH: Forced recovery failed: %s", exc)

        return issues

    def _check_system_resources(self) -> list[str]:
        issues: list[str] = []
        try:
            import psutil
            mem = psutil.virtual_memory()
            if mem.percent > 90:
                issues.append(
                    f"RAM usage critical: {mem.percent:.0f}% "
                    f"({mem.available / (1024**3):.1f} GB free)"
                )
                self._degraded_components.add("memory")

            cpu = psutil.cpu_percent(interval=0)
            self._last_cpu = cpu
            if cpu > 95:
                issues.append(f"CPU usage critical: {cpu:.0f}%")
                self._degraded_components.add("cpu")

            if self._governor_enabled:
                self._governor_update(cpu)

        except ImportError:
            pass
        except Exception as exc:
            issues.append(f"System resource check failed: {exc}")
        return issues

    def _governor_update(self, cpu: float) -> None:
        """CPU governor: widen/restore check intervals based on load."""
        if cpu > self._governor_threshold:
            self._governor_hot_count += 1
            self._governor_cool_count = 0
            if not self._governor_throttled and self._governor_hot_count >= 2:
                self._governor_throttled = True
                self._interval = self._base_interval * _GOVERNOR_THROTTLE_MULTIPLIER
                logger.warning(
                    "CPU Governor: CPU at %.0f%% (threshold %d%%) -- "
                    "throttling interval %.0fs -> %.0fs",
                    cpu, self._governor_threshold,
                    self._base_interval, self._interval,
                )
                self._bus.emit_fast("governor_throttle", cpu=cpu)
        else:
            self._governor_cool_count += 1
            self._governor_hot_count = 0
            if self._governor_throttled and self._governor_cool_count >= _GOVERNOR_COOLDOWN_CHECKS:
                self._governor_throttled = False
                self._interval = self._base_interval
                logger.info(
                    "CPU Governor: CPU normalized (%.0f%%) -- "
                    "restoring interval %.0fs",
                    cpu, self._base_interval,
                )
                self._bus.emit_fast("governor_normal", cpu=cpu)

    @property
    def is_throttled(self) -> bool:
        return self._governor_throttled

    # ── Idle Detection ────────────────────────────────────────────────

    def _check_idle(self) -> None:
        idle_sec = time.monotonic() - self._last_user_activity
        idle_min = idle_sec / 60.0
        if idle_min >= self._idle_timeout_min and not self._idle_notified:
            self._idle_notified = True
            self._bus.emit_fast(
                "idle_detected", idle_minutes=idle_min,
            )
            logger.info("User idle for %.0f minutes", idle_min)

    @property
    def idle_minutes(self) -> float:
        return (time.monotonic() - self._last_user_activity) / 60.0

    # ── Context Snapshot ──────────────────────────────────────────────

    def _emit_context_snapshot(self) -> None:
        """Emit current system context for the autonomy engine."""
        now = datetime.now()
        active_app = ""
        try:
            import sys
            if sys.platform == "win32":
                import ctypes
                hwnd = ctypes.windll.user32.GetForegroundWindow()
                length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
                    active_app = buf.value[:80]
        except Exception:
            pass

        cpu = self._last_cpu
        ram = 0.0
        try:
            import psutil
            ram = psutil.virtual_memory().percent
        except Exception:
            pass

        self._bus.emit_fast(
            "context_snapshot",
            time_of_day=_time_of_day(now.hour),
            hour=now.hour,
            cpu=cpu,
            ram=ram,
            idle_minutes=self.idle_minutes,
            active_app=active_app,
            is_weekday=now.weekday() < 5,
            weekday=now.weekday(),
        )

    # ── Subsystem Checks ─────────────────────────────────────────────

    def _check_mic(self) -> list[str]:
        issues: list[str] = []
        if self._stt is None:
            return issues

        mic_name = getattr(self._stt, "mic_name", None)
        device_idx = getattr(self._stt, "_mic_device_index", None)
        if mic_name and device_idx is not None:
            logger.debug("HEALTH mic: [%s] '%s'", device_idx, mic_name)
        elif not mic_name:
            issues.append("STT: no microphone detected")
            self._degraded_components.add("stt")

        return issues

    def _check_tts(self) -> list[str]:
        issues: list[str] = []
        if self._tts is None:
            return issues

        mixer_ready = getattr(self._tts, "_mixer_ready", False)
        if not mixer_ready:
            issues.append("TTS: mixer not initialized")
            self._degraded_components.add("tts")

        failures = getattr(self._tts, "_consecutive_failures", 0)
        if failures >= 3:
            issues.append(f"TTS: {failures} consecutive failures -- may need restart")
            self._degraded_components.add("tts")

        return issues

    def _check_bluetooth_devices(self) -> None:
        if self._stt is not None and hasattr(self._stt, "refresh_mic"):
            try:
                changed = self._stt.refresh_mic()
                if changed:
                    self._bus.emit("mic_changed",
                                   name=getattr(self._stt, "mic_name", ""))
            except Exception as exc:
                logger.debug("BT input check error: %s", exc)

        if self._tts is not None and hasattr(self._tts, "refresh_output_device"):
            try:
                from core.state_manager import AtomState
                if self._state.current not in (AtomState.SPEAKING,):
                    self._tts.refresh_output_device()
            except Exception as exc:
                logger.debug("BT output check error: %s", exc)
