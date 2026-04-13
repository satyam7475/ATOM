"""
ATOM -- Diagnostics Handler (extracted from Router).

Handles all self-check and diagnostic actions:
  - self_check: mic, TTS, brain, CPU, RAM status
  - self_diagnostic: evolution engine report
  - behavior_report: behavioral pattern analysis

Previously inlined as _do_self_check, _do_self_diagnostic, _do_behavior_report
in the Router's 1000+ line file. Extracted for single-responsibility.

Contract:
    self_check(config) -> str
    self_diagnostic() -> str
    behavior_report() -> str
    configure(stt, tts, metrics, local_brain, health_monitor)

Owner: Satyam
"""

from __future__ import annotations

import logging
from typing import Any, Callable
import psutil

logger = logging.getLogger("atom.diagnostics")


class DiagnosticsHandler:
    """Handles ATOM's self-diagnostic and self-check operations."""

    __slots__ = (
        "_stt", "_tts", "_metrics", "_local_brain",
        "_health_monitor", "_evolution", "_behavior_tracker",
        "_config", "_state_snapshot_provider", "_report_publisher",
    )

    def __init__(self, config: dict | None = None) -> None:
        self._config = config or {}
        self._stt: Any = None
        self._tts: Any = None
        self._metrics: Any = None
        self._local_brain: Any = None
        self._health_monitor: Any = None
        self._evolution: Any = None
        self._behavior_tracker: Any = None
        self._state_snapshot_provider: Callable[[], dict[str, Any]] | None = None
        self._report_publisher: Callable[[dict[str, Any]], Any] | None = None

    def configure(
        self,
        stt: Any = None,
        tts: Any = None,
        metrics: Any = None,
        local_brain: Any = None,
        health_monitor: Any = None,
        evolution: Any = None,
        behavior_tracker: Any = None,
        state_snapshot_provider: Callable[[], dict[str, Any]] | None = None,
        report_publisher: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        """Wire diagnostic dependencies after construction."""
        if stt is not None:
            self._stt = stt
        if tts is not None:
            self._tts = tts
        if metrics is not None:
            self._metrics = metrics
        if local_brain is not None:
            self._local_brain = local_brain
        if health_monitor is not None:
            self._health_monitor = health_monitor
        if evolution is not None:
            self._evolution = evolution
        if behavior_tracker is not None:
            self._behavior_tracker = behavior_tracker
        if state_snapshot_provider is not None:
            self._state_snapshot_provider = state_snapshot_provider
        if report_publisher is not None:
            self._report_publisher = report_publisher

    def _get_state_snapshot(self) -> dict[str, Any]:
        if self._state_snapshot_provider is None:
            return {}
        try:
            snapshot = self._state_snapshot_provider()
            return dict(snapshot) if isinstance(snapshot, dict) else {}
        except Exception:
            logger.debug("self_check state snapshot unavailable", exc_info=True)
            return {}

    def _publish_report(self, report: dict[str, Any]) -> None:
        if self._report_publisher is None:
            return
        try:
            self._report_publisher(report)
        except Exception:
            logger.debug("self_check report publish failed", exc_info=True)

    def self_check(self) -> str:
        """Run a comprehensive self-check of ATOM's subsystems."""
        report = self._build_self_check_report()
        self._publish_report(report)
        return str(report.get("summary_text") or "")

    def mode_status(self) -> str:
        """Return the active runtime mode from shared state, not LLM guesswork."""
        snapshot = self._get_state_snapshot()
        state_mode = snapshot.get("mode", {}) if isinstance(snapshot.get("mode"), dict) else {}
        state_reasoning = (
            snapshot.get("reasoning", {})
            if isinstance(snapshot.get("reasoning"), dict)
            else {}
        )
        requested = str(
            state_mode.get("requested")
            or self._config.get("performance", {}).get("mode", "auto")
        ).replace("_", " ")
        effective = str(state_mode.get("effective") or requested or "optimal").replace("_", " ")
        reason = str(
            state_mode.get("reason")
            or state_reasoning.get("why_this_mode")
            or "No explicit mode reason is recorded yet."
        ).strip()
        assistant_mode = str(state_mode.get("assistant_mode") or "").replace("_", " ").strip()

        parts = [
            (
                f"I'm in {effective} mode, Boss, because {reason[0].lower() + reason[1:]}"
                if reason and len(reason) > 1
                else f"I'm in {effective} mode, Boss."
            )
        ]
        if requested and requested != effective:
            parts.append(f"Requested mode is {requested}.")
        if assistant_mode:
            parts.append(f"Assistant mode is {assistant_mode}.")
        return " ".join(parts)

    def detailed_status(self) -> str:
        """Return a richer owner-facing system report grounded in shared state."""
        report = self._build_self_check_report()
        self._publish_report(report)

        system = report.get("system", {}) if isinstance(report.get("system"), dict) else {}
        context = report.get("context", {}) if isinstance(report.get("context"), dict) else {}
        voice = report.get("voice", {}) if isinstance(report.get("voice"), dict) else {}
        mode = report.get("mode", {}) if isinstance(report.get("mode"), dict) else {}
        running = report.get("running", {}) if isinstance(report.get("running"), dict) else {}
        warnings = [str(w) for w in report.get("warnings", [])]
        snapshot = self._get_state_snapshot()
        state_health = snapshot.get("health", {}) if isinstance(snapshot.get("health"), dict) else {}

        top_processes = running.get("top_processes") or []
        proc_bits: list[str] = []
        if isinstance(top_processes, list):
            for proc in top_processes[:4]:
                if isinstance(proc, dict):
                    name = str(proc.get("name") or "unknown process")
                    cpu = float(proc.get("cpu_percent", 0.0) or 0.0)
                    mem = float(proc.get("memory_percent", 0.0) or 0.0)
                    proc_bits.append(f"{name} ({cpu:.0f}% CPU, {mem:.0f}% RAM)")
                else:
                    proc_bits.append(str(proc))
        proc_summary = ", ".join(proc_bits) if proc_bits else "No heavy processes detected."

        media = context.get("media", {}) if isinstance(context.get("media"), dict) else {}
        media_summary = str(media.get("summary") or "No media playing.")
        mode_reason = str(mode.get("reason") or "No mode reason recorded yet.")
        readiness_summary = str(state_health.get("readiness_summary") or "").strip()
        warning_summary = "; ".join(warnings) if warnings else "none"
        stt_state = "ready" if bool(voice.get("stt_ok")) else "unavailable"
        tts_state = "ready" if bool(voice.get("tts_ok")) else "unavailable"
        battery_pct = system.get("battery_pct")
        battery_part = (
            f"Battery {float(battery_pct):.0f}%"
            if isinstance(battery_pct, (int, float))
            else "Battery unknown"
        )
        charge_part = "charging" if system.get("charging") else "on battery"

        parts = [
            f"Warnings first: {warning_summary}.",
            f"Health score {float(report.get('health_score', 0.0) or 0.0):.1f} out of 10.",
            f"CPU {float(system.get('cpu', 0.0) or 0.0):.0f}% and RAM {float(system.get('memory_pct', 0.0) or 0.0):.0f}%.",
            f"{battery_part}, {charge_part}. Disk free {float(system.get('disk_free_gb', 0.0) or 0.0):.1f} GB.",
            f"Foreground app {context.get('active_app') or 'unknown'} with window {context.get('window_title') or 'unknown'}.",
            f"Activity is {context.get('activity_type') or 'idle'} at confidence {float(context.get('confidence', 0.0) or 0.0):.2f}.",
            f"Idle time {float(context.get('idle_minutes', 0.0) or 0.0):.1f} minutes. Media: {media_summary}",
            f"Speech input is {stt_state} via {voice.get('stt_engine') or 'unknown'}; TTS is {tts_state} via {voice.get('tts_engine') or 'unknown'}.",
            f"Microphone: {voice.get('mic') or 'unknown'}.",
            f"Mode is {str(mode.get('effective') or 'unknown').replace('_', ' ')} because {mode_reason}",
            f"Top processes: {proc_summary}.",
            f"Recommendation: {report.get('recommendation') or 'No recommendation.'}",
        ]
        if readiness_summary:
            parts.append(f"Readiness summary: {readiness_summary}")
        return " ".join(parts)

    def _build_self_check_report(self) -> dict[str, Any]:
        from core.system.system_monitor import get_system_state

        snapshot = self._get_state_snapshot()
        state_system = snapshot.get("system", {}) if isinstance(snapshot.get("system"), dict) else {}
        state_context = snapshot.get("context", {}) if isinstance(snapshot.get("context"), dict) else {}
        state_voice = snapshot.get("voice", {}) if isinstance(snapshot.get("voice"), dict) else {}
        state_mode = snapshot.get("mode", {}) if isinstance(snapshot.get("mode"), dict) else {}
        state_health = snapshot.get("health", {}) if isinstance(snapshot.get("health"), dict) else {}
        state_reasoning = (
            snapshot.get("reasoning", {})
            if isinstance(snapshot.get("reasoning"), dict)
            else {}
        )

        warnings: list[str] = []
        stt_engine = str(
            state_voice.get("stt_engine")
            or getattr(self._stt, "backend_name", type(self._stt).__name__ if self._stt else "")
        )
        tts_engine = str(
            state_voice.get("tts_engine")
            or getattr(self._tts, "_backend", type(self._tts).__name__ if self._tts else "")
        )
        mic_name = str(
            state_voice.get("mic")
            or (getattr(self._stt, "mic_name", "") if self._stt else "")
        )
        stt_error = getattr(self._stt, "_last_error", None) or state_voice.get("error")
        permissions = state_voice.get("permissions", {}) if isinstance(state_voice.get("permissions"), dict) else {}
        speech_permission = str(
            permissions.get("speech")
            or getattr(self._stt, "speech_permission_status", "unknown")
            or "unknown"
        )
        microphone_permission = str(
            permissions.get("microphone")
            or getattr(self._stt, "microphone_permission_status", "unknown")
            or "unknown"
        )
        stt_ok = bool(
            self._stt
            and stt_engine.strip().lower() not in {"", "disabled", "unavailable"}
            and mic_name.strip().lower() not in {"", "voice input unavailable"}
            and not stt_error
        )
        tts_ok = bool(
            self._tts
            and tts_engine.strip().lower() not in {"", "disabled", "unavailable"}
        )
        brain_ok = bool(
            self._local_brain
            and getattr(self._local_brain, "available", False)
        )

        cpu_val = float(state_system.get("cpu", 0.0) or 0.0)
        ram_val = float(state_system.get("memory_pct", 0.0) or 0.0)
        battery_pct = state_system.get("battery_pct")
        charging = state_system.get("charging")
        disk_free_gb = state_system.get("disk_free_gb")
        running: dict[str, Any] = {}

        try:
            if cpu_val <= 0.0:
                cpu_val = float(psutil.cpu_percent(interval=0.1) or 0.0)
            if ram_val <= 0.0:
                ram = psutil.virtual_memory()
                ram_val = float(ram.percent or 0.0)
            if disk_free_gb in (None, 0):
                disk = psutil.disk_usage("/")
                disk_free_gb = round(disk.free / (1024 ** 3), 1)
            if battery_pct is None or charging is None:
                battery = psutil.sensors_battery()
                if battery is not None:
                    battery_pct = round(float(battery.percent), 1)
                    charging = bool(battery.power_plugged)
            running = get_system_state()
        except Exception:
            logger.debug("self_check_report system probe failed", exc_info=True)

        def _append_warning(message: str) -> None:
            msg = str(message or "").strip()
            if msg and msg not in warnings:
                warnings.append(msg)

        if not stt_ok:
            _append_warning("Speech input unavailable")
        if stt_error:
            _append_warning(f"STT error: {stt_error}")
        if speech_permission not in {"authorized", "granted", "unknown", "not_determined"}:
            _append_warning(f"Speech permission: {speech_permission}")
        if microphone_permission not in {"authorized", "granted", "unknown", "not_determined"}:
            _append_warning(f"Microphone permission: {microphone_permission}")
        if not tts_ok:
            _append_warning("TTS unavailable")
        if not brain_ok:
            _append_warning("Local brain unavailable")
        for warning in list(state_health.get("warnings") or [])[:6]:
            _append_warning(str(warning))

        readiness = state_health.get("readiness", {}) if isinstance(state_health.get("readiness"), dict) else {}
        readiness_summary = readiness.get("summary", {}) if isinstance(readiness.get("summary"), dict) else {}
        readiness_failures = int(readiness_summary.get("failures", 0) or 0)
        readiness_warnings = int(readiness_summary.get("warnings", 0) or 0)

        score = 10.0
        if not stt_ok:
            score -= 4.0
        if not tts_ok:
            score -= 2.0
        if not brain_ok:
            score -= 2.0
        score -= min(2.0, readiness_failures * 0.5)
        score -= min(1.0, readiness_warnings * 0.2)
        if ram_val >= 85.0:
            score -= 0.5
        if cpu_val >= 85.0:
            score -= 0.5
        score = max(0.0, round(score, 1))

        top_apps = list(
            running.get("active_applications")
            or state_system.get("top_processes")
            or []
        )[:5]
        active_app = str(
            running.get("foreground_app")
            or state_context.get("active_app")
            or ""
        )
        window_title = str(
            running.get("foreground_window_title")
            or state_context.get("window_title")
            or ""
        )
        running_summary = (
            f"{active_app or 'No foreground app'}"
            + (f" — {window_title}" if window_title and window_title != active_app else "")
        )

        context_report = {
            "active_app": active_app,
            "window_title": window_title,
            "activity_type": str(state_context.get("activity_type") or "idle"),
            "confidence": float(state_context.get("confidence", 0.0) or 0.0),
            "idle_minutes": float(state_context.get("idle_minutes", 0.0) or 0.0),
            "media": dict(state_context.get("media") or {}),
        }

        requested_mode = str(
            state_mode.get("requested")
            or self._config.get("performance", {}).get("mode", "auto")
        )
        effective_mode = str(state_mode.get("effective") or requested_mode)
        mode_reason = str(
            state_mode.get("reason")
            or state_reasoning.get("why_this_mode")
            or ""
        )

        recommendation = "All systems green."
        if not stt_ok:
            if stt_error and "NSSpeechRecognitionUsageDescription" in str(stt_error):
                recommendation = (
                    "Launch ATOM from the speech-enabled app bundle or enable a CLI microphone "
                    "fallback before relying on voice control."
                )
            elif stt_error and "PyAudio" in str(stt_error):
                recommendation = (
                    "Install the PortAudio microphone stack or launch through the macOS-native "
                    "speech path before relying on voice control."
                )
            else:
                recommendation = "Restore speech input before relying on hands-free control."
        elif warnings:
            recommendation = "Investigate the unavailable subsystems before long sessions."
        elif battery_pct is not None and charging is False and float(battery_pct) < 25:
            recommendation = "Running well, but low battery suggests staying in a lighter mode."
        elif context_report["activity_type"] == "meeting":
            recommendation = "Meeting context detected; keep ATOM quieter unless directly invoked."
        elif context_report["activity_type"] == "media":
            recommendation = "Media playback detected; suppress routine spoken feedback."

        issues = []
        if not stt_ok:
            issues.append("speech input")
        if not tts_ok:
            issues.append("TTS")
        if not brain_ok:
            issues.append("brain")
        if readiness_failures:
            issues.append(f"{readiness_failures} readiness checks")

        if not issues and not warnings:
            summary_text = (
                f"All systems green, Boss. CPU {cpu_val:.0f}%, RAM {ram_val:.0f}%. "
                f"Speech input, TTS, and brain are online. Mode: {effective_mode}. "
                f"Context: {running_summary}."
            )
        else:
            issue_summary = ", ".join(issues) if issues else "warnings present"
            summary_text = (
                f"System is degraded, Boss. Issues: {issue_summary}. "
                f"Health score {score:.1f} out of 10. "
                f"CPU {cpu_val:.0f}%, RAM {ram_val:.0f}%. Mode: {effective_mode}. "
                f"Context: {running_summary}."
            )

        return {
            "health_score": score,
            "system": {
                "cpu": round(cpu_val, 1),
                "memory_pct": round(ram_val, 1),
                "battery_pct": battery_pct,
                "charging": charging,
                "disk_free_gb": disk_free_gb,
                "foreground_app": active_app,
                "foreground_window_title": window_title,
                "top_processes": top_apps,
            },
            "context": context_report,
            "voice": {
                "stt_engine": stt_engine,
                "tts_engine": tts_engine,
                "mic": mic_name,
                "stt_ok": stt_ok,
                "tts_ok": tts_ok,
                "error": stt_error,
                "permissions": {
                    "speech": speech_permission,
                    "microphone": microphone_permission,
                },
            },
            "mode": {
                "requested": requested_mode,
                "effective": effective_mode,
                "reason": mode_reason,
            },
            "brain": {
                "available": brain_ok,
            },
            "running": {
                "summary": running_summary,
                "foreground_app": active_app,
                "window_title": window_title,
                "top_processes": top_apps,
            },
            "warnings": warnings,
            "recommendation": recommendation,
            "summary_text": summary_text,
        }

    def self_check_report(self) -> dict[str, Any]:
        """Structured self-check report for UI + voice parity."""
        report = self._build_self_check_report()
        self._publish_report(report)
        return report

    def self_diagnostic(self) -> str:
        """Get evolution engine diagnostic report."""
        if self._evolution is None:
            return "Self-evolution engine is not active."
        return self._evolution.format_diagnostic()

    def behavior_report(self) -> str:
        """Get behavioral pattern analysis."""
        if self._behavior_tracker is None:
            return "Behavior tracker is not active, Boss."
        suggestions = self._behavior_tracker.predict()
        self._behavior_tracker.persist()
        if not suggestions:
            return (
                "No clear usage patterns yet, Boss. "
                "Keep using me and I'll learn your habits."
            )
        return "Here are your patterns, Boss. " + " ".join(suggestions)
