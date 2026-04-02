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
from typing import Any
import psutil

logger = logging.getLogger("atom.diagnostics")


class DiagnosticsHandler:
    """Handles ATOM's self-diagnostic and self-check operations."""

    __slots__ = (
        "_stt", "_tts", "_metrics", "_local_brain",
        "_health_monitor", "_evolution", "_behavior_tracker",
        "_config",
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

    def configure(
        self,
        stt: Any = None,
        tts: Any = None,
        metrics: Any = None,
        local_brain: Any = None,
        health_monitor: Any = None,
        evolution: Any = None,
        behavior_tracker: Any = None,
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

    def self_check(self) -> str:
        """Run a comprehensive self-check of ATOM's subsystems."""
        checks_ok = 0
        checks_total = 0

        checks_total += 1
        mic_ok = bool(self._stt and getattr(self._stt, "mic_name", None))
        if mic_ok:
            checks_ok += 1

        checks_total += 1
        tts_ok = bool(self._tts)
        if tts_ok:
            checks_ok += 1

        checks_total += 1
        brain_ok = bool(
            self._local_brain
            and getattr(self._local_brain, "available", False)
        )
        if brain_ok:
            checks_ok += 1

        cpu_val, ram_val = 0.0, 0.0
        checks_total += 1
        try:
            cpu_val = psutil.cpu_percent(interval=0.1)
            ram_val = psutil.virtual_memory().percent
            checks_ok += 1
        except Exception:
            pass

        perf_mode = self._config.get("performance", {}).get("mode", "full")

        if checks_ok == checks_total:
            return (
                f"All systems green, Boss. "
                f"CPU {cpu_val:.0f}%, RAM {ram_val:.0f}%. "
                f"Mic, TTS, and brain are online. Mode: {perf_mode}."
            )

        issues = []
        if not mic_ok:
            issues.append("mic")
        if not tts_ok:
            issues.append("TTS")
        if not brain_ok:
            issues.append("brain")

        return (
            f"{checks_ok} of {checks_total} systems ok. "
            f"Issues: {', '.join(issues)}. "
            f"CPU {cpu_val:.0f}%, RAM {ram_val:.0f}%. Mode: {perf_mode}."
        )

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
