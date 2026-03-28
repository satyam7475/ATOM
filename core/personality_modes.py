"""
ATOM -- Multi-Mode Personality System.

Four operational modes that change ATOM's entire behavior:

  WORK   - Professional, brisk. Productivity tips. Reminders allowed.
  FOCUS  - Minimal voice, no suggestions. Only urgent interruptions.
  CHILL  - Friendly, relaxed. Casual suggestions. Entertainment.
  SLEEP  - Silent or ultra-soft. No interruptions except emergency.

Activation:
  - Voice: "focus mode", "chill mode", "work mode", "sleep mode"
  - Auto: BehaviorModel can suggest mode changes based on energy
  - Dashboard: visual toggle

Each mode adjusts:
  - Voice profile overrides (rate, pitch, volume)
  - Suggestion gating (allow / block / urgent-only)
  - Response verbosity (full / minimal / silent)
  - Proactive alerts (on / off)

SAFETY: Focus mode defines "urgent" as:
  - Battery < 10%
  - Reminders marked critical
  - Explicit "break focus" voice command
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.cognitive.behavior_model import BehaviorModel

logger = logging.getLogger("atom.modes")


class ModeConfig:
    """Configuration for a personality mode."""
    __slots__ = (
        "name", "voice_rate_adj", "voice_pitch_adj", "voice_volume_adj",
        "allow_suggestions", "allow_interruptions", "urgent_only",
        "verbosity", "proactive_alerts",
    )

    def __init__(
        self, name: str, *,
        voice_rate_adj: int = 0, voice_pitch_adj: int = 0, voice_volume_adj: int = 0,
        allow_suggestions: bool = True, allow_interruptions: bool = True,
        urgent_only: bool = False, verbosity: str = "full",
        proactive_alerts: bool = True,
    ) -> None:
        self.name = name
        self.voice_rate_adj = voice_rate_adj
        self.voice_pitch_adj = voice_pitch_adj
        self.voice_volume_adj = voice_volume_adj
        self.allow_suggestions = allow_suggestions
        self.allow_interruptions = allow_interruptions
        self.urgent_only = urgent_only
        self.verbosity = verbosity
        self.proactive_alerts = proactive_alerts


_MODES: dict[str, ModeConfig] = {
    "work": ModeConfig(
        "work",
        voice_rate_adj=2, voice_pitch_adj=0, voice_volume_adj=0,
        allow_suggestions=True, allow_interruptions=True,
        verbosity="full", proactive_alerts=True,
    ),
    "focus": ModeConfig(
        "focus",
        voice_rate_adj=5, voice_pitch_adj=-2, voice_volume_adj=-5,
        allow_suggestions=False, allow_interruptions=False,
        urgent_only=True, verbosity="minimal", proactive_alerts=False,
    ),
    "chill": ModeConfig(
        "chill",
        voice_rate_adj=-3, voice_pitch_adj=-1, voice_volume_adj=-2,
        allow_suggestions=True, allow_interruptions=True,
        verbosity="full", proactive_alerts=False,
    ),
    "sleep": ModeConfig(
        "sleep",
        voice_rate_adj=-5, voice_pitch_adj=-3, voice_volume_adj=-10,
        allow_suggestions=False, allow_interruptions=False,
        urgent_only=True, verbosity="silent", proactive_alerts=False,
    ),
}

_URGENT_EVENTS = frozenset({
    "battery_critical", "shutdown_requested", "critical_reminder",
})


class PersonalityModes:
    """Multi-mode personality controller for ATOM."""

    __slots__ = (
        "_bus", "_bmodel", "_config",
        "_current_mode", "_auto_switching", "_default_mode",
        "_queued_suggestions", "_mode_history",
    )

    def __init__(
        self,
        bus: AsyncEventBus,
        behavior_model: BehaviorModel | None = None,
        config: dict | None = None,
    ) -> None:
        self._bus = bus
        self._bmodel = behavior_model
        cfg = (config or {}).get("cognitive", {})
        self._config = cfg
        self._default_mode = cfg.get("default_mode", "work")
        self._auto_switching = cfg.get("auto_mode_switching", True)
        self._current_mode: str = self._default_mode
        self._queued_suggestions: list[dict] = []
        self._mode_history: list[tuple[str, float]] = []

    def start(self) -> None:
        if self._auto_switching:
            self._bus.on("user_energy_state", self._on_energy_state)
        logger.info("Personality modes started (mode=%s, auto=%s)",
                     self._current_mode, self._auto_switching)

    def stop(self) -> None:
        pass

    # ── Mode Switching ─────────────────────────────────────────────────

    def switch_mode(self, mode: str) -> str:
        """Switch to a new personality mode."""
        mode = mode.lower().strip()
        if mode not in _MODES:
            return f"Unknown mode '{mode}'. Available: work, focus, chill, sleep."

        old = self._current_mode
        if mode == old:
            return f"Already in {mode} mode, Boss."

        self._current_mode = mode
        self._mode_history.append((mode, __import__("time").time()))

        mc = _MODES[mode]
        self._bus.emit_fast(
            "mode_changed",
            mode=mode,
            old_mode=old,
            voice_rate_adj=mc.voice_rate_adj,
            voice_pitch_adj=mc.voice_pitch_adj,
            voice_volume_adj=mc.voice_volume_adj,
            allow_suggestions=mc.allow_suggestions,
            verbosity=mc.verbosity,
        )

        if mode == "focus" and self._queued_suggestions:
            self._queued_suggestions.clear()
        elif mode != "focus" and old == "focus" and self._queued_suggestions:
            for s in self._queued_suggestions[:3]:
                self._bus.emit_fast("habit_suggestion", **s)
            self._queued_suggestions.clear()

        logger.info("Mode switched: %s -> %s", old, mode)
        _MODE_MESSAGES = {
            "work": "Work mode, Boss. Let's get productive. I'll be sharp and efficient.",
            "focus": "Focus mode activated, Boss. I'll keep quiet unless it's urgent. Deep work time.",
            "chill": "Chill mode, Boss. We're taking it easy. I'm here for whatever, no pressure.",
            "sleep": "Sleep mode, Boss. I'll be silent. Only emergencies will reach you. Rest well.",
        }
        return _MODE_MESSAGES.get(mode, f"Switched to {mode} mode, Boss.")

    async def _on_energy_state(self, energy: str = "", **_kw: Any) -> None:
        """Auto-suggest mode changes based on energy state."""
        if not self._auto_switching:
            return

        hour = datetime.now().hour
        current = self._current_mode

        if energy == "resting" and hour >= 23 and current != "sleep":
            self._bus.emit_fast(
                "habit_suggestion",
                text="It's late and you seem to be resting. Want me to switch to sleep mode?",
                habit_id="_auto_sleep_mode",
                confidence=0.7,
            )
        elif energy == "high" and current == "chill":
            pass

    # ── Gate Checks ────────────────────────────────────────────────────

    def should_allow_suggestion(self) -> bool:
        return _MODES[self._current_mode].allow_suggestions

    def should_allow_interruption(self, event_type: str = "") -> bool:
        mc = _MODES[self._current_mode]
        if mc.allow_interruptions:
            return True
        if mc.urgent_only and event_type in _URGENT_EVENTS:
            return True
        return False

    def queue_suggestion(self, suggestion: dict) -> None:
        """Queue a suggestion for delivery when exiting focus mode."""
        if self._current_mode == "focus":
            self._queued_suggestions.append(suggestion)
            if len(self._queued_suggestions) > 10:
                self._queued_suggestions = self._queued_suggestions[-10:]

    @property
    def current_mode(self) -> str:
        return self._current_mode

    @property
    def current_config(self) -> ModeConfig:
        return _MODES[self._current_mode]

    @property
    def verbosity(self) -> str:
        return _MODES[self._current_mode].verbosity

    def get_mode_for_dashboard(self) -> dict:
        mc = _MODES[self._current_mode]
        return {
            "current": self._current_mode,
            "auto_switching": self._auto_switching,
            "allow_suggestions": mc.allow_suggestions,
            "allow_interruptions": mc.allow_interruptions,
            "verbosity": mc.verbosity,
            "queued_count": len(self._queued_suggestions),
        }
