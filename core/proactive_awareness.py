"""
ATOM -- Proactive awareness engine.

Generates *safe*, *dismissible* hints based on local-only signals:
  - Time-of-day greetings ("Good morning, Boss")
  - Battery warnings (already partial in system_watcher; this centralizes)
  - Idle nudges ("Still here whenever you need me")
  - App-context tips ("You're in VS Code — want me to check git status?")

All hints are gated by:
  1. features.proactive_awareness toggle
  2. SecurityPolicy (no action runs without policy)
  3. Cooldowns (same hint type not repeated within window)
  4. State machine (only fires from IDLE / LISTENING)

FRIDAY-style: present, never intrusive.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

logger = logging.getLogger("atom.proactive")

_GREETING_COOLDOWN_S = 7200
_APP_HINT_COOLDOWN_S = 300
_IDLE_HINT_COOLDOWN_S = 900
_IPHONE_PRESENCE_COOLDOWN_S = 180  # 3 min between the same presence state firing a hint
_IPHONE_TRIGGER_COOLDOWN_S = 30    # same named trigger no more than 2x / min

_APP_HINTS: dict[str, str] = {
    "code": "Want me to check git status or run tests?",
    "cursor": "Want me to check git status or run tests?",
    "vscode": "Want me to check git status or run tests?",
    "pycharm": "Want me to check git status or run tests?",
    "teams": "Need me to check your calendar?",
    "outlook": "Want me to summarize your recent emails?",
    "chrome": "Need me to search for something?",
    "edge": "Need me to search for something?",
    "firefox": "Need me to search for something?",
    "excel": "Want me to summarize the data on screen?",
}

# Presence-state -> spoken acknowledgement. Deliberately short; the
# goal is "Boss knows ATOM saw the Shortcut fire", not a monologue.
_IPHONE_PRESENCE_HINTS: dict[str, str] = {
    "at_desk": "Welcome back, Boss. Systems ready.",
    "leaving": "Got it, Boss. I'll hold everything until you're back.",
    "home": "Welcome home, Boss.",
    "away": "Got it, Boss. Quiet mode until you're home.",
    "busy": "Understood, Boss. Holding proactive hints.",
}

_IPHONE_TRIGGER_ACKS: dict[str, str] = {
    "morning_routine": "Running the morning routine, Boss.",
    "evening_wrap": "Wrapping the day, Boss.",
    "focus_on": "Focus mode on, Boss. Holding non-critical hints.",
    "focus_off": "Focus mode off, Boss.",
}


class ProactiveAwareness:
    """Generates hints from local signals. Never executes actions directly."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        feats = (config or {}).get("features", {}) or {}
        self._enabled: bool = bool(feats.get("proactive_awareness", False))
        _neg = -max(
            _GREETING_COOLDOWN_S, _APP_HINT_COOLDOWN_S, _IDLE_HINT_COOLDOWN_S,
            _IPHONE_PRESENCE_COOLDOWN_S, _IPHONE_TRIGGER_COOLDOWN_S,
        ) - 1
        self._last_greeting: float = _neg
        self._last_app_hint: float = _neg
        self._last_idle_hint: float = _neg
        self._greeted_today: bool = False
        self._last_app: str = ""

        self._last_iphone_presence_state: str = ""
        self._last_iphone_presence_ts: float = _neg
        self._iphone_trigger_last_ts: dict[str, float] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def check_greeting(self) -> str | None:
        """Return a time-of-day greeting if appropriate. Call periodically."""
        if not self._enabled:
            return None
        now = time.monotonic()
        if now - self._last_greeting < _GREETING_COOLDOWN_S:
            return None
        if self._greeted_today:
            return None
        hour = datetime.now().hour
        if 5 <= hour < 12:
            greeting = "Good morning, Boss. Systems are online and ready."
        elif 12 <= hour < 17:
            greeting = "Good afternoon, Boss. All systems nominal."
        elif 17 <= hour < 22:
            greeting = "Good evening, Boss. ATOM is standing by."
        else:
            return None
        self._last_greeting = now
        self._greeted_today = True
        return greeting

    def check_app_context(self, active_app: str) -> str | None:
        """Return an app-relevant hint if the user switched apps recently."""
        if not self._enabled or not active_app:
            return None
        now = time.monotonic()
        if now - self._last_app_hint < _APP_HINT_COOLDOWN_S:
            return None
        app_low = active_app.lower()
        if app_low == self._last_app:
            return None
        self._last_app = app_low
        for key, hint in _APP_HINTS.items():
            if key in app_low:
                self._last_app_hint = now
                return hint
        return None

    def check_idle(self, idle_seconds: float) -> str | None:
        """Return an idle hint if user has been quiet for a while.

        Uses a 45s threshold for lightweight checks (delegated to JarvisCore
        for deeper intelligence). The generic "still here" message fires
        only after 5+ minutes.
        """
        if not self._enabled:
            return None
        now = time.monotonic()
        if now - self._last_idle_hint < _IDLE_HINT_COOLDOWN_S:
            return None
        if idle_seconds < 45:
            return None
        self._last_idle_hint = now
        if idle_seconds >= 300:
            return "Still here whenever you need me, Boss."
        return None

    # ── iPhone-driven hints (Phase 1 cross_device) ────────────────────

    def handle_iphone_presence(
        self,
        state: str,
        *,
        timestamp: float | None = None,
    ) -> str | None:
        """Emit a short greeting/acknowledgement when iPhone reports a
        presence change. Suppresses:

        * duplicate same-state events within :py:data:`_IPHONE_PRESENCE_COOLDOWN_S`
        * the ``greeted_today`` flag, so ``at_desk`` during work hours
          does not fight with ``check_greeting`` -- iPhone-driven
          greetings count as "greeted" for the day.
        * the global ``enabled`` flag (feature switch).
        """
        if not self._enabled:
            return None
        st = str(state or "").strip().lower()
        hint = _IPHONE_PRESENCE_HINTS.get(st)
        if not hint:
            return None

        now = time.monotonic()
        if (
            st == self._last_iphone_presence_state
            and (now - self._last_iphone_presence_ts) < _IPHONE_PRESENCE_COOLDOWN_S
        ):
            return None

        self._last_iphone_presence_state = st
        self._last_iphone_presence_ts = now

        # Presence-driven greeting counts as today's greeting so the
        # next check_greeting() stays quiet. timestamp arg is accepted
        # for API symmetry with the bridge payload but not used here.
        _ = timestamp
        if st in {"at_desk", "home"}:
            self._last_greeting = now
            self._greeted_today = True

        return hint

    def handle_iphone_trigger(
        self,
        name: str,
        *,
        args: dict[str, Any] | None = None,
    ) -> dict[str, str] | None:
        """Acknowledge a named iPhone trigger and return the dispatch
        envelope.

        Returns ``None`` when the trigger is ignored (disabled engine,
        rate-limited, unknown name). Otherwise returns a dict with:

        * ``trigger`` -- canonical trigger name (lowercased, stripped)
        * ``ack`` -- short spoken acknowledgement
        """
        if not self._enabled:
            return None
        key = str(name or "").strip().lower()
        if not key:
            return None

        now = time.monotonic()
        last = self._iphone_trigger_last_ts.get(key, -_IPHONE_TRIGGER_COOLDOWN_S - 1)
        if (now - last) < _IPHONE_TRIGGER_COOLDOWN_S:
            return None
        self._iphone_trigger_last_ts[key] = now

        _ = args  # Phase 1 ignores args; Phase 2 trigger registry will consume.
        ack = _IPHONE_TRIGGER_ACKS.get(key, f"On it, Boss. Running {key.replace('_', ' ')}.")
        return {"trigger": key, "ack": ack}

    def on_new_day(self) -> None:
        """Reset daily state. Call from midnight check or startup."""
        self._greeted_today = False
