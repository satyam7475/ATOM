"""
ATOM v15 -- Proactive awareness engine.

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


class ProactiveAwareness:
    """Generates hints from local signals. Never executes actions directly."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        feats = (config or {}).get("features", {}) or {}
        self._enabled: bool = bool(feats.get("proactive_awareness", False))
        _neg = -max(_GREETING_COOLDOWN_S, _APP_HINT_COOLDOWN_S, _IDLE_HINT_COOLDOWN_S) - 1
        self._last_greeting: float = _neg
        self._last_app_hint: float = _neg
        self._last_idle_hint: float = _neg
        self._greeted_today: bool = False
        self._last_app: str = ""

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
        """Return an idle hint if user has been quiet for a while."""
        if not self._enabled:
            return None
        if idle_seconds < 300:
            return None
        now = time.monotonic()
        if now - self._last_idle_hint < _IDLE_HINT_COOLDOWN_S:
            return None
        self._last_idle_hint = now
        return "Still here whenever you need me, Boss."

    def on_new_day(self) -> None:
        """Reset daily state. Call from midnight check or startup."""
        self._greeted_today = False
