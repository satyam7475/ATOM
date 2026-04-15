"""
ATOM -- Inline Suggestion Engine.

Fires immediately after a command completes, analyzing the command + intent +
system context to offer a contextual follow-up suggestion. This is different
from the background ProactiveEngine which runs on a 5-minute loop.

Rules:
  - App-context hints ("You opened Safari -- want me to search for something?")
  - Trip/budget follow-ups
  - Repeat-command shortcuts
  - Cooldown: same suggestion type not repeated within 5 minutes
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("atom.suggestion_engine")

_COOLDOWN_S = 300  # 5 minutes

_APP_SUGGESTIONS: dict[str, str] = {
    "safari": "Want me to search for something, Boss?",
    "google chrome": "Want me to search for something or open a specific site?",
    "arc": "Want me to search for something, Boss?",
    "firefox": "Need me to look anything up for you?",
    "spotify": "Want me to play something specific?",
    "music": "Want me to play something specific?",
    "terminal": "Need me to run a command for you?",
    "iterm": "Need me to run a command for you?",
    "notes": "Want me to create a new note or read your recent ones?",
    "calendar": "Want me to check your schedule or add an event?",
    "maps": "Need directions somewhere, Boss?",
    "mail": "Want me to check for new emails?",
    "messages": "Want me to read your recent messages?",
    "finder": "Looking for a specific file?",
}

_INTENT_FOLLOW_UPS: dict[str, str] = {
    "open_app": "Anything specific you want to do in there?",
    "web_search": "Want me to dig deeper into that?",
    "set_timer": "Need another timer or reminder?",
    "set_reminder": "Anything else to remember?",
    "play_music": "Want to adjust the volume or queue another track?",
    "screenshot": "Want me to analyze what's on screen?",
    "send_message": "Any follow-up to send?",
}

_REPEAT_THRESHOLD = 3


class SuggestionEngine:
    """Inline post-command suggestion generator with cooldown tracking."""

    __slots__ = ("_last_fired", "_repeat_counts")

    def __init__(self) -> None:
        self._last_fired: dict[str, float] = {}
        self._repeat_counts: dict[str, int] = {}

    def suggest(
        self,
        command_text: str,
        *,
        intent: str = "",
        active_app: str = "",
        response_text: str = "",
    ) -> str | None:
        """Return an optional follow-up suggestion, or None.

        Args:
            command_text: The user's original command.
            intent: Classified intent name (e.g. "open_app").
            active_app: Currently focused application name.
            response_text: The response ATOM just gave.
        """
        now = time.monotonic()
        cmd_lower = command_text.lower()

        # Repeat-command shortcut detection
        self._repeat_counts[cmd_lower] = self._repeat_counts.get(cmd_lower, 0) + 1
        if self._repeat_counts[cmd_lower] >= _REPEAT_THRESHOLD:
            suggestion = self._try_emit(
                "repeat",
                f"You've done this {self._repeat_counts[cmd_lower]} times. "
                "Want me to make it a shortcut?",
                now,
            )
            if suggestion:
                return suggestion

        # Intent-based follow-ups
        if intent and intent in _INTENT_FOLLOW_UPS:
            suggestion = self._try_emit(
                f"intent:{intent}",
                _INTENT_FOLLOW_UPS[intent],
                now,
            )
            if suggestion:
                return suggestion

        # App-context hints
        if active_app:
            app_key = active_app.lower()
            for name, hint in _APP_SUGGESTIONS.items():
                if name in app_key:
                    suggestion = self._try_emit(f"app:{name}", hint, now)
                    if suggestion:
                        return suggestion
                    break

        return None

    def _try_emit(self, key: str, text: str, now: float) -> str | None:
        """Emit suggestion only if cooldown has elapsed."""
        last = self._last_fired.get(key, 0.0)
        if now - last < _COOLDOWN_S:
            return None
        self._last_fired[key] = now
        logger.debug("Suggestion fired [%s]: %s", key, text[:60])
        return text

    def reset_cooldowns(self) -> None:
        """Clear all cooldown timers (useful for testing)."""
        self._last_fired.clear()
        self._repeat_counts.clear()


__all__ = ["SuggestionEngine"]
