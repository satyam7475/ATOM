"""
ATOM -- Instant Acknowledgement Engine.

Plays a sub-100ms verbal acknowledgement the moment speech is finalized,
BEFORE intent classification or LLM processing begins. This is what makes
the system feel responsive -- the user hears "On it." within 100ms of
finishing their sentence, then the real response follows.

Siri does: [beep] -> silence -> response
ATOM does: "On it." -> streaming response

The AckEngine selects contextually appropriate phrases and prevents
duplicate or redundant acks.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

logger = logging.getLogger("atom.ack_engine")

_QUICK_ACKS = [
    "On it.",
    "Got it.",
    "Working on it.",
    "One moment.",
    "Let me check.",
    "Right away.",
]

_GREETING_ACKS = [
    "Yes, Boss?",
    "I'm here.",
    "Go ahead.",
    "What do you need?",
]

_FOLLOW_UP_ACKS = [
    "Sure.",
    "Alright.",
    "Okay.",
]

_FAST_INTENT_ACKS: dict[str, str] = {
    "time": "",
    "date": "",
    "battery": "",
    "cpu": "",
    "ram": "",
    "greeting": "",
    "thanks": "",
    "confirm": "",
    "deny": "",
    # Sprint Ω.13 (Apr 27 2026): expand the synchronous suppression
    # layer. The deferred-ACK timer in ``CommandLoop`` already cancels
    # any ack whose response arrives in <280 ms, but for *known* fast
    # intents (clock readouts, system status, info lookups) we never
    # want to spawn the timer in the first place — both belt and braces.
    "status": "",
    "info": "",
    "clock": "",
}

_MIN_QUERY_LEN_FOR_ACK = 3
_ACK_COOLDOWN_S = 1.0


class AckEngine:
    """Instant acknowledgement before command processing."""

    def __init__(self) -> None:
        self._last_ack_time: float = 0.0
        self._ack_count: int = 0
        self._suppressed_count: int = 0

    def should_ack(self, text: str, *, intent: str = "") -> bool:
        """Decide whether an ack is appropriate for this input."""
        if not text or len(text.strip()) < _MIN_QUERY_LEN_FOR_ACK:
            return False

        if intent in _FAST_INTENT_ACKS:
            return False

        now = time.monotonic()
        if now - self._last_ack_time < _ACK_COOLDOWN_S:
            self._suppressed_count += 1
            return False

        return True

    def get_ack(self, text: str, *, is_follow_up: bool = False) -> str:
        """Select a contextually appropriate ack phrase.

        Returns empty string if no ack should play.
        """
        if not self.should_ack(text):
            return ""

        self._last_ack_time = time.monotonic()
        self._ack_count += 1

        if is_follow_up:
            return random.choice(_FOLLOW_UP_ACKS)

        lower = text.lower().strip()
        if any(w in lower for w in ("hey atom", "atom", "hey")):
            return random.choice(_GREETING_ACKS)

        return random.choice(_QUICK_ACKS)

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            "ack_count": self._ack_count,
            "suppressed_count": self._suppressed_count,
        }


__all__ = ["AckEngine"]
