"""
ATOM -- Unified speech parameter controller.

Single merge point for perception (real-time emotion/urgency) and
adaptive (learned user profile) rate/pause signals.  Both inputs
compose multiplicatively so neither overwrites the other.

Owner: Satyam
"""

from __future__ import annotations

import logging

logger = logging.getLogger("atom.speech_controller")


class SpeechController:
    """Merge perception, adaptive, and mood speech params for TTS.

    Sprint N7 added the third ``mood`` input so a steady mood signal
    (e.g. focused / tired / alert) can shape Boss's prosody without
    interfering with the burst-y perception channel that reacts to
    the user's current utterance.
    """

    __slots__ = ("_perception", "_adaptive", "_mood")

    def __init__(self) -> None:
        self._perception = {"rate_multiplier": 1.0, "pause_multiplier": 1.0}
        self._adaptive = {"rate_multiplier": 1.0, "pause_multiplier": 1.0}
        self._mood = {"rate_multiplier": 1.0, "pause_multiplier": 1.0}

    def set_perception(
        self,
        rate_multiplier: float = 1.0,
        pause_multiplier: float = 1.0,
    ) -> None:
        self._perception = {
            "rate_multiplier": rate_multiplier,
            "pause_multiplier": pause_multiplier,
        }

    def set_adaptive(
        self,
        rate_multiplier: float = 1.0,
        pause_multiplier: float = 1.0,
    ) -> None:
        self._adaptive = {
            "rate_multiplier": rate_multiplier,
            "pause_multiplier": pause_multiplier,
        }

    def set_mood(
        self,
        rate_multiplier: float = 1.0,
        pause_multiplier: float = 1.0,
    ) -> None:
        """Steady mood-driven channel (third multiplier)."""
        self._mood = {
            "rate_multiplier": rate_multiplier,
            "pause_multiplier": pause_multiplier,
        }

    def merged(self) -> dict[str, float]:
        """Multiplicative composition of all signal sources."""
        return {
            "rate_multiplier": round(
                self._perception["rate_multiplier"]
                * self._adaptive["rate_multiplier"]
                * self._mood["rate_multiplier"],
                3,
            ),
            "pause_multiplier": round(
                self._perception["pause_multiplier"]
                * self._adaptive["pause_multiplier"]
                * self._mood["pause_multiplier"],
                3,
            ),
        }

    def reset(self) -> None:
        """Reset perception + adaptive (mood persists across turns)."""
        self._perception = {"rate_multiplier": 1.0, "pause_multiplier": 1.0}
        self._adaptive = {"rate_multiplier": 1.0, "pause_multiplier": 1.0}
