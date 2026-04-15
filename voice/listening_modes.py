"""
ATOM -- Dual-Channel Listening Modes.

Separates STT behavior into two modes so the microphone never fully
stops -- Siri/Alexa style:

  PASSIVE: STT runs but only wake word detection is active.
           Audio is captured at low cost; transcripts are checked
           for wake phrases but NOT emitted as ``speech_final``.

  ACTIVE:  Full transcription. Every partial/final is emitted to
           the command pipeline.

The VoicePipeline controls mode switching:
  - ``wake_word_detected`` -> ACTIVE
  - Command completes (TTS done) -> PASSIVE (if wake word mode)
  - Always-listen config -> permanent ACTIVE

This eliminates "dead ears" -- STT never blocks on state.
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from typing import Any

logger = logging.getLogger("atom.listening_modes")


class ListeningMode(Enum):
    PASSIVE = 1
    ACTIVE = 2


class ListeningModeController:
    """Thread-safe controller for STT listening mode."""

    def __init__(self, *, always_active: bool = False) -> None:
        self._mode = ListeningMode.ACTIVE if always_active else ListeningMode.PASSIVE
        self._always_active = always_active
        self._lock = threading.Lock()
        self._last_switch_time: float = 0.0
        self._active_count: int = 0
        self._passive_count: int = 0

    @property
    def mode(self) -> ListeningMode:
        with self._lock:
            return self._mode

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._mode is ListeningMode.ACTIVE

    @property
    def is_passive(self) -> bool:
        with self._lock:
            return self._mode is ListeningMode.PASSIVE

    def activate(self, reason: str = "") -> bool:
        """Switch to ACTIVE mode. Returns True if mode actually changed."""
        with self._lock:
            if self._mode is ListeningMode.ACTIVE:
                return False
            self._mode = ListeningMode.ACTIVE
            self._last_switch_time = time.monotonic()
            self._active_count += 1
        logger.info("ListeningMode -> ACTIVE (%s)", reason or "unspecified")
        return True

    def deactivate(self, reason: str = "") -> bool:
        """Switch to PASSIVE mode. Returns True if mode actually changed."""
        if self._always_active:
            return False
        with self._lock:
            if self._mode is ListeningMode.PASSIVE:
                return False
            self._mode = ListeningMode.PASSIVE
            self._last_switch_time = time.monotonic()
            self._passive_count += 1
        logger.info("ListeningMode -> PASSIVE (%s)", reason or "unspecified")
        return True

    def set_always_active(self, always: bool) -> None:
        with self._lock:
            self._always_active = always
            if always and self._mode is ListeningMode.PASSIVE:
                self._mode = ListeningMode.ACTIVE

    def get_diagnostics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "mode": self._mode.name,
                "always_active": self._always_active,
                "active_count": self._active_count,
                "passive_count": self._passive_count,
                "last_switch_time": self._last_switch_time,
            }


class WakeWordFilter:
    """Separate wake word detection from STT transcription.

    Processes partial transcripts and detects wake phrases independently
    of the main STT pipeline, with configurable sensitivity and cooldown.
    """

    WAKE_PHRASES = frozenset({"hey atom", "atom", "hey computer"})

    def __init__(self, *, cooldown_s: float = 1.5) -> None:
        self._cooldown_s = max(0.3, float(cooldown_s))
        self._last_trigger_time: float = 0.0
        self._trigger_count: int = 0
        self._lock = threading.Lock()

    def check(self, text: str) -> str | None:
        """Check partial text for wake phrases.

        Returns the matched wake phrase, or None.
        Thread-safe with cooldown to prevent rapid re-triggering.
        """
        if not text:
            return None

        lower = text.lower().strip()
        now = time.monotonic()

        with self._lock:
            if now - self._last_trigger_time < self._cooldown_s:
                return None

        for phrase in self.WAKE_PHRASES:
            if lower.endswith(phrase) or lower == phrase:
                with self._lock:
                    self._last_trigger_time = now
                    self._trigger_count += 1
                return phrase

        return None

    @property
    def trigger_count(self) -> int:
        return self._trigger_count


__all__ = ["ListeningMode", "ListeningModeController", "WakeWordFilter"]
