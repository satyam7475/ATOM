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
import re
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

    STT-mishearing tolerance
    ------------------------
    Apple's SFSpeechRecognizer regularly renders "atom" as the nearest
    in-vocabulary English word — especially under the en-IN locale
    which has no "atom" as a common word. Observed mishearings include
    "adam", "atum", "autumn", "atam", "adom", "atan", and "adtan". We therefore accept all
    of these as equivalent to "atom" for wake purposes. The tradeoff is
    a small false-positive risk if the user addresses a real person
    named Adam within earshot; in a personal-assistant context that is
    acceptable and far preferable to the pre-fix behaviour of silently
    dropping every query because the recognizer rendered "adam".
    """

    _ATOM_VARIANTS: tuple[str, ...] = (
        "atom", "adam", "atum", "autumn", "atam", "adom",
        "atan", "adtan", "adton", "attom", "adum",
        "ottam", "autam", "odum", "aadam", "aatom", "atam",
    )

    # "Boss"-family direct-address tokens. Owner addresses ATOM as "Boss"
    # so any utterance that opens with these equivalents is treated as a
    # direct call to ATOM. Crucially, "Dear Boss" is the recurring SFSpeech
    # mishearing of "Hey Boss" / "Hey ATOM" we observe in production logs;
    # promoting it here stops a bare "dear boss" partial from being routed
    # to the LLM as a noisy fallback query.
    # Multi-word boss / owner-address openers. Multi-word forms are safe to
    # match mid-sentence because their false-positive surface is tiny.
    # ``yeah/yep/cool/right/thanks/sure boss`` are common conversational
    # acks that Boss says back to ATOM; treating them as bare-wake stops
    # the LLM from spinning up a 5-second response for a one-word reply.
    _BOSS_OPENERS: tuple[str, ...] = (
        "hey boss", "hi boss", "yo boss", "ok boss", "okay boss",
        "yes boss", "yeah boss", "yep boss", "yup boss",
        "cool boss", "right boss", "sure boss",
        "thanks boss", "thank you boss",
        "hello boss", "dear boss",
        "hey satyam", "hi satyam",
    )

    # Bare wake/owner tokens that ONLY count as a wake when the utterance
    # is essentially just that token (handled by the router's bare-wake
    # short-circuit, never by ``_DIRECT_ADDRESS_RE``). Keeping ``"satyam"``
    # out of the direct-address regex prevents mid-sentence mentions like
    # "I called my friend Satyam" from waking ATOM.
    _BARE_OWNER_TOKENS: tuple[str, ...] = ("satyam",)

    DIRECT_ADDRESS: tuple[str, ...] = (
        "are you there", "you there", "can you hear me",
        "do you hear me", "hear me", "you listening",
        "are you listening", "hello atom", "hello adam",
        "respond", "say something",
    ) + _BOSS_OPENERS

    WAKE_PHRASES: frozenset[str] = frozenset(
        {v for v in _ATOM_VARIANTS}
        | {f"hey {v}" for v in _ATOM_VARIANTS}
        | {f"hi {v}" for v in _ATOM_VARIANTS}
        | {"hey computer", "hey jarvis"}
        | set(_BOSS_OPENERS)
        | set(_BARE_OWNER_TOKENS)
    )

    _WAKE_RE = re.compile(
        r"\b(?:hey|hi|ok|okay|yo)?\s*("
        + "|".join(re.escape(v) for v in _ATOM_VARIANTS)
        + r"|computer|jarvis)\b",
        re.IGNORECASE,
    )

    _DIRECT_ADDRESS_RE = re.compile(
        r"\b(?:" + "|".join(re.escape(p) for p in DIRECT_ADDRESS) + r")\b",
        re.IGNORECASE,
    )

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

        # Prefer the word-boundary regex (accepts "hey adam how are you",
        # "adam?", "ok atom ...") and fall back to direct-address phrases
        # ("are you there") so a confused user doesn't get stranded.
        m = self._WAKE_RE.search(lower)
        if m is not None:
            with self._lock:
                self._last_trigger_time = now
                self._trigger_count += 1
            return m.group(0).strip()

        if self._DIRECT_ADDRESS_RE.search(lower):
            with self._lock:
                self._last_trigger_time = now
                self._trigger_count += 1
            return "direct_address"

        return None

    @classmethod
    def contains_wake(cls, text: str) -> bool:
        """Cheap substring-level check used by the PASSIVE final-gate.

        Unlike :meth:`check`, this does not consume cooldown and does not
        activate state. It's the predicate "does this utterance look like
        it was addressed at ATOM?".
        """
        if not text:
            return False
        lower = text.lower()
        return bool(cls._WAKE_RE.search(lower) or cls._DIRECT_ADDRESS_RE.search(lower))

    @property
    def trigger_count(self) -> int:
        return self._trigger_count


__all__ = ["ListeningMode", "ListeningModeController", "WakeWordFilter"]
