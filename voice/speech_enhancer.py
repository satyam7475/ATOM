"""
ATOM -- Speech Enhancement Layer for JARVIS-Level TTS.

Preprocesses text before speech synthesis to add:
  - Dynamic rate control based on content type and emotion
  - Micro-pause insertion at natural breath points
  - Emotion-to-prosody mapping for expressive speech

Integrates with MacOSTTSAsync._speak_internal to make TTS "alive"
rather than robotic.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("atom.speech_enhancer")

_RE_QUESTION = re.compile(r"\?\s*$")
_RE_EXCLAMATION = re.compile(r"!\s*$")
_RE_ELLIPSIS = re.compile(r"\.{2,3}")

_PAUSE_MARKERS = re.compile(r"(,\s|;\s|—\s|--\s|…\s|\.\.\.\s)")

_ERROR_WORDS = frozenset({
    "error", "fail", "sorry", "unfortunately", "couldn't", "can't",
    "unable", "issue", "problem", "wrong", "broke", "crash",
})
_URGENT_WORDS = frozenset({
    "critical", "urgent", "warning", "alert", "danger", "immediately",
    "now", "attention", "battery", "emergency",
})
_SUCCESS_WORDS = frozenset({
    "done", "complete", "success", "ready", "finished", "perfect",
    "great", "opened", "started", "saved", "created",
})

_EMOTION_RATE_MAP = {
    "neutral": 0,
    "happy": 8,
    "excited": 12,
    "frustrated": -10,
    "stressed": -8,
    "tired": -15,
    "calm": -5,
    "curious": 5,
    "empathetic": -8,
}


@dataclass
class EnhancedSpeech:
    """Preprocessed speech parameters."""
    text: str
    rate: int = 165
    pause_points: list[int] = field(default_factory=list)
    emotion: str = "neutral"

    @property
    def say_silence_text(self) -> str:
        """Insert macOS `say` silence markers at pause points.

        `[[slnc N]]` inserts N milliseconds of silence.
        """
        if not self.pause_points:
            return self.text
        result = list(self.text)
        for offset in reversed(self.pause_points):
            if 0 <= offset < len(result):
                result.insert(offset + 1, " [[slnc 120]] ")
        return "".join(result)


class SpeechEnhancer:
    """Preprocesses text for expressive, JARVIS-quality speech."""

    def __init__(self, base_rate: int = 165) -> None:
        self._base_rate = base_rate
        self._pause_multiplier: float = 1.0

    def enhance(
        self,
        text: str,
        emotion: str = "neutral",
    ) -> EnhancedSpeech:
        """Analyze text and return enhanced speech parameters."""
        rate = self._compute_rate(text, emotion)
        pauses = self._find_pause_points(text)
        return EnhancedSpeech(
            text=text,
            rate=rate,
            pause_points=pauses,
            emotion=emotion,
        )

    def _compute_rate(self, text: str, emotion: str) -> int:
        """Dynamic rate based on content type and emotional context."""
        rate = self._base_rate
        lower = text.lower()
        word_count = len(lower.split())

        if word_count <= 6:
            rate += 15
        elif word_count >= 40:
            rate -= 10

        if _RE_QUESTION.search(text):
            rate -= 12
        elif _RE_EXCLAMATION.search(text):
            rate += 8

        words = set(lower.split())
        if words & _ERROR_WORDS:
            rate -= 15
        elif words & _URGENT_WORDS:
            rate += 20
        elif words & _SUCCESS_WORDS:
            rate += 8

        rate += _EMOTION_RATE_MAP.get(emotion, 0)

        return max(155, min(240, rate))

    def _find_pause_points(self, text: str) -> list[int]:
        """Find positions where micro-pauses improve naturalism."""
        pauses: list[int] = []
        for m in _PAUSE_MARKERS.finditer(text):
            pauses.append(m.end() - 1)
        for m in _RE_ELLIPSIS.finditer(text):
            pauses.append(m.end() - 1)
        return pauses

    def compute_inter_sentence_pause(
        self,
        sentence: str,
        emotion: str = "neutral",
    ) -> float:
        """Seconds to pause between sentences for natural rhythm."""
        if _RE_ELLIPSIS.search(sentence):
            base = 0.22
        elif emotion in ("tired", "calm", "empathetic"):
            base = 0.18
        elif emotion in ("excited", "happy"):
            base = 0.08
        elif emotion == "curious":
            base = 0.10
        elif _RE_QUESTION.search(sentence):
            base = 0.14
        else:
            base = 0.12
        return base * self._pause_multiplier


__all__ = ["SpeechEnhancer", "EnhancedSpeech"]
