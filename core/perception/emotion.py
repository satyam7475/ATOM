"""
ATOM -- Lightweight text-based emotion analyzer.

Runs in <0.1ms per call using keyword + punctuation heuristics.
No ML dependencies -- designed for real-time pipeline use.

Feeds into the existing SpeechEnhancer via the ``user_emotion_detected``
bus event so TTS prosody adapts to user mood.
"""

from __future__ import annotations

from dataclasses import dataclass

_POSITIVE = frozenset({
    "good", "great", "awesome", "nice", "love", "thanks", "thank",
    "perfect", "excellent", "wonderful", "cool", "amazing", "happy",
    "brilliant", "fantastic",
})

_NEGATIVE = frozenset({
    "bad", "hate", "annoying", "issue", "problem", "error", "wrong",
    "broken", "fail", "failed", "terrible", "awful", "stupid", "slow",
    "bug", "crash", "stuck", "frustrated", "angry", "useless",
    "not", "doesn't", "isn't", "can't", "won't", "never",
    "sucks", "horrible", "rubbish", "garbage", "pathetic",
})

_URGENT = frozenset({
    "now", "immediately", "fast", "quick", "urgent", "asap", "hurry",
    "right away", "quickly",
})


@dataclass(frozen=True, slots=True)
class EmotionResult:
    """Detected emotional state from user input."""
    label: str
    intensity: float


class EmotionAnalyzer:
    """Keyword + punctuation heuristic emotion classifier."""

    __slots__ = ()

    def analyze(self, text: str) -> EmotionResult:
        lower = text.lower()
        words = set(lower.split())

        score = 0
        score += len(words & _POSITIVE)
        score -= len(words & _NEGATIVE)

        excl_count = text.count("!")
        if excl_count >= 2:
            score += 1
        elif excl_count == 1:
            score += 0.5

        if "??" in text or "???" in text:
            score -= 1

        if text.isupper() and len(text) > 4:
            score -= 0.5

        urgent = bool(words & _URGENT)
        intensity = min(1.0, abs(score) / 2.0)

        if score >= 2:
            return EmotionResult("happy", max(intensity, 0.5))
        if score <= -2:
            return EmotionResult("angry", max(intensity, 0.7))
        if score <= -1:
            return EmotionResult("frustrated", max(intensity, 0.4))
        if urgent:
            return EmotionResult("stressed", 0.6)
        if score >= 1:
            return EmotionResult("calm", 0.3)
        return EmotionResult("neutral", 0.15)
