"""
ATOM -- Intent urgency classifier.

Scores user input by urgency (low / medium / high) using keyword
matching and structural heuristics.  Runs in <0.05ms.

Urgency drives TTS speech rate and inter-sentence pauses via
the SpeechStyleController.
"""

from __future__ import annotations

from dataclasses import dataclass

_HIGH_KEYWORDS = frozenset({
    "now", "urgent", "immediately", "asap", "quick", "fast",
    "hurry", "right away", "quickly", "stop", "cancel", "abort",
    "emergency",
})

_COMMAND_PREFIXES = (
    "open", "close", "stop", "start", "run", "execute", "launch",
    "kill", "play", "pause", "mute", "unmute", "search", "find",
    "show", "hide", "set", "turn", "switch", "toggle",
)


@dataclass(frozen=True, slots=True)
class UrgencyResult:
    """Classified urgency level."""
    level: str
    score: float


class UrgencyClassifier:
    """Weighted urgency scorer: keyword (0.6) + structure (0.3) + brevity (0.1).

    Avoids false positives like "explain now what is quantum physics"
    where a keyword is present but intent is clearly not urgent.
    """

    __slots__ = ()

    def classify(self, text: str) -> UrgencyResult:
        lower = text.lower().strip()
        words = lower.split()
        n_words = len(words)

        keyword_score = 0.0
        for kw in _HIGH_KEYWORDS:
            if kw in lower:
                keyword_score = 1.0
                break

        is_command = 1.0 if (words and words[0] in _COMMAND_PREFIXES) else 0.0
        is_short = 1.0 if n_words <= 4 else (0.5 if n_words <= 7 else 0.0)
        exclaim = 0.3 if text.endswith("!") else 0.0

        score = (
            0.50 * keyword_score
            + 0.30 * is_command
            + 0.10 * is_short
            + 0.10 * exclaim
        )

        if score >= 0.7:
            return UrgencyResult("high", round(score, 3))
        if score >= 0.4:
            return UrgencyResult("medium", round(score, 3))
        return UrgencyResult("low", round(score, 3))
