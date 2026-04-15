"""
ATOM -- Adaptive response shaper.

Trims or preserves response text based on the learned verbosity
preference.  Applied to **full** (non-streaming) responses before
they reach TTS.

For streaming responses the verbosity signal is used upstream
via the system instruction (concise mode) rather than post-hoc
truncation.

Owner: Satyam
"""

from __future__ import annotations

import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class ResponseShaper:
    """Shape response length based on learned verbosity."""

    __slots__ = ()

    def shape(self, text: str, profile: dict) -> str:
        if not text:
            return text

        verbosity = profile.get("verbosity", 0.5)
        sentences = _SENTENCE_SPLIT.split(text)

        if len(sentences) <= 2:
            return text

        if verbosity < 0.35:
            return " ".join(sentences[:2])

        if verbosity > 0.7:
            return text

        cap = max(2, int(len(sentences) * 0.6))
        return " ".join(sentences[:cap])
