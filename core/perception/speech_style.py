"""
ATOM -- Adaptive speech style controller.

Maps (emotion, urgency) → speech parameters so TTS automatically
shifts between fast command confirmations, slow explanations, and
emotionally adapted responses.

Integrates with the existing ``SpeechEnhancer`` by adjusting its
base rate, and with ``MacOSTTSAsync`` via the ``apply_perception_style``
method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.perception.emotion import EmotionResult
    from core.perception.urgency import UrgencyResult


@dataclass(frozen=True, slots=True)
class SpeechStyle:
    """Target speech parameters for current interaction context."""
    rate_multiplier: float
    pause_multiplier: float


class SpeechStyleController:
    """Decide TTS pacing from emotion + urgency signals."""

    __slots__ = ()

    def decide(self, emotion: EmotionResult, urgency: UrgencyResult) -> SpeechStyle:
        if urgency.level == "high":
            return SpeechStyle(rate_multiplier=1.20, pause_multiplier=0.4)

        if urgency.level == "low" and emotion.label in ("neutral", "calm"):
            return SpeechStyle(rate_multiplier=0.92, pause_multiplier=1.5)

        if emotion.label in ("frustrated", "angry"):
            return SpeechStyle(rate_multiplier=1.10, pause_multiplier=0.6)

        if emotion.label == "stressed":
            return SpeechStyle(rate_multiplier=1.05, pause_multiplier=0.7)

        if emotion.label == "happy":
            return SpeechStyle(rate_multiplier=1.05, pause_multiplier=0.9)

        if emotion.label == "sad":
            return SpeechStyle(rate_multiplier=0.88, pause_multiplier=1.8)

        return SpeechStyle(rate_multiplier=1.0, pause_multiplier=1.0)
