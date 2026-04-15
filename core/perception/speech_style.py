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
    """Decide TTS pacing from emotion + urgency + adaptive feedback."""

    __slots__ = ()

    def decide(
        self,
        emotion: EmotionResult,
        urgency: UrgencyResult,
        *,
        rate_boost: float = 0.0,
    ) -> SpeechStyle:
        if urgency.level == "high":
            base = SpeechStyle(rate_multiplier=1.20, pause_multiplier=0.4)
        elif urgency.level == "low" and emotion.label in ("neutral", "calm"):
            base = SpeechStyle(rate_multiplier=0.92, pause_multiplier=1.5)
        elif emotion.label in ("frustrated", "angry"):
            base = SpeechStyle(rate_multiplier=1.10, pause_multiplier=0.6)
        elif emotion.label == "stressed":
            base = SpeechStyle(rate_multiplier=1.05, pause_multiplier=0.7)
        elif emotion.label == "happy":
            base = SpeechStyle(rate_multiplier=1.05, pause_multiplier=0.9)
        elif emotion.label == "sad":
            base = SpeechStyle(rate_multiplier=0.88, pause_multiplier=1.8)
        else:
            base = SpeechStyle(rate_multiplier=1.0, pause_multiplier=1.0)

        if rate_boost:
            return SpeechStyle(
                rate_multiplier=round(base.rate_multiplier + rate_boost, 3),
                pause_multiplier=base.pause_multiplier,
            )
        return base
