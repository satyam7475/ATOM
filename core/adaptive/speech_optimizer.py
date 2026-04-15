"""
ATOM -- Adaptive speech parameter optimizer.

Combines the learned user profile (preferred rate/pause) with the
current perception state (emotion, urgency) to produce final TTS
parameters.  All values are clamped to safe ranges.

Owner: Satyam
"""

from __future__ import annotations


class SpeechOptimizer:
    """Merge user profile with real-time perception into TTS params."""

    __slots__ = ()

    def optimize(
        self,
        perception: dict,
        profile: dict,
    ) -> dict[str, float]:
        rate = profile.get("preferred_rate", 1.0)
        pause = profile.get("preferred_pause", 1.0)

        urgency = perception.get("urgency", "medium")
        emotion = perception.get("emotion", "neutral")

        if urgency == "high":
            rate *= 1.2
            pause *= 0.7
        elif urgency == "low":
            rate *= 0.9
            pause *= 1.2

        if emotion in ("angry", "frustrated"):
            rate *= 1.1
            pause *= 0.8
        elif emotion in ("calm", "sad"):
            rate *= 0.95
            pause *= 1.1

        return {
            "rate_multiplier": round(max(0.85, min(rate, 1.35)), 3),
            "pause_multiplier": round(max(0.5, min(pause, 1.5)), 3),
        }
