"""ATOM Sprint N7 -- mood-driven voice prosody catalogue.

The :class:`SpeechController` has a dedicated ``mood`` channel that
multiplies into the final TTS rate / pause params. This module is the
*lookup table* the wiring layer consults when it sees a
``mood_changed`` bus event.

Each mood maps to a steady prosody profile. The numbers are
intentionally subtle (within ±20%) -- big swings break the FRIDAY
voice continuity. The intent is "Boss can hear ATOM lean in or step
back" without the voice ever sounding like a different model.

We also expose a *voice preset hint* (e.g. ``"friday"`` / ``"focused"``
/ ``"calm"``) that the existing voice picker in ``voice/tts_macos.py``
can honour when it next reseeds.

Owner: Boss (Satyam).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MoodProsody:
    """Prosody preset for one mood label."""

    mood: str
    rate_multiplier: float = 1.0
    pause_multiplier: float = 1.0
    voice_preset: str = "friday"
    intent: str = ""

    def to_speech_params(self) -> dict[str, float]:
        return {
            "rate_multiplier": float(self.rate_multiplier),
            "pause_multiplier": float(self.pause_multiplier),
        }


# Tuned for the existing _enhancer base rate (~190 wpm). All values
# stay inside the SpeechController guardrails (rate ≥ 0.7 / ≤ 1.4,
# pause ≥ 0.3 / ≤ 2.0).
_PROFILES: dict[str, MoodProsody] = {
    "neutral": MoodProsody(
        "neutral", 1.00, 1.00, "friday",
        "Default FRIDAY voice for unstated mood.",
    ),
    "focused": MoodProsody(
        "focused", 1.05, 0.85, "friday",
        "Slightly faster, tighter pauses -- match Boss's flow state.",
    ),
    "engaged": MoodProsody(
        "engaged", 1.03, 0.95, "friday",
        "Lean-in voice when Boss is actively in dialogue.",
    ),
    "calm": MoodProsody(
        "calm", 0.95, 1.10, "calm",
        "Slow + slightly longer pauses for relaxed/idle moments.",
    ),
    "relaxed": MoodProsody(
        "relaxed", 0.93, 1.10, "calm",
        "Same family as calm, used when Boss is decompressing.",
    ),
    "tired": MoodProsody(
        "tired", 0.90, 1.20, "calm",
        "Softer, slower -- Boss looks fatigued; don't bark commands.",
    ),
    "stressed": MoodProsody(
        "stressed", 0.92, 1.15, "calm",
        "Drop the speed, give Boss more breath room.",
    ),
    "alert": MoodProsody(
        "alert", 1.10, 0.80, "focused",
        "Snap to attention voice -- urgent context detected.",
    ),
    "urgent": MoodProsody(
        "urgent", 1.12, 0.75, "focused",
        "Tight, clipped delivery for critical / time-pressed tasks.",
    ),
    "happy": MoodProsody(
        "happy", 1.05, 0.95, "friday",
        "Slight lift -- Boss is in a good mood, lean into it.",
    ),
    "distracted": MoodProsody(
        "distracted", 0.98, 1.05, "friday",
        "Extra clarity -- Boss is splitting attention.",
    ),
    "idle": MoodProsody(
        "idle", 0.95, 1.10, "calm",
        "Boss isn't engaged; lower energy proactive voice.",
    ),
    "absent": MoodProsody(
        "absent", 1.00, 1.00, "friday",
        "Boss is away -- prosody stays neutral until they return.",
    ),
}


def for_mood(mood: str | None) -> MoodProsody:
    """Return the prosody profile for ``mood`` (defaults to neutral)."""
    if not mood:
        return _PROFILES["neutral"]
    return _PROFILES.get(mood.strip().lower(), _PROFILES["neutral"])


def known_moods() -> list[str]:
    return sorted(_PROFILES.keys())


__all__ = ["MoodProsody", "for_mood", "known_moods"]
