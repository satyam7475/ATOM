"""
ATOM -- Voice emotion profiles for Edge Neural TTS.

Tuned for en-GB-RyanNeural (British male) to achieve a Vision/JARVIS hybrid:
  - Vision: calm, measured, empathetic warmth, deeper pitch
  - JARVIS: British polish, confident, efficient, precise

Maps contextual emotions to SSML prosody parameters (rate, pitch, volume).

Time-of-day awareness:
  - Night (21:00-05:00): softer volume, slower rate, minimal interruption feel
  - Morning (05:00-12:00): normal baseline
  - Afternoon (12:00-17:00): slightly more energetic
  - Evening (17:00-21:00): calmer, winding down
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class VoiceProfile:
    """SSML prosody parameters for a specific emotion/context."""
    name: str
    rate: str = "+2%"
    pitch: str = "-2Hz"
    volume: str = "-10%"


PROFILES: dict[str, VoiceProfile] = {
    # Vision-like: composed, measured, thoughtful
    "neutral":  VoiceProfile("neutral",  rate="+2%",   pitch="-2Hz",  volume="-10%"),
    # JARVIS warmth: slightly brighter, welcoming
    "friendly": VoiceProfile("friendly", rate="+5%",   pitch="+0Hz",  volume="-8%"),
    # JARVIS alert: crisp, clear, elevated urgency
    "urgent":   VoiceProfile("urgent",   rate="+10%",  pitch="+2Hz",  volume="+0%"),
    # Vision contemplating: deep, soothing, unhurried
    "calm":     VoiceProfile("calm",     rate="-2%",   pitch="-4Hz",  volume="-15%"),
    # JARVIS quick ack: efficient but not robotic
    "ack":      VoiceProfile("ack",      rate="+8%",   pitch="+0Hz",  volume="-8%"),
    # JARVIS system report: professional clarity
    "status":   VoiceProfile("status",   rate="+3%",   pitch="-1Hz",  volume="-10%"),
    # Vision empathetic concern: gentle, slightly deeper
    "error":    VoiceProfile("error",    rate="-1%",   pitch="-3Hz",  volume="-5%"),
    # Vision emotional warmth: tender, genuine care
    "warm":     VoiceProfile("warm",     rate="+0%",   pitch="-2Hz",  volume="-12%"),
    # Autonomous execution: confident, decisive, clear
    "executing": VoiceProfile("executing", rate="+12%", pitch="+1Hz",  volume="+0%"),
}

# ── Time-of-day modifiers ─────────────────────────────────────────────

_TIME_MODIFIERS: dict[str, dict[str, int]] = {
    "night":     {"rate_adj": -5, "volume_adj": -10, "pitch_adj": -2},
    "morning":   {"rate_adj":  0, "volume_adj":   0, "pitch_adj":  0},
    "afternoon": {"rate_adj": +2, "volume_adj":   0, "pitch_adj":  0},
    "evening":   {"rate_adj": -2, "volume_adj":  -5, "pitch_adj": -1},
}

_TIME_SLOTS = {
    range(5, 12): "morning",
    range(12, 17): "afternoon",
    range(17, 21): "evening",
}


def _current_time_slot(hour: int | None = None) -> str:
    if hour is None:
        hour = datetime.now().hour
    for rng, label in _TIME_SLOTS.items():
        if hour in rng:
            return label
    return "night"


def _parse_prosody(value: str) -> int:
    """Extract numeric portion from SSML prosody string like '+5%' or '-2Hz'."""
    stripped = value.replace("%", "").replace("Hz", "").replace("hz", "")
    try:
        return int(stripped)
    except ValueError:
        return 0


def _format_rate(val: int) -> str:
    return f"{val:+d}%"


def _format_pitch(val: int) -> str:
    return f"{val:+d}Hz"


def _format_volume(val: int) -> str:
    return f"{val:+d}%"


def get_profile(name: str) -> VoiceProfile:
    """Return a voice profile by name, defaulting to neutral."""
    return PROFILES.get(name, PROFILES["neutral"])


def get_time_aware_profile(
    emotion: str,
    hour: int | None = None,
) -> VoiceProfile:
    """Return a voice profile adjusted for the current time of day.

    Night: softer, slower, deeper -- minimal disturbance.
    Afternoon: slightly brighter.
    Evening: calmer transition.
    """
    base = get_profile(emotion)
    slot = _current_time_slot(hour)
    mods = _TIME_MODIFIERS.get(slot)
    if not mods:
        return base

    new_rate = _parse_prosody(base.rate) + mods["rate_adj"]
    new_pitch = _parse_prosody(base.pitch) + mods["pitch_adj"]
    new_volume = _parse_prosody(base.volume) + mods["volume_adj"]

    return VoiceProfile(
        name=f"{base.name}_{slot}",
        rate=_format_rate(new_rate),
        pitch=_format_pitch(new_pitch),
        volume=_format_volume(new_volume),
    )


# ── Emotion detection ────────────────────────────────────────────────

_EMOTION_HINTS: list[tuple[str, list[str]]] = [
    ("warm",     ["glad", "happy", "great", "wonderful", "appreciate",
                   "thank", "pleasure", "lovely", "beautiful", "nice to",
                   "take care", "stay safe", "miss you"]),
    ("friendly", ["good morning", "hello", "hi boss", "welcome", "greetings",
                   "hey", "good afternoon", "good evening", "boss"]),
    ("urgent",   ["warning", "critical", "failed", "cannot", "emergency",
                   "immediately", "attention", "danger"]),
    ("error",    ["sorry", "unable", "not found", "don't understand",
                   "didn't work", "error", "oops", "couldn't", "apologize"]),
    ("calm",     ["battery", "cpu", "ram", "disk", "status", "system info",
                   "percent", "usage", "free", "available", "everything is",
                   "all systems"]),
    ("status",   ["running", "active", "online", "operational", "ready",
                   "loaded", "initialized", "connected"]),
    ("executing", ["executing", "auto-executing", "opening for you",
                    "running that", "on it automatically"]),
]


def detect_emotion(text: str) -> str:
    """Auto-detect emotion profile from response text content."""
    lower = text.lower()
    for emotion, keywords in _EMOTION_HINTS:
        if any(kw in lower for kw in keywords):
            return emotion
    return "neutral"
