"""
ATOM Intent Engine -- Media intents (play_youtube, stop_music, volume, mute, unmute).
"""

from __future__ import annotations

import re

from .base import IntentResult, clean_slot

_PLAY_YOUTUBE = re.compile(
    r"\b(play|start|bajao|chalao)\s+(?P<query>.+?)\s+(on\s+)?(youtube|screen)\b|"
    r"\b(play|start|bajao)\s+youtube\b|"
    r"\b(gana|gaana|song|music|sangeet)\s+(play|bajao|chalao|chala|suna|sunao)\s*(karo|kar\s+do|do)?\b|"
    r"\b(play|bajao|chalao|suna)\s+(gana|gaana|song|music|sangeet)\b",
    re.I,
)

_STOP_MUSIC = re.compile(
    r"\b(stop\s+(music|song|audio)|pause\s+(music|song|audio)|stop\s+the\s+music)\b", re.I)

_VOLUME_SET = re.compile(
    r"\b(increase|decrease|set|make|volume|sound|awaaz)\s*(ko\s+|to\s+|at\s+)?(?P<pct>\d{1,3})\s*(percent|%)\s*(pe|par|per|karo|kar\s+do|le\s+aao)?\b",
    re.I,
)

_VOLUME_100 = re.compile(
    r"\b(full\s+volume|max(imum)?\s+volume|volume\s+(full|max)|100\s*percent\s*volume|volume\s+full\s+karo)\b",
    re.I,
)

_VOLUME_50 = re.compile(
    r"\b(half\s+volume|volume\s+50|50\s*percent\s*volume|volume\s+half)\b", re.I)

_MUTE = re.compile(
    r"\b(mute|mute\s+(system|pc|computer|audio|sound|volume)|silent\s+mode|"
    r"awaaz\s+band)\b", re.I)

_UNMUTE = re.compile(
    r"\b(unmute|un\s*mute|unmute\s+(system|pc|computer|audio|sound)|"
    r"awaaz\s+chalu)\b", re.I)


def check(text: str) -> IntentResult | None:
    m = _PLAY_YOUTUBE.search(text)
    if m:
        query = clean_slot(m.groupdict().get("query"))
        if not query:
            t_lower = text.lower()
            for strip_word in ("gana", "gaana", "song", "music", "sangeet",
                               "play", "bajao", "chalao", "chala", "suna",
                               "sunao", "karo", "kar", "do", "start",
                               "on", "youtube", "screen", "my"):
                t_lower = re.sub(r"\b" + strip_word + r"\b", "", t_lower)
            leftover = t_lower.strip()
            query = leftover if leftover else "music"
        return IntentResult("play_youtube", response=f"Playing {query} on YouTube.",
                            action="play_youtube", action_args={"query": query})

    if _STOP_MUSIC.search(text):
        return IntentResult("stop_music", response="Stopping music.",
                            action="stop_music", action_args={})

    if _VOLUME_100.search(text):
        return IntentResult("set_volume", response="Setting full volume.",
                            action="set_volume", action_args={"percent": 100})
    if _VOLUME_50.search(text):
        return IntentResult("set_volume", response="Setting volume to 50 percent.",
                            action="set_volume", action_args={"percent": 50})
    m = _VOLUME_SET.search(text)
    if m:
        try:
            pct = max(0, min(100, int(m.group("pct"))))
        except (TypeError, ValueError):
            return None
        return IntentResult("set_volume", response=f"Setting volume to {pct} percent.",
                            action="set_volume", action_args={"percent": pct})

    if _MUTE.search(text):
        return IntentResult("mute", action="mute", action_args={})
    if _UNMUTE.search(text):
        return IntentResult("unmute", action="unmute", action_args={})
    return None


def quick_match(text: str) -> str | None:
    if _VOLUME_100.search(text) or _VOLUME_50.search(text) or _VOLUME_SET.search(text):
        return "set_volume"
    if _MUTE.search(text):
        return "mute"
    if _UNMUTE.search(text):
        return "unmute"
    if _STOP_MUSIC.search(text):
        return "stop_music"
    return None
