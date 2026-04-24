"""
ATOM Intent Engine -- Focus / Do-Not-Disturb intents (Phase F3).

Resolved actions::

    focus_on    -> turn on macOS DND
    focus_off   -> turn off macOS DND
    focus_state -> ask "is focus on?" (status query)

The pattern set is intentionally generous on synonyms ("deep work
mode", "do not disturb", "focus mode for 30 minutes", etc.) because
this is a marquee feature for a Jarvis-grade assistant. ``check()``
runs *before* ``meta_intents``-style "go silent" handlers so the OS
focus state -- not just ATOM's listening state -- toggles.
"""

from __future__ import annotations

import re

from .base import IntentResult


_FOCUS_ON = re.compile(
    r"\b("
    r"focus(?:\s+mode)?\s+on|"
    r"turn\s+on\s+(?:focus|do\s+not\s+disturb|dnd)|"
    r"enable\s+(?:focus|do\s+not\s+disturb|dnd)|"
    r"start\s+(?:focus(?:\s+mode)?|deep\s+work(?:\s+mode)?)|"
    r"do\s+not\s+disturb(?:\s+on)?|"
    r"deep\s+work(?:\s+mode)?(?:\s+on)?|"
    r"i\s+(?:am\s+|'m\s+)?(?:going\s+)?(?:into\s+)?deep\s+work|"
    r"silence\s+(?:my\s+)?notifications?|"
    r"mute\s+notifications?|"
    r"hold\s+(?:my\s+)?notifications?"
    r")\b",
    re.I,
)

_FOCUS_OFF = re.compile(
    r"\b("
    r"focus(?:\s+mode)?\s+off|"
    r"turn\s+off\s+(?:focus|do\s+not\s+disturb|dnd)|"
    r"disable\s+(?:focus|do\s+not\s+disturb|dnd)|"
    r"end\s+(?:focus(?:\s+mode)?|deep\s+work(?:\s+mode)?)|"
    r"exit\s+(?:focus|deep\s+work|dnd)|"
    r"do\s+not\s+disturb\s+off|"
    r"unmute\s+notifications?|"
    r"resume\s+notifications?|"
    r"i'?m\s+(?:done|out)(?:\s+with\s+(?:focus|deep\s+work))?"
    r")\b",
    re.I,
)

_FOCUS_STATUS = re.compile(
    r"\b("
    r"is\s+(?:focus|do\s+not\s+disturb|dnd)\s+on|"
    r"is\s+focus(?:\s+mode)?\s+(?:on|enabled|active)|"
    r"am\s+i\s+(?:in\s+)?focus(?:\s+mode)?|"
    r"focus\s+status|"
    r"what(?:'s|\s+is)\s+(?:my\s+)?focus(?:\s+state)?"
    r")\b",
    re.I,
)

# Capture "for N minutes" / "for an hour" affixed to enable phrases.
_FOCUS_DURATION = re.compile(
    r"\bfor\s+(?:(?P<num>\d+)\s*(?P<unit>min|mins|minutes|hour|hours|hr|hrs)|"
    r"an?\s+(?P<unit_only>hour|minute))\b",
    re.I,
)


def _extract_duration_minutes(text: str) -> int | None:
    m = _FOCUS_DURATION.search(text)
    if not m:
        return None
    if m.group("unit_only"):
        return 60 if "hour" in m.group("unit_only").lower() else 1
    try:
        n = int(m.group("num"))
    except (TypeError, ValueError):
        return None
    unit = (m.group("unit") or "").lower()
    if unit.startswith("hr") or unit.startswith("hour"):
        return n * 60
    return n


def check(text: str) -> IntentResult | None:
    if _FOCUS_STATUS.search(text):
        return IntentResult("focus_state", action="focus_state",
                            action_args={})
    if _FOCUS_OFF.search(text):
        return IntentResult("focus_off", action="focus_off",
                            action_args={})
    if _FOCUS_ON.search(text):
        args: dict = {}
        minutes = _extract_duration_minutes(text)
        if minutes:
            args["duration_minutes"] = minutes
        return IntentResult("focus_on", action="focus_on",
                            action_args=args)
    return None


def quick_match(text: str) -> str | None:
    if _FOCUS_STATUS.search(text):
        return "focus_state"
    if _FOCUS_OFF.search(text):
        return "focus_off"
    if _FOCUS_ON.search(text):
        return "focus_on"
    return None
