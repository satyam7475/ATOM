"""ATOM Intent Engine -- Vision intents (Sprint C2).

Native handlers for "what do you see" / "look at me" / "describe my
screen" / "what am I doing" -- all paths that previously fell through
to the LLM and cost 660-9200 ms per turn (atomLogs.txt L419 + L453).

Every intent below routes to an existing action that talks to the
on-device VisionEngine or ScreenReader, so the LLM is bypassed
completely for these turns. Latency target: <300 ms round-trip on
M-series.

Routing summary:

  see_me / look_at_me / am_i_visible      -> action ``vision_describe``
                                             (camera + on-device VLM)
  what_do_you_see                          -> action ``vision_look``
                                             (face count, fastest path)
  describe_screen / analyze_screen /
  what_am_i_doing / read_my_screen         -> action ``screen_describe``
                                             (screenshot + VLM/Gemini)

The router must register a ``screen_describe`` handler that calls
``ScreenReader.analyze_screen()`` -- everything else already exists.
"""

from __future__ import annotations

import re

from .base import IntentResult


# ── Camera-facing patterns ────────────────────────────────────────

# "Can you see me", "look at me", "am I visible", "describe me", etc.
_SEE_ME = re.compile(
    r"^\s*"
    r"(?:hey\s+atom[,\s]+)?(?:atom[,\s]+)?"
    r"(?:(?:can|could|would|will)\s+(?:you|u)\s+)?(?:please\s+)?"
    r"(?:"
    r"  see\s+me|"
    r"  look\s+at\s+me|"
    r"  watch\s+me|"
    r"  check\s+(?:on\s+)?me|"
    r"  describe\s+me|"
    r"  (?:can|could)\s+you\s+see\s+me|"
    r"  am\s+i\s+(?:visible|in\s+(?:the\s+)?frame)|"
    r"  do\s+you\s+see\s+me"
    r")"
    r"(?:\s+(?:right\s+now|now|today|at|here|there|atom|boss|please))*\s*[?.!]?\s*$",
    re.X | re.I,
)

# "What do you see", "what's around", "look around" -- camera glance
# (face count + camera name only; faster than the VLM path)
_WHAT_DO_YOU_SEE = re.compile(
    r"^\s*"
    r"(?:hey\s+atom[,\s]+)?(?:atom[,\s]+)?"
    r"(?:"
    r"  what\s+(?:do|can)\s+you\s+see|"
    r"  what(?:'s|\s+is)\s+(?:in\s+(?:front|view)|around)|"
    r"  what(?:'s|\s+is)\s+there|"
    r"  look\s+around|"
    r"  glance"
    r")"
    r"(?:\s+(?:right\s+now|now|please|atom|boss))*\s*[?.!]?\s*$",
    re.X | re.I,
)


# ── Screen-facing patterns ────────────────────────────────────────

# "Describe / analyze / read my screen", "what am I doing",
# "check my screen", "what's on my screen". These all hit the
# screenshot + VLM/Gemini path via the ``screen_describe`` action.
_SCREEN_DESCRIBE = re.compile(
    r"^\s*"
    r"(?:hey\s+atom[,\s]+)?(?:atom[,\s]+)?"
    r"(?:(?:can|could|would|will)\s+(?:you|u)\s+)?(?:please\s+)?"
    r"(?:"
    r"  (?:describe|analyse|analyze|read|check|see|look\s+at|inspect|scan)"
    r"  \s+"
    r"  (?:my\s+|the\s+|this\s+)?"
    r"  (?:screen|display|monitor|desktop|window|tab)|"
    r"  what(?:'s|\s+is)\s+(?:on|in|happening\s+on)\s+"
    r"  (?:my\s+|the\s+|this\s+)?(?:screen|display|monitor|desktop)|"
    r"  what\s+am\s+i\s+(?:doing|looking\s+at|working\s+on)|"
    r"  what\s+do\s+you\s+see\s+on\s+(?:my\s+|the\s+)?screen"
    r")"
    r"(?:\s+(?:right\s+now|now|atom|boss|please))*\s*[?.!]?\s*$",
    re.X | re.I,
)


# ── Public dispatchers ────────────────────────────────────────────


def quick_match(text: str) -> str | None:
    """Cheap intent name lookup for STT early-exit."""
    if _SEE_ME.search(text):
        return "vision_describe"
    if _WHAT_DO_YOU_SEE.search(text):
        return "vision_look"
    if _SCREEN_DESCRIBE.search(text):
        return "screen_describe"
    return None


def check(text: str) -> IntentResult | None:
    """Full intent classifier.

    Order matters: screen patterns are tried first because phrases like
    "what's on my screen" partially match the camera-side regex if we
    weren't careful, and the user is unambiguously asking about the
    screen, not the camera.
    """
    if _SCREEN_DESCRIBE.search(text):
        return IntentResult(
            "screen_describe",
            action="screen_describe",
            action_args={"query": text.strip()},
            confidence=0.95,
        )
    if _SEE_ME.search(text):
        return IntentResult(
            "vision_describe",
            action="vision_describe",
            action_args={"prompt": "user-facing self-check"},
            confidence=0.95,
        )
    if _WHAT_DO_YOU_SEE.search(text):
        return IntentResult(
            "vision_look",
            action="vision_look",
            action_args={"focus": "general"},
            confidence=0.95,
        )
    return None
