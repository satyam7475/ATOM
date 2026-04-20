"""
ATOM -- Productivity intents (Sprint D2).

Quick summaries of the user's immediate obligations:

    * "what's on my plate"
    * "what do I have today"
    * "what's my schedule"
    * "what's coming up today"
    * "what does my day look like"
    * "what's happening today"

Matched patterns emit the ``whats_on_my_plate`` action; the router
dispatches it to :func:`core.proactive.whats_on_plate.generate_plate_summary_sync`
which merges calendar + reminder state into a single spoken line.

Design rule: keep the regexes narrow enough that we never shadow
``memory_recall_intents`` (e.g. "what did I ask yesterday").
"""

from __future__ import annotations

import re

from .base import IntentResult


_PLATE = re.compile(
    r"\bwhat(?:'?s| is|\s+do\s+i\s+have)\b[^?]*?"
    r"(?:on\s+my\s+plate|my\s+(?:schedule|day|calendar|agenda)|coming\s+up|happening|going\s+on)\b",
    re.IGNORECASE,
)

_HAVE_TODAY = re.compile(
    r"\bwhat\s+do\s+i\s+have\s+(?:today|going\s+on|on\s+my\s+plate|planned|"
    r"scheduled|coming\s+up|for\s+today|this\s+morning|this\s+afternoon)\b",
    re.IGNORECASE,
)

_MY_DAY = re.compile(
    r"\b(?:how\s+does\s+my\s+day\s+look|what\s+does\s+my\s+day\s+look|"
    r"tell\s+me\s+about\s+my\s+day|how\s+is\s+my\s+day\s+looking|"
    r"brief\s+me\s+on\s+my\s+day|run\s+me\s+through\s+my\s+day)\b",
    re.IGNORECASE,
)

_SCHEDULE_DIRECT = re.compile(
    r"\b(?:my\s+(?:schedule|agenda|calendar)\s+(?:today|for\s+today))\b",
    re.IGNORECASE,
)


def check(text: str) -> IntentResult | None:
    if not text:
        return None
    if (
        _PLATE.search(text)
        or _HAVE_TODAY.search(text)
        or _MY_DAY.search(text)
        or _SCHEDULE_DIRECT.search(text)
    ):
        return IntentResult(
            "whats_on_my_plate",
            action="whats_on_my_plate",
            action_args={},
        )
    return None


__all__ = ["check"]
