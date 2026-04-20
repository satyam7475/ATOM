"""
ATOM Intent Engine — Memory / timeline recall (Sprint A4).

Detects queries like:

  * "what did I ask you yesterday about the invoice?"
  * "did I say anything earlier about the car?"
  * "remind me what we talked about this morning"
  * "remember when I asked about the tax form?"

and answers directly from ATOM's timeline memory without any LLM roundtrip.
The IntentEngine chain wires this before the generic conversational LLM so
ATOM can answer memory questions with a pre-built response and the TTS
layer can speak it immediately.

TimelineMemory is injected at boot via :func:`set_timeline`. When no
timeline is wired the handler becomes a no-op (returns ``None``).
"""

from __future__ import annotations

import datetime
import logging
import re
import time
from typing import Any

from .base import IntentResult

logger = logging.getLogger("atom.intent.memory_recall")

_timeline: Any | None = None


def set_timeline(instance: Any) -> None:
    """Wire the singleton :class:`TimelineMemory` for intent lookups."""
    global _timeline
    _timeline = instance


# ── Trigger patterns ─────────────────────────────────────────────

# Primary: "what did I ask ... about X"
_ASK_ABOUT = re.compile(
    r"\bwhat\s+did\s+(i|we)\s+(ask|say|talk|discuss|mention|tell\s+you)\b.*?\babout\s+(?P<topic>.+)",
    re.I,
)

# Also: "did I ask you about X <when>"
_DID_I_ASK_ABOUT = re.compile(
    r"\b(did\s+(i|we)|have\s+(i|we))\s+(ask|say|talk|discuss|mention|tell\s+you)\b.*?\babout\s+(?P<topic>.+)",
    re.I,
)

# "remember when I asked about X" / "remember when we discussed X"
_REMEMBER_WHEN = re.compile(
    r"\b(remember|recall)\s+(when|what)\s+(i|we)\s+(ask(ed)?|said|talk(ed)?|mention(ed)?|discuss(ed)?)\b.*?\babout\s+(?P<topic>.+)",
    re.I,
)

# "what did I ask <time scope>" (topic-less — any query in the window)
_WHAT_DID_I_ASK_WHEN = re.compile(
    r"\bwhat\s+did\s+(i|we)\s+(ask|say|talk|discuss|mention|tell\s+you)\b",
    re.I,
)

# "remind me what we talked about today" — no "about X" required.
_REMIND_ME = re.compile(
    r"\b(remind\s+me\s+)?what\s+(did\s+)?(i|we)\s+(talk(ed)?|discuss(ed)?|say|said)\b",
    re.I,
)

# "did I mention <topic>", "did I bring up X", "did I talk about X" — any form
# where "about" is optional so short utterances still land.
_MENTION_TOPIC = re.compile(
    r"\bdid\s+(i|we)\s+(mention|bring\s+up|talk\s+about|discuss|ask|say)\s+(?P<topic>.+)",
    re.I,
)

# Temporal scope tokens we understand without NLP.
_SCOPE_TOKENS = (
    ("last week", 7 * 86400.0, 1 * 86400.0),
    ("last month", 30 * 86400.0, 1 * 86400.0),
    ("yesterday", 48 * 3600.0, 12 * 3600.0),
    ("this morning", 18 * 3600.0, 0.0),
    ("this afternoon", 12 * 3600.0, 0.0),
    ("this evening", 8 * 3600.0, 0.0),
    ("today", 24 * 3600.0, 0.0),
    ("earlier today", 24 * 3600.0, 0.0),
    ("earlier", 6 * 3600.0, 0.0),
    ("a while ago", 12 * 3600.0, 0.0),
    ("recently", 6 * 3600.0, 0.0),
    ("just now", 600.0, 0.0),
)


def _parse_scope(text: str) -> tuple[float | None, float | None, str]:
    """Return (since_ts, until_ts, label) extracted from ``text``.

    The label is a short human-readable phrase used by the response
    formatter. If no temporal scope is detected we default to the last
    48 hours which covers the most natural "did I ask about X" queries.
    """
    low = text.lower()
    now = time.time()

    for token, lookback_s, _ in _SCOPE_TOKENS:
        if token in low:
            if token in {"yesterday"}:
                try:
                    today_start = datetime.datetime.fromtimestamp(now).replace(
                        hour=0, minute=0, second=0, microsecond=0,
                    ).timestamp()
                    return (today_start - 86400.0, today_start, token)
                except Exception:
                    pass
            since = now - float(lookback_s)
            return (since, None, token)

    return (now - 48 * 3600.0, None, "")


def _extract_topic(text: str) -> str:
    for pat in (_REMEMBER_WHEN, _ASK_ABOUT, _DID_I_ASK_ABOUT, _MENTION_TOPIC):
        m = pat.search(text)
        if m:
            raw = (m.group("topic") or "").strip()
            for tok, _, _ in _SCOPE_TOKENS:
                idx = raw.lower().find(tok)
                if idx >= 0:
                    raw = raw[:idx]
            raw = re.sub(r"^(the|a|an|that|about)\s+", "", raw, flags=re.I)
            raw = re.sub(r"[?!.,]+\s*$", "", raw).strip()
            return raw[:80]
    return ""


def check(text: str) -> IntentResult | None:
    if not text or _timeline is None:
        return None
    t = text.strip()
    if len(t) < 8:
        return None

    has_match = bool(
        _REMEMBER_WHEN.search(t)
        or _ASK_ABOUT.search(t)
        or _DID_I_ASK_ABOUT.search(t)
        or _WHAT_DID_I_ASK_WHEN.search(t)
        or _REMIND_ME.search(t)
        or _MENTION_TOPIC.search(t)
    )
    if not has_match:
        return None

    topic = _extract_topic(t)
    since_ts, until_ts, scope_label = _parse_scope(t)

    try:
        scoped_hits = _timeline.search_user_queries(
            topic, since_ts=since_ts, until_ts=until_ts, limit=50,
        )
        widened = False
        if not scoped_hits and topic:
            # Try the full retention window — user often says "yesterday"
            # when they actually mean "some time in the last few days".
            scoped_hits = _timeline.search_user_queries(
                topic, since_ts=None, until_ts=None, limit=50,
            )
            widened = bool(scoped_hits)

        if not scoped_hits:
            kw_txt = f" about \"{topic}\"" if topic else ""
            scope_txt = f" {scope_label}" if scope_label else ""
            response = f"I don't see any questions{scope_txt}{kw_txt} in my memory."
        else:
            latest = scoped_hits[-1]
            n = len(scoped_hits)
            latest_text = (latest.data.get("text") or "").strip()
            if len(latest_text) > 140:
                latest_text = latest_text[:137] + "..."
            from core.memory.timeline_memory import (
                _count_phrase as _count, _relative_time as _rel,
            )
            count_word = _count(n)
            rel = _rel(latest.timestamp)
            if topic:
                lead = f"You asked me about \"{topic}\" {count_word}"
                if scope_label and not widened:
                    lead += f" {scope_label}"
                elif widened:
                    lead += " (not specifically in that window)"
            else:
                lead = f"You've asked me {count_word}"
                if scope_label:
                    lead += f" {scope_label}"
            detail = f" Most recent was {rel}: \"{latest_text}\"." if latest_text else "."
            response = f"{lead}.{detail}"
    except Exception:
        logger.debug("timeline recall failed", exc_info=True)
        return None

    logger.info(
        "memory_recall_intent: topic=%r scope=%r hits=%d",
        topic[:40], scope_label or "(default)", len(scoped_hits),
    )
    return IntentResult("memory_recall", response=response)


__all__ = ["check", "set_timeline"]
