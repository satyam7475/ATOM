"""
ATOM v15 -- Two-tier quick replies (no LLM).

Tier 1: Pattern-based replies for common conversational phrases.
Tier 2: Config-driven substring match from settings.json.

Together these skip the LLM for ~80% of casual queries, leaving it
for genuine open questions only.
"""

from __future__ import annotations

import logging
import random
import re
from typing import Any

logger = logging.getLogger("atom.quick_replies")

_MAX_REPLY_LEN = 500
_MAX_KEY_LEN = 80

# ── Tier 1: Pattern-based quick replies ──────────────────────────────
# Each entry: (compiled regex, list of possible responses).
# Checked before config table — handles greetings, farewells, meta-questions,
# acknowledgments, and simple factual queries the intent engine missed.

_PATTERN_REPLIES: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"^(hi|hello|hey|yo|howdy|hola|greetings)\b", re.I), [
        "Hey Boss, what do you need?",
        "Hello Boss. Ready when you are.",
        "Hey! How can I help?",
    ]),
    (re.compile(r"^good\s*(morning|afternoon|evening)", re.I), [
        "Good {0}, Boss. Systems are ready.",
        "Good {0}! What can I do for you?",
    ]),
    (re.compile(r"(how are you\b.*|how.?s it going|how do you feel|you doing ok|how.?s (?:everything|life)|what.?s up)", re.I), [
        "Running smooth, Boss. All systems nominal.",
        "I'm good, Boss. Ready and waiting.",
        "All good here. What do you need?",
    ]),
    (re.compile(r"^(thanks?|thank you|thx|cheers|appreciate)", re.I), [
        "Anytime, Boss.",
        "Happy to help.",
        "You got it, Boss.",
    ]),
    (re.compile(r"^(bye|goodbye|see you|good\s*night|later|peace out)", re.I), [
        "See you later, Boss. I'll keep things running.",
        "Goodbye, Boss. I'll be here when you need me.",
    ]),
    (re.compile(r"^(ok|okay|alright|got it|understood|cool|nice|great|perfect)", re.I), [
        "Got it, Boss.",
        "Alright. Let me know if you need anything else.",
    ]),
    (re.compile(r"(who are you|what are you|what.?s your name|tell me about yourself)", re.I), [
        "I'm ATOM, your personal AI operating system. Version 15, fully offline.",
    ]),
    (re.compile(r"(what can you do|what.?s your (capabilit|function)|help me)", re.I), [
        "I can open apps, check system stats, manage files, set goals, search the web, and answer questions — all offline, Boss.",
    ]),
    (re.compile(r"^(never\s*mind|forget\s*(it|about it)|cancel|nah|nope)", re.I), [
        "No problem, Boss. Standing by.",
        "Alright, cancelled.",
    ]),
    (re.compile(r"(you.?re (great|awesome|amazing|the best)|good job|well done|nice work)", re.I), [
        "Thanks Boss, I appreciate that.",
        "Glad I could help!",
    ]),
    (re.compile(r"^(yes|yeah|yep|yup|sure|absolutely|definitely|of course)$", re.I), [
        "Got it. What's next?",
    ]),
    (re.compile(r"(what.?s the date|today.?s date|what day is)", re.I), [
        None,  # sentinel — let intent engine handle via info_intents
    ]),
]


def _try_pattern_reply(norm: str) -> str | None:
    """Check Tier 1 pattern-based replies. Returns response or None."""
    for pattern, responses in _PATTERN_REPLIES:
        m = pattern.search(norm)
        if m:
            resp = random.choice(responses)
            if resp is None:
                return None
            if "{0}" in resp and m.groups():
                resp = resp.format(m.group(1).lower())
            return resp
    return None


def normalize_for_match(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t[:300]


def try_quick_reply(user_text: str, config: dict | None) -> str | None:
    """Two-tier quick reply: pattern match first, then config table."""
    norm = normalize_for_match(user_text)
    if not norm:
        return None

    # Tier 1: pattern-based
    pattern_hit = _try_pattern_reply(norm)
    if pattern_hit:
        logger.debug("Quick reply (pattern tier) for: %s", norm[:40])
        return pattern_hit

    # Tier 2: config-driven substring match
    cfg = config or {}
    ab = cfg.get("assistant_brain", {})
    if not ab.get("quick_replies_enabled", True):
        return None
    table = ab.get("quick_replies")
    if not isinstance(table, dict) or not table:
        return None

    best_key = ""
    best_resp = ""
    for key, resp in table.items():
        if not isinstance(key, str) or not isinstance(resp, str):
            continue
        k = key.strip().lower()[:_MAX_KEY_LEN]
        if len(k) < 2:
            continue
        if k in norm or norm == k:
            if len(k) > len(best_key):
                best_key = k
                best_resp = resp.strip()
    if not best_resp:
        return None
    if len(best_resp) > _MAX_REPLY_LEN:
        best_resp = best_resp[:_MAX_REPLY_LEN].rsplit(" ", 1)[0] + "…"
    logger.debug("Quick reply (config tier) key=%s", best_key)
    return best_resp
