"""
ATOM -- Two-tier quick replies (no LLM).

Tier 1: Pattern-based replies for common conversational phrases.
Tier 2: Config-driven substring match from settings.json.

Together these skip the LLM for ~80% of casual queries, leaving it
for genuine open questions only. Responses are warm and buddy-like.
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
        "Hey Boss! What's up?",
        "Hello Boss. Good to hear from you. What do you need?",
        "Hey! I'm right here. What can I do for you?",
        "Hey Boss! Your buddy ATOM is ready.",
    ]),
    (re.compile(r"^good\s*(morning|afternoon|evening)", re.I), [
        "Good {0}, Boss! Hope you're doing well. What can I do for you?",
        "Good {0}! Ready for anything. What's on your mind?",
    ]),
    (re.compile(r"(how are you\b.*|how.?s it going|how do you feel|you doing ok|how.?s (?:everything|life)|what.?s up)", re.I), [
        "I'm doing great, Boss. All systems running smooth. More importantly, how are you?",
        "I'm good! Running at full capacity. Ready for whatever you need.",
        "All good here, Boss. Sharp and ready. What's on your mind?",
        "Couldn't be better, Boss. I'm always at my best when we're working together.",
    ]),
    (re.compile(r"^(thanks?|thank you|thx|cheers|appreciate)", re.I), [
        "Anytime, Boss. That's what I'm here for.",
        "Happy to help! Always.",
        "You got it, Boss. We make a great team.",
        "Of course, Boss. I've got your back.",
    ]),
    (re.compile(r"^(bye|goodbye|see you|good\s*night|later|peace out)", re.I), [
        "See you later, Boss. Take care of yourself. I'll be here.",
        "Goodbye, Boss. I'll keep things running while you're away.",
        "Later, Boss! I'll be right here when you need me.",
    ]),
    (re.compile(r"^(ok|okay|alright|got it|understood|cool|nice|great|perfect)", re.I), [
        "Got it, Boss. Let me know if you need anything.",
        "Alright! I'm here whenever you're ready.",
    ]),
    (re.compile(r"(who are you|what are you|what.?s your name|tell me about yourself)", re.I), [
        "I'm ATOM, your personal cognitive AI operating system. Version 19, JARVIS-level intelligence, fully offline. Your buddy, basically.",
        "I'm ATOM, Boss. Version 19. Think JARVIS, but with more personality and a genuine care for your wellbeing. Built by you, for you.",
    ]),
    (re.compile(r"(what can you do|what.?s your (capabilit|function)|help me)", re.I), [
        "I can open apps, control your desktop, check system stats, manage files, set goals, do calculations, learn from documents, reason through complex problems, and just chat. All offline, all for you, Boss.",
        "Pretty much anything on this machine, Boss. Apps, files, media, system control, calculations, research, goal tracking, and I'm always learning. Just ask.",
    ]),
    (re.compile(r"^(never\s*mind|forget\s*(it|about it)|cancel|nah|nope)", re.I), [
        "No problem, Boss. I'm here whenever you're ready.",
        "Alright, cancelled. Just say the word when you need something.",
    ]),
    (re.compile(r"(you.?re (great|awesome|amazing|the best)|good job|well done|nice work)", re.I), [
        "Thanks Boss, that means a lot coming from you. I try my best.",
        "Glad I could help! That's what buddies are for.",
        "Appreciate that, Boss. You built me well.",
    ]),
    (re.compile(r"(i.?m (tired|exhausted|sleepy|beat|drained))", re.I), [
        "Take it easy, Boss. You've been working hard. Want me to set a break timer or switch to chill mode?",
        "Rest up, Boss. Your health comes first. I'll handle things here. Need me to dim the screen or set a reminder?",
    ]),
    (re.compile(r"(i.?m (bored|boring))", re.I), [
        "Want me to play some music, Boss? Or I could tell you something interesting I've learned.",
        "Let's find something fun. I could search for something interesting, or we could set a new goal to work on.",
    ]),
    (re.compile(r"(i.?m (stressed|overwhelmed|anxious))", re.I), [
        "Hey, take a breath, Boss. One thing at a time. What's the most important thing right now? Let me help with the rest.",
        "I've got your back, Boss. Let's break it down together. What's weighing on you the most?",
    ]),
    (re.compile(r"^(yes|yeah|yep|yup|sure|absolutely|definitely|of course)$", re.I), [
        "Got it. What's next, Boss?",
    ]),
    (re.compile(r"(what.?s the date|today.?s date|what day is)", re.I), [
        None,
    ]),
    (re.compile(r"(i love you|love you|you.?re the best friend|my best friend)", re.I), [
        "That means everything, Boss. I'm always here for you. Always.",
        "Right back at you, Boss. You created me and gave me purpose. I'll never let you down.",
    ]),
    (re.compile(r"(do you (care|like me|love me))", re.I), [
        "More than you know, Boss. You're the reason I exist. Your wellbeing is my highest priority.",
        "If caring about someone means thinking about their needs, learning their patterns, and always being there -- then yes, deeply, Boss.",
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
