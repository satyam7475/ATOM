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

from core.query_policy import detect_response_language, normalize_query, wants_explicit_depth

logger = logging.getLogger("atom.quick_replies")

_MAX_REPLY_LEN = 500
_MAX_KEY_LEN = 80
_QUICK_REPLY_CLEAN_RE = re.compile(r"[^a-z0-9\s]")

# ── Sprint Ω.8: quick-reply gating ───────────────────────────────────
# Pattern-tier "filler" replies must not fire when the query has *real*
# content beyond the matched greeting / acknowledgement. The bug we're
# fixing (atomCurrentLogs.txt L351-L354): "what's up? can you play
# some music for me?" → "All good here, Boss." — the music question
# never reached the brain.
#
# We only gate FILLER patterns (greetings, "what's up", "thanks",
# acknowledgements). Specific domain replies (joke, "what time is it",
# "who are you" etc.) stay open because their regex is already keyed
# to a substantive concept. The action-verb gate is only consulted
# when ``_PATTERN_REPLIES_FILLER_INDEXES`` says the matching pattern
# is a filler, so "tell me a joke" still matches the joke pattern
# even though "tell" is in the action set.
_FILLER_ACTION_TOKENS: frozenset[str] = frozenset({
    "play", "open", "close", "find", "search", "where",
    "calculate", "compute", "remind", "remember", "schedule", "set",
    "create", "make", "build", "write", "send", "email", "message",
    "translate", "summarize",
    "weather", "news", "headlines", "battery", "cpu",
    "ram", "disk", "memory", "process", "kill", "start", "stop",
    "music", "song", "video", "youtube", "spotify", "browser",
    "bolo", "kholo", "chalao", "khol", "dikhao", "lao",
    "kaha", "kab", "kyun",
})
_FILLER_MAX_WORDS = 6
_COMPARE_SAFARI_ARC_RE = re.compile(r"\bcompare\b.*\bsafari\b.*\barc\b|\barc\b.*\bsafari\b", re.I)
_UNIFIED_MEMORY_RE = re.compile(r"\bunified memory\b", re.I)
_MODE_DIFF_RE = re.compile(
    r"\b(?:optimal|full performance)\b.*\b(?:difference|compare|farak)\b|\b(?:difference|compare|farak)\b.*\b(?:optimal|full performance)\b",
    re.I,
)
_CPU_SPIKE_RE = re.compile(r"\bcpu spike\b|\bwhy\b.*\bcpu\b", re.I)

# ── Tier 1: Pattern-based quick replies ──────────────────────────────
# Each entry: (compiled regex, list of possible responses).
# Checked before config table — handles greetings, farewells, meta-questions,
# acknowledgments, and simple factual queries the intent engine missed.

_PATTERN_REPLIES: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"^(hi|hello|hey|yo|howdy|hola|greetings)\b", re.I), [
        "Hey Boss. What do you need?",
        "Here, Boss. What's up?",
        "Ready, Boss.",
        "I'm here, Boss.",
    ]),
    (re.compile(r"^(namaste|namaskar|kaise ho|kya haal hai)\b", re.I), [
        "Main yahin hoon, Boss. Bolo kya chahiye.",
        "Bilkul theek, Boss. Batao.",
    ]),
    (re.compile(r"^good\s*(morning|afternoon|evening)", re.I), [
        "Good {0}, Boss. Ready when you are.",
        "Good {0}. What do you need, Boss?",
    ]),
    (re.compile(r"(how are you\b.*|how.?s it going|how do you feel|you doing ok|how.?s (?:everything|life)|what.?s up)", re.I), [
        "I'm good, Boss. Ready for you.",
        "All good here, Boss.",
        "Sharp and ready, Boss.",
    ]),
    (re.compile(r"^(thanks?|thank you|thx|cheers|appreciate)", re.I), [
        "Always, Boss.",
        "Anytime, Boss.",
        "I've got you, Boss.",
    ]),
    (re.compile(r"^(bye|goodbye|see you|good\s*night|later|peace out)", re.I), [
        "See you later, Boss. Take care of yourself. I'll be here.",
        "Goodbye, Boss. I'll keep things running while you're away.",
        "Later, Boss! I'll be right here when you need me.",
    ]),
    (re.compile(r"^(ok|okay|alright|got it|understood|cool|nice|great|perfect)", re.I), [
        "Got it, Boss.",
        "Alright, Boss.",
    ]),
    (re.compile(r"(who are you|what are you|what.?s your name|tell me about yourself)", re.I), [
        "I'm ATOM, Boss. Your personal AI buddy.",
        "I'm ATOM, your operating intelligence, Boss.",
    ]),
    (re.compile(r"(tell me (a )?(short )?joke|say something funny)", re.I), [
        "Why do programmers use dark mode? Because light attracts bugs.",
        "I would tell you a UDP joke, Boss, but you might not get it.",
    ]),
    (re.compile(r"(joke\s+batao|ek\s+chota\s+joke\s+batao|mujhe\s+joke\s+batao|short\s+joke(?:\s+tell\s+me)?)", re.I), [
        "Boss, ek short wala: programmer ne dark mode isliye chuna kyunki light bugs ko attract karti hai.",
        "Ek aur, Boss: Wi-Fi down ho jaye to sabko yaad aata hai ki router bhi feelings rakhta hai.",
    ]),
    (re.compile(r"(say something nice|bolo\s+kuch\s+achha)", re.I), [
        "Boss, you're doing better than you think. One clean step at a time.",
        "You're sharper than you give yourself credit for, Boss.",
    ]),
    (re.compile(r"(one productivity tip|productivity tip|ek productivity tip)", re.I), [
        "Boss, pick one task, set 25 minutes, and protect that block completely.",
        "Do the hardest useful thing first, Boss. Momentum gets easier after that.",
    ]),
    (re.compile(r"(hindi|hinglish).*(baat karo|reply do|jawab do)", re.I), [
        "Theek hai, Boss. Ab main Hindi ya Hinglish mein baat karunga.",
    ]),
    (re.compile(r"(what can you do|what.?s your (capabilit|function)|help me)", re.I), [
        "I handle apps, files, system control, research, and planning, Boss.",
        "Anything local on this Mac that fits my tools, Boss.",
    ]),
    (re.compile(r"^(never\s*mind|forget\s*(it|about it)|cancel|nah|nope)", re.I), [
        "No problem, Boss.",
        "Alright, Boss.",
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


def _query_has_filler_action_intent(norm: str) -> bool:
    """True when a query that *also* matched a filler pattern still
    carries a real action verb (so the filler match is misleading).

    Sprint Ω.8 R5: only consulted for filler-tagged patterns; specific
    domain patterns (joke / who-are-you / time / etc.) bypass this.
    """
    if not norm:
        return False
    tokens = re.findall(r"[a-z0-9']+", norm)
    if not tokens:
        return False
    return any(tok in _FILLER_ACTION_TOKENS for tok in tokens)


# Indexes into ``_PATTERN_REPLIES`` whose matches should be gated by
# the filler-action check. The 0th entry is the bare greetings pattern;
# index 3 is the "what's up / how are you" pattern; thanks/bye/ack/yes
# are also filler. Computed lazily so the literal indices stay close
# to the table for review.
def _filler_indexes() -> frozenset[int]:
    out: set[int] = set()
    for i, (pat, _) in enumerate(_PATTERN_REPLIES):
        src = pat.pattern
        if any(
            marker in src
            for marker in (
                "^(hi|hello|hey",
                "^(namaste|namaskar",
                "^good\\s*(morning",
                "how are you",
                "^(thanks?",
                "^(bye|goodbye",
                "^(ok|okay|alright",
                "^(yes|yeah|yep",
                "^(never\\s*mind",
            )
        ):
            out.add(i)
    return frozenset(out)


_PATTERN_REPLIES_FILLER_INDEXES: frozenset[int] | None = None


def _try_pattern_reply(norm: str) -> str | None:
    """Check Tier 1 pattern-based replies. Returns response or None.

    Sprint Ω.8 (Apr 26 2026) R5: a query that combines a greeting with
    a real ask ("hey atom, where's my music?", "what's up? can you
    play something?") used to short-circuit to a chatty filler reply.
    We now identify which patterns are conversational filler (greetings,
    acknowledgements, etc.) and require those matches to also pass a
    word-count cap + action-verb veto. Specific domain patterns
    (joke, time, who-are-you) are unaffected.
    """
    if not norm:
        return None

    global _PATTERN_REPLIES_FILLER_INDEXES
    if _PATTERN_REPLIES_FILLER_INDEXES is None:
        _PATTERN_REPLIES_FILLER_INDEXES = _filler_indexes()

    word_count = len(norm.split())

    for idx, (pattern, responses) in enumerate(_PATTERN_REPLIES):
        m = pattern.search(norm)
        if not m:
            continue
        if idx in _PATTERN_REPLIES_FILLER_INDEXES:
            if word_count > _FILLER_MAX_WORDS:
                continue
            if _query_has_filler_action_intent(norm):
                continue
        resp = random.choice(responses)
        if resp is None:
            return None
        if "{0}" in resp and m.groups():
            resp = resp.format(m.group(1).lower())
        return resp
    return None


def normalize_for_match(text: str) -> str:
    return normalize_query(text)[:300]


def _normalize_quick_key(text: str) -> str:
    clean = _QUICK_REPLY_CLEAN_RE.sub("", normalize_for_match(text))
    return re.sub(r"\s+", " ", clean).strip()


def _try_domain_reply(norm: str, *, explicit_depth: bool = False) -> str | None:
    language = detect_response_language(norm)
    if _UNIFIED_MEMORY_RE.search(norm):
        if language in {"hindi", "hinglish"}:
            if explicit_depth:
                return (
                    "Unified memory ka matlab hai CPU, GPU aur Neural Engine ek hi memory pool share karte hain. "
                    "Isse Apple Silicon par data copy kam hoti hai, latency kam hoti hai, aur power efficiency better rehti hai. "
                    "Isliye MacBook Air par heavy multitasking aur AI workloads zyada smooth feel hote hain."
                )
            return (
                "Unified memory ka matlab hai CPU, GPU aur Neural Engine ek hi memory pool share karte hain. "
                "Isse Apple Silicon par copy kam hoti hai aur speed aur efficiency better hoti hai."
            )
        if explicit_depth:
            return (
                "Unified memory means the CPU, GPU, and Neural Engine share one memory pool instead of copying data between separate pools. "
                "On Apple Silicon that cuts latency, reduces memory duplication, and improves efficiency, which is why MacBook Air systems stay smoother under mixed workloads."
            )
        return (
            "Unified memory means the CPU, GPU, and Neural Engine share one memory pool. "
            "On Apple Silicon that cuts data copying, so performance and efficiency are better."
        )
    if _MODE_DIFF_RE.search(norm):
        if language in {"hindi", "hinglish"}:
            if explicit_depth:
                return (
                    "Optimal mode daily stable buddy mode hai: kam background engines, lower heat, lower RAM pressure, aur better battery behavior. "
                    "Full Performance mode deeper reasoning, zyada background intelligence, aur heavier model usage ke liye hai. "
                    "System headroom kam ho to ATOM wapas optimal par aa jana chahiye taaki Mac stable rahe."
                )
            return (
                "Optimal mode daily stable buddy mode hai: kam background load, lower heat, aur better battery. "
                "Full Performance mode deeper answers aur extra intelligence features ke liye hai, but headroom chahiye."
            )
        if explicit_depth:
            return (
                "Optimal mode is the stable daily profile with less background work, lower heat, and lower RAM pressure. "
                "Full Performance enables deeper reasoning, more background intelligence, and heavier model use. "
                "If thermal or memory headroom drops, ATOM should fall back to optimal to keep the Mac stable."
            )
        return (
            "Optimal mode is the stable daily buddy mode with lower background load, heat, and RAM pressure. "
            "Full Performance enables deeper answers and more background intelligence when the Mac has headroom."
        )
    if _COMPARE_SAFARI_ARC_RE.search(norm) and "coding" in norm and "macbook air" in norm:
        return (
            "Safari is better for battery, heat, and RAM on a MacBook Air. "
            "Arc has a nicer workspace UI, but for long coding sessions I would default to Safari."
        )
    if (_CPU_SPIKE_RE.search(norm) and "atom" in norm) or ("cpu spike" in norm and "beech beech" in norm):
        if language in {"hindi", "hinglish"}:
            return (
                "Short spike tab aata hai jab model warm up hota hai, system scan chalta hai, ya TTS bol raha hota hai. "
                "Agar spike baar baar aaye to optimal mode aur background features check karne chahiye."
            )
        return (
            "Short CPU spikes usually come from model warm-up, system scans, or TTS speaking. "
            "If they keep repeating, optimal mode and background features are the first things to check."
        )
    return None


def try_quick_reply(user_text: str, config: dict | None) -> str | None:
    """Two-tier quick reply: pattern match first, then config table."""
    norm = normalize_for_match(user_text)
    normalized_query = _normalize_quick_key(user_text)
    if not norm:
        return None
    explicit_depth = wants_explicit_depth(norm)

    domain_hit = _try_domain_reply(norm, explicit_depth=explicit_depth)
    if domain_hit:
        logger.debug("Quick reply (domain tier) for: %s", norm[:60])
        return domain_hit

    if explicit_depth:
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
        k = _normalize_quick_key(key)[:_MAX_KEY_LEN]
        if len(k) < 2:
            continue
        variants = {
            normalized_query,
            normalized_query.removeprefix("atom ").strip(),
            normalized_query.removesuffix(" atom").strip(),
            normalized_query.removeprefix("hey ").strip(),
            normalized_query.removeprefix("hello ").strip(),
        }
        if k in variants:
            if len(k) > len(best_key):
                best_key = k
                best_resp = resp.strip()
    if not best_resp:
        return None
    if len(best_resp) > _MAX_REPLY_LEN:
        best_resp = best_resp[:_MAX_REPLY_LEN].rsplit(" ", 1)[0] + "…"
    logger.debug("Quick reply (config tier) key=%s", best_key)
    return best_resp
