"""
ATOM -- Speech detection helpers.

Text corrections and noise word filtering for STT post-processing.
"""

from __future__ import annotations

import re

MAX_IDLE_LISTEN_S = 8.0

# ── Post-STT text corrections ──────────────────────────────────────────

_TEXT_CORRECTIONS: list[tuple[str, str]] = [
    ("one morning at one", "good morning atom"),
    ("one morning atom", "good morning atom"),
    ("one morning atome", "good morning atom"),
    ("good morning item", "good morning atom"),
    ("good morning at one", "good morning atom"),
    ("good morning atome", "good morning atom"),
    ("good money", "good morning"),
    ("hello welcome", "hello atom"),
    ("hey adam", "hey atom"),
    ("hey a tom", "hey atom"),
    ("hey at um", "hey atom"),
    ("hey at on", "hey atom"),
    ("hey at one", "hey atom"),
    ("hey item", "hey atom"),
    ("hello item", "hello atom"),
    ("hello at one", "hello atom"),
    ("hello adam", "hello atom"),

    ("open the room", "open chrome"),
    ("open crumb", "open chrome"),
    ("open crome", "open chrome"),
    ("open grow", "open chrome"),
    ("open grown", "open chrome"),
    ("open gram", "open chrome"),
    ("open crime", "open chrome"),
    ("open brome", "open chrome"),

    ("check system", "check system"),
    ("take system", "check system"),
    ("check the system", "check system"),
    ("system status", "check system"),
    ("system check", "check system"),

    ("what time is it", "what time is it"),
    ("what's the time", "what time is it"),
    ("what time it is", "what time is it"),
    ("what is the time", "what time is it"),

    ("shut down", "shutdown"),
    ("shut down the pc", "shutdown pc"),
    ("shot down", "shutdown"),
    ("lock the screen", "lock screen"),
    ("lock screen", "lock screen"),
    ("log off", "logoff"),
    ("sign out", "logoff"),

    ("open no pad", "open notepad"),
    ("open know pad", "open notepad"),
    ("open note pad", "open notepad"),
    ("open the note pad", "open notepad"),
    ("close know pad", "close notepad"),
    ("close no pad", "close notepad"),

    ("atome", "atom"),
    # NOTE: ("item", "atom") used to live here. Removed 2026-04 because
    # it rewrote legitimate trailing wake-echoes in commands like
    # "play some music for me item" -> "play some music for me atom",
    # which broke music-intent matching (atom_log.txt L320-L325). The
    # wake-only confusable substitution now happens in
    # ``_correct_leading_wake_token`` and only fires when the suspect
    # token is at the head of the utterance.
    ("no pad", "notepad"),
    ("know pad", "notepad"),
    ("note pad", "notepad"),
    ("full wall you", "full volume"),
    ("wall you", "volume"),
    ("wall um", "volume"),
    ("power shell", "powershell"),
    ("out look", "outlook"),
    ("you tube", "youtube"),
    ("v s code", "vscode"),
    ("in tell i j", "intellij"),
    ("adam", "atom"),
    ("adm", "atom"),
    ("one morning", "good morning"),

    ("what can you do", "what can you do"),
    ("what are you", "what are you"),
    ("who are you", "who are you"),
    ("how are you", "how are you"),
    ("what do you do", "what can you do"),
]

_FILLER_PATTERN = re.compile(r"\b(um+|uh+|hmm+|hm+|ah+|oh+)\b", re.I)

NOISE_WORDS = frozenset({
    "the", "a", "an", "it", "is", "in", "on", "to", "of", "i",
    "and", "or", "but", "so", "do", "at", "by", "up", "if", "as",
    "he", "we", "be", "no", "my", "me", "that", "this", "was",
    "with", "not", "for", "you", "are", "had", "has", "him",
    "happen", "her", "them", "been", "have", "they",
    "yeah", "yes", "yep", "yup", "nah", "nope", "okay", "ok",
    "oh", "ah", "uh", "um", "hmm", "huh", "mm", "mhm",
    "what", "who", "how", "why", "when", "where", "which",
    "just", "like", "well", "right", "here", "there",
    "one", "two", "three", "four", "five", "six", "seven",
    "hey", "hi", "hello", "bye", "thanks", "thank",
    # Hindi filler/noise words
    "haan", "nahi", "ji", "aur", "toh", "hai", "ka", "ki",
    "ke", "ko", "se", "par", "mein", "yeh", "woh", "kya",
    "accha", "theek", "bas", "hm",
})

# ── Hindi text corrections ──────────────────────────────────────────
_HINDI_CORRECTIONS: list[tuple[str, str]] = [
    ("atom ko bolo", "atom ko bolo"),
    ("at um", "atom"),
    ("at on", "atom"),
    ("a tom", "atom"),
    ("aye tom", "atom"),
]


# Tokens we believe are STT misrecognitions of the wake word "atom" when
# they appear in the wake position (head of utterance). Body-position
# matches are NOT corrected because they are far more likely to be the
# user's actual word (e.g. "add this item to the list", "play music for
# me item" where the trailing token is a wake echo we tolerate).
_WAKE_CONFUSABLES: frozenset[str] = frozenset({
    "item",
    "atum",
})

_LEADING_WAKE_RE = re.compile(
    r"^\s*(?P<wake>" + "|".join(re.escape(w) for w in sorted(_WAKE_CONFUSABLES)) + r")\b",
    re.IGNORECASE,
)


def _correct_leading_wake_token(text: str) -> str:
    """Promote a confusable first token to the canonical wake word "atom".

    Only the FIRST token is rewritten — body-position occurrences belong
    to the user (e.g. "play music for me item"). Single-token utterances
    ("item") still get rewritten because they're almost certainly a bare
    wake call.
    """
    if not text:
        return text
    return _LEADING_WAKE_RE.sub("atom", text, count=1)


def correct_text(text: str) -> str:
    """Fix common misrecognitions and strip filler words (English + Hindi)."""
    t = text.lower().strip()
    t = _correct_leading_wake_token(t)
    for wrong, right in _TEXT_CORRECTIONS:
        pattern = r'\b' + re.escape(wrong) + r'\b'
        t = re.sub(pattern, right, t)
    for wrong, right in _HINDI_CORRECTIONS:
        pattern = r'\b' + re.escape(wrong) + r'\b'
        t = re.sub(pattern, right, t)
    t = _FILLER_PATTERN.sub("", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t


_FILLER_ONLY = re.compile(r"^[\s,.!?]+$")


def is_noise_word(text: str) -> bool:
    """True if the text is a hallucinated noise word or filler.

    Rejects: single noise words, two-word noise combos,
    and strings that are only punctuation/whitespace.
    """
    t = text.strip().lower()
    if not t or _FILLER_ONLY.match(t):
        return True
    words = t.split()
    if len(words) == 1:
        return t in NOISE_WORDS
    if len(words) == 2:
        return all(w in NOISE_WORDS for w in words)
    return False
