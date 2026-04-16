"""
Shared response-policy helpers for ATOM.

Keeps prompt hints, routing, quick replies, and report export behavior aligned
around one rule set: short by default, detailed only when explicitly asked.
"""

from __future__ import annotations

import logging

logger = logging.getLogger('atom.core.query_policy')

import re
from enum import Enum

_WS_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_SHORT_HINT_RE = re.compile(
    r"\b("
    r"short(?:\s+answer)?|brief(?:ly)?|in\s+short|one\s+line|"
    r"short\s+me|short\s+mein|chh?ot[ae]\s+me|chh?ota\s+jawab"
    r")\b",
    re.I,
)
_NORMALIZATION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bwhats\b", re.I), "what is"),
    (re.compile(r"\bwhats\s+the\s+tiem\b", re.I), "what is the time"),
    (re.compile(r"\btiem\b", re.I), "time"),
    (re.compile(r"\byuo\b", re.I), "you"),
    (re.compile(r"\bopitmal\b", re.I), "optimal"),
    (re.compile(r"\bperformnce\b", re.I), "performance"),
    (re.compile(r"\bmemroy\b", re.I), "memory"),
    (re.compile(r"\bsamjha\s*do\b", re.I), "explain"),
    (re.compile(r"\bsamjhao\b", re.I), "explain"),
    (re.compile(r"\bbatao\b", re.I), "tell me"),
    (re.compile(r"\bbata\s*do\b", re.I), "tell me"),
    (re.compile(r"\bkyu+n?\b", re.I), "why"),
    (re.compile(r"\bfarak\b", re.I), "difference"),
    (re.compile(r"\bkya\s+hota\s+hai\b", re.I), "what is"),
    (re.compile(r"\bkya\s+hai\b", re.I), "what is"),
    (re.compile(r"\bmatlab\b", re.I), "meaning"),
    (re.compile(r"\bhindi\s+me(?:in)?\b", re.I), "in hindi"),
    (re.compile(r"\bhinglish\s+me(?:in)?\b", re.I), "in hinglish"),
    (re.compile(r"\bdetail\s+me(?:in)?\b", re.I), "in detail"),
    (re.compile(r"\bshort\s+me(?:in)?\b", re.I), "short"),
    (re.compile(r"\bchh?ota\s+joke\b", re.I), "short joke"),
)
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")
_ROMANIZED_HINDI_TOKENS = frozenset(
    {
        "kya",
        "kyu",
        "kyun",
        "kaise",
        "mujhe",
        "mera",
        "meri",
        "mere",
        "samjhao",
        "samjha",
        "batao",
        "farak",
        "kya",
        "hota",
        "hai",
        "aur",
        "nahi",
        "bolo",
        "kar",
        "karo",
        "mein",
        "acha",
        "achha",
        "kyunki",
    }
)

_DETAIL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        r"\bexplain(?:\s+(?:this|that|it))?\s+properly\b",
        r"\bexplain(?:\s+(?:this|that|it))?\s+in\s+detail\b",
        r"\bgive(?:\s+me)?\s+(?:a\s+)?detailed\s+(?:answer|analysis|explanation)\b",
        r"\bdetailed\s+(?:answer|analysis|explanation)\b",
        r"\bdeep\s+analysis\b",
        r"\banaly[sz]e(?:\s+(?:this|that|it))?\s+(?:deeply|in\s+detail)\b",
        r"\bwalk\s+me\s+through\b",
        r"\bproper\s+explanation\b",
        r"\bin\s+detail\b",
        r"\bexplain\b",
        r"\bach(?:h|)e?\s+se\s+samjha(?:o|do)\b",
        r"\bproper(?:ly)?\s+samjha(?:o|do)\b",
    )
)
_REPORT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        r"\bresearch(?:\s+(?:this|that|it|best|the|why|how|what|which))?\b",
        r"\bfull\s+report\b",
        r"\bdetailed\s+report\b",
        r"\bwrite(?:\s+me)?\s+(?:a\s+)?report\b",
        r"\breport\s+on\b",
        r"\binvestigate\b",
        r"\bpoori?\s+report\b",
        r"\bfull\s+detailed\s+report\b",
    )
)
_SHORT_QUERY_RE = re.compile(
    r"\b("
    r"hi|hello|hey|yo|good\s+morning|good\s+afternoon|good\s+evening|"
    r"how\s+are\s+you|how'?s\s+it\s+going|what'?s\s+up|"
    r"who\s+are\s+you|what\s+are\s+you|what'?s\s+your\s+name|"
    r"what\s+time\s+is\s+it|what'?s\s+the\s+time|"
    r"what\s+date\s+is\s+it|what\s+day\s+is\s+it|"
    r"are\s+you\s+there|status(?:\s+report)?"
    r")\b",
    re.I,
)
_QUESTION_STARTERS = frozenset({"what", "who", "how", "why", "when", "where"})
_COMMAND_STARTERS = frozenset(
    {
        "open",
        "close",
        "kill",
        "launch",
        "write",
        "create",
        "build",
        "make",
        "add",
        "update",
        "change",
        "fix",
        "remove",
        "delete",
        "install",
        "configure",
        "set",
        "turn",
        "enable",
        "disable",
        "restart",
    }
)


class ResponseMode(str, Enum):
    SHORT = "short"
    NORMAL = "normal"
    DETAIL = "detail"
    REPORT = "report"


def normalize_query(text: str) -> str:
    q = (text or "").strip()
    if not q:
        return ""
    try:
        from voice.speech_detector import correct_text

        q = correct_text(q)
    except Exception:
        logger.debug('core query policy optional step failed', exc_info=True)
    q = _WS_RE.sub(" ", q.lower())
    for pattern, replacement in _NORMALIZATION_PATTERNS:
        q = pattern.sub(replacement, q)
    return _WS_RE.sub(" ", q).strip()


def wants_detailed_answer(query: str) -> bool:
    q = normalize_query(query)
    return bool(q) and any(pattern.search(q) for pattern in _DETAIL_PATTERNS)


def wants_research_report(query: str) -> bool:
    q = normalize_query(query)
    return bool(q) and any(pattern.search(q) for pattern in _REPORT_PATTERNS)


def wants_explicit_depth(query: str) -> bool:
    return wants_detailed_answer(query) or wants_research_report(query)


def classify_response_mode(query: str) -> ResponseMode:
    q = normalize_query(query)
    if not q:
        return ResponseMode.SHORT
    if wants_research_report(q):
        return ResponseMode.REPORT
    if wants_detailed_answer(q):
        return ResponseMode.DETAIL
    if _SHORT_HINT_RE.search(q):
        return ResponseMode.SHORT
    if _SHORT_QUERY_RE.search(q):
        return ResponseMode.SHORT

    words = q.split()
    if not words:
        return ResponseMode.SHORT
    if len(words) <= 4 and words[0] in _QUESTION_STARTERS:
        return ResponseMode.SHORT
    if len(words) <= 3 and words[0] not in _COMMAND_STARTERS:
        return ResponseMode.SHORT
    return ResponseMode.NORMAL


def detect_response_language(query: str, *, previous: str = "english") -> str:
    raw = (query or "").strip()
    if not raw:
        return previous or "english"
    q = normalize_query(raw)
    if "in english" in q or "english mein" in q:
        return "english"
    if "in hindi" in q:
        return "hindi"
    if "in hinglish" in q:
        return "hinglish"
    if _DEVANAGARI_RE.search(raw):
        return "hindi"
    hindi_hits = sum(1 for token in q.split() if token in _ROMANIZED_HINDI_TOKENS)
    if hindi_hits >= 2:
        return "hinglish"
    if previous in {"hindi", "hinglish"} and len(q.split()) <= 6:
        return previous
    return "english"


def should_export_report(
    query: str,
    response_text: str,
    *,
    min_words: int = 140,
    min_chars: int = 900,
) -> bool:
    text = _WS_RE.sub(" ", (response_text or "").strip())
    if not text:
        return False
    if wants_research_report(query):
        return True
    if wants_detailed_answer(query):
        return len(text.split()) >= min_words or len(text) >= min_chars
    return False


def summarize_report(
    text: str,
    *,
    max_sentences: int = 2,
    max_chars: int = 220,
) -> str:
    clean = _WS_RE.sub(" ", (text or "").strip())
    if not clean:
        return ""
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(clean) if part.strip()]
    summary = " ".join(parts[:max_sentences]).strip() if parts else clean
    if not summary:
        summary = clean
    if len(summary) > max_chars:
        summary = summary[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:")
        summary = f"{summary}..."
    if summary and summary[-1] not in ".!?":
        summary += "."
    return summary


def slugify_query(query: str, *, max_words: int = 8, max_chars: int = 60) -> str:
    tokens = re.findall(r"[a-z0-9]+", normalize_query(query))
    if not tokens:
        return "report"
    slug = "-".join(tokens[:max_words]).strip("-")
    if len(slug) > max_chars:
        slug = slug[:max_chars].rstrip("-")
    return slug or "report"
