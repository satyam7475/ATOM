"""Classify user queries for smart RAG skip (latency reduction)."""

from __future__ import annotations

import re
from enum import Enum


class QueryComplexity(str, Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"


_SIMPLE_GREETINGS = re.compile(
    r"^(hi|hello|hey|good\s*(morning|afternoon|evening|night)|"
    r"thanks?|thank you|bye|goodbye|ok|okay|yes|no|sure|stop|silence)\b",
    re.I,
)
_SIMPLE_META = re.compile(
    r"^(what\s+time|what\s+date|how\s+are\s+you|who\s+are\s+you)\b",
    re.I,
)


def classify_query(text: str) -> QueryComplexity:
    """Return SIMPLE for trivial utterances — skip RAG to save latency."""
    t = (text or "").strip()
    if len(t) < 2:
        return QueryComplexity.SIMPLE
    if len(t) < 40 and _SIMPLE_GREETINGS.search(t):
        return QueryComplexity.SIMPLE
    if len(t) < 50 and _SIMPLE_META.search(t):
        return QueryComplexity.SIMPLE
    # Memory / recall / technical → need retrieval
    recall_kw = (
        "remember", "recall", "last time", "what did i", "my task",
        "document", "file", "explain", "how does", "why", "compare",
        "architecture", "schedule", "gpu", "error", "debug",
    )
    low = t.lower()
    if any(k in low for k in recall_kw):
        return QueryComplexity.COMPLEX
    if len(t) > 80:
        return QueryComplexity.COMPLEX
    return QueryComplexity.SIMPLE
