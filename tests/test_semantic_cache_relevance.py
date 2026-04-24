"""Regression suite for ``core.semantic_cache.SemanticCache`` relevance.

atom_log.txt 2026-04-25 captured a high-impact false-positive: a question
about "slbc shoes in the terminal" returned a cached answer about
chicken-wing nutrition facts (L605 → L632). The two queries shared no
content vocabulary but happened to embed within 0.92 cosine similarity.

The Jaccard token-overlap guard added in Phase D4 must block these
"semantically near, topically distant" hits while still allowing genuine
paraphrases like "what's the time" → "what time is it" through.
"""
from __future__ import annotations

from typing import Any

from core.semantic_cache import (
    SemanticCache,
    _content_tokens,
    _jaccard_overlap,
)


class _StubEmbeddings:
    """Returns a fixed unit vector for every query → cosine = 1.0 always.

    Lets us focus the test on the Jaccard guard without depending on the
    real embedding engine. The real engine is unit tested elsewhere; this
    test cares only that the guard rejects false-positive cosine hits.
    """

    def embed_sync(self, _query: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def batch_similarity(
        self,
        _query_embedding: list[float],
        embeddings: list[list[float]],
    ) -> list[float]:
        return [1.0] * len(embeddings)


def _build_cache(*, jaccard: float = 0.4, threshold: float = 0.92) -> SemanticCache:
    cache = SemanticCache(config={
        "semantic_cache": {
            "enabled": True,
            "max_size": 8,
            "ttl_seconds": 600,
            "threshold": threshold,
            "min_jaccard_overlap": jaccard,
            "persistent": False,
        },
    })
    cache._embedding_engine = _StubEmbeddings()
    cache._has_embeddings = True
    return cache


def test_content_tokens_strips_stopwords_and_atom_persona() -> None:
    """Boss / atom / function words must not contribute to the overlap."""
    toks = _content_tokens("Boss, atom, what is the slbc shoes in the terminal?")
    assert "boss" not in toks
    assert "atom" not in toks
    assert "the" not in toks
    assert "is" not in toks
    assert "slbc" in toks
    assert "shoes" in toks
    assert "terminal" in toks


def test_jaccard_zero_for_unrelated_topics() -> None:
    overlap = _jaccard_overlap(
        "what is slbc shoes in the terminal",
        "tell me chicken wing nutrition facts",
    )
    assert overlap == 0.0


def test_jaccard_high_for_paraphrase() -> None:
    overlap = _jaccard_overlap(
        "what is machine learning",
        "explain machine learning to me",
    )
    assert overlap >= 0.5


def test_cache_rejects_chicken_wings_for_slbc_query() -> None:
    """The exact regression from atom_log.txt L605-L632. Even when the
    embedding engine reports cosine=1.0 (worst-case false positive), the
    Jaccard guard must block the answer because the queries share zero
    content tokens."""
    cache = _build_cache()
    cache.put(
        "tell me chicken wing nutrition facts",
        "Chicken wings: 42g protein per serving, Boss.",
    )

    out = cache.get("what is slbc shoes in the terminal")

    assert out is None, (
        f"chicken-wing answer must NOT be served for an unrelated SLBC "
        f"query: got {out!r}"
    )


def test_cache_serves_genuine_paraphrase_through_jaccard_guard() -> None:
    """The guard must NOT block legitimate paraphrases that share content
    vocabulary, otherwise we destroy cache hit rate for the common case."""
    cache = _build_cache()
    cache.put(
        "what is machine learning",
        "Machine learning is the study of algorithms that learn from data.",
    )

    out = cache.get("explain machine learning to me")

    assert out is not None, (
        "machine-learning paraphrase must still hit the cache once Jaccard "
        "overlap >= 0.4 -- regressing this destroys cache hit rate"
    )
    assert "Machine learning" in out


def test_cache_disables_jaccard_guard_with_zero_threshold() -> None:
    """Operators can opt out by setting min_jaccard_overlap=0 in config —
    confirms the knob is wired through __init__."""
    cache = _build_cache(jaccard=0.0)
    cache.put(
        "tell me chicken wing nutrition facts",
        "Chicken wings: 42g protein.",
    )

    out = cache.get("what is slbc shoes in the terminal")

    assert out is not None, (
        "with min_jaccard_overlap=0 the guard must be disabled, restoring "
        "v1 behaviour (semantic similarity alone)"
    )


def test_diagnostics_expose_min_jaccard_overlap() -> None:
    cache = _build_cache(jaccard=0.55)
    diag: dict[str, Any] = cache.get_diagnostics()
    assert diag["min_jaccard_overlap"] == 0.55
