"""
Focused tests for RagEngine smart scoring.

Run: python -m tests.test_rag_engine
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _DummyEmbed:
    async def embed(self, _text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    def embed_sync(self, _text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def shutdown(self) -> None:
        return None


def _build_engine():
    import core.rag.rag_engine as rag_mod

    original = rag_mod.get_embedding_engine
    rag_mod.get_embedding_engine = lambda _cfg=None: _DummyEmbed()
    try:
        return rag_mod.RagEngine(
            config={
                "owner": {"name": "Satyam"},
                "rag": {
                    "enabled": True,
                    "persistent_embed_cache": False,
                    "top_k": 3,
                    "max_snippets": 3,
                },
            },
        )
    finally:
        rag_mod.get_embedding_engine = original


def test_owner_priority_boost() -> None:
    engine = _build_engine()
    now = time.time()
    ranked = engine._hybrid_merge(
        "dark mode preference",
        [
            ("Generic dark theme note", 0.86, {"timestamp": now, "_collection": "facts"}),
            ("Boss prefers dark mode", 0.62, {"source": "preference", "timestamp": now, "_collection": "facts"}),
        ],
        now,
        2,
    )
    assert ranked[0][0] == "Boss prefers dark mode"
    print("  PASS: owner-priority boosts direct owner facts")


def test_usage_frequency_boost() -> None:
    engine = _build_engine()
    for _ in range(6):
        engine._record_chunk_usage(["Important kubernetes note"])
    now = time.time()
    ranked = engine._hybrid_merge(
        "kubernetes note",
        [
            ("Fresh but unused note", 0.63, {"timestamp": now, "_collection": "facts"}),
            ("Important kubernetes note", 0.56, {"timestamp": now, "_collection": "facts"}),
        ],
        now,
        2,
    )
    assert ranked[0][0] == "Important kubernetes note"
    print("  PASS: usage-frequency boost promotes repeatedly used chunks")


def test_stale_chunks_are_marked() -> None:
    engine = _build_engine()
    now = time.time()
    old_ts = now - (90 * 24 * 3600)
    chunk = engine._format_chunk(
        "Current roadmap status for the release",
        {"source": "voice", "timestamp": old_ts, "_collection": "facts"},
        now,
    )
    assert chunk.startswith("[Possibly outdated] ")
    print("  PASS: stale time-sensitive chunks are labeled")


def test_preferences_do_not_stale_without_signal() -> None:
    engine = _build_engine()
    now = time.time()
    old_ts = now - (120 * 24 * 3600)
    chunk = engine._format_chunk(
        "Boss prefers concise answers",
        {"source": "preference", "timestamp": old_ts, "_collection": "facts"},
        now,
    )
    assert not chunk.startswith("[Possibly outdated] ")
    print("  PASS: stable preferences avoid stale labeling")


def run_all() -> None:
    test_owner_priority_boost()
    test_usage_frequency_boost()
    test_stale_chunks_are_marked()
    test_preferences_do_not_stale_without_signal()
    print("\n=== RAG SMART SCORING TESTS PASSED ===\n")


if __name__ == "__main__":
    run_all()
