"""Dream engine M5 consolidation helpers.

Run: python3 -m tests.test_dream_engine_m5
"""

from __future__ import annotations


def test_pattern_summary() -> None:
    from core.cognitive.dream_engine import _build_pattern_summary

    p = [{"insight": "Boss often uses search", "count": 5, "action": "search"}]
    c = [{"topic": "Python", "occurrences": 3, "type": "semantic_cluster"}]
    s = _build_pattern_summary(p, c)
    assert "Boss often" in s
    assert "Python" in s


def test_second_brain_prune() -> None:
    from unittest.mock import MagicMock

    from core.cognitive.second_brain import SecondBrain

    mem = MagicMock()
    mem.preferences = {}
    beh = MagicMock()
    eng = SecondBrain(mem, beh, {"cognitive": {}})
    now = __import__("time").time()
    eng._facts = [
        {"text": "old dream noise", "source": "dream_consolidation", "importance": 0.2, "ts": now - 10 * 86400, "tags": [], "keywords": []},
        {"text": "keep me", "source": "conversation", "importance": 0.5, "ts": now, "tags": [], "keywords": []},
    ]
    n = eng.prune_for_consolidation()
    assert n == 1
    assert len(eng._facts) == 1


def main() -> None:
    test_pattern_summary()
    test_second_brain_prune()
    print("Dream M5 tests passed.")


if __name__ == "__main__":
    main()
