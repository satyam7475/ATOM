"""Tests for Jarvis-level upgrades: fast_path, conversation_memory, proactive_awareness, skills v2."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_latency_budget() -> None:
    from core.fast_path import LatencyBudget
    import time
    b = LatencyBudget(budget_ms=50, label="test")
    assert not b.overbudget
    assert b.remaining_ms > 0
    assert b.elapsed_ms < 50
    time.sleep(0.06)
    assert b.overbudget
    b.warn_if_slow("final")
    print("  PASS: LatencyBudget")


def test_conversation_memory_topics() -> None:
    from core.conversation_memory import ConversationMemory
    cm = ConversationMemory()
    cm.record("check cpu usage", "cpu", "CPU is at 12 percent, Boss.")
    cm.record("deploy the api to staging", "fallback", "Deploying now, Boss.")
    topics = cm.active_topics
    assert "cpu" in topics or "deploy" in topics or "api" in topics
    assert cm.turn_count == 2
    summary = cm.recent_summary()
    assert "Q:" in summary
    pairs = cm.get_pairs()
    assert len(pairs) == 2
    print("  PASS: ConversationMemory topics + summary")


def test_proactive_awareness_greeting() -> None:
    from core.proactive_awareness import ProactiveAwareness
    p = ProactiveAwareness({"features": {"proactive_awareness": True}})
    greeting = p.check_greeting()
    from datetime import datetime
    hour = datetime.now().hour
    if 5 <= hour < 22:
        assert greeting is not None, f"Expected greeting at hour {hour}"
        assert "Boss" in greeting
        second = p.check_greeting()
        assert second is None
    else:
        assert greeting is None, f"No greeting expected at hour {hour} (night)"
    assert p.check_idle(10) is None
    idle_hint = p.check_idle(600)
    assert idle_hint is not None
    assert "Boss" in idle_hint
    print("  PASS: ProactiveAwareness greeting + idle")


def test_proactive_disabled() -> None:
    p_off = __import__("core.proactive_awareness", fromlist=["ProactiveAwareness"]).ProactiveAwareness({})
    assert p_off.check_greeting() is None
    assert p_off.check_idle(9999) is None
    print("  PASS: ProactiveAwareness disabled")


def test_skills_v2_chain() -> None:
    from core.skills_registry import SkillsRegistry, SkillMatch
    import json, tempfile
    from pathlib import Path
    payload = {
        "skills": [
            {
                "id": "morning",
                "triggers": ["start my day"],
                "expand_to": "open chrome",
                "chain": ["open teams"],
            },
        ],
    }
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "s.json"
        p.write_text(json.dumps(payload), encoding="utf-8")
        reg = SkillsRegistry({"skills": {"enabled": True, "path": str(p)}})
        m = reg.try_expand_full("start my day")
        assert m is not None
        assert isinstance(m, SkillMatch)
        assert m.primary == "open chrome"
        assert m.chain == ["open teams"]
        assert m.skill_id == "morning"
        simple = reg.try_expand("start my day")
        assert simple is not None
        assert simple[0] == "open chrome"
    print("  PASS: SkillsRegistry v2 chain")


if __name__ == "__main__":
    test_latency_budget()
    test_conversation_memory_topics()
    test_proactive_awareness_greeting()
    test_proactive_disabled()
    test_skills_v2_chain()
    print("\ntest_jarvis_upgrades: ALL PASSED")
