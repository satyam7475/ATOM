"""Tests for goal step ↔ tool mapping (Phase 6.2).

Run: python3 -m tests.test_goal_engine_tools
"""

from __future__ import annotations


def test_infer_open_app() -> None:
    from core.cognitive.goal_engine import infer_suggested_tool

    t, a = infer_suggested_tool("Open Safari for UI testing")
    assert t == "open_app"
    assert "safari" in a.get("name", "").lower()


def test_infer_bracket_tool() -> None:
    from core.cognitive.goal_engine import infer_suggested_tool

    t, a = infer_suggested_tool("Read release notes [tool:remember]")
    assert t == "remember"
    assert "release" in a.get("fact", "").lower()


def test_finalize_strip_brackets() -> None:
    from core.cognitive.goal_engine import _finalize_step_record

    step = {"title": "Ship v2 [tool:screenshot]", "status": "pending"}
    _finalize_step_record(step)
    assert "[" not in step["title"]
    assert step.get("suggested_tool") == "screenshot"


def test_apply_tool_completion() -> None:
    from unittest.mock import MagicMock

    from core.cognitive.goal_engine import GoalEngine

    bus = MagicMock()
    bus.on = MagicMock()
    bus.off = MagicMock()
    bus.emit_fast = MagicMock()
    brain = MagicMock()
    cfg = {
        "cognitive": {
            "goals_enabled": True,
            "goal_tool_auto_complete": True,
            "goal_tool_match_strict": False,
        },
    }
    eng = GoalEngine(bus, brain, cfg)
    eng._goals = []
    g = eng.create_goal("Demo goal")
    assert "error" not in g
    gid = g["id"]
    goal = eng._find_by_id(gid)
    assert goal is not None
    goal["steps"] = [
        {
            "id": "s1",
            "title": "Remember the launch checklist",
            "status": "pending",
            "minutes_logged": 0,
            "created_at": "",
            "updated_at": "",
        },
    ]
    from core.cognitive.goal_engine import _finalize_step_record

    _finalize_step_record(goal["steps"][0])
    assert goal["steps"][0].get("suggested_tool") == "remember"

    ok = eng.apply_tool_completion("remember", {"fact": "the launch checklist"})
    assert ok is True
    assert goal["steps"][0]["status"] == "completed"


def main() -> None:
    test_infer_open_app()
    test_infer_bracket_tool()
    test_finalize_strip_brackets()
    test_apply_tool_completion()
    print("All goal_engine tool tests passed.")


if __name__ == "__main__":
    main()
