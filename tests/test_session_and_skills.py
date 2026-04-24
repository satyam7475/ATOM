"""ConversationMemory + SkillsRegistry unit tests."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.conversation_memory import ConversationMemory  # noqa: E402
from core.skills_registry import SkillsRegistry  # noqa: E402


def test_session_promotes_prior_turn_for_prompt() -> None:
    cfg = {"session": {"enabled": True, "max_query_snippet_chars": 200}}
    s = ConversationMemory(cfg)
    s.on_new_user_query("first query here")
    s.set_classified("time", None)
    s.on_new_user_query("second question")
    line = s.summary_for_prompt()
    assert "Prior turn" in line
    assert "first query" in line
    assert "time" in line


def test_skills_expand_from_file() -> None:
    payload = {
        "skills": [{"id": "t", "triggers": ["hello skill test"], "expand_to": "self check"}],
    }
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "s.json"
        p.write_text(json.dumps(payload), encoding="utf-8")
        reg = SkillsRegistry({"skills": {"enabled": True, "path": str(p)}})
        out = reg.try_expand("hello skill test")
        assert out is not None
        assert out[0] == "self check"
        assert out[1] == "t"


def test_skills_no_op_when_same_as_expand() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "s.json"
        p.write_text(
            json.dumps({"skills": [{"id": "x", "triggers": ["self check"], "expand_to": "self check"}]}),
            encoding="utf-8",
        )
        reg = SkillsRegistry({"skills": {"enabled": True, "path": str(p)}})
        assert reg.try_expand("self check") is None


def test_give_me_a_summary_does_not_trigger_self_check() -> None:
    """Regression for atom_log.txt L596-600: 'give me a summary' was
    matched by the over-generic whats_up trigger and expanded to
    'self check', producing identity boilerplate. The query should
    fall through to the LLM with the actual context as topic anchor.
    """
    cfg_path = Path(__file__).parent.parent / "config" / "skills.json"
    reg = SkillsRegistry({"skills": {"enabled": True, "path": str(cfg_path)}})
    for query in [
        "give me a summary",
        "give me a quick summary",
        "give me a summary of that file",
        "summarize this for me",
    ]:
        out = reg.try_expand(query)
        assert out is None or out[1] != "whats_up", (
            f"{query!r} still expands via whats_up: {out}"
        )


def test_status_report_phrases_still_route_to_self_check() -> None:
    """Tightened triggers must still serve the legitimate use-case
    without colliding with `system_check.system status`."""
    cfg_path = Path(__file__).parent.parent / "config" / "skills.json"
    reg = SkillsRegistry({"skills": {"enabled": True, "path": str(cfg_path)}})
    for query in ["status report", "atom self check", "whats up atom"]:
        out = reg.try_expand(query)
        assert out is not None, f"status query {query!r} no longer expands: {out}"
        assert out[0] == "self check"
        assert out[1] == "whats_up"


if __name__ == "__main__":
    test_session_promotes_prior_turn_for_prompt()
    test_skills_expand_from_file()
    test_skills_no_op_when_same_as_expand()
    test_give_me_a_summary_does_not_trigger_self_check()
    test_status_report_phrases_still_route_to_self_check()
    print("test_session_and_skills: ok")
