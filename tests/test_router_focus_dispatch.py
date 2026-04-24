"""
ATOM -- regression for F3 router wiring of macOS Focus actions.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.router import focus_actions
from core.router.router import Router


def _make_router_stub() -> Router:
    router = Router.__new__(Router)
    router._bus = MagicMock()
    return router


@pytest.mark.parametrize(
    "action",
    ["focus_on", "focus_off", "focus_state"],
)
def test_focus_actions_registered_in_dispatch(action: str) -> None:
    assert action in Router._ACTION_DISPATCH, (
        f"{action} missing from Router._ACTION_DISPATCH -- voice "
        f"command for {action} will fall through to the LLM."
    )


def test_focus_on_invokes_enable_focus(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def _fake_enable(*, duration_minutes: int | None = None) -> tuple[bool, str]:
        captured["duration"] = duration_minutes
        return True, "Focus on, Boss."

    monkeypatch.setattr(focus_actions, "enable_focus", _fake_enable)
    router = _make_router_stub()
    out = Router._do_focus_on(router, "focus_on", {"duration_minutes": 30})
    assert "Focus on" in out
    assert captured["duration"] == 30
    router._bus.emit_fast.assert_called_with(
        "focus_changed", state="on", duration_minutes=30,
    )


def test_focus_on_handles_no_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def _fake_enable(*, duration_minutes: int | None = None) -> tuple[bool, str]:
        captured["duration"] = duration_minutes
        return True, "Focus on."

    monkeypatch.setattr(focus_actions, "enable_focus", _fake_enable)
    router = _make_router_stub()
    out = Router._do_focus_on(router, "focus_on", {})
    assert "Focus on" in out
    assert captured["duration"] is None


def test_focus_on_does_not_emit_event_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(focus_actions, "enable_focus",
                        lambda **_kw: (False, "Setup hint."))
    router = _make_router_stub()
    out = Router._do_focus_on(router, "focus_on", {})
    assert "Setup hint." in out
    router._bus.emit_fast.assert_not_called()


def test_focus_off_invokes_disable_focus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(focus_actions, "disable_focus",
                        lambda: (True, "Focus off, Boss."))
    router = _make_router_stub()
    out = Router._do_focus_off(router, "focus_off", {})
    assert "Focus off" in out
    router._bus.emit_fast.assert_called_with(
        "focus_changed", state="off", duration_minutes=None,
    )


def test_focus_state_describes_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(focus_actions, "focus_state", lambda: "on")
    router = _make_router_stub()
    out = Router._do_focus_state(router, "focus_state", {})
    assert "Focus is on" in out


def test_focus_state_describes_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(focus_actions, "focus_state", lambda: "off")
    router = _make_router_stub()
    out = Router._do_focus_state(router, "focus_state", {})
    assert "Focus is off" in out


def test_focus_state_explains_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(focus_actions, "focus_state", lambda: "unknown")
    router = _make_router_stub()
    out = Router._do_focus_state(router, "focus_state", {})
    assert "FOCUS_SETUP.md" in out


# ── tool registry / security ────────────────────────────────────────


@pytest.mark.parametrize("name", ["focus_on", "focus_off", "focus_state"])
def test_focus_tools_registered_safely(name: str) -> None:
    from core.reasoning.tool_registry import get_tool_registry
    tool = get_tool_registry().get(name)
    assert tool is not None, f"{name} missing from tool registry"
    assert tool.safety_level == "safe"
    assert tool.requires_confirmation is False


def test_focus_actions_are_safe_always_intents() -> None:
    from core.security_policy import _SAFE_ALWAYS_INTENTS
    for action in ("focus_on", "focus_off", "focus_state"):
        assert action in _SAFE_ALWAYS_INTENTS, (
            f"{action} missing from _SAFE_ALWAYS_INTENTS -- "
            f"rate-limit will block rapid focus toggles."
        )


# ── planner & skills.json ──────────────────────────────────────────


def test_focus_mode_skill_uses_real_focus_command() -> None:
    """The legacy "go silent" expansion only put ATOM on mute. F3
    rewires it to actual macOS DND via 'focus mode on'."""
    import json
    import pathlib
    skills_path = (
        pathlib.Path(__file__).resolve().parents[1] / "config" / "skills.json"
    )
    skills = json.loads(skills_path.read_text())
    focus_skill = next((s for s in skills["skills"] if s["id"] == "focus_mode"), None)
    assert focus_skill is not None
    assert focus_skill["expand_to"] != "go silent", (
        "focus_mode skill must call actual focus, not just mute ATOM"
    )
    assert "focus" in focus_skill["expand_to"].lower()


def test_focus_mode_planner_includes_focus_on_step() -> None:
    """The planner's 'focus_mode' macro should now drive macOS DND."""
    from core.reasoning.planner import _PLAN_TEMPLATES
    steps = _PLAN_TEMPLATES.get("focus_mode")
    assert steps, "focus_mode plan template missing from planner"
    tools = [step.get("tool") for step in steps]
    assert "focus_on" in tools, (
        "focus_mode planner template must include the focus_on tool"
    )
