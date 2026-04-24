"""
ATOM -- F4 system-control audit.

Pins:

* Every casual-control voice path (open_app, lock_screen, sleep_pc,
  brightness, volume, screenshot, switch_window, switch_space,
  next_window) is wired end-to-end:
    intent -> dispatch table -> handler module function.
* Newly-added macOS verbs (``switch_space`` and ``next_window_in_app``)
  produce the right keystroke commands without crashing.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from core.intent_engine import IntentEngine
from core.router import utility_actions
from core.router.router import Router


# ── intent ↔ action mapping ─────────────────────────────────────────


@pytest.mark.parametrize(
    "phrase, expected_action",
    [
        ("open chrome", "open_app"),
        ("lock the screen", "lock_screen"),
        ("put my mac to sleep", "sleep_pc"),
        ("set brightness to 60 percent", "set_brightness"),
        ("set volume to 40 percent", "set_volume"),
        ("take a screenshot", "screenshot"),
        ("switch app", "switch_window"),
        ("alt tab", "switch_window"),
        ("next space", "switch_space"),
        ("previous workspace", "switch_space"),
        ("cycle windows", "next_window_in_app"),
        ("other window", "next_window_in_app"),
    ],
)
def test_voice_phrase_resolves_to_expected_action(
    phrase: str, expected_action: str,
) -> None:
    engine = IntentEngine()
    out = engine.classify(phrase)
    assert out.action == expected_action, (
        f"phrase {phrase!r} -> action {out.action!r} (intent={out.intent!r})"
    )


def test_switch_space_arg_direction_is_extracted() -> None:
    engine = IntentEngine()
    nxt = engine.classify("next space")
    assert nxt.action == "switch_space"
    assert nxt.action_args is not None
    assert nxt.action_args.get("direction") == "right"

    prev = engine.classify("previous workspace")
    assert prev.action == "switch_space"
    assert prev.action_args is not None
    assert prev.action_args.get("direction") == "left"


# ── dispatch table coverage ─────────────────────────────────────────


@pytest.mark.parametrize(
    "action",
    [
        "open_app", "lock_screen", "sleep_pc", "set_brightness",
        "set_volume", "screenshot",
        "switch_window", "next_window_in_app", "switch_space",
        "minimize_window", "maximize_window",
        "music_play", "music_pause", "music_next", "music_prev",
        "music_current", "music_play_specific",
        "focus_on", "focus_off", "focus_state",
    ],
)
def test_action_present_in_dispatch_table(action: str) -> None:
    assert action in Router._ACTION_DISPATCH, (
        f"{action!r} missing from Router._ACTION_DISPATCH -- the "
        f"router will fall through to the LLM and the model will "
        f"hallucinate the answer instead of acting."
    )


# ── handlers route to utility_actions ───────────────────────────────


def test_next_window_in_app_handler_invokes_utility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[str] = []
    monkeypatch.setattr(utility_actions, "next_window_in_app",
                        lambda: called.append("next") or None)
    router = Router.__new__(Router)
    router._bus = MagicMock()
    out = Router._do_next_window_in_app(router, "next_window_in_app", {})
    assert called == ["next"]
    assert isinstance(out, str) and out


def test_switch_space_handler_forwards_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []
    monkeypatch.setattr(utility_actions, "switch_space",
                        lambda direction="right": captured.append(direction))
    router = Router.__new__(Router)
    router._bus = MagicMock()

    out_right = Router._do_switch_space(router, "switch_space",
                                        {"direction": "right"})
    out_left = Router._do_switch_space(router, "switch_space",
                                       {"direction": "left"})
    assert captured == ["right", "left"]
    assert "right" in out_right.lower() or "switched" in out_right.lower()
    assert "back" in out_left.lower() or "previous" in out_left.lower() \
        or "left" in out_left.lower()


def test_switch_space_handler_defaults_to_right_for_garbage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []
    monkeypatch.setattr(utility_actions, "switch_space",
                        lambda direction="right": captured.append(direction))
    router = Router.__new__(Router)
    router._bus = MagicMock()
    Router._do_switch_space(router, "switch_space", {"direction": "diagonal"})
    Router._do_switch_space(router, "switch_space", {})
    assert captured == ["right", "right"]


# ── osascript shape (smoke; no real subprocess) ────────────────────


def test_next_window_in_app_uses_cmd_backtick_on_darwin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(utility_actions.sys, "platform", "darwin")
    captured: list[list[str]] = []

    class _FakeProc:
        returncode = 0

    def _fake_run(cmd: list[str], **_kw: object) -> _FakeProc:
        captured.append(cmd)
        return _FakeProc()

    monkeypatch.setattr(utility_actions.subprocess, "run", _fake_run)
    utility_actions.next_window_in_app()
    assert captured, "subprocess.run was not called"
    script = captured[0][-1]
    assert "`" in script and "command down" in script


def test_switch_space_uses_arrow_keycodes_on_darwin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(utility_actions.sys, "platform", "darwin")
    captured: list[list[str]] = []

    class _FakeProc:
        returncode = 0

    def _fake_run(cmd: list[str], **_kw: object) -> _FakeProc:
        captured.append(cmd)
        return _FakeProc()

    monkeypatch.setattr(utility_actions.subprocess, "run", _fake_run)
    utility_actions.switch_space("right")
    utility_actions.switch_space("left")
    assert len(captured) == 2
    right_script = captured[0][-1]
    left_script = captured[1][-1]
    # AppleScript right-arrow key code is 124, left-arrow is 123.
    assert "124" in right_script
    assert "123" in left_script
    assert "control down" in right_script
    assert "control down" in left_script


# ── tool registry ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name", ["next_window_in_app", "switch_space"],
)
def test_new_window_tools_registered(name: str) -> None:
    from core.reasoning.tool_registry import get_tool_registry
    tool = get_tool_registry().get(name)
    assert tool is not None
    assert tool.safety_level == "safe"
    assert tool.requires_confirmation is False
    assert tool.category == "desktop"


def test_new_window_actions_in_safe_always_intents() -> None:
    from core.security_policy import _SAFE_ALWAYS_INTENTS
    assert "switch_space" in _SAFE_ALWAYS_INTENTS
    assert "next_window_in_app" in _SAFE_ALWAYS_INTENTS
