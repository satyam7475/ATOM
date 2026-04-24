"""
ATOM -- F5: confirmation policy regression suite.

Pins what should and should NOT prompt the user. The whole point of
the Jarvis-grade rewrite is that ATOM ACTS for safe, frequent things
(music, focus, lock, screenshot, open) and only pauses for
destructive ones (sleep, restart, file mutations).
"""

from __future__ import annotations

import pytest

from core.action_safety import (
    ActionRisk,
    default_risk_for_action,
    risk_requires_confirmation,
)
from core.reasoning.tool_registry import get_tool_registry


# Frictionless: should NEVER prompt the user.
_NO_CONFIRM = [
    "music_play", "music_pause", "music_next", "music_prev",
    "music_current", "music_play_specific",
    "focus_on", "focus_off", "focus_state",
    "lock_screen", "screenshot",
    "set_brightness", "set_volume", "mute", "unmute",
    "open_app", "open_url",
    "play_youtube",
    "minimize_window", "maximize_window",
    "switch_window", "next_window_in_app", "switch_space",
    "stop_music",
]


# Must always prompt (consequential / destructive).
_MUST_CONFIRM = [
    "sleep_pc", "shutdown_pc", "restart_pc", "logoff",
    "close_app",
    "empty_recycle_bin", "kill_process",
    "move_path", "copy_path", "create_folder",
    "set_focused_text", "click_ui_element", "type_text",
]


@pytest.mark.parametrize("action", _NO_CONFIRM)
def test_safe_actions_skip_confirmation(action: str) -> None:
    registry = get_tool_registry()
    assert registry.requires_confirmation(action) is False, (
        f"{action!r} unexpectedly requires confirmation. The "
        f"frictionless action set must run without a "
        f"\"are you sure?\" prompt."
    )


@pytest.mark.parametrize("action", _MUST_CONFIRM)
def test_dangerous_actions_demand_confirmation(action: str) -> None:
    registry = get_tool_registry()
    assert registry.requires_confirmation(action) is True, (
        f"{action!r} should require confirmation but the registry "
        f"reports otherwise -- this is a safety regression."
    )


# ── default-risk fallback (used when a tool is not in the registry) ─


@pytest.mark.parametrize(
    "action, expected",
    [
        ("lock_screen", ActionRisk.LOW),
        ("screenshot", ActionRisk.LOW),
        ("music_play", ActionRisk.LOW),
        ("focus_on", ActionRisk.LOW),
        ("set_volume", ActionRisk.LOW),
        ("play_youtube", ActionRisk.LOW),
        ("open_app", ActionRisk.MEDIUM),
        ("close_app", ActionRisk.HIGH),
        ("type_text", ActionRisk.HIGH),
        ("sleep_pc", ActionRisk.CRITICAL),
        ("shutdown_pc", ActionRisk.CRITICAL),
    ],
)
def test_default_risk_classification(action: str, expected: ActionRisk) -> None:
    assert default_risk_for_action(action) is expected


def test_risk_threshold_is_high() -> None:
    """Confirmation kicks in at HIGH and above; LOW/MEDIUM run silently."""
    assert risk_requires_confirmation(ActionRisk.LOW) is False
    assert risk_requires_confirmation(ActionRisk.MEDIUM) is False
    assert risk_requires_confirmation(ActionRisk.HIGH) is True
    assert risk_requires_confirmation(ActionRisk.CRITICAL) is True


# ── settings-driven extra confirmations ────────────────────────────


def test_settings_extra_confirmation_keeps_destructive_only() -> None:
    """The configurable allow-list must NOT contain music/focus/lock
    or other frictionless verbs; otherwise voice 'lock screen' would
    bounce off a confirmation prompt."""
    import json
    import pathlib

    settings_path = (
        pathlib.Path(__file__).resolve().parents[1] / "config" / "settings.json"
    )
    settings = json.loads(settings_path.read_text())
    extras = settings.get("security", {}).get("require_confirmation_for", [])

    forbidden = {
        "music_play", "music_pause", "music_next", "music_prev",
        "focus_on", "focus_off",
        "lock_screen", "screenshot", "play_youtube",
        "open_app", "open_url",
        "set_volume", "set_brightness",
    }
    overlap = forbidden.intersection(extras)
    assert not overlap, (
        f"settings.json security.require_confirmation_for contains "
        f"frictionless actions that must run without prompting: {overlap}"
    )


def test_settings_extra_confirmation_keeps_destructive_actions() -> None:
    """And conversely, destructive actions must still be in the
    confirmation allow-list."""
    import json
    import pathlib

    settings_path = (
        pathlib.Path(__file__).resolve().parents[1] / "config" / "settings.json"
    )
    settings = json.loads(settings_path.read_text())
    extras = set(
        settings.get("security", {}).get("require_confirmation_for", []),
    )
    expected = {
        "shutdown_pc", "restart_pc", "logoff", "sleep_pc",
        "empty_recycle_bin", "kill_process", "close_app",
        "move_path", "copy_path", "create_folder",
    }
    missing = expected - extras
    assert not missing, (
        f"settings.json security.require_confirmation_for is missing "
        f"required destructive actions: {missing}"
    )
