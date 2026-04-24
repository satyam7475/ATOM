"""
ATOM -- regression suite for ``core.router.focus_actions`` (F3).

Tests are 100% offline: every shortcut invocation is monkeypatched to
return scripted results. We never spawn a real ``shortcuts`` CLI.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.router import focus_actions


# ── helpers ──────────────────────────────────────────────────────────


class _FakeProc:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _make_run(handler: Any) -> Any:
    """Build a fake ``subprocess.run`` that delegates to ``handler``."""

    def _run(cmd: list[str], **kwargs: Any) -> _FakeProc:
        return handler(cmd, kwargs)

    return _run


def _force_mac(monkeypatch: pytest.MonkeyPatch, *, has_cli: bool = True) -> None:
    monkeypatch.setattr(focus_actions, "_IS_MAC", True)
    if has_cli:
        monkeypatch.setattr(focus_actions.shutil, "which",
                            lambda _name: "/usr/bin/shortcuts")
    else:
        monkeypatch.setattr(focus_actions.shutil, "which", lambda _name: None)


# ── enable_focus ─────────────────────────────────────────────────────


def test_enable_focus_calls_named_shortcut(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_mac(monkeypatch)
    captured: list[list[str]] = []

    def _handler(cmd: list[str], _kwargs: dict) -> _FakeProc:
        captured.append(cmd)
        # Real shortcuts often produce no stdout when they just toggle
        # state -- mirror that here so we exercise the friendly default.
        return _FakeProc(returncode=0, stdout="")

    monkeypatch.setattr(focus_actions.subprocess, "run", _make_run(_handler))
    ok, msg = focus_actions.enable_focus()
    assert ok is True
    assert "Focus on" in msg or "guard your attention" in msg
    assert any("ATOM Focus On" in part for part in captured[0])


def test_enable_focus_passes_duration_via_input(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_mac(monkeypatch)
    captured: dict = {}

    def _handler(cmd: list[str], kwargs: dict) -> _FakeProc:
        captured["cmd"] = cmd
        captured["input"] = kwargs.get("input")
        return _FakeProc(returncode=0, stdout="")

    monkeypatch.setattr(focus_actions.subprocess, "run", _make_run(_handler))
    ok, msg = focus_actions.enable_focus(duration_minutes=45)
    assert ok is True
    assert "45 minutes" in msg
    assert captured["input"] == "45"
    assert "--input-path" in captured["cmd"]


def test_enable_focus_returns_setup_hint_when_shortcut_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_mac(monkeypatch)

    def _handler(_cmd: list[str], _kwargs: dict) -> _FakeProc:
        return _FakeProc(returncode=1, stderr="The shortcut couldn't be found.")

    monkeypatch.setattr(focus_actions.subprocess, "run", _make_run(_handler))
    ok, msg = focus_actions.enable_focus()
    assert ok is False
    assert "ATOM Focus On" in msg
    assert "FOCUS_SETUP.md" in msg


def test_enable_focus_returns_friendly_error_on_generic_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_mac(monkeypatch)

    def _handler(_cmd: list[str], _kwargs: dict) -> _FakeProc:
        return _FakeProc(returncode=2, stderr="permission denied")

    monkeypatch.setattr(focus_actions.subprocess, "run", _make_run(_handler))
    ok, msg = focus_actions.enable_focus()
    assert ok is False
    assert "permission denied" in msg.lower() or "failed" in msg.lower()


def test_enable_focus_short_circuits_on_non_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(focus_actions, "_IS_MAC", False)
    ok, msg = focus_actions.enable_focus()
    assert ok is False
    assert "macOS" in msg or "Boss" in msg


def test_enable_focus_reports_missing_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_mac(monkeypatch, has_cli=False)
    ok, msg = focus_actions.enable_focus()
    assert ok is False
    assert "shortcuts" in msg.lower() or "xcode-select" in msg.lower()


def test_enable_focus_handles_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_mac(monkeypatch)

    def _raise(_cmd: list[str], _kwargs: dict) -> _FakeProc:
        raise focus_actions.subprocess.TimeoutExpired(cmd=["shortcuts"], timeout=6.0)

    monkeypatch.setattr(focus_actions.subprocess, "run", _make_run(_raise))
    ok, msg = focus_actions.enable_focus()
    assert ok is False
    assert "timed out" in msg.lower()


# ── disable_focus ────────────────────────────────────────────────────


def test_disable_focus_calls_off_shortcut(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_mac(monkeypatch)
    captured: list[list[str]] = []

    def _handler(cmd: list[str], _kwargs: dict) -> _FakeProc:
        captured.append(cmd)
        return _FakeProc(returncode=0, stdout="")

    monkeypatch.setattr(focus_actions.subprocess, "run", _make_run(_handler))
    ok, msg = focus_actions.disable_focus()
    assert ok is True
    assert any(token in msg.lower() for token in ("off", "back"))
    assert any("ATOM Focus Off" in part for part in captured[0])


# ── focus_state ──────────────────────────────────────────────────────


def test_focus_state_returns_unknown_without_status_shortcut(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_mac(monkeypatch)
    monkeypatch.setattr(focus_actions, "_list_shortcuts", lambda: set())
    assert focus_actions.focus_state() == "unknown"


def test_focus_state_returns_on(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_mac(monkeypatch)
    monkeypatch.setattr(focus_actions, "_list_shortcuts",
                        lambda: {focus_actions.SHORTCUT_STATUS.lower()})

    def _handler(_cmd: list[str], _kwargs: dict) -> _FakeProc:
        return _FakeProc(returncode=0, stdout="On\n")

    monkeypatch.setattr(focus_actions.subprocess, "run", _make_run(_handler))
    assert focus_actions.focus_state() == "on"


def test_focus_state_returns_off(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_mac(monkeypatch)
    monkeypatch.setattr(focus_actions, "_list_shortcuts",
                        lambda: {focus_actions.SHORTCUT_STATUS.lower()})

    def _handler(_cmd: list[str], _kwargs: dict) -> _FakeProc:
        return _FakeProc(returncode=0, stdout="off")

    monkeypatch.setattr(focus_actions.subprocess, "run", _make_run(_handler))
    assert focus_actions.focus_state() == "off"


def test_focus_state_treats_garbage_output_as_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_mac(monkeypatch)
    monkeypatch.setattr(focus_actions, "_list_shortcuts",
                        lambda: {focus_actions.SHORTCUT_STATUS.lower()})

    def _handler(_cmd: list[str], _kwargs: dict) -> _FakeProc:
        return _FakeProc(returncode=0, stdout="maybe?")

    monkeypatch.setattr(focus_actions.subprocess, "run", _make_run(_handler))
    assert focus_actions.focus_state() == "unknown"


# ── toggle_focus ─────────────────────────────────────────────────────


def test_toggle_focus_disables_when_currently_on(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_mac(monkeypatch)
    monkeypatch.setattr(focus_actions, "focus_state", lambda: "on")
    called: list[str] = []
    monkeypatch.setattr(focus_actions, "disable_focus",
                        lambda: (True, called.append("off") or "off."))
    monkeypatch.setattr(focus_actions, "enable_focus",
                        lambda **_kw: (True, called.append("on") or "on."))
    ok, msg = focus_actions.toggle_focus()
    assert ok is True
    assert called == ["off"]


def test_toggle_focus_enables_when_state_unknown_or_off(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_mac(monkeypatch)
    monkeypatch.setattr(focus_actions, "focus_state", lambda: "unknown")
    called: list[str] = []
    monkeypatch.setattr(focus_actions, "enable_focus",
                        lambda **_kw: (True, called.append("on") or "on."))
    monkeypatch.setattr(focus_actions, "disable_focus",
                        lambda: (True, called.append("off") or "off."))
    ok, _msg = focus_actions.toggle_focus()
    assert ok is True
    assert called == ["on"]


# ── diagnostics ──────────────────────────────────────────────────────


def test_diagnostics_marks_unavailable_off_mac(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(focus_actions, "_IS_MAC", False)
    snap = focus_actions.diagnostics()
    assert snap == {"available": False, "reason": "non-darwin"}


def test_diagnostics_reports_shortcut_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_mac(monkeypatch)
    monkeypatch.setattr(focus_actions, "_list_shortcuts",
                        lambda: {focus_actions.SHORTCUT_ON.lower(),
                                 focus_actions.SHORTCUT_OFF.lower()})
    monkeypatch.setattr(focus_actions, "focus_state", lambda: "unknown")
    snap = focus_actions.diagnostics()
    assert snap["available"] is True
    assert snap["shortcut_on_present"] is True
    assert snap["shortcut_off_present"] is True
    assert snap["shortcut_status_present"] is False
    assert snap["current_state"] == "unknown"
