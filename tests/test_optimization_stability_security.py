"""Regression tests for the optimization / stability / security hardening batch.

Covers:

1. ``system_monitor.get_system_state`` emits at DEBUG, not INFO (log-volume fix).
2. :func:`core.security_secret_scrub.scrub_sensitive_env` clears known secrets
   from the environment and returns the pre-scrub snapshot.
3. :class:`core.runtime_watchdog.RuntimeWatchdog` escalates after the
   configured number of consecutive LLM/TTS timeouts and calls
   ``request_profile_demote`` on the local brain.
4. :func:`core.macos.fs_watcher._should_ignore` blocks paths under ``~/.ssh``,
   ``~/.aws``, keychains etc. regardless of the watched root.
5. :meth:`cursor_bridge.local_brain_controller.LocalBrainController.request_profile_demote`
   flips the brain to ``FAST`` mode and is idempotent.

These are pure unit-level checks so they stay under 1s and never import MLX
or the full voice pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# 1. system_monitor log level
# ---------------------------------------------------------------------------


def test_system_monitor_v7_line_is_debug(caplog: pytest.LogCaptureFixture) -> None:
    """``v7_system_state`` must fire at DEBUG, not INFO, to keep logs readable."""
    from core.system.system_monitor import get_system_state

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="atom.system_monitor"):
        get_system_state()

    info_lines = [
        r for r in caplog.records
        if r.name == "atom.system_monitor" and r.levelno >= logging.INFO
        and "v7_system_state" in r.getMessage()
    ]
    debug_lines = [
        r for r in caplog.records
        if r.name == "atom.system_monitor" and r.levelno == logging.DEBUG
        and "v7_system_state" in r.getMessage()
    ]
    assert not info_lines, (
        f"v7_system_state must not emit at INFO anymore; got {info_lines!r}"
    )
    assert debug_lines, "Expected at least one DEBUG v7_system_state record"


# ---------------------------------------------------------------------------
# 2. secret scrub
# ---------------------------------------------------------------------------


def test_secret_scrub_snapshots_and_blanks_env() -> None:
    from core.security_secret_scrub import (
        reset_for_tests,
        scrub_sensitive_env,
        sensitive_env_vars,
    )

    reset_for_tests()
    env: dict[str, str] = {
        "HF_TOKEN": "hf_xxxx",
        "OPENAI_API_KEY": "sk-test",
        "GEMINI_API_KEY": "AIza...",
        "ATOM_DASHBOARD_TOKEN": "token_keep",
        "UNRELATED_VAR": "safe",
    }
    snapshot = scrub_sensitive_env(
        env=env, preserve=("ATOM_DASHBOARD_TOKEN",),
    )

    assert snapshot["HF_TOKEN"] == "hf_xxxx"
    assert snapshot["OPENAI_API_KEY"] == "sk-test"
    assert snapshot["GEMINI_API_KEY"] == "AIza..."
    assert snapshot["ATOM_DASHBOARD_TOKEN"] == "token_keep"

    assert env["HF_TOKEN"] == ""
    assert env["OPENAI_API_KEY"] == ""
    assert env["GEMINI_API_KEY"] == ""
    assert env["ATOM_DASHBOARD_TOKEN"] == "token_keep"
    assert env["UNRELATED_VAR"] == "safe"

    known = set(sensitive_env_vars())
    assert "HF_TOKEN" in known and "OPENAI_API_KEY" in known
    assert "ATOM_ADMIN_TOKEN" in known


def test_secret_scrub_is_idempotent() -> None:
    from core.security_secret_scrub import reset_for_tests, scrub_sensitive_env

    reset_for_tests()
    env: dict[str, str] = {"HF_TOKEN": "first"}
    snap1 = scrub_sensitive_env(env=env)
    env["HF_TOKEN"] = "reinjected"
    snap2 = scrub_sensitive_env(env=env)

    assert snap1["HF_TOKEN"] == "first"
    assert snap2 == snap1, (
        "scrub must be idempotent -- re-injected secret must NOT be picked up"
    )
    # value that was re-injected after scrub is left untouched (caller's job)
    assert env["HF_TOKEN"] == "reinjected"


# ---------------------------------------------------------------------------
# 3. runtime watchdog consecutive-timeout demotion
# ---------------------------------------------------------------------------


class _DummyBus:
    def __init__(self) -> None:
        self.emitted: list[tuple[str, dict[str, Any]]] = []
        self.handlers: dict[str, Any] = {}

    def on(self, event: str, handler: Any) -> None:
        self.handlers[event] = handler

    def emit(self, event: str, **kwargs: Any) -> None:
        self.emitted.append((event, kwargs))

    def emit_fast(self, event: str, **kwargs: Any) -> None:
        self.emitted.append((event, kwargs))


class _DummyState:
    def __init__(self) -> None:
        self.current = SimpleNamespace(value="idle")


def _build_watchdog() -> tuple[Any, _DummyBus]:
    from core.runtime_watchdog import RuntimeWatchdog

    bus = _DummyBus()
    state = _DummyState()
    config = {
        "performance": {
            "watchdog_llm_timeout_s": 1.0,
            "watchdog_tts_timeout_s": 1.0,
            "watchdog_timeout_demote_threshold": 2,
            "supervisor_restart_cooldown_s": 0.0,
        },
    }
    wd = RuntimeWatchdog(bus, state, config)
    return wd, bus


def test_watchdog_demotes_brain_after_consecutive_llm_timeouts() -> None:
    async def _run() -> None:
        wd, bus = _build_watchdog()
        brain = MagicMock()
        brain.request_profile_demote.return_value = True
        wd.attach_local_brain(brain)

        wd._handle_budget_timeout("llm_inference", 1.0, metadata={"turn": 1})
        assert brain.request_profile_demote.call_count == 0, (
            "First timeout must not demote"
        )
        wd._handle_budget_timeout("llm_inference", 1.0, metadata={"turn": 2})
        assert brain.request_profile_demote.call_count == 1, (
            "Second consecutive LLM timeout must trigger demotion"
        )
        kwargs = brain.request_profile_demote.call_args.kwargs
        assert kwargs.get("reason") == "llm_timeout_streak"

    asyncio.new_event_loop().run_until_complete(_run())


def test_watchdog_resets_streak_on_successful_turn() -> None:
    async def _run() -> None:
        wd, _bus = _build_watchdog()
        brain = MagicMock()
        brain.request_profile_demote.return_value = True
        wd.attach_local_brain(brain)

        wd._handle_budget_timeout("llm_inference", 1.0)
        assert wd._consecutive_llm_timeouts == 1

        await wd._on_successful_turn()
        assert wd._consecutive_llm_timeouts == 0
        assert wd._consecutive_tts_timeouts == 0

        # After a successful turn, a single new timeout must NOT trip demotion.
        wd._handle_budget_timeout("llm_inference", 1.0)
        assert brain.request_profile_demote.call_count == 0

    asyncio.new_event_loop().run_until_complete(_run())


def test_watchdog_tts_timeout_streak_triggers_demotion() -> None:
    async def _run() -> None:
        wd, _bus = _build_watchdog()
        brain = MagicMock()
        brain.request_profile_demote.return_value = True
        wd.attach_local_brain(brain)

        wd._handle_budget_timeout("tts_synthesis", 1.0)
        wd._handle_budget_timeout("tts_synthesis", 1.0)

        assert brain.request_profile_demote.call_count == 1
        kwargs = brain.request_profile_demote.call_args.kwargs
        assert kwargs.get("reason") == "tts_timeout_streak"

    asyncio.new_event_loop().run_until_complete(_run())


# ---------------------------------------------------------------------------
# 4. FSWatcher sensitive-path deny
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/Users/satyam/.ssh/id_ed25519",
        "/Users/satyam/.aws/credentials",
        "/Users/satyam/.gnupg/gpg.conf",
        "/Users/satyam/.kube/config",
        "/Users/satyam/.docker/config.json",
        "/Users/satyam/.netrc",
        "/Users/satyam/Library/Keychains/login.keychain-db",
        "/Users/satyam/Library/Cookies/Cookies.binarycookies",
        "/Users/satyam/.pgpass",
        "/Users/satyam/.config/gh/hosts.yml",
        "/Users/satyam/.atom_secrets",
    ],
)
def test_fs_watcher_skips_sensitive_paths(path: str) -> None:
    from core.macos.fs_watcher import _should_ignore

    assert _should_ignore(path), f"FSWatcher must deny sensitive path: {path}"


@pytest.mark.parametrize(
    "path",
    [
        "/Users/satyam/Desktop/notes.txt",
        "/Users/satyam/Downloads/report.pdf",
        "/Users/satyam/Documents/todo.md",
    ],
)
def test_fs_watcher_allows_normal_user_paths(path: str) -> None:
    from core.macos.fs_watcher import _should_ignore

    assert not _should_ignore(path), (
        f"FSWatcher must still surface non-sensitive path: {path}"
    )


# ---------------------------------------------------------------------------
# 5. request_profile_demote on LocalBrainController
# ---------------------------------------------------------------------------


class _FakeLLM:
    def __init__(self) -> None:
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1

    def is_available(self) -> bool:
        return True


def test_local_brain_request_profile_demote_flips_to_fast() -> None:
    from cursor_bridge.local_brain_controller import LocalBrainController

    ctrl = LocalBrainController.__new__(LocalBrainController)
    ctrl._current_runtime_mode = "SMART"  # type: ignore[attr-defined]
    ctrl._last_mode_info = {}  # type: ignore[attr-defined]
    ctrl._llm = _FakeLLM()  # type: ignore[attr-defined]

    result = ctrl.request_profile_demote(reason="llm_timeout_streak")

    assert result is True
    assert ctrl._current_runtime_mode == "FAST"  # type: ignore[attr-defined]
    assert ctrl._last_mode_info.get("runtime_mode") == "FAST"  # type: ignore[attr-defined]
    assert ctrl._last_mode_info.get("reason", "").startswith(  # type: ignore[attr-defined]
        "watchdog_demote:",
    )
    assert ctrl._llm.shutdown_calls == 1  # type: ignore[attr-defined]


def test_local_brain_request_profile_demote_is_idempotent() -> None:
    from cursor_bridge.local_brain_controller import LocalBrainController

    ctrl = LocalBrainController.__new__(LocalBrainController)
    ctrl._current_runtime_mode = "FAST"  # type: ignore[attr-defined]
    ctrl._last_mode_info = {}  # type: ignore[attr-defined]
    ctrl._llm = _FakeLLM()  # type: ignore[attr-defined]

    assert ctrl.request_profile_demote(reason="second_call") is False
    assert ctrl._llm.shutdown_calls == 0  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 6. voice_debug default must be OFF in shipped settings.json
# ---------------------------------------------------------------------------


def test_voice_debug_default_is_off() -> None:
    import json

    cfg_path = _REPO_ROOT / "config" / "settings.json"
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    voice_debug = raw.get("stt", {}).get("voice_debug")
    assert voice_debug is False, (
        "Shipped settings.json must default voice_debug=false so production "
        "logs don't drown in STT partials / tap-feed debug lines."
    )
