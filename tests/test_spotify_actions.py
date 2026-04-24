"""
ATOM -- regression suite for ``core.router.spotify_actions`` (Phase F1).

Tests are 100% offline: every public path goes through ``_run_osascript``
which is monkeypatched to return scripted ``(ok, stdout)`` tuples. No
real Spotify, no real subprocess, no real macOS -- so they pass on
CI Linux runners as well.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from core.router import spotify_actions


# ── helpers ──────────────────────────────────────────────────────────


def _install_runner(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[str], tuple[bool, str]],
) -> list[str]:
    """Replace ``_run_osascript`` with ``handler`` and record calls."""
    calls: list[str] = []

    def _fake(script: str, *, timeout_s: float = 4.0) -> tuple[bool, str]:
        calls.append(script)
        return handler(script)

    monkeypatch.setattr(spotify_actions, "_run_osascript", _fake)
    monkeypatch.setattr(spotify_actions, "_IS_MAC", True)
    return calls


def _force_mac(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(spotify_actions, "_IS_MAC", True)


# ── is_spotify_running ───────────────────────────────────────────────


def test_is_spotify_running_returns_true_when_process_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_runner(monkeypatch, lambda _s: (True, "true"))
    assert spotify_actions.is_spotify_running() is True


def test_is_spotify_running_returns_false_on_false_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_runner(monkeypatch, lambda _s: (True, "false"))
    assert spotify_actions.is_spotify_running() is False


def test_is_spotify_running_returns_false_on_runner_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_runner(monkeypatch, lambda _s: (False, "boom"))
    assert spotify_actions.is_spotify_running() is False


def test_is_spotify_running_returns_false_on_non_darwin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(spotify_actions, "_IS_MAC", False)
    assert spotify_actions.is_spotify_running() is False


# ── ensure_spotify_running ───────────────────────────────────────────


def test_ensure_spotify_running_short_circuits_when_already_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the app is already running we must NOT issue an activate."""
    calls = _install_runner(monkeypatch, lambda _s: (True, "true"))
    assert spotify_actions.ensure_spotify_running() is True
    assert all("activate" not in c for c in calls)


def test_ensure_spotify_running_launches_then_polls_until_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First check says "false", we send activate, next check says "true"."""
    sequence = iter([
        (True, "false"),  # initial is_running -> not yet
        (True, ""),       # activate Spotify -> ok
        (True, "true"),   # poll says alive
    ])

    def _handler(_s: str) -> tuple[bool, str]:
        return next(sequence)

    _install_runner(monkeypatch, _handler)
    monkeypatch.setattr(spotify_actions.time, "sleep", lambda _s: None)
    assert spotify_actions.ensure_spotify_running(launch_timeout_s=0.5) is True


def test_ensure_spotify_running_returns_false_when_launch_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence = iter([
        (True, "false"),     # is_running -> no
        (False, "denied"),   # activate -> failed
    ])

    def _handler(_s: str) -> tuple[bool, str]:
        return next(sequence)

    _install_runner(monkeypatch, _handler)
    monkeypatch.setattr(spotify_actions.time, "sleep", lambda _s: None)
    assert spotify_actions.ensure_spotify_running(launch_timeout_s=0.2) is False


def test_ensure_spotify_running_times_out_when_app_never_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The poll loop should give up after ``launch_timeout_s``."""
    answers: list[tuple[bool, str]] = [(True, "false"),  # initial
                                       (True, "")]      # activate
    answers.extend([(True, "false")] * 20)
    sequence = iter(answers)

    def _handler(_s: str) -> tuple[bool, str]:
        try:
            return next(sequence)
        except StopIteration:
            return (True, "false")

    _install_runner(monkeypatch, _handler)
    monkeypatch.setattr(spotify_actions.time, "sleep", lambda _s: None)
    fake_now = iter([0.0, 0.05, 0.1, 0.2, 0.4, 0.6])

    def _now() -> float:
        try:
            return next(fake_now)
        except StopIteration:
            return 99.0

    monkeypatch.setattr(spotify_actions.time, "monotonic", _now)
    assert spotify_actions.ensure_spotify_running(launch_timeout_s=0.3) is False


# ── transport verbs ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fn, expected_substring",
    [
        (spotify_actions.play, " to play"),
        (spotify_actions.pause, " to pause"),
        (spotify_actions.next_track, "next track"),
        (spotify_actions.previous_track, "previous track"),
    ],
)
def test_transport_verbs_send_correct_applescript(
    monkeypatch: pytest.MonkeyPatch,
    fn: Callable[[], bool],
    expected_substring: str,
) -> None:
    def _handler(script: str) -> tuple[bool, str]:
        # Pretend Spotify is already running so ensure_spotify_running()
        # short-circuits and we go straight to the transport verb.
        if "processes" in script:
            return True, "true"
        return True, ""

    calls = _install_runner(monkeypatch, _handler)
    assert fn() is True
    transport = [c for c in calls if expected_substring in c]
    assert transport, (
        f"expected an AppleScript containing '{expected_substring}', "
        f"got {calls!r}"
    )


def test_transport_verbs_return_false_on_runner_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failures in osascript propagate as False, not raised exceptions."""

    def _handler(script: str) -> tuple[bool, str]:
        if "processes" in script:  # is_running probe
            return True, "true"
        return False, "kaboom"

    _install_runner(monkeypatch, _handler)
    assert spotify_actions.play() is False
    assert spotify_actions.pause() is False
    assert spotify_actions.next_track() is False
    assert spotify_actions.previous_track() is False


def test_transport_verbs_auto_launch_when_not_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When Spotify is dead, ``play()`` should activate then resume."""
    state = {"running": False, "activated": False}

    def _handler(script: str) -> tuple[bool, str]:
        if "processes" in script:
            return True, ("true" if state["running"] else "false")
        if "activate" in script:
            state["activated"] = True
            state["running"] = True
            return True, ""
        if script.endswith('to play'):
            return True, ""
        return True, ""

    _install_runner(monkeypatch, _handler)
    monkeypatch.setattr(spotify_actions.time, "sleep", lambda _s: None)
    assert spotify_actions.play() is True
    assert state["activated"] is True


# ── current_track ────────────────────────────────────────────────────


def test_current_track_parses_full_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = "Bohemian Rhapsody\tQueen\tA Night at the Opera\t354320\t42.5\tplaying"

    def _handler(script: str) -> tuple[bool, str]:
        if "processes" in script:
            return True, "true"
        return True, payload

    _install_runner(monkeypatch, _handler)
    track = spotify_actions.current_track()
    assert track == {
        "name": "Bohemian Rhapsody",
        "artist": "Queen",
        "album": "A Night at the Opera",
        "duration_s": 354.32,
        "position_s": 42.5,
        "state": "playing",
    }


def test_current_track_returns_none_when_spotify_dead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_runner(monkeypatch, lambda _s: (True, "false"))
    assert spotify_actions.current_track() is None


def test_current_track_returns_none_on_blank_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If Spotify is alive but nothing is loaded, the script returns ""."""

    def _handler(script: str) -> tuple[bool, str]:
        if "processes" in script:
            return True, "true"
        return True, ""

    _install_runner(monkeypatch, _handler)
    assert spotify_actions.current_track() is None


def test_current_track_returns_none_on_malformed_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _handler(script: str) -> tuple[bool, str]:
        if "processes" in script:
            return True, "true"
        return True, "only\ttwo\tparts"

    _install_runner(monkeypatch, _handler)
    assert spotify_actions.current_track() is None


# ── play_search ──────────────────────────────────────────────────────


def test_play_search_builds_search_uri_for_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def _handler(script: str) -> tuple[bool, str]:
        captured.append(script)
        if "processes" in script:
            return True, "true"
        return True, ""

    _install_runner(monkeypatch, _handler)
    assert spotify_actions.play_search("Bohemian Rhapsody") is True
    play_calls = [c for c in captured if "play track" in c]
    assert play_calls, captured
    assert "spotify:search:track:Bohemian Rhapsody" in play_calls[0]


def test_play_search_supports_album_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def _handler(script: str) -> tuple[bool, str]:
        captured.append(script)
        if "processes" in script:
            return True, "true"
        return True, ""

    _install_runner(monkeypatch, _handler)
    assert spotify_actions.play_search("OK Computer", kind="album") is True
    play_calls = [c for c in captured if "play track" in c]
    assert "spotify:search:album:OK Computer" in play_calls[0]


def test_play_search_falls_back_to_track_for_invalid_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def _handler(script: str) -> tuple[bool, str]:
        captured.append(script)
        if "processes" in script:
            return True, "true"
        return True, ""

    _install_runner(monkeypatch, _handler)
    assert spotify_actions.play_search("In Bloom", kind="potato") is True
    play_calls = [c for c in captured if "play track" in c]
    assert "spotify:search:track:In Bloom" in play_calls[0]


def test_play_search_returns_false_on_blank_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_mac(monkeypatch)
    monkeypatch.setattr(spotify_actions, "_run_osascript",
                        lambda *_a, **_kw: (True, ""))
    assert spotify_actions.play_search("   ") is False
    assert spotify_actions.play_search("") is False


def test_play_search_strips_double_quotes_from_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quotes inside the user's query must not break out of the script."""
    captured: list[str] = []

    def _handler(script: str) -> tuple[bool, str]:
        captured.append(script)
        if "processes" in script:
            return True, "true"
        return True, ""

    _install_runner(monkeypatch, _handler)
    assert spotify_actions.play_search('"); do something evil; tell app "Spotify') is True
    play_calls = [c for c in captured if "play track" in c]
    assert play_calls
    # Isolate the URI literal (everything between the URI's wrapping quotes)
    # and assert that no extra double-quote characters survived inside it.
    script = play_calls[0]
    uri_literal = script.split('play track "', 1)[1].rstrip('"')
    assert '"' not in uri_literal, uri_literal


def test_play_search_returns_false_when_spotify_wont_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence = iter([
        (True, "false"),    # is_running
        (False, "denied"),  # activate
    ])

    def _handler(_s: str) -> tuple[bool, str]:
        return next(sequence)

    _install_runner(monkeypatch, _handler)
    monkeypatch.setattr(spotify_actions.time, "sleep", lambda _s: None)
    assert spotify_actions.play_search("Whatever") is False


# ── diagnostics ──────────────────────────────────────────────────────


def test_diagnostics_includes_track_when_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = "X\tY\tZ\t180000\t10\tplaying"

    def _handler(script: str) -> tuple[bool, str]:
        if "processes" in script:
            return True, "true"
        return True, payload

    _install_runner(monkeypatch, _handler)
    snap = spotify_actions.diagnostics()
    assert snap["available"] is True
    assert snap["running"] is True
    assert snap["track"]["name"] == "X"
    assert snap["track"]["state"] == "playing"


def test_diagnostics_marks_unavailable_off_mac(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(spotify_actions, "_IS_MAC", False)
    snap = spotify_actions.diagnostics()
    assert snap == {"available": False, "reason": "non-darwin"}


# ── runner contract ──────────────────────────────────────────────────


def test_runner_returns_false_on_non_darwin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real ``_run_osascript`` must short-circuit when not on macOS."""
    monkeypatch.setattr(spotify_actions, "_IS_MAC", False)
    ok, out = spotify_actions._run_osascript("does not matter")
    assert ok is False
    assert out == "non-darwin"


def test_runner_treats_missing_value_output_as_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``missing value`` is Spotify's idiom for "I can't answer that"."""

    class _FakeProc:
        def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def _fake_run(*_a: Any, **_kw: Any) -> _FakeProc:
        return _FakeProc(0, stdout="missing value\n")

    monkeypatch.setattr(spotify_actions, "_IS_MAC", True)
    monkeypatch.setattr(spotify_actions.subprocess, "run", _fake_run)
    ok, out = spotify_actions._run_osascript("ignored")
    assert ok is False
    assert out == "missing value"


def test_runner_returns_false_on_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeProc:
        returncode = 1
        stdout = ""
        stderr = "Application isn't running.\n"

    def _fake_run(*_a: Any, **_kw: Any) -> _FakeProc:
        return _FakeProc()

    monkeypatch.setattr(spotify_actions, "_IS_MAC", True)
    monkeypatch.setattr(spotify_actions.subprocess, "run", _fake_run)
    ok, err = spotify_actions._run_osascript("ignored")
    assert ok is False
    assert "Application isn't running" in err


def test_runner_handles_timeout_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(*_a: Any, **_kw: Any) -> Any:
        raise spotify_actions.subprocess.TimeoutExpired(cmd=["osascript"], timeout=1.0)

    monkeypatch.setattr(spotify_actions, "_IS_MAC", True)
    monkeypatch.setattr(spotify_actions.subprocess, "run", _raise)
    ok, out = spotify_actions._run_osascript("ignored", timeout_s=1.0)
    assert ok is False
    assert out == "timeout"


def test_runner_handles_missing_osascript_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(*_a: Any, **_kw: Any) -> Any:
        raise FileNotFoundError("osascript missing")

    monkeypatch.setattr(spotify_actions, "_IS_MAC", True)
    monkeypatch.setattr(spotify_actions.subprocess, "run", _raise)
    ok, out = spotify_actions._run_osascript("ignored")
    assert ok is False
    assert out == "no-osascript"
