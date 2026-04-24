"""
ATOM -- regression for F2 router wiring of Spotify ``music_*`` actions.

We don't spin up a full ``Router`` (it has heavy bus / state / kernel
deps), instead we instantiate a bare router via ``Router.__new__`` and
patch the ``spotify_actions`` module to verify that:

  - All six ``music_*`` actions are present in ``_ACTION_DISPATCH``.
  - Each handler routes to the matching ``spotify_actions`` function.
  - Failed Spotify calls degrade to a personality.error_response()
    string instead of bubbling exceptions.
  - ``music_current`` formats track / artist / state into a friendly
    one-liner (and handles "nothing playing" cleanly).
  - ``music_play_specific`` rejects empty queries and forwards the
    ``kind`` slot to ``play_search``.
  - ``play_youtube`` no longer requires confirmation in the tool
    registry (it was the friction point for casual music control).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.router import spotify_actions
from core.router.router import Router


# ── presence ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "action",
    [
        "music_play",
        "music_pause",
        "music_next",
        "music_prev",
        "music_current",
        "music_play_specific",
    ],
)
def test_music_actions_registered_in_dispatch(action: str) -> None:
    assert action in Router._ACTION_DISPATCH, (
        f"{action} missing from Router._ACTION_DISPATCH -- the router "
        f"will fall through to the LLM instead of calling Spotify."
    )


# ── handler routing ─────────────────────────────────────────────────


def _make_router_stub() -> Router:
    router = Router.__new__(Router)
    router._bus = MagicMock()
    return router


def test_music_play_invokes_spotify_play(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(spotify_actions, "play", lambda: True)
    router = _make_router_stub()
    out = Router._do_music_play(router, "music_play", {})
    assert isinstance(out, str) and out
    router._bus.emit_fast.assert_called_with("media_started")


def test_music_play_returns_error_string_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(spotify_actions, "play", lambda: False)
    router = _make_router_stub()
    out = Router._do_music_play(router, "music_play", {})
    assert isinstance(out, str) and out
    router._bus.emit_fast.assert_not_called()


def test_music_pause_invokes_spotify_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(spotify_actions, "pause",
                        lambda: calls.append("pause") or True)
    router = _make_router_stub()
    out = Router._do_music_pause(router, "music_pause", {})
    assert calls == ["pause"]
    assert isinstance(out, str) and out


def test_music_next_invokes_spotify_next_track(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(spotify_actions, "next_track",
                        lambda: calls.append("next") or True)
    router = _make_router_stub()
    out = Router._do_music_next(router, "music_next", {})
    assert calls == ["next"]
    assert isinstance(out, str) and out


def test_music_prev_invokes_spotify_previous_track(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(spotify_actions, "previous_track",
                        lambda: calls.append("prev") or True)
    router = _make_router_stub()
    out = Router._do_music_prev(router, "music_prev", {})
    assert calls == ["prev"]
    assert isinstance(out, str) and out


# ── current track narrator ──────────────────────────────────────────


def test_music_current_announces_track_name_and_artist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(spotify_actions, "current_track",
                        lambda: {"name": "Bohemian Rhapsody",
                                 "artist": "Queen",
                                 "album": "A Night at the Opera",
                                 "duration_s": 354.32,
                                 "position_s": 12.4,
                                 "state": "playing"})
    router = _make_router_stub()
    out = Router._do_music_current(router, "music_current", {})
    assert "Bohemian Rhapsody" in out
    assert "Queen" in out
    assert out.lower().startswith("playing")


def test_music_current_handles_paused_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(spotify_actions, "current_track",
                        lambda: {"name": "Yesterday",
                                 "artist": "The Beatles",
                                 "album": "Help!",
                                 "duration_s": 125.0,
                                 "position_s": 30.0,
                                 "state": "paused"})
    router = _make_router_stub()
    out = Router._do_music_current(router, "music_current", {})
    assert out.lower().startswith("paused")
    assert "Yesterday" in out


def test_music_current_handles_nothing_playing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(spotify_actions, "current_track", lambda: None)
    router = _make_router_stub()
    out = Router._do_music_current(router, "music_current", {})
    assert "nothing" in out.lower() or "not playing" in out.lower()


# ── play_specific ────────────────────────────────────────────────────


def test_music_play_specific_forwards_query_and_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def _fake_search(query: str, *, kind: str = "track") -> bool:
        captured["query"] = query
        captured["kind"] = kind
        return True

    monkeypatch.setattr(spotify_actions, "play_search", _fake_search)
    # Force full verbosity so the music template renders the song name
    # (otherwise minimal mode collapses to "Done." and we can't assert).
    from core import adaptive_personality
    monkeypatch.setattr(adaptive_personality, "_verbosity", lambda: "full")
    monkeypatch.setattr(adaptive_personality, "_emotion", lambda: "neutral")
    router = _make_router_stub()
    out = Router._do_music_play_specific(
        router, "music_play_specific",
        {"query": "Smells Like Teen Spirit", "kind": "track"},
    )
    assert captured == {"query": "Smells Like Teen Spirit", "kind": "track"}
    assert "Smells Like Teen Spirit" in out
    router._bus.emit_fast.assert_called_with("media_started")


def test_music_play_specific_defaults_kind_to_track(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def _fake_search(query: str, *, kind: str = "track") -> bool:
        captured["query"] = query
        captured["kind"] = kind
        return True

    monkeypatch.setattr(spotify_actions, "play_search", _fake_search)
    router = _make_router_stub()
    Router._do_music_play_specific(
        router, "music_play_specific", {"query": "Despacito"},
    )
    assert captured["kind"] == "track"


def test_music_play_specific_rejects_empty_query() -> None:
    router = _make_router_stub()
    out = Router._do_music_play_specific(
        router, "music_play_specific", {"query": "   "},
    )
    assert isinstance(out, str) and out
    assert "song" in out.lower() or "didn't" in out.lower()
    router._bus.emit_fast.assert_not_called()


def test_music_play_specific_returns_error_string_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(spotify_actions, "play_search",
                        lambda *_a, **_kw: False)
    router = _make_router_stub()
    out = Router._do_music_play_specific(
        router, "music_play_specific", {"query": "X"},
    )
    assert isinstance(out, str) and out
    router._bus.emit_fast.assert_not_called()


# ── tool-registry confirmation policy (F5 prep) ─────────────────────


def test_play_youtube_no_longer_requires_confirmation() -> None:
    """Casual music control was blocked by the YouTube confirmation
    prompt (atom_log L325). Phase F flips this to safe-by-default."""
    from core.reasoning.tool_registry import get_tool_registry
    tool = get_tool_registry().get("play_youtube")
    assert tool is not None
    assert tool.requires_confirmation is False, (
        "play_youtube must NOT require confirmation -- it's the "
        "casual-music fallback path."
    )


@pytest.mark.parametrize(
    "name",
    [
        "music_play",
        "music_pause",
        "music_next",
        "music_prev",
        "music_current",
        "music_play_specific",
    ],
)
def test_music_tools_are_registered_safely(name: str) -> None:
    from core.reasoning.tool_registry import get_tool_registry
    tool = get_tool_registry().get(name)
    assert tool is not None, f"{name} missing from tool registry"
    assert tool.safety_level == "safe"
    assert tool.requires_confirmation is False
    assert tool.category == "media"


# ── security policy ──────────────────────────────────────────────────


def test_music_actions_are_safe_always_intents() -> None:
    """Music transport must skip rate-limit and lock-mode gates so it
    works during heavy use (e.g. mid-conversation 'pause music')."""
    from core.security_policy import _SAFE_ALWAYS_INTENTS
    for action in ("music_play", "music_pause", "music_next",
                   "music_prev", "music_current", "music_play_specific"):
        assert action in _SAFE_ALWAYS_INTENTS, (
            f"{action} missing from _SAFE_ALWAYS_INTENTS -- it will "
            f"be rate-limited and blocked under restricted lock mode."
        )
