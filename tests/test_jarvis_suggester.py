"""Regression tests for the Phase G5 Jarvis suggester."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from core.cognitive.jarvis_suggester import (
    JarvisSuggester,
    SuggestionCandidate,
    default_candidates,
)


# ── fakes ──────────────────────────────────────────────────────────


class _FakeBus:
    def __init__(self) -> None:
        self.subs: dict[str, list] = {}
        self.emitted_long: list[tuple[str, dict[str, Any]]] = []

    def on(self, event: str, handler) -> None:
        self.subs.setdefault(event, []).append(handler)

    def off(self, event: str, handler) -> None:
        if event in self.subs and handler in self.subs[event]:
            self.subs[event].remove(handler)

    def emit_long(self, event: str, **data: Any) -> None:
        self.emitted_long.append((event, data))

    async def fire(self, event: str, **data: Any) -> None:
        for fn in list(self.subs.get(event, [])):
            await fn(**data)


def _always_strong_candidate(**_kw: Any) -> list[SuggestionCandidate]:
    return [SuggestionCandidate(
        text="Boss, want me to dim the screen?",
        category="wind_down",
        score=0.95,
        rationale="test override",
    )]


def _empty_candidates(**_kw: Any) -> list[SuggestionCandidate]:
    return []


def _weak_candidate(**_kw: Any) -> list[SuggestionCandidate]:
    return [SuggestionCandidate(text="Maybe?", category="x", score=0.3)]


def _make(bus: _FakeBus, **overrides: Any) -> JarvisSuggester:
    kwargs: dict[str, Any] = dict(
        candidate_provider=_always_strong_candidate,
        cooldown_s=0.0,
        category_cooldown_s=0.0,
        daily_cap=10,
        quiet_hours=(0, 0),  # disabled
        relevance_threshold=0.7,
        suppress_moods=("frustrated", "focused", "idle"),
    )
    kwargs.update(overrides)
    s = JarvisSuggester(bus, **kwargs)
    s.attach()
    return s


# ── default_candidates pure function ───────────────────────────────


def test_default_candidates_returns_wind_down_when_tired_and_late() -> None:
    out = default_candidates(
        mood="tired", session_minutes=30,
        hour_of_day=23, presence_present=True,
    )
    cats = {c.category for c in out}
    assert "wind_down" in cats


def test_default_candidates_returns_break_when_long_session() -> None:
    out = default_candidates(
        mood="tired", session_minutes=120,
        hour_of_day=14, presence_present=True,
    )
    cats = {c.category for c in out}
    assert "break_suggest" in cats


def test_default_candidates_offers_focus_when_engaged_and_short() -> None:
    out = default_candidates(
        mood="engaged", session_minutes=10,
        hour_of_day=10, presence_present=True,
    )
    assert any(c.category == "focus_offer" for c in out)


def test_default_candidates_returns_empty_for_focused_user() -> None:
    out = default_candidates(
        mood="focused", session_minutes=20,
        hour_of_day=11, presence_present=True,
    )
    assert out == []


# ── basic emission ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_emits_suggestion_when_all_gates_pass() -> None:
    bus = _FakeBus()
    s = _make(bus)
    await bus.fire("mood.state", mood="tired")
    assert any(evt == "response_ready" for evt, _ in bus.emitted_long)
    payload = next(d for evt, d in bus.emitted_long if evt == "response_ready")
    assert payload["proactive"] is True
    assert payload["source"] == "jarvis_suggester"
    assert payload["category"] == "wind_down"
    assert s.metrics["emits"] == 1


@pytest.mark.asyncio
async def test_skipped_when_mood_in_suppress_set() -> None:
    bus = _FakeBus()
    s = _make(bus)
    for mood in ("frustrated", "focused", "idle"):
        await bus.fire("mood.state", mood=mood)
    assert bus.emitted_long == []
    assert s.metrics["emits"] == 0


@pytest.mark.asyncio
async def test_skipped_when_no_candidates() -> None:
    bus = _FakeBus()
    s = _make(bus, candidate_provider=_empty_candidates)
    await bus.fire("mood.state", mood="tired")
    assert bus.emitted_long == []
    assert s.metrics["emits"] == 0


@pytest.mark.asyncio
async def test_skipped_when_score_below_threshold() -> None:
    bus = _FakeBus()
    s = _make(bus, candidate_provider=_weak_candidate)
    await bus.fire("mood.state", mood="tired")
    assert bus.emitted_long == []
    assert s.metrics["emits"] == 0


# ── cooldown gates ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_global_cooldown_prevents_back_to_back_emits() -> None:
    bus = _FakeBus()
    s = _make(bus, cooldown_s=10.0)
    await bus.fire("mood.state", mood="tired")
    await bus.fire("mood.state", mood="tired")
    assert s.metrics["emits"] == 1


@pytest.mark.asyncio
async def test_category_cooldown_prevents_repeat_category() -> None:
    bus = _FakeBus()
    s = _make(bus, cooldown_s=0.0, category_cooldown_s=60.0)
    await bus.fire("mood.state", mood="tired")
    await bus.fire("mood.state", mood="tired")
    assert s.metrics["emits"] == 1


@pytest.mark.asyncio
async def test_daily_cap_locks_after_threshold() -> None:
    bus = _FakeBus()
    s = _make(bus, daily_cap=2, category_cooldown_s=0.0)
    for _ in range(5):
        await bus.fire("mood.state", mood="tired")
    assert s.metrics["emits"] == 2


# ── quiet hours ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_quiet_hours_suppress_emission(monkeypatch: pytest.MonkeyPatch) -> None:
    import datetime as dt
    import core.cognitive.jarvis_suggester as mod

    class _FakeDT(dt.datetime):
        @classmethod
        def now(cls, tz: Any = None) -> "dt.datetime":
            return dt.datetime(2026, 1, 1, 1, 30, 0)

    monkeypatch.setattr(mod._dt, "datetime", _FakeDT)
    bus = _FakeBus()
    s = _make(bus, quiet_hours=(23, 6))
    await bus.fire("mood.state", mood="tired")
    assert bus.emitted_long == []
    assert s.metrics["blocked"] >= 1


@pytest.mark.asyncio
async def test_quiet_hours_disabled_when_equal() -> None:
    bus = _FakeBus()
    s = _make(bus, quiet_hours=(0, 0))
    await bus.fire("mood.state", mood="tired")
    assert s.metrics["emits"] == 1


# ── presence + trace signal updates ───────────────────────────────


@pytest.mark.asyncio
async def test_presence_event_updates_internal_state() -> None:
    bus = _FakeBus()
    seen: list[bool | None] = []

    def grabber(**kw: Any) -> list[SuggestionCandidate]:
        seen.append(kw.get("presence_present"))
        return [SuggestionCandidate("hi", "x", 0.95)]

    s = _make(bus, candidate_provider=grabber)
    await bus.fire("presence.snapshot", present=True, face_count=1, quality="good")
    await bus.fire("mood.state", mood="tired")
    assert seen and seen[-1] is True


@pytest.mark.asyncio
async def test_command_trace_updates_last_user_chars() -> None:
    bus = _FakeBus()
    seen_chars: list[int | None] = []

    def grabber(**kw: Any) -> list[SuggestionCandidate]:
        seen_chars.append(kw.get("last_user_chars"))
        return [SuggestionCandidate("hi", "x", 0.95)]

    s = _make(bus, candidate_provider=grabber)
    await bus.fire("command_loop_trace", stage="start", text="hello there ATOM")
    await bus.fire("mood.state", mood="tired")
    assert seen_chars and seen_chars[-1] == len("hello there ATOM")


# ── response_emitter override path ────────────────────────────────


@pytest.mark.asyncio
async def test_custom_response_emitter_short_circuits_bus() -> None:
    bus = _FakeBus()
    captured: list[str] = []

    def emitter(text: str) -> None:
        captured.append(text)

    s = _make(bus, response_emitter=emitter)
    await bus.fire("mood.state", mood="tired")
    assert captured and "dim the screen" in captured[0]
    # Bus path should *not* be used when emitter is provided.
    assert not any(evt == "response_ready" for evt, _ in bus.emitted_long)


@pytest.mark.asyncio
async def test_emitter_failure_falls_back_to_bus() -> None:
    bus = _FakeBus()

    def boom(_: str) -> None:
        raise RuntimeError("boom")

    s = _make(bus, response_emitter=boom)
    await bus.fire("mood.state", mood="tired")
    assert any(evt == "response_ready" for evt, _ in bus.emitted_long)


# ── attach/detach + metrics ───────────────────────────────────────


@pytest.mark.asyncio
async def test_attach_idempotent_and_detach_clears_handlers() -> None:
    bus = _FakeBus()
    s = JarvisSuggester(bus, candidate_provider=_always_strong_candidate)
    s.attach()
    s.attach()
    assert len(bus.subs.get("mood.state", [])) == 1
    s.detach()
    assert bus.subs.get("mood.state", []) == []


@pytest.mark.asyncio
async def test_provider_exception_does_not_leak() -> None:
    bus = _FakeBus()

    def crashy(**_kw: Any) -> list[SuggestionCandidate]:
        raise RuntimeError("oops")

    s = _make(bus, candidate_provider=crashy)
    await bus.fire("mood.state", mood="tired")
    assert s.metrics["emits"] == 0
    assert s.metrics["blocked"] >= 1


def test_session_minutes_is_monotonic_nonnegative() -> None:
    bus = _FakeBus()
    s = _make(bus)
    m = s.metrics["session_minutes"]
    assert m >= 0.0


def test_reset_session_resets_clock() -> None:
    bus = _FakeBus()
    s = _make(bus)
    s._session_started_at -= 600  # type: ignore[attr-defined]
    assert s.metrics["session_minutes"] >= 10.0
    s.reset_session()
    assert s.metrics["session_minutes"] < 1.0
