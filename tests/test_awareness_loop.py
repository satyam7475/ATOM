"""Sprint F1 -- continuous awareness loop fusion."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from core.cognitive.awareness_loop import AwarenessConfig, AwarenessLoop


pytestmark = pytest.mark.asyncio


class _StubBus:
    def __init__(self) -> None:
        self.handlers: dict[str, list[Any]] = {}
        self.emitted: list[tuple[str, dict[str, Any]]] = []

    def on(self, event: str, handler: Any) -> None:
        self.handlers.setdefault(event, []).append(handler)

    def off(self, event: str, handler: Any) -> None:
        try:
            self.handlers.get(event, []).remove(handler)
        except ValueError:
            pass

    def emit_fast(self, event: str, **payload: Any) -> None:
        self.emitted.append((event, payload))

    async def fire(self, event: str, **payload: Any) -> None:
        for h in list(self.handlers.get(event, [])):
            await h(**payload)


class _StubSuggester:
    def __init__(self, *, accept: bool = True) -> None:
        self.calls: list[tuple[list, str]] = []
        self.accept = accept

    async def consider_candidates(self, candidates: list, *, reason: str = "") -> bool:
        self.calls.append((candidates, reason))
        return self.accept


# ── presence transitions ──────────────────────────────────────────


async def test_welcome_back_after_long_absence_emits_directly_when_no_suggester() -> None:
    bus = _StubBus()
    loop = AwarenessLoop(
        bus,
        config=AwarenessConfig(welcome_back_after_absent_s=1.0, min_emit_gap_s=0.1),
    )
    loop.attach()

    # Simulate "absent" two seconds ago.
    loop._snapshot.last_seen_absent_at = time.time() - 2.0
    loop._snapshot.presence_present = False

    await bus.fire("presence.snapshot", present=True)

    welcomes = [e for e in bus.emitted if e[0] == "response_ready"]
    assert welcomes, f"expected a response_ready emission, got {bus.emitted}"
    assert "Welcome back" in welcomes[0][1]["text"]
    assert welcomes[0][1]["source"] == "awareness_loop"


async def test_welcome_back_routes_through_suggester_when_present() -> None:
    bus = _StubBus()
    sug = _StubSuggester()
    loop = AwarenessLoop(
        bus, suggester=sug,
        config=AwarenessConfig(welcome_back_after_absent_s=1.0, min_emit_gap_s=0.1),
    )
    loop.attach()
    loop._snapshot.last_seen_absent_at = time.time() - 2.0
    loop._snapshot.presence_present = False

    await bus.fire("presence.snapshot", present=True)

    assert sug.calls, "suggester.consider_candidates should be called"
    candidates, reason = sug.calls[0]
    assert candidates and "Welcome back" in candidates[0].text
    assert reason == "welcome_back"


async def test_short_absence_does_not_trigger_welcome() -> None:
    bus = _StubBus()
    loop = AwarenessLoop(bus, config=AwarenessConfig(welcome_back_after_absent_s=300.0))
    loop.attach()
    loop._snapshot.last_seen_absent_at = time.time() - 5.0
    loop._snapshot.presence_present = False
    await bus.fire("presence.snapshot", present=True)
    assert not [e for e in bus.emitted if e[0] == "response_ready"]


# ── silent-present nudge ───────────────────────────────────────────


async def test_silent_present_emits_through_suggester_after_threshold() -> None:
    bus = _StubBus()
    sug = _StubSuggester()
    loop = AwarenessLoop(
        bus, suggester=sug,
        config=AwarenessConfig(silent_present_warn_s=1.0, min_emit_gap_s=0.05),
    )
    loop.attach()
    loop._snapshot.presence_present = True
    loop._snapshot.last_user_speech_at = time.time() - 2.0

    await bus.fire("mood.state", mood="engaged")

    assert sug.calls, "silent-present should push a candidate"
    candidates, reason = sug.calls[0]
    assert candidates[0].category == "awareness.silent_present"
    assert reason == "silent_present"


async def test_silent_present_skips_when_mood_idle() -> None:
    bus = _StubBus()
    sug = _StubSuggester()
    loop = AwarenessLoop(
        bus, suggester=sug,
        config=AwarenessConfig(silent_present_warn_s=1.0, min_emit_gap_s=0.05),
    )
    loop.attach()
    loop._snapshot.presence_present = True
    loop._snapshot.last_user_speech_at = time.time() - 2.0
    await bus.fire("mood.state", mood="idle")
    assert not sug.calls


# ── scene dwell ────────────────────────────────────────────────────


async def test_scene_dwell_triggers_break_offer() -> None:
    bus = _StubBus()
    sug = _StubSuggester()
    loop = AwarenessLoop(
        bus, suggester=sug,
        config=AwarenessConfig(scene_dwell_warn_s=1.0, min_emit_gap_s=0.05),
    )
    loop.attach()
    loop._snapshot.presence_present = True

    await bus.fire("scene.context", label="Visual Studio Code")
    # First scene event sets the timer; second one (same label) triggers dwell.
    loop._snapshot.scene_changed_at = time.time() - 2.0
    await bus.fire("scene.context", label="Visual Studio Code")

    assert sug.calls, "scene-dwell should push a candidate after threshold"
    candidates, _ = sug.calls[0]
    assert "heads-down" in candidates[0].text
    assert candidates[0].category.startswith("awareness.scene_dwell")


# ── snapshot --------------------------------------------------------


async def test_snapshot_includes_metrics_and_state() -> None:
    bus = _StubBus()
    loop = AwarenessLoop(bus)
    loop.attach()
    await bus.fire("speech_final", text="hello")
    await bus.fire("response_ready", text="hi Boss")
    snap = loop.snapshot
    assert snap["last_user_speech_age_s"] is not None
    assert snap["last_atom_speech_age_s"] is not None
    assert "metrics" in snap
