"""
ATOM -- Phase G4 regression suite for mood inference.

Pins:
  * Pure ``infer_mood`` returns the right mood for canonical signal
    bundles (focused, engaged, frustrated, tired, idle, distracted,
    unknown).
  * The engine emits ``mood.state`` only when the *category* changes
    AND the hysteresis streak is satisfied.
  * Bus subscribers update signals live (presence, sentiment, voice
    metrics, command trace) without raising.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.cognitive.mood_inference import (
    MoodInferenceEngine,
    MoodResult,
    MoodSignals,
    VALID_MOODS,
    infer_mood,
)


# ── pure inference ─────────────────────────────────────────────────


def test_no_signals_returns_unknown() -> None:
    out = infer_mood(MoodSignals())
    assert out.mood == "unknown"
    assert out.confidence == 0.0


def test_no_face_returns_idle() -> None:
    out = infer_mood(MoodSignals(presence_present=False, presence_face_count=0))
    assert out.mood == "idle"
    assert "no face" in " ".join(out.rationale).lower()


def test_face_with_good_quality_skews_focused() -> None:
    out = infer_mood(MoodSignals(
        presence_present=True, presence_face_count=1,
        presence_quality="good", hour_of_day=10,
    ))
    assert out.mood in {"focused", "engaged"}


def test_negative_sentiment_yields_frustrated() -> None:
    out = infer_mood(MoodSignals(
        presence_present=True, presence_face_count=1,
        presence_quality="good", sentiment="negative",
        repeat_count=2,
    ))
    assert out.mood == "frustrated"


def test_late_hour_pushes_tired() -> None:
    out = infer_mood(MoodSignals(
        presence_present=True, presence_face_count=1,
        presence_quality="good", hour_of_day=2,
        session_minutes=120,
    ))
    assert out.mood == "tired"


def test_multiple_faces_yields_distracted() -> None:
    out = infer_mood(MoodSignals(
        presence_present=True, presence_face_count=3,
        presence_quality="good",
    ))
    assert out.mood == "distracted"


def test_positive_sentiment_promotes_engaged() -> None:
    out = infer_mood(MoodSignals(
        presence_present=True, presence_face_count=1,
        presence_quality="good", sentiment="positive",
        last_user_chars=120,
    ))
    assert out.mood == "engaged"


def test_low_face_quality_pushes_distracted() -> None:
    out = infer_mood(MoodSignals(
        presence_present=True, presence_face_count=1,
        presence_quality="low_light",
    ))
    assert out.mood == "distracted"


def test_valid_moods_constant_pinned() -> None:
    assert "focused" in VALID_MOODS
    assert "frustrated" in VALID_MOODS
    assert "tired" in VALID_MOODS


def test_mood_result_as_dict_round_trip() -> None:
    r = MoodResult(mood="focused", confidence=0.42, rationale=["x"], ts=1.0)
    d = r.as_dict()
    assert d["mood"] == "focused"
    assert d["confidence"] == 0.42
    assert d["rationale"] == ["x"]
    assert d["ts"] == 1.0


# ── engine + hysteresis ───────────────────────────────────────────


class _FakeBus:
    def __init__(self) -> None:
        self.handlers: dict[str, list] = {}
        self.emitted_long: list[tuple[str, dict]] = []

    def on(self, event: str, handler: Any) -> None:
        self.handlers.setdefault(event, []).append(handler)

    def off(self, event: str, handler: Any) -> None:
        try:
            self.handlers[event].remove(handler)
        except (KeyError, ValueError):
            pass

    def emit_long(self, event: str, **payload: Any) -> None:
        self.emitted_long.append((event, payload))

    def emit_fast(self, *_a: Any, **_kw: Any) -> None: ...

    async def fire(self, event: str, **payload: Any) -> None:
        for h in list(self.handlers.get(event, ())):
            await h(**payload)


@pytest.mark.asyncio
async def test_engine_does_not_emit_until_streak_satisfied() -> None:
    bus = _FakeBus()
    engine = MoodInferenceEngine(bus, min_consecutive=2)
    engine.attach()
    await bus.fire("presence.snapshot",
                   present=True, face_count=1, quality="good")
    # First sample: streak count = 1, no emit.
    assert bus.emitted_long == []
    await bus.fire("presence.snapshot",
                   present=True, face_count=1, quality="good")
    # Second sample of the same mood: streak satisfied, emit.
    assert any(evt == "mood.state" for evt, _ in bus.emitted_long)


@pytest.mark.asyncio
async def test_engine_does_not_re_emit_same_mood() -> None:
    bus = _FakeBus()
    engine = MoodInferenceEngine(bus, min_consecutive=1)
    engine.attach()
    for _ in range(4):
        await bus.fire("presence.snapshot",
                       present=False, face_count=0, quality="no_face")
    moods = [p["mood"] for evt, p in bus.emitted_long if evt == "mood.state"]
    assert moods.count("idle") == 1
    assert engine.current_mood == "idle"


@pytest.mark.asyncio
async def test_engine_emits_on_mood_change() -> None:
    """Sprint K7: 'no face' is now debounced (3 samples + 90 s before
    we declare 'idle'), so a single absent snapshot followed by an
    engaged one should NOT flap to 'idle' mid-sentence -- but it MUST
    still emit a mood transition when the visible state changes."""
    bus = _FakeBus()
    engine = MoodInferenceEngine(bus, min_consecutive=1)
    engine.attach()
    await bus.fire("presence.snapshot",
                   present=False, face_count=0, quality="no_face")
    await bus.fire("presence.snapshot",
                   present=True, face_count=3, quality="good")
    moods = [p["mood"] for evt, p in bus.emitted_long if evt == "mood.state"]
    assert moods, "mood engine should emit at least one mood.state"
    # No flap to "idle" on the very first absent sample (Sprint K7).
    assert "idle" not in moods, (
        f"single 'no face' sample must not flap to 'idle' "
        f"(K7 debouncer): saw {moods}"
    )
    assert moods[-1] == "distracted"


@pytest.mark.asyncio
async def test_engine_uses_emotion_signal() -> None:
    bus = _FakeBus()
    engine = MoodInferenceEngine(bus, min_consecutive=1)
    engine.attach()
    engine.update_signals(presence_present=True, presence_face_count=1,
                          presence_quality="good")
    await bus.fire("user_emotion_detected", emotion="negative")
    engine.update_signals(repeat_count=2)
    await bus.fire("user_emotion_detected", emotion="negative")
    assert any(p["mood"] == "frustrated"
               for evt, p in bus.emitted_long if evt == "mood.state")


@pytest.mark.asyncio
async def test_engine_uses_command_trace_for_user_chars() -> None:
    bus = _FakeBus()
    engine = MoodInferenceEngine(bus, min_consecutive=1)
    engine.attach()
    engine.update_signals(presence_present=True, presence_face_count=1,
                          presence_quality="good")
    await bus.fire("command_loop_trace", stage="start",
                   text="x" * 200)
    # Should at least register the chars; mood can be focused/engaged
    assert engine._signals.last_user_chars == 200


@pytest.mark.asyncio
async def test_engine_uses_voice_metrics() -> None:
    bus = _FakeBus()
    engine = MoodInferenceEngine(bus)
    engine.attach()
    await bus.fire("voice_metrics", rms_dbfs=-50.0)
    assert engine._signals.voice_rms_db == -50.0


@pytest.mark.asyncio
async def test_attach_is_idempotent() -> None:
    bus = _FakeBus()
    engine = MoodInferenceEngine(bus)
    engine.attach()
    engine.attach()
    assert len(bus.handlers["presence.snapshot"]) == 1


@pytest.mark.asyncio
async def test_detach_removes_subscribers() -> None:
    bus = _FakeBus()
    engine = MoodInferenceEngine(bus)
    engine.attach()
    engine.detach()
    assert bus.handlers.get("presence.snapshot", []) == []
    assert bus.handlers.get("user_emotion_detected", []) == []


@pytest.mark.asyncio
async def test_engine_metrics_track_streak() -> None:
    bus = _FakeBus()
    engine = MoodInferenceEngine(bus, min_consecutive=2)
    engine.attach()
    await bus.fire("presence.snapshot",
                   present=True, face_count=1, quality="good")
    metrics = engine.metrics
    assert metrics["updates"] >= 1
    assert metrics["streak_count"] >= 1
    assert metrics["current_mood"] in ("unknown", "focused", "engaged")
