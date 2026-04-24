"""Focused tests for VoicePipeline.describe_on_wake safeguards.

Exercises the three guards that ``_schedule_describe_on_wake`` must
honour so the on-wake VLM dispatch never overloads the M5 Air:

1. Single-flight latch -- a previous in-flight describe blocks new ones.
2. Speaking-state guard -- skip when ATOM is currently SPEAKING.
3. Recent-caption dedupe -- skip when the engine already has a fresh
   caption (within the dedupe window).

These tests intentionally never construct a real VoicePipeline (which
would require STT/TTS/wake-word/audio plumbing); instead they create
the minimal object shape that ``_schedule_describe_on_wake`` reads from
``self``. This keeps the test surface tight and lets us prove the
guard *order* (single-flight > speaking > dedupe > schedule) without
spinning up the entire voice subsystem.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice.voice_pipeline import VoicePipeline  # noqa: E402


class _FakeEngine:
    """Captures look() calls so tests can assert what fired and why."""

    def __init__(self, *, recent: str = "", captioner_available: bool = True) -> None:
        self._recent = recent
        self.captioner_available = captioner_available
        self.look_calls: list[dict[str, Any]] = []

    def recent_caption(self, *, max_age_s: float | None = None) -> str:  # noqa: ARG002
        return self._recent

    def look(self, **kwargs: Any) -> Any:
        self.look_calls.append(kwargs)
        return object()


class _FakeStateManager:
    """Mimics StateManager.current_state for the speaking-guard check."""

    def __init__(self, current_state: Any) -> None:
        self.current_state = current_state


def _make_pipeline(
    *,
    engine: Any,
    state: Any,
    in_flight: bool = False,
) -> VoicePipeline:
    pipeline = VoicePipeline.__new__(VoicePipeline)
    pipeline._bus = object()
    pipeline._state = state
    pipeline._config = {}
    pipeline._mic_manager = None
    pipeline._intent_engine = None
    pipeline.stt = None
    pipeline.tts = None
    pipeline.stt_runtime_label = ""
    pipeline.stt_runtime_error = ""
    pipeline.stt_runtime_fallbacks = []
    pipeline.tts_runtime_label = ""
    pipeline._wake_word = None
    pipeline._interrupt_handler = None
    pipeline._stt_watchdog = None
    pipeline._listening_mode = None
    pipeline._loop_task = None
    pipeline._audio_intel = None
    pipeline._earcons = None
    pipeline._vision_engine = engine
    pipeline._on_wake_describe_in_flight = in_flight
    return pipeline


def _wait_for_executor_drain(loop: asyncio.AbstractEventLoop) -> None:
    """Drain the default executor so a fire-and-forget executor task
    has a chance to actually run (and call ``look``) before we assert."""
    fut = loop.run_in_executor(None, lambda: None)
    loop.run_until_complete(fut)


def test_describe_on_wake_skips_when_in_flight() -> None:
    """Single-flight latch must short-circuit before scheduling."""
    engine = _FakeEngine(recent="")
    state = _FakeStateManager(current_state=None)
    pipeline = _make_pipeline(engine=engine, state=state, in_flight=True)

    async def _run() -> None:
        pipeline._schedule_describe_on_wake(trigger="wake:test")

    asyncio.run(_run())
    assert engine.look_calls == [], "in-flight latch should suppress dispatch"


def test_describe_on_wake_skips_when_speaking() -> None:
    """Wake during TTS must NOT fire VLM (would compete for CPU)."""
    from core.state_manager import AtomState

    engine = _FakeEngine(recent="")
    state = _FakeStateManager(current_state=AtomState.SPEAKING)
    pipeline = _make_pipeline(engine=engine, state=state)

    loop = asyncio.new_event_loop()
    try:
        async def _run() -> None:
            pipeline._schedule_describe_on_wake(trigger="wake:test")
        loop.run_until_complete(_run())
        _wait_for_executor_drain(loop)
    finally:
        loop.close()
    assert engine.look_calls == [], "speaking-guard should suppress dispatch"


def test_describe_on_wake_dedupes_against_fresh_caption() -> None:
    """A fresh recent_caption must short-circuit -- saves CPU on rapid wakes."""
    from core.state_manager import AtomState

    engine = _FakeEngine(recent="A laptop on a wooden desk.")
    state = _FakeStateManager(current_state=AtomState.IDLE)
    pipeline = _make_pipeline(engine=engine, state=state)

    loop = asyncio.new_event_loop()
    try:
        async def _run() -> None:
            pipeline._schedule_describe_on_wake(trigger="wake:test")
        loop.run_until_complete(_run())
        _wait_for_executor_drain(loop)
    finally:
        loop.close()
    assert engine.look_calls == [], "fresh-caption dedupe should suppress dispatch"


def test_describe_on_wake_fires_when_all_clear() -> None:
    """No latch, not speaking, no fresh caption -- must dispatch describe."""
    from core.state_manager import AtomState

    engine = _FakeEngine(recent="")
    state = _FakeStateManager(current_state=AtomState.IDLE)
    pipeline = _make_pipeline(engine=engine, state=state)

    loop = asyncio.new_event_loop()
    try:
        async def _run() -> None:
            pipeline._schedule_describe_on_wake(trigger="wake:hey-atom")
        loop.run_until_complete(_run())
        _wait_for_executor_drain(loop)
    finally:
        loop.close()

    assert len(engine.look_calls) == 1, "should fire one describe pass"
    call = engine.look_calls[0]
    assert call.get("describe") is True
    assert call.get("detect_faces") is True
    assert call.get("detect_barcodes") is False
    assert "on_wake:wake:hey-atom" == call.get("reason")


def test_describe_on_wake_clears_in_flight_after_completion() -> None:
    """Completion callback must reset the latch so future wakes can fire."""
    from core.state_manager import AtomState

    engine = _FakeEngine(recent="")
    state = _FakeStateManager(current_state=AtomState.IDLE)
    pipeline = _make_pipeline(engine=engine, state=state)

    loop = asyncio.new_event_loop()
    try:
        async def _run() -> None:
            pipeline._schedule_describe_on_wake(trigger="wake:first")
        loop.run_until_complete(_run())
        _wait_for_executor_drain(loop)
    finally:
        loop.close()

    assert pipeline._on_wake_describe_in_flight is False
    assert len(engine.look_calls) == 1


def test_is_speaking_now_handles_missing_state() -> None:
    """Speaking-guard must fail open (return False) when state is None."""
    pipeline = _make_pipeline(engine=_FakeEngine(), state=None)
    assert pipeline._is_speaking_now() is False


if __name__ == "__main__":
    test_describe_on_wake_skips_when_in_flight()
    test_describe_on_wake_skips_when_speaking()
    test_describe_on_wake_dedupes_against_fresh_caption()
    test_describe_on_wake_fires_when_all_clear()
    test_describe_on_wake_clears_in_flight_after_completion()
    test_is_speaking_now_handles_missing_state()
    print("test_voice_pipeline_describe_on_wake: ALL PASSED")
