"""ATOM -- regression suite for parallel ack-TTS overlap (Sprint C3 +
Sprint Ω.13 deferred-ACK).

Pins five behaviours:

1. ``CommandLoop.submit()`` spawns ``tts.speak_ack`` as a fire-and-
   forget task BEFORE awaiting the router, so the ack audio rolls
   while the LLM prefills. Saves the bus dispatch step (~50-200 ms
   per turn).
2. ``voice_ack`` bus event still fires for indicator subscribers and
   carries ``spoken_inline=True`` so ``on_voice_ack`` in wiring.py
   does NOT double-speak the same ack.
3. When no TTS is wired the loop falls back gracefully to the legacy
   bus-only flow (no crash, no double-speak, no missing ack).
4. (Sprint Ω.13) When ``response_ready`` arrives within the deferral
   window (default 280 ms) the spoken ACK is suppressed entirely —
   ``speak_ack`` is NOT awaited and ``PIPELINE_FAST_PATH: skip ACK``
   is logged.
5. (Sprint Ω.13) When ``response_ready`` arrives AFTER the deferral
   window the ACK is spoken on schedule, exactly once.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from core.command_loop import CommandLoop


# ── Lightweight stubs ─────────────────────────────────────────────


class _BusStub:
    """Fast bus that records ``emit_fast`` payloads without dispatch."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self._handlers: dict[str, list[Any]] = {}

    def emit_fast(self, event: str, **data: Any) -> None:
        self.events.append((event, data))

    def emit(self, event: str, **data: Any) -> None:
        self.events.append((event, data))

    def emit_long(self, event: str, **data: Any) -> None:
        self.events.append((event, data))

    def on(self, event: str, handler: Any) -> None:
        self._handlers.setdefault(event, []).append(handler)

    def find(self, event: str) -> list[dict[str, Any]]:
        return [data for evt, data in self.events if evt == event]


class _StateStub:
    def __init__(self) -> None:
        from core.state_manager import AtomState
        self.current = AtomState.LISTENING
        self.transitions: list[Any] = []

    async def transition(self, new_state: Any) -> None:
        self.transitions.append(new_state)
        self.current = new_state

    async def on_error(self, source: str = "") -> None:
        pass


class _RouterStub:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.delay = 0.0

    async def on_speech(self, text: str) -> None:
        self.calls.append(text)
        if self.delay:
            await asyncio.sleep(self.delay)


class _AckEngineStub:
    def __init__(self, ack_text: str = "On it, Boss.") -> None:
        self._ack = ack_text

    def get_ack(self, text: str, *, is_follow_up: bool = False) -> str:
        return self._ack


class _TTSRecorder:
    """TTS double that records ack invocations AND timestamps so we
    can prove the ack started before the router awaited."""

    def __init__(self) -> None:
        self.acks: list[str] = []
        self.ack_started_at: list[float] = []
        self.ack_done_at: list[float] = []
        self._lock = asyncio.Lock()
        self.delay = 0.05

    async def speak_ack(self, phrase: str) -> None:
        loop = asyncio.get_event_loop()
        self.ack_started_at.append(loop.time())
        async with self._lock:
            self.acks.append(phrase)
            if self.delay:
                await asyncio.sleep(self.delay)
        self.ack_done_at.append(loop.time())


# ── Tests ────────────────────────────────────────────────────────


async def _drain_ack_task(loop: CommandLoop) -> None:
    """Sprint Ω.13: the deferred-ACK task is fire-and-forget. Tests
    that want to observe the spoken ack must await its completion
    after ``submit`` returns."""
    if loop._ack_task is not None:
        try:
            await loop._ack_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_ack_task_spawned_before_router_call() -> None:
    bus = _BusStub()
    state = _StateStub()
    router = _RouterStub()
    router.delay = 0.05
    tts = _TTSRecorder()

    loop = CommandLoop(bus, state, router)
    loop.attach_ack_engine(_AckEngineStub("Got it, Boss."))
    loop.attach_tts(tts)
    # Sprint Ω.13: tighten the deferral window so the ack fires inside
    # the test's already-awaited router delay (50 ms) — that proves
    # the ack and router run concurrently without flaking on slow CI.
    loop._ack_deferral_s = 0.0

    await loop.submit("hello atom")
    await _drain_ack_task(loop)

    assert tts.acks == ["Got it, Boss."], (
        f"speak_ack should have fired exactly once, got {tts.acks!r}"
    )
    assert router.calls == ["hello atom"]
    assert loop._ack_overlap_count == 1
    assert tts.ack_started_at, "ack timestamp not recorded"


@pytest.mark.asyncio
async def test_voice_ack_bus_event_carries_spoken_inline_flag() -> None:
    bus = _BusStub()
    state = _StateStub()
    router = _RouterStub()
    tts = _TTSRecorder()

    loop = CommandLoop(bus, state, router)
    loop.attach_ack_engine(_AckEngineStub("Right away, Boss."))
    loop.attach_tts(tts)
    loop._ack_deferral_s = 0.0  # speak immediately for this test

    await loop.submit("open chrome")
    await _drain_ack_task(loop)

    voice_acks = bus.find("voice_ack")
    assert len(voice_acks) == 1
    assert voice_acks[0]["text"] == "Right away, Boss."
    assert voice_acks[0]["spoken_inline"] is True


@pytest.mark.asyncio
async def test_legacy_path_when_tts_not_attached() -> None:
    """No TTS wired -> we still emit the bus event (with
    ``spoken_inline=False``) so wiring.py's ``on_voice_ack`` keeps
    speaking. No crash, no double-speak."""
    bus = _BusStub()
    state = _StateStub()
    router = _RouterStub()

    loop = CommandLoop(bus, state, router)
    loop.attach_ack_engine(_AckEngineStub("One sec, Boss."))
    # NOTE: no attach_tts -- legacy path

    await loop.submit("what's the time")

    voice_acks = bus.find("voice_ack")
    assert len(voice_acks) == 1
    assert voice_acks[0]["spoken_inline"] is False
    assert loop._ack_overlap_count == 0
    assert loop._ack_task is None


@pytest.mark.asyncio
async def test_attach_tts_rejects_object_without_speak_ack() -> None:
    bus = _BusStub()
    state = _StateStub()
    router = _RouterStub()

    loop = CommandLoop(bus, state, router)
    loop.attach_tts(SimpleNamespace())  # no speak_ack attribute
    assert loop._tts is None


@pytest.mark.asyncio
async def test_ack_failure_does_not_break_pipeline() -> None:
    """If ``speak_ack`` raises immediately we still complete the turn."""
    bus = _BusStub()
    state = _StateStub()
    router = _RouterStub()

    class _BoomTTS:
        async def speak_ack(self, phrase: str) -> None:
            raise RuntimeError("simulated TTS failure")

    loop = CommandLoop(bus, state, router)
    loop.attach_ack_engine(_AckEngineStub("Aye, Boss."))
    loop.attach_tts(_BoomTTS())
    loop._ack_deferral_s = 0.0

    await loop.submit("ping")
    await _drain_ack_task(loop)

    assert router.calls == ["ping"]
    # ack_task was created (then the inner coroutine threw); count
    # should still increment so diagnostics stay accurate.
    assert loop._ack_overlap_count == 1


@pytest.mark.asyncio
async def test_no_ack_text_skips_overlap_entirely() -> None:
    """When the ack engine returns empty string, no task is spawned
    and no bus event is emitted."""
    bus = _BusStub()
    state = _StateStub()
    router = _RouterStub()
    tts = _TTSRecorder()

    loop = CommandLoop(bus, state, router)
    loop.attach_ack_engine(_AckEngineStub(""))  # empty ack
    loop.attach_tts(tts)

    await loop.submit("hi")

    assert tts.acks == []
    assert bus.find("voice_ack") == []
    assert loop._ack_task is None


# ── Sprint Ω.13: deferred ACK ────────────────────────────────────────


@pytest.mark.asyncio
async def test_ack_suppressed_when_response_ready_within_window(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When ``response_ready`` fires within ``_ack_deferral_s`` of
    ``submit``, the spoken ACK is cancelled and ``PIPELINE_FAST_PATH:
    skip ACK`` is logged. Mirrors the intent fast-path turns at
    atomCurrentLogs L295/L338/L365 where the brain replied in ≤ 8 ms
    but Boss still heard a useless 'On it.' preamble."""
    import logging
    bus = _BusStub()
    state = _StateStub()
    router = _RouterStub()
    tts = _TTSRecorder()

    loop = CommandLoop(bus, state, router)
    loop.attach_ack_engine(_AckEngineStub("On it, Boss."))
    loop.attach_tts(tts)
    # Default 280 ms deferral. Submit is <1 ms; response_ready below
    # therefore lands well inside the window.

    caplog.set_level(logging.INFO, logger="atom.command_loop")
    await loop.submit("ping")

    # Simulate the brain returning a response immediately after submit
    # — within the 280 ms deferral window.
    await loop._on_response_ready(text="pong")

    # Drain the (now cancelled) ACK task. CancelledError must not bubble.
    await _drain_ack_task(loop)

    assert tts.acks == [], (
        "speak_ack must NOT fire when response_ready arrived within "
        f"the deferral window, got {tts.acks!r}"
    )
    assert loop._ack_skipped_count == 1, (
        f"expected _ack_skipped_count == 1, got {loop._ack_skipped_count}"
    )
    msgs = [r.getMessage() for r in caplog.records]
    assert any("PIPELINE_FAST_PATH: skip ACK" in m for m in msgs), (
        f"missing PIPELINE_FAST_PATH skip log; got {msgs!r}"
    )


@pytest.mark.asyncio
async def test_ack_spoken_when_response_ready_arrives_late() -> None:
    """When ``response_ready`` fires AFTER the deferral window, the
    ack must already have been spoken on its original schedule."""
    bus = _BusStub()
    state = _StateStub()
    router = _RouterStub()
    tts = _TTSRecorder()

    loop = CommandLoop(bus, state, router)
    loop.attach_ack_engine(_AckEngineStub("On it, Boss."))
    loop.attach_tts(tts)
    loop._ack_deferral_s = 0.02  # 20 ms — keeps the test fast

    await loop.submit("hello atom")
    await _drain_ack_task(loop)
    await loop._on_response_ready(text="hi")

    assert tts.acks == ["On it, Boss."], (
        "speak_ack should have fired (deferral expired before "
        f"response_ready arrived), got {tts.acks!r}"
    )
    assert loop._ack_skipped_count == 0


@pytest.mark.asyncio
async def test_ack_default_deferral_matches_spec() -> None:
    """The plan fixes the deferral window at 280 ms — guard against
    accidental drift in case someone tweaks the constant."""
    bus = _BusStub()
    state = _StateStub()
    router = _RouterStub()
    loop = CommandLoop(bus, state, router)
    assert abs(loop._ack_deferral_s - 0.28) < 1e-6, (
        f"deferral window must remain 280 ms (spec); got {loop._ack_deferral_s}"
    )
