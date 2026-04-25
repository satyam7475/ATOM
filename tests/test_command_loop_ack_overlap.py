"""ATOM -- regression suite for parallel ack-TTS overlap (Sprint C3).

Pins three behaviours:

1. ``CommandLoop.submit()`` spawns ``tts.speak_ack`` as a fire-and-
   forget task BEFORE awaiting the router, so the ack audio rolls
   while the LLM prefills. Saves the bus dispatch step (~50-200 ms
   per turn).
2. ``voice_ack`` bus event still fires for indicator subscribers and
   carries ``spoken_inline=True`` so ``on_voice_ack`` in wiring.py
   does NOT double-speak the same ack.
3. When no TTS is wired the loop falls back gracefully to the legacy
   bus-only flow (no crash, no double-speak, no missing ack).
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

    await loop.submit("hello atom")

    assert tts.acks == ["Got it, Boss."], (
        f"speak_ack should have fired exactly once, got {tts.acks!r}"
    )
    assert router.calls == ["hello atom"]
    assert loop._ack_overlap_count == 1
    # ack must have *started* before the router slept (proven by the
    # ack start timestamp preceding the router's awaited completion).
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

    await loop.submit("open chrome")

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

    await loop.submit("ping")

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
