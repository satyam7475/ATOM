"""Focused tests for macOS TTS streaming behavior."""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.state_manager import AtomState, StateManager
from voice.tts_macos import MacOSTTSAsync


class FakeEventBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def on(self, event: str, handler) -> None:
        pass

    def emit(self, event: str, **data) -> None:
        self.events.append((event, data))

    def emit_fast(self, event: str, **data) -> None:
        self.events.append((event, data))


async def _make_tts() -> tuple[MacOSTTSAsync, FakeEventBus, list[str]]:
    bus = FakeEventBus()
    state = StateManager(bus, initial=AtomState.IDLE)
    await state.transition(AtomState.LISTENING)
    await state.transition(AtomState.THINKING)

    tts = MacOSTTSAsync(bus, state)
    spoken: list[str] = []

    async def fake_speak_internal(text: str, emotion: str | None = None) -> None:
        spoken.append(text)
        await asyncio.sleep(0)

    async def fake_kill_procs() -> None:
        return None

    tts._available = True
    tts._backend = "test"
    tts._speak_internal = fake_speak_internal  # type: ignore[method-assign]
    tts._kill_procs = fake_kill_procs  # type: ignore[method-assign]
    return tts, bus, spoken


async def test_streaming_starts_before_last_chunk() -> None:
    tts, bus, spoken = await _make_tts()

    await tts.on_partial_response(
        "Hello there.",
        is_first=True,
        is_last=False,
        source="local",
        stream_id="stream-1",
    )
    await asyncio.sleep(0.05)

    assert spoken == ["Hello there."]
    assert not any(event == "tts_complete" for event, _ in bus.events)

    await tts.on_partial_response(
        "How are you?",
        is_first=False,
        is_last=True,
        source="local",
        stream_id="stream-1",
    )
    await asyncio.sleep(0.05)

    assert spoken == ["Hello there.", "How are you?"]
    assert any(event == "tts_complete" for event, _ in bus.events)
    print("  PASS: macOS TTS starts speaking on the first streamed chunk")


async def test_stale_stream_id_is_ignored() -> None:
    tts, bus, spoken = await _make_tts()

    await tts.on_partial_response(
        "Current answer.",
        is_first=True,
        is_last=False,
        source="local",
        stream_id="stream-live",
    )
    await asyncio.sleep(0.05)

    await tts.on_partial_response(
        "Old stale chunk.",
        is_first=False,
        is_last=False,
        source="local",
        stream_id="stream-stale",
    )
    await asyncio.sleep(0.05)

    await tts.on_partial_response(
        "Final live chunk.",
        is_first=False,
        is_last=True,
        source="local",
        stream_id="stream-live",
    )
    await asyncio.sleep(0.05)

    assert spoken == ["Current answer.", "Final live chunk."]
    assert not any(
        event == "text_display" and "Old stale chunk." in data.get("text", "")
        for event, data in bus.events
    )
    print("  PASS: macOS TTS ignores stale chunks from an older stream")


async def run_all() -> None:
    print("\n=== macOS TTS Streaming Tests ===\n")
    await test_streaming_starts_before_last_chunk()
    await test_stale_stream_id_is_ignored()
    print("\n=== ALL TESTS PASSED ===\n")


if __name__ == "__main__":
    asyncio.run(run_all())
