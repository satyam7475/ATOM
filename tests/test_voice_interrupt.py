"""Focused tests for the voice interrupt coordinator."""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.state_manager import AtomState, StateManager
from voice.interrupt_handler import VoiceInterruptHandler


class FakeEventBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def on(self, event: str, handler) -> None:
        pass

    def emit(self, event: str, **data) -> None:
        self.events.append((event, data))

    def emit_fast(self, event: str, **data) -> None:
        self.events.append((event, data))


class FakeTTS:
    def __init__(self) -> None:
        self.stop_calls = 0

    async def stop(self) -> None:
        self.stop_calls += 1


class FakeInterruptManager:
    def __init__(self) -> None:
        self.broadcasts = 0

    async def broadcast_interrupt(self) -> None:
        self.broadcasts += 1


class FakeBrain:
    def __init__(self) -> None:
        self.preempt_calls = 0

    def request_preempt(self) -> None:
        self.preempt_calls += 1


class FakeIndicator:
    def __init__(self) -> None:
        self.logs: list[tuple[str, str]] = []

    def add_log(self, level: str, text: str) -> None:
        self.logs.append((level, text))


async def test_interrupt_from_speaking() -> None:
    bus = FakeEventBus()
    state = StateManager(bus, initial=AtomState.IDLE)
    tts = FakeTTS()
    interrupt_mgr = FakeInterruptManager()
    brain = FakeBrain()
    indicator = FakeIndicator()
    handler = VoiceInterruptHandler(
        bus=bus,
        state=state,
        tts=tts,
        interrupt_manager=interrupt_mgr,
        local_brain=brain,
        indicator=indicator,
    )

    await state.transition(AtomState.LISTENING)
    await state.transition(AtomState.THINKING)
    await state.transition(AtomState.SPEAKING)

    changed = await handler.interrupt_to_listening(
        trigger="speech_final",
        reason="new_speech",
        partial_text="open chrome",
        user_interrupt=True,
    )

    assert changed is True
    assert state.current is AtomState.LISTENING
    assert tts.stop_calls == 1
    assert interrupt_mgr.broadcasts == 1
    assert brain.preempt_calls == 1
    assert any(event == "user_interrupt" for event, _ in bus.events)
    print("  PASS: Voice interrupt stops TTS and returns to LISTENING")


async def test_partial_arming_status_is_ignored() -> None:
    bus = FakeEventBus()
    state = StateManager(bus, initial=AtomState.SPEAKING)
    handler = VoiceInterruptHandler(
        bus=bus,
        state=state,
        tts=FakeTTS(),
        interrupt_manager=FakeInterruptManager(),
        local_brain=FakeBrain(),
        indicator=FakeIndicator(),
    )

    await handler.on_speech_partial("Listening...")

    assert not any(event == "resume_listening" for event, _ in bus.events)
    print("  PASS: Listening status alone does not trigger barge-in")


async def test_processing_status_is_ignored() -> None:
    bus = FakeEventBus()
    state = StateManager(bus, initial=AtomState.SPEAKING)
    handler = VoiceInterruptHandler(
        bus=bus,
        state=state,
        tts=FakeTTS(),
        interrupt_manager=FakeInterruptManager(),
        local_brain=FakeBrain(),
        indicator=FakeIndicator(),
    )

    await handler.on_speech_partial("Processing...")

    assert not any(event == "resume_listening" for event, _ in bus.events)
    print("  PASS: Processing status alone does not trigger barge-in")


async def test_transcript_partial_emits_resume() -> None:
    bus = FakeEventBus()
    state = StateManager(bus, initial=AtomState.SPEAKING)
    handler = VoiceInterruptHandler(
        bus=bus,
        state=state,
        tts=FakeTTS(),
        interrupt_manager=FakeInterruptManager(),
        local_brain=FakeBrain(),
        indicator=FakeIndicator(),
    )

    await handler.on_speech_partial("open chrome")

    matches = [
        data for event, data in bus.events
        if event == "resume_listening" and data.get("source") == "voice_interrupt"
    ]
    assert len(matches) == 1
    assert matches[0].get("reason") == "speech_partial"
    print("  PASS: Transcript partial emits resume_listening")


async def test_prepare_for_new_speech_from_thinking() -> None:
    bus = FakeEventBus()
    state = StateManager(bus, initial=AtomState.IDLE)
    tts = FakeTTS()
    interrupt_mgr = FakeInterruptManager()
    brain = FakeBrain()
    indicator = FakeIndicator()
    handler = VoiceInterruptHandler(
        bus=bus,
        state=state,
        tts=tts,
        interrupt_manager=interrupt_mgr,
        local_brain=brain,
        indicator=indicator,
    )

    await state.transition(AtomState.LISTENING)
    await state.transition(AtomState.THINKING)

    changed = await handler.prepare_for_new_speech("what time is it")

    assert changed is True
    assert state.current is AtomState.LISTENING
    assert tts.stop_calls == 0
    assert interrupt_mgr.broadcasts == 1
    assert brain.preempt_calls == 1
    assert any("Interrupted. Go ahead, Boss." in text for _, text in indicator.logs)
    print("  PASS: New speech preempts THINKING before routing")


async def run_all() -> None:
    print("\n=== Voice Interrupt Tests ===\n")
    await test_interrupt_from_speaking()
    await test_partial_arming_status_is_ignored()
    await test_processing_status_is_ignored()
    await test_transcript_partial_emits_resume()
    await test_prepare_for_new_speech_from_thinking()
    print("\n=== ALL TESTS PASSED ===\n")


if __name__ == "__main__":
    asyncio.run(run_all())
