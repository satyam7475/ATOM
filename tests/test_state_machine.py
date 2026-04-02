"""
ATOM v9 -- Phase 1 State Machine Verification.

Tests all valid transitions and verifies illegal transitions are blocked.
Run: python -m tests.test_state_machine
"""

from __future__ import annotations

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.state_manager import AtomState, StateManager, VALID_TRANSITIONS


class FakeEventBus:
    """Minimal event bus stub that records emitted events."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def on(self, event: str, handler) -> None:
        pass

    def emit(self, event: str, **data) -> None:
        self.events.append((event, data))

    def clear(self) -> None:
        self.events.clear()


async def test_happy_path() -> None:
    """Simulate the normal v9 lifecycle: IDLE -> LISTENING -> THINKING -> SPEAKING -> IDLE."""
    bus = FakeEventBus()
    sm = StateManager(bus, initial=AtomState.IDLE)

    assert sm.current is AtomState.IDLE, f"Expected IDLE, got {sm.current}"

    # Wake word detected -> LISTENING
    await sm.transition(AtomState.LISTENING)
    assert sm.current is AtomState.LISTENING, f"Expected LISTENING, got {sm.current}"

    # Speech finalized -> THINKING
    await sm.transition(AtomState.THINKING)
    assert sm.current is AtomState.THINKING, f"Expected THINKING, got {sm.current}"

    # First partial response -> SPEAKING
    await sm.transition(AtomState.SPEAKING)
    assert sm.current is AtomState.SPEAKING, f"Expected SPEAKING, got {sm.current}"

    # TTS complete -> IDLE (mic idle until next utterance / resume)
    await sm.on_tts_complete()
    assert sm.current is AtomState.IDLE, f"Expected IDLE after tts_complete, got {sm.current}"

    print("  PASS: Happy path (IDLE -> LISTENING -> THINKING -> SPEAKING -> IDLE)")


async def test_silence_timeout() -> None:
    """LISTENING -> IDLE when no speech is detected."""
    bus = FakeEventBus()
    sm = StateManager(bus, initial=AtomState.IDLE)

    await sm.transition(AtomState.LISTENING)
    assert sm.current is AtomState.LISTENING

    await sm.on_silence_timeout()
    assert sm.current is AtomState.IDLE, f"Expected IDLE after silence timeout, got {sm.current}"

    print("  PASS: Silence timeout (LISTENING -> IDLE)")


async def test_barge_in() -> None:
    """SPEAKING -> LISTENING when user interrupts (barge-in / resume)."""
    bus = FakeEventBus()
    sm = StateManager(bus, initial=AtomState.IDLE)

    await sm.transition(AtomState.LISTENING)
    await sm.transition(AtomState.THINKING)
    await sm.transition(AtomState.SPEAKING)
    assert sm.current is AtomState.SPEAKING

    # Barge-in during speech
    await sm.transition(AtomState.LISTENING)
    assert sm.current is AtomState.LISTENING, f"Expected LISTENING after barge-in, got {sm.current}"

    print("  PASS: Barge-in (SPEAKING -> LISTENING)")


async def test_llm_error_fallback() -> None:
    """THINKING -> LISTENING when LLM request fails."""
    bus = FakeEventBus()
    sm = StateManager(bus, initial=AtomState.IDLE)

    await sm.transition(AtomState.LISTENING)
    await sm.transition(AtomState.THINKING)
    assert sm.current is AtomState.THINKING

    # LLM error -> fall back to LISTENING so user can retry
    await sm.transition(AtomState.LISTENING)
    assert sm.current is AtomState.LISTENING, f"Expected LISTENING after LLM error, got {sm.current}"

    print("  PASS: LLM error fallback (THINKING -> LISTENING)")


async def test_shutdown_from_any_state() -> None:
    """Every state must allow transition to SLEEP."""
    for state in AtomState:
        if state is AtomState.SLEEP:
            continue
        bus = FakeEventBus()
        sm = StateManager(bus, initial=state)
        await sm.transition(AtomState.SLEEP)
        assert sm.current is AtomState.SLEEP, \
            f"Expected SLEEP from {state.value}, got {sm.current}"

    print("  PASS: Shutdown from every state -> SLEEP")


async def test_startup_from_sleep() -> None:
    """SLEEP -> IDLE (system boot)."""
    bus = FakeEventBus()
    sm = StateManager(bus, initial=AtomState.SLEEP)

    await sm.transition(AtomState.IDLE)
    assert sm.current is AtomState.IDLE, f"Expected IDLE from SLEEP, got {sm.current}"

    print("  PASS: Startup (SLEEP -> IDLE)")


async def test_illegal_transitions_blocked() -> None:
    """Verify that illegal transitions are silently blocked.

    Must stay aligned with ``core.state_manager.VALID_TRANSITIONS`` (v14+):
    - LISTENING -> SPEAKING is **legal** (fast-path local reply, skips THINKING).
    - SLEEP -> LISTENING is **legal** (resume from sleep / silent mode).
    - THINKING -> IDLE is **legal** (cache/cognitive paths with no TTS).
    """
    bus = FakeEventBus()

    illegal_pairs = [
        (AtomState.IDLE, AtomState.THINKING),       # can't skip LISTENING
        (AtomState.IDLE, AtomState.SPEAKING),       # can't skip LISTENING (+ usually THINKING)
        (AtomState.IDLE, AtomState.ERROR_RECOVERY), # no direct path from idle
        (AtomState.SPEAKING, AtomState.THINKING),   # can't go backwards
        (AtomState.SLEEP, AtomState.THINKING),      # can't skip to THINKING
        (AtomState.SLEEP, AtomState.SPEAKING),      # can't skip everything
        (AtomState.ERROR_RECOVERY, AtomState.LISTENING),  # recovery only to IDLE (or SLEEP)
    ]

    for from_state, to_state in illegal_pairs:
        sm = StateManager(bus, initial=from_state)
        await sm.transition(to_state)
        assert sm.current is from_state, \
            f"Illegal transition {from_state.value} -> {to_state.value} was NOT blocked!"

    print(f"  PASS: {len(illegal_pairs)} illegal transitions correctly blocked")


async def test_noop_transition() -> None:
    """Transition to the same state is a silent no-op."""
    bus = FakeEventBus()
    sm = StateManager(bus, initial=AtomState.IDLE)
    bus.events.clear()

    await sm.transition(AtomState.IDLE)
    assert sm.current is AtomState.IDLE
    assert len(bus.events) == 0, "No-op transition should not emit events"

    print("  PASS: No-op transition (IDLE -> IDLE) emits no events")


async def test_events_emitted() -> None:
    """Verify that state_changed events carry correct old/new values."""
    bus = FakeEventBus()
    sm = StateManager(bus, initial=AtomState.IDLE)
    bus.events.clear()

    await sm.transition(AtomState.LISTENING)
    assert len(bus.events) == 1
    event_name, event_data = bus.events[0]
    assert event_name == "state_changed"
    assert event_data["old"] is AtomState.IDLE
    assert event_data["new"] is AtomState.LISTENING

    print("  PASS: state_changed event emitted with correct old/new")


async def test_transition_table_completeness() -> None:
    """Every AtomState must appear as a key in VALID_TRANSITIONS."""
    for state in AtomState:
        assert state in VALID_TRANSITIONS, \
            f"State {state.value} missing from VALID_TRANSITIONS"

    print(f"  PASS: All {len(AtomState)} states have transition entries")


async def test_barge_in_full_cycle() -> None:
    """Full barge-in lifecycle: SPEAKING -> LISTENING -> THINKING -> SPEAKING -> IDLE."""
    bus = FakeEventBus()
    sm = StateManager(bus, initial=AtomState.IDLE)

    await sm.transition(AtomState.LISTENING)
    await sm.transition(AtomState.THINKING)
    await sm.transition(AtomState.SPEAKING)
    assert sm.current is AtomState.SPEAKING

    await sm.transition(AtomState.LISTENING)
    assert sm.current is AtomState.LISTENING

    await sm.transition(AtomState.THINKING)
    await sm.transition(AtomState.SPEAKING)
    await sm.on_tts_complete()
    assert sm.current is AtomState.IDLE

    print("  PASS: Full barge-in cycle (SPEAKING -> LISTENING -> ... -> IDLE)")


async def test_silence_returns_to_idle() -> None:
    """After LISTENING -> silence, system returns to IDLE (can re-enter LISTENING)."""
    bus = FakeEventBus()
    sm = StateManager(bus, initial=AtomState.IDLE)

    await sm.transition(AtomState.LISTENING)
    assert sm.current is AtomState.LISTENING

    await sm.on_silence_timeout()
    assert sm.current is AtomState.IDLE

    await sm.transition(AtomState.LISTENING)
    assert sm.current is AtomState.LISTENING

    print("  PASS: Silence -> IDLE -> re-activate works")


async def run_all() -> None:
    print("\n=== ATOM v9 Phase 1+6 -- State Machine Tests ===\n")

    await test_happy_path()
    await test_silence_timeout()
    await test_barge_in()
    await test_barge_in_full_cycle()
    await test_silence_returns_to_idle()
    await test_llm_error_fallback()
    await test_shutdown_from_any_state()
    await test_startup_from_sleep()
    await test_illegal_transitions_blocked()
    await test_noop_transition()
    await test_events_emitted()
    await test_transition_table_completeness()

    print("\n=== ALL TESTS PASSED ===\n")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_all())
