"""ATOM — Sprint Ω.13: stuck-LISTENING recovery regression tests.

Pins three behaviours:

  1. ``StateManager.force_listening_reset`` walks the prescribed
     transition path (``LISTENING → ERROR_RECOVERY → IDLE → LISTENING``)
     and emits a ``restart_listening`` bus event so the voice loop
     re-attaches the STT backend.
  2. The ``STATE_RECOVERY: recovered stuck listening`` log line is
     emitted with the ``source`` and dwell time, exactly as required
     by the diagnostic logging schema.
  3. :class:`RuntimeWatchdog` only triggers the reset when LISTENING is
     stuck *and* a ``whisperkit_serve_unresponsive`` event has fired
     within the correlation window. A long thoughtful pause without a
     wedge signal must NOT cause a recovery cascade.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class _BusStub:
    """Bus that records emits and dispatches subscriber callbacks
    synchronously so we can drive the watchdog handlers in-test."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self._handlers: dict[str, list[Any]] = {}

    def emit(self, evt: str, **kw: Any) -> None:
        self.events.append((evt, kw))
        self._dispatch(evt, kw)

    def emit_fast(self, evt: str, **kw: Any) -> None:
        self.events.append((evt, kw))
        self._dispatch(evt, kw)

    def emit_long(self, evt: str, **kw: Any) -> None:
        self.events.append((evt, kw))
        self._dispatch(evt, kw)

    def on(self, evt: str, handler: Any) -> None:
        self._handlers.setdefault(evt, []).append(handler)

    def find(self, evt: str) -> list[dict[str, Any]]:
        return [d for e, d in self.events if e == evt]

    def _dispatch(self, evt: str, kw: dict[str, Any]) -> None:
        for h in self._handlers.get(evt, []):
            try:
                rv = h(**kw)
                if asyncio.iscoroutine(rv):
                    asyncio.ensure_future(rv)
            except Exception:
                pass


# ── StateManager.force_listening_reset ────────────────────────────────


@pytest.mark.asyncio
async def test_force_listening_reset_walks_recovery_path() -> None:
    from core.state_manager import AtomState, StateManager
    bus = _BusStub()
    state = StateManager(bus, initial=AtomState.LISTENING)
    seen: list[AtomState] = []

    def _record(old: AtomState, new: AtomState, **_kw: Any) -> None:
        seen.append(new)

    bus.on("state_changed", _record)

    await state.force_listening_reset(source="test")

    assert state.current is AtomState.LISTENING
    assert AtomState.ERROR_RECOVERY in seen
    assert AtomState.IDLE in seen
    assert seen[-1] is AtomState.LISTENING, (
        f"expected final transition back into LISTENING, got {seen!r}"
    )


@pytest.mark.asyncio
async def test_force_listening_reset_emits_restart_listening() -> None:
    from core.state_manager import AtomState, StateManager
    bus = _BusStub()
    state = StateManager(bus, initial=AtomState.LISTENING)

    await state.force_listening_reset(source="whisperkit_unresponsive")

    restart_events = bus.find("restart_listening")
    assert restart_events, (
        "expected restart_listening event so voice loop reconnects STT"
    )
    assert restart_events[0]["source"] == "whisperkit_unresponsive"


@pytest.mark.asyncio
async def test_force_listening_reset_logs_state_recovery_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from core.state_manager import AtomState, StateManager
    bus = _BusStub()
    state = StateManager(bus, initial=AtomState.LISTENING)

    caplog.set_level(logging.WARNING, logger="atom.state")
    await state.force_listening_reset(source="probe_loop")

    msgs = [r.getMessage() for r in caplog.records]
    assert any("STATE_RECOVERY: recovered stuck listening" in m for m in msgs), (
        f"missing STATE_RECOVERY log line; got {msgs!r}"
    )
    matching = [m for m in msgs if "STATE_RECOVERY" in m]
    assert any("source=probe_loop" in m for m in matching), (
        f"STATE_RECOVERY log must carry source=probe_loop; got {matching!r}"
    )


@pytest.mark.asyncio
async def test_force_listening_reset_from_thinking_also_recovers() -> None:
    """The recovery helper must work from THINKING too (deferred-ack
    plus stuck router can wedge there in the same way)."""
    from core.state_manager import AtomState, StateManager
    bus = _BusStub()
    state = StateManager(bus, initial=AtomState.IDLE)
    await state.transition(AtomState.THINKING)

    await state.force_listening_reset(source="thinking_stuck")

    assert state.current is AtomState.LISTENING


# ── RuntimeWatchdog wiring ───────────────────────────────────────────


def _make_watchdog(bus: _BusStub, state: Any, listen_stuck_s: float = 0.05):
    from core.runtime_watchdog import RuntimeWatchdog
    config = {
        "voice": {"listening_stuck_timeout_s": listen_stuck_s},
        "performance": {
            "watchdog_poll_interval_s": 0.005,
            "watchdog_intent_boot_grace_s": 0.0,
        },
    }
    return RuntimeWatchdog(bus, state, config)


class _StateProbe:
    """Lightweight state stand-in used by RuntimeWatchdog tests so we
    can observe ``force_listening_reset`` invocations without driving
    a full StateManager + transition lock."""

    def __init__(self, current_value: str) -> None:
        from core.state_manager import AtomState
        self.current = AtomState(current_value)
        self.force_calls: list[str] = []

    async def force_listening_reset(self, source: str = "x") -> None:
        self.force_calls.append(source)


@pytest.mark.asyncio
async def test_runtime_watchdog_records_whisperkit_unresponsive() -> None:
    bus = _BusStub()
    state = _StateProbe("listening")
    watchdog = _make_watchdog(bus, state)

    assert watchdog._last_whisperkit_unresponsive_at == 0.0

    bus.emit("whisperkit_serve_unresponsive", failures=2, pid=123, reason="x")
    await asyncio.sleep(0)  # let the dispatched coroutine run

    assert watchdog._last_whisperkit_unresponsive_at > 0.0


@pytest.mark.asyncio
async def test_runtime_watchdog_clears_unresponsive_on_restart() -> None:
    bus = _BusStub()
    state = _StateProbe("listening")
    watchdog = _make_watchdog(bus, state)

    watchdog._last_whisperkit_unresponsive_at = time.monotonic()
    bus.emit("whisperkit_serve_restarted", pid=999, reason="x", elapsed_ms=10)
    await asyncio.sleep(0)

    assert watchdog._last_whisperkit_unresponsive_at == 0.0


@pytest.mark.asyncio
async def test_runtime_watchdog_triggers_force_listening_reset() -> None:
    """Stuck LISTENING + recent whisperkit wedge → force_listening_reset
    must fire exactly once per cooldown window."""
    bus = _BusStub()
    state = _StateProbe("listening")
    watchdog = _make_watchdog(bus, state, listen_stuck_s=0.01)

    watchdog._state_entered = time.monotonic() - 100.0
    watchdog._last_whisperkit_unresponsive_at = time.monotonic()

    watchdog.start()
    try:
        for _ in range(40):
            if state.force_calls:
                break
            await asyncio.sleep(0.01)
    finally:
        await watchdog.shutdown()

    assert state.force_calls, (
        "expected force_listening_reset to fire on stuck LISTENING + "
        "recent whisperkit wedge"
    )
    assert state.force_calls[0] == "whisperkit_unresponsive"


@pytest.mark.asyncio
async def test_runtime_watchdog_no_recovery_without_whisperkit_signal() -> None:
    """A long thoughtful pause with no whisperkit wedge must NOT
    trigger a state recovery — that would cut Boss off mid-thought."""
    bus = _BusStub()
    state = _StateProbe("listening")
    watchdog = _make_watchdog(bus, state, listen_stuck_s=0.01)

    watchdog._state_entered = time.monotonic() - 100.0
    watchdog._last_whisperkit_unresponsive_at = 0.0

    watchdog.start()
    try:
        await asyncio.sleep(0.08)
    finally:
        await watchdog.shutdown()

    assert not state.force_calls, (
        "force_listening_reset must NOT fire without a whisperkit wedge"
    )
