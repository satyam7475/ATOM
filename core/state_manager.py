"""
ATOM -- Thread-safe async state machine with transition metrics.

Six states with strictly validated transitions.
Every transition emits a ``state_changed`` event on the bus.

v20 enhancements over v14:
  - Transition timing: tracks time spent in each state
  - Transition counters: how often each transition occurs
  - Last-transition timestamp for health monitoring
  - Fast no-op path: skips lock acquisition when current == target
  - State duration API for diagnostics

States:
    SLEEP          -- fully shut down, no audio processing
    IDLE           -- resting state, minimal background work (<0.5% CPU)
    LISTENING      -- faster-whisper STT active, processing speech
    THINKING       -- LLM query in flight
    SPEAKING       -- TTS playing + barge-in (hotkey / resume_listening)
    ERROR_RECOVERY -- transient error state, auto-recovers to IDLE
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .async_event_bus import AsyncEventBus

logger = logging.getLogger("atom.state")


class AtomState(enum.Enum):
    SLEEP = "sleep"
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR_RECOVERY = "error_recovery"


VALID_TRANSITIONS: dict[AtomState, frozenset[AtomState]] = {
    AtomState.SLEEP: frozenset({
        AtomState.IDLE,
        AtomState.LISTENING,
    }),
    AtomState.IDLE: frozenset({
        AtomState.LISTENING,
        AtomState.SLEEP,
    }),
    AtomState.LISTENING: frozenset({
        AtomState.THINKING,
        AtomState.SPEAKING,
        AtomState.IDLE,
        AtomState.ERROR_RECOVERY,
        AtomState.SLEEP,
    }),
    AtomState.THINKING: frozenset({
        AtomState.SPEAKING,
        AtomState.LISTENING,
        AtomState.IDLE,
        AtomState.ERROR_RECOVERY,
        AtomState.SLEEP,
    }),
    AtomState.SPEAKING: frozenset({
        AtomState.IDLE,
        AtomState.LISTENING,
        AtomState.ERROR_RECOVERY,
        AtomState.SLEEP,
    }),
    AtomState.ERROR_RECOVERY: frozenset({
        AtomState.IDLE,
        AtomState.SLEEP,
    }),
}




class StateManager:
    """
    Manages the ATOM lifecycle through a strict state machine.

    Thread-safe: the transition is guarded by an asyncio.Lock.
    Read-only access to ``current`` is lock-free (CPython GIL guarantee).

    Tracks transition counts and time-in-state for diagnostics.
    """

    __slots__ = (
        "_state", "_lock", "_bus", "_always_listen",
        "_state_enter_time", "_transition_counts",
        "_state_durations", "_last_transition_time",
        "_total_transitions", "_error_recovery_hold_s",
    )

    def __init__(
        self,
        bus: AsyncEventBus,
        initial: AtomState = AtomState.IDLE,
        *,
        error_recovery_hold_s: float = 0.35,
    ) -> None:
        self._state = initial
        self._lock = asyncio.Lock()
        self._bus = bus
        self._always_listen = False
        # Keep ERROR_RECOVERY visible long enough for observers to react.
        self._error_recovery_hold_s = max(0.0, float(error_recovery_hold_s))
        now = time.monotonic()
        self._state_enter_time: float = now
        self._last_transition_time: float = now
        self._total_transitions: int = 0
        self._transition_counts: dict[tuple[AtomState, AtomState], int] = defaultdict(int)
        self._state_durations: dict[AtomState, float] = defaultdict(float)
        logger.info("StateManager initialised -> %s", initial.value)

    @property
    def current(self) -> AtomState:
        return self._state

    @property
    def time_in_current_state(self) -> float:
        """Seconds spent in the current state."""
        return time.monotonic() - self._state_enter_time

    @property
    def last_transition_age(self) -> float:
        """Seconds since the last state transition."""
        return time.monotonic() - self._last_transition_time

    async def transition(self, new_state: AtomState) -> None:
        """
        Atomically move to *new_state* if the transition is legal.

        Emits ``state_changed`` with ``old`` and ``new`` kwargs.
        Silently ignores no-op transitions (current == new) without
        acquiring the lock (fast path).
        Logs a warning and returns for illegal transitions.
        """
        if self._state is new_state:
            return

        async with self._lock:
            old = self._state
            if old is new_state:
                return
            if new_state not in VALID_TRANSITIONS[old]:
                logger.warning("Blocked illegal transition %s -> %s", old.value, new_state.value)
                return

            now = time.monotonic()
            duration = now - self._state_enter_time
            self._state_durations[old] += duration

            self._state = new_state
            self._state_enter_time = now
            self._last_transition_time = now
            self._total_transitions += 1
            self._transition_counts[(old, new_state)] += 1

            logger.info("State: %s -> %s (was %s for %.1fs)",
                        old.value, new_state.value, old.value, duration)

        self._bus.emit("state_changed", old=old, new=new_state)

    def get_diagnostics(self) -> dict:
        """Return state machine diagnostics for health monitoring."""
        return {
            "current": self._state.value,
            "time_in_state_s": round(self.time_in_current_state, 1),
            "total_transitions": self._total_transitions,
            "state_durations": {
                s.value: round(d, 1)
                for s, d in self._state_durations.items()
            },
            "top_transitions": {
                f"{a.value}->{b.value}": c
                for (a, b), c in sorted(
                    self._transition_counts.items(),
                    key=lambda x: x[1], reverse=True,
                )[:10]
            },
        }

    @property
    def always_listen(self) -> bool:
        return self._always_listen

    @always_listen.setter
    def always_listen(self, value: bool) -> None:
        self._always_listen = value

    async def on_tts_complete(self, **_kw) -> None:
        """After speech finishes, return to IDLE or LISTENING (if always-listen)."""
        if self._state is AtomState.SPEAKING:
            if self._always_listen:
                await self.transition(AtomState.LISTENING)
            else:
                await self.transition(AtomState.IDLE)

    async def on_silence_timeout(self, **_kw) -> None:
        """No speech detected within timeout -- restart LISTENING or return to IDLE."""
        if self._state in (AtomState.LISTENING, AtomState.SPEAKING):
            if self._always_listen:
                self._bus.emit("restart_listening")
            else:
                await self.transition(AtomState.IDLE)

    async def on_error(self, source: str = "unknown", **_kw) -> None:
        """Transition to ERROR_RECOVERY on component failure, then auto-recover to IDLE."""
        if self._state in (AtomState.LISTENING, AtomState.THINKING, AtomState.SPEAKING):
            logger.error("Error from %s in state %s -- entering recovery", source, self._state.value)
            await self.transition(AtomState.ERROR_RECOVERY)
            if self._error_recovery_hold_s > 0:
                await asyncio.sleep(self._error_recovery_hold_s)
            await self.transition(AtomState.IDLE)

