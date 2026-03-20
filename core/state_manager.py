"""
ATOM v14 -- Thread-safe async state machine.

Six states with strictly validated transitions.
Every transition emits a ``state_changed`` event on the bus.

States:
    SLEEP          -- fully shut down, no audio processing
    IDLE           -- resting state, minimal background work (<0.5% CPU)
    LISTENING      -- Vosk STT active, processing speech
    THINKING       -- LLM query in flight
    SPEAKING       -- TTS playing + barge-in (hotkey / resume_listening)
    ERROR_RECOVERY -- transient error state, auto-recovers to IDLE

Key design decisions:
    - IDLE is the primary resting state (not full LISTENING)
    - SPEAKING -> IDLE on tts_complete (returns to idle / always-listen per config)
    - LISTENING -> IDLE on silence timeout
    - LISTENING / THINKING / SPEAKING -> ERROR_RECOVERY on failure
    - ERROR_RECOVERY -> IDLE auto-recovery
    - Any state -> SLEEP (shutdown)
"""

from __future__ import annotations

import asyncio
import enum
import logging
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
        AtomState.LISTENING,          # resume from sleep / silent mode
    }),
    AtomState.IDLE: frozenset({
        AtomState.LISTENING,          # start listening
        AtomState.SLEEP,              # shutdown
    }),
    AtomState.LISTENING: frozenset({
        AtomState.THINKING,           # speech finalized -> route to LLM
        AtomState.SPEAKING,           # fast-path: local command skips THINKING
        AtomState.IDLE,               # silence timeout -> back to idle
        AtomState.ERROR_RECOVERY,     # STT failure
        AtomState.SLEEP,              # shutdown
    }),
    AtomState.THINKING: frozenset({
        AtomState.SPEAKING,           # first partial response arrived
        AtomState.LISTENING,          # LLM error -> let user retry
        AtomState.ERROR_RECOVERY,     # LLM failure/timeout
        AtomState.SLEEP,              # shutdown
    }),
    AtomState.SPEAKING: frozenset({
        AtomState.IDLE,               # tts_complete -> rest
        AtomState.LISTENING,          # barge-in during speech
        AtomState.ERROR_RECOVERY,     # TTS failure
        AtomState.SLEEP,              # shutdown
    }),
    AtomState.ERROR_RECOVERY: frozenset({
        AtomState.IDLE,               # auto-recovery
        AtomState.SLEEP,              # shutdown
    }),
}

class InvalidStateTransition(Exception):
    """Raised when a state change violates the transition table."""


class StateManager:
    """
    Manages the ATOM lifecycle through a strict state machine.

    Thread-safe: the transition is guarded by an asyncio.Lock.
    Read-only access to ``current`` is lock-free (CPython GIL guarantee).

    v10: Default initial state is IDLE.
    """

    __slots__ = ("_state", "_lock", "_bus", "_always_listen")

    def __init__(self, bus: AsyncEventBus, initial: AtomState = AtomState.IDLE) -> None:
        self._state = initial
        self._lock = asyncio.Lock()
        self._bus = bus
        self._always_listen = False
        logger.info("StateManager initialised -> %s", initial.value)

    @property
    def current(self) -> AtomState:
        # Atomic in CPython (GIL guarantees single-bytecode enum ref assignment).
        # If future no-GIL interpreters are used, wrap reads with self._lock.
        return self._state

    async def transition(self, new_state: AtomState) -> None:
        """
        Atomically move to *new_state* if the transition is legal.

        Emits ``state_changed`` with ``old`` and ``new`` kwargs.
        Silently ignores no-op transitions (current == new).
        Logs a warning and returns for illegal transitions.
        """
        async with self._lock:
            old = self._state
            if old is new_state:
                return
            if new_state not in VALID_TRANSITIONS[old]:
                logger.warning("Blocked illegal transition %s -> %s", old.value, new_state.value)
                return
            self._state = new_state
            logger.info("State: %s -> %s", old.value, new_state.value)

        self._bus.emit("state_changed", old=old, new=new_state)

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
        if self._state is AtomState.LISTENING:
            if self._always_listen:
                self._bus.emit("restart_listening")
            else:
                await self.transition(AtomState.IDLE)

    async def on_error(self, source: str = "unknown", **_kw) -> None:
        """Transition to ERROR_RECOVERY on component failure, then auto-recover to IDLE."""
        if self._state in (AtomState.LISTENING, AtomState.THINKING, AtomState.SPEAKING):
            logger.error("Error from %s in state %s -- entering recovery", source, self._state.value)
            await self.transition(AtomState.ERROR_RECOVERY)
            await self.transition(AtomState.IDLE)

