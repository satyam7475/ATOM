"""
ATOM -- Voice interrupt coordinator.

Coordinates barge-in across STT, TTS, state transitions, and optional
cross-worker interrupts so a new utterance can take over cleanly.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import Any

logger = logging.getLogger("atom.voice_interrupt")

_LISTENING_STATUSES = frozenset({
    "listening",
    "listening...",
    "listening…",
})
_NON_INTERRUPT_STATUSES = _LISTENING_STATUSES | frozenset({
    "processing",
    "processing...",
    "processing…",
})


class VoiceInterruptHandler:
    """Coordinate speech-driven and manual interrupts into LISTENING."""

    def __init__(
        self,
        *,
        bus: Any,
        state: Any,
        tts: Any,
        interrupt_manager: Any = None,
        local_brain: Any = None,
        indicator: Any = None,
        emit_cooldown_s: float = 0.35,
    ) -> None:
        self._bus = bus
        self._state = state
        self._tts = tts
        self._interrupt_mgr = interrupt_manager
        self._local_brain = local_brain
        self._indicator = indicator
        self._emit_cooldown_s = max(0.0, float(emit_cooldown_s))
        self._lock = asyncio.Lock()
        self._last_resume_emit = -self._emit_cooldown_s

    @staticmethod
    def partial_indicates_voice_interrupt(text: str) -> bool:
        """True when partial STT output likely means real user speech."""
        normalized = " ".join((text or "").strip().lower().split())
        if not normalized:
            return False
        return normalized not in _NON_INTERRUPT_STATUSES

    async def on_speech_partial(self, text: str = "", **_kw: Any) -> None:
        """Early interrupt path from STT partials while TTS is speaking."""
        from core.state_manager import AtomState

        if self._state.current is not AtomState.SPEAKING:
            return
        if not self.partial_indicates_voice_interrupt(text):
            return

        now = time.monotonic()
        if self._emit_cooldown_s > 0 and (now - self._last_resume_emit) < self._emit_cooldown_s:
            return

        self._last_resume_emit = now
        logger.info("Voice interrupt partial detected: '%s'", (text or "")[:80])
        self._bus.emit_fast(
            "resume_listening",
            source="voice_interrupt",
            reason="speech_partial",
            partial_text=(text or "")[:160],
            user_interrupt=True,
        )

    async def prepare_for_new_speech(self, text: str = "", **_kw: Any) -> bool:
        """Ensure stale speech/thinking is interrupted before routing new speech."""
        from core.state_manager import AtomState

        if self._state.current not in (AtomState.SPEAKING, AtomState.THINKING):
            return False
        return await self.interrupt_to_listening(
            trigger="speech_final",
            reason="new_speech",
            partial_text=text,
            user_interrupt=True,
        )

    async def interrupt_to_listening(
        self,
        *,
        trigger: str,
        reason: str = "",
        partial_text: str = "",
        user_interrupt: bool = False,
    ) -> bool:
        """Move the system into LISTENING and stop stale output if needed."""
        from core.state_manager import AtomState

        async with self._lock:
            current = self._state.current

            if current is AtomState.SLEEP:
                logger.info("Voice interrupt leaving SLEEP via %s", trigger)
                await self._state.transition(AtomState.LISTENING)
                if self._indicator is not None:
                    try:
                        self._indicator.add_log("action", "I'm back, Boss.")
                    except Exception:
                        logger.debug("Voice interrupt indicator wake log failed", exc_info=True)
                return True

            if current is AtomState.ERROR_RECOVERY:
                logger.info("Voice interrupt recovering from ERROR_RECOVERY via %s", trigger)
                await self._state.transition(AtomState.IDLE)
                current = self._state.current

            if current is AtomState.LISTENING:
                return False

            if current in (AtomState.SPEAKING, AtomState.THINKING):
                if self._interrupt_mgr is not None:
                    await self._interrupt_mgr.broadcast_interrupt()
                if self._local_brain is not None:
                    try:
                        self._local_brain.request_preempt()
                    except Exception:
                        logger.debug("Voice interrupt brain preempt failed", exc_info=True)

            if current is AtomState.SPEAKING:
                await self._stop_tts()

            if user_interrupt:
                try:
                    self._bus.emit_fast(
                        "user_interrupt",
                        trigger=trigger,
                        reason=reason,
                        text=(partial_text or "")[:160],
                    )
                except Exception:
                    logger.debug("Voice interrupt user_interrupt emit failed", exc_info=True)

            if current is AtomState.THINKING and self._indicator is not None:
                try:
                    self._indicator.add_log("info", "Interrupted. Go ahead, Boss.")
                except Exception:
                    logger.debug("Voice interrupt indicator log failed", exc_info=True)

            await self._state.transition(AtomState.LISTENING)
            logger.info(
                "Voice interrupt -> LISTENING (trigger=%s reason=%s)",
                trigger,
                reason or "n/a",
            )
            return True

    async def _stop_tts(self) -> None:
        stop_fn = getattr(self._tts, "stop", None)
        if not callable(stop_fn):
            return
        try:
            result = stop_fn()
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.debug("Voice interrupt TTS stop failed", exc_info=True)


__all__ = ["VoiceInterruptHandler"]
