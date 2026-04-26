"""
ATOM -- Voice interrupt coordinator.

Coordinates barge-in across STT, TTS, state transitions, and optional
cross-worker interrupts so a new utterance can take over cleanly.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
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


_PARTIAL_BURST_WINDOW_S = 0.5
_PARTIAL_BURST_THRESHOLD = 3
# A single 1-token partial like "Dear" while ATOM is SPEAKING is almost
# always the mic catching ATOM's own voice. Require either the burst
# threshold or at least this many words before we treat a partial as a
# real barge-in. A real interruption nearly always crosses 2 tokens
# within 200ms ("hey atom", "stop please", "no no").
_PARTIAL_MIN_WORDS_FOR_INTERRUPT = 2
_THINKING_PARTIAL_INTERRUPT_RE = re.compile(
    r"\b("
    r"stop|cancel|abort|pause|wait|hold on|never mind|nevermind|"
    r"leave it|forget it|shut up"
    r")\b",
    re.I,
)


class VoiceInterruptHandler:
    """Coordinate speech-driven and manual interrupts into LISTENING.

    Supports predictive barge-in (pre-stop on rapid partial bursts),
    soft pause/resume, and "as I was saying" resume capability.
    """

    def __init__(
        self,
        *,
        bus: Any,
        state: Any,
        tts: Any,
        interrupt_manager: Any = None,
        local_brain: Any = None,
        llm_queue: Any = None,
        indicator: Any = None,
        command_loop: Any = None,
        emit_cooldown_s: float = 0.15,
    ) -> None:
        self._bus = bus
        self._state = state
        self._tts = tts
        self._interrupt_mgr = interrupt_manager
        self._local_brain = local_brain
        self._llm_queue = llm_queue
        self._indicator = indicator
        self._command_loop = command_loop
        self._emit_cooldown_s = max(0.0, float(emit_cooldown_s))
        self._lock = asyncio.Lock()
        self._last_resume_emit = -self._emit_cooldown_s

        self._partial_timestamps: list[float] = []
        self._paused = False
        self._interrupted_text: str = ""
        self._interrupted_position: int = 0

    def attach_command_loop(self, command_loop: Any) -> None:
        """Late-bind the CommandLoop (may not exist at construction time)."""
        self._command_loop = command_loop

    @staticmethod
    def partial_indicates_voice_interrupt(text: str) -> bool:
        """True when partial STT output likely means real user speech."""
        normalized = " ".join((text or "").strip().lower().split())
        if not normalized:
            return False
        return normalized not in _NON_INTERRUPT_STATUSES

    @staticmethod
    def partial_indicates_thinking_interrupt(text: str) -> bool:
        """True when a THINKING partial is an explicit abort command."""
        normalized = " ".join((text or "").strip().lower().split())
        if not normalized or normalized in _NON_INTERRUPT_STATUSES:
            return False
        return bool(_THINKING_PARTIAL_INTERRUPT_RE.search(normalized))

    async def on_speech_partial(self, text: str = "", **_kw: Any) -> None:
        """Early interrupt path from STT partials while TTS is speaking or thinking.

        Includes predictive barge-in: if partials arrive in rapid bursts
        (3+ within 500ms), pre-pause TTS before a full word forms.

        Echo suppression: skips partials that match ATOM's own recent TTS
        output to prevent the mic-speaker feedback loop on MacBook.
        """
        from core.state_manager import AtomState

        cur = self._state.current
        if cur not in (AtomState.SPEAKING, AtomState.THINKING):
            self._partial_timestamps.clear()
            return
        if not self.partial_indicates_voice_interrupt(text):
            return

        tts = self._tts
        if tts is not None and hasattr(tts, "is_echo") and tts.is_echo(text):
            logger.info(
                "Echo suppressed (TTS self-feedback): '%s'", (text or "")[:80],
            )
            return

        if cur is AtomState.THINKING and not self.partial_indicates_thinking_interrupt(text):
            logger.debug(
                "Ignoring non-explicit THINKING partial interrupt: '%s'",
                (text or "")[:80],
            )
            return

        # Defensive floor on single partials. A 1-token partial like "Dear"
        # captured while we are still SPEAKING is almost always our own
        # voice; require at least N words before treating it as a real
        # barge-in. The burst path below still fires when the user really
        # is talking over us (3 partials in 500ms).
        word_count = len((text or "").split())

        now = time.monotonic()

        self._partial_timestamps.append(now)
        cutoff = now - _PARTIAL_BURST_WINDOW_S
        self._partial_timestamps = [
            t for t in self._partial_timestamps if t >= cutoff
        ]

        if (
            len(self._partial_timestamps) >= _PARTIAL_BURST_THRESHOLD
            and cur is AtomState.SPEAKING
            and not self._paused
        ):
            logger.info(
                "Predictive barge-in: %d partials in %.0fms, pausing TTS",
                len(self._partial_timestamps),
                _PARTIAL_BURST_WINDOW_S * 1000,
            )
            await self._pause_tts()
            self._partial_timestamps.clear()
            return

        if (
            cur is AtomState.SPEAKING
            and word_count < _PARTIAL_MIN_WORDS_FOR_INTERRUPT
        ):
            # Too thin a signal to cut TTS off; wait for the burst path
            # or a richer partial / final.
            return

        if self._emit_cooldown_s > 0 and (now - self._last_resume_emit) < self._emit_cooldown_s:
            return

        self._last_resume_emit = now

        if self._paused:
            logger.info("User confirmed speech during pause, escalating to full stop")
            await self._hard_stop_from_pause()

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
                if self._command_loop is not None:
                    try:
                        await self._command_loop.cancel_current()
                    except Exception:
                        logger.debug("Voice interrupt command_loop cancel failed", exc_info=True)
                if self._interrupt_mgr is not None:
                    await self._interrupt_mgr.broadcast_interrupt()
                if self._local_brain is not None:
                    try:
                        self._local_brain.request_preempt()
                    except Exception:
                        logger.debug("Voice interrupt brain preempt failed", exc_info=True)
                if self._llm_queue is not None:
                    try:
                        await self._llm_queue.clear_pending()
                    except Exception:
                        logger.debug("Voice interrupt LLM queue clear failed", exc_info=True)

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

    async def _pause_tts(self) -> None:
        """Soft pause: suspend TTS via NSSpeechSynthesizer boundary pause.

        Stores the interrupted state for potential resume.
        """
        tts = self._tts
        if tts is None:
            return

        try:
            native = getattr(tts, "_native_synth", None)
            if native is not None:
                synth = getattr(native, "_synth", None)
                if synth is not None and hasattr(synth, "pauseSpeakingAtBoundary_"):
                    synth.pauseSpeakingAtBoundary_(1)
                    self._paused = True
                    logger.info("TTS paused at word boundary")

                    if hasattr(tts, "_stream_speak_buffer"):
                        self._interrupted_text = tts._stream_speak_buffer or ""
                    return

            self._paused = True
            logger.debug("TTS pause: no native synth, marking paused flag only")
        except Exception:
            logger.debug("TTS pause failed", exc_info=True)

    async def _resume_tts(self) -> None:
        """Resume TTS from a soft pause."""
        tts = self._tts
        if tts is None or not self._paused:
            return

        try:
            native = getattr(tts, "_native_synth", None)
            if native is not None:
                synth = getattr(native, "_synth", None)
                if synth is not None and hasattr(synth, "continueSpeaking"):
                    synth.continueSpeaking()
                    logger.info("TTS resumed from pause")

            self._paused = False
        except Exception:
            logger.debug("TTS resume failed", exc_info=True)
            self._paused = False

    async def _hard_stop_from_pause(self) -> None:
        """Escalate from pause to full stop when user confirms speech."""
        self._paused = False
        await self._stop_tts()

    def get_interrupted_context(self) -> str | None:
        """Return the text that was being spoken when interrupted.

        Used by the response pipeline to offer: "As I was saying, ..."
        """
        text = self._interrupted_text
        if text:
            self._interrupted_text = ""
            return text
        return None

    def clear_interrupted(self) -> None:
        """Clear the interrupted context after it's been used."""
        self._interrupted_text = ""
        self._interrupted_position = 0

    async def _stop_tts(self) -> None:
        """Immediately kill TTS -- not graceful, instant silence.

        Uses ``force_stop()`` (public API) when available so we never
        mutate private TTS fields from outside the TTS module.
        """
        tts = self._tts
        if tts is None:
            return

        if self._paused:
            self._paused = False

        try:
            force_fn = getattr(tts, "force_stop", None)
            if callable(force_fn):
                result = force_fn()
                if inspect.isawaitable(result):
                    await result
                return

            stop_fn = getattr(tts, "stop", None)
            if callable(stop_fn):
                result = stop_fn()
                if inspect.isawaitable(result):
                    await result
        except Exception:
            logger.debug("Voice interrupt TTS stop failed", exc_info=True)


__all__ = ["VoiceInterruptHandler"]
