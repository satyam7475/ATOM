"""
ATOM v13 -- Persistent SAPI COM TTS via comtypes.

Single SpVoice COM object created at startup in a dedicated thread.
Subsequent Speak() calls reuse it with zero init overhead (~5ms).
Markdown is stripped via clean_for_tts() before speaking.

Features:
- SPF_ASYNC (non-blocking speak, auto-queues)
- SPF_PURGEBEFORESPEAK (instant barge-in stop)
- Streaming partial_response (sentence-by-sentence as the LLM streams)
"""

from __future__ import annotations

import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

logger = logging.getLogger("atom.tts")

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.state_manager import StateManager

SPF_ASYNC = 1
SPF_PURGEBEFORESPEAK = 2

_RE_CODE_BLOCK = re.compile(r'```.*?```', re.DOTALL)
_RE_INLINE_CODE = re.compile(r'`([^`]*)`')
_RE_BOLD = re.compile(r'\*\*([^*]+)\*\*')
_RE_ITALIC_STAR = re.compile(r'\*([^*]+)\*')
_RE_ITALIC_UNDER = re.compile(r'_([^_]+)_')
_RE_HEADER = re.compile(r'^#+\s*', re.MULTILINE)
_RE_BULLET = re.compile(r'^\s*[-*\u2022]\s+', re.MULTILINE)
_RE_NUMBERED = re.compile(r'^\s*\d+\.\s+', re.MULTILINE)
_RE_BLOCKQUOTE = re.compile(r'^\s*>\s+', re.MULTILINE)


def clean_for_tts(text: str) -> str:
    """Strip markdown formatting so SAPI speaks clean plain text."""
    text = _RE_CODE_BLOCK.sub('', text)
    text = _RE_INLINE_CODE.sub(r'\1', text)
    text = _RE_BOLD.sub(r'\1', text)
    text = _RE_ITALIC_STAR.sub(r'\1', text)
    text = _RE_ITALIC_UNDER.sub(r'\1', text)
    text = _RE_HEADER.sub('', text)
    text = _RE_BULLET.sub('', text)
    text = _RE_NUMBERED.sub('', text)
    text = _RE_BLOCKQUOTE.sub('', text)
    return text.strip()


def _truncate(text: str, max_lines: int = 4) -> str:
    text = clean_for_tts(text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) <= max_lines:
        return " ".join(lines)
    return " ".join(lines[:max_lines])


class TTSAsync:
    """Persistent SAPI COM TTS -- near-zero start latency, cancelable."""

    def __init__(
        self,
        bus: AsyncEventBus,
        state: StateManager,
        max_lines: int = 4,
        rate: int = 1,
    ) -> None:
        self._bus = bus
        self._state = state
        self._max_lines = max_lines
        self._rate = rate
        self._voice = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")

    # -- COM lifecycle (runs in dedicated TTS thread) --

    def _init_voice(self) -> None:
        """Initialize COM and create SpVoice. Called once in the TTS thread."""
        import comtypes
        import comtypes.client
        comtypes.CoInitialize()
        self._voice = comtypes.client.CreateObject("SAPI.SpVoice")
        self._voice.Rate = self._rate
        logger.info("SAPI COM voice ready (rate=%d)", self._rate)

    async def init_voice(self) -> None:
        """Preload the COM voice object at startup (non-blocking)."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._init_voice)

    # -- Core speak / stop (run in TTS executor thread) --

    def _speak_sync(self, text: str) -> None:
        """Queue text for async SAPI speech, then block until done."""
        if not self._voice:
            return
        self._voice.Speak(text, SPF_ASYNC)
        self._voice.WaitUntilDone(-1)

    def _speak_async_queue(self, text: str) -> None:
        """Queue text behind any in-flight speech (no wait)."""
        if not self._voice:
            return
        self._voice.Speak(text, SPF_ASYNC)

    def _stop_sync(self) -> None:
        """Immediately purge all queued speech."""
        if not self._voice:
            return
        self._voice.Speak("", SPF_PURGEBEFORESPEAK)

    # -- Public async API --

    async def speak(self, text: str) -> None:
        text = _truncate(text, self._max_lines)
        if not text:
            self._bus.emit("tts_complete")
            return

        logger.info("TTS speaking: '%s'", text[:100])
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._speak_sync, text)
        except asyncio.CancelledError:
            await asyncio.get_running_loop().run_in_executor(
                self._executor, self._stop_sync
            )
            raise
        except Exception:
            logger.exception("TTS error")

        self._bus.emit("tts_complete")

    async def speak_ack(self, phrase: str) -> None:
        """Speak a very short acknowledgement phrase (non-blocking).

        Queued via SPF_ASYNC so it plays immediately.  The first streaming
        partial_response from the brain will naturally queue behind this short
        phrase, giving the user instant audio feedback while the LLM thinks.
        """
        if not phrase:
            return
        logger.info("TTS ack: '%s'", phrase)
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._speak_async_queue, phrase)
        except Exception:
            logger.exception("TTS ack error")

    async def stop(self) -> None:
        """Barge-in: immediately stop all speech."""
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._stop_sync)
        except Exception:
            logger.exception("TTS stop error")

    # -- Event handlers --

    async def on_response(self, text: str, is_exit: bool = False, **_kw) -> None:
        """Handle full response_ready event.

        Spawns speech as a background task so the EventBus handler returns
        immediately and doesn't hit the 10-second timeout for long responses.
        """
        from core.state_manager import AtomState

        if self._state.current is AtomState.SPEAKING:
            return

        await self._state.transition(AtomState.SPEAKING)

        async def _speak_bg() -> None:
            try:
                await self.speak(text)
            except Exception:
                logger.exception("TTS background speak error")
                self._bus.emit("tts_complete")
            if is_exit:
                self._bus.emit("shutdown_requested")

        asyncio.create_task(_speak_bg())

    async def on_partial_response(self, text: str, is_first: bool = False, is_last: bool = False, **_kw) -> None:
        """Handle streaming partial_response events from the LLM.

        First chunk transitions to SPEAKING and starts audio immediately.
        Subsequent chunks queue behind the current speech via SPF_ASYNC.
        Last chunk spawns a background task that waits for all queued speech
        to finish, then emits tts_complete.  This prevents the event-bus
        handler timeout (10 s) from killing the handler mid-speech.
        """
        from core.state_manager import AtomState

        text = text.strip()
        if not text and not is_last:
            return

        if is_first:
            await self._state.transition(AtomState.SPEAKING)

        truncated = _truncate(text, self._max_lines) if text else ""

        if truncated:
            logger.info("TTS partial: '%s'", truncated[:80])

        loop = asyncio.get_running_loop()

        if is_last:
            asyncio.create_task(self._finish_speaking(loop, truncated))
        elif truncated:
            try:
                await loop.run_in_executor(self._executor, self._speak_async_queue, truncated)
            except Exception:
                logger.exception("TTS partial error")

    async def _finish_speaking(self, loop: asyncio.AbstractEventLoop, text: str) -> None:
        """Background task: speak final chunk, wait for SAPI to finish, emit tts_complete."""
        try:
            if text:
                await loop.run_in_executor(self._executor, self._speak_sync, text)
            else:
                await loop.run_in_executor(self._executor, self._wait_until_done)
        except Exception:
            logger.exception("TTS finish error")
        self._bus.emit("tts_complete")

    def _wait_until_done(self) -> None:
        """Block until all queued SAPI speech finishes."""
        if self._voice:
            self._voice.WaitUntilDone(-1)

    # -- Shutdown --

    async def shutdown(self) -> None:
        self._stop_sync()
        if self._voice:
            try:
                import comtypes
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self._executor, comtypes.CoUninitialize)
            except Exception:
                pass
        self._executor.shutdown(wait=False)
        logger.info("TTS shut down (SAPI COM released)")
