"""
ATOM -- Kokoro TTS Engine (JARVIS-Level Local Voice).

Ultra-fast, fully offline, natural neural TTS.
Requires: pip install kokoro-tts sounddevice

Implements the same public interface as MacOSTTSAsync / EdgeTTSAsync
so it can be used as a drop-in replacement via config.

Owner: Satyam
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

logger = logging.getLogger("atom.tts.kokoro")

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.state_manager import StateManager


class KokoroTTSAsync:
    """Offline neural TTS using Kokoro (ultra-low latency).

    Matches the public interface expected by ``wire_events`` in
    ``core/boot/wiring.py``: ``on_response``, ``on_partial_response``,
    ``on_speech_partial``, ``speak``, ``speak_ack``, ``stop``,
    ``force_stop``, ``init_voice``, ``set_emotion``, ``shutdown``.
    """

    def __init__(
        self,
        bus: AsyncEventBus,
        state: StateManager,
        max_lines: int = 4,
        voice: str = "af_heart",
    ) -> None:
        self._bus = bus
        self._state = state
        self._max_lines = max_lines
        self._voice = voice
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kokoro")
        self._running = False
        self._current_task: asyncio.Task | None = None
        self._model: Any = None
        self._available = False
        self._emotion: str = "neutral"
        self._stream_buffer: str = ""

        self._init_model()

    def _init_model(self) -> None:
        try:
            from kokoro_tts import Kokoro
            self._model = Kokoro(voice=self._voice)
            self._available = True
            logger.info("Kokoro TTS initialized with voice: %s", self._voice)
        except ImportError:
            logger.error(
                "kokoro-tts not installed. Run: pip install kokoro-tts sounddevice"
            )
        except Exception as e:
            logger.error("Failed to initialize Kokoro TTS: %s", e)

    # ── Public interface (matches MacOSTTSAsync) ─────────────────────

    async def init_voice(self) -> None:
        """No-op — Kokoro model is loaded in ``__init__``."""

    async def speak(self, text: str, **_kw: Any) -> None:
        """Speak text with Kokoro TTS."""
        if not (text or "").strip():
            return

        if not self._available:
            logger.warning("Kokoro TTS unavailable, skipping speech")
            self._bus.emit("tts_complete")
            return

        await self.stop()
        self._running = True
        self._current_task = asyncio.create_task(self._speak_task(text))

    async def speak_ack(self, text: str) -> None:
        """Speak a short acknowledgment."""
        await self.speak(text)

    def set_emotion(self, emotion: str) -> None:
        """Store emotion hint for future prosody control."""
        self._emotion = emotion or "neutral"

    def next_ack_phrase(self) -> str:
        return "On it, Boss."

    async def stop(self) -> None:
        """Immediately stop current speech."""
        self._running = False
        task = self._current_task
        if task is not None and not task.done():
            task.cancel()
        try:
            import sounddevice as sd
            sd.stop()
        except ImportError:
            pass

    async def force_stop(self) -> None:
        """Hard stop for the interrupt handler."""
        await self.stop()

    # ── Event handlers (wired by core/boot/wiring.py) ────────────────

    async def on_response(
        self, text: str, is_exit: bool = False, is_sleep: bool = False, **_kw: Any,
    ) -> None:
        """Handle a complete response from the router."""
        from core.state_manager import AtomState

        if self._running:
            await self.stop()

        if not (text or "").strip():
            self._bus.emit("tts_complete")
            return

        await self._state.transition(AtomState.SPEAKING)
        await self.speak(text)

        if is_exit:
            self._bus.emit("shutdown_requested")

    _STREAM_FLUSH_WORDS: int = 12

    async def on_partial_response(
        self,
        text: str,
        is_first: bool = False,
        is_last: bool = False,
        source: str = "",
        stream_id: str = "",
        **_kw: Any,
    ) -> None:
        """Accumulate streaming tokens and flush to speech periodically.

        Instead of waiting for ``is_last`` (which can take many seconds for
        long responses), flush every ``_STREAM_FLUSH_WORDS`` words so the
        user hears continuous speech while the LLM is still generating.
        """
        if is_first:
            self._stream_buffer = ""

        if (text or "").strip():
            self._stream_buffer += " " + text.strip()

        buf = self._stream_buffer.strip()
        word_count = len(buf.split()) if buf else 0

        if is_last:
            if buf:
                await self.speak(buf)
                self._stream_buffer = ""
        elif word_count >= self._STREAM_FLUSH_WORDS:
            await self.speak(buf)
            self._stream_buffer = ""

    async def on_speech_partial(self, text: str = "", **_kw: Any) -> None:
        """Barge-in: stop if user speaks while Kokoro is playing."""
        from core.state_manager import AtomState

        if self._state.current is AtomState.SPEAKING:
            await self.stop()

    async def on_state_changed(self, old: Any, new: Any, **_kw: Any) -> None:
        from core.state_manager import AtomState

        if new is AtomState.LISTENING and old is AtomState.SPEAKING:
            await self.stop()

    # ── Governor hooks (no-ops for compatibility) ────────────────────

    def set_postprocess(self, enabled: bool) -> None:
        pass

    def restore_postprocess(self) -> None:
        pass

    def refresh_output_device(self) -> bool:
        return False

    # ── Internals ────────────────────────────────────────────────────

    async def _speak_task(self, text: str) -> None:
        from core.state_manager import AtomState

        try:
            if self._state.current is not AtomState.SPEAKING:
                await self._state.transition(AtomState.SPEAKING)

            lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
            if len(lines) > self._max_lines:
                text = " ".join(lines[: self._max_lines]) + " ... and more."

            clean_text = text.replace("*", "").replace("_", "").replace("`", "")

            loop = asyncio.get_running_loop()
            t0 = time.monotonic()

            await loop.run_in_executor(
                self._executor, self._generate_and_play, clean_text,
            )

            elapsed = time.monotonic() - t0
            logger.info("Kokoro TTS completed in %.2fs", elapsed)

        except asyncio.CancelledError:
            logger.debug("Kokoro TTS task cancelled")
        except Exception:
            logger.exception("Kokoro TTS error")
        finally:
            self._running = False
            self._bus.emit("tts_complete")

    def _generate_and_play(self, text: str) -> None:
        """Blocking: generate and play audio chunks via Kokoro."""
        if not self._running or not self._model:
            return
        try:
            import sounddevice as sd

            for audio, sr in self._model.create_stream(text):
                if not self._running:
                    break
                sd.play(audio, sr)
                sd.wait()
        except Exception as e:
            logger.error("Kokoro playback error: %s", e)

    # ── Lifecycle ────────────────────────────────────────────────────

    def shutdown(self) -> None:
        self._running = False
        task = self._current_task
        if task is not None and not task.done():
            task.cancel()
        self._executor.shutdown(wait=False)
        logger.info("Kokoro TTS shut down")


__all__ = ["KokoroTTSAsync"]
