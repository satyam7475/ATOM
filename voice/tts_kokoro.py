"""
ATOM -- Kokoro TTS Engine (JARVIS-Level Local Voice).

Ultra-fast, fully offline, natural neural TTS.
Requires: pip install kokoro-tts sounddevice

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
    """Offline neural TTS using Kokoro (ultra-low latency)."""

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
        self._model = None
        self._available = False

        self._init_model()

    def _init_model(self):
        try:
            from kokoro_tts import Kokoro
            self._model = Kokoro(voice=self._voice)
            self._available = True
            logger.info("Kokoro TTS initialized with voice: %s", self._voice)
        except ImportError:
            logger.error("kokoro-tts not installed. Run: pip install kokoro-tts sounddevice")
            self._available = False
        except Exception as e:
            logger.error("Failed to initialize Kokoro TTS: %s", e)
            self._available = False

    async def speak(self, text: str, **_kw) -> None:
        """Speak text with Kokoro TTS."""
        if not text.strip():
            return

        if not self._available:
            logger.warning("Kokoro TTS unavailable, skipping speech")
            self._bus.emit("tts_done")
            return

        self.stop()
        self._running = True
        self._current_task = asyncio.create_task(self._speak_task(text))

    async def speak_ack(self, text: str) -> None:
        """Speak a short acknowledgment."""
        await self.speak(text)

    async def _speak_task(self, text: str) -> None:
        from core.state_manager import AtomState
        try:
            await self._state.transition(AtomState.SPEAKING)
            
            # Truncate if too long (similar to EdgeTTS)
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            if len(lines) > self._max_lines:
                text = " ".join(lines[:self._max_lines]) + " ... and more."

            # Clean up text for TTS
            clean_text = text.replace("*", "").replace("_", "").replace("`", "")

            loop = asyncio.get_running_loop()
            t0 = time.monotonic()
            
            # Run Kokoro generation and playback in executor
            await loop.run_in_executor(self._executor, self._generate_and_play, clean_text)
            
            elapsed = time.monotonic() - t0
            logger.info("Kokoro TTS completed in %.2fs", elapsed)

        except asyncio.CancelledError:
            logger.debug("Kokoro TTS task cancelled")
        except Exception as e:
            logger.exception("Kokoro TTS error: %s", e)
        finally:
            self._running = False
            self._bus.emit("tts_done")
            if self._state.current is AtomState.SPEAKING:
                await self._state.transition(AtomState.IDLE)

    def _generate_and_play(self, text: str):
        """Blocking call to generate and play audio via Kokoro."""
        if not self._running or not self._model:
            return
        try:
            import sounddevice as sd
            # Kokoro returns a generator of (audio_array, sample_rate)
            # We iterate and play chunks as they arrive for streaming latency
            for audio, sr in self._model.create_stream(text):
                if not self._running:
                    break
                sd.play(audio, sr)
                sd.wait() # Wait for chunk to finish playing
        except Exception as e:
            logger.error("Kokoro playback error: %s", e)

    def stop(self) -> None:
        """Stop current speech."""
        self._running = False
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
        
        try:
            import sounddevice as sd
            sd.stop()
        except ImportError:
            pass

    async def on_state_changed(self, old, new, **_kw) -> None:
        from core.state_manager import AtomState
        if new is AtomState.LISTENING and old is AtomState.SPEAKING:
            self.stop()

    def shutdown(self) -> None:
        self.stop()
        self._executor.shutdown(wait=False)
        logger.info("Kokoro TTS shut down")
