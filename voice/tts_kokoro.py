"""
ATOM -- Kokoro TTS Engine (JARVIS-Level Local Voice).

Ultra-fast, fully offline neural TTS via the kokoro-onnx runtime
(82M-param Kokoro model, ~24kHz output, real-time on Apple Silicon).

Setup
-----
Three things need to be on disk before this engine becomes available:

1. ``pip install kokoro-onnx sounddevice`` (handled by requirements.txt).
2. ``brew install espeak-ng`` -- required for the phonemizer; without
   it the ONNX runtime cannot synthesise speech.
3. Model + voices files. Default path is ``models/kokoro/``::

       kokoro-v1.0.onnx     (~310 MB)
       voices-v1.0.bin      (~24 MB)

   Run ``python scripts/install_kokoro.py`` to download both, or set
   ``tts.kokoro_model_path`` / ``tts.kokoro_voices_path`` in
   ``config/settings.json`` to point at custom locations.

Until all three exist this engine reports ``available=False`` and
voice_pipeline will fall back to the macOS Native voice (``Daniel``)
without crashing the boot path.

Implements the same public interface as MacOSTTSAsync / EdgeTTSAsync
so it can be used as a drop-in replacement via config.

Owner: Satyam
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger("atom.tts.kokoro")

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.state_manager import StateManager

_DEFAULT_MODEL_REL = "models/kokoro/kokoro-v1.0.onnx"
_DEFAULT_VOICES_REL = "models/kokoro/voices-v1.0.bin"
_DEFAULT_LANG = "en-us"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


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
        *,
        model_path: str | Path | None = None,
        voices_path: str | Path | None = None,
        speed: float = 1.0,
        language: str = _DEFAULT_LANG,
    ) -> None:
        self._bus = bus
        self._state = state
        self._max_lines = max_lines
        self._voice = voice
        self._speed = float(speed)
        self._language = language
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="kokoro",
        )
        self._running = False
        self._current_task: asyncio.Task | None = None
        self._model: Any = None
        self._available = False
        self._emotion: str = "neutral"
        self._stream_buffer: str = ""

        root = _project_root()
        self._model_path = Path(model_path) if model_path else root / _DEFAULT_MODEL_REL
        self._voices_path = Path(voices_path) if voices_path else root / _DEFAULT_VOICES_REL

        self._init_model()

    def _init_model(self) -> None:
        if not self._model_path.exists():
            logger.warning(
                "Kokoro TTS unavailable -- model file missing at %s "
                "(run `python scripts/install_kokoro.py`)",
                self._model_path,
            )
            return
        if not self._voices_path.exists():
            logger.warning(
                "Kokoro TTS unavailable -- voices file missing at %s "
                "(run `python scripts/install_kokoro.py`)",
                self._voices_path,
            )
            return
        try:
            from kokoro_onnx import Kokoro
        except ImportError:
            logger.warning(
                "Kokoro TTS unavailable -- run `pip install kokoro-onnx sounddevice`",
            )
            return
        try:
            self._model = Kokoro(
                str(self._model_path), str(self._voices_path),
            )
        except Exception as exc:
            msg = str(exc).lower()
            if "espeak" in msg or "phonemizer" in msg:
                logger.warning(
                    "Kokoro TTS unavailable -- espeak-ng not on PATH "
                    "(install via `brew install espeak-ng`): %s", exc,
                )
            else:
                logger.error(
                    "Failed to initialize Kokoro TTS: %s", exc,
                )
            return
        self._available = True
        logger.info(
            "Kokoro TTS initialized (voice=%s speed=%.2f lang=%s)",
            self._voice, self._speed, self._language,
        )

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

            t0 = time.monotonic()
            await self._stream_and_play(clean_text)
            elapsed = time.monotonic() - t0
            logger.info("Kokoro TTS completed in %.2fs", elapsed)

        except asyncio.CancelledError:
            logger.debug("Kokoro TTS task cancelled")
        except Exception:
            logger.exception("Kokoro TTS error")
        finally:
            self._running = False
            self._bus.emit("tts_complete")

    async def _stream_and_play(self, text: str) -> None:
        """Stream Kokoro chunks straight into a single sounddevice
        OutputStream so the user hears the first phoneme in <200 ms
        instead of waiting for the whole sentence to synthesise.
        """
        if not self._running or not self._model:
            return
        try:
            import sounddevice as sd
        except ImportError:
            logger.warning(
                "sounddevice missing -- install with `pip install sounddevice`",
            )
            return

        loop = asyncio.get_running_loop()
        stream = None
        sample_rate = 0
        try:
            async for audio, sr in self._model.create_stream(
                text,
                voice=self._voice,
                speed=self._speed,
                lang=self._language,
            ):
                if not self._running:
                    break
                if stream is None:
                    sample_rate = int(sr)
                    stream = sd.OutputStream(
                        samplerate=sample_rate,
                        channels=1,
                        dtype="float32",
                    )
                    await loop.run_in_executor(self._executor, stream.start)
                await loop.run_in_executor(
                    self._executor, stream.write, audio,
                )
        except Exception:
            logger.exception("Kokoro playback error")
        finally:
            if stream is not None:
                try:
                    await loop.run_in_executor(self._executor, stream.stop)
                    await loop.run_in_executor(self._executor, stream.close)
                except Exception:
                    pass

    # ── Lifecycle ────────────────────────────────────────────────────

    def shutdown(self) -> None:
        self._running = False
        task = self._current_task
        if task is not None and not task.done():
            task.cancel()
        self._executor.shutdown(wait=False)
        logger.info("Kokoro TTS shut down")


__all__ = ["KokoroTTSAsync"]
