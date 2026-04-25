"""ATOM -- whisper.cpp STT backend (Sprint B2).

Drop-in replacement for ``voice.stt_macos.NativeSTT`` that runs
on-device whisper.cpp inference on Metal via ``pywhispercpp``. Cures
the SFSpeechRecognizer idle-timeout cliff that produced the 5
starvations-in-4-minutes cascade documented in atomLogs.txt
(L310, L437, L553, L571, L587, L589).

Architecture::

    Microphone (sounddevice 16-kHz mono PCM)
        v
    WebRTC VAD (30 ms frames, energy + harmonic gate)
        v
    Rolling 30 s ring buffer
        v
    whisper.cpp (Metal, small.en-q5_1)
        +-> partial transcript every 1.0 s of speech
        +-> final transcript after 600 ms of trailing silence
        v
    AsyncEventBus -- speech_partial / speech_final / voice.partial /
                     voice.final (same shape as NativeSTT)

Public surface mirrors :class:`NativeSTT` so the voice_pipeline factory
(B3) can swap engines without touching callers:

* ``preload`` / ``async_preload`` -- model + audio init.
* ``start_listening(loop, on_final, on_partial)`` -- begin streaming.
* ``stop_listening`` -- finalise pending audio, return last final.
* ``shutdown`` -- release model + audio.
* ``async_start_listening`` -- continuous background loop.
* properties: ``is_available``, ``is_listening``, ``backend_name``,
  ``mic_name``.

Event-bus events emitted (identical shape to NativeSTT)::

    speech_partial(text)
    speech_final(text, language)
    voice.partial(text, confidence, engine, mic)
    voice.final(text, language, confidence, engine, mic)

Latency budget: ~30 ms VAD + ~200-300 ms whisper-small.en-q5_1 transcribe
= sub-500 ms end-of-speech to final on M-series.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

logger = logging.getLogger("atom.stt_whisper")

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.state_manager import StateManager

# ── Optional native deps ──────────────────────────────────────────
# Soft-import so the module can be imported in CI / test envs without
# the heavy native binaries being installed. The factory falls back
# to NativeSTT in that case.
try:
    import pywhispercpp.model as _pwc_model  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - environment-specific
    _pwc_model = None  # type: ignore[assignment]

try:
    import sounddevice as _sd  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    _sd = None  # type: ignore[assignment]

try:
    import webrtcvad as _webrtcvad  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    _webrtcvad = None  # type: ignore[assignment]

try:
    import numpy as _np  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    _np = None  # type: ignore[assignment]


# ── Constants ─────────────────────────────────────────────────────

_SAMPLE_RATE = 16000
_FRAME_MS = 30
_FRAME_SAMPLES = (_SAMPLE_RATE * _FRAME_MS) // 1000  # 480
_RING_SECONDS = 30.0
_PARTIAL_INTERVAL_S = 1.0
_TRAILING_SILENCE_S = 0.6
_MAX_UTTERANCE_S = 20.0
_MIN_UTTERANCE_MS = 250

# Where install_whisper_model.py writes the GGML weights. As of
# Apr 25 2026 ggerganov/whisper.cpp removed the q5_0 small.en quant
# from HF (404), so we now default to the q5_1 variant of the same
# model. Old configs pointing at the q5_0 filename are auto-redirected
# in voice/whisper_install.py.
_DEFAULT_MODEL_FILENAME = "ggml-small.en-q5_1.bin"
_LEGACY_FILENAME_REDIRECTS = {
    "ggml-small.en-q5_0.bin": "ggml-small.en-q5_1.bin",
    "ggml-base.en-q5_0.bin": "ggml-base.en-q5_1.bin",
}


def _resolve_model_path(config_path: str | None) -> Path:
    """Return the effective ggml model path.

    Resolution order:
      1. explicit ``stt.whisper_model_path`` (if absolute).
      2. ``./models/<filename>`` if the explicit path is just a name.
      3. ``./models/ggml-small.en-q5_1.bin`` (default).
    """
    root = Path(__file__).resolve().parent.parent

    def _maybe_redirect(p: Path) -> Path:
        if p.name in _LEGACY_FILENAME_REDIRECTS:
            return p.with_name(_LEGACY_FILENAME_REDIRECTS[p.name])
        return p

    if config_path:
        p = _maybe_redirect(Path(config_path))
        if p.is_absolute():
            return p
        return (root / "models" / p.name).resolve()
    return (root / "models" / _DEFAULT_MODEL_FILENAME).resolve()


# ── Backend ───────────────────────────────────────────────────────


class WhisperSTT:
    """whisper.cpp Metal STT backend that mirrors NativeSTT's surface."""

    def __init__(
        self,
        bus: "AsyncEventBus",
        state: "StateManager",
        config: dict | None = None,
        mic_manager: Any = None,
        intent_engine: Any = None,
    ) -> None:
        self._bus = bus
        self._state = state
        full_cfg = config or {}
        self._config = full_cfg.get("stt", {})
        self._mic_manager = mic_manager
        self._intent_engine = intent_engine

        self._model_path: Path = _resolve_model_path(
            self._config.get("whisper_model_path"),
        )
        self._n_threads: int = int(self._config.get("whisper_n_threads", 4))
        self._language: str = str(self._config.get("whisper_language", "en"))
        self._partial_interval_s: float = float(
            self._config.get("whisper_partial_interval_s", _PARTIAL_INTERVAL_S),
        )
        self._trailing_silence_s: float = float(
            self._config.get("whisper_trailing_silence_s", _TRAILING_SILENCE_S),
        )
        self._max_utterance_s: float = float(
            self._config.get("whisper_max_utterance_s", _MAX_UTTERANCE_S),
        )
        # Sprint Ω9: adaptive end-of-turn detector. Built lazily in
        # ``preload`` only when stt.smart_turn_taker.enabled = true.
        # When unavailable, all decisions route to the legacy
        # ``trailing_silence_s`` ceiling so behaviour is unchanged.
        from voice.smart_turn_taker import (  # local import keeps boot cheap
            SmartTurnTaker,
            SmartTurnTakerConfig,
        )
        _stt_cfg = self._config.get("smart_turn_taker") or {}
        self._turn_taker: SmartTurnTaker | None = None
        if isinstance(_stt_cfg, dict) and _stt_cfg.get("enabled"):
            self._turn_taker = SmartTurnTaker(SmartTurnTakerConfig(
                enabled=True,
                sample_rate=int(_stt_cfg.get("sample_rate", _SAMPLE_RATE)),
                decision_window_s=float(_stt_cfg.get("decision_window_s", 1.0)),
                min_silence_s=float(_stt_cfg.get(
                    "min_silence_s", 0.18,
                )),
                max_silence_s=float(_stt_cfg.get(
                    "max_silence_s", max(self._trailing_silence_s, 1.20),
                )),
                eot_probability_threshold=float(_stt_cfg.get(
                    "eot_probability_threshold", 0.78,
                )),
                midthought_lockout_threshold=float(_stt_cfg.get(
                    "midthought_lockout_threshold", 0.92,
                )),
                min_eval_interval_ms=float(_stt_cfg.get(
                    "min_eval_interval_ms", 60.0,
                )),
            ))
        self._vad_aggressiveness: int = max(
            0, min(3, int(self._config.get("whisper_vad_aggressiveness", 2))),
        )
        # Apple's VPIO / hardware noise gate is irrelevant here -- we
        # gate on RMS in software so Bluetooth + USB mics behave the
        # same way.
        self._noise_floor_dbfs: float = float(
            self._config.get("noise_floor_dbfs", -55.0),
        )
        self.mic_name: str = "sounddevice (PortAudio/CoreAudio)"

        # State
        self._model: Any = None
        self._stream: Any = None
        self._available: bool = False
        self._listening: bool = False
        self._running_async: bool = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._on_final: Callable[[str], None] | None = None
        self._on_partial: Callable[[str], None] | None = None
        self._last_partial: str = ""
        self._last_final: str = ""
        self._last_confidence: float = 0.95
        self._last_error: str | None = None
        self._last_speech_time: float = 0.0
        self._tap_buffer_count: int = 0
        self._last_audio_rms_db: float = -96.0
        self._last_speech_candidate_time: float = 0.0
        self._permanently_disabled: bool = False

        # Audio + VAD ring
        self._ring_lock = threading.Lock()
        self._ring: collections.deque[bytes] = collections.deque(
            maxlen=int(_RING_SECONDS * _SAMPLE_RATE * 2 / _FRAME_SAMPLES),
        )
        self._vad: Any = None
        self._utterance_frames: list[bytes] = []
        self._silence_frames: int = 0
        self._utterance_started_at: float = 0.0
        self._last_partial_emit_at: float = 0.0
        self._worker_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

        # Echo guard (wired by VoicePipeline; identical contract to
        # NativeSTT._echo_guard).
        self._echo_guard: Callable[[str], bool] | None = None

    # ── public properties ───────────────────────────────────────

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def is_listening(self) -> bool:
        return self._listening

    @property
    def backend_name(self) -> str:
        if self._available:
            return f"whisper.cpp Metal ({self._model_path.name})"
        return "whisper.cpp (unavailable)"

    # ── lifecycle ───────────────────────────────────────────────

    def preload(self) -> bool:
        """Load whisper model + VAD. Idempotent."""
        if self._available:
            return True
        if _pwc_model is None:
            self._last_error = (
                "pywhispercpp not installed; run `pip install pywhispercpp`"
            )
            logger.warning("WhisperSTT: %s", self._last_error)
            return False
        if _sd is None:
            self._last_error = "sounddevice missing -- pip install sounddevice"
            logger.warning("WhisperSTT: %s", self._last_error)
            return False
        if _webrtcvad is None:
            self._last_error = "webrtcvad missing -- pip install webrtcvad"
            logger.warning("WhisperSTT: %s", self._last_error)
            return False
        if _np is None:
            self._last_error = "numpy missing"
            logger.warning("WhisperSTT: %s", self._last_error)
            return False
        if not self._model_path.exists():
            self._last_error = (
                f"Whisper model not found at {self._model_path} "
                f"-- run `python scripts/install_whisper_model.py`"
            )
            logger.warning("WhisperSTT: %s", self._last_error)
            return False

        try:
            t0 = time.perf_counter()
            self._model = _pwc_model.Model(
                str(self._model_path),
                n_threads=self._n_threads,
            )
            self._vad = _webrtcvad.Vad(self._vad_aggressiveness)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._available = True
            self._last_error = None
            logger.info(
                "WhisperSTT preloaded (%s, n_threads=%d, vad=%d) in %.0f ms",
                self._model_path.name,
                self._n_threads,
                self._vad_aggressiveness,
                elapsed_ms,
            )
            # Sprint Ω9: best-effort load of the adaptive turn-taker.
            # Failure is non-fatal -- the legacy trailing-silence wait
            # remains the default decision.
            if self._turn_taker is not None:
                try:
                    if self._turn_taker.preload():
                        logger.info(
                            "WhisperSTT smart turn-taker active "
                            "(min=%.2fs max=%.2fs eot_thr=%.2f)",
                            self._turn_taker.config.min_silence_s,
                            self._turn_taker.config.max_silence_s,
                            self._turn_taker.config.eot_probability_threshold,
                        )
                    else:
                        logger.info(
                            "WhisperSTT smart turn-taker requested but "
                            "unavailable -- falling back to fixed "
                            "trailing-silence (%.2fs)",
                            self._trailing_silence_s,
                        )
                except Exception:
                    logger.debug(
                        "Smart turn-taker preload raised; falling back",
                        exc_info=True,
                    )
            return True
        except Exception as exc:
            self._available = False
            self._last_error = f"WhisperSTT preload failed: {exc}"
            logger.warning("WhisperSTT: %s", self._last_error, exc_info=True)
            return False

    async def async_preload(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.preload)

    def start_listening(
        self,
        loop: asyncio.AbstractEventLoop | None = None,
        on_final: Callable[[str], None] | None = None,
        on_partial: Callable[[str], None] | None = None,
    ) -> bool:
        """Open the mic stream and begin background processing."""
        if not self._available:
            if self.preload() is False:
                return False
        if self._listening:
            return True

        self._loop = loop
        self._on_final = on_final
        self._on_partial = on_partial
        self._stop_event = asyncio.Event()
        self._reset_utterance_state()

        try:
            self._stream = _sd.RawInputStream(
                samplerate=_SAMPLE_RATE,
                blocksize=_FRAME_SAMPLES,
                dtype="int16",
                channels=1,
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as exc:
            self._last_error = f"sounddevice open failed: {exc}"
            logger.warning("WhisperSTT: %s", self._last_error, exc_info=True)
            self._stream = None
            return False

        self._listening = True
        if loop is not None:
            self._worker_task = loop.create_task(self._consume_loop())
        logger.info(
            "WhisperSTT listening (%d Hz, %d-ms VAD, partial @%.1fs, "
            "trail %.2fs)",
            _SAMPLE_RATE,
            _FRAME_MS,
            self._partial_interval_s,
            self._trailing_silence_s,
        )
        return True

    def stop_listening(self) -> str:
        """Stop the mic, flush any pending utterance, return its text."""
        if not self._listening:
            return self._last_final
        self._listening = False
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception:
            logger.debug("stt_whisper: stream close raised", exc_info=True)
        self._stream = None

        # Drain whatever speech is sitting in the buffer so we never
        # silently lose the last utterance.
        text = self._flush_utterance(force=True)
        if text:
            self._emit_final(text)
            self._last_final = text

        try:
            self._stop_event.set()
        except Exception:
            logger.debug("stop_event.set raised", exc_info=True)
        if self._worker_task is not None:
            try:
                self._worker_task.cancel()
            except Exception:
                logger.debug("worker_task cancel raised", exc_info=True)
            self._worker_task = None

        logger.info("WhisperSTT stopped")
        return self._last_final

    def shutdown(self) -> None:
        self.stop_listening()
        self._model = None
        self._vad = None
        self._available = False
        logger.info("WhisperSTT shut down")

    # ── async-compatible wrappers ───────────────────────────────

    async def async_start_listening(self, **_kw: Any) -> None:
        if self._permanently_disabled:
            logger.info("WhisperSTT permanently disabled — voice input unavailable")
            return
        loop = asyncio.get_running_loop()
        self._loop = loop
        self._running_async = True

        def _on_final(text: str) -> None:
            if not text:
                return
            loop.call_soon_threadsafe(
                lambda t=text: (
                    self._bus.emit_fast(
                        "voice.final",
                        text=t,
                        language=self._language,
                        confidence=float(self._last_confidence),
                        engine=self.backend_name,
                        mic=self.mic_name,
                    ),
                    self._bus.emit("speech_final", text=t, language=self._language),
                ),
            )

        def _on_partial(text: str) -> None:
            if not text:
                return
            loop.call_soon_threadsafe(
                lambda t=text: (
                    self._bus.emit_fast(
                        "voice.partial",
                        text=t,
                        confidence=float(self._last_confidence),
                        engine=self.backend_name,
                        mic=self.mic_name,
                    ),
                    self._bus.emit("speech_partial", text=t),
                ),
            )

        if not self.start_listening(
            loop=loop, on_final=_on_final, on_partial=_on_partial,
        ):
            logger.warning(
                "WhisperSTT async_start_listening: start failed (%s)",
                self._last_error or "unknown",
            )
            return

        try:
            while self._running_async and self._listening:
                await asyncio.sleep(0.5)
        finally:
            self._running_async = False

    # ── audio path ──────────────────────────────────────────────

    def _audio_callback(self, indata: bytes, frames: int, time_info, status) -> None:
        """Sounddevice callback (audio-thread)."""
        if status:
            logger.debug("WhisperSTT audio status: %s", status)
        try:
            self._tap_buffer_count += 1
            data = bytes(indata)
            if not data:
                return
            with self._ring_lock:
                self._ring.append(data)
            # Update RMS for the watchdog / diagnostics.
            samples = _np.frombuffer(data, dtype=_np.int16) if _np else None
            if samples is not None and samples.size:
                rms = float(_np.sqrt(_np.mean(samples.astype(_np.float32) ** 2)))
                if rms > 0.0:
                    self._last_audio_rms_db = 20.0 * (
                        # log10(rms / 32768) ~ -6 dB at full scale
                        # use _np.log10 to stay numpy-only
                        _np.log10(max(rms, 1.0) / 32768.0)
                    )
        except Exception:
            logger.debug("WhisperSTT audio_callback raised", exc_info=True)

    async def _consume_loop(self) -> None:
        """Drain the ring buffer, run VAD, push speech frames into the
        active utterance, and call whisper.cpp at the configured cadence."""
        try:
            while self._listening and not self._stop_event.is_set():
                await asyncio.sleep(_FRAME_MS / 1000.0)
                self._consume_once()
        except asyncio.CancelledError:  # pragma: no cover -- normal stop
            pass
        except Exception:
            logger.exception("WhisperSTT consume loop crashed")

    def _consume_once(self) -> None:
        """Consume all frames currently in the ring."""
        with self._ring_lock:
            frames = list(self._ring)
            self._ring.clear()
        if not frames:
            return

        for frame in frames:
            try:
                is_speech = self._vad.is_speech(frame, _SAMPLE_RATE)
            except Exception:
                is_speech = False
            now = time.monotonic()
            if is_speech:
                if not self._utterance_frames:
                    self._utterance_started_at = now
                self._utterance_frames.append(frame)
                self._silence_frames = 0
                self._last_speech_time = now
                self._last_speech_candidate_time = now
            elif self._utterance_frames:
                self._utterance_frames.append(frame)
                self._silence_frames += 1

        if not self._utterance_frames:
            return

        utterance_duration_s = (
            len(self._utterance_frames) * _FRAME_MS / 1000.0
        )
        silence_duration_s = self._silence_frames * _FRAME_MS / 1000.0

        # Sprint Ω9: consult the smart turn-taker first. It either
        # produces an explicit finalize/no-finalize decision (with a
        # cooldown so we never burn CPU per frame) or returns
        # ``eval_skipped=True`` and the loop continues with the legacy
        # trailing-silence rule. Mid-thought lockout actively delays
        # the legacy 600 ms trigger when Silero is highly confident
        # the speaker hasn't finished.
        legacy_trailing_finalize = (
            silence_duration_s >= self._trailing_silence_s
        )
        max_utterance_finalize = (
            utterance_duration_s >= self._max_utterance_s
        )
        early_final = False
        midthought_lock = False
        if (
            self._turn_taker is not None
            and self._turn_taker.is_available
            and self._utterance_frames
        ):
            try:
                decision = self._turn_taker.should_finalize(
                    b"".join(self._utterance_frames),
                    silence_s=silence_duration_s,
                    utterance_s=utterance_duration_s,
                )
                early_final = bool(decision.finalize)
                midthought_lock = (
                    decision.reason == "midthought_lockout"
                )
                if decision.finalize:
                    logger.debug(
                        "WhisperSTT smart-final: reason=%s "
                        "silence=%.2fs prob=%.2f eot=%.2f",
                        decision.reason,
                        silence_duration_s,
                        decision.probability,
                        decision.eot_score,
                    )
            except Exception:
                logger.debug(
                    "Smart turn-taker raised; using legacy fallback",
                    exc_info=True,
                )

        if (
            early_final
            or max_utterance_finalize
            or (legacy_trailing_finalize and not midthought_lock)
        ):
            text = self._flush_utterance(force=True)
            if text:
                self._emit_final(text)
                self._last_final = text
            return

        # Partial: emit at most every _partial_interval_s.
        now = time.monotonic()
        since_partial = now - self._last_partial_emit_at
        if (
            utterance_duration_s >= self._partial_interval_s
            and since_partial >= self._partial_interval_s
        ):
            text = self._transcribe(self._utterance_frames, partial=True)
            if text and text != self._last_partial:
                self._last_partial = text
                self._emit_partial(text)
                self._last_partial_emit_at = now

    def _flush_utterance(self, *, force: bool) -> str:
        if not self._utterance_frames:
            return ""
        duration_ms = len(self._utterance_frames) * _FRAME_MS
        if duration_ms < _MIN_UTTERANCE_MS and not force:
            return ""
        text = self._transcribe(self._utterance_frames, partial=False)
        self._reset_utterance_state()
        return text

    def _transcribe(self, frames: list[bytes], *, partial: bool) -> str:
        if self._model is None or _np is None:
            return ""
        try:
            audio_bytes = b"".join(frames)
            audio = (
                _np.frombuffer(audio_bytes, dtype=_np.int16)
                .astype(_np.float32) / 32768.0
            )
            t0 = time.perf_counter()
            segments = self._model.transcribe(
                audio,
                language=self._language,
                # Smaller param when partial -> ~30% faster
                n_processors=1,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            text = " ".join(
                getattr(seg, "text", "") for seg in segments
            ).strip()
            logger.debug(
                "WhisperSTT %s: %d-ms audio -> %d chars in %.0f ms",
                "partial" if partial else "final",
                len(frames) * _FRAME_MS,
                len(text),
                elapsed_ms,
            )
            return text
        except Exception:
            logger.exception("WhisperSTT transcribe raised")
            return ""

    def _reset_utterance_state(self) -> None:
        self._utterance_frames = []
        self._silence_frames = 0
        self._utterance_started_at = 0.0
        self._last_partial_emit_at = 0.0
        self._last_partial = ""
        if self._turn_taker is not None:
            self._turn_taker.reset()

    # ── emit helpers (duck-compatible with NativeSTT) ───────────

    def _emit_partial(self, text: str) -> None:
        if self._echo_guard is not None:
            try:
                if self._echo_guard(text):
                    return
            except Exception:
                logger.debug("echo_guard raised", exc_info=True)
        cb = self._on_partial
        if cb is not None:
            try:
                cb(text)
            except Exception:
                logger.debug("on_partial raised", exc_info=True)
        try:
            self._bus.emit("speech_partial", text=text)
        except Exception:
            logger.debug("emit speech_partial raised", exc_info=True)

    def _emit_final(self, text: str) -> None:
        if self._echo_guard is not None:
            try:
                if self._echo_guard(text):
                    return
            except Exception:
                logger.debug("echo_guard raised", exc_info=True)
        cb = self._on_final
        if cb is not None:
            try:
                cb(text)
            except Exception:
                logger.debug("on_final raised", exc_info=True)
        try:
            self._bus.emit("speech_final", text=text, language=self._language)
        except Exception:
            logger.debug("emit speech_final raised", exc_info=True)

    # ── diagnostics (mirrors NativeSTT.get_diagnostics shape) ──

    def get_diagnostics(self) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "engine": self.backend_name,
            "available": self._available,
            "listening": self._listening,
            "running_async": self._running_async,
            "model_path": str(self._model_path),
            "model_present": self._model_path.exists(),
            "tap_buffer_count": self._tap_buffer_count,
            "last_audio_rms_db": round(self._last_audio_rms_db, 1),
            "since_last_speech_s": (
                round(now - self._last_speech_time, 1)
                if self._last_speech_time else None
            ),
            "last_error": self._last_error,
        }

    # ── compatibility shims for STTWatchdog ───────────────────

    def _restart_recognition_chain(self) -> None:
        """Watchdog-compatible 'soft restart'. Whisper.cpp doesn't have
        the SFSpeechRecognizer-style chain, so we just bounce the
        sounddevice stream."""
        if not self._listening:
            return
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception:
            logger.debug("WhisperSTT chain-restart stream-close raised",
                         exc_info=True)
        self._stream = None
        try:
            self._stream = _sd.RawInputStream(
                samplerate=_SAMPLE_RATE,
                blocksize=_FRAME_SAMPLES,
                dtype="int16",
                channels=1,
                callback=self._audio_callback,
            )
            self._stream.start()
            logger.info("WhisperSTT: sounddevice stream re-bound")
        except Exception:
            logger.warning("WhisperSTT: stream rebind failed", exc_info=True)


def is_whisper_available(config: dict | None = None) -> bool:
    """Best-effort import-time check used by the auto-engine factory.
    Returns True iff every native dep is present AND the model file
    exists on disk."""
    if _pwc_model is None or _sd is None or _webrtcvad is None or _np is None:
        return False
    full_cfg = (config or {}).get("stt", {})
    model_path = _resolve_model_path(full_cfg.get("whisper_model_path"))
    return model_path.exists()


__all__ = ("WhisperSTT", "is_whisper_available")
