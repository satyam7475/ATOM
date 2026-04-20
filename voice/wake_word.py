"""
ATOM -- Wake Word Detection Engine.

Config-driven OpenWakeWord wake detection for the passive listening path.
Runs on CPU with low overhead and promotes the voice pipeline from passive
monitoring to active command routing when the configured model triggers.

Text-level wake matching for "Atom" / "Hey Atom" still lives in
``voice.listening_modes.WakeWordFilter`` so ATOM branding remains stable even
when the underlying acoustic model is a different built-in OpenWakeWord name
such as ``hey_jarvis``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import TYPE_CHECKING, Any

logger = logging.getLogger("atom.wake_word")

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.state_manager import StateManager


class WakeWordEngine:
    """Wake word detector for passive -> active promotion.

    Runs in a dedicated thread, consuming minimal CPU.
    Emits 'wake_word_detected' event when triggered.
    """

    _WAKE_WORDS = ["hey_atom", "hey atom", "atom", "hey computer"]
    _COOLDOWN_S = 2.0

    def __init__(
        self,
        bus: "AsyncEventBus",
        state: "StateManager",
        config: dict | None = None,
    ) -> None:
        self._bus = bus
        self._state = state
        self._config = (config or {}).get("wake_word", {})
        self._enabled = self._config.get("enabled", True)
        self._sensitivity = self._config.get("sensitivity", 0.6)
        self._model: Any = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_detection: float = 0.0
        self._available = False
        self._detection_count = 0
        self._capture_gate = threading.Event()
        self._capture_gate.set()
        self._wakeword_models = self._configured_model_names(self._config)

    @staticmethod
    def _configured_model_names(config: dict | None) -> list[str]:
        """Normalize configured OpenWakeWord model names.

        Supports either ``wake_word.model`` (string) or ``wake_word.models``
        (list of strings). Spaces are normalized to underscores so
        ``"hey jarvis"`` and ``"hey_jarvis"`` resolve identically.
        """
        cfg = config or {}
        raw_models = cfg.get("models")
        if isinstance(raw_models, (list, tuple)):
            items = raw_models
        else:
            items = [cfg.get("model", "hey_jarvis")]

        normalized: list[str] = []
        for item in items:
            name = str(item or "").strip().lower().replace(" ", "_")
            if name:
                normalized.append(name)
        return normalized or ["hey_jarvis"]

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def detection_count(self) -> int:
        return self._detection_count

    def preload(self) -> bool:
        """Load the wake word model."""
        if not self._enabled:
            logger.info("Wake word disabled in config")
            return False

        try:
            from openwakeword.model import Model
            self._model = Model(
                wakeword_models=self._wakeword_models,
                inference_framework="onnx",
            )
            self._available = True
            logger.info(
                "Wake word engine loaded (OpenWakeWord: %s)",
                ", ".join(self._wakeword_models),
            )
            return True
        except ImportError:
            logger.info(
                "OpenWakeWord not installed -- using always-listen mode. "
                "Install with: pip install openwakeword"
            )
            self._available = False
            return False
        except Exception:
            logger.debug("Wake word load failed", exc_info=True)
            self._available = False
            return False

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Start wake word detection in a background thread."""
        if not self._available or not self._enabled:
            return
        if self._running:
            return

        self._loop = loop
        self._running = True
        self._thread = threading.Thread(
            target=self._detection_loop,
            name="wake_word",
            daemon=True,
        )
        self._thread.start()
        logger.info("Wake word detection started (sensitivity=%.2f)", self._sensitivity)

    def pause(self) -> None:
        """Pause audio capture (e.g. while STT has its own mic stream)."""
        if self._capture_gate.is_set():
            self._capture_gate.clear()
            logger.debug("Wake word capture paused (STT active)")

    def resume(self) -> None:
        """Resume audio capture after STT releases the mic."""
        if not self._capture_gate.is_set():
            self._capture_gate.set()
            logger.debug("Wake word capture resumed")

    def stop(self) -> None:
        self._running = False
        self._capture_gate.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        logger.info("Wake word detection stopped (%d detections total)", self._detection_count)

    def _detection_loop(self) -> None:
        """Main detection loop -- runs in a dedicated thread.

        Pauses automatically when STT has its own mic stream open
        (via the _capture_gate event) to avoid PortAudio conflicts.
        """
        try:
            import sounddevice as sd
            import numpy as np

            _RATE = 16000
            _BLOCK = 1280

            logger.info("Wake word listening on microphone (sounddevice %d Hz)...", _RATE)

            while self._running:
                if not self._capture_gate.wait(timeout=1.0):
                    continue

                try:
                    audio_f32 = sd.rec(
                        _BLOCK, samplerate=_RATE, channels=1,
                        dtype="float32", blocking=True,
                    )
                    flat = audio_f32.ravel()
                    if np.any(np.isnan(flat)):
                        continue
                    samples = (flat * 32767).astype(np.int16)

                    prediction = self._model.predict(samples)

                    for wake_word, score in prediction.items():
                        if score >= self._sensitivity:
                            now = time.monotonic()
                            if now - self._last_detection < self._COOLDOWN_S:
                                continue
                            self._last_detection = now
                            self._detection_count += 1
                            logger.info(
                                "Wake word detected: '%s' (score=%.2f, count=%d)",
                                wake_word, score, self._detection_count,
                            )
                            self._emit_detection(wake_word)

                except Exception as e:
                    if self._running:
                        logger.debug("Wake word audio error: %s", e)
                        time.sleep(0.5)

        except Exception:
            logger.exception("Wake word detection loop failed")
            self._available = False

    def _emit_detection(self, wake_word: str) -> None:
        """Thread-safe event emission."""
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(
                    lambda ww=wake_word: self._bus.emit(
                        "wake_word_detected", wake_word=ww,
                    ),
                )
            except RuntimeError:
                pass

    def shutdown(self) -> None:
        self.stop()
        self._model = None
        self._available = False
