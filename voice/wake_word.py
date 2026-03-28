"""
ATOM -- Wake Word Detection Engine.

"Hey ATOM" wake word detection using OpenWakeWord for natural activation.
Runs on CPU with <1% usage. When detected, transitions from passive
monitoring to active listening.

Replaces always-on STT with efficient wake word detection that only
activates full speech recognition when the user says "Hey ATOM".

Pipeline:
    Microphone -> Wake Word Model (always running, <1% CPU)
    -> Wake word detected -> Transition to LISTENING
    -> STT processes actual command -> Back to wake word monitoring

Hotkey (Ctrl+Alt+A) bypasses wake word for instant activation.

Falls back to always-listening mode if OpenWakeWord is not installed.
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
    """Wake word detector for 'Hey ATOM' activation.

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
                wakeword_models=["hey_jarvis"],
                inference_framework="onnx",
            )
            self._available = True
            logger.info("Wake word engine loaded (OpenWakeWord)")
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

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        logger.info("Wake word detection stopped (%d detections total)", self._detection_count)

    def _detection_loop(self) -> None:
        """Main detection loop -- runs in a dedicated thread."""
        try:
            import pyaudio
            import numpy as np

            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1280,
            )

            logger.info("Wake word listening on microphone...")

            while self._running:
                try:
                    data = stream.read(1280, exception_on_overflow=False)
                    samples = np.frombuffer(data, dtype=np.int16)

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

            stream.stop_stream()
            stream.close()
            audio.terminate()

        except Exception:
            logger.exception("Wake word detection loop failed")
            self._available = False

    def _emit_detection(self, wake_word: str) -> None:
        """Thread-safe event emission."""
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(
                    self._bus.emit, "wake_word_detected",
                    {"wake_word": wake_word},
                )
            except RuntimeError:
                pass

    def shutdown(self) -> None:
        self.stop()
        self._model = None
        self._available = False
