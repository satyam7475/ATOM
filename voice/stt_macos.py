"""
ATOM -- Native macOS Speech-to-Text via SFSpeechRecognizer.

Runs entirely on-device using the Neural Engine. Zero external
dependencies beyond pyobjc. Replaces faster-whisper + SpeechRecognition
+ PyAudio for the command recognition path.

Features:
  - On-device recognition (no network, Neural Engine accelerated)
  - Real-time streaming with partial results
  - ~50ms latency for short commands
  - Automatic microphone handling via AVAudioEngine (no PyAudio/PortAudio)
  - Hardware echo cancellation & noise suppression via Voice Processing I/O
  - Built-in wake word detection (checks partials for "atom"/"hey atom")
  - Language support: en-US (primary), extensible to other locales

Falls back to faster-whisper STT if SFSpeechRecognizer is unavailable.

Requires: macOS 10.15+, pyobjc-framework-Speech, pyobjc-framework-AVFoundation
Authorization: user must grant Speech Recognition + Microphone permissions.

Owner: Satyam
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Callable

logger = logging.getLogger("atom.stt_macos")

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.state_manager import StateManager
    from voice.mic_manager import MicManager

_HAS_SPEECH = False
_Speech: Any = None
_AVFoundation: Any = None
_Foundation: Any = None

try:
    import Speech as _Speech            # type: ignore[import-untyped]
    import AVFoundation as _AVFoundation  # type: ignore[import-untyped]
    import Foundation as _Foundation      # type: ignore[import-untyped]
    _HAS_SPEECH = True
except ImportError:
    pass

_WAKE_PHRASES = {"hey atom", "atom", "hey computer"}
_SILENCE_TIMEOUT_S = 2.0
_MAX_RECORD_S = 15.0


class NativeSTT:
    """macOS native STT using SFSpeechRecognizer + AVAudioEngine.

    Lifecycle:
      1. preload() — check authorization, create recognizer
      2. start_listening() — begin mic capture + recognition
      3. stop_listening() — stop mic, finalize result
      4. shutdown() — release all resources
    """

    def __init__(
        self,
        bus: AsyncEventBus,
        state: StateManager,
        config: dict | None = None,
        mic_manager: MicManager | None = None,
        intent_engine: Any = None,
    ) -> None:
        self._bus = bus
        self._state = state
        self._config = (config or {}).get("stt", {})
        self._mic_manager = mic_manager
        self._intent_engine = intent_engine

        self._locale: str = self._config.get("locale", "en-US")
        self._recognizer: Any = None
        self._audio_engine: Any = None
        self._recognition_request: Any = None
        self._recognition_task: Any = None

        self._available = False
        self._listening = False
        self._last_partial: str = ""
        self._last_final: str = ""
        self._last_speech_time: float = 0.0
        self._loop: asyncio.AbstractEventLoop | None = None

        self._on_final: Callable[[str], None] | None = None
        self._on_partial: Callable[[str], None] | None = None

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def is_listening(self) -> bool:
        return self._listening

    @property
    def backend_name(self) -> str:
        return "SFSpeechRecognizer" if self._available else "unavailable"

    # ── Initialization ─────────────────────────────────────────────

    def preload(self) -> bool:
        """Initialize recognizer and check authorization."""
        if not _HAS_SPEECH:
            logger.info(
                "SFSpeechRecognizer not available (pyobjc-framework-Speech "
                "not installed). Install with: pip install pyobjc-framework-Speech"
            )
            return False

        if sys.platform != "darwin":
            logger.info("SFSpeechRecognizer only available on macOS")
            return False

        locale = _Foundation.NSLocale.alloc().initWithLocaleIdentifier_(
            self._locale
        )
        self._recognizer = _Speech.SFSpeechRecognizer.alloc().initWithLocale_(
            locale
        )
        if self._recognizer is None or not self._recognizer.isAvailable():
            logger.warning(
                "SFSpeechRecognizer not available for locale '%s'", self._locale
            )
            return False

        self._recognizer.setSupportsOnDeviceRecognition_(True)

        auth_status = _Speech.SFSpeechRecognizer.authorizationStatus()
        if auth_status == 3:  # authorized
            self._available = True
            logger.info(
                "Native STT ready (SFSpeechRecognizer, locale=%s, on-device=True)",
                self._locale,
            )
            return True
        elif auth_status == 0:  # notDetermined
            logger.info("Requesting Speech Recognition authorization...")
            granted_event = threading.Event()
            granted_result = [False]

            def _auth_callback(status: int) -> None:
                granted_result[0] = (status == 3)
                granted_event.set()

            _Speech.SFSpeechRecognizer.requestAuthorization_(_auth_callback)
            granted_event.wait(timeout=30.0)

            if granted_result[0]:
                self._available = True
                logger.info("Speech Recognition authorized by user")
                return True
            else:
                logger.warning(
                    "Speech Recognition denied. Go to System Settings > "
                    "Privacy & Security > Speech Recognition to enable."
                )
                return False
        else:
            status_names = {1: "denied", 2: "restricted", 3: "authorized"}
            logger.warning(
                "Speech Recognition authorization: %s",
                status_names.get(auth_status, f"unknown({auth_status})"),
            )
            return False

    # ── Listening ──────────────────────────────────────────────────

    def start_listening(
        self,
        loop: asyncio.AbstractEventLoop | None = None,
        on_final: Callable[[str], None] | None = None,
        on_partial: Callable[[str], None] | None = None,
    ) -> bool:
        """Start mic capture and speech recognition.

        Callbacks are called from the recognition thread:
          on_final(text) — called when a complete utterance is recognized
          on_partial(text) — called with interim results
        """
        if not self._available or self._listening:
            return False

        self._loop = loop
        self._on_final = on_final
        self._on_partial = on_partial
        self._last_partial = ""
        self._last_final = ""

        try:
            self._audio_engine = _AVFoundation.AVAudioEngine.alloc().init()

            input_node = self._audio_engine.inputNode()

            try:
                input_node.setVoiceProcessingEnabled_error_(True, None)
                logger.debug("Voice Processing I/O enabled (HW noise suppression)")
            except Exception:
                logger.debug("Voice Processing I/O not available")

            self._recognition_request = (
                _Speech.SFSpeechAudioBufferRecognitionRequest.alloc().init()
            )
            self._recognition_request.setShouldReportPartialResults_(True)
            self._recognition_request.setRequiresOnDeviceRecognition_(True)

            recording_format = input_node.outputFormatForBus_(0)

            input_node.installTapOnBus_bufferSize_format_block_(
                0, 1024, recording_format, self._audio_buffer_callback,
            )

            self._audio_engine.prepare()
            success, error = self._audio_engine.startAndReturnError_(None)
            if not success:
                logger.error("AVAudioEngine start failed: %s", error)
                self._cleanup()
                return False

            self._recognition_task = self._recognizer.recognitionTaskWithRequest_resultHandler_(
                self._recognition_request, self._recognition_result_handler,
            )

            self._listening = True
            self._last_speech_time = time.monotonic()
            logger.info("Native STT listening started (on-device)")
            return True

        except Exception:
            logger.exception("Failed to start native STT")
            self._cleanup()
            return False

    def _audio_buffer_callback(self, buffer: Any, when: Any) -> None:
        """Tap callback: forward audio buffers to the recognition request."""
        if self._recognition_request is not None:
            self._recognition_request.appendAudioPCMBuffer_(buffer)

    def _recognition_result_handler(self, result: Any, error: Any) -> None:
        """Called by SFSpeechRecognizer with partial/final results."""
        if error is not None:
            err_desc = str(error)
            if "kAFAssistantErrorDomain" not in err_desc:
                logger.debug("Recognition error: %s", err_desc)
            return

        if result is None:
            return

        transcript = str(result.bestTranscription().formattedString())
        is_final = result.isFinal()

        self._last_speech_time = time.monotonic()

        if is_final:
            self._last_final = transcript
            logger.info("STT final: '%s'", transcript)
            if self._on_final:
                self._emit_threadsafe(self._on_final, transcript)
        else:
            if transcript != self._last_partial:
                self._last_partial = transcript
                logger.debug("STT partial: '%s'", transcript)

                lower = transcript.lower().strip()
                for phrase in _WAKE_PHRASES:
                    if lower.endswith(phrase) or lower == phrase:
                        logger.info("Wake phrase detected in partial: '%s'", phrase)
                        self._emit_threadsafe(
                            lambda p=phrase: self._bus.emit(
                                "wake_word_detected", wake_word=p,
                            ),
                            None,
                        )
                        break

                if self._on_partial:
                    self._emit_threadsafe(self._on_partial, transcript)

    def _emit_threadsafe(self, callback: Callable, arg: Any) -> None:
        """Safely call a callback from the recognition thread."""
        loop = self._loop
        if loop is not None and loop.is_running():
            if arg is not None:
                loop.call_soon_threadsafe(callback, arg)
            else:
                loop.call_soon_threadsafe(callback)
        else:
            try:
                if arg is not None:
                    callback(arg)
                else:
                    callback()
            except Exception:
                logger.debug("Callback error", exc_info=True)

    # ── Stop / Cleanup ─────────────────────────────────────────────

    def stop_listening(self) -> str:
        """Stop mic capture and return the last recognized text."""
        if not self._listening:
            return self._last_final

        self._listening = False

        if self._audio_engine is not None:
            try:
                self._audio_engine.inputNode().removeTapOnBus_(0)
                self._audio_engine.stop()
            except Exception:
                logger.debug("Audio engine stop error", exc_info=True)

        if self._recognition_request is not None:
            try:
                self._recognition_request.endAudio()
            except Exception:
                pass

        if self._recognition_task is not None:
            try:
                self._recognition_task.cancel()
            except Exception:
                pass

        logger.info("Native STT listening stopped")
        return self._last_final or self._last_partial

    def _cleanup(self) -> None:
        """Release all resources."""
        self._listening = False
        if self._audio_engine is not None:
            try:
                self._audio_engine.inputNode().removeTapOnBus_(0)
            except Exception:
                pass
            try:
                self._audio_engine.stop()
            except Exception:
                pass
        self._audio_engine = None
        self._recognition_request = None
        self._recognition_task = None

    def shutdown(self) -> None:
        """Full shutdown."""
        self.stop_listening()
        self._cleanup()
        self._recognizer = None
        self._available = False
        logger.info("Native STT shut down")
