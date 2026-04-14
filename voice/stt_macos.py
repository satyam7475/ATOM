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
import os
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


def _main_bundle_info() -> dict[str, Any] | None:
    if _Foundation is None:
        return None
    try:
        bundle = _Foundation.NSBundle.mainBundle()
        info = bundle.infoDictionary()
        if info is None:
            return None
        return dict(info)
    except Exception:
        logger.debug("Main bundle probe failed", exc_info=True)
        return None


def _bundle_has_usage_description(key: str) -> bool:
    info = _main_bundle_info()
    return bool(info and info.get(key))


def _native_stt_bundle_supported() -> tuple[bool, str]:
    if sys.platform != "darwin":
        return False, "SFSpeechRecognizer only available on macOS"
    if os.environ.get("ATOM_LAUNCH_MODE") == "venv":
        return (
            False,
            "Native macOS STT expects the ATOM.app bundle process; running under venv "
            "(browser/dashboard SpeechRecognition remains available).",
        )
    if not _HAS_SPEECH or _Speech is None or _Foundation is None:
        return False, "pyobjc speech frameworks are unavailable"
    if not _bundle_has_usage_description("NSSpeechRecognitionUsageDescription"):
        return False, "current process bundle lacks NSSpeechRecognitionUsageDescription"
    if not _bundle_has_usage_description("NSMicrophoneUsageDescription"):
        return False, "current process bundle lacks NSMicrophoneUsageDescription"
    return True, ""


def _speech_permission_status() -> str:
    if not _HAS_SPEECH or _Speech is None:
        return "unavailable"
    if not _bundle_has_usage_description("NSSpeechRecognitionUsageDescription"):
        return "bundle_missing_usage_description"
    try:
        status = int(_Speech.SFSpeechRecognizer.authorizationStatus())
    except Exception:
        return "unknown"
    return {
        0: "not_determined",
        1: "denied",
        2: "restricted",
        3: "authorized",
    }.get(status, f"unknown({status})")


def _microphone_permission_status() -> str:
    if _AVFoundation is None:
        return "unknown"
    if not _bundle_has_usage_description("NSMicrophoneUsageDescription"):
        return "bundle_missing_usage_description"
    try:
        device_cls = getattr(_AVFoundation, "AVCaptureDevice", None)
        media_type = getattr(_AVFoundation, "AVMediaTypeAudio", "soun")
        if device_cls is not None and hasattr(device_cls, "authorizationStatusForMediaType_"):
            status = int(device_cls.authorizationStatusForMediaType_(media_type))
            return {
                0: "not_determined",
                1: "restricted",
                2: "denied",
                3: "authorized",
            }.get(status, f"unknown({status})")
    except Exception:
        logger.debug("Microphone permission probe failed", exc_info=True)
    return "unknown"


def native_stt_launch_supported() -> tuple[bool, str]:
    """Return whether the current process can safely use SFSpeechRecognizer."""
    launch_ok, launch_reason = _native_stt_bundle_supported()
    if not launch_ok:
        return False, launch_reason

    try:
        locale = _Foundation.NSLocale.alloc().initWithLocaleIdentifier_("en-US")
        recognizer = _Speech.SFSpeechRecognizer.alloc().initWithLocale_(locale)
        if recognizer is None or not recognizer.isAvailable():
            return False, "SFSpeechRecognizer is unavailable for the current locale"

        auth_status = _Speech.SFSpeechRecognizer.authorizationStatus()
        if auth_status == 3:
            return True, ""
        if auth_status == 0:
            return True, ""
        return False, f"speech recognition authorization status={auth_status}"
    except Exception as exc:
        return False, str(exc)


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
        self.mic_name: str = "macOS AVAudioEngine"
        self._recognizer: Any = None
        self._audio_engine: Any = None
        self._recognition_request: Any = None
        self._recognition_task: Any = None

        self._available = False
        self._permanently_disabled = False
        self._listening = False
        self._running_async = False
        self._async_task: asyncio.Task | None = None
        self._last_partial: str = ""
        self._last_final: str = ""
        self._last_confidence: float = 0.95
        self._last_error: str | None = None
        self._last_speech_time: float = 0.0
        self._loop: asyncio.AbstractEventLoop | None = None
        self.speech_permission_status: str = "unknown"
        self.microphone_permission_status: str = "unknown"

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
        return "macOS Native (SFSpeechRecognizer)" if self._available else "unavailable"

    # ── Initialization ─────────────────────────────────────────────

    def preload(self) -> bool:
        """Initialize recognizer and check authorization."""
        launch_ok, launch_reason = _native_stt_bundle_supported()
        if not launch_ok:
            self._available = False
            self._last_error = launch_reason
            if "NSSpeechRecognitionUsageDescription" in launch_reason:
                self.speech_permission_status = "bundle_missing_usage_description"
            if "NSMicrophoneUsageDescription" in launch_reason:
                self.microphone_permission_status = "bundle_missing_usage_description"
            logger.warning("Native STT unavailable: %s", launch_reason)
            return False
        self.speech_permission_status = _speech_permission_status()
        self.microphone_permission_status = _microphone_permission_status()

        locale = _Foundation.NSLocale.alloc().initWithLocaleIdentifier_(
            self._locale
        )
        self._recognizer = _Speech.SFSpeechRecognizer.alloc().initWithLocale_(
            locale
        )
        if self._recognizer is None or not self._recognizer.isAvailable():
            self._last_error = f"SFSpeechRecognizer unavailable for locale {self._locale}"
            logger.warning(
                "SFSpeechRecognizer not available for locale '%s'", self._locale
            )
            return False

        self._recognizer.setSupportsOnDeviceRecognition_(True)

        auth_status = _Speech.SFSpeechRecognizer.authorizationStatus()
        if auth_status == 3:  # authorized
            self._available = True
            self.speech_permission_status = "authorized"
            self._last_error = None
            logger.info("Native STT ready (SFSpeechRecognizer, locale=%s, on-device=True)", self._locale)
            return True
        elif auth_status == 0:  # notDetermined
            if threading.current_thread() is not threading.main_thread():
                self._last_error = "Speech Recognition authorization request must run on the main thread"
                logger.warning("Native STT authorization skipped: %s", self._last_error)
                return False
            self.speech_permission_status = "not_determined"
            logger.info("Requesting Speech Recognition authorization...")
            granted_event = threading.Event()
            granted_result = [False]

            def _auth_callback(status: int) -> None:
                granted_result[0] = (status == 3)
                granted_event.set()

            _Speech.SFSpeechRecognizer.requestAuthorization_(_auth_callback)
            granted_event.wait(timeout=30.0)
            if not granted_event.is_set():
                self._last_error = "Speech Recognition authorization request timed out"
                logger.warning("Speech Recognition authorization timed out")
                return False

            if granted_result[0]:
                self._available = True
                self.speech_permission_status = "authorized"
                self._last_error = None
                logger.info("Speech Recognition authorized by user")
                return True
            else:
                self.speech_permission_status = "denied"
                self._last_error = "Speech Recognition permission denied"
                logger.warning(
                    "Speech Recognition denied. Go to System Settings > Privacy & Security > Speech Recognition to enable."
                )
                return False
        else:
            status_names = {1: "denied", 2: "restricted", 3: "authorized"}
            self.speech_permission_status = status_names.get(auth_status, f"unknown({auth_status})")
            self._last_error = f"Speech Recognition authorization: {self.speech_permission_status}"
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
            if not self._available and self._last_error is None:
                self._last_error = "Native STT is unavailable"
            return False

        self._loop = loop
        self._on_final = on_final
        self._on_partial = on_partial
        self._last_partial = ""
        self._last_final = ""
        self.microphone_permission_status = _microphone_permission_status()
        if self.microphone_permission_status in {"denied", "restricted"}:
            self._last_error = f"Microphone permission {self.microphone_permission_status}"
            logger.warning("Native STT blocked: %s", self._last_error)
            return False

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
                self._last_error = str(error or "AVAudioEngine start failed")
                logger.error("AVAudioEngine start failed: %s", error)
                self._cleanup()
                return False

            self._recognition_task = self._recognizer.recognitionTaskWithRequest_resultHandler_(
                self._recognition_request, self._recognition_result_handler,
            )

            self._listening = True
            self._last_speech_time = time.monotonic()
            self._last_error = None
            logger.info("Native STT listening started (on-device)")
            return True

        except Exception as exc:
            self._last_error = str(exc)
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
            self._last_error = err_desc
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
            self._last_error = None
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

    # ── Async-compatible wrappers (match STTAsync interface for main.py) ──

    async def async_preload(self) -> None:
        """Run preload on the process main thread.

        macOS speech authorization is tied to the current app bundle and can
        hard-abort the process if requested from an invalid or worker-thread
        context. Keep preload on the main thread and let `preload()` fail safe
        when the bundle is not speech-capable.
        """
        self.preload()

    async def async_start_listening(self, **_kw) -> None:
        """Continuous listen loop matching STTAsync.start_listening() contract.

        Starts native recognition, then sleeps while active.  When
        a final result arrives the text is emitted on the bus just like
        STTAsync does, and recognition is restarted for the next utterance.
        """
        if self._permanently_disabled:
            logger.info("STT permanently disabled (TCC/entitlements) — voice input unavailable")
            return

        loop = asyncio.get_running_loop()
        self._loop = loop
        self._running_async = True

        def _on_final(text: str) -> None:
            if text and text.strip():
                self._last_error = None
                loop.call_soon_threadsafe(
                    lambda t=text: (
                        self._bus.emit_fast(
                            "voice.final",
                            text=t,
                            language="en",
                            confidence=float(self._last_confidence),
                            engine=self.backend_name,
                            mic=self.mic_name,
                        ),
                        self._bus.emit("speech_final", text=t, language="en"),
                    ),
                )

        def _on_partial(text: str) -> None:
            if text:
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

        max_retries = 5
        retries = 0
        while getattr(self, "_running_async", False):
            if not self._listening:
                ok = self.start_listening(
                    loop=loop, on_final=_on_final, on_partial=_on_partial,
                )
                if not ok:
                    retries += 1
                    self._last_error = self._last_error or "Native STT failed to start"
                    if not self._available or retries > max_retries:
                        logger.warning(
                            "Native STT unavailable (retries=%d, available=%s) — "
                            "voice input disabled for this session",
                            retries, self._available,
                        )
                        self._running_async = False
                        return
                    logger.warning("Native STT start failed, retrying in 5s (%d/%d)", retries, max_retries)
                    await asyncio.sleep(5.0)
                    continue
                retries = 0
            await asyncio.sleep(0.5)

    def stop(self) -> None:
        """Stop async listen loop + underlying recognition."""
        self._running_async = False
        self.stop_listening()

    async def on_state_changed(self, old, new, **_kw) -> None:
        """Handle ATOM state transitions (mirrors STTAsync behaviour)."""
        from core.state_manager import AtomState

        try:
            if self._permanently_disabled:
                return

            if new in (AtomState.LISTENING, AtomState.SPEAKING):
                already_running = (
                    self._async_task is not None
                    and not self._async_task.done()
                )
                if not self._running_async and not already_running:
                    self._running_async = True
                    self._async_task = asyncio.create_task(
                        self.async_start_listening()
                    )
            elif old in (AtomState.LISTENING, AtomState.SPEAKING) and \
                    new not in (AtomState.LISTENING, AtomState.SPEAKING):
                self.stop()
        except Exception:
            logger.debug("on_state_changed error", exc_info=True)
