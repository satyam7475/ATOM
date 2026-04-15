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

Requires: macOS 10.15+, pyobjc-framework-Speech, pyobjc-framework-AVFoundation
Authorization: user must grant Speech Recognition + Microphone permissions.

Owner: Satyam
"""

from __future__ import annotations

import asyncio
import logging
import os
import plistlib
import sys
import threading
import time
import weakref
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


def _format_ns_error(error: Any) -> str:
    """Best-effort NSError / PyObjc exception string for STT diagnostics."""
    if error is None:
        return ""
    try:
        domain = error.domain() if hasattr(error, "domain") else ""
        code = int(error.code()) if hasattr(error, "code") else -1
        return f"{domain} code={code} | {error}"
    except Exception:
        return str(error)


def _probe_default_input_mic_label() -> str:
    """Best-effort name for the default input device (menu bar / Sound Input)."""
    if _AVFoundation is None:
        return "unknown (AVFoundation unavailable)"
    try:
        mt = getattr(_AVFoundation, "AVMediaTypeAudio", None)
        if mt is None:
            mt = "soun"
        dev = _AVFoundation.AVCaptureDevice.defaultDeviceWithMediaType_(mt)
        if dev is None:
            return "no default AVCaptureDevice — grant Microphone in Privacy, or pick input in Sound"
        return str(dev.localizedName())
    except Exception as exc:
        return f"mic probe error: {exc}"


def _format_av_audio_format(fmt: Any) -> str:
    try:
        sr = float(fmt.sampleRate())
        ch = int(fmt.channelCount())
        return f"{sr:.0f} Hz, {ch} ch"
    except Exception:
        return str(fmt)


def _resolve_stt_locale(raw: str | None) -> str:
    """Match Siri/Dictation: use the Mac’s primary locale when ``locale`` is ``auto``."""
    s = (raw or "auto").strip()
    if s.lower() != "auto":
        return s
    if _Foundation is None:
        return "en-US"
    try:
        loc = _Foundation.NSLocale.currentLocale()
        ident = str(loc.localeIdentifier())
        return ident.replace("_", "-")
    except Exception:
        logger.debug("NSLocale.currentLocale failed", exc_info=True)
        return "en-US"


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


def _atom_app_bundle_info_plist() -> dict[str, Any] | None:
    """When the real process image is ``.venv/bin/python`` (shell launcher exec), ``NSBundle.mainBundle`` is Python.framework, not ATOM.app. Read the launched app's Info.plist from ATOM_APP_BUNDLE."""
    if os.environ.get("ATOM_LAUNCH_MODE") != "bundle":
        return None
    root = (os.environ.get("ATOM_APP_BUNDLE") or "").strip()
    if not root:
        return None
    plist_path = os.path.join(root, "Contents", "Info.plist")
    try:
        with open(plist_path, "rb") as f:
            raw = plistlib.load(f)
        return raw if isinstance(raw, dict) else None
    except OSError:
        logger.debug("ATOM_APP_BUNDLE Info.plist missing or unreadable: %s", plist_path)
        return None
    except Exception:
        logger.debug("ATOM_APP_BUNDLE Info.plist parse failed", exc_info=True)
        return None


def _effective_bundle_info() -> dict[str, Any] | None:
    """Merge main bundle dict with ATOM.app plist so usage strings resolve after ``exec`` to venv Python."""
    main = _main_bundle_info()
    extra = _atom_app_bundle_info_plist()
    if not extra:
        return main
    if not main:
        return extra
    merged = dict(main)
    merged.update(extra)
    return merged


def _bundle_has_usage_description(key: str) -> bool:
    info = _effective_bundle_info()
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

        self._locale: str = _resolve_stt_locale(self._config.get("locale", "auto"))
        self._audio_buffer_frames: int = int(self._config.get("audio_buffer_frames", 2048))
        self._native_stop_audio_delay_s: float = (
            float(self._config.get("native_stop_audio_delay_ms", 120) or 0) / 1000.0
        )
        self.mic_name: str = "macOS AVAudioEngine"
        self._recognizer: Any = None
        self._audio_engine: Any = None
        self._recognition_request: Any = None
        self._recognition_task: Any = None
        self._recognition_lock = threading.Lock()

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
        # After TTS, wait before reopening mic (avoids AVAudioEngine vs output conflict).
        self._need_post_tts_cooldown: bool = False

        self._voice_debug: bool = bool(self._config.get("voice_debug", False))
        self._voice_debug_interval_s: float = max(
            8.0, float(self._config.get("voice_debug_interval_s", 25) or 25),
        )
        self._dbg_next_status_log: float = 0.0
        # When True, keep mic open during SPEAKING so partials can trigger TTS barge-in (test with headphones first; echo risk).
        self._barge_in_during_speak: bool = bool(
            self._config.get("barge_in_during_speak", False),
        )
        # On-device-only often yields zero partials on Bluetooth + some locales (e.g. en-IN).
        # When False, Apple may use server-assisted recognition (requires network; not fully private).
        self._native_requires_on_device: bool = bool(
            self._config.get("native_requires_on_device", True),
        )
        # Voice Processing I/O can break or silence some Bluetooth headsets — disable to test.
        self._native_voice_processing: bool = bool(
            self._config.get("native_voice_processing", True),
        )
        self._tap_buffer_count: int = 0
        self._recognizer_supports_on_device: bool = False
        # Plain function passed to recognitionTaskWithRequest — bound methods can fail to bridge as ObjC blocks.
        self._speech_pyobjc_block: Callable[..., None] | None = None

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
            logger.warning(
                "Native STT unavailable: %s — VOICE_INPUT: use ATOM.app (not raw venv) for "
                "SFSpeechRecognizer + microphone TCC when bundle usage strings are required.",
                launch_reason,
            )
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

        try:
            _sod = bool(self._recognizer.supportsOnDeviceRecognition())
        except Exception:
            _sod = False
        self._recognizer_supports_on_device: bool = _sod

        auth_status = _Speech.SFSpeechRecognizer.authorizationStatus()
        if auth_status == 3:  # authorized
            self._available = True
            self.speech_permission_status = "authorized"
            self._last_error = None
            logger.info(
                "Native STT ready (SFSpeechRecognizer, locale=%s, recognizer.supportsOnDeviceRecognition=%s)",
                self._locale,
                _sod,
            )
            logger.info(
                "VOICE_INPUT: probe default mic=%s | launch_mode=%s",
                _probe_default_input_mic_label(),
                os.environ.get("ATOM_LAUNCH_MODE", ""),
            )
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
                logger.info(
                    "VOICE_INPUT: probe default mic=%s | launch_mode=%s",
                    _probe_default_input_mic_label(),
                    os.environ.get("ATOM_LAUNCH_MODE", ""),
                )
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

    def _apply_speech_request_policy(self, req: Any) -> None:
        """Partial results + on-device vs server-assisted (Bluetooth/locale dependent)."""
        req.setShouldReportPartialResults_(True)
        want_on_device = self._native_requires_on_device
        if want_on_device:
            try:
                if not getattr(self, "_recognizer_supports_on_device", True):
                    logger.warning(
                        "Native STT: on-device recognition not supported for locale %s — using "
                        "server-assisted streaming (set stt.locale=en-US for on-device, or "
                        "stt.native_requires_on_device=false to allow network by choice)",
                        self._locale,
                    )
                    want_on_device = False
            except Exception:
                pass
        req.setRequiresOnDeviceRecognition_(bool(want_on_device))
        if not want_on_device:
            logger.info(
                "Native STT: requiresOnDeviceRecognition=False (locale=%s) — network speech may be used",
                self._locale,
            )

    def _apply_speech_request_hints(self, req: Any) -> None:
        """Prefer dictation-style streaming (helps some macOS Speech code paths)."""
        if not _HAS_SPEECH or _Speech is None:
            return
        hint = getattr(_Speech, "SFSpeechRecognitionTaskHintDictation", None)
        if hint is None or not hasattr(req, "setTaskHint_"):
            return
        try:
            req.setTaskHint_(hint)
        except Exception as exc:
            logger.debug("Native STT: setTaskHint(Dictation) skipped: %s", exc)

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
            self._tap_buffer_count = 0
            self._logged_stt_handler_shape = False
            self._audio_engine = _AVFoundation.AVAudioEngine.alloc().init()

            input_node = self._audio_engine.inputNode()

            if self._native_voice_processing:
                try:
                    input_node.setVoiceProcessingEnabled_error_(True, None)
                    logger.debug("Voice Processing I/O enabled (HW noise suppression)")
                except Exception:
                    logger.debug("Voice Processing I/O not available")
            else:
                try:
                    input_node.setVoiceProcessingEnabled_error_(False, None)
                    logger.info(
                        "Native STT: Voice Processing I/O disabled (stt.native_voice_processing=false) — "
                        "often better for Bluetooth",
                    )
                except Exception:
                    logger.debug("Voice Processing I/O toggle skipped", exc_info=True)

            self._recognition_request = (
                _Speech.SFSpeechAudioBufferRecognitionRequest.alloc().init()
            )
            self._apply_speech_request_policy(self._recognition_request)
            self._apply_speech_request_hints(self._recognition_request)

            recording_format = input_node.outputFormatForBus_(0)
            n_channels = int(recording_format.channelCount())
            sample_rate = float(recording_format.sampleRate())
            tap_sr_override = float(self._config.get("native_tap_sample_rate") or 0)
            # Multi-channel Bluetooth formats can crash AVAudioConverter in CoreAudio; let the
            # input node convert in the tap instead (same sample rate, mono Float32 for Speech).
            # Optional native_tap_sample_rate (e.g. 48000): macOS Speech often behaves better at 44.1/48 kHz.
            tap_format: Any = recording_format
            if tap_sr_override > 0:
                mono_fmt = _AVFoundation.AVAudioFormat.alloc().initStandardFormatWithSampleRate_channels_(
                    tap_sr_override, 1,
                )
                if mono_fmt is not None:
                    tap_format = mono_fmt
                    logger.info(
                        "Native STT: tap forced to %.0f Hz mono (stt.native_tap_sample_rate) — was %.0f Hz, %d ch",
                        tap_sr_override,
                        sample_rate,
                        n_channels,
                    )
                else:
                    logger.warning("Native STT: tap sample-rate format alloc failed; using device format")
            elif n_channels > 1:
                mono_fmt = _AVFoundation.AVAudioFormat.alloc().initStandardFormatWithSampleRate_channels_(
                    sample_rate, 1,
                )
                if mono_fmt is not None:
                    tap_format = mono_fmt
                    logger.info(
                        "Native STT: %d ch @ %.0f Hz device — tap delivers mono for SFSpeech (no AVAudioConverter)",
                        n_channels,
                        sample_rate,
                    )
                else:
                    logger.warning("Native STT: mono tap format alloc failed; using device format")

            try:
                input_node.installTapOnBus_bufferSize_format_block_(
                    0, self._audio_buffer_frames, tap_format, self._audio_buffer_callback,
                )
            except Exception as exc:
                if tap_format is not recording_format:
                    logger.warning(
                        "Native STT: mono tap install failed (%s); retrying with raw device format",
                        exc,
                    )
                    tap_format = recording_format
                    input_node.installTapOnBus_bufferSize_format_block_(
                        0, self._audio_buffer_frames, tap_format, self._audio_buffer_callback,
                    )
                else:
                    raise

            self._audio_engine.prepare()
            success, error = self._audio_engine.startAndReturnError_(None)
            if not success:
                self._last_error = str(error or "AVAudioEngine start failed")
                logger.error("AVAudioEngine start failed: %s", error)
                self._cleanup()
                return False

            _wr = weakref.ref(self)

            def _pyobjc_speech_block(*args: Any) -> None:
                inst = _wr()
                if inst is not None:
                    inst._recognition_result_handler(*args)

            self._speech_pyobjc_block = _pyobjc_speech_block
            self._recognition_task = self._recognizer.recognitionTaskWithRequest_resultHandler_(
                self._recognition_request, self._speech_pyobjc_block,
            )
            if self._recognition_task is None:
                self._last_error = "SFSpeechRecognizer returned nil recognition task"
                logger.error("Native STT: recognitionTaskWithRequest returned nil — Speech unavailable")
                self._cleanup()
                return False

            self._listening = True
            self._last_speech_time = time.monotonic()
            self._last_error = None
            mic_label = _probe_default_input_mic_label()
            self.mic_name = f"{mic_label} (AVAudioEngine)"
            logger.info(
                "Native STT listening started (buffer_frames=%d, locale=%s, requires_on_device=%s, "
                "voice_processing=%s)",
                self._audio_buffer_frames,
                self._locale,
                self._native_requires_on_device,
                self._native_voice_processing,
            )
            logger.info(
                "VOICE_INPUT: mic=%s | device=%s | tap=%s | engine_running=%s | speech=%s | mic_perm=%s | "
                "launch_mode=%s",
                mic_label,
                _format_av_audio_format(recording_format),
                _format_av_audio_format(tap_format),
                bool(self._audio_engine and self._audio_engine.isRunning()),
                self.speech_permission_status,
                self.microphone_permission_status,
                os.environ.get("ATOM_LAUNCH_MODE", ""),
            )
            return True

        except Exception as exc:
            self._last_error = str(exc)
            logger.exception("Failed to start native STT")
            self._cleanup()
            return False

    def _audio_buffer_callback(self, buffer: Any, when: Any) -> None:
        """Tap callback: forward audio buffers to the recognition request."""
        if self._voice_debug:
            self._tap_buffer_count += 1
            if self._tap_buffer_count == 1 or self._tap_buffer_count % 200 == 0:
                logger.info(
                    "VOICE_INPUT: tap feeding Speech (buffers=%d) — no STT partial lines? try "
                    "stt.native_requires_on_device=false and stt.native_voice_processing=false",
                    self._tap_buffer_count,
                )
        with self._recognition_lock:
            req = self._recognition_request
            if req is not None:
                req.appendAudioPCMBuffer_(buffer)

    def _recognition_result_handler(self, *args: Any) -> None:
        """Called by SFSpeechRecognizer with partial/final results.

        PyObjC may invoke the block as ``(result, error)`` or ``(context, result, error)``.
        A wrong arity mis-binds arguments and yields no transcripts (often silently).
        """
        if len(args) == 2:
            result, error = args[0], args[1]
        elif len(args) == 3:
            _, result, error = args[0], args[1], args[2]
        else:
            logger.warning(
                "Native STT: unexpected resultHandler arity=%d (args=%r) — check PyObjC/Speech",
                len(args),
                args,
            )
            return

        if not getattr(self, "_logged_stt_handler_shape", False):
            self._logged_stt_handler_shape = True
            logger.info(
                "Native STT: recognition resultHandler invoked (arity=%d) — callbacks are wired",
                len(args),
            )

        if error is not None:
            err_desc = _format_ns_error(error)
            self._last_error = err_desc
            if "kAFAssistantErrorDomain" in err_desc or "216" in err_desc:
                logger.debug("Recognition (cancel/expected): %s", err_desc)
            else:
                logger.warning("STT recognition error: %s — attempting auto-restart", err_desc)
                try:
                    self._restart_recognition_chain()
                except Exception:
                    logger.warning("STT auto-restart after error failed, will retry via async loop")
                    self._listening = False
            return

        if result is None:
            return

        transcript = str(result.bestTranscription().formattedString())
        is_final = result.isFinal()

        try:
            segments = result.bestTranscription().segments()
            if segments and len(segments) > 0:
                confidences = [
                    float(seg.confidence()) for seg in segments
                    if float(seg.confidence()) > 0
                ]
                if confidences:
                    self._last_confidence = sum(confidences) / len(confidences)
        except Exception:
            pass

        self._last_speech_time = time.monotonic()

        if is_final:
            self._last_final = transcript
            self._last_error = None
            logger.info("STT final: '%s'", transcript)
            if self._on_final:
                self._emit_threadsafe(self._on_final, transcript)
            # One SFSpeechAudioBufferRecognitionRequest completes after each final; start a new
            # recognition task while keeping AVAudioEngine + tap (continuous listen).
            self._restart_recognition_chain()
        else:
            if transcript != self._last_partial:
                self._last_partial = transcript
                if self._voice_debug:
                    logger.info("STT partial: '%s'", transcript[:200])
                else:
                    logger.debug("STT partial: '%s'", transcript)

                from voice.listening_modes import WakeWordFilter
                if not hasattr(self, "_wake_filter"):
                    self._wake_filter = WakeWordFilter(cooldown_s=1.5)
                wake_match = self._wake_filter.check(transcript)
                if wake_match:
                    logger.info("Wake phrase detected in partial: '%s'", wake_match)
                    self._emit_threadsafe(
                        lambda p=wake_match: self._bus.emit(
                            "wake_word_detected", wake_word=p,
                        ),
                        None,
                    )

                if self._on_partial:
                    self._emit_threadsafe(self._on_partial, transcript)

    def _restart_recognition_chain(self) -> None:
        """After each isFinal, start a new request+task; the engine tap keeps running.

        SFSpeechRecognizer completes one streaming request per final; without a new
        request, buffers keep appending to a finished session and no further results fire.
        """
        if not self._listening or self._audio_engine is None or self._recognizer is None:
            return
        try:
            with self._recognition_lock:
                old_task = self._recognition_task
                self._recognition_request = None
                self._recognition_task = None
            if old_task is not None:
                try:
                    old_task.cancel()
                except Exception:
                    pass

            new_req = _Speech.SFSpeechAudioBufferRecognitionRequest.alloc().init()
            self._apply_speech_request_policy(new_req)
            self._apply_speech_request_hints(new_req)

            block = self._speech_pyobjc_block or self._recognition_result_handler
            new_task = self._recognizer.recognitionTaskWithRequest_resultHandler_(
                new_req, block,
            )
            if new_task is None:
                logger.warning("Native STT: recognition restart returned nil task — forcing full restart")
                self._last_error = "Speech recognition task nil on restart"
                self._listening = False
                return
            with self._recognition_lock:
                self._recognition_request = new_req
                self._recognition_task = new_task
            self._last_partial = ""
            logger.debug("Native STT: recognition chain restarted for next utterance")
        except Exception as exc:
            self._last_error = str(exc)
            logger.warning("STT: failed to restart recognition chain: %s — forcing full restart", exc)
            self._listening = False

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

        if self._native_stop_audio_delay_s > 0:
            time.sleep(self._native_stop_audio_delay_s)

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

        self._speech_pyobjc_block = None

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
        self._speech_pyobjc_block = None

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
        from core.state_manager import AtomState

        post_cd_s = float(self._config.get("post_tts_cooldown_ms", 800) or 0) / 1000.0

        while getattr(self, "_running_async", False):
            now = time.monotonic()
            if self._voice_debug and now >= self._dbg_next_status_log:
                self._dbg_next_status_log = now + self._voice_debug_interval_s
                cur = self._state.current
                hint = ""
                if cur is not AtomState.LISTENING and not (
                    self._barge_in_during_speak and cur is AtomState.SPEAKING
                ):
                    hint = (
                        " | NOTE: mic only opens when AtomState=LISTENING "
                        "(blocked during THINKING/SPEAKING/SLEEP/IDLE)"
                        + (
                            " [barge_in_during_speak: mic may open during SPEAKING]"
                            if self._barge_in_during_speak
                            else ""
                        )
                    )
                logger.info(
                    "VOICE_DEBUG: atom_state=%s | native_stt_listening=%s | mic_field=%s | "
                    "stt_available=%s | speech_perm=%s | mic_perm=%s | last_err=%s | launch_mode=%s%s",
                    getattr(cur, "value", cur),
                    self._listening,
                    (self.mic_name or "")[:120],
                    self._available,
                    self.speech_permission_status,
                    self.microphone_permission_status,
                    self._last_error,
                    os.environ.get("ATOM_LAUNCH_MODE", ""),
                    hint,
                )

            if not self._listening:
                cur = self._state.current
                allow_mic = (
                    cur is AtomState.LISTENING
                    or cur is AtomState.IDLE
                    or (self._barge_in_during_speak and cur is AtomState.SPEAKING)
                    or cur is AtomState.THINKING
                )
                if cur is AtomState.SLEEP:
                    await asyncio.sleep(0.5)
                    continue
                if not allow_mic:
                    await asyncio.sleep(0.12)
                    continue
                if self._need_post_tts_cooldown:
                    self._need_post_tts_cooldown = False
                    if post_cd_s > 0:
                        logger.info(
                            "Native STT: post-TTS cooldown %.2fs before mic (config stt.post_tts_cooldown_ms)",
                            post_cd_s,
                        )
                        await asyncio.sleep(post_cd_s)
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
            elif self._last_speech_time > 0:
                idle_s = time.monotonic() - self._last_speech_time
                if idle_s > _MAX_RECORD_S:
                    logger.info(
                        "STT: no speech for %.1fs (max=%.1fs) — restarting recognition chain",
                        idle_s, _MAX_RECORD_S,
                    )
                    self._last_speech_time = time.monotonic()
                    try:
                        self._restart_recognition_chain()
                    except Exception:
                        logger.warning("STT: timeout restart failed, forcing full restart")
                        self._listening = False
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

            if old is AtomState.SPEAKING and new is AtomState.LISTENING:
                self._need_post_tts_cooldown = True

            if new is AtomState.SLEEP:
                self.stop()
            elif new in (AtomState.LISTENING, AtomState.SPEAKING, AtomState.THINKING, AtomState.IDLE):
                already_running = (
                    self._async_task is not None
                    and not self._async_task.done()
                )
                if not self._running_async and not already_running:
                    self._running_async = True
                    self._async_task = asyncio.create_task(
                        self.async_start_listening()
                    )
        except Exception:
            logger.debug("on_state_changed error", exc_info=True)
