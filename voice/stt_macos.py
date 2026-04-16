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
import collections
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


_BLUETOOTH_HINTS = frozenset({
    "airpods", "buds", "bluetooth", "bt ", "wireless", "jbl", "sony",
    "bose", "beats", "jabra", "galaxy", "oneplus", "nord",
})


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
        name = str(dev.localizedName())
        if any(hint in name.lower() for hint in _BLUETOOTH_HINTS):
            logger.warning(
                "VOICE_INPUT: active mic '%s' appears to be Bluetooth — "
                "for best STT quality, switch macOS Sound Input to MacBook "
                "built-in mic (System Settings > Sound > Input)",
                name,
            )
        return name
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
        self._partial_stable_since: float = 0.0
        self._partial_finalize_s: float = 1.8
        # After TTS, wait before reopening mic (avoids AVAudioEngine vs output conflict).
        self._need_post_tts_cooldown: bool = False
        self._audio_prebuffer: collections.deque = collections.deque(maxlen=12)

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
        self._native_requires_on_device_cfg: bool = bool(
            self._config.get("native_requires_on_device", True),
        )
        self._native_requires_on_device: bool = self._native_requires_on_device_cfg
        # Voice Processing I/O can break or silence some Bluetooth headsets — disable to test.
        self._native_voice_processing: bool = bool(
            self._config.get("native_voice_processing", True),
        )
        self._tap_buffer_count: int = 0
        self._recognizer_supports_on_device: bool = False
        self._consecutive_silent_buffers: int = 0
        self._engine_restart_count: int = 0
        self._max_engine_restarts: int = 3
        self._rebind_audio_device: Callable[[], bool] | None = None
        self._preferred_device_name: str = ""
        self._signal_verified: bool = False
        self._sd_stream: Any = None
        self._sd_audio_format: Any = None
        self._using_sounddevice: bool = False
        self._speech_runloop_task: asyncio.Task | None = None
        self._last_result_callback_time: float = 0.0
        self._last_listen_start_time: float = 0.0
        self._last_audio_rms_db: float = -96.0
        self._last_speech_candidate_time: float = 0.0
        self._callback_starvation_count: int = 0
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

    def set_audio_device_rebinder(
        self,
        rebind_fn: Callable[[], bool],
        device_name: str,
    ) -> None:
        """Provide a callable that re-asserts the CoreAudio system default input.

        Called by main.py after AudioIntelligenceEngine selects a device, so
        that on engine restarts we can re-lock the device before rebuilding
        AVAudioEngine (prevents macOS from silently switching back to BT).
        """
        self._rebind_audio_device = rebind_fn
        self._preferred_device_name = device_name

    def preflight_mic_check(self, timeout_s: float = 0.5) -> bool:
        """Quick sounddevice capture to verify the mic is actually live.

        Returns True if RMS > -80 dB (real audio detected), False otherwise.
        Called before starting AVAudioEngine to catch dead/muted mics early.
        """
        try:
            import sounddevice as sd
            import numpy as np

            sr = 16000
            frames = int(sr * timeout_s)
            audio = sd.rec(frames, samplerate=sr, channels=1, dtype="float32")
            sd.wait()
            rms = float(np.sqrt(np.mean(audio ** 2)))
            rms_db = 20.0 * float(np.log10(max(rms, 1e-10)))
            ok = rms_db > -80.0
            logger.info(
                "VOICE_INPUT: preflight mic check — rms=%.1f dB, signal=%s",
                rms_db, "OK" if ok else "DEAD",
            )
            return ok
        except Exception as exc:
            logger.debug("VOICE_INPUT: preflight mic check skipped: %s", exc)
            return True

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
            self._consecutive_silent_buffers = 0
            self._signal_verified = False
            self._logged_stt_handler_shape = False
            self._logged_buf_fail = False
            self._last_result_callback_time = 0.0
            self._last_audio_rms_db = -96.0
            self._last_speech_candidate_time = 0.0

            self._recognition_request = (
                _Speech.SFSpeechAudioBufferRecognitionRequest.alloc().init()
            )
            self._apply_speech_request_policy(self._recognition_request)
            self._apply_speech_request_hints(self._recognition_request)

            # --- capture path: sounddevice (preferred) or AVAudioEngine (fallback) ---
            sd_ok = self._start_sounddevice_capture()

            if sd_ok:
                self._using_sounddevice = True
                mic_label = _probe_default_input_mic_label()
                device_fmt_label = "16000 Hz, 1 ch"
                capture_label = "sounddevice (PortAudio/CoreAudio direct)"
            else:
                self._using_sounddevice = False
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

                try:
                    hw_format = input_node.inputFormatForBus_(0)
                    hw_ch = int(hw_format.channelCount()) if hw_format else 0
                    hw_sr = float(hw_format.sampleRate()) if hw_format else 0.0
                    if hw_ch == 0 or hw_sr == 0:
                        logger.warning(
                            "VOICE_INPUT: inputNode hardware format is dead "
                            "(ch=%d, sr=%.0f) — IO graph stale, aborting start",
                            hw_ch, hw_sr,
                        )
                        self._last_error = "Dead IO graph (hardware format ch=0 or sr=0)"
                        self._cleanup()
                        return False
                    else:
                        logger.info(
                            "VOICE_INPUT: hardware input format — %.0f Hz, %d ch",
                            hw_sr, hw_ch,
                        )
                except Exception:
                    logger.debug("VOICE_INPUT: inputFormatForBus_ query failed", exc_info=True)

                recording_format = input_node.outputFormatForBus_(0)
                n_channels = int(recording_format.channelCount())
                sample_rate = float(recording_format.sampleRate())

                input_node.installTapOnBus_bufferSize_format_block_(
                    0, self._audio_buffer_frames, recording_format,
                    self._audio_buffer_callback,
                )

                self._audio_engine.prepare()
                success, error = self._audio_engine.startAndReturnError_(None)
                if not success:
                    self._last_error = str(error or "AVAudioEngine start failed")
                    logger.error("AVAudioEngine start failed: %s", error)
                    self._cleanup()
                    return False

                mic_label = _probe_default_input_mic_label()
                device_fmt_label = _format_av_audio_format(recording_format)
                capture_label = f"{sample_rate:.0f} Hz, {n_channels} ch (AVAudioEngine tap)"

            # --- recognition task (shared by both capture paths) ---
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
            self._last_listen_start_time = self._last_speech_time
            self._last_error = None
            self._ensure_speech_runloop_pump()
            capture_mode = "sounddevice" if self._using_sounddevice else "AVAudioEngine"
            self.mic_name = f"{mic_label} ({capture_mode})"
            logger.info(
                "Native STT listening started (capture=%s, buffer_frames=%d, locale=%s, "
                "requires_on_device=%s, voice_processing=%s)",
                capture_mode,
                self._audio_buffer_frames,
                self._locale,
                self._native_requires_on_device,
                self._native_voice_processing,
            )
            logger.info(
                "VOICE_INPUT: mic=%s | device=%s | capture=%s | running=%s | speech=%s | mic_perm=%s | "
                "launch_mode=%s",
                mic_label,
                device_fmt_label,
                capture_label,
                self._using_sounddevice or bool(self._audio_engine and self._audio_engine.isRunning()),
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

    def _note_audio_activity(self, rms_db: float | None) -> None:
        """Track recent audio energy so watchdog can distinguish speech from ambience."""
        if rms_db is None:
            return
        self._last_audio_rms_db = float(rms_db)
        # Rough speech-likelihood gate: stronger than quiet-room ambience.
        if rms_db > -42.0:
            self._last_speech_candidate_time = time.monotonic()

    def _ensure_speech_runloop_pump(self) -> None:
        """Pump Cocoa's run loop so Speech framework result handlers actually fire.

        SFSpeechRecognizer callbacks are delivered through the Foundation run loop.
        ATOM primarily runs on asyncio, so without this pump the recognizer task
        can stay silent forever even while audio buffers flow correctly.
        """
        loop = self._loop
        if loop is None or _Foundation is None:
            return
        task = self._speech_runloop_task
        if task is not None and not task.done():
            return
        self._speech_runloop_task = loop.create_task(self._pump_speech_runloop())

    def _cancel_speech_runloop_pump(self) -> None:
        task = self._speech_runloop_task
        self._speech_runloop_task = None
        if task is not None and not task.done():
            task.cancel()

    async def _pump_speech_runloop(self) -> None:
        """Service the main-thread Cocoa run loop alongside asyncio."""
        if _Foundation is None:
            return
        runloop = _Foundation.NSRunLoop.currentRunLoop()
        try:
            while self._running_async or self._listening:
                try:
                    runloop.runUntilDate_(
                        _Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.01),
                    )
                except Exception:
                    logger.debug("Native STT: NSRunLoop pump error", exc_info=True)
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass

    def _on_recognition_starvation(self) -> None:
        """Progressively relax recognizer constraints when mic is healthy but callbacks never arrive."""
        self._callback_starvation_count += 1
        if self._callback_starvation_count == 1 and self._native_requires_on_device:
            self._native_requires_on_device = False
            logger.warning(
                "VOICE_INPUT: healthy mic but no Speech callbacks — disabling "
                "requiresOnDeviceRecognition for the next recognition task",
            )
        elif self._callback_starvation_count >= 2:
            logger.warning(
                "VOICE_INPUT: repeated recognizer callback starvation (count=%d, locale=%s)",
                self._callback_starvation_count,
                self._locale,
            )

    def _audio_buffer_callback(self, buffer: Any, when: Any) -> None:
        """Tap callback: forward audio buffers to the recognition request.

        Signal gate: RMS is checked on the first 60 buffers (~2.8s at 44.1kHz
        with 2048-frame buffers).  If real audio is detected (RMS > -80 dB),
        ``_signal_verified`` is set so callers know the mic path is healthy.

        Silence detection: if 15 consecutive buffers are digital-silence
        (RMS <= -95 dB), the engine is torn down and rebuilt with progressive
        configuration fallback (VPIO disable, on-device relaxation, device
        rebinding).
        """
        self._tap_buffer_count += 1

        check_rms = self._tap_buffer_count <= 60 or (
            self._voice_debug and self._tap_buffer_count % 200 == 0
        )
        rms_db = self._estimate_rms_db(buffer) if check_rms else None
        self._note_audio_activity(rms_db)

        if rms_db is not None:
            if rms_db <= -95.0:
                self._consecutive_silent_buffers += 1
            else:
                self._consecutive_silent_buffers = 0
                if not self._signal_verified:
                    self._signal_verified = True
                    logger.info(
                        "VOICE_INPUT: mic signal verified (rms=%.1f dB at buffer %d) — audio path healthy",
                        rms_db, self._tap_buffer_count,
                    )

        if (self._consecutive_silent_buffers >= 15
                and self._engine_restart_count < self._max_engine_restarts):
            self._engine_restart_count += 1
            logger.warning(
                "VOICE_INPUT: %d consecutive silent buffers (rms<=−95 dB) — "
                "restarting AVAudioEngine to rebind audio session (attempt %d/%d)",
                self._consecutive_silent_buffers,
                self._engine_restart_count,
                self._max_engine_restarts,
            )
            self._consecutive_silent_buffers = 0
            self._schedule_engine_restart()
            return

        if self._voice_debug and rms_db is not None and (
            self._tap_buffer_count == 1 or self._tap_buffer_count % 200 == 0
        ):
            logger.info(
                "VOICE_INPUT: tap feeding Speech (buffers=%d, rms=%.1f dB, signal_ok=%s)",
                self._tap_buffer_count, rms_db, self._signal_verified,
            )

        with self._recognition_lock:
            req = self._recognition_request
            if req is not None:
                req.appendAudioPCMBuffer_(buffer)

    # ── sounddevice capture (bypasses AVAudioEngine input tap) ──

    def _start_sounddevice_capture(self) -> bool:
        """Open a sounddevice InputStream that feeds audio to SFSpeechRecognitionRequest.

        AVAudioEngine's input tap delivers zero-valued buffers when input and
        output devices use different CoreAudio clock domains (e.g. built-in mic +
        Bluetooth output).  sounddevice (PortAudio) reads from CoreAudio directly
        and is immune to this issue.

        SFSpeechRecognizer requires audio at the mic's native sample rate (typically
        48 kHz on MacBook Air).  Resampling to 16 kHz causes "No speech detected".
        """
        try:
            import sounddevice as sd

            dev_info = sd.query_devices(kind="input")
            native_sr = int(dev_info.get("default_samplerate", 48000))
            sr = native_sr if native_sr > 0 else 48000
            blocksize = max(self._audio_buffer_frames, sr // 8)

            self._sd_audio_format = (
                _AVFoundation.AVAudioFormat.alloc()
                .initStandardFormatWithSampleRate_channels_(float(sr), 1)
            )
            if self._sd_audio_format is None:
                logger.debug("sounddevice capture: AVAudioFormat alloc failed")
                return False

            self._sd_stream = sd.InputStream(
                samplerate=sr,
                channels=1,
                dtype="float32",
                blocksize=blocksize,
                callback=self._sd_audio_callback,
            )
            self._sd_stream.start()
            logger.info(
                "Native STT: using sounddevice capture (PortAudio/CoreAudio) — "
                "%d Hz (native), 1 ch, blocksize=%d — bypassing AVAudioEngine input tap",
                sr, blocksize,
            )
            return True
        except Exception as exc:
            logger.info(
                "sounddevice capture unavailable (%s), falling back to AVAudioEngine", exc,
            )
            self._sd_stream = None
            self._sd_audio_format = None
            return False

    def _sd_audio_callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        """sounddevice InputStream callback — mirrors _audio_buffer_callback."""
        import numpy as np

        self._tap_buffer_count += 1

        check_rms = self._tap_buffer_count <= 60 or (
            self._voice_debug and self._tap_buffer_count % 200 == 0
        )
        rms_db: float | None = None
        if check_rms:
            rms = float(np.sqrt(np.mean(indata ** 2)))
            rms_db = 20.0 * float(np.log10(max(rms, 1e-10))) if rms > 1e-10 else -96.0
        self._note_audio_activity(rms_db)

        if rms_db is not None:
            if rms_db <= -95.0:
                self._consecutive_silent_buffers += 1
            else:
                self._consecutive_silent_buffers = 0
                if not self._signal_verified:
                    self._signal_verified = True
                    logger.info(
                        "VOICE_INPUT: mic signal verified (rms=%.1f dB at buffer %d) — audio path healthy",
                        rms_db, self._tap_buffer_count,
                    )

        if rms_db is not None and (
            self._tap_buffer_count == 1 or self._tap_buffer_count % 200 == 0
        ):
            logger.info(
                "VOICE_INPUT: tap feeding Speech (buffers=%d, rms=%.1f dB, signal_ok=%s)",
                self._tap_buffer_count, rms_db, self._signal_verified,
            )

        self._maybe_finalize_stable_partial()

        import numpy as np
        self._audio_prebuffer.append(np.array(indata, copy=True))

        with self._recognition_lock:
            req = self._recognition_request
            if req is not None:
                buf = self._numpy_to_pcm_buffer(indata, frames)
                if buf is not None:
                    req.appendAudioPCMBuffer_(buf)
                elif not getattr(self, "_logged_buf_fail", False):
                    self._logged_buf_fail = True
                    logger.error(
                        "VOICE_INPUT: _numpy_to_pcm_buffer returned None — "
                        "no audio will reach speech recognizer (check AVAudioPCMBuffer creation)",
                    )

    def _maybe_finalize_stable_partial(self) -> None:
        """Promote a partial to final if text has been stable for _partial_finalize_s.

        Called from the audio callback so it runs at capture rate (~60-100 Hz).
        SFSpeechRecognizer in streaming mode often never sets isFinal=True,
        so this is the primary mechanism to emit speech_final events.
        """
        if not self._partial_stable_since:
            return
        text = self._last_partial
        if not text or not text.strip():
            return
        now = time.monotonic()
        elapsed = now - self._partial_stable_since
        if elapsed < self._partial_finalize_s:
            return
        logger.info(
            "STT: partial stable for %.1fs — promoting to final: '%s'",
            elapsed, text,
        )
        self._last_final = text
        self._last_error = None
        self._last_partial = ""
        self._partial_stable_since = 0.0
        if self._on_final:
            self._emit_threadsafe(self._on_final, text)
        self._restart_recognition_chain()

    def _numpy_to_pcm_buffer(self, np_data: Any, frames: int) -> Any:
        """Convert a numpy float32 array to AVAudioPCMBuffer for SFSpeech.

        Uses pyobjc varlist element-wise assignment (verified at 0.05ms / 2048
        frames).  ctypes.cast / memmove cannot handle objc.varlist pointers.
        """
        try:
            fmt = self._sd_audio_format
            if fmt is None:
                return None
            buf = _AVFoundation.AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(
                fmt, frames,
            )
            if buf is None:
                return None
            buf.setFrameLength_(frames)
            fcd = buf.floatChannelData()
            if fcd is None:
                return None
            ch0 = fcd[0]
            vals = np_data.ravel().tolist()
            for i, v in enumerate(vals):
                ch0[i] = v
            return buf
        except Exception:
            if not getattr(self, "_logged_buf_fail", False):
                self._logged_buf_fail = True
                logger.warning("VOICE_INPUT: AVAudioPCMBuffer creation failed", exc_info=True)
            return None

    @staticmethod
    def _estimate_rms_db(buffer: Any) -> float:
        """Quick RMS estimate from the first channel of an AVAudioPCMBuffer (dBFS).

        Uses objc.varlist.as_buffer() for zero-copy numpy access.
        ctypes.cast does NOT work with pyobjc varlist pointers.
        """
        try:
            import numpy as np

            data = buffer.floatChannelData()
            if data is None:
                return -96.0
            count = int(buffer.frameLength())
            if count == 0:
                return -96.0
            raw = data[0].as_buffer(count)
            arr = np.frombuffer(raw, dtype=np.float32)
            rms = float(np.sqrt(np.mean(arr ** 2)))
            if rms < 1e-10:
                return -96.0
            return 20.0 * float(np.log10(rms))
        except Exception:
            return -96.0

    def _schedule_engine_restart(self) -> None:
        """Schedule a full AVAudioEngine teardown + rebuild on the main thread.

        Called from the tap callback when sustained digital silence is detected
        (typically after a CoreAudio default-device switch that the engine
        didn't pick up).
        """
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(self._restart_audio_engine)
        except RuntimeError:
            pass

    def _restart_audio_engine(self) -> None:
        """Tear down and rebuild the capture path to rebind to the current system default input.

        Uses progressive fallback to resolve persistent silence:
          attempt 1: same config + re-assert CoreAudio default
          attempt 2: disable Voice Processing I/O (AEC silences mic when
                     input/output are on different hardware)
          attempt 3: also relax on-device recognition requirement
        """
        has_capture = self._audio_engine is not None or self._using_sounddevice
        if not self._listening or not has_capture:
            return

        from core.state_manager import AtomState
        cur = self._state.current
        if cur is AtomState.SPEAKING:
            logger.debug(
                "VOICE_INPUT: deferring engine restart — TTS is speaking "
                "(VPIO echo cancellation causes silence during playback)",
            )
            if self._loop:
                self._loop.call_later(2.0, self._restart_audio_engine)
            return

        attempt = self._engine_restart_count

        if attempt >= 2 and self._native_voice_processing:
            self._native_voice_processing = False
            logger.info(
                "VOICE_INPUT: attempt %d — disabling Voice Processing I/O "
                "(echo cancellation causes silence when input != output hardware)",
                attempt,
            )

        if attempt >= 3 and self._native_requires_on_device:
            self._native_requires_on_device = False
            logger.info(
                "VOICE_INPUT: attempt %d — allowing server-assisted recognition "
                "(on-device may yield no partials for some locale/device combos)",
                attempt,
            )

        logger.info(
            "VOICE_INPUT: rebuilding AVAudioEngine (attempt %d/%d, vpio=%s, on_device=%s)",
            attempt, self._max_engine_restarts,
            self._native_voice_processing, self._native_requires_on_device,
        )

        on_final = self._on_final
        on_partial = self._on_partial
        loop = self._loop

        self.stop_listening()
        self._cleanup()

        if self._rebind_audio_device:
            try:
                ok = self._rebind_audio_device()
                logger.info(
                    "VOICE_INPUT: re-asserted CoreAudio default to '%s' before rebuild (ok=%s)",
                    self._preferred_device_name, ok,
                )
            except Exception:
                logger.debug("VOICE_INPUT: device rebind failed", exc_info=True)

        import time as _time
        _time.sleep(0.8)

        ok = self.start_listening(loop=loop, on_final=on_final, on_partial=on_partial)
        if ok:
            logger.info("VOICE_INPUT: AVAudioEngine restarted successfully — mic should be live now")
        else:
            logger.error("VOICE_INPUT: AVAudioEngine restart failed — %s", self._last_error)

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
        self._last_result_callback_time = time.monotonic()
        self._callback_starvation_count = 0
        # Restore on-device preference once callbacks are flowing
        if not self._native_requires_on_device and self._native_requires_on_device_cfg:
            self._native_requires_on_device = True
            logger.info("Native STT: callbacks flowing — re-enabling requiresOnDeviceRecognition")

        if error is not None:
            err_desc = _format_ns_error(error)
            is_no_speech = "1110" in err_desc or "No speech detected" in err_desc
            is_cancel = "216" in err_desc or "kLSRErrorDomain" in err_desc
            is_expected = "kAFAssistantErrorDomain" in err_desc or is_cancel

            if is_no_speech:
                logger.debug("Recognition: no speech detected (normal silence) — restarting chain quietly")
                self._last_error = None
                try:
                    self._restart_recognition_chain()
                except Exception:
                    self._listening = False
            elif is_expected:
                logger.debug("Recognition (cancel/expected): %s", err_desc)
                self._last_error = err_desc
            else:
                self._last_error = err_desc
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
            self._partial_stable_since = 0.0
            logger.info("STT final: '%s'", transcript)
            if self._on_final:
                self._emit_threadsafe(self._on_final, transcript)
            self._restart_recognition_chain()
        else:
            now = time.monotonic()
            if transcript != self._last_partial:
                self._last_partial = transcript
                self._partial_stable_since = now
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
            elif (
                transcript.strip()
                and self._partial_stable_since > 0
                and (now - self._partial_stable_since) >= self._partial_finalize_s
            ):
                logger.info(
                    "STT: partial stable for %.1fs — treating as final: '%s'",
                    now - self._partial_stable_since, transcript,
                )
                self._last_final = transcript
                self._last_error = None
                self._last_partial = ""
                self._partial_stable_since = 0.0
                if self._on_final:
                    self._emit_threadsafe(self._on_final, transcript)
                self._restart_recognition_chain()

    def _flush_prebuffer_to_request(self, req: Any) -> int:
        """Replay buffered audio frames into a fresh recognition request.

        Returns the number of frames flushed.  This ensures speech that arrived
        during a recognition-chain swap is not lost.
        """
        frames_flushed = 0
        snapshot = list(self._audio_prebuffer)
        for chunk in snapshot:
            buf = self._numpy_to_pcm_buffer(chunk, chunk.shape[0])
            if buf is not None:
                try:
                    req.appendAudioPCMBuffer_(buf)
                    frames_flushed += chunk.shape[0]
                except Exception:
                    break
        return frames_flushed

    def _restart_recognition_chain(self) -> None:
        """After each isFinal, start a new request+task; the engine tap keeps running.

        SFSpeechRecognizer completes one streaming request per final; without a new
        request, buffers keep appending to a finished session and no further results fire.
        """
        has_capture = self._audio_engine is not None or self._using_sounddevice
        if not self._listening or not has_capture or self._recognizer is None:
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

            flushed = self._flush_prebuffer_to_request(new_req)

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
            if flushed:
                logger.debug(
                    "Native STT: recognition chain restarted (pre-buffered %d frames)",
                    flushed,
                )
            else:
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
        self._partial_stable_since = 0.0

        if self._native_stop_audio_delay_s > 0:
            time.sleep(self._native_stop_audio_delay_s)

        if self._sd_stream is not None:
            try:
                self._sd_stream.stop()
            except Exception:
                logger.debug("sounddevice stream stop error", exc_info=True)

        if self._audio_engine is not None:
            try:
                self._audio_engine.inputNode().removeTapOnBus_(0)
                self._audio_engine.stop()
                self._audio_engine.reset()
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
        if self._sd_stream is not None:
            try:
                self._sd_stream.stop()
            except Exception:
                pass
            try:
                self._sd_stream.close()
            except Exception:
                pass
        self._sd_stream = None
        self._sd_audio_format = None
        if self._audio_engine is not None:
            try:
                self._audio_engine.inputNode().removeTapOnBus_(0)
            except Exception:
                pass
            try:
                self._audio_engine.stop()
            except Exception:
                pass
            try:
                self._audio_engine.reset()
            except Exception:
                pass
        self._audio_engine = None
        self._recognition_request = None
        self._recognition_task = None
        self._speech_pyobjc_block = None

    def shutdown(self) -> None:
        """Full shutdown."""
        self._cancel_speech_runloop_pump()
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
        self._ensure_speech_runloop_pump()

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

        try:
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
        finally:
            self._cancel_speech_runloop_pump()

    def stop(self) -> None:
        """Stop async listen loop + underlying recognition."""
        self._running_async = False
        self._cancel_speech_runloop_pump()
        self.stop_listening()

    async def on_state_changed(self, old, new, **_kw) -> None:
        """Handle ATOM state transitions (mirrors STTAsync behaviour)."""
        from core.state_manager import AtomState

        try:
            if self._permanently_disabled:
                return

            if old is AtomState.SPEAKING and new is AtomState.LISTENING:
                self._need_post_tts_cooldown = True
                if self._listening:
                    self._restart_recognition_chain()
                    self._audio_prebuffer.clear()

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
