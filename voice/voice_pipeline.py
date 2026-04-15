"""
ATOM -- Voice Pipeline (unified voice lifecycle owner).

Consolidates STT/TTS construction, wake word, and interrupt handling
into a single class. Replaces the 300+ line factory previously inlined
in main.py.

Lifecycle:
    1. IDLE/MONITORING -- WakeWordEngine runs at <1 % CPU
    2. LISTENING       -- STT captures speech (silence timeout)
    3. PROCESSING      -- CommandLoop handles the command
    4. SPEAKING         -- TTS plays response
    5. Return to 1.

Pipeline ownership:
    VoicePipeline owns STT, TTS, WakeWordEngine, VoiceInterruptHandler.
    It does NOT own the Router or CommandLoop -- those are passed in.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.intent_engine import IntentEngine
    from core.state_manager import StateManager
    from voice.mic_manager import MicManager

logger = logging.getLogger("atom.voice_pipeline")


class _DisabledSTT:
    """Placeholder when no STT backend is available."""

    def __init__(self, reason: str) -> None:
        self._reason = reason
        self._last_error = reason
        self.mic_name = "Voice input unavailable"
        self.backend_name = "Disabled"
        self.fallback_chain = [reason]
        self.speech_permission_status = (
            "bundle_missing_usage_description"
            if "NSSpeechRecognitionUsageDescription" in reason
            else "unavailable"
        )
        self.microphone_permission_status = (
            "dependency_missing"
            if "PyAudio/PortAudio" in reason
            else "unknown"
        )

    async def async_preload(self) -> None:
        logger.warning("STT disabled: %s", self._reason)

    async def async_start_listening(self, **_kw: Any) -> None:
        logger.warning("STT disabled: %s", self._reason)

    async def start_listening(self, **_kw: Any) -> None:
        await self.async_start_listening(**_kw)

    async def on_state_changed(self, old: Any, new: Any, **_kw: Any) -> None:
        return None

    def stop(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


class VoicePipeline:
    """Single owner of STT + TTS + WakeWord + Interrupt lifecycle."""

    def __init__(
        self,
        bus: AsyncEventBus,
        state: StateManager,
        config: dict,
        *,
        mic_manager: MicManager | None = None,
        intent_engine: IntentEngine | None = None,
    ) -> None:
        self._bus = bus
        self._state = state
        self._config = config
        self._mic_manager = mic_manager
        self._intent_engine = intent_engine

        self.stt: Any = None
        self.tts: Any = None
        self.stt_runtime_label: str = "Voice input unavailable"
        self.stt_runtime_error: str = ""
        self.stt_runtime_fallbacks: list[str] = []
        self.tts_runtime_label: str = "macOS Native"

        self._wake_word: Any = None
        self._interrupt_handler: Any = None
        self._stt_watchdog: Any = None
        self._listening_mode: Any = None
        self._loop_task: asyncio.Task | None = None
        self._audio_intel: Any = None

    def build(self) -> None:
        """Construct STT and TTS engines based on config + platform."""
        self._build_stt()
        self._build_tts()
        logger.info(
            "VoicePipeline built: stt=%s tts=%s",
            self.stt_runtime_label,
            self.tts_runtime_label,
        )

    def _build_disabled_stt(self, reason: str) -> _DisabledSTT:
        logger.error("Voice input unavailable: %s", reason)
        if sys.platform == "darwin" and "NSSpeechRecognitionUsageDescription" in reason:
            logger.warning(
                "VOICE_INPUT: On macOS, speech needs the ATOM.app bundle "
                "(usage strings in Info.plist). Run via 'Run ATOM.command'.",
            )
        elif sys.platform != "darwin" and (
            "PyAudio" in reason or "faster-whisper" in reason or "SpeechRecognition" in reason
        ):
            logger.warning(
                "VOICE_INPUT: Install offline STT deps "
                "(`pip install faster-whisper SpeechRecognition pyaudio`).",
            )
        elif sys.platform == "darwin" and (
            "PyAudio" in reason or "faster-whisper" in reason or "SpeechRecognition" in reason
        ):
            logger.warning(
                "VOICE_INPUT: On macOS only Apple native STT is wired -- "
                "use ATOM.app / Run ATOM.command with stt.engine macos_native or auto.",
            )
        return _DisabledSTT(reason)

    def _build_google_stt(self) -> tuple[Any | None, str]:
        missing: list[str] = []
        try:
            import speech_recognition  # noqa: F401
        except ImportError:
            missing.append("SpeechRecognition")
        try:
            import pyaudio  # noqa: F401
        except ImportError:
            missing.append("PyAudio/PortAudio")

        if missing:
            return None, "Google STT dependencies missing: " + ", ".join(missing)

        from voice.stt_google import STTGoogle

        logger.info("STT: Google Online (free, fast, bilingual)")
        return STTGoogle(
            self._bus,
            self._state,
            self._config,
            mic_manager=self._mic_manager,
            intent_engine=self._intent_engine,
        ), ""

    def _build_faster_whisper_stt(self) -> Any:
        missing: list[str] = []
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            missing.append("faster-whisper")
        try:
            import speech_recognition  # noqa: F401
        except ImportError:
            missing.append("SpeechRecognition")
        try:
            import pyaudio  # noqa: F401
        except ImportError:
            missing.append("PyAudio/PortAudio")

        if missing:
            return self._build_disabled_stt(
                "Offline STT dependencies missing: " + ", ".join(missing),
            )

        from voice.stt_async import STTAsync

        logger.info("STT: faster-whisper (offline fallback)")
        return STTAsync(
            self._bus,
            self._state,
            self._config,
            mic_manager=self._mic_manager,
            intent_engine=self._intent_engine,
        )

    def _build_native_stt(self) -> tuple[Any | None, str]:
        from voice.stt_macos import NativeSTT, native_stt_launch_supported

        native_ok, native_reason = native_stt_launch_supported()
        if not native_ok:
            return None, native_reason
        return NativeSTT(
            self._bus,
            self._state,
            self._config,
            mic_manager=self._mic_manager,
            intent_engine=self._intent_engine,
        ), ""

    def _build_stt(self) -> None:
        stt_cfg = self._config.get("stt", {})
        engine_pref = str(stt_cfg.get("engine", "macos_native") or "macos_native").strip().lower()
        logger.info("STT engine preference: %s (platform=%s)", engine_pref, sys.platform)

        if sys.platform == "darwin":
            if engine_pref in ("macos_native", "auto"):
                native_stt, native_reason = self._build_native_stt()
                if native_stt is not None:
                    self.stt = native_stt
                    self.stt_runtime_label = "macOS Native (SFSpeechRecognizer)"
                    logger.info("STT: macOS Native -- Apple stack only")
                else:
                    self.stt_runtime_error = native_reason or ""
                    self.stt_runtime_fallbacks.append(f"native unavailable: {native_reason}")
                    self.stt = self._build_disabled_stt(
                        native_reason
                        or "Native STT unavailable -- use ATOM.app / Run ATOM.command",
                    )
                    self.stt_runtime_label = "Disabled"
            elif engine_pref in ("faster_whisper", "google_online", "google"):
                msg = (
                    f"stt.engine={engine_pref} is not used on macOS -- "
                    "use macos_native or auto (SFSpeechRecognizer only)"
                )
                self.stt_runtime_fallbacks.append(msg)
                self.stt = self._build_disabled_stt(msg)
                self.stt_runtime_label = "Disabled"
            else:
                self.stt = self._build_disabled_stt(f"Unknown STT engine: {engine_pref}")
                self.stt_runtime_label = "Disabled"
        elif engine_pref in ("google_online", "google"):
            google_stt, google_err = self._build_google_stt()
            if google_stt is not None:
                self.stt = google_stt
                self.stt_runtime_label = "Google Online (free)"
            else:
                self.stt_runtime_error = google_err or ""
                self.stt_runtime_fallbacks.append(f"google unavailable: {google_err}")
                logger.warning("Google STT unavailable (%s) -- trying offline fallback", google_err)
                self.stt = self._build_faster_whisper_stt()
                self.stt_runtime_label = (
                    "Disabled"
                    if type(self.stt).__name__ == "DisabledSTT" or isinstance(self.stt, _DisabledSTT)
                    else "Faster-Whisper (offline fallback)"
                )
        elif engine_pref == "auto":
            self.stt = self._build_faster_whisper_stt()
            if isinstance(self.stt, _DisabledSTT):
                whisper_reason = getattr(self.stt, "_reason", "offline fallback unavailable")
                self.stt_runtime_fallbacks.append(f"offline unavailable: {whisper_reason}")
                google_stt, google_err = self._build_google_stt()
                if google_stt is not None:
                    self.stt = google_stt
                    self.stt_runtime_label = "Google Online (auto fallback)"
                else:
                    self.stt_runtime_fallbacks.append(f"google unavailable: {google_err}")
                    self.stt = self._build_disabled_stt(
                        f"Offline unavailable ({whisper_reason}); google unavailable ({google_err})"
                    )
                    self.stt_runtime_label = "Disabled"
            else:
                self.stt_runtime_label = "Faster-Whisper (auto)"
        elif engine_pref == "macos_native":
            self.stt = self._build_disabled_stt(
                "macOS native STT (SFSpeechRecognizer) is only available on macOS",
            )
            self.stt_runtime_label = "Disabled"
        elif engine_pref == "faster_whisper":
            self.stt = self._build_faster_whisper_stt()
            self.stt_runtime_label = (
                "Disabled"
                if isinstance(self.stt, _DisabledSTT)
                else "Faster-Whisper (explicit)"
            )
        else:
            self.stt = self._build_disabled_stt(f"Unknown STT engine: {engine_pref}")
            self.stt_runtime_label = "Disabled"

        logger.info("STT backend selected: %s", type(self.stt).__name__)
        logger.info(
            "VOICE_LAUNCH_DIAG: ATOM_LAUNCH_MODE=%s ATOM_APP_BUNDLE=%s label=%s",
            os.environ.get("ATOM_LAUNCH_MODE", ""),
            os.environ.get("ATOM_APP_BUNDLE", ""),
            self.stt_runtime_label,
        )

    def _build_tts(self) -> None:
        tts_cfg = self._config.get("tts", {})
        tts_engine = (tts_cfg.get("engine") or "macos_native").lower()

        if sys.platform == "darwin" and tts_engine not in ("macos_native", "kokoro"):
            logger.warning(
                "TTS: engine=%r ignored on macOS -- using macos_native",
                tts_cfg.get("engine"),
            )
            tts_engine = "macos_native"

        if tts_engine == "macos_native":
            from voice.tts_macos import MacOSTTSAsync
            self.tts = MacOSTTSAsync(
                self._bus, self._state,
                max_lines=tts_cfg.get("max_lines", 4),
                voice=tts_cfg.get("macos_voice", "system"),
                rate=tts_cfg.get("macos_rate", 165),
            )
            logger.info("TTS: macOS Native (voice=%s)", tts_cfg.get("macos_voice", "system"))
            self.tts_runtime_label = f"macOS Native ({tts_cfg.get('macos_voice', 'system')})"
        elif tts_engine == "kokoro":
            from voice.tts_kokoro import KokoroTTSAsync
            self.tts = KokoroTTSAsync(
                self._bus, self._state,
                max_lines=tts_cfg.get("max_lines", 4),
                voice=tts_cfg.get("kokoro_voice", "af_heart"),
            )
            logger.info("TTS: Kokoro Neural fallback (offline)")
            self.tts_runtime_label = f"Kokoro ({tts_cfg.get('kokoro_voice', 'af_heart')})"
        else:
            from voice.tts_edge import EdgeTTSAsync
            self.tts = EdgeTTSAsync(
                self._bus, self._state,
                max_lines=tts_cfg.get("max_lines", 4),
                voice=tts_cfg.get("edge_voice", "en-GB-RyanNeural"),
                rate=tts_cfg.get("edge_rate", "+15%"),
                enable_postprocess=tts_cfg.get("edge_postprocess", True),
                enable_ack_cache=tts_cfg.get("edge_ack_cache", True),
            )
            logger.info("TTS: Edge Neural fallback")
            self.tts_runtime_label = f"Edge ({tts_cfg.get('edge_voice')})"

    def build_wake_word(self) -> Any:
        """Build and preload the optional wake word engine."""
        from voice.wake_word import WakeWordEngine

        self._wake_word = WakeWordEngine(self._bus, self._state, self._config)
        loaded = self._wake_word.preload()
        if loaded:
            logger.info("WakeWordEngine ready")
        else:
            logger.info("WakeWordEngine unavailable -- always-listen mode")
        return self._wake_word

    def build_interrupt_handler(
        self,
        *,
        local_brain: Any = None,
        llm_queue: Any = None,
        indicator: Any = None,
        interrupt_manager: Any = None,
    ) -> Any:
        """Build the voice interrupt handler and return it."""
        from voice.interrupt_handler import VoiceInterruptHandler

        self._interrupt_handler = VoiceInterruptHandler(
            bus=self._bus,
            state=self._state,
            tts=self.tts,
            interrupt_manager=interrupt_manager,
            local_brain=local_brain,
            llm_queue=llm_queue,
            indicator=indicator,
        )
        return self._interrupt_handler

    @property
    def wake_word(self) -> Any:
        return self._wake_word

    @property
    def interrupt_handler(self) -> Any:
        return self._interrupt_handler

    def build_stt_watchdog(self) -> Any:
        """Build and start the STT self-healing watchdog."""
        from voice.stt_watchdog import STTWatchdog

        self._stt_watchdog = STTWatchdog(self._bus)
        if self.stt is not None:
            self._stt_watchdog.attach_stt(self.stt)
        self._bus.on("speech_partial", self._stt_watchdog.on_speech_partial)
        self._bus.on("speech_final", self._stt_watchdog.on_speech_final)
        return self._stt_watchdog

    def build_listening_mode(self) -> Any:
        """Build the dual-channel listening mode controller."""
        from voice.listening_modes import ListeningModeController

        always_active = self._wake_word is None or not getattr(self._wake_word, "is_available", False)
        self._listening_mode = ListeningModeController(always_active=always_active)
        return self._listening_mode

    @property
    def stt_watchdog(self) -> Any:
        return self._stt_watchdog

    @property
    def listening_mode(self) -> Any:
        return self._listening_mode

    @property
    def audio_intelligence(self) -> Any:
        return self._audio_intel

    def build_audio_intelligence(self) -> Any:
        """Build the Audio Intelligence Engine for device auto-selection."""
        from voice.audio_intelligence import AudioIntelligenceEngine

        self._audio_intel = AudioIntelligenceEngine(
            self._bus, self._state, self._config,
            mic_manager=self._mic_manager,
        )
        return self._audio_intel

    def configure_audio_intelligence(self) -> None:
        """Wire STT/TTS into the Audio Intelligence Engine after build()."""
        if self._audio_intel is not None:
            self._audio_intel.configure(stt=self.stt, tts=self.tts)

    async def start_voice_loop(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Start the wake word engine, watchdog, and continuous listening loop."""
        if self._wake_word is not None and self._wake_word.is_available:
            self._wake_word.start(loop)

        if self._listening_mode is None:
            self.build_listening_mode()

        if self._stt_watchdog is None:
            self.build_stt_watchdog()
        self._stt_watchdog.start()

        if self._audio_intel is not None:
            self.configure_audio_intelligence()
            try:
                self._audio_intel.wire_context()
            except Exception:
                logger.debug("Audio intelligence context wiring failed", exc_info=True)
            try:
                await self._audio_intel.start_watchdog()
            except Exception:
                logger.debug("Audio watchdog start failed", exc_info=True)

        from core.state_manager import AtomState

        if self._state.current is AtomState.IDLE:
            self._state.always_listen = self._wake_word is None or not self._wake_word.is_available
            if self._state.always_listen:
                await self._state.transition(AtomState.LISTENING)
                logger.info("Voice loop: always-listen mode (no wake word, STT never blocks)")
            else:
                logger.info("Voice loop: wake word mode (say 'Hey ATOM')")

    def shutdown(self) -> None:
        """Cleanly shut down all voice components."""
        if self._audio_intel is not None:
            try:
                self._audio_intel.shutdown()
            except Exception:
                logger.debug("AudioIntelligence shutdown error", exc_info=True)
        if self._stt_watchdog is not None:
            try:
                self._stt_watchdog.stop()
            except Exception:
                logger.debug("STT Watchdog shutdown error", exc_info=True)
        if self._wake_word is not None:
            try:
                self._wake_word.shutdown()
            except Exception:
                logger.debug("WakeWordEngine shutdown error", exc_info=True)
        if self.stt is not None:
            try:
                self.stt.shutdown()
            except Exception:
                logger.debug("STT shutdown error", exc_info=True)
        if self.tts is not None:
            stop_fn = getattr(self.tts, "shutdown", getattr(self.tts, "stop", None))
            if callable(stop_fn):
                try:
                    result = stop_fn()
                    if asyncio.iscoroutine(result):
                        try:
                            loop = asyncio.get_running_loop()
                            loop.create_task(result)
                        except RuntimeError:
                            asyncio.get_event_loop().run_until_complete(result)
                except Exception:
                    logger.debug("TTS shutdown error", exc_info=True)
        logger.info("VoicePipeline shutdown complete")
