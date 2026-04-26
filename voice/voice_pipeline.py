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
import time as _time
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
        self._earcons: Any = None
        # Optional reference to the VisionEngine so the wake-word
        # handler can fire a VLM describe pass in the background.
        # Attached post-init via :py:meth:`attach_vision_engine` (see
        # main.py), mirroring how the router receives the same engine.
        self._vision_engine: Any = None
        self._on_wake_describe_in_flight: bool = False

    def build(self) -> None:
        """Construct STT and TTS engines based on config + platform."""
        self._build_tts()
        self._build_stt()
        self._wire_echo_guard()
        logger.info(
            "VoicePipeline built: stt=%s tts=%s",
            self.stt_runtime_label,
            self.tts_runtime_label,
        )

    def attach_vision_engine(self, vision_engine: Any) -> None:
        """Wire the VisionEngine so ``describe_on_wake`` can fire.

        Safe to call multiple times or with ``None``. Must be called
        AFTER :py:meth:`build_listening_mode_controller` has registered
        its handlers; the wake handler dereferences ``self._vision_engine``
        lazily at call time, so the order actually doesn't matter.
        """
        self._vision_engine = vision_engine

    def _describe_on_wake_enabled(self) -> bool:
        """Cheap check: is every prerequisite in place to fire the
        describe-on-wake path right now?"""
        vision_cfg = (self._config.get("vision") or {})
        if not vision_cfg.get("enabled", False):
            return False
        if not vision_cfg.get("describe_on_wake", False):
            return False
        engine = self._vision_engine
        if engine is None:
            return False
        if not getattr(engine, "captioner_available", False):
            return False
        return True

    # How long a freshly captured caption is considered "still good
    # enough" to skip the on-wake describe entirely. Shorter than
    # ``vision.caption_max_age_s`` (used by the router for context
    # injection) on purpose -- we want fast back-and-forth wakes to
    # reuse a recent caption (saves CPU + 1.5s latency), but a 15s
    # window also lets the scene update if the user has moved away
    # from the desk between turns.
    _ON_WAKE_DEDUPE_MAX_AGE_S: float = 15.0

    def _schedule_describe_on_wake(self, *, trigger: str) -> None:
        """Fire ``engine.look(describe=True)`` in the default executor.

        Safeguards (in order of evaluation):

        * **Single-flight per pipeline instance.** If a previous
          describe pass is still running we skip -- AVCapture only
          cooperates with one session at a time, and the camera lock
          inside the engine would just block us anyway.
        * **Speaking-state guard.** If TTS is currently playing we
          skip; the VLM eats GIL + neural-engine cycles and would
          stretch perceived TTS latency. The next user turn will get a
          fresh describe via the router-side caption injection.
        * **Recent-caption dedupe.** If the engine has a caption
          younger than ``_ON_WAKE_DEDUPE_MAX_AGE_S`` we skip -- a 5-15s
          old caption is still good enough for the next LLM turn and
          firing a fresh describe just heats the M5 Air for nothing.
        * **Engine disabled-check** is re-run inside the executor so a
          late config flip doesn't race the coroutine.
        * **Exceptions are swallowed**; the camera path already
          audit-logs failures internally.
        """
        if self._on_wake_describe_in_flight:
            logger.debug(
                "describe_on_wake skipped; prior pass still in flight",
            )
            return
        engine = self._vision_engine
        if engine is None:
            return

        # Don't compete with TTS for CPU/Neural-Engine. SPEAKING is the
        # only state where the M5 Air feels the VLM as a TTS hiccup --
        # IDLE / LISTENING / THINKING are all fine because the user
        # already accepts a 100-300ms gap before ATOM responds.
        if self._is_speaking_now():
            logger.debug(
                "describe_on_wake skipped; ATOM is currently speaking",
            )
            return

        # Re-use a fresh caption rather than burning another inference.
        # Most rapid wakes (user calls ATOM twice in a row) hit this.
        try:
            recent_caption_fn = getattr(engine, "recent_caption", None)
            if callable(recent_caption_fn):
                fresh = recent_caption_fn(
                    max_age_s=self._ON_WAKE_DEDUPE_MAX_AGE_S,
                )
                if fresh:
                    logger.debug(
                        "describe_on_wake skipped; reusing fresh "
                        "caption (%d chars, age <= %.0fs)",
                        len(fresh),
                        self._ON_WAKE_DEDUPE_MAX_AGE_S,
                    )
                    return
        except Exception:
            logger.debug(
                "describe_on_wake dedupe-check raised", exc_info=True,
            )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug(
                "describe_on_wake: no running loop; skipping",
            )
            return

        self._on_wake_describe_in_flight = True

        def _blocking_describe() -> None:
            try:
                engine.look(
                    reason=f"on_wake:{trigger[:40]}",
                    detect_faces=True,
                    detect_barcodes=False,
                    describe=True,
                )
            except Exception:
                logger.debug(
                    "describe_on_wake worker raised", exc_info=True,
                )

        future = loop.run_in_executor(None, _blocking_describe)

        def _clear(_fut: asyncio.Future) -> None:
            self._on_wake_describe_in_flight = False

        future.add_done_callback(_clear)

    def _is_speaking_now(self) -> bool:
        """Return True if StateManager reports ATOM is currently
        speaking (TTS in flight). Defensive against any unexpected
        StateManager API shape -- returns False on any error so the
        VLM can still fire if we can't read the state.
        """
        state = self._state
        if state is None:
            return False
        try:
            from core.state_manager import AtomState  # local import: avoid cycles
        except Exception:
            return False
        try:
            current = getattr(state, "current_state", None) or getattr(state, "current", None)
            if current is None:
                return False
            return bool(current == AtomState.SPEAKING)
        except Exception:
            return False

    def _wire_echo_guard(self) -> None:
        """Connect TTS.is_echo() to STT so stable partials that match
        ATOM's own voice cannot be promoted to finals (Jarvis-loop fix).
        """
        try:
            stt = self.stt
            tts = self.tts
            if stt is None or tts is None:
                return
            attach = getattr(stt, "attach_echo_guard", None)
            is_echo = getattr(tts, "is_echo", None)
            if callable(attach) and callable(is_echo):
                attach(lambda text: bool(is_echo(text, window_s=30.0)))
                logger.info("Voice pipeline: echo guard wired (TTS.is_echo -> STT finalization)")
        except Exception:
            logger.debug("echo guard wiring failed", exc_info=True)

    def _voice_activation_mode(self) -> str:
        """Return the configured voice activation mode.

        ``always_on`` keeps STT in the ACTIVE command path all the time.
        ``wake_word`` preserves the dual-channel passive/active flow.
        ``jarvis`` is accepted as a friendly alias for ``always_on``.
        """
        voice_cfg = self._config.get("voice", {}) or {}
        raw = str(voice_cfg.get("activation_mode", "") or "").strip().lower()
        if raw in {"always_on", "always-on", "alwayson", "jarvis"}:
            return "always_on"
        return "wake_word"

    def _wake_word_requested(self) -> bool:
        wake_cfg = self._config.get("wake_word", {}) or {}
        return bool(wake_cfg.get("enabled", True))

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

    def _build_whisperkit_stt(self) -> tuple[Any | None, str]:
        """Construct the WhisperKit (CoreML / ANE) backend (Sprint P3.3).

        Returns ``(stt, "")`` on success and ``(None, reason)`` if any
        dependency is missing. The factory falls back to whisper.cpp
        when ``whisperkit-cli`` is not on $PATH.
        """
        try:
            from voice.stt_whisperkit import (
                WhisperKitSTT,
                is_whisperkit_available,
            )
        except Exception as exc:
            return None, (
                f"WhisperKit module import failed: {exc}; "
                "did you `pip install sounddevice webrtcvad`?"
            )
        if not is_whisperkit_available(self._config):
            return None, (
                "WhisperKit unavailable: install with "
                "`brew install whisperkit-cli`. Falling back to whisper.cpp."
            )
        stt = WhisperKitSTT(
            self._bus,
            self._state,
            self._config,
            mic_manager=self._mic_manager,
            intent_engine=self._intent_engine,
        )
        return stt, ""

    def _build_whisper_cpp_stt(self) -> tuple[Any | None, str]:
        """Construct the whisper.cpp Metal backend (Sprint B3).

        Returns ``(stt, "")`` on success and ``(None, reason)`` if any
        dependency is missing. Sprint K blocks the first boot to install
        the GGML model instead of silently falling back to Apple's
        unstable SFSpeechRecognizer.
        """
        try:
            from voice.stt_whisper import WhisperSTT, is_whisper_available
        except Exception as exc:
            return None, (
                f"whisper.cpp module import failed: {exc}; "
                "did you `pip install pywhispercpp sounddevice webrtcvad`?"
            )
        if not is_whisper_available(self._config):
            from voice.stt_whisper import _resolve_model_path  # noqa: PLC0415
            model_path = _resolve_model_path(
                (self._config.get("stt") or {}).get("whisper_model_path"),
            )
            if not model_path.exists():
                try:
                    from voice.whisper_install import ensure_model

                    logger.warning(
                        "whisper.cpp model missing at %s -- blocking boot "
                        "to install it now",
                        model_path,
                    )
                    try:
                        self._bus.emit_fast(
                            "tts_say",
                            text=(
                                "Setting up speech recognition, Boss. "
                                "This is a one-time download."
                            ),
                            source="whisper_install",
                        )
                    except Exception:
                        logger.debug("whisper install TTS cue failed", exc_info=True)

                    ensure_model(
                        model_path=model_path,
                        progress_cb=lambda msg: logger.info(
                            "Whisper install: %s", msg,
                        ),
                    )
                except Exception as exc:
                    return None, (
                        f"Whisper model install failed for {model_path}: {exc}"
                    )
                if not is_whisper_available(self._config):
                    return None, (
                        f"Whisper model installed at {model_path}, but "
                        "whisper.cpp is still unavailable; check pywhispercpp "
                        "/ sounddevice / webrtcvad dependencies."
                    )
            if not is_whisper_available(self._config):
                return None, (
                    "whisper.cpp deps missing -- "
                    "`pip install pywhispercpp sounddevice webrtcvad`"
                )
        stt = WhisperSTT(
            self._bus,
            self._state,
            self._config,
            mic_manager=self._mic_manager,
            intent_engine=self._intent_engine,
        )
        return stt, ""

    def _load_persisted_failover_reason(self) -> str:
        """Read data/atom_runtime.json, return the stored failover reason
        (empty string if none). Used to honor a prior session's decision
        to stay on Whisper instead of re-provoking native STT."""
        try:
            import json as _json
            import pathlib as _pl
            path = _pl.Path("data/atom_runtime.json")
            if not path.exists():
                return ""
            data = _json.loads(path.read_text() or "{}")
            return str(data.get("stt_failover_reason") or "").strip()
        except Exception:
            logger.debug("failover flag load failed", exc_info=True)
            return ""

    _WHISPER_CPP_ALIASES = ("whisper_cpp", "whispercpp", "whisper", "whisper.cpp")
    # Sprint P3.3 (Apr 26 2026): WhisperKit (CoreML / ANE) is the highest-
    # ROI engine on Apple Silicon. We accept several spellings so config
    # files and docs can be casual.
    _WHISPERKIT_ALIASES = ("whisperkit", "whisper_kit", "whisper-kit", "wk")

    def _build_stt(self) -> None:
        stt_cfg = self._config.get("stt", {})
        engine_pref = str(
            stt_cfg.get("engine", "macos_native") or "macos_native",
        ).strip().lower()
        # Sprint B3: alias collapse so "whisper", "whispercpp", "whisper.cpp"
        # all map to the canonical whisper_cpp branch.
        if engine_pref in self._WHISPER_CPP_ALIASES:
            engine_pref = "whisper_cpp"
        if engine_pref in self._WHISPERKIT_ALIASES:
            engine_pref = "whisperkit"
        logger.info(
            "STT engine preference: %s (platform=%s)",
            engine_pref, sys.platform,
        )

        # If a previous session failed over to Whisper due to unrecoverable
        # native breakage, honor that decision on this boot — unless the
        # user has explicitly set stt.engine = macos_native (which implies
        # "I want to try native again"). The "auto" preference respects
        # the persisted flag.
        persisted_reason = self._load_persisted_failover_reason()
        if (persisted_reason
                and engine_pref in ("auto",)
                and sys.platform == "darwin"):
            logger.warning(
                "STT: persisted failover flag present (%s) — starting with Whisper",
                persisted_reason[:80],
            )
            self.stt = self._build_faster_whisper_stt()
            if isinstance(self.stt, _DisabledSTT):
                self.stt_runtime_label = "Disabled (whisper unavailable)"
            else:
                self.stt_runtime_label = "Faster-Whisper (persisted failover)"
                self._stt_failed_over = True
            return

        if sys.platform == "darwin":
            if engine_pref == "whisperkit":
                # Sprint P3.3 (Apr 26 2026): explicit WhisperKit. If the
                # CLI is missing we still gracefully fall back to
                # whisper.cpp -- ANE > Metal > nothing.
                wk_stt, wk_reason = self._build_whisperkit_stt()
                if wk_stt is not None:
                    self.stt = wk_stt
                    self.stt_runtime_label = (
                        "WhisperKit CoreML/ANE (explicit)"
                    )
                    logger.info(
                        "STT: WhisperKit (CoreML on Apple Neural Engine)",
                    )
                else:
                    self.stt_runtime_fallbacks.append(
                        f"whisperkit unavailable: {wk_reason}",
                    )
                    logger.warning(
                        "STT whisperkit: %s -- falling back to whisper.cpp",
                        wk_reason,
                    )
                    whisper_stt, whisper_reason = (
                        self._build_whisper_cpp_stt()
                    )
                    if whisper_stt is not None:
                        self.stt = whisper_stt
                        self.stt_runtime_label = (
                            "whisper.cpp Metal (whisperkit fallback)"
                        )
                    else:
                        self.stt_runtime_error = whisper_reason or ""
                        self.stt = self._build_disabled_stt(
                            f"whisperkit + whisper.cpp both unavailable",
                        )
                        self.stt_runtime_label = "Disabled"
            elif engine_pref == "whisper_cpp":
                whisper_stt, whisper_reason = self._build_whisper_cpp_stt()
                if whisper_stt is not None:
                    self.stt = whisper_stt
                    self.stt_runtime_label = "whisper.cpp Metal (small.en-q5_1)"
                    logger.info(
                        "STT: whisper.cpp Metal (long-session reliable, "
                        "no idle timeout)",
                    )
                else:
                    self.stt_runtime_error = whisper_reason or ""
                    self.stt_runtime_fallbacks.append(
                        f"whisper.cpp unavailable: {whisper_reason}",
                    )
                    logger.error(
                        "whisper.cpp STT unavailable (%s) -- refusing "
                        "SFSpeech fallback after Sprint K",
                        whisper_reason,
                    )
                    self.stt = self._build_disabled_stt(
                        f"whisper.cpp unavailable ({whisper_reason})",
                    )
                    self.stt_runtime_label = "Disabled (whisper.cpp required)"
            elif engine_pref == "auto":
                # Sprint P3.3 (Apr 26 2026): prefer WhisperKit if the CLI
                # is present (ANE > Metal). Else fall back to whisper.cpp,
                # which is still better than SFSpeechRecognizer for long
                # sessions (atomLogs.txt L310/437).
                wk_stt, wk_reason = self._build_whisperkit_stt()
                if wk_stt is not None:
                    self.stt = wk_stt
                    self.stt_runtime_label = "WhisperKit CoreML/ANE (auto)"
                    logger.info(
                        "STT auto: WhisperKit selected -- ANE on hot path",
                    )
                else:
                    self.stt_runtime_fallbacks.append(
                        f"whisperkit unavailable: {wk_reason}",
                    )
                    whisper_stt, whisper_reason = (
                        self._build_whisper_cpp_stt()
                    )
                    if whisper_stt is not None:
                        self.stt = whisper_stt
                        self.stt_runtime_label = "whisper.cpp Metal (auto)"
                        logger.info(
                            "STT auto: whisper.cpp selected -- "
                            "model present (whisperkit unavailable)",
                        )
                    else:
                        self.stt_runtime_fallbacks.append(
                            f"whisper.cpp unavailable: {whisper_reason}",
                        )
                        self.stt_runtime_error = whisper_reason or ""
                        logger.error(
                            "STT auto: whisper.cpp unavailable (%s) -- "
                            "not falling back to SFSpeechRecognizer "
                            "after Sprint K",
                            whisper_reason,
                        )
                        self.stt = self._build_disabled_stt(
                            f"whisper.cpp unavailable ({whisper_reason})",
                        )
                        self.stt_runtime_label = (
                            "Disabled (whisper.cpp required)"
                        )
            elif engine_pref == "macos_native":
                native_stt, native_reason = self._build_native_stt()
                if native_stt is not None:
                    self.stt = native_stt
                    self.stt_runtime_label = "macOS Native (SFSpeechRecognizer)"
                    logger.info("STT: macOS Native -- Apple stack only")
                else:
                    self.stt_runtime_error = native_reason or ""
                    self.stt_runtime_fallbacks.append(
                        f"native unavailable: {native_reason}",
                    )
                    logger.warning(
                        "Native STT unavailable (%s) -- "
                        "trying Faster-Whisper fallback", native_reason,
                    )
                    self.stt = self._build_faster_whisper_stt()
                    if isinstance(self.stt, _DisabledSTT):
                        self.stt_runtime_label = "Disabled"
                    else:
                        self.stt_runtime_label = (
                            "Faster-Whisper (native fallback)"
                        )
                        logger.info(
                            "STT: Faster-Whisper (native STT unavailable)",
                        )
            elif engine_pref == "faster_whisper":
                self.stt = self._build_faster_whisper_stt()
                self.stt_runtime_label = (
                    "Disabled"
                    if isinstance(self.stt, _DisabledSTT)
                    else "Faster-Whisper (explicit)"
                )
            elif engine_pref in ("google_online", "google"):
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
        elif engine_pref == "whisper_cpp":
            whisper_stt, whisper_reason = self._build_whisper_cpp_stt()
            if whisper_stt is not None:
                self.stt = whisper_stt
                self.stt_runtime_label = "whisper.cpp (Metal/CPU)"
            else:
                self.stt_runtime_error = whisper_reason or ""
                self.stt_runtime_fallbacks.append(
                    f"whisper.cpp unavailable: {whisper_reason}",
                )
                self.stt = self._build_disabled_stt(
                    f"whisper.cpp unavailable ({whisper_reason})",
                )
                self.stt_runtime_label = "Disabled"
        elif engine_pref == "auto":
            whisper_stt, whisper_reason = self._build_whisper_cpp_stt()
            if whisper_stt is not None:
                self.stt = whisper_stt
                self.stt_runtime_label = "whisper.cpp (auto)"
                logger.info(
                    "STT auto: whisper.cpp selected -- model present",
                )
                logger.info("STT backend selected: %s", type(self.stt).__name__)
                logger.info(
                    "VOICE_LAUNCH_DIAG: ATOM_LAUNCH_MODE=%s "
                    "ATOM_APP_BUNDLE=%s label=%s",
                    os.environ.get("ATOM_LAUNCH_MODE", ""),
                    os.environ.get("ATOM_APP_BUNDLE", ""),
                    self.stt_runtime_label,
                )
                return
            self.stt_runtime_fallbacks.append(
                f"whisper.cpp unavailable: {whisper_reason}",
            )
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
                first_word_warmup_ms=int(tts_cfg.get("macos_first_word_warmup_ms", 140)),
                tail_drain_ms=int(tts_cfg.get("macos_tail_drain_ms", 120)),
                tail_drain_bluetooth_ms=int(tts_cfg.get("macos_tail_drain_bluetooth_ms", 200)),
                warmup_skip_window_ms=int(tts_cfg.get("macos_warmup_skip_window_ms", 800)),
            )
            logger.info("TTS: macOS Native (voice=%s)", tts_cfg.get("macos_voice", "system"))
            self.tts_runtime_label = f"macOS Native ({tts_cfg.get('macos_voice', 'system')})"
        elif tts_engine == "kokoro":
            from voice.tts_kokoro import KokoroTTSAsync
            self.tts = KokoroTTSAsync(
                self._bus, self._state,
                max_lines=tts_cfg.get("max_lines", 4),
                voice=tts_cfg.get("kokoro_voice", "af_heart"),
                model_path=tts_cfg.get("kokoro_model_path"),
                voices_path=tts_cfg.get("kokoro_voices_path"),
                speed=float(tts_cfg.get("kokoro_speed", 1.0)),
                language=tts_cfg.get("kokoro_language", "en-us"),
            )
            # Sprint Ω2: graceful auto-fallback so a misconfigured
            # Kokoro install (missing model files, missing espeak-ng)
            # never silences ATOM at boot.
            if not getattr(self.tts, "_available", False):
                logger.warning(
                    "TTS: Kokoro requested but unavailable -- "
                    "falling back to macOS Native (Daniel)",
                )
                from voice.tts_macos import MacOSTTSAsync
                self.tts = MacOSTTSAsync(
                    self._bus, self._state,
                    max_lines=tts_cfg.get("max_lines", 4),
                    voice=tts_cfg.get("macos_voice", "system"),
                    rate=tts_cfg.get("macos_rate", 165),
                    first_word_warmup_ms=int(tts_cfg.get("macos_first_word_warmup_ms", 140)),
                    tail_drain_ms=int(tts_cfg.get("macos_tail_drain_ms", 120)),
                    tail_drain_bluetooth_ms=int(tts_cfg.get("macos_tail_drain_bluetooth_ms", 200)),
                    warmup_skip_window_ms=int(tts_cfg.get("macos_warmup_skip_window_ms", 800)),
                )
                self.tts_runtime_label = (
                    f"macOS Native ({tts_cfg.get('macos_voice', 'system')})"
                    " [kokoro-unavailable-fallback]"
                )
            else:
                logger.info(
                    "TTS: Kokoro Neural offline (voice=%s)",
                    tts_cfg.get("kokoro_voice", "af_heart"),
                )
                self.tts_runtime_label = (
                    f"Kokoro ({tts_cfg.get('kokoro_voice', 'af_heart')})"
                )
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
        """Build and preload the optional wake word engine.

        Sprint Ω.6.B (Apr 26 2026): Surface the
        ``activation_mode=always_on`` + ``wake_word.enabled=true``
        contradiction at WARNING level so operators stop wondering why
        ``Hey ATOM`` does nothing — always_on routes STT continuously and
        intentionally bypasses OpenWakeWord. The fix is one of:
          (a) set ``wake_word.enabled=false`` in ``config/settings.json``
              (recommended for the always-on Jarvis default), or
          (b) set ``voice.activation_mode="wake_word"`` to actually use
              the wake-word gating.
        """
        self._wake_word = None
        if self._voice_activation_mode() == "always_on":
            if self._wake_word_requested():
                logger.warning(
                    "Wake-word config contradiction: voice.activation_mode="
                    "'always_on' supersedes wake_word.enabled=true (the wake "
                    "word engine will NOT run). Set wake_word.enabled=false "
                    "in config/settings.json to silence this warning, or "
                    "switch voice.activation_mode to 'wake_word' to actually "
                    "use the wake gate.",
                )
            else:
                logger.info("WakeWordEngine bypassed: voice.activation_mode=always_on")
            return None
        if not self._wake_word_requested():
            logger.info("WakeWordEngine disabled in config")
            return None

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
        # Hardened NativeSTT emits this event when it detects empty-isFinal
        # cascades or recreate storms it cannot recover from locally.
        self._bus.on("stt_needs_full_restart", self._stt_watchdog.on_needs_full_restart)
        # Watchdog asks us to swap the engine after repeated failovers.
        self._bus.on("stt_swap_to_whisper", self._on_swap_to_whisper_event)
        return self._stt_watchdog

    async def _on_swap_to_whisper_event(self, reason: str = "", **_kw: Any) -> None:
        """Bus handler: watchdog asked us to swap to Whisper."""
        try:
            self.swap_to_whisper(reason=reason or "watchdog_request")
        except Exception:
            logger.exception("STT failover: swap_to_whisper failed")

    def swap_to_whisper(self, *, reason: str = "failover") -> bool:
        """Swap the primary STT from NativeSTT (SFSpeechRecognizer) to
        faster-whisper (STTAsync) at runtime, after the native engine has
        proven unrecoverable in this process.

        Returns True if the swap succeeded (Whisper now active), False if
        Whisper is unavailable (dependencies missing etc.).
        """
        # Idempotent: if we've already swapped, don't try again.
        if getattr(self, "_stt_failed_over", False):
            return True
        if self.stt is not None:
            current_name = type(self.stt).__name__
            if current_name in ("STTAsync", "_DisabledSTT"):
                return False

        logger.warning(
            "STT failover: swapping %s -> faster-whisper (%s)",
            type(self.stt).__name__ if self.stt is not None else "None",
            reason,
        )

        old = self.stt
        if old is not None:
            try:
                old.shutdown()
            except Exception:
                logger.debug("STT failover: old stt shutdown failed", exc_info=True)

        new_stt = self._build_faster_whisper_stt()
        if isinstance(new_stt, _DisabledSTT):
            logger.error(
                "STT failover: Whisper unavailable — '%s'. Voice input will stay in disabled state.",
                getattr(new_stt, "_reason", "unknown"),
            )
            self.stt = new_stt
            self.stt_runtime_label = "Disabled (whisper unavailable)"
            self._stt_failed_over = True
            self._persist_failover_flag(reason=f"whisper_unavailable:{reason}")
            return False

        self.stt = new_stt
        self.stt_runtime_label = "Faster-Whisper (runtime failover)"
        self._stt_failed_over = True

        # Re-wire the watchdog to the new engine.
        if self._stt_watchdog is not None:
            try:
                self._stt_watchdog.attach_stt(new_stt)
            except Exception:
                logger.debug("STT failover: watchdog rewire failed", exc_info=True)

        # Re-wire state-change handler so the new engine participates in
        # LISTENING/SPEAKING transitions.
        on_state = getattr(new_stt, "on_state_changed", None)
        if callable(on_state):
            try:
                self._bus.on("state_changed", on_state)
            except Exception:
                logger.debug("STT failover: state_changed rewire failed", exc_info=True)

        # Kick the new engine into life.
        try:
            import asyncio as _aio
            loop = _aio.get_running_loop()
            loop.create_task(new_stt.async_start_listening())
        except RuntimeError:
            logger.debug("STT failover: no running loop to start new engine")
        except Exception:
            logger.debug("STT failover: new engine start failed", exc_info=True)

        self._persist_failover_flag(reason=reason)
        logger.info(
            "STT failover complete: voice input now via faster-whisper (reason=%s)",
            reason,
        )
        try:
            self._bus.emit_fast("stt_failover_complete", reason=reason)
        except Exception:
            logger.debug("STT failover: emit_fast failed", exc_info=True)
        return True

    def _persist_failover_flag(self, *, reason: str) -> None:
        """Persist a one-shot flag so ATOM boots with Whisper next time if
        native was unrecoverable in this session."""
        try:
            import json as _json
            import pathlib as _pl
            path = _pl.Path("data/atom_runtime.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            existing: dict = {}
            if path.exists():
                try:
                    existing = _json.loads(path.read_text() or "{}")
                except Exception:
                    existing = {}
            existing["stt_failover_reason"] = reason
            try:
                import datetime as _dt
                existing["stt_failover_at"] = _dt.datetime.utcnow().isoformat() + "Z"
            except Exception:
                pass
            path.write_text(_json.dumps(existing, indent=2))
        except Exception:
            logger.debug("STT failover: persist flag failed", exc_info=True)

    def build_listening_mode(self) -> Any:
        """Build the dual-channel listening mode controller and wire the
        mode-flip handlers.

        Flow:
            always_on                -> ACTIVE permanently
            wake_word_detected       -> ACTIVE (route speech_final to Router)
            idle after tts_done      -> PASSIVE (ignore speech unless wake)
        """
        from voice.listening_modes import ListeningModeController

        always_active = (
            self._voice_activation_mode() == "always_on"
            or self._wake_word is None
            or not getattr(self._wake_word, "is_available", False)
        )
        self._listening_mode = ListeningModeController(always_active=always_active)

        self._passive_revert_task: asyncio.Task | None = None
        # Extended to 30s so short pauses / half-second breaths during a
        # follow-up correction don't flip us back to PASSIVE mid-dialog.
        self._passive_revert_delay_s: float = float(
            self._config.get("stt", {}).get("passive_revert_delay_s", 30.0),
        )
        # Minimum gap between partial-triggered kicks so we don't thrash
        # the scheduler on every 50-ms partial.
        self._passive_revert_kick_min_s: float = 1.0
        self._passive_revert_last_kick_t: float = 0.0

        def _cancel_revert_task() -> None:
            task = self._passive_revert_task
            if task is not None and not task.done():
                task.cancel()
                self._passive_revert_task = None

        def _arm_revert_task(delay_s: float) -> None:
            """Always cancel + re-arm. Safe to call from any event."""
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            _cancel_revert_task()
            if self._listening_mode is None:
                return
            if getattr(self._listening_mode, "_always_active", False):
                return
            if getattr(self._listening_mode, "is_passive", False):
                return
            self._passive_revert_task = loop.create_task(
                self._schedule_passive_revert(delay_s),
            )

        self._arm_passive_revert = _arm_revert_task  # type: ignore[attr-defined]
        self._cancel_passive_revert = _cancel_revert_task  # type: ignore[attr-defined]

        async def _on_wake_word(**kw: Any) -> None:
            try:
                word = str(kw.get("wake_word") or kw.get("word") or "wake")
                self._listening_mode.activate(f"wake_phrase:{word}")
                _cancel_revert_task()
            except Exception:
                logger.debug("wake_word_detected handler failed", exc_info=True)
            # Fire-and-forget a VLM describe pass so the NEXT cognitive
            # turn inherits a ``visual_context`` entry. The camera
            # warm-up + VLM inference can take 1-3s on first call; we
            # run it in the default executor so wake-word handling
            # itself stays instantaneous.
            if self._describe_on_wake_enabled():
                self._schedule_describe_on_wake(trigger=f"wake:{word}")

        async def _on_describe_request(**kw: Any) -> None:
            # Same path as wake-on-describe but driven by an explicit
            # request (e.g. a button press, a shortcut, or an STT
            # keyphrase bypass). Kept separate so the wake handler
            # doesn't have to guard both sources.
            reason = str(kw.get("reason") or "manual")
            if not self._describe_on_wake_enabled():
                logger.debug(
                    "describe request ignored; prerequisites not met",
                )
                return
            self._schedule_describe_on_wake(trigger=f"manual:{reason}")

        async def _on_tts_complete(**_kw: Any) -> None:
            # Whenever we finish speaking, the user is still in an active
            # exchange — give them the full 30s window to respond before
            # we quietly slip back to PASSIVE.
            _arm_revert_task(self._passive_revert_delay_s)

        async def _on_partial_or_final(**_kw: Any) -> None:
            # Any detected speech resets the revert timer so a user who is
            # mid-sentence / mid-correction never falls off a cliff.
            now = _time.time()
            if (now - self._passive_revert_last_kick_t) < self._passive_revert_kick_min_s:
                return
            self._passive_revert_last_kick_t = now
            _arm_revert_task(self._passive_revert_delay_s)

        # Also hook tts_delivery_metrics because some backends emit that
        # *before* tts_complete and sometimes skip tts_complete entirely
        # on cancellation.
        async def _on_tts_metrics(**_kw: Any) -> None:
            _arm_revert_task(self._passive_revert_delay_s)

        async def _on_tts_start(**_kw: Any) -> None:
            # Cancel any pending passive revert while ATOM is actively
            # speaking — without this the 30s timer can fire mid-TTS, flip
            # us to PASSIVE, and silently swallow the user's follow-up.
            _cancel_revert_task()

        async def _on_user_query(**_kw: Any) -> None:
            # A cursor_query event means the user's speech was captured and
            # forwarded for a reply — keep us ACTIVE until TTS completes.
            _cancel_revert_task()

        async def _on_wake_hint(**kw: Any) -> None:
            # STT noticed the user is clearly trying to address ATOM (3+
            # suppressed finals in PASSIVE mode within 60s) but no wake
            # phrase slipped through. Speak a short, warm prompt so the
            # user knows how to actually call us. Throttled upstream
            # inside stt_macos._note_passive_suppression so we only get
            # one of these per minute.
            sample = str(kw.get("sample_text") or "")[:60]
            logger.info(
                "Wake-hint fired — sample='%s' (suppressed_count=%s)",
                sample, kw.get("suppressed_count"),
            )
            try:
                self._bus.emit_long(
                    "partial_response",
                    text=(
                        "Boss, if you're calling me, say Atom or Hey Atom. "
                        "I'll listen."
                    ),
                    is_first=True,
                    is_last=True,
                )
            except Exception:
                logger.debug("wake_hint TTS emit failed", exc_info=True)

        self._bus.on("wake_word_detected", _on_wake_word)
        self._bus.on("vision.describe.request", _on_describe_request)
        self._bus.on("tts_complete", _on_tts_complete)
        self._bus.on("speech_partial", _on_partial_or_final)
        self._bus.on("speech_final", _on_partial_or_final)
        self._bus.on("tts_delivery_metrics", _on_tts_metrics)
        self._bus.on("tts_start", _on_tts_start)
        self._bus.on("tts_stream", _on_tts_start)
        self._bus.on("cursor_query", _on_user_query)
        self._bus.on("partial_response", _on_user_query)
        self._bus.on("wake_hint_needed", _on_wake_hint)

        # Give the native STT a reference so it can gate speech_final
        # emission while in PASSIVE mode (no wake phrase -> ignore).
        try:
            if self.stt is not None:
                set_mode = getattr(self.stt, "attach_listening_mode", None)
                if callable(set_mode):
                    set_mode(self._listening_mode)
                else:
                    setattr(self.stt, "_listening_mode_ref", self._listening_mode)
        except Exception:
            logger.debug("stt listening mode wire failed", exc_info=True)

        return self._listening_mode

    async def _schedule_passive_revert(self, delay_s: float) -> None:
        try:
            await asyncio.sleep(max(1.0, float(delay_s)))
            if self._listening_mode is not None:
                self._listening_mode.deactivate("idle_timeout")
        except asyncio.CancelledError:
            return
        except Exception:
            logger.debug("passive revert task error", exc_info=True)

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

    def build_earcons(self) -> Any:
        """Build the (optional) Earcons engine and wire lifecycle cues.

        Driven by config:
          voice.earcons.enabled              bool  (default True)
          voice.earcons.volume               float (0..1, default 0.45)
          voice.earcons.heartbeat_enabled    bool  (default False)
          voice.earcons.heartbeat_interval_s float (default 600)
        """
        from voice.earcons import Earcons

        vcfg = (self._config.get("voice") or {}).get("earcons", {}) or {}
        self._earcons = Earcons(
            enabled=bool(vcfg.get("enabled", True)),
            volume=float(vcfg.get("volume", 0.45)),
            heartbeat_enabled=bool(vcfg.get("heartbeat_enabled", False)),
            heartbeat_interval_s=float(vcfg.get("heartbeat_interval_s", 600.0)),
        )
        if not self._earcons.is_enabled:
            return self._earcons

        async def _on_wake(**_kw: Any) -> None:
            try:
                self._earcons.play("wake", min_interval_s=0.4)
            except Exception:
                logger.debug("earcons wake play failed", exc_info=True)

        async def _on_done(**_kw: Any) -> None:
            try:
                self._earcons.play("done", min_interval_s=0.4)
            except Exception:
                logger.debug("earcons done play failed", exc_info=True)

        async def _on_failover(**_kw: Any) -> None:
            try:
                self._earcons.play("error", min_interval_s=1.0)
            except Exception:
                logger.debug("earcons error play failed", exc_info=True)

        async def _on_thinking(**_kw: Any) -> None:
            # Soft click played ~1.2s after the brain accepts a query but
            # before the first TTS sentence is ready. Eliminates the dead
            # silence window without forcing a verbose "let me check" ack.
            try:
                self._earcons.play("thinking", min_interval_s=2.0)
            except Exception:
                logger.debug("earcons thinking play failed", exc_info=True)

        self._bus.on("wake_word_detected", _on_wake)
        self._bus.on("tts_complete", _on_done)
        self._bus.on("stt_failover_complete", _on_failover)
        self._bus.on("stt_watchdog_restart", _on_failover)
        self._bus.on("thinking_earcon", _on_thinking)

        if self._earcons._heartbeat_enabled:
            try:
                self._earcons.start_heartbeat(
                    lambda: (self.stt is not None
                             and getattr(self.stt, "_listening", False)),
                )
            except Exception:
                logger.debug("earcons heartbeat start failed", exc_info=True)

        return self._earcons

    def configure_audio_intelligence(self) -> None:
        """Wire STT/TTS into the Audio Intelligence Engine after build()."""
        if self._audio_intel is not None:
            self._audio_intel.configure(stt=self.stt, tts=self.tts)

    async def start_voice_loop(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Start the wake word engine, watchdog, and continuous listening loop."""
        import time as _time
        _vl_t0 = _time.perf_counter()

        if (
            self._voice_activation_mode() != "always_on"
            and self._wake_word is not None
            and self._wake_word.is_available
        ):
            self._wake_word.start(loop)
            self._wire_wake_word_gate()

        if self._listening_mode is None:
            self.build_listening_mode()

        if self._stt_watchdog is None:
            self.build_stt_watchdog()
        self._stt_watchdog.start()

        if self._earcons is None:
            try:
                self.build_earcons()
            except Exception:
                logger.debug("Earcons build failed", exc_info=True)

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
            activation_mode = self._voice_activation_mode()
            wake_available = self._wake_word is not None and self._wake_word.is_available
            self._state.always_listen = activation_mode == "always_on" or not wake_available
            if self._state.always_listen:
                await self._state.transition(AtomState.LISTENING)
                if activation_mode == "always_on":
                    logger.info(
                        "Voice loop: Jarvis always-on mode "
                        "(continuous STT routing, duplex-ready)",
                    )
                else:
                    logger.info(
                        "Voice loop: always-listen fallback "
                        "(wake word unavailable, STT never blocks)",
                    )
            else:
                logger.info("Voice loop: wake word mode (say 'Hey ATOM')")

        _vl_elapsed = (_time.perf_counter() - _vl_t0) * 1000
        logger.info(
            "VOICE_LOOP_READY: %.0fms | stt=%s | tts=%s | wake_word=%s",
            _vl_elapsed,
            self.stt_runtime_label,
            self.tts_runtime_label,
            "enabled" if (self._wake_word and self._wake_word.is_available) else "disabled",
        )

    def _wire_wake_word_gate(self) -> None:
        """Pause wake word audio capture while STT has a mic stream open.

        Both use sounddevice; running two streams to the same device
        simultaneously causes PortAudio conflicts / segfaults on macOS.
        STT already does wake-phrase detection on partials via WakeWordFilter,
        so the neural wake word engine only needs to run when STT is idle.
        """
        from core.state_manager import AtomState

        ww = self._wake_word

        async def _gate_wake_word(old=None, new=None, **_kw) -> None:
            if new in (AtomState.LISTENING, AtomState.THINKING, AtomState.SPEAKING):
                ww.pause()
            elif new in (AtomState.IDLE, AtomState.SLEEP):
                ww.resume()

        self._bus.on("state_changed", _gate_wake_word)

    def shutdown(self) -> None:
        """Cleanly shut down all voice components."""
        if self._earcons is not None:
            try:
                self._earcons.shutdown()
            except Exception:
                logger.debug("Earcons shutdown error", exc_info=True)
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
