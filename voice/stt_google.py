"""
ATOM -- Speech-to-Text Engine (Google Cloud STT — Free Tier).

Production-grade online STT pipeline:
    - Engine: Google Web Speech API via SpeechRecognition library (free, no API key)
    - Bilingual: English + Hindi (language auto-detect via config)
    - Mic management: Bluetooth priority, persistent stream, auto-fallback
    - Noise handling: energy calibration, threshold escalation, noise flood cooldown
    - Text validation: noise word rejection, text corrections, command filtering

Pipeline:
    sr.Microphone -> sr.Recognizer.listen() -> recognize_google()
    -> text corrections -> command filter -> Intent Engine
    -> language tag emitted for response routing

Advantages over offline Whisper:
    - No model download (~0 MB vs 460+ MB)
    - Sub-500ms latency (vs 1-3s local)
    - Higher accuracy for conversational speech
    - Zero GPU/CPU load for transcription
    - Free, no API key required

Dependencies: SpeechRecognition, pyaudio (PortAudio)

Owner: Satyam
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from voice.speech_detector import MAX_IDLE_LISTEN_S, correct_text, is_noise_word

logger = logging.getLogger("atom.stt")

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.state_manager import StateManager
    from voice.mic_manager import MicManager

MAX_RECORD_S = 10.0
MIN_AUDIO_DURATION_S = 0.4
_BT_MIN_THRESHOLD = 1800.0
_RECALIBRATE_AFTER_S = 90.0


class STTGoogle:
    """Google Web Speech API STT with bilingual support.

    Uses the free Google endpoint via SpeechRecognition library.
    No API key needed. Requires internet connectivity.
    """

    _BT_KEYWORDS = (
        "headset", "hands-free", "bluetooth", "bt", "buds",
        "airpods", "earbuds", "jbl", "bose", "sony", "mivi",
        "oneplus", "realme", "yealink", "blaupunkt", "jabra",
    )

    _BT_DRIVER_BLACKLIST = (
        "@system32\\drivers", "\\drivers\\",
        ".sys,", ".sys)", "bthhfenum", "bthenum",
    )

    _MAX_ENERGY_THRESHOLD: float = 6000.0
    _SUPPORTED_LANGUAGES = {"en", "hi"}

    def __init__(
        self,
        bus: "AsyncEventBus",
        state: "StateManager",
        config: dict | None = None,
        mic_manager: "MicManager | None" = None,
        intent_engine: Any = None,
    ) -> None:
        self._bus = bus
        self._state = state
        self._mic_manager = mic_manager
        self._intent_engine = intent_engine
        self._config = config or {}
        self._audio: Any = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
        self._running = False
        self._came_from_speaking = False
        self._mic_device_index: int | None = None
        self._consecutive_errors: int = 0
        self._consecutive_noise: int = 0
        self._MAX_BACKOFF_S: float = 30.0
        self._sr_calibrated = False
        self._calibrated_threshold: float = 300.0
        self._base_threshold: float = 300.0
        self._recognizer: Any = None

        # Persistent mic handle for zero-latency wake
        self._persistent_mic: Any = None
        self._persistent_source: Any = None

        self._rejected_bt_indices: set[int] = set()
        self._last_successful_speech: float = 0.0
        self._threshold_elevated: bool = False
        self._too_noisy_emitted: bool = False
        self._last_confidence: float = 0.85
        self._last_error: str | None = None
        self._is_bt_mic: bool = False
        self._detected_language: str = "en"
        self._language_history: list[str] = []
        self.speech_permission_status: str = "unknown"
        self.microphone_permission_status: str = "unknown"

        stt_cfg = self._config.get("stt", {})
        self.POST_TTS_COOLDOWN: float = stt_cfg.get("post_tts_cooldown_ms", 600) / 1000
        self.CALIBRATION_DELAY_S: float = stt_cfg.get("calibration_delay_s", 2.0)
        self.MIN_ENERGY_THRESHOLD: float = float(stt_cfg.get("min_energy_threshold", 400))
        mic_cfg = self._config.get("mic", {})
        self.PREFER_BLUETOOTH: bool = mic_cfg.get("prefer_bluetooth", True)

        # Google STT language config
        self._bilingual: bool = stt_cfg.get("bilingual", True)
        self._google_lang: str = stt_cfg.get("google_language", "en-IN")

        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_hearing_text: str = ""
        self.mic_name: str = "Unknown Mic"
        self._listen_wait_count: int = 0

    # ── Public API ─────────────────────────────────────────────────────

    async def preload(self) -> None:
        """Detect best input mic. No model download needed for Google STT."""
        import pyaudio

        if self._audio is None:
            self._audio = pyaudio.PyAudio()

        if self._mic_manager is not None and self._mic_manager.is_profiled:
            active = self._mic_manager.active_device
            if active is not None:
                self._mic_device_index = active.index
                self.mic_name = active.name
                self._is_bt_mic = (active.device_type == "bluetooth")
                logger.info(
                    "Using MicManager-profiled device: [%d] '%s' (%s, quality=%d)",
                    active.index, active.name, active.device_type, active.quality_score,
                )
            else:
                self._select_mic_legacy()
        else:
            self._select_mic_legacy()

        lang_str = f"bilingual (en+hi, default={self._google_lang})" if self._bilingual else self._google_lang
        logger.info(
            "STT ready -- engine=google_online, mic='%s', lang=%s",
            self.mic_name, lang_str,
        )
        self._last_error = None

    @property
    def backend_name(self) -> str:
        return "Google Online"

    # Alias expected by main.py wiring
    async def async_preload(self) -> None:
        await self.preload()

    def _select_mic_legacy(self) -> None:
        """Legacy mic selection with Bluetooth priority."""
        if self.PREFER_BLUETOOTH:
            bt_idx, bt_name = self._find_bluetooth_input()
            if bt_idx is not None:
                self._mic_device_index = bt_idx
                self.mic_name = bt_name
                self._is_bt_mic = True
                logger.info("Bluetooth input detected: [%d] '%s'", bt_idx, bt_name)
            else:
                self._set_default_mic()
        else:
            self._set_default_mic()

    # ── Mic management ─────────────────────────────────────────────────

    def _set_default_mic(self) -> None:
        try:
            info = self._audio.get_default_input_device_info()
            self._mic_device_index = int(info.get("index", 0))
            self.mic_name = info.get("name", "System Default")
            logger.info("Using system default mic: [%d] '%s'",
                        self._mic_device_index, self.mic_name)
        except Exception:
            self._mic_device_index = None
            self.mic_name = "System Default"

    def _fallback_to_default_mic(self) -> None:
        import pyaudio

        if self._audio is None:
            self._audio = pyaudio.PyAudio()

        try:
            info = self._audio.get_default_input_device_info()
            new_idx = int(info.get("index", 0))
            new_name = info.get("name", "System Default")

            lower_name = new_name.lower()
            if any(blk in lower_name for blk in self._BT_DRIVER_BLACKLIST):
                new_idx = None
                new_name = "System Default"
                self._is_bt_mic = False
            elif any(kw in lower_name for kw in self._BT_KEYWORDS):
                new_idx = None
                new_name = "System Default (auto)"
                self._is_bt_mic = True
            else:
                self._is_bt_mic = False

            old_name = self.mic_name
            self._mic_device_index = new_idx
            self.mic_name = new_name
            self._sr_calibrated = False
            self._calibrated_threshold = self.MIN_ENERGY_THRESHOLD
            self._recognizer = None
            logger.info("Mic fallback: '%s' -> '%s'", old_name, new_name)

            if self._loop:
                import functools
                self._loop.call_soon_threadsafe(
                    functools.partial(self._bus.emit, "mic_changed", name=new_name))
        except Exception:
            self._mic_device_index = None
            self.mic_name = "System Default"
            self._is_bt_mic = False
            self._sr_calibrated = False
            self._calibrated_threshold = self.MIN_ENERGY_THRESHOLD
            self._recognizer = None

    def _find_bluetooth_input(self) -> tuple[int | None, str]:
        """Find the best Bluetooth input device."""
        if self._audio is None:
            return None, ""
        best_idx, best_name, best_score = None, "", -1
        for i in range(self._audio.get_device_count()):
            try:
                info = self._audio.get_device_info_by_index(i)
            except Exception:
                continue
            if info.get("maxInputChannels", 0) <= 0:
                continue
            name = info.get("name", "").lower()
            if not any(kw in name for kw in self._BT_KEYWORDS):
                continue
            if any(blk in name for blk in self._BT_DRIVER_BLACKLIST):
                continue
            if i in self._rejected_bt_indices:
                continue
            rate = int(info.get("defaultSampleRate", 0))
            score = 0
            if rate == 16000:
                score = 100
            elif 8000 < rate < 44100:
                score = 60
            elif rate == 8000:
                score = 30
            elif rate >= 44100:
                score = 10
            if score > best_score:
                best_idx = i
                best_name = info.get("name", "Bluetooth")
                best_score = score
        if best_idx is not None:
            logger.info("BT mic selected: [%d] '%s' (score=%d)",
                        best_idx, best_name, best_score)
        return best_idx, best_name

    def refresh_mic(self) -> bool:
        """Re-scan for Bluetooth devices and switch if better one found."""
        import pyaudio

        if self._audio is None:
            self._audio = pyaudio.PyAudio()

        bt_idx, bt_name = self._find_bluetooth_input()
        if bt_idx is not None and bt_idx != self._mic_device_index:
            if bt_idx in self._rejected_bt_indices:
                return False
            old_name = self.mic_name
            self._mic_device_index = bt_idx
            self.mic_name = bt_name
            self._is_bt_mic = True
            self._sr_calibrated = False
            self._calibrated_threshold = self.MIN_ENERGY_THRESHOLD
            self._recognizer = None
            logger.info("Bluetooth input switched: '%s' -> '%s'", old_name, bt_name)
            return True
        return False

    # ── Recognizer management ──────────────────────────────────────────

    def _get_recognizer(self):
        """Return the persistent recognizer for mic capture."""
        import speech_recognition as sr

        if self._recognizer is None:
            r = sr.Recognizer()
            r.energy_threshold = self._calibrated_threshold
            if self._is_bt_mic:
                r.dynamic_energy_threshold = False
                r.pause_threshold = 1.2
                r.phrase_threshold = 0.4
                r.non_speaking_duration = 0.7
            else:
                r.dynamic_energy_threshold = True
                r.dynamic_energy_adjustment_damping = 0.15
                r.dynamic_energy_ratio = 1.5
                r.pause_threshold = 1.2
                r.phrase_threshold = 0.3
                r.non_speaking_duration = 0.8
            self._recognizer = r
        else:
            effective = min(
                max(self._calibrated_threshold,
                    self._recognizer.energy_threshold),
                self._MAX_ENERGY_THRESHOLD,
            )
            self._recognizer.energy_threshold = effective
        return self._recognizer

    # ── Core listen pipeline ───────────────────────────────────────────

    def _open_mic(self):
        """Open the microphone device persistently (zero-latency wake)."""
        if self._persistent_mic is not None and self._persistent_source is not None:
            return self._persistent_mic, self._persistent_source

        import speech_recognition as sr

        mic_kwargs = {}
        if self._mic_device_index is not None:
            mic_kwargs["device_index"] = self._mic_device_index

        try:
            mic_obj = sr.Microphone(**mic_kwargs)
        except Exception as mic_err:
            logger.warning("Mic device %s init failed: %s -- falling back",
                           self._mic_device_index, mic_err)
            if self._mic_device_index is not None:
                self._rejected_bt_indices.add(self._mic_device_index)
            self._fallback_to_default_mic()
            self._sr_calibrated = False
            self._recognizer = None
            return None, None

        try:
            source = mic_obj.__enter__()
        except Exception as e:
            logger.warning("Mic stream open failed: %s -- falling back", e)
            try:
                mic_obj.__exit__(None, None, None)
            except Exception:
                pass
            if self._mic_device_index is not None:
                self._rejected_bt_indices.add(self._mic_device_index)
            self._fallback_to_default_mic()
            self._sr_calibrated = False
            self._recognizer = None
            return None, None

        if source.stream is None:
            logger.warning("Mic stream is None -- falling back")
            try:
                mic_obj.__exit__(None, None, None)
            except Exception:
                pass
            time.sleep(3.0)
            if self._mic_device_index is not None:
                self._rejected_bt_indices.add(self._mic_device_index)
            self._fallback_to_default_mic()
            self._sr_calibrated = False
            self._recognizer = None
            return None, None

        self._persistent_mic = mic_obj
        self._persistent_source = source
        logger.info("Persistent mic stream opened (zero-latency wake)")
        return mic_obj, source

    def _calibrate(self, recognizer, source) -> bool:
        """Calibrate recognizer against ambient noise."""
        from core.state_manager import AtomState

        min_thr = self.MIN_ENERGY_THRESHOLD

        if self.CALIBRATION_DELAY_S > 0:
            logger.info("Waiting %.1fs before calibration...", self.CALIBRATION_DELAY_S)
            time.sleep(self.CALIBRATION_DELAY_S)

        if not self._running or self._state.current not in (AtomState.LISTENING, AtomState.SPEAKING):
            return False

        logger.info("Calibrating for ambient noise (1.5s)...")
        recognizer.adjust_for_ambient_noise(source, duration=1.5)
        threshold = recognizer.energy_threshold
        raw_threshold = threshold

        if threshold < min_thr:
            threshold = min_thr

        if threshold > 4000:
            if self._state.current is AtomState.SPEAKING:
                self._came_from_speaking = True
                return False
            if self._is_bt_mic and threshold > 50000:
                logger.warning("BT mic '%s' noise %.0f -- unusable, falling back",
                               self.mic_name, threshold)
                if self._mic_device_index is not None:
                    self._rejected_bt_indices.add(self._mic_device_index)
                self._fallback_to_default_mic()
                self._recognizer = None
                return False
            elif self._is_bt_mic:
                clamp = max(1800, min(int(threshold * 0.20), 4000))
                threshold = clamp
            else:
                threshold = min(threshold, 4000)

        if self._is_bt_mic and threshold < _BT_MIN_THRESHOLD:
            threshold = _BT_MIN_THRESHOLD

        recognizer.energy_threshold = threshold
        self._calibrated_threshold = threshold
        self._base_threshold = threshold
        self._sr_calibrated = True
        self._last_successful_speech = time.monotonic()
        logger.info("Calibrated: raw=%.0f, effective=%.0f (bt=%s)",
                     raw_threshold, threshold, self._is_bt_mic)
        return True

    def _needs_recalibration(self) -> bool:
        if self._last_successful_speech <= 0:
            return False
        return (time.monotonic() - self._last_successful_speech) > _RECALIBRATE_AFTER_S

    def _capture_audio(self, recognizer, source):
        """Capture audio from the mic. Returns audio data or None on timeout."""
        import speech_recognition as sr
        from core.state_manager import AtomState

        if not self._running or self._state.current not in (AtomState.LISTENING, AtomState.SPEAKING):
            return None

        self._listen_wait_count += 1
        if self._listen_wait_count <= 1 or self._listen_wait_count % 15 == 0:
            logger.info("Waiting for speech (energy_threshold=%.0f)...",
                        recognizer.energy_threshold)

        try:
            return recognizer.listen(
                source, timeout=MAX_IDLE_LISTEN_S, phrase_time_limit=MAX_RECORD_S)
        except sr.WaitTimeoutError:
            return None

    def _transcribe(self, audio) -> str:
        """Transcribe audio using Google Web Speech API (free, online).

        Sends audio to Google's servers for transcription.
        Supports bilingual (en-IN covers Hindi + English mixed speech).
        """
        import speech_recognition as sr

        recognizer = sr.Recognizer()
        lang = self._google_lang if self._bilingual else "en-US"

        try:
            t0 = time.monotonic()
            text = recognizer.recognize_google(audio, language=lang)
            elapsed_ms = (time.monotonic() - t0) * 1000

            if not text or not text.strip():
                return ""

            text = text.strip()

            # Update tracking
            self._last_confidence = 0.9  # Google generally high confidence
            self._listen_wait_count = 0
            self._consecutive_noise = 0
            self._last_successful_speech = time.monotonic()

            if self._threshold_elevated:
                self._calibrated_threshold = self._base_threshold
                if self._recognizer:
                    self._recognizer.energy_threshold = self._base_threshold
                self._threshold_elevated = False

            # Detect language heuristic (Hindi characters present?)
            detected_lang = "en"
            if any("\u0900" <= c <= "\u097F" for c in text):
                detected_lang = "hi"
            self._detected_language = detected_lang
            self._language_history.append(detected_lang)
            if len(self._language_history) > 20:
                self._language_history = self._language_history[-20:]

            logger.info(
                "Google STT [%s]: '%s' (%.0fms, lang=%s)",
                detected_lang, text, elapsed_ms, lang,
            )
            return text

        except sr.UnknownValueError:
            logger.debug("Google STT: unintelligible audio -- skipped")
            self._last_error = None
            return ""
        except sr.RequestError as e:
            logger.warning("Google STT network error: %s", e)
            self._consecutive_errors += 1
            self._last_error = str(e)
            return ""
        except Exception as exc:
            self._last_error = str(exc)
            logger.exception("Google STT unexpected error")
            return ""

    def _handle_noise_flood(self, recognizer) -> None:
        """Escalate threshold on consecutive noise captures."""
        self._consecutive_noise += 1

        if self._consecutive_noise >= 2:
            cooldown = min(1.5 + self._consecutive_noise * 0.8, 8.0)
            logger.info("Noise flood (%d) -- cooling down %.1fs",
                        self._consecutive_noise, cooldown)
            if not self._too_noisy_emitted and self._consecutive_noise >= 3:
                self._too_noisy_emitted = True
                self._emit_stt_event("stt_too_noisy")
            time.sleep(cooldown)

            old_thr = self._calibrated_threshold
            new_thr = min(old_thr * 1.5, self._MAX_ENERGY_THRESHOLD)
            if new_thr != old_thr:
                self._calibrated_threshold = new_thr
                recognizer.energy_threshold = new_thr
                self._threshold_elevated = True
                logger.info("Raised energy_threshold %.0f -> %.0f", old_thr, new_thr)

            if self._consecutive_noise >= 5:
                logger.info("Persistent noise -- forcing recalibration")
                self._sr_calibrated = False
                self._consecutive_noise = 0
        else:
            time.sleep(0.5)

    def _validate_text(self, text: str) -> str | None:
        """Apply text corrections, noise filtering, and command validation."""
        text = text.strip()
        if not text:
            return None

        if is_noise_word(text):
            logger.info("Rejected noise word: '%s'", text)
            return None

        original = text
        text = correct_text(text)
        if text != original:
            logger.info("Text correction: '%s' -> '%s'", original, text)

        from core.command_filter import is_valid_command
        if not is_valid_command(text, self._last_confidence):
            return None

        return text

    def _listen_loop(self) -> str | None:
        """Listen for speech: acquire mic -> capture -> Google STT -> validate."""
        if self._mic_manager is not None:
            t_acq = time.monotonic()
            if not self._mic_manager.acquire("stt"):
                logger.warning("STT: mic acquire timed out")
                return None
            mic_acq_ms = (time.monotonic() - t_acq) * 1000
            if mic_acq_ms > 5:
                logger.info("Mic acquire wait: %.0fms", mic_acq_ms)

        try:
            if self._came_from_speaking:
                time.sleep(self.POST_TTS_COOLDOWN)
                self._came_from_speaking = False

            recognizer = self._get_recognizer()
            listen_start = time.monotonic()
            self._emit_hearing("Listening...")

            mic_obj, source = self._open_mic()
            if mic_obj is None:
                return None

            if not self._sr_calibrated or self._needs_recalibration():
                if self._needs_recalibration():
                    logger.info("Forcing recalibration (no speech in %.0fs)",
                                 _RECALIBRATE_AFTER_S)
                    self._sr_calibrated = False
                if not self._calibrate(recognizer, source):
                    return None

            audio = self._capture_audio(recognizer, source)
            if audio is None:
                return None

            audio_duration_s = len(audio.get_raw_data()) / (
                audio.sample_rate * audio.sample_width)

            if audio_duration_s < MIN_AUDIO_DURATION_S:
                logger.debug("Audio too short (%.1fs) -- noise click, skipping",
                             audio_duration_s)
                return None

            logger.info("Captured %.1fs of audio, sending to Google...", audio_duration_s)
            self._emit_hearing("Processing...")

            text = self._transcribe(audio)
            if not text:
                self._handle_noise_flood(recognizer)
                return None

            self._too_noisy_emitted = False

            total_ms = (time.monotonic() - listen_start) * 1000
            logger.info("STT latency: total=%.0fms, audio=%.1fs", total_ms, audio_duration_s)

            return self._validate_text(text)

        except OSError:
            logger.exception("Mic device error -- resetting")
            self._consecutive_errors += 1
            self._sr_calibrated = False
            self._calibrated_threshold = self.MIN_ENERGY_THRESHOLD
            self._recognizer = None
            self._reset_audio()
            return None
        except Exception:
            logger.exception("STT error (non-device)")
            self._consecutive_errors += 1
            return None
        finally:
            if self._mic_manager is not None:
                self._mic_manager.release("stt")

    # ── Thread-safe event emission ─────────────────────────────────────

    def _emit_hearing(self, text: str) -> None:
        if text == self._last_hearing_text:
            return
        self._last_hearing_text = text
        loop = self._loop
        if loop is not None:
            try:
                t = text
                loop.call_soon_threadsafe(
                    lambda: (
                        self._bus.emit_fast(
                            "voice.partial",
                            text=t,
                            confidence=float(self._last_confidence),
                            engine=self.backend_name,
                            mic=self.mic_name,
                        ),
                        self._bus.emit("speech_partial", text=t),
                    ))
            except RuntimeError:
                pass

    def _emit_stt_event(self, event: str) -> None:
        loop = self._loop
        if loop is not None:
            try:
                bus = self._bus
                loop.call_soon_threadsafe(lambda: bus.emit(event))
            except RuntimeError:
                pass

    def _reset_audio(self) -> None:
        if self._persistent_mic is not None:
            try:
                self._persistent_mic.__exit__(None, None, None)
            except Exception:
                pass
            self._persistent_mic = None
            self._persistent_source = None

        if self._audio is not None:
            try:
                self._audio.terminate()
            except Exception:
                pass
            self._audio = None
            logger.info("PyAudio reset -- will re-detect mic on next listen")

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def detected_language(self) -> str:
        return self._detected_language

    @property
    def dominant_language(self) -> str:
        if not self._language_history:
            return "en"
        from collections import Counter
        counts = Counter(self._language_history[-10:])
        return counts.most_common(1)[0][0]

    # ── Async wrappers (match STTAsync interface) ──────────────────────

    async def start_listening(self, **_kw) -> None:
        try:
            loop = asyncio.get_running_loop()
            self._loop = loop
            self._running = True
            text = await loop.run_in_executor(self._executor, self._listen_loop)

            if text:
                self._consecutive_errors = 0
                self._consecutive_noise = 0
                self._last_error = None
                lang = self._detected_language
                logger.info("STT final [%s]: '%s'", lang, text)
                self._bus.emit_fast(
                    "voice.final",
                    text=text,
                    language=lang,
                    confidence=float(self._last_confidence),
                    engine=self.backend_name,
                    mic=self.mic_name,
                )
                self._bus.emit("speech_final", text=text, language=lang)
            elif self._consecutive_errors > 0:
                backoff = min(2 ** self._consecutive_errors, self._MAX_BACKOFF_S)
                logger.warning("STT error backoff: %.1fs (attempt %d)",
                               backoff, self._consecutive_errors)
                await asyncio.sleep(backoff)
                self._bus.emit("silence_timeout")
            else:
                self._consecutive_errors = 0
                await asyncio.sleep(0.15)
                self._bus.emit("silence_timeout")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("STT start_listening failed: %s", exc)
            self._consecutive_errors += 1
            self._last_error = str(exc)
            try:
                self._bus.emit("silence_timeout")
            except Exception:
                pass

    # Alias expected by some wiring paths
    async def async_start_listening(self, **kw) -> None:
        await self.start_listening(**kw)

    def stop(self) -> None:
        self._running = False

    async def on_state_changed(self, old, new, **_kw) -> None:
        from core.state_manager import AtomState

        try:
            if new in (AtomState.LISTENING, AtomState.SPEAKING):
                if new is AtomState.SPEAKING:
                    if self._recognizer is not None:
                        self._calibrated_threshold = max(1500.0, self._base_threshold * 2.5)
                        self._recognizer.energy_threshold = self._calibrated_threshold
                        self._threshold_elevated = True

                if old is AtomState.SPEAKING and new is AtomState.LISTENING:
                    self._came_from_speaking = True
                    self._consecutive_noise = 0
                    self._too_noisy_emitted = False
                    if self._recognizer is not None:
                        restore = self._base_threshold
                        self._calibrated_threshold = restore
                        self._recognizer.energy_threshold = restore
                        self._threshold_elevated = False

                if not self._running:
                    asyncio.create_task(self.start_listening())
            elif old in (AtomState.LISTENING, AtomState.SPEAKING) and new not in (AtomState.LISTENING, AtomState.SPEAKING):
                self.stop()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.stop()
            logger.exception("STT on_state_changed failed: %s", exc)

    def on_media_started(self) -> None:
        """Pre-raise threshold when media starts."""
        if self._calibrated_threshold < 600:
            old = self._calibrated_threshold
            self._calibrated_threshold = 600.0
            self._threshold_elevated = True
            if self._recognizer is not None:
                self._recognizer.energy_threshold = 600.0
            logger.info("Media started -- threshold %.0f -> 600", old)

    def shutdown(self) -> None:
        try:
            self._running = False
            if self._mic_manager is not None:
                self._mic_manager.release("stt")
            if self._persistent_mic is not None:
                try:
                    self._persistent_mic.__exit__(None, None, None)
                except Exception:
                    pass
                self._persistent_mic = None
                self._persistent_source = None
            if self._audio:
                self._audio.terminate()
                self._audio = None
            self._executor.shutdown(wait=False)
            logger.info("STT shut down")
        except Exception:
            logger.exception("STT shutdown failed")


__all__ = ["STTGoogle"]
