"""
ATOM v16 -- Multi-engine STT orchestrator.

Supports:
    - faster-whisper (offline, ~300ms, default -- base.en model, ~140MB)
    - Vosk (offline, 100-200ms, legacy fallback)
    - Google Web Speech API (cloud, 300-700ms, fallback)

Pipeline:
    sr.Microphone -> sr.Recognizer.listen() -> faster-whisper/Vosk/Google -> text
    -> text corrections -> command filter -> Intent Engine

Noise hardening (corporate office environment):
    - BT mic minimum threshold 1800 (filters HVAC, keyboard, chatter)
    - Dynamic energy disabled for BT (prevents threshold drift)
    - Minimum audio duration 0.5s (rejects clicks/pops)
    - Aggressive noise flood: escalate after 2 fails, +50% threshold
    - Periodic recalibration every 90s without successful speech
    - Post-TTS cooldown 600ms to absorb earbuds echo

Dependencies: SpeechRecognition, pyaudio, faster-whisper (primary), vosk (optional)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
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
MIN_AUDIO_DURATION_S = 0.5
_BT_MIN_THRESHOLD = 1800.0
_RECALIBRATE_AFTER_S = 90.0


class STTAsync:
    """Multi-engine speech-to-text: Vosk (offline) or Google Web Speech (cloud)."""

    _BT_KEYWORDS = ("headset", "hands-free", "bluetooth", "bt", "buds",
                     "airpods", "earbuds", "jbl", "bose", "sony", "mivi",
                     "oneplus", "realme", "yealink", "blaupunkt", "jabra")

    _MAX_ENERGY_THRESHOLD: float = 6000.0

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
        self._rejected_bt_indices: set[int] = set()
        self._last_successful_speech: float = 0.0
        self._threshold_elevated: bool = False
        self._too_noisy_emitted: bool = False
        self._last_confidence: float = 0.85
        self._is_bt_mic: bool = False

        stt_cfg = self._config.get("stt", {})
        self.POST_TTS_COOLDOWN: float = stt_cfg.get("post_tts_cooldown_ms", 600) / 1000
        self.CALIBRATION_DELAY_S: float = stt_cfg.get("calibration_delay_s", 2.0)
        self.MIN_ENERGY_THRESHOLD: float = float(stt_cfg.get("min_energy_threshold", 400))
        mic_cfg = self._config.get("mic", {})
        self.PREFER_BLUETOOTH: bool = mic_cfg.get("prefer_bluetooth", True)

        self._stt_engine: str = stt_cfg.get("engine", "faster_whisper")
        self._vosk_model_path: str = stt_cfg.get("vosk_model_path",
                                                   "models/vosk-model-en-us-0.22")
        self._vosk_model: Any = None
        self._vosk_recognizer: Any = None
        self._vosk_available: bool = False

        self._whisper_model: Any = None
        self._whisper_available: bool = False
        self._whisper_model_size: str = stt_cfg.get("whisper_model_size", "base.en")

        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_hearing_text: str = ""
        self.mic_name: str = "Unknown Mic"
        self._listen_wait_count: int = 0

    # ── Public API ─────────────────────────────────────────────────────

    async def preload(self) -> None:
        """Detect best input mic, load Vosk model if configured."""
        import pyaudio

        if self._audio is None:
            self._audio = pyaudio.PyAudio()

        mic_cfg = self._config.get("mic", {})
        device_name_cfg = mic_cfg.get("device_name")

        if self.PREFER_BLUETOOTH:
            bt_idx, bt_name = self._find_bluetooth_input()
            if bt_idx is not None:
                self._mic_device_index = bt_idx
                self.mic_name = bt_name
                self._is_bt_mic = True
                logger.info("Bluetooth input detected -- prioritizing: [%d] '%s'",
                            bt_idx, bt_name)
            else:
                self._set_default_mic()
        else:
            from voice.mic_selector import find_device
            idx, _rate, _ch = find_device(
                self._audio, device_name=device_name_cfg, exclude_bluetooth=True)
            if idx is not None:
                try:
                    info = self._audio.get_device_info_by_index(idx)
                    self._mic_device_index = idx
                    self.mic_name = info.get("name", "Selected Mic")
                    logger.info("Using non-Bluetooth mic: [%d] '%s'", idx, self.mic_name)
                except Exception:
                    self._set_default_mic()
            else:
                self._set_default_mic()

        if self._stt_engine == "faster_whisper":
            self._load_whisper_model()
        elif self._stt_engine in ("vosk", "vosk_with_fallback"):
            self._load_vosk_model()

        engine_label = self._stt_engine
        if self._stt_engine == "faster_whisper" and not self._whisper_available:
            engine_label = "google (whisper unavailable)"
        elif self._stt_engine in ("vosk", "vosk_with_fallback") and not self._vosk_available:
            engine_label = "google (vosk unavailable)"
        logger.info("STT ready -- engine=%s, mic=%s", engine_label, self.mic_name)

    def _load_vosk_model(self) -> None:
        """Load the Vosk model for offline speech recognition."""
        try:
            import vosk
            vosk.SetLogLevel(-1)

            atom_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(atom_root, self._vosk_model_path)

            if not os.path.isdir(model_path):
                logger.warning("Vosk model not found at '%s' -- will use Google STT", model_path)
                self._vosk_available = False
                return

            self._vosk_model = vosk.Model(model_path)
            self._vosk_recognizer = vosk.KaldiRecognizer(self._vosk_model, 16000)
            self._vosk_available = True
            logger.info("Vosk model loaded from '%s'", model_path)
        except ImportError:
            logger.warning("vosk package not installed -- falling back to Google STT")
            self._vosk_available = False
        except Exception:
            logger.exception("Failed to load Vosk model -- falling back to Google STT")
            self._vosk_available = False

    def _load_whisper_model(self) -> None:
        """Load faster-whisper model for offline STT (~140MB for base.en)."""
        try:
            from faster_whisper import WhisperModel
            t0 = time.monotonic()
            self._whisper_model = WhisperModel(
                self._whisper_model_size,
                device="cpu",
                compute_type="int8",
            )
            self._whisper_available = True
            elapsed = (time.monotonic() - t0) * 1000
            logger.info("faster-whisper loaded model=%s in %.0fms",
                        self._whisper_model_size, elapsed)
        except ImportError:
            logger.warning("faster-whisper not installed -- falling back to Google STT")
            self._whisper_available = False
        except Exception:
            logger.exception("Failed to load faster-whisper model")
            self._whisper_available = False

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
                logger.debug("Default mic is driver path, using system default: '%s'", new_name)
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

    _BT_DRIVER_BLACKLIST = (
        "@system32\\drivers",
        "\\drivers\\",
        ".sys,",
        ".sys)",
        "bthhfenum",
        "bthenum",
    )

    def _find_bluetooth_input(self) -> tuple[int | None, str]:
        """Find the best Bluetooth input device for speech recognition.

        Prefers 16000Hz endpoints (HFP speech profile) over 44100Hz
        endpoints (A2DP media profile) because Google Web Speech API
        expects 16kHz and the speech profile has lower noise.
        Filters out driver path entries (e.g. @System32\\drivers\\bthhfenum.sys).
        """
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
                logger.debug("BT skip driver path [%d] '%s'", i, info.get("name", ""))
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
                logger.debug("BT candidate [%d] '%s' rate=%d score=%d",
                             i, best_name, rate, score)
        if best_idx is not None:
            logger.info("BT mic selected: [%d] '%s' (score=%d)",
                        best_idx, best_name, best_score)
        return best_idx, best_name

    def refresh_mic(self) -> bool:
        """Re-scan for Bluetooth devices and switch if a better one appears."""
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

        if bt_idx is None and self.mic_name and any(
            kw in self.mic_name.lower() for kw in self._BT_KEYWORDS
        ):
            try:
                info = self._audio.get_default_input_device_info()
                self._mic_device_index = int(info.get("index", 0))
                old_name = self.mic_name
                self.mic_name = info.get("name", "System Default")
                self._is_bt_mic = False
                self._sr_calibrated = False
                self._calibrated_threshold = self.MIN_ENERGY_THRESHOLD
                self._recognizer = None
                logger.info("Bluetooth disconnected, falling back: '%s' -> '%s'",
                            old_name, self.mic_name)
                return True
            except Exception:
                pass
        return False

    # ── Recognizer management ──────────────────────────────────────────

    def _get_recognizer(self):
        """Return the persistent recognizer, creating it only once.

        BT mics: dynamic threshold disabled (it drifts and captures noise).
        Wired mics: dynamic threshold enabled for auto-adjustment.
        """
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

    # ── Core listen pipeline (decomposed) ──────────────────────────────

    def _open_mic(self):
        """Open the microphone device. Returns (mic_obj, source) or (None, None)."""
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
            logger.warning("Mic stream is None -- waiting 3s then falling back")
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

        return mic_obj, source

    def _calibrate(self, recognizer, source) -> bool:
        """Calibrate the recognizer against ambient noise.

        BT mics get a minimum threshold of 3500 to filter office noise.
        Returns True if calibration succeeded, False to abort.
        """
        from core.state_manager import AtomState

        min_thr = self.MIN_ENERGY_THRESHOLD

        if self.CALIBRATION_DELAY_S > 0:
            logger.info("Waiting %.1fs before calibration...", self.CALIBRATION_DELAY_S)
            time.sleep(self.CALIBRATION_DELAY_S)

        if not self._running or self._state.current is not AtomState.LISTENING:
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
        """Force recalibration if no successful speech in _RECALIBRATE_AFTER_S."""
        if self._last_successful_speech <= 0:
            return False
        elapsed = time.monotonic() - self._last_successful_speech
        return elapsed > _RECALIBRATE_AFTER_S

    def _capture_audio(self, recognizer, source):
        """Capture audio from the mic. Returns audio data or None on timeout."""
        import speech_recognition as sr
        from core.state_manager import AtomState

        if not self._running or self._state.current is not AtomState.LISTENING:
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

    def _transcribe(self, recognizer, audio) -> str:
        """Transcribe audio using the configured engine.

        Priority: faster-whisper > Vosk > Google Web Speech.
        """
        if self._whisper_available and self._stt_engine == "faster_whisper":
            text = self._transcribe_whisper(audio)
            if text:
                return text
            self._handle_noise_flood(recognizer)
            return ""

        use_vosk = (
            self._vosk_available
            and self._stt_engine in ("vosk", "vosk_with_fallback")
        )
        if use_vosk:
            text = self._transcribe_vosk(audio)
            if text:
                return text
            if self._stt_engine == "vosk":
                self._handle_noise_flood(recognizer)
                return ""

        return self._transcribe_google(recognizer, audio)

    def _transcribe_whisper(self, audio) -> str:
        """Transcribe audio using faster-whisper (offline, high accuracy)."""
        if self._whisper_model is None:
            return ""
        try:
            import io
            import numpy as np

            wav_data = audio.get_wav_data(convert_rate=16000, convert_width=2)
            raw_pcm = wav_data[44:]
            samples = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0

            segments, info = self._whisper_model.transcribe(
                samples,
                beam_size=1,
                language="en",
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
            )
            parts = []
            total_prob = 0.0
            count = 0
            for seg in segments:
                parts.append(seg.text.strip())
                total_prob += seg.avg_log_prob
                count += 1

            text = " ".join(parts).strip()
            if not text:
                return ""

            avg_prob = total_prob / count if count > 0 else -1.0
            if avg_prob < -1.2:
                logger.info("Whisper LOW CONFIDENCE: '%.60s' (avg_log_prob=%.2f) -- rejected",
                            text, avg_prob)
                return ""

            self._last_confidence = min(1.0, max(0.0, 1.0 + avg_prob))
            self._listen_wait_count = 0
            self._consecutive_noise = 0
            self._last_successful_speech = time.monotonic()

            if self._threshold_elevated:
                self._calibrated_threshold = self._base_threshold
                if self._recognizer:
                    self._recognizer.energy_threshold = self._base_threshold
                self._threshold_elevated = False

            logger.info("Whisper STT: '%s' (avg_log_prob=%.2f)", text, avg_prob)
            return text
        except Exception:
            logger.exception("faster-whisper transcription error")
            return ""

    def _transcribe_vosk(self, audio) -> str:
        """Transcribe captured audio using offline Vosk recognizer.

        Creates a fresh KaldiRecognizer per utterance to prevent context
        leakage from previous audio producing garbage results.
        """
        if self._vosk_model is None:
            return ""

        try:
            import vosk
            rec = vosk.KaldiRecognizer(self._vosk_model, 16000)
            rec.SetWords(True)

            wav_data = audio.get_wav_data(convert_rate=16000, convert_width=2)
            raw_pcm = wav_data[44:]

            rec.AcceptWaveform(raw_pcm)
            result = json.loads(rec.FinalResult())
            text = result.get("text", "").strip()

            words_info = result.get("result", [])
            if words_info:
                avg_conf = sum(w.get("conf", 0) for w in words_info) / len(words_info)
                if avg_conf < 0.55:
                    logger.info("Vosk LOW CONFIDENCE: '%.60s' (avg_conf=%.2f) -- rejected",
                                text, avg_conf)
                    return ""
                self._last_confidence = avg_conf
            else:
                self._last_confidence = 0.70

            if not text:
                return ""

            self._last_confidence = 0.85
            self._listen_wait_count = 0
            self._consecutive_noise = 0
            self._last_successful_speech = time.monotonic()

            if self._threshold_elevated:
                self._calibrated_threshold = self._base_threshold
                if self._recognizer:
                    self._recognizer.energy_threshold = self._base_threshold
                self._threshold_elevated = False
                logger.info("Restored energy_threshold to base %.0f", self._base_threshold)

            logger.info("Vosk STT: '%s'", text)
            return text

        except Exception:
            logger.exception("Vosk transcription error")
            return ""

    def _transcribe_google(self, recognizer, audio) -> str:
        """Send audio to Google Web Speech API and return transcribed text.

        Returns empty string on failure, low confidence, or unknown audio.
        Handles noise flood detection and threshold escalation.
        """
        import speech_recognition as sr

        try:
            raw_result = recognizer.recognize_google(audio, show_all=True)

            text = ""
            confidence = 0.0

            if raw_result and isinstance(raw_result, dict):
                alts = raw_result.get("alternative", [])
                if alts:
                    text = alts[0].get("transcript", "")
                    confidence = float(alts[0].get("confidence", 0.85))
            elif raw_result and isinstance(raw_result, list):
                for entry in raw_result:
                    alts = entry.get("alternative", []) if isinstance(entry, dict) else []
                    if alts:
                        text = alts[0].get("transcript", "")
                        confidence = float(alts[0].get("confidence", 0.85))
                        break

            if not text:
                raise sr.UnknownValueError()

            self._last_confidence = confidence

            if confidence < 0.5:
                logger.info("LOW CONFIDENCE: '%.40s' (conf=%.2f) -- rejected", text, confidence)
                self._emit_stt_event("stt_did_not_catch")
                time.sleep(0.3)
                return ""

            self._listen_wait_count = 0
            logger.info("Google STT: '%s' (conf=%.2f)", text, confidence)
            self._consecutive_noise = 0
            self._last_successful_speech = time.monotonic()

            if self._threshold_elevated:
                self._calibrated_threshold = self._base_threshold
                recognizer.energy_threshold = self._base_threshold
                self._threshold_elevated = False
                logger.info("Restored energy_threshold to base %.0f", self._base_threshold)

            return text

        except sr.UnknownValueError:
            self._handle_noise_flood(recognizer)
            return ""
        except sr.RequestError as e:
            logger.warning("Google STT request failed: %s", e)
            time.sleep(1.0)
            return ""

    def _handle_noise_flood(self, recognizer) -> None:
        """Aggressively escalate threshold on consecutive noise.

        Triggers after just 2 consecutive failures (not 3).
        Raises threshold by 50% per step (not 40%).
        Caps at _MAX_ENERGY_THRESHOLD.
        """
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
                logger.info("Persistent noise -- forcing recalibration next cycle")
                self._sr_calibrated = False
                self._consecutive_noise = 0
        else:
            logger.info("Google STT: could not understand audio")
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
        """Listen for speech using the decomposed pipeline.

        Steps: acquire mic -> open device -> calibrate -> capture -> transcribe -> validate.
        """
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

            try:
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
            finally:
                try:
                    mic_obj.__exit__(None, None, None)
                except Exception:
                    pass

            audio_duration_s = len(audio.get_raw_data()) / (
                audio.sample_rate * audio.sample_width)

            if audio_duration_s < MIN_AUDIO_DURATION_S:
                logger.debug("Audio too short (%.1fs < %.1fs) -- noise click, skipping",
                             audio_duration_s, MIN_AUDIO_DURATION_S)
                return None

            logger.info("Captured %.1fs of audio, sending to Google...", audio_duration_s)
            self._emit_hearing("Processing...")

            text = self._transcribe(recognizer, audio)
            if not text:
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
                    lambda: self._bus.emit("speech_partial", text=t))
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
        if self._audio is not None:
            try:
                self._audio.terminate()
            except Exception:
                pass
            self._audio = None
            logger.info("PyAudio reset -- will re-detect mic on next listen")

    # ── Async wrappers ─────────────────────────────────────────────────

    async def start_listening(self, **_kw) -> None:
        loop = asyncio.get_running_loop()
        self._loop = loop
        self._running = True
        text = await loop.run_in_executor(self._executor, self._listen_loop)

        if text:
            self._consecutive_errors = 0
            self._consecutive_noise = 0
            logger.info("STT final: '%s'", text)
            self._bus.emit("speech_final", text=text)
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

    def stop(self) -> None:
        self._running = False

    async def on_state_changed(self, old, new, **_kw) -> None:
        from core.state_manager import AtomState

        if new is AtomState.LISTENING:
            if old is AtomState.SPEAKING:
                self._came_from_speaking = True
                self._consecutive_noise = 0
                self._too_noisy_emitted = False
                if self._recognizer is not None:
                    restore = self._base_threshold
                    self._calibrated_threshold = restore
                    self._recognizer.energy_threshold = restore
                    self._threshold_elevated = False
                    logger.info("Post-TTS: threshold restored to base %.0f", restore)
            asyncio.create_task(self.start_listening())
        elif old is AtomState.LISTENING:
            self.stop()

    def on_media_started(self) -> None:
        """Pre-raise threshold when media starts to filter speaker bleed."""
        if self._calibrated_threshold < 600:
            old = self._calibrated_threshold
            self._calibrated_threshold = 600.0
            self._threshold_elevated = True
            if self._recognizer is not None:
                self._recognizer.energy_threshold = 600.0
            logger.info("Media started -- threshold %.0f -> 600", old)

    def shutdown(self) -> None:
        self._running = False
        if self._mic_manager is not None:
            self._mic_manager.release("stt")
        if self._audio:
            self._audio.terminate()
            self._audio = None
        self._executor.shutdown(wait=False)
        logger.info("STT shut down")
