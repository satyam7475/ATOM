"""
ATOM -- Speech-to-Text Engine (faster-whisper only).

Production-grade STT pipeline:
    - Engine: faster-whisper (offline, GPU-accelerated, ~300ms)
    - Bilingual: English + Hindi with auto-language detection
    - Audio preprocessing: DC removal, pre-emphasis, spectral noise gate
    - Active noise handling with learned ambient profile
    - Garbage sound rejection (spectral + energy + confidence)
    - Whisper hallucination detection
    - Language-aware response routing

Pipeline:
    sr.Microphone -> AudioPreprocessor (noise gate + normalization)
    -> faster-whisper (multilingual, auto-detect language)
    -> text corrections -> command filter -> Intent Engine
    -> language tag emitted for response routing

Recommended model: 'small' (multilingual, 244M params, ~460MB)
    - English WER: 3.4%
    - Multilingual WER: ~7%
    - Speed: 0.036 RTF on GPU (27x real-time)
    - CPU int8: ~300-500ms per utterance

Dependencies: SpeechRecognition (mic capture), pyaudio, faster-whisper

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
MIN_AUDIO_DURATION_S = 0.5
_BT_MIN_THRESHOLD = 1800.0
_RECALIBRATE_AFTER_S = 90.0
_MIN_AUDIO_QUALITY_SCORE = 0.15


class STTAsync:
    """faster-whisper STT with bilingual support and audio preprocessing."""

    _BT_KEYWORDS = ("headset", "hands-free", "bluetooth", "bt", "buds",
                     "airpods", "earbuds", "jbl", "bose", "sony", "mivi",
                     "oneplus", "realme", "yealink", "blaupunkt", "jabra")

    _MAX_ENERGY_THRESHOLD: float = 6000.0

    _SUPPORTED_LANGUAGES = {"en", "hi"}

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
        
        # Apple CoreAudio-style persistent mic (zero-latency wake)
        self._persistent_mic: Any = None
        self._persistent_source: Any = None
        
        self._rejected_bt_indices: set[int] = set()
        self._last_successful_speech: float = 0.0
        self._threshold_elevated: bool = False
        self._too_noisy_emitted: bool = False
        self._last_confidence: float = 0.85
        self._is_bt_mic: bool = False
        self._detected_language: str = "en"
        self._language_history: list[str] = []

        stt_cfg = self._config.get("stt", {})
        self.POST_TTS_COOLDOWN: float = stt_cfg.get("post_tts_cooldown_ms", 600) / 1000
        self.CALIBRATION_DELAY_S: float = stt_cfg.get("calibration_delay_s", 2.0)
        self.MIN_ENERGY_THRESHOLD: float = float(stt_cfg.get("min_energy_threshold", 400))
        mic_cfg = self._config.get("mic", {})
        self.PREFER_BLUETOOTH: bool = mic_cfg.get("prefer_bluetooth", True)

        self._whisper_model: Any = None
        self._whisper_available: bool = False
        self._whisper_model_size: str = stt_cfg.get("whisper_model_size", "small")
        self._whisper_bilingual: bool = stt_cfg.get("bilingual", True)

        from voice.audio_preprocessor import AudioPreprocessor
        self._preprocessor = AudioPreprocessor(self._config)

        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_hearing_text: str = ""
        self.mic_name: str = "Unknown Mic"
        self._listen_wait_count: int = 0

    # ── Public API ─────────────────────────────────────────────────────

    async def preload(self) -> None:
        """Detect best input mic and load faster-whisper model.

        Uses MicManager's device profiling (if available) for intelligent
        device selection. Falls back to legacy BT/default detection otherwise.
        """
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

        self._load_whisper_model()

        bilingual_str = "bilingual (en+hi)" if self._whisper_bilingual else "English-only"
        logger.info(
            "STT ready -- engine=faster-whisper, model=%s, mic='%s', lang=%s, preprocessor=%s",
            self._whisper_model_size, self.mic_name, bilingual_str,
            "enabled" if self._preprocessor._enabled else "disabled",
        )

    def _select_mic_legacy(self) -> None:
        """Legacy mic selection (before MicManager profiling existed)."""
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
            self._set_default_mic()

    def _load_whisper_model(self) -> None:
        """Load faster-whisper model (multilingual for Hindi+English).

        Recommended: 'small' (244M params, ~460MB, 3.4% WER English)
        Uses GPU (cuda) if available, otherwise CPU with int8 quantization.
        """
        try:
            from faster_whisper import WhisperModel

            model_name = self._whisper_model_size
            if self._whisper_bilingual and model_name.endswith(".en"):
                model_name = model_name.replace(".en", "")
                logger.info(
                    "Bilingual mode: switching from '%s' to '%s' (multilingual)",
                    self._whisper_model_size, model_name,
                )

            device = "cpu"
            compute_type = "int8"
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                    compute_type = "float16"
                    logger.info("CUDA available -- whisper will use GPU acceleration")
            except ImportError:
                pass

            t0 = time.monotonic()
            self._whisper_model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
            )
            self._whisper_available = True
            elapsed = (time.monotonic() - t0) * 1000
            lang_label = "bilingual (en+hi)" if self._whisper_bilingual else "English-only"
            logger.info(
                "faster-whisper loaded: model=%s, device=%s, %s, %.0fms",
                model_name, device, lang_label, elapsed,
            )
        except ImportError:
            logger.error("faster-whisper not installed -- STT will not work")
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
        """Find the best Bluetooth input device for speech recognition."""
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
        """Return the persistent recognizer for mic capture.

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

    # ── Core listen pipeline ───────────────────────────────────────────

    def _open_mic(self):
        """Open the microphone device persistently (Apple CoreAudio style).
        
        Instead of opening and closing the mic for every utterance (which adds
        300-500ms latency and drops the first syllable), we keep the audio
        stream hot in the background.
        """
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

        self._persistent_mic = mic_obj
        self._persistent_source = source
        logger.info("Persistent mic stream opened (zero-latency wake enabled)")
        return mic_obj, source

    def _calibrate(self, recognizer, source) -> bool:
        """Calibrate recognizer AND AudioPreprocessor against ambient noise.

        Dual calibration:
            1. SpeechRecognition energy threshold (for listen() triggering)
            2. AudioPreprocessor noise profile (for spectral noise gating)
        """
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

        try:
            import speech_recognition as sr
            ambient_audio = recognizer.listen(
                source, timeout=1.5, phrase_time_limit=1.5,
            )
            ambient_wav = ambient_audio.get_wav_data(convert_rate=16000, convert_width=2)
            ambient_pcm = ambient_wav[44:]
            if len(ambient_pcm) > 3200:
                self._preprocessor.learn_noise(ambient_pcm)
                logger.info("AudioPreprocessor noise profile learned from ambient audio")
        except Exception:
            logger.debug("Ambient capture for preprocessor skipped (timeout or error)")

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
        logger.info("Calibrated: raw=%.0f, effective=%.0f (bt=%s, preprocessor=%s)",
                     raw_threshold, threshold, self._is_bt_mic,
                     "stable" if self._preprocessor._noise_profile.is_stable else "learning")
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
        """Transcribe audio using faster-whisper with preprocessing + bilingual.

        Pipeline:
            1. Extract raw PCM from audio
            2. Run AudioPreprocessor (noise gate, normalize, quality check)
            3. Reject if quality too low
            4. Transcribe with language=None for auto-detection (en/hi)
            5. Track detected language for response routing
        """
        if not self._whisper_available or self._whisper_model is None:
            logger.warning("Whisper not available -- cannot transcribe")
            return ""
        try:
            import numpy as np

            wav_data = audio.get_wav_data(convert_rate=16000, convert_width=2)
            raw_pcm = wav_data[44:]

            clean_audio, quality = self._preprocessor.process(raw_pcm)

            if quality.is_silence:
                logger.debug("Preprocessor: silence detected, skipping STT")
                return ""

            if quality.score < _MIN_AUDIO_QUALITY_SCORE:
                logger.info(
                    "Preprocessor: low quality (%.2f) -- SNR=%.1fdB, energy=%.3f -- rejected",
                    quality.score, quality.snr_db, quality.energy_rms,
                )
                return ""

            if quality.is_clipped:
                logger.warning("Audio clipping detected -- mic gain may be too high")

            whisper_lang = None if self._whisper_bilingual else "en"

            segments, info = self._whisper_model.transcribe(
                clean_audio,
                beam_size=3 if self._whisper_bilingual else 1,
                language=whisper_lang,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 250,
                    "speech_pad_ms": 200,
                    "threshold": 0.35,
                },
                condition_on_previous_text=False,
                no_speech_threshold=0.5,
            )

            parts = []
            total_prob = 0.0
            count = 0
            for seg in segments:
                seg_text = seg.text.strip()
                if seg_text:
                    parts.append(seg_text)
                    total_prob += seg.avg_log_prob
                    count += 1

            text = " ".join(parts).strip()
            if not text:
                return ""

            detected_lang = getattr(info, "language", "en") or "en"
            lang_prob = getattr(info, "language_probability", 0.0) or 0.0

            if detected_lang in self._SUPPORTED_LANGUAGES:
                self._detected_language = detected_lang
                self._language_history.append(detected_lang)
                if len(self._language_history) > 20:
                    self._language_history = self._language_history[-20:]

            avg_prob = total_prob / count if count > 0 else -1.0

            confidence_threshold = -1.0 if detected_lang == "hi" else -1.2
            if avg_prob < confidence_threshold:
                logger.info(
                    "Whisper LOW CONFIDENCE: '%.60s' (avg_log_prob=%.2f, lang=%s) -- rejected",
                    text, avg_prob, detected_lang,
                )
                return ""

            if self._is_whisper_hallucination(text):
                logger.info("Whisper hallucination rejected: '%.60s'", text)
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

            lang_label = "hi" if detected_lang == "hi" else "en"
            logger.info(
                "Whisper STT [%s]: '%s' (avg_log_prob=%.2f, lang_prob=%.2f, quality=%.2f)",
                lang_label, text, avg_prob, lang_prob, quality.score,
            )
            return text
        except Exception:
            logger.exception("faster-whisper transcription error")
            return ""

    @staticmethod
    def _is_whisper_hallucination(text: str) -> bool:
        """Detect common whisper hallucinations on noise/silence."""
        t = text.strip().lower()

        hallucination_patterns = [
            "thank you for watching",
            "thanks for watching",
            "please subscribe",
            "like and subscribe",
            "thank you for listening",
            "thanks for listening",
            "the end",
            "subtitles by",
            "copyright",
            "music playing",
            "[music]",
            "(music)",
            "you",
        ]
        for pattern in hallucination_patterns:
            if t == pattern or t.startswith(pattern):
                return True

        if len(t) > 0 and t == t[0] * len(t):
            return True

        words = t.split()
        if len(words) >= 3 and len(set(words)) == 1:
            return True

        return False

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
                logger.info("Persistent noise -- forcing recalibration next cycle")
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
        """Listen for speech: acquire mic -> capture -> transcribe -> validate."""
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
                logger.debug("Audio too short (%.1fs < %.1fs) -- noise click, skipping",
                             audio_duration_s, MIN_AUDIO_DURATION_S)
                return None

            logger.info("Captured %.1fs of audio, transcribing...", audio_duration_s)
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
        """Last detected language code ('en' or 'hi')."""
        return self._detected_language

    @property
    def dominant_language(self) -> str:
        """Most frequently detected language in recent history."""
        if not self._language_history:
            return "en"
        from collections import Counter
        counts = Counter(self._language_history[-10:])
        return counts.most_common(1)[0][0]

    @property
    def preprocessor(self):
        return self._preprocessor

    async def _handle_runtime_error(
        self,
        source: str,
        exc: Exception,
        *,
        notify_timeout: bool = True,
    ) -> None:
        logger.exception("STT %s failed: %s", source, exc)
        self._consecutive_errors += 1
        try:
            self._bus.emit_fast("metrics_event", counter="errors_total")
        except Exception:
            pass
        if notify_timeout:
            try:
                self._bus.emit("silence_timeout")
            except Exception:
                logger.debug("STT fallback timeout emit failed", exc_info=True)

    # ── Async wrappers ─────────────────────────────────────────────────

    async def start_listening(self, **_kw) -> None:
        try:
            loop = asyncio.get_running_loop()
            self._loop = loop
            self._running = True
            text = await loop.run_in_executor(self._executor, self._listen_loop)

            if text:
                self._consecutive_errors = 0
                self._consecutive_noise = 0
                lang = self._detected_language
                logger.info("STT final [%s]: '%s'", lang, text)
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
            await self._handle_runtime_error("start_listening", exc)

    def stop(self) -> None:
        self._running = False

    async def on_state_changed(self, old, new, **_kw) -> None:
        from core.state_manager import AtomState

        try:
            if new in (AtomState.LISTENING, AtomState.SPEAKING):
                if new is AtomState.SPEAKING:
                    # Elevate threshold during speaking to prevent self-triggering
                    if self._recognizer is not None:
                        self._calibrated_threshold = max(1500.0, self._base_threshold * 2.5)
                        self._recognizer.energy_threshold = self._calibrated_threshold
                        self._threshold_elevated = True
                        logger.info("Speaking: elevated threshold to %.0f for barge-in", self._calibrated_threshold)

                if old is AtomState.SPEAKING and new is AtomState.LISTENING:
                    self._came_from_speaking = True
                    self._consecutive_noise = 0
                    self._too_noisy_emitted = False
                    if self._recognizer is not None:
                        restore = self._base_threshold
                        self._calibrated_threshold = restore
                        self._recognizer.energy_threshold = restore
                        self._threshold_elevated = False
                        logger.info("Post-TTS: threshold restored to base %.0f", restore)

                if not self._running:
                    asyncio.create_task(self.start_listening())
            elif old in (AtomState.LISTENING, AtomState.SPEAKING) and new not in (AtomState.LISTENING, AtomState.SPEAKING):
                self.stop()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.stop()
            await self._handle_runtime_error("on_state_changed", exc)

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
