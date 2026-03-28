"""
ATOM -- Voice Emotion Detection Engine.

Analyzes voice characteristics to detect the user's emotional state:
  - Pitch (fundamental frequency)
  - Speed (words per second)
  - Volume (RMS energy)
  - Spectral features (brightness/warmth)

Classifies into: neutral, stressed, excited, tired, frustrated, happy, calm

This allows ATOM to respond empathetically -- like JARVIS reading
Tony Stark's stress levels and adjusting its tone accordingly.

Output is fed into:
  - PersonalityModes (adjusts ATOM's response tone)
  - SecondBrain (episodic memory -- "Boss was stressed at 11pm")
  - Prompt Builder (emotion hint in LLM context)

No external ML models required -- pure signal processing.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("atom.emotion")


@dataclass
class EmotionResult:
    """Result of voice emotion analysis."""
    emotion: str = "neutral"
    confidence: float = 0.5
    energy: float = 0.0
    pitch_hz: float = 0.0
    speech_rate: float = 0.0
    spectral_centroid: float = 0.0

    def __repr__(self) -> str:
        return (
            f"Emotion({self.emotion}, conf={self.confidence:.2f}, "
            f"energy={self.energy:.0f}, pitch={self.pitch_hz:.0f}Hz)"
        )

    @property
    def is_negative(self) -> bool:
        return self.emotion in ("stressed", "frustrated", "tired")

    @property
    def is_positive(self) -> bool:
        return self.emotion in ("happy", "excited")

    @property
    def prompt_hint(self) -> str:
        """One-line hint for the LLM prompt."""
        hints = {
            "stressed": "The user sounds stressed. Be calm, supportive, and concise.",
            "frustrated": "The user sounds frustrated. Acknowledge the frustration, be patient.",
            "tired": "The user sounds tired. Be gentle, brief, and consider suggesting rest.",
            "excited": "The user sounds excited. Match their energy, be enthusiastic.",
            "happy": "The user sounds happy. Be warm and positive.",
            "calm": "The user is calm. Respond naturally and thoughtfully.",
            "neutral": "",
        }
        return hints.get(self.emotion, "")


class EmotionDetector:
    """Voice-based emotion detection using acoustic features."""

    _BASELINE_ENERGY = 2000.0
    _BASELINE_PITCH = 160.0

    def __init__(self, config: dict | None = None) -> None:
        self._config = (config or {}).get("emotion", {})
        self._enabled = self._config.get("enabled", True)
        self._history: list[EmotionResult] = []
        self._max_history = 20
        self._baseline_energy = self._BASELINE_ENERGY
        self._baseline_pitch = self._BASELINE_PITCH
        self._calibrated = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def current_emotion(self) -> str:
        if self._history:
            return self._history[-1].emotion
        return "neutral"

    @property
    def last_result(self) -> EmotionResult | None:
        return self._history[-1] if self._history else None

    def analyze_audio(self, audio_data: bytes, sample_rate: int = 16000) -> EmotionResult:
        """Analyze raw audio data for emotional indicators."""
        if not self._enabled:
            return EmotionResult()

        try:
            import numpy as np
            samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)

            if len(samples) < sample_rate * 0.3:
                return EmotionResult()

            energy = self._compute_rms(samples)
            pitch = self._estimate_pitch(samples, sample_rate)
            spectral = self._spectral_centroid(samples, sample_rate)
            duration = len(samples) / sample_rate

            if not self._calibrated and energy > 100:
                self._baseline_energy = energy
                self._baseline_pitch = pitch if pitch > 0 else self._BASELINE_PITCH
                self._calibrated = True

            emotion, confidence = self._classify(energy, pitch, spectral, duration)

            result = EmotionResult(
                emotion=emotion,
                confidence=confidence,
                energy=energy,
                pitch_hz=pitch,
                speech_rate=0.0,
                spectral_centroid=spectral,
            )

            self._history.append(result)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

            if emotion != "neutral":
                logger.info("Emotion: %s (conf=%.2f, energy=%.0f, pitch=%.0fHz)",
                            emotion, confidence, energy, pitch)

            return result

        except ImportError:
            return EmotionResult()
        except Exception:
            logger.debug("Emotion analysis failed", exc_info=True)
            return EmotionResult()

    def analyze_text_emotion(self, text: str) -> EmotionResult:
        """Lightweight text-based emotion heuristic as fallback."""
        if not self._enabled or not text:
            return EmotionResult()

        t = text.lower()

        frustrated_words = ["damn", "ugh", "annoying", "stupid", "broken",
                           "not working", "again", "frustrated", "hate"]
        stressed_words = ["urgent", "deadline", "hurry", "asap", "quickly",
                         "running out", "pressure", "stress"]
        happy_words = ["awesome", "great", "perfect", "amazing", "love",
                      "excellent", "wonderful", "fantastic", "nice"]
        tired_words = ["tired", "exhausted", "sleepy", "can't focus",
                      "drowsy", "yawn", "long day"]
        excited_words = ["wow", "incredible", "unbelievable", "exciting",
                        "can't wait", "thrilling"]

        if any(w in t for w in frustrated_words):
            return EmotionResult(emotion="frustrated", confidence=0.6)
        if any(w in t for w in stressed_words):
            return EmotionResult(emotion="stressed", confidence=0.6)
        if any(w in t for w in excited_words):
            return EmotionResult(emotion="excited", confidence=0.6)
        if any(w in t for w in happy_words):
            return EmotionResult(emotion="happy", confidence=0.5)
        if any(w in t for w in tired_words):
            return EmotionResult(emotion="tired", confidence=0.6)

        if text.endswith("!") or text.endswith("!!"):
            return EmotionResult(emotion="excited", confidence=0.4)
        if text.isupper() and len(text) > 5:
            return EmotionResult(emotion="frustrated", confidence=0.5)

        return EmotionResult()

    @staticmethod
    def _compute_rms(samples: Any) -> float:
        import numpy as np
        return float(np.sqrt(np.mean(samples ** 2)))

    @staticmethod
    def _estimate_pitch(samples: Any, sample_rate: int) -> float:
        """Simple autocorrelation-based pitch estimation."""
        import numpy as np
        if len(samples) < 1024:
            return 0.0

        samples = samples - np.mean(samples)
        windowed = samples[:4096] * np.hanning(min(4096, len(samples)))

        corr = np.correlate(windowed, windowed, mode='full')
        corr = corr[len(corr) // 2:]

        min_lag = sample_rate // 500
        max_lag = sample_rate // 50

        if max_lag >= len(corr):
            return 0.0

        search_region = corr[min_lag:max_lag]
        if len(search_region) == 0:
            return 0.0

        peak_idx = np.argmax(search_region) + min_lag

        if corr[peak_idx] < corr[0] * 0.2:
            return 0.0

        if peak_idx > 0:
            return float(sample_rate / peak_idx)
        return 0.0

    @staticmethod
    def _spectral_centroid(samples: Any, sample_rate: int) -> float:
        import numpy as np
        spectrum = np.abs(np.fft.rfft(samples[:4096]))
        freqs = np.fft.rfftfreq(min(4096, len(samples)), d=1.0 / sample_rate)
        total = np.sum(spectrum)
        if total < 1e-6:
            return 0.0
        return float(np.sum(freqs * spectrum) / total)

    def _classify(self, energy: float, pitch: float,
                  spectral: float, duration: float) -> tuple[str, float]:
        """Classify emotion from acoustic features."""
        energy_ratio = energy / max(self._baseline_energy, 1.0)
        pitch_ratio = pitch / max(self._baseline_pitch, 1.0) if pitch > 0 else 1.0

        if energy_ratio > 1.8 and pitch_ratio > 1.3:
            return "excited", min(0.9, 0.5 + (energy_ratio - 1.5) * 0.2)
        if energy_ratio > 1.5 and spectral > 3000:
            return "stressed", min(0.85, 0.5 + (energy_ratio - 1.3) * 0.2)
        if energy_ratio > 1.4 and pitch_ratio > 1.2 and spectral > 2500:
            return "frustrated", min(0.8, 0.5 + (energy_ratio - 1.2) * 0.2)
        if energy_ratio < 0.6 and pitch_ratio < 0.8:
            return "tired", min(0.8, 0.5 + (1.0 - energy_ratio) * 0.3)
        if pitch_ratio > 1.1 and energy_ratio > 0.9 and spectral > 2000:
            return "happy", 0.6
        if energy_ratio < 0.8 and pitch_ratio < 0.95:
            return "calm", 0.5

        return "neutral", 0.5

    def get_emotion_summary(self) -> str:
        if not self._history:
            return "No emotion data yet."
        recent = self._history[-5:]
        emotions = [r.emotion for r in recent]
        dominant = max(set(emotions), key=emotions.count)
        return f"Recent emotional state: {dominant} (based on last {len(recent)} readings)"
