"""
ATOM -- Audio Preprocessor (JARVIS-Level Input Conditioning).

Before ANY audio reaches the STT engine, it passes through this
preprocessor. Raw microphone audio is noisy, inconsistent, and
unpredictable. This module transforms it into clean, normalized,
speech-optimized audio.

Pipeline:
    Raw PCM -> DC Offset Removal -> Pre-Emphasis -> Spectral Noise Gate
    -> Normalization -> Energy Check -> Quality Score -> Clean PCM

Features:
    1. DC OFFSET REMOVAL -- Centers waveform around zero (fixes mic bias)
    2. PRE-EMPHASIS -- Boosts high frequencies for clearer consonants
    3. SPECTRAL NOISE GATE -- FFT-based noise floor subtraction
       (learns ambient noise profile, subtracts it from speech)
    4. PEAK NORMALIZATION -- Consistent volume regardless of mic gain
    5. ENERGY-BASED SILENCE DETECTION -- Rejects pure silence/very low energy
    6. CLIPPING DETECTION -- Warns if audio is clipped (mic gain too high)
    7. AUDIO QUALITY SCORING -- 0.0 to 1.0 assessment of input quality
    8. AMBIENT NOISE PROFILING -- Learns and adapts to environment

All operations use numpy only (no extra dependencies beyond what
faster-whisper already requires).

Owner: Satyam
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger("atom.audio_preprocess")

_SAMPLE_RATE = 16000
_PRE_EMPHASIS_COEFF = 0.97
_NOISE_GATE_FACTOR = 1.5
_NOISE_SMOOTHING_ALPHA = 0.85
_NORMALIZATION_TARGET = 0.85
_SILENCE_ENERGY_THRESHOLD = 0.005
_CLIPPING_THRESHOLD = 0.98
_MIN_SPEECH_ENERGY = 0.01
_NOISE_PROFILE_FRAMES = 8


@dataclass
class AudioQuality:
    """Quality assessment of a single audio chunk."""
    score: float = 0.0
    energy_rms: float = 0.0
    peak_level: float = 0.0
    snr_db: float = 0.0
    is_clipped: bool = False
    is_silence: bool = False
    is_speech_likely: bool = False
    noise_floor: float = 0.0
    duration_s: float = 0.0


@dataclass
class NoiseProfile:
    """Learned ambient noise characteristics."""
    spectral_floor: np.ndarray | None = None
    energy_floor: float = 0.0
    sample_count: int = 0
    last_updated: float = 0.0
    is_stable: bool = False


class AudioPreprocessor:
    """Production-grade audio conditioning for voice input.

    Usage:
        preprocessor = AudioPreprocessor()
        preprocessor.learn_noise(ambient_audio_samples)
        clean_audio, quality = preprocessor.process(raw_pcm_int16)
    """

    __slots__ = (
        "_noise_profile", "_config", "_enabled",
        "_pre_emphasis", "_normalize", "_spectral_gate",
        "_dc_remove", "_process_count", "_total_process_ms",
    )

    def __init__(self, config: dict | None = None) -> None:
        cfg = (config or {}).get("audio_preprocessor", {})
        self._config = cfg
        self._enabled = cfg.get("enabled", True)
        self._pre_emphasis = cfg.get("pre_emphasis", True)
        self._normalize = cfg.get("normalize", True)
        self._spectral_gate = cfg.get("spectral_gate", True)
        self._dc_remove = cfg.get("dc_remove", True)
        self._noise_profile = NoiseProfile()
        self._process_count = 0
        self._total_process_ms = 0.0

    # ── Public API ───────────────────────────────────────────────────

    def process(self, raw_pcm_int16: bytes | np.ndarray,
                sample_rate: int = _SAMPLE_RATE) -> tuple[np.ndarray, AudioQuality]:
        """Process raw PCM audio through the full conditioning pipeline.

        Args:
            raw_pcm_int16: Raw PCM audio as bytes (int16) or numpy array
            sample_rate: Sample rate (default 16000)

        Returns:
            (clean_float32_audio, quality_assessment)
        """
        t0 = time.perf_counter()

        if isinstance(raw_pcm_int16, bytes):
            samples = np.frombuffer(raw_pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0
        elif raw_pcm_int16.dtype == np.int16:
            samples = raw_pcm_int16.astype(np.float32) / 32768.0
        else:
            samples = raw_pcm_int16.astype(np.float32)

        if len(samples) == 0:
            return samples, AudioQuality(is_silence=True)

        quality = self._assess_raw_quality(samples, sample_rate)

        if not self._enabled:
            self._track_perf(t0)
            return samples, quality

        if quality.is_silence:
            self._track_perf(t0)
            return samples, quality

        if self._dc_remove:
            samples = self._remove_dc_offset(samples)

        if self._pre_emphasis:
            samples = self._apply_pre_emphasis(samples)

        if self._spectral_gate and self._noise_profile.is_stable:
            samples = self._apply_spectral_gate(samples)

        if self._normalize:
            samples = self._peak_normalize(samples)

        quality = self._assess_processed_quality(samples, quality, sample_rate)
        self._track_perf(t0)

        return samples, quality

    def learn_noise(self, raw_pcm_int16: bytes | np.ndarray,
                    sample_rate: int = _SAMPLE_RATE) -> None:
        """Learn ambient noise profile from a sample of silence/background.

        Call this during calibration with 1-2 seconds of ambient audio.
        The spectral profile will be used for noise subtraction.
        """
        if isinstance(raw_pcm_int16, bytes):
            samples = np.frombuffer(raw_pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0
        elif raw_pcm_int16.dtype == np.int16:
            samples = raw_pcm_int16.astype(np.float32) / 32768.0
        else:
            samples = raw_pcm_int16.astype(np.float32)

        if len(samples) < 1600:
            return

        frame_size = 1024
        hop = 512
        n_frames = max(1, (len(samples) - frame_size) // hop)
        window = np.hanning(frame_size).astype(np.float32)

        spectra = []
        for i in range(min(n_frames, 32)):
            start = i * hop
            frame = samples[start:start + frame_size]
            if len(frame) < frame_size:
                break
            windowed = frame * window
            spectrum = np.abs(np.fft.rfft(windowed))
            spectra.append(spectrum)

        if not spectra:
            return

        mean_spectrum = np.mean(spectra, axis=0).astype(np.float32)
        energy = float(np.sqrt(np.mean(samples ** 2)))

        profile = self._noise_profile
        if profile.spectral_floor is not None and profile.sample_count > 0:
            alpha = _NOISE_SMOOTHING_ALPHA
            profile.spectral_floor = (
                alpha * profile.spectral_floor + (1 - alpha) * mean_spectrum
            )
            profile.energy_floor = alpha * profile.energy_floor + (1 - alpha) * energy
        else:
            profile.spectral_floor = mean_spectrum
            profile.energy_floor = energy

        profile.sample_count += 1
        profile.last_updated = time.time()
        profile.is_stable = profile.sample_count >= 2

        logger.info(
            "Noise profile updated (sample %d): energy_floor=%.4f, stable=%s",
            profile.sample_count, profile.energy_floor, profile.is_stable,
        )

    def reset_noise_profile(self) -> None:
        """Reset the learned noise profile (e.g. after environment change)."""
        self._noise_profile = NoiseProfile()
        logger.info("Noise profile reset")

    def get_stats(self) -> dict:
        """Get preprocessing statistics."""
        avg_ms = (
            self._total_process_ms / self._process_count
            if self._process_count > 0 else 0
        )
        return {
            "process_count": self._process_count,
            "avg_process_ms": round(avg_ms, 2),
            "noise_profile_stable": self._noise_profile.is_stable,
            "noise_energy_floor": round(self._noise_profile.energy_floor, 4),
            "enabled": self._enabled,
        }

    # ── Processing Stages ────────────────────────────────────────────

    @staticmethod
    def _remove_dc_offset(samples: np.ndarray) -> np.ndarray:
        """Remove DC bias from the signal."""
        return samples - np.mean(samples)

    @staticmethod
    def _apply_pre_emphasis(samples: np.ndarray) -> np.ndarray:
        """Boost high frequencies for clearer consonants (s, t, k, etc.)."""
        emphasized = np.empty_like(samples)
        emphasized[0] = samples[0]
        emphasized[1:] = samples[1:] - _PRE_EMPHASIS_COEFF * samples[:-1]
        return emphasized

    def _apply_spectral_gate(self, samples: np.ndarray) -> np.ndarray:
        """Subtract learned noise spectrum from the signal.

        Uses overlap-add with Hanning window for artifact-free output.
        """
        profile = self._noise_profile
        if profile.spectral_floor is None:
            return samples

        frame_size = 1024
        hop = 512
        window = np.hanning(frame_size).astype(np.float32)
        n_frames = max(1, (len(samples) - frame_size) // hop)

        output = np.zeros(len(samples), dtype=np.float32)
        window_sum = np.zeros(len(samples), dtype=np.float32)

        noise_floor = profile.spectral_floor
        gate_factor = _NOISE_GATE_FACTOR

        for i in range(n_frames):
            start = i * hop
            end = start + frame_size
            if end > len(samples):
                break

            frame = samples[start:end] * window
            spectrum = np.fft.rfft(frame)
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)

            noise_scaled = noise_floor[:len(magnitude)] * gate_factor
            clean_mag = np.maximum(magnitude - noise_scaled, magnitude * 0.05)

            clean_spectrum = clean_mag * np.exp(1j * phase)
            clean_frame = np.fft.irfft(clean_spectrum, n=frame_size).astype(np.float32)

            output[start:end] += clean_frame * window
            window_sum[start:end] += window ** 2

        mask = window_sum > 1e-8
        output[mask] /= window_sum[mask]

        remaining = samples[n_frames * hop:]
        if len(remaining) > 0:
            output[n_frames * hop:n_frames * hop + len(remaining)] = remaining

        return output

    @staticmethod
    def _peak_normalize(samples: np.ndarray) -> np.ndarray:
        """Normalize audio to consistent peak level."""
        peak = np.max(np.abs(samples))
        if peak < 1e-6:
            return samples
        return samples * (_NORMALIZATION_TARGET / peak)

    # ── Quality Assessment ───────────────────────────────────────────

    def _assess_raw_quality(self, samples: np.ndarray,
                            sample_rate: int) -> AudioQuality:
        """Assess the raw audio quality before processing."""
        q = AudioQuality()
        q.duration_s = len(samples) / sample_rate
        q.energy_rms = float(np.sqrt(np.mean(samples ** 2)))
        q.peak_level = float(np.max(np.abs(samples)))
        q.is_clipped = q.peak_level > _CLIPPING_THRESHOLD
        q.is_silence = q.energy_rms < _SILENCE_ENERGY_THRESHOLD
        q.noise_floor = self._noise_profile.energy_floor

        if q.energy_rms > 0 and self._noise_profile.energy_floor > 0:
            signal_power = q.energy_rms ** 2
            noise_power = self._noise_profile.energy_floor ** 2
            if noise_power > 0:
                snr = signal_power / noise_power
                q.snr_db = float(10 * np.log10(max(snr, 1e-10)))
            else:
                q.snr_db = 40.0
        else:
            q.snr_db = 0.0

        q.is_speech_likely = (
            q.energy_rms > _MIN_SPEECH_ENERGY
            and not q.is_silence
            and q.snr_db > 3.0
        )

        return q

    def _assess_processed_quality(self, samples: np.ndarray,
                                  raw_quality: AudioQuality,
                                  sample_rate: int) -> AudioQuality:
        """Assess quality after processing and compute final score."""
        q = raw_quality
        q.energy_rms = float(np.sqrt(np.mean(samples ** 2)))
        q.peak_level = float(np.max(np.abs(samples)))

        score = 0.0

        if q.is_silence:
            q.score = 0.0
            return q

        if q.snr_db > 20:
            score += 0.3
        elif q.snr_db > 10:
            score += 0.2
        elif q.snr_db > 5:
            score += 0.1

        if 0.1 < q.energy_rms < 0.9:
            score += 0.2
        elif q.energy_rms > 0.02:
            score += 0.1

        if not q.is_clipped:
            score += 0.15

        if 0.3 <= q.duration_s <= 10.0:
            score += 0.2
        elif q.duration_s > 0.5:
            score += 0.1

        if q.is_speech_likely:
            score += 0.15

        q.score = min(1.0, score)
        return q

    # ── Internal ─────────────────────────────────────────────────────

    def _track_perf(self, t0: float) -> None:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._process_count += 1
        self._total_process_ms += elapsed_ms
        if self._process_count <= 3 or self._process_count % 50 == 0:
            logger.debug("Audio preprocess: %.1fms", elapsed_ms)
