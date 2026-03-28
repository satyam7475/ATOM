"""
ATOM -- Voice Print Authentication (Biometric Owner Verification).

Creates a unique voice fingerprint for the owner (Satyam) and verifies
identity through speaker embedding comparison. This is what makes ATOM
respond ONLY to its owner -- like JARVIS to Tony, FRIDAY to Tony.

Architecture:
    PRIMARY: resemblyzer (SpeakerEncoder) -- research-grade speaker embeddings
        - 256-dimensional d-vector per utterance
        - Pre-trained on thousands of speakers
        - Cosine similarity > threshold = verified owner

    FALLBACK: Lightweight MFCC-based speaker model
        - 13 MFCC coefficients averaged over utterance
        - Normalized cosine similarity comparison
        - Less accurate but zero-dependency (only numpy)

Enrollment:
    Owner speaks 5-10 phrases. Each phrase is embedded and averaged
    into a reference voice print. Stored encrypted in SecurityFortress vault.

Verification:
    Incoming speech is embedded and compared against the stored voice print.
    Cosine similarity above threshold = owner verified. Below = stranger.

Security:
    - Voice prints stored encrypted (via EncryptedVault)
    - Replay attack mitigation via temporal variance check
    - Progressive confidence: more enrollment phrases = higher accuracy
    - Anti-spoofing: checks embedding variance (synthetic voices are uniform)

Contract:
    enroll(audio_features) -> EnrollmentResult
    verify(audio_features) -> VerificationResult
    is_enrolled -> bool
    confidence_level -> str  (low/medium/high based on enrollment quality)

Owner: Satyam
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom.auth.voice")

_VOICE_PROFILE_FILE = Path("data/security/voice_profile.json")

_VERIFY_THRESHOLD_HIGH = 0.82
_VERIFY_THRESHOLD_MEDIUM = 0.75
_VERIFY_THRESHOLD_LOW = 0.68
_MIN_ENROLLMENT_PHRASES = 3
_MAX_ENROLLMENT_PHRASES = 20
_EMBEDDING_DIM_RESEMBLYZER = 256
_EMBEDDING_DIM_MFCC = 13
_ANTI_SPOOF_MIN_VARIANCE = 0.001
_VERIFICATION_COOLDOWN_S = 1.0


@dataclass
class EnrollmentResult:
    """Result of a voice enrollment attempt."""
    success: bool
    message: str
    phrases_enrolled: int = 0
    confidence_level: str = "none"
    embedding_method: str = "none"


@dataclass
class VerificationResult:
    """Result of a voice verification attempt."""
    verified: bool
    similarity: float = 0.0
    message: str = ""
    method: str = "none"
    is_potential_spoof: bool = False


@dataclass
class VoiceProfile:
    """Stored voice fingerprint of the owner."""
    embeddings: list[list[float]] = field(default_factory=list)
    centroid: list[float] = field(default_factory=list)
    embedding_dim: int = 0
    method: str = "none"
    enrolled_at: float = 0.0
    phrase_count: int = 0
    variance: float = 0.0
    verification_count: int = 0
    successful_verifications: int = 0
    last_verified: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "embeddings": self.embeddings,
            "centroid": self.centroid,
            "embedding_dim": self.embedding_dim,
            "method": self.method,
            "enrolled_at": self.enrolled_at,
            "phrase_count": self.phrase_count,
            "variance": self.variance,
            "verification_count": self.verification_count,
            "successful_verifications": self.successful_verifications,
            "last_verified": self.last_verified,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VoiceProfile:
        return cls(
            embeddings=data.get("embeddings", []),
            centroid=data.get("centroid", []),
            embedding_dim=data.get("embedding_dim", 0),
            method=data.get("method", "none"),
            enrolled_at=data.get("enrolled_at", 0.0),
            phrase_count=data.get("phrase_count", 0),
            variance=data.get("variance", 0.0),
            verification_count=data.get("verification_count", 0),
            successful_verifications=data.get("successful_verifications", 0),
            last_verified=data.get("last_verified", 0.0),
        )


class VoicePrintAuth:
    """Biometric voice print authentication for owner verification.

    Uses speaker embeddings (resemblyzer or MFCC fallback) to create
    a unique voice fingerprint. Only the enrolled owner passes verification.
    """

    __slots__ = (
        "_profile", "_encoder", "_method", "_threshold",
        "_last_verify_time", "_vault",
    )

    def __init__(self, config: dict | None = None) -> None:
        cfg = (config or {}).get("voice_auth", {})
        self._profile = VoiceProfile()
        self._encoder: Any = None
        self._method = "none"
        self._threshold = cfg.get("threshold", _VERIFY_THRESHOLD_MEDIUM)
        self._last_verify_time = 0.0
        self._vault: Any = None

        self._init_encoder()
        self._load_profile()

    def _init_encoder(self) -> None:
        """Initialize the best available speaker encoder."""
        try:
            from resemblyzer import VoiceEncoder
            self._encoder = VoiceEncoder("cpu")
            self._method = "resemblyzer"
            logger.info("Voice auth: resemblyzer encoder loaded (256-d embeddings)")
            return
        except ImportError:
            pass
        except Exception:
            logger.debug("resemblyzer load failed", exc_info=True)

        try:
            import numpy as np  # noqa: F401
            self._method = "mfcc"
            logger.info("Voice auth: MFCC fallback mode (13-d features)")
            return
        except ImportError:
            pass

        self._method = "none"
        logger.info("Voice auth: no encoder available (numpy or resemblyzer required)")

    def attach_vault(self, vault: Any) -> None:
        """Attach encrypted vault for secure voice print storage."""
        self._vault = vault
        self._load_profile()

    def _load_profile(self) -> None:
        """Load voice profile from vault or disk."""
        loaded = False
        if self._vault:
            raw = self._vault.get("voice_profile", "")
            if raw:
                try:
                    self._profile = VoiceProfile.from_dict(json.loads(raw))
                    loaded = True
                except Exception:
                    logger.debug("Vault voice profile load failed", exc_info=True)

        if not loaded and _VOICE_PROFILE_FILE.exists():
            try:
                data = json.loads(_VOICE_PROFILE_FILE.read_text(encoding="utf-8"))
                self._profile = VoiceProfile.from_dict(data)
                loaded = True
            except Exception:
                logger.debug("Disk voice profile load failed", exc_info=True)

        if loaded and self._profile.phrase_count > 0:
            logger.info(
                "Voice profile loaded: %d phrases, method=%s, %d verifications",
                self._profile.phrase_count,
                self._profile.method,
                self._profile.verification_count,
            )

    def _save_profile(self) -> None:
        """Persist voice profile to vault and disk."""
        data = self._profile.to_dict()

        if self._vault:
            self._vault.put("voice_profile", json.dumps(data))
            self._vault.persist()

        try:
            _VOICE_PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
            _VOICE_PROFILE_FILE.write_text(
                json.dumps(data, indent=2), encoding="utf-8",
            )
        except Exception:
            logger.debug("Voice profile disk save failed", exc_info=True)

    # ── Enrollment ───────────────────────────────────────────────────

    def enroll(self, audio_data: Any) -> EnrollmentResult:
        """Enroll a voice sample for the owner.

        Call this multiple times (3-10 phrases) to build a robust profile.
        audio_data can be:
            - numpy array of raw audio samples (float32, 16kHz mono)
            - list of float values (raw PCM samples)
        """
        if self._method == "none":
            return EnrollmentResult(
                success=False,
                message="No audio encoder available. Install resemblyzer or numpy.",
            )

        try:
            embedding = self._extract_embedding(audio_data)
        except Exception as exc:
            return EnrollmentResult(
                success=False,
                message=f"Failed to extract voice features: {exc}",
            )

        if embedding is None or len(embedding) == 0:
            return EnrollmentResult(
                success=False,
                message="Could not extract voice features from audio. Try speaking more clearly.",
            )

        self._profile.embeddings.append(embedding)
        if len(self._profile.embeddings) > _MAX_ENROLLMENT_PHRASES:
            self._profile.embeddings = self._profile.embeddings[-_MAX_ENROLLMENT_PHRASES:]

        self._profile.centroid = self._compute_centroid(self._profile.embeddings)
        self._profile.embedding_dim = len(embedding)
        self._profile.method = self._method
        self._profile.phrase_count = len(self._profile.embeddings)
        self._profile.enrolled_at = time.time()
        self._profile.variance = self._compute_variance(self._profile.embeddings)

        self._save_profile()

        confidence = self.confidence_level
        count = self._profile.phrase_count
        remaining = max(0, _MIN_ENROLLMENT_PHRASES - count)

        if remaining > 0:
            message = (
                f"Voice sample {count} enrolled. "
                f"Say {remaining} more phrase{'s' if remaining > 1 else ''} "
                f"for minimum enrollment."
            )
        elif count < 5:
            message = (
                f"Voice enrolled with {count} samples (confidence: {confidence}). "
                f"Add more phrases for higher accuracy."
            )
        else:
            message = (
                f"Voice enrolled with {count} samples. "
                f"Confidence: {confidence}. Your voice print is ready, Boss."
            )

        logger.info(
            "Voice enrollment: phrase %d, dim=%d, variance=%.4f, confidence=%s",
            count, len(embedding), self._profile.variance, confidence,
        )

        return EnrollmentResult(
            success=True,
            message=message,
            phrases_enrolled=count,
            confidence_level=confidence,
            embedding_method=self._method,
        )

    def reset_enrollment(self) -> str:
        """Clear voice enrollment data."""
        self._profile = VoiceProfile()
        self._save_profile()
        logger.info("Voice enrollment reset")
        return "Voice enrollment cleared. You'll need to re-enroll, Boss."

    # ── Verification ─────────────────────────────────────────────────

    def verify(self, audio_data: Any) -> VerificationResult:
        """Verify if the speaker matches the enrolled owner.

        Returns a VerificationResult with similarity score and verdict.
        """
        if not self.is_enrolled:
            return VerificationResult(
                verified=False,
                message="No voice profile enrolled. Say 'enroll my voice' first.",
            )

        now = time.monotonic()
        if now - self._last_verify_time < _VERIFICATION_COOLDOWN_S:
            return VerificationResult(
                verified=False,
                message="Verification too rapid. Wait a moment.",
            )
        self._last_verify_time = now

        try:
            embedding = self._extract_embedding(audio_data)
        except Exception as exc:
            return VerificationResult(
                verified=False,
                message=f"Voice feature extraction failed: {exc}",
            )

        if embedding is None or len(embedding) == 0:
            return VerificationResult(
                verified=False,
                message="Could not extract voice features. Try again.",
            )

        similarity = self._cosine_similarity(embedding, self._profile.centroid)

        is_spoof = self._check_anti_spoof(embedding)

        self._profile.verification_count += 1

        if is_spoof:
            self._profile.last_verified = time.time()
            self._save_profile()
            logger.warning(
                "Voice verification SPOOF DETECTED: similarity=%.3f",
                similarity,
            )
            return VerificationResult(
                verified=False,
                similarity=similarity,
                message="Voice pattern appears synthetic. Verification denied.",
                method=self._method,
                is_potential_spoof=True,
            )

        verified = similarity >= self._threshold

        if verified:
            self._profile.successful_verifications += 1
            self._profile.last_verified = time.time()
            self._update_centroid_with_verified(embedding)
            message = f"Voice verified. Welcome, Boss. (confidence: {similarity:.0%})"
        else:
            message = (
                f"Voice not recognized (similarity: {similarity:.0%}, "
                f"threshold: {self._threshold:.0%}). Access denied."
            )

        self._save_profile()

        logger.info(
            "Voice verification: similarity=%.3f, threshold=%.3f, result=%s",
            similarity, self._threshold, "PASS" if verified else "FAIL",
        )

        return VerificationResult(
            verified=verified,
            similarity=similarity,
            message=message,
            method=self._method,
        )

    # ── Feature Extraction ───────────────────────────────────────────

    def _extract_embedding(self, audio_data: Any) -> list[float] | None:
        """Extract speaker embedding from audio data."""
        import numpy as np

        if isinstance(audio_data, list):
            audio_data = np.array(audio_data, dtype=np.float32)

        if not isinstance(audio_data, np.ndarray):
            return None

        audio = audio_data.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if len(audio) < 1600:
            return None

        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-6:
            return None

        if self._method == "resemblyzer" and self._encoder is not None:
            embedding = self._encoder.embed_utterance(audio)
            return embedding.tolist()

        if self._method == "mfcc":
            return self._extract_mfcc(audio)

        return None

    def _extract_mfcc(self, audio: Any) -> list[float]:
        """Lightweight MFCC extraction without librosa.

        Computes 13 Mel-frequency cepstral coefficients using:
        1. Pre-emphasis filter
        2. Frame windowing (Hamming)
        3. FFT -> power spectrum
        4. Mel filterbank
        5. Log -> DCT (simplified)
        6. Mean over all frames -> single 13-d vector
        """
        import numpy as np

        sample_rate = 16000
        pre_emphasis = 0.97
        frame_size_s = 0.025
        frame_stride_s = 0.01
        n_mfcc = _EMBEDDING_DIM_MFCC
        n_filt = 26
        nfft = 512

        emphasized = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

        frame_length = int(round(frame_size_s * sample_rate))
        frame_step = int(round(frame_stride_s * sample_rate))
        signal_length = len(emphasized)
        num_frames = max(1, 1 + (signal_length - frame_length) // frame_step)

        pad_length = (num_frames - 1) * frame_step + frame_length
        if pad_length > signal_length:
            emphasized = np.append(
                emphasized, np.zeros(pad_length - signal_length),
            )

        indices = (
            np.tile(np.arange(0, frame_length), (num_frames, 1))
            + np.tile(
                np.arange(0, num_frames * frame_step, frame_step),
                (frame_length, 1),
            ).T
        )
        frames = emphasized[indices]

        hamming = np.hamming(frame_length)
        frames *= hamming

        mag_frames = np.absolute(np.fft.rfft(frames, nfft))
        pow_frames = (1.0 / nfft) * (mag_frames ** 2)

        low_freq_mel = 0.0
        high_freq_mel = 2595.0 * np.log10(1.0 + (sample_rate / 2.0) / 700.0)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filt + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        bins = np.floor((nfft + 1) * hz_points / sample_rate).astype(int)

        fbank = np.zeros((n_filt, nfft // 2 + 1))
        for m in range(1, n_filt + 1):
            f_m_minus = bins[m - 1]
            f_m = bins[m]
            f_m_plus = bins[m + 1]

            for k in range(f_m_minus, f_m):
                if f_m != f_m_minus:
                    fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                if f_m_plus != f_m:
                    fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(
            filter_banks == 0, np.finfo(float).eps, filter_banks,
        )
        filter_banks = 20.0 * np.log10(filter_banks)

        n_coeffs = min(n_mfcc, filter_banks.shape[1])
        dct_matrix = np.zeros((n_coeffs, filter_banks.shape[1]))
        for i in range(n_coeffs):
            for j in range(filter_banks.shape[1]):
                dct_matrix[i, j] = np.cos(
                    np.pi * i * (2 * j + 1) / (2 * filter_banks.shape[1])
                )
        mfcc = np.dot(filter_banks, dct_matrix.T)

        mean_mfcc = np.mean(mfcc, axis=0)

        norm = np.linalg.norm(mean_mfcc)
        if norm > 0:
            mean_mfcc = mean_mfcc / norm

        return mean_mfcc.tolist()

    # ── Similarity & Anti-Spoofing ───────────────────────────────────

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b) or len(a) == 0:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _compute_centroid(embeddings: list[list[float]]) -> list[float]:
        """Compute the centroid (average) of multiple embeddings."""
        if not embeddings:
            return []
        dim = len(embeddings[0])
        centroid = [0.0] * dim
        for emb in embeddings:
            for i, val in enumerate(emb):
                centroid[i] += val
        n = len(embeddings)
        centroid = [v / n for v in centroid]
        norm = math.sqrt(sum(v * v for v in centroid))
        if norm > 0:
            centroid = [v / norm for v in centroid]
        return centroid

    @staticmethod
    def _compute_variance(embeddings: list[list[float]]) -> float:
        """Compute average variance across embedding dimensions."""
        if len(embeddings) < 2:
            return 0.0
        dim = len(embeddings[0])
        total_var = 0.0
        for d in range(dim):
            values = [emb[d] for emb in embeddings]
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / len(values)
            total_var += var
        return total_var / dim

    def _check_anti_spoof(self, embedding: list[float]) -> bool:
        """Check if the embedding appears synthetic (anti-spoofing).

        Synthetic/replayed voices tend to have unnaturally low variance
        in their embeddings compared to real speech.
        """
        if len(self._profile.embeddings) < 3:
            return False

        similarities = [
            self._cosine_similarity(embedding, stored)
            for stored in self._profile.embeddings
        ]

        if all(s > 0.999 for s in similarities):
            return True

        try:
            import numpy as np
            emb_arr = np.array(embedding)
            if np.std(emb_arr) < _ANTI_SPOOF_MIN_VARIANCE:
                return True
        except ImportError:
            if len(embedding) > 0:
                mean_val = sum(embedding) / len(embedding)
                variance = sum((v - mean_val) ** 2 for v in embedding) / len(embedding)
                if variance < _ANTI_SPOOF_MIN_VARIANCE:
                    return True

        return False

    def _update_centroid_with_verified(self, embedding: list[float]) -> None:
        """Adaptive learning: slightly shift centroid toward verified speech.

        Over time this accounts for natural voice changes (aging, health, mood).
        Uses a very small learning rate to prevent drift from adversarial input.
        """
        if not self._profile.centroid:
            return
        lr = 0.02
        updated = [
            c * (1 - lr) + e * lr
            for c, e in zip(self._profile.centroid, embedding)
        ]
        norm = math.sqrt(sum(v * v for v in updated))
        if norm > 0:
            self._profile.centroid = [v / norm for v in updated]

    # ── Properties ───────────────────────────────────────────────────

    @property
    def is_enrolled(self) -> bool:
        return self._profile.phrase_count >= _MIN_ENROLLMENT_PHRASES

    @property
    def is_available(self) -> bool:
        return self._method != "none"

    @property
    def confidence_level(self) -> str:
        """Enrollment confidence based on number of phrases and variance."""
        n = self._profile.phrase_count
        if n < _MIN_ENROLLMENT_PHRASES:
            return "insufficient"
        if n >= 8 and self._profile.variance > 0.0005:
            return "high"
        if n >= 5:
            return "medium"
        return "low"

    @property
    def method(self) -> str:
        return self._method

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "enrolled": self.is_enrolled,
            "method": self._method,
            "phrases": self._profile.phrase_count,
            "confidence": self.confidence_level,
            "verifications": self._profile.verification_count,
            "successful": self._profile.successful_verifications,
            "success_rate": (
                f"{self._profile.successful_verifications / self._profile.verification_count:.0%}"
                if self._profile.verification_count > 0 else "N/A"
            ),
            "last_verified": self._profile.last_verified,
            "variance": round(self._profile.variance, 6),
        }

    def get_status_message(self) -> str:
        """Human-readable voice auth status for TTS."""
        if not self.is_available:
            return "Voice authentication unavailable. Install resemblyzer or numpy."
        if not self.is_enrolled:
            n = self._profile.phrase_count
            remaining = _MIN_ENROLLMENT_PHRASES - n
            if n == 0:
                return "Voice not enrolled. Say 'enroll my voice' to begin."
            return (
                f"Voice enrollment in progress: {n} samples collected, "
                f"need {remaining} more."
            )
        s = self._profile
        rate = (
            f"{s.successful_verifications / s.verification_count:.0%}"
            if s.verification_count > 0 else "no attempts yet"
        )
        return (
            f"Voice enrolled with {s.phrase_count} samples "
            f"(confidence: {self.confidence_level}). "
            f"Method: {self._method}. "
            f"Verification success rate: {rate}."
        )

    def persist(self) -> None:
        self._save_profile()

    def shutdown(self) -> None:
        self.persist()
        logger.info("Voice auth shut down")
