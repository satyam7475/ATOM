"""voice.whisper_confirmer — v3 Phase 4: high-accuracy second-pass STT.

Why this exists
---------------
The streaming STT (macOS Speech / SFSpeechRecognizer) is fast (sub-300ms
final) but mistranscribes short, low-context utterances under noise. Real
field logs show patterns like:

  * "" / "."  -- empty or single-char finals (mic hiccup)
  * "uh"     -- noise classified as a token
  * "hire"   when the user said "hi"
  * mixed Hindi-English where the streaming engine downranks the wrong
    language

Whisper-small/tiny (faster-whisper int8 on CPU) re-decodes the same
~3-4 second audio window in 80-180ms with materially lower WER on these
exact cases. Running it on EVERY final would double STT latency, so we
only call it when the streaming output is suspect.

Public API
----------
``WhisperConfirmer.feed_audio(samples, sample_rate)``
    Append a PCM mono float32 chunk to the rolling ring buffer. Safe to
    call from the audio-tap thread.

``WhisperConfirmer.confirm(text, confidence, language=None) -> ConfirmResult``
    Inspect ``text``/``confidence``. If suspect, run Whisper on the last
    N seconds of the ring buffer and return the corrected text. If not
    suspect, return the original. Synchronous, intended to be invoked
    from ``_on_final``.

Wiring is *opt-in* via ``config["stt"]["whisper_confirm"]``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger("atom.voice.whisper_confirmer")

# ── Configuration defaults (kept on this constant so tests can introspect) ──
_DEFAULTS = {
    "enabled": False,
    "model_size": "tiny",          # tiny is ~75MB, ~80ms per 4s clip
    "ring_seconds": 5.0,           # how much rolling audio we keep
    "decode_seconds": 4.0,         # how much we hand to Whisper
    "min_confidence": 0.55,        # below this, force a re-decode
    "max_confirm_ms": 250.0,       # advisory budget — log warn if exceeded
    "language": None,              # None = auto-detect (multilingual)
    "min_text_chars": 3,           # finals shorter than this look suspect
}

# ── Suspect-text fingerprints ──────────────────────────────────────────
# Single-token / noise-token finals the streaming engine sometimes emits.
_NOISE_TOKENS = frozenset({
    "", ".", "..", "...", "uh", "um", "eh", "hm", "hmm",
    "ah", "oh", "huh", "mm", "mhm", "okay.", "ok.", "yeah.",
})


@dataclass
class ConfirmResult:
    """Result of a confirmation pass.

    Attributes
    ----------
    text:
        Final text -- either the original (when not suspect or when
        Whisper unavailable) or the Whisper re-decode.
    used_whisper:
        True iff Whisper actually ran.
    elapsed_ms:
        Wall-clock time spent in confirm(). Useful for telemetry.
    reason:
        One of "ok", "low_conf", "short", "noise", "whisper_failed",
        "whisper_unavailable". Drives logging + nightly eval.
    """

    text: str
    used_whisper: bool
    elapsed_ms: float
    reason: str


class _RingBuffer:
    """Thread-safe rolling float32 PCM buffer.

    We store as a flat ``bytearray`` of float32 little-endian samples to
    avoid per-chunk numpy allocations on the audio-tap hot path.
    """

    __slots__ = ("_buf", "_cap_bytes", "_lock", "_sample_rate", "_max_seconds")

    def __init__(self, max_seconds: float, sample_rate: int = 16000) -> None:
        self._max_seconds = float(max_seconds)
        self._sample_rate = int(sample_rate)
        self._cap_bytes = int(self._max_seconds * self._sample_rate * 4)
        self._buf = bytearray()
        self._lock = threading.Lock()

    def feed(self, samples_bytes: bytes) -> None:
        """Append raw float32-LE PCM bytes; trim to capacity."""
        if not samples_bytes:
            return
        with self._lock:
            self._buf.extend(samples_bytes)
            overflow = len(self._buf) - self._cap_bytes
            if overflow > 0:
                # Trim to a frame boundary (4 bytes per float32 sample)
                trim = overflow + (4 - overflow % 4) % 4
                del self._buf[:trim]

    def snapshot(self, seconds: float) -> bytes:
        """Return the most recent ``seconds`` of audio as float32-LE bytes."""
        want = int(max(0.0, seconds) * self._sample_rate * 4)
        with self._lock:
            if want >= len(self._buf):
                return bytes(self._buf)
            return bytes(self._buf[-want:])

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    def duration_s(self) -> float:
        with self._lock:
            return len(self._buf) / 4.0 / max(1, self._sample_rate)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate


class WhisperConfirmer:
    """High-accuracy second-pass STT used only when the streaming
    transcript looks suspect.

    The model is loaded lazily on the first ``confirm()`` that decides
    Whisper is needed -- not at construction -- so cold boot stays fast.
    """

    # Whisper expects 16kHz mono float32. Audio fed in at any other rate
    # is resampled inside _whisper_decode().
    _WHISPER_SR = 16000

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        sample_rate: int = 16000,
    ) -> None:
        self._cfg = dict(_DEFAULTS)
        self._cfg.update((config or {}).get("whisper_confirm", {}) or {})

        self._enabled = bool(self._cfg.get("enabled", False))
        self._model_size = str(self._cfg.get("model_size", "tiny"))
        self._ring_seconds = float(self._cfg.get("ring_seconds", 5.0))
        self._decode_seconds = float(self._cfg.get("decode_seconds", 4.0))
        self._min_confidence = float(self._cfg.get("min_confidence", 0.55))
        self._max_confirm_ms = float(self._cfg.get("max_confirm_ms", 250.0))
        self._language = self._cfg.get("language", None)
        self._min_text_chars = int(self._cfg.get("min_text_chars", 3))

        self._capture_sample_rate = int(sample_rate)
        self._ring = _RingBuffer(
            self._ring_seconds, sample_rate=self._capture_sample_rate,
        )
        self._model: Any = None
        self._model_lock = threading.Lock()
        self._model_unavailable = False  # one-shot: stop retrying after failure
        self._model_load_attempted = False

        self.calls = 0
        self.confirmed = 0
        self.last_elapsed_ms = 0.0

    # ── audio capture ──────────────────────────────────────────────

    def feed_audio(self, samples_bytes: bytes) -> None:
        """Push float32-LE PCM samples into the ring buffer.

        Must be safe to call from the audio-tap thread; the ring buffer
        does its own locking and never allocates while the lock is held.
        """
        if not self._enabled:
            return
        self._ring.feed(samples_bytes)

    def reset_audio(self) -> None:
        """Clear the ring buffer (e.g. after TTS cooldown)."""
        self._ring.clear()

    def set_sample_rate(self, sample_rate: int) -> None:
        """Update the capture sample rate. Resets the ring buffer because
        a stale buffer at the wrong rate would corrupt the next decode."""
        sr = int(sample_rate)
        if sr <= 0 or sr == self._capture_sample_rate:
            return
        self._capture_sample_rate = sr
        self._ring = _RingBuffer(self._ring_seconds, sample_rate=sr)

    # ── decision logic ─────────────────────────────────────────────

    def is_enabled(self) -> bool:
        return bool(self._enabled)

    def _is_suspect(
        self,
        text: str,
        confidence: float,
    ) -> tuple[bool, str]:
        """Return (suspect, reason)."""
        if text is None:
            return True, "noise"
        t = text.strip()
        if not t:
            return True, "noise"
        if t.lower() in _NOISE_TOKENS:
            return True, "noise"
        if len(t) < self._min_text_chars:
            return True, "short"
        try:
            conf_f = float(confidence)
        except (TypeError, ValueError):
            conf_f = 1.0
        if conf_f < self._min_confidence:
            return True, "low_conf"
        return False, "ok"

    # ── public confirm() ───────────────────────────────────────────

    def confirm(
        self,
        text: str,
        confidence: float = 1.0,
        *,
        language: Optional[str] = None,
    ) -> ConfirmResult:
        """Decide whether to re-decode; return the (possibly corrected) text.

        Synchronous: runs Whisper on the calling thread when needed.
        The caller (``_on_final`` in stt_macos) already runs off the
        audio thread, so this is fine.
        """
        t_start = time.perf_counter()
        self.calls += 1

        if not self._enabled:
            return ConfirmResult(text or "", False, 0.0, "ok")

        suspect, reason = self._is_suspect(text or "", confidence)
        if not suspect:
            return ConfirmResult(text, False, 0.0, "ok")

        model = self._get_model()
        if model is None:
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            self.last_elapsed_ms = elapsed_ms
            return ConfirmResult(
                text or "", False, elapsed_ms, "whisper_unavailable",
            )

        decoded = self._whisper_decode(
            model,
            language=language or self._language,
        )
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.last_elapsed_ms = elapsed_ms

        if elapsed_ms > self._max_confirm_ms:
            logger.warning(
                "WhisperConfirmer over budget: %.1fms (cap=%.1fms) reason=%s",
                elapsed_ms, self._max_confirm_ms, reason,
            )

        if decoded is None:
            return ConfirmResult(text or "", False, elapsed_ms, "whisper_failed")

        decoded = decoded.strip()
        if not decoded:
            # Sprint Ω.5.B (Apr 26 2026): a previous version trusted the
            # empty decode unconditionally -- which silently wiped real
            # user turns whenever Whisper happened to fail or the audio
            # ring buffer was thinner than the streaming utterance (e.g.
            # network mic, BT codec hiccup). The contract now is:
            #
            #   * suspect-by-noise / suspect-by-short  -> trust empty
            #     (the streaming output was a noise token to begin with)
            #   * suspect-by-low_conf with substantive original text
            #     -> KEEP THE ORIGINAL. Streaming had a real transcript;
            #     refusing to re-speak it just because Whisper couldn't
            #     decode the same audio drops the user.
            #
            # ``len(original) >= 2 * min_text_chars`` is the substance
            # heuristic -- one full short sentence ("turn off the lights"
            # is 4 chars * "turn" alone, but the whole utterance is well
            # over the threshold). Anything below that is short enough
            # we'd rather be safe and drop.
            self.confirmed += 1
            original_clean = (text or "").strip()
            keep_original = (
                reason == "low_conf"
                and len(original_clean) >= max(
                    self._min_text_chars * 2, 6,
                )
            )
            if keep_original:
                logger.info(
                    "WhisperConfirmer empty-decode on low_conf text -- "
                    "keeping original to avoid a silent drop: %r [%.1fms]",
                    original_clean[:60], elapsed_ms,
                )
                return ConfirmResult(
                    original_clean, False, elapsed_ms, reason,
                )
            logger.debug(
                "WhisperConfirmer collapsed suspect (%s) input %r to empty "
                "[%.1fms]", reason, original_clean[:60], elapsed_ms,
            )
            return ConfirmResult("", True, elapsed_ms, reason)

        self.confirmed += 1
        logger.info(
            "WhisperConfirmer corrected (%s): %r -> %r [%.1fms conf=%.2f]",
            reason, (text or "")[:60], decoded[:60], elapsed_ms, confidence,
        )
        return ConfirmResult(decoded, True, elapsed_ms, reason)

    # ── model lifecycle ────────────────────────────────────────────

    def unload(self) -> bool:
        """Drop the faster-whisper model from RAM.

        Sprint Ω.4.C (Apr 26 2026): exposed for the memory governor so a
        16 GB box can release the second-pass STT model under pressure.
        The next ``confirm()`` after ``unload()`` will re-load the model
        on its first invocation (back to cold-load latency, ~250 ms for
        the tiny model on CPU).

        Returns True iff a model was actually unloaded.
        """
        with self._model_lock:
            if self._model is None:
                return False
            self._model = None
            # Allow the next access to re-attempt even if a previous
            # cold-load had failed; the user just freed RAM, give it a
            # second chance.
            self._model_unavailable = False
            self._model_load_attempted = False
        logger.info(
            "WhisperConfirmer: model unloaded (memory governor)",
        )
        return True

    def _get_model(self) -> Any:
        """Lazily load faster-whisper. Once a load fails we stay off."""
        if self._model_unavailable:
            return None
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            if self._model_unavailable:
                return None
            self._model_load_attempted = True
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                logger.warning(
                    "WhisperConfirmer disabled: faster-whisper not installed.",
                )
                self._model_unavailable = True
                return None
            try:
                t0 = time.monotonic()
                self._model = WhisperModel(
                    self._model_size,
                    device="cpu",
                    compute_type="int8",
                )
                logger.info(
                    "WhisperConfirmer loaded model=%s (%.0fms cold)",
                    self._model_size,
                    (time.monotonic() - t0) * 1000.0,
                )
                return self._model
            except Exception:
                logger.exception(
                    "WhisperConfirmer model load failed -- disabling.",
                )
                self._model_unavailable = True
                return None

    def _whisper_decode(
        self,
        model: Any,
        *,
        language: Optional[str],
    ) -> Optional[str]:
        """Pull last N seconds from the ring buffer and run Whisper."""
        try:
            import numpy as np  # type: ignore
        except ImportError:
            logger.warning("numpy missing -- WhisperConfirmer cannot decode")
            return None

        raw = self._ring.snapshot(self._decode_seconds)
        if not raw:
            return None
        try:
            samples = np.frombuffer(raw, dtype=np.float32)
            if samples.size == 0:
                return None

            # Resample to 16kHz if the capture rate differs. Cheap linear
            # interp is enough for Whisper -- it does its own front-end
            # mel filtering and is robust to small interpolation noise.
            src_sr = self._ring.sample_rate
            if src_sr != self._WHISPER_SR:
                src_n = samples.size
                ratio = self._WHISPER_SR / float(src_sr)
                tgt_n = max(1, int(src_n * ratio))
                samples = np.interp(
                    np.linspace(0.0, src_n - 1, tgt_n, dtype=np.float64),
                    np.arange(src_n, dtype=np.float64),
                    samples.astype(np.float64),
                ).astype(np.float32)

            # faster-whisper accepts a mono float32 numpy array directly
            # when sampled at 16kHz.
            segments_iter, _info = model.transcribe(
                samples,
                language=language,
                vad_filter=False,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
            )
            chunks = []
            for seg in segments_iter:
                txt = getattr(seg, "text", "") or ""
                if txt:
                    chunks.append(txt)
            return "".join(chunks).strip()
        except Exception:
            logger.exception("WhisperConfirmer decode failed")
            return None

    # ── introspection (used by tests + nightly eval) ───────────────

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self._enabled,
            "calls": self.calls,
            "confirmed": self.confirmed,
            "last_elapsed_ms": round(self.last_elapsed_ms, 2),
            "ring_duration_s": round(self._ring.duration_s(), 2),
            "model_loaded": self._model is not None,
            "model_unavailable": self._model_unavailable,
            "model_size": self._model_size,
        }
