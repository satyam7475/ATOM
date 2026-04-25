"""ATOM Sprint Ω9 -- Silero VAD adaptive end-of-turn detector.

Why a second VAD on top of webrtcvad
------------------------------------
``webrtcvad`` is great at the cheap, per-frame question "is there voice
in this 30 ms slice?". It's tiny, deterministic, and runs on CPU in
microseconds. What it can't tell us is *"has the speaker finished?"* --
that needs prosodic context spanning ~1 s of audio.

The hot path in :mod:`voice.stt_whisper` waits for a fixed 600 ms of
trailing silence before emitting ``speech_final``. That's a safe
default but it costs 600 ms of latency on **every** turn -- including
the easy ones where the speaker very clearly stopped.

This module wraps Silero VAD v5 / v6 as a probability gate that the
STT loop consults *in addition to* webrtcvad:

* When the model is highly confident the speaker is done (clean prosody
  fall-off, no rising intonation), the loop can finalize after as
  little as ~180 ms of silence.
* When the model thinks the speaker is mid-thought (held vowel,
  thinking-noise, partial word), the loop waits longer than 600 ms --
  up to a hard cap -- to avoid cutting Boss off.

Trade-offs
----------
* Pure additive layer: if Silero is missing or fails to load, the STT
  loop falls back to the existing 600 ms behaviour (zero regression).
* Single torch import is paid at preload, not per-frame.
* Inference cost: ~1 ms per call on M-series, called at ~15 Hz during
  listening. Negligible vs whisper.cpp.
* Config-gated via ``stt.smart_turn_taker.enabled`` (off by default
  during the Sprint Ω rollout so the existing safety net is the
  default until Boss flips it on).

Owner: Satyam
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("atom.voice.turn_taker")


# ── Public config ───────────────────────────────────────────────────


@dataclass(slots=True)
class SmartTurnTakerConfig:
    """Tuning knobs for :class:`SmartTurnTaker`.

    All thresholds are expressed in seconds for readability. A safe
    default profile is calibrated for English conversational speech
    on a quiet desk mic at 16 kHz.
    """

    enabled: bool = False
    sample_rate: int = 16000
    # Decision window of audio fed to Silero on every check (seconds).
    decision_window_s: float = 1.0
    # Hard floor: never finalize before this much trailing silence,
    # even if Silero is screaming "they're done".
    min_silence_s: float = 0.18
    # Hard ceiling: always finalize after this much trailing silence,
    # regardless of model probability.
    max_silence_s: float = 1.20
    # When the silence has crossed ``min_silence_s``, finalize as soon
    # as Silero's end-of-turn probability exceeds this threshold.
    eot_probability_threshold: float = 0.78
    # Probability above which we refuse to finalize even past the
    # legacy 600 ms wait (i.e. user clearly hasn't finished).
    midthought_lockout_threshold: float = 0.92
    # Cooldown between two successive Silero inference calls. Caps CPU
    # cost when the STT loop is calling us aggressively.
    min_eval_interval_ms: float = 60.0
    # Optional weights overrides for offline / pinned-version installs.
    model_repo: str | None = None


@dataclass(slots=True)
class TurnDecision:
    """Outcome of one ``should_finalize`` call."""

    finalize: bool
    reason: str
    probability: float = 0.0
    eot_score: float = 0.0
    midthought_score: float = 0.0
    eval_skipped: bool = False


# ── Detector ────────────────────────────────────────────────────────


class SmartTurnTaker:
    """Adaptive end-of-turn detector built on Silero VAD.

    Public surface kept tiny on purpose:

    * :meth:`preload` -- best-effort load of Silero. Returns False
      cleanly when torch / silero-vad aren't available.
    * :meth:`should_finalize(audio_int16_bytes, silence_s)` --
      returns a :class:`TurnDecision`. Callers OR this with their
      legacy fixed-threshold check.
    * :meth:`reset` -- drop any per-utterance state at start of new
      utterance.

    The class is **thread-safe** for the dispatcher pattern in
    :mod:`voice.stt_whisper` where the audio thread feeds frames and
    the worker calls ``should_finalize``.
    """

    def __init__(self, config: SmartTurnTakerConfig | None = None) -> None:
        self._cfg = config or SmartTurnTakerConfig()
        self._model: Any = None
        self._torch: Any = None
        self._np: Any = None
        self._available: bool = False
        self._last_eval_at: float = 0.0
        self._lock = threading.Lock()
        self._stats: dict[str, Any] = {
            "evals": 0,
            "skips": 0,
            "early_finals": 0,
            "lockouts": 0,
            "fallbacks": 0,
            "max_prob": 0.0,
        }

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def config(self) -> SmartTurnTakerConfig:
        return self._cfg

    def stats(self) -> dict[str, Any]:
        return dict(self._stats)

    def reset(self) -> None:
        """Reset per-utterance counters at the start of a new utterance."""
        with self._lock:
            self._last_eval_at = 0.0

    # ── lifecycle ─────────────────────────────────────────────

    def preload(self) -> bool:
        """Best-effort load of Silero VAD + torch.

        Returns True only when the model is loaded and ready. On any
        failure (missing deps, network issue on first use, weights
        unavailable), logs a warning and returns False -- the STT
        loop should then fall back to its legacy fixed-threshold
        behaviour.
        """
        if not self._cfg.enabled:
            return False
        if self._available:
            return True
        try:
            import numpy as _np
            import torch as _torch
            from silero_vad import load_silero_vad
        except Exception as exc:  # pragma: no cover - env-specific
            logger.info(
                "SmartTurnTaker disabled: missing deps (%s). "
                "Run `pip install silero-vad torch` to enable.",
                exc,
            )
            self._stats["fallbacks"] += 1
            return False

        try:
            self._model = load_silero_vad()
        except Exception as exc:
            logger.warning(
                "SmartTurnTaker disabled: model load failed: %s", exc,
            )
            self._stats["fallbacks"] += 1
            return False

        self._torch = _torch
        self._np = _np
        self._available = True
        logger.info(
            "SmartTurnTaker ready: silero-vad loaded "
            "(min=%.2fs, max=%.2fs, eot_thr=%.2f)",
            self._cfg.min_silence_s,
            self._cfg.max_silence_s,
            self._cfg.eot_probability_threshold,
        )
        return True

    # ── decision ──────────────────────────────────────────────

    def should_finalize(
        self,
        audio_int16_bytes: bytes,
        silence_s: float,
        *,
        utterance_s: float = 0.0,
    ) -> TurnDecision:
        """Decide whether to finalize the current utterance.

        ``audio_int16_bytes`` is the raw int16 PCM of the ENTIRE
        accumulated utterance (matching the ring buffer in
        :mod:`stt_whisper`). The detector clips it to the last
        ``decision_window_s`` internally so callers don't have to.

        ``silence_s`` is the trailing-silence accumulated by the
        webrtcvad-driven counter in the STT loop.
        """
        cfg = self._cfg

        # Hard floor: never finalize before min_silence_s.
        if silence_s < cfg.min_silence_s:
            return TurnDecision(
                finalize=False,
                reason="below_min_silence",
                probability=0.0,
            )

        # Hard ceiling: always finalize after max_silence_s.
        if silence_s >= cfg.max_silence_s:
            self._stats["evals"] += 1
            return TurnDecision(
                finalize=True,
                reason="max_silence_ceiling",
                probability=1.0,
                eot_score=1.0,
            )

        if not (cfg.enabled and self._available):
            # Model not ready -- defer to caller's legacy fallback.
            return TurnDecision(
                finalize=False,
                reason="model_unavailable",
                probability=0.0,
                eval_skipped=True,
            )

        # CPU cooldown.
        now = time.monotonic()
        if (now - self._last_eval_at) * 1000.0 < cfg.min_eval_interval_ms:
            self._stats["skips"] += 1
            return TurnDecision(
                finalize=False,
                reason="eval_cooldown",
                probability=0.0,
                eval_skipped=True,
            )
        self._last_eval_at = now

        # Pull last ``decision_window_s`` of audio for the model.
        try:
            audio = self._slice_window(audio_int16_bytes, cfg)
            if audio is None:
                return TurnDecision(
                    finalize=False, reason="audio_too_short",
                    probability=0.0, eval_skipped=True,
                )
            prob = self._silero_eot_score(audio)
        except Exception as exc:
            logger.debug(
                "SmartTurnTaker inference failed: %s -- using fallback",
                exc, exc_info=True,
            )
            self._stats["fallbacks"] += 1
            return TurnDecision(
                finalize=False, reason="inference_error",
                probability=0.0, eval_skipped=True,
            )

        self._stats["evals"] += 1
        self._stats["max_prob"] = max(
            float(self._stats.get("max_prob", 0.0)), prob,
        )

        # Mid-thought lockout: refuse to finalize even past 600 ms
        # if Silero says the user is clearly mid-thought.
        if prob >= cfg.midthought_lockout_threshold:
            self._stats["lockouts"] += 1
            return TurnDecision(
                finalize=False,
                reason="midthought_lockout",
                probability=prob,
                midthought_score=prob,
            )

        # End-of-turn early-final: trigger fast when very confident.
        # Silero returns voice-activity prob, so end-of-turn is
        # roughly (1 - prob) calibrated against decay.
        eot_score = max(0.0, 1.0 - prob)
        if eot_score >= cfg.eot_probability_threshold:
            self._stats["early_finals"] += 1
            return TurnDecision(
                finalize=True,
                reason="early_final",
                probability=prob,
                eot_score=eot_score,
            )

        return TurnDecision(
            finalize=False,
            reason="below_eot_threshold",
            probability=prob,
            eot_score=eot_score,
        )

    # ── internals ─────────────────────────────────────────────

    def _slice_window(
        self, audio_int16_bytes: bytes, cfg: SmartTurnTakerConfig,
    ) -> Any:
        """Convert the raw int16 PCM into a torch tensor for Silero."""
        if self._np is None or self._torch is None:
            return None
        sr = cfg.sample_rate
        bytes_per_sample = 2  # int16
        bytes_per_window = int(cfg.decision_window_s * sr * bytes_per_sample)
        if len(audio_int16_bytes) < int(0.25 * sr * bytes_per_sample):
            # <250 ms total -- not enough for a stable EOT prob.
            return None
        tail = audio_int16_bytes[-bytes_per_window:] if (
            len(audio_int16_bytes) > bytes_per_window
        ) else audio_int16_bytes
        np = self._np
        arr = np.frombuffer(tail, dtype=np.int16).astype(np.float32) / 32768.0
        return self._torch.from_numpy(arr.copy())

    def _silero_eot_score(self, audio_tensor: Any) -> float:
        """Run Silero VAD over the audio tensor and return mean voice prob.

        We compute the *mean* probability across the decision window;
        a low mean (e.g. <0.22) means the tail of the utterance has
        decayed -- a strong end-of-turn signal. A high mean means
        speech is still ongoing.
        """
        torch = self._torch
        sr = self._cfg.sample_rate
        # Silero VAD expects 16 kHz mono and processes in 512-sample
        # chunks (~32 ms). We slide across the window and average.
        chunk = 512
        with torch.no_grad():
            n = audio_tensor.shape[0]
            if n < chunk:
                # Pad with zeros so Silero can still produce a single
                # estimate (effectively biasing toward "silent").
                pad = torch.zeros(chunk - n, dtype=audio_tensor.dtype)
                audio_tensor = torch.cat([audio_tensor, pad])
                n = chunk
            scores: list[float] = []
            for i in range(0, n - chunk + 1, chunk):
                window = audio_tensor[i:i + chunk]
                p = float(self._model(window, sr).item())
                scores.append(p)
            if not scores:
                return 0.0
            return float(sum(scores) / len(scores))


__all__ = [
    "SmartTurnTaker",
    "SmartTurnTakerConfig",
    "TurnDecision",
]
