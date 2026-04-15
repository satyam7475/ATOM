"""
ATOM -- Perception Engine (Phase 1.5 — closed-loop adaptive layer).

Orchestrates emotion analysis, urgency classification, interrupt
prediction, speech style adaptation, and **session-level adaptive
tuning** based on TTS delivery feedback.

The feedback loop:

    STT → Perception → style/profile → Router + TTS
                 ↑                          │
                 └── tts_delivery_metrics ───┘

Bus events emitted:
    ``perception_result``     -- emotion, urgency, style for every speech_final
    ``user_emotion_detected`` -- emotion label forwarded to existing TTS prosody
    ``interrupt_predicted``   -- early barge-in signal during TTS playback
    ``perception_adaptive``   -- concise/rate hints for Router + TTS

Owner: Satyam
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from core.perception.emotion import EmotionAnalyzer, EmotionResult
from core.perception.urgency import UrgencyClassifier
from core.perception.interrupt_predictor import InterruptPredictor
from core.perception.speech_style import SpeechStyleController

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus

logger = logging.getLogger("atom.perception")

_EMOTION_SMOOTH_ALPHA = 0.6
_SESSION_EMA_ALPHA = 0.3


class PerceptionEngine:
    """Real-time perception layer with session-level adaptive tuning.

    Lightweight, deterministic, zero external dependencies.
    """

    __slots__ = (
        "_bus",
        "_emotion",
        "_urgency",
        "_interrupt",
        "_style",
        "_last_emotion",
        "_last_urgency",
        "_smoothed_emotion",
        "_style_locked",
        "_active_style",
        "_interrupt_count",
        "_session_stats",
        "_adaptive_profile",
    )

    def __init__(self, bus: AsyncEventBus) -> None:
        self._bus = bus
        self._emotion = EmotionAnalyzer()
        self._urgency = UrgencyClassifier()
        self._interrupt = InterruptPredictor()
        self._style = SpeechStyleController()
        self._last_emotion: Any = None
        self._last_urgency: Any = None
        self._smoothed_emotion: EmotionResult | None = None
        self._style_locked = False
        self._active_style: Any = None
        self._interrupt_count: int = 0

        self._session_stats: dict[str, float] = {
            "interrupt_rate": 0.0,
            "avg_duration_ms": 0.0,
            "frustration_score": 0.0,
            "response_count": 0.0,
        }
        self._adaptive_profile: dict[str, Any] = {
            "concise": False,
            "rate_boost": 0.0,
        }

    # ── Emotion smoothing ────────────────────────────────────────

    def _smooth_emotion(self, raw: EmotionResult) -> EmotionResult:
        """Weighted blend so emotion doesn't jitter between turns.

        Uses exponential smoothing on intensity, and keeps the new label
        unless the intensity delta is negligible (prevents flicker between
        ``calm`` and ``neutral`` on every sentence).
        """
        prev = self._smoothed_emotion
        if prev is None:
            self._smoothed_emotion = raw
            return raw

        alpha = _EMOTION_SMOOTH_ALPHA
        blended_intensity = alpha * raw.intensity + (1.0 - alpha) * prev.intensity

        label = raw.label
        if abs(raw.intensity - prev.intensity) < 0.15 and raw.label != prev.label:
            label = prev.label

        smoothed = EmotionResult(label, round(blended_intensity, 3))
        self._smoothed_emotion = smoothed
        return smoothed

    # ── STT event handlers ────────────────────────────────────────

    async def on_speech_partial(self, text: str = "", **_kw: Any) -> None:
        """Check for predicted interrupts on every partial."""
        if self._interrupt.on_partial(text):
            self._interrupt_count += 1
            logger.info("Interrupt predicted from partial: '%s'", (text or "")[:60])
            self._bus.emit_fast("interrupt_predicted", text=text)

    async def on_speech_final(self, text: str = "", **_kw: Any) -> None:
        """Full perception pass on every final utterance."""
        if not text or not text.strip():
            return

        raw_emotion = self._emotion.analyze(text)
        emotion = self._smooth_emotion(raw_emotion)
        urgency = self._urgency.classify(text)

        rate_boost = self._adaptive_profile["rate_boost"]
        style = self._style.decide(emotion, urgency, rate_boost=rate_boost)

        self._last_emotion = emotion
        self._last_urgency = urgency
        self._active_style = style
        self._style_locked = True

        logger.info(
            "Perception: emotion=%s(%.1f) urgency=%s(%.2f) "
            "rate=%.2fx pause=%.1fx concise=%s | '%s'",
            emotion.label, emotion.intensity,
            urgency.level, urgency.score,
            style.rate_multiplier, style.pause_multiplier,
            self._adaptive_profile["concise"],
            text[:60],
        )

        self._bus.emit_fast(
            "perception_result",
            emotion=emotion,
            urgency=urgency,
            style=style,
            text=text,
        )

        self._bus.emit_fast("user_emotion_detected", emotion=emotion.label)

        self._bus.emit_fast(
            "perception_adaptive",
            concise=self._adaptive_profile["concise"],
            rate_boost=self._adaptive_profile["rate_boost"],
            session_stats=dict(self._session_stats),
        )

    # ── TTS lifecycle handlers ────────────────────────────────────

    async def on_tts_speaking(self, **_kw: Any) -> None:
        """Called when TTS starts playing audio."""
        self._interrupt.on_tts_start()

    async def on_tts_done(self, **_kw: Any) -> None:
        """Called when TTS finishes playing audio."""
        self._interrupt.on_tts_end()
        self._style_locked = False

        self._bus.emit_fast(
            "tts_metrics",
            interrupt_count=self._interrupt_count,
            last_emotion=(
                self._last_emotion.label if self._last_emotion else "neutral"
            ),
            last_urgency=(
                self._last_urgency.level if self._last_urgency else "low"
            ),
            session_stats=dict(self._session_stats),
        )
        self._interrupt_count = 0

    # ── Delivery feedback (closed loop) ───────────────────────────

    async def on_tts_delivery_metrics(
        self,
        words_spoken: int = 0,
        duration_ms: float = 0.0,
        **_kw: Any,
    ) -> None:
        """Process TTS delivery feedback and update session stats.

        This is the feedback path: TTS tells us how delivery went,
        and we adjust the adaptive profile for the *next* response.
        """
        stats = self._session_stats
        profile = self._adaptive_profile
        alpha = _SESSION_EMA_ALPHA

        stats["response_count"] += 1

        # Normalized interrupt rate: interrupts per 10 spoken words.
        # A long explanation with 1 interrupt is fine; a short command
        # with 1 interrupt means the user really didn't want it.
        norm_interrupts = self._interrupt_count / max(1.0, words_spoken / 10.0)

        # EMA updates
        stats["interrupt_rate"] = (
            alpha * norm_interrupts + (1.0 - alpha) * stats["interrupt_rate"]
        )
        if duration_ms > 0:
            stats["avg_duration_ms"] = (
                alpha * duration_ms + (1.0 - alpha) * stats["avg_duration_ms"]
            )

        # Frustration composite: negative emotion + high interrupt rate
        emo_label = self._last_emotion.label if self._last_emotion else "neutral"
        frustration_input = (
            1.0 if emo_label in ("frustrated", "angry", "stressed") else 0.0
        )
        frustration_input += min(1.0, norm_interrupts)
        stats["frustration_score"] = (
            alpha * frustration_input + (1.0 - alpha) * stats["frustration_score"]
        )

        # ── Adaptive decisions for next response ──────────────────
        was_concise = profile["concise"]

        profile["concise"] = (
            stats["interrupt_rate"] > 0.3
            or self._interrupt_count >= 2
            or stats["frustration_score"] > 0.6
        )

        profile["rate_boost"] = (
            0.08 if stats["avg_duration_ms"] > 4000 else 0.0
        )

        if profile["concise"] != was_concise:
            logger.info(
                "Adaptive profile changed: concise=%s (int_rate=%.2f frust=%.2f)",
                profile["concise"],
                stats["interrupt_rate"],
                stats["frustration_score"],
            )

    # ── Diagnostics ───────────────────────────────────────────────

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            "last_emotion": (
                {"label": self._last_emotion.label, "intensity": self._last_emotion.intensity}
                if self._last_emotion else None
            ),
            "last_urgency": (
                {"level": self._last_urgency.level, "score": self._last_urgency.score}
                if self._last_urgency else None
            ),
            "style_locked": self._style_locked,
            "smoothed_emotion": (
                {"label": self._smoothed_emotion.label, "intensity": self._smoothed_emotion.intensity}
                if self._smoothed_emotion else None
            ),
            "session_stats": dict(self._session_stats),
            "adaptive_profile": dict(self._adaptive_profile),
        }
