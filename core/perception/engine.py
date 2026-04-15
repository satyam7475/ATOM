"""
ATOM -- Perception Engine (Phase 1 integration layer).

Orchestrates emotion analysis, urgency classification, interrupt
prediction, and speech style adaptation.  Subscribes to STT and
TTS lifecycle events on the bus and emits enriched perception
signals that downstream components (Router, TTS, InterruptHandler)
consume.

Total added latency per speech event: ~0.2ms.

Bus events emitted:
    ``perception_result``     -- emotion, urgency, style for every speech_final
    ``user_emotion_detected`` -- emotion label forwarded to existing TTS prosody
    ``interrupt_predicted``   -- early barge-in signal during TTS playback

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


class PerceptionEngine:
    """Real-time perception layer sitting between STT and the Router.

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
        style = self._style.decide(emotion, urgency)

        self._last_emotion = emotion
        self._last_urgency = urgency
        self._active_style = style
        self._style_locked = True

        logger.info(
            "Perception: emotion=%s(%.1f) urgency=%s(%.2f) "
            "rate=%.2fx pause=%.1fx | '%s'",
            emotion.label, emotion.intensity,
            urgency.level, urgency.score,
            style.rate_multiplier, style.pause_multiplier,
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
        )
        self._interrupt_count = 0

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
        }
