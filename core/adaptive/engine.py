"""
ATOM -- Adaptive Intelligence Engine (Phase 2).

Closes the perception → action → delivery → learning loop:

    Perception → Router/TTS → tts_delivery_metrics
         ↑                          │
         └── AdaptiveEngine ────────┘

The engine maintains a :class:`BehaviorMemory` that learns user
preferences over a rolling window, a :class:`SpeechOptimizer` that
blends those preferences with real-time perception, and a
:class:`ResponseShaper` that adjusts response verbosity.

Bus events consumed:
    ``perception_result``      -- current emotion / urgency / style
    ``tts_delivery_metrics``   -- words_spoken, duration_ms from TTS

Bus events emitted:
    ``adaptive_speech_update`` -- optimized rate/pause for TTS
    ``adaptive_profile_update`` -- full profile snapshot for diagnostics

Owner: Satyam
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from core.adaptive.behavior_memory import BehaviorMemory
from core.adaptive.speech_optimizer import SpeechOptimizer
from core.adaptive.response_shaper import ResponseShaper

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus

logger = logging.getLogger("atom.adaptive")


class AdaptiveEngine:
    """Session-level adaptive intelligence layer."""

    __slots__ = (
        "_bus",
        "_memory",
        "_optimizer",
        "_shaper",
        "_last_perception",
    )

    def __init__(self, bus: AsyncEventBus) -> None:
        self._bus = bus
        self._memory = BehaviorMemory()
        self._optimizer = SpeechOptimizer()
        self._shaper = ResponseShaper()
        self._last_perception: dict[str, Any] = {}

    # ── Input handlers (bus subscribers) ──────────────────────────

    async def on_perception(
        self,
        emotion: Any = None,
        urgency: Any = None,
        style: Any = None,
        **_kw: Any,
    ) -> None:
        """Capture latest perception for speech optimization."""
        self._last_perception = {
            "emotion": getattr(emotion, "label", "neutral") if emotion else "neutral",
            "urgency": getattr(urgency, "level", "medium") if urgency else "medium",
        }

    async def on_tts_delivery_metrics(
        self,
        words_spoken: int = 0,
        duration_ms: float = 0.0,
        backend: str = "",
        **_kw: Any,
    ) -> None:
        """Learn from TTS delivery and update adaptive profile."""
        self._memory.record({
            "words_spoken": words_spoken,
            "duration_ms": duration_ms,
            "interrupt_count": _kw.get("interrupt_count", 0),
            "backend": backend,
        })
        self._memory.update_from_metrics()

        profile = self._memory.get_profile()

        speech_params = self._optimizer.optimize(
            self._last_perception, profile,
        )

        self._bus.emit_fast(
            "adaptive_speech_update",
            rate_multiplier=speech_params["rate_multiplier"],
            pause_multiplier=speech_params["pause_multiplier"],
        )

        self._bus.emit_fast(
            "adaptive_profile_update",
            profile=profile,
        )

        logger.info(
            "Adaptive cycle: verb=%.2f rate=%.2f pause=%.2f | "
            "speech → rate=%.2fx pause=%.1fx",
            profile["verbosity"],
            profile["preferred_rate"],
            profile["preferred_pause"],
            speech_params["rate_multiplier"],
            speech_params["pause_multiplier"],
        )

    # ── Output hooks (called by router) ──────────────────────────

    def process_response(self, text: str) -> tuple[str, dict[str, float]]:
        """Shape response text and compute speech params.

        Returns ``(shaped_text, speech_params)`` where speech_params
        has ``rate_multiplier`` and ``pause_multiplier``.
        """
        profile = self._memory.get_profile()
        shaped = self._shaper.shape(text, profile)
        speech = self._optimizer.optimize(self._last_perception, profile)
        return shaped, speech

    def get_verbosity(self) -> float:
        """Current learned verbosity (0 = concise, 1 = verbose)."""
        return self._memory.get_profile()["verbosity"]

    def should_be_concise(self) -> bool:
        """True when learned profile indicates user prefers short responses."""
        return self._memory.get_profile()["verbosity"] < 0.35

    # ── Diagnostics ───────────────────────────────────────────────

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            "profile": self._memory.get_profile(),
            "last_perception": self._last_perception,
            "history_size": len(self._memory._history),
        }
