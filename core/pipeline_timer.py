"""
ATOM -- End-to-end pipeline latency tracker.

Tracks timing across the full voice command pipeline:
    speech_final -> intent -> router -> action -> tts_start

Also records **pipeline_llm_first_partial**: time from ``cursor_query`` to the
first ``partial_response`` (streaming first chunk) — validates the LLM hot path
without blocking the bus on generation.

Emits a structured one-line summary log per query:

    PIPELINE | STT: 110ms | Intent: 2ms | Action: 4ms | TTS: 6ms | Total: 122ms

Hooks into the event bus -- zero coupling to individual modules.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.metrics import MetricsCollector

logger = logging.getLogger("atom.pipeline")


class PipelineTimer:
    """Per-query latency tracker for the ATOM voice pipeline."""

    __slots__ = (
        "_bus", "_metrics",
        "_t_speech_final", "_t_intent_done", "_t_action_done",
        "_t_tts_start", "_t_tts_complete",
        "_t_cursor_query",
        "_current_query", "_active",
    )

    def __init__(self, bus: AsyncEventBus,
                 metrics: MetricsCollector | None = None) -> None:
        self._bus = bus
        self._metrics = metrics
        self._reset()

    def _reset(self) -> None:
        self._t_speech_final: float = 0.0
        self._t_intent_done: float = 0.0
        self._t_action_done: float = 0.0
        self._t_tts_start: float = 0.0
        self._t_tts_complete: float = 0.0
        self._t_cursor_query: float = 0.0
        self._current_query: str = ""
        self._active: bool = False

    def register(self) -> None:
        """Subscribe to bus events. Call once during startup."""
        self._bus.on("speech_final", self._on_speech_final)
        self._bus.on("intent_classified", self._on_intent_classified)
        self._bus.on("cursor_query", self._on_cursor_query)
        self._bus.on("partial_response", self._on_partial_response)
        self._bus.on("response_ready", self._on_response_ready)
        self._bus.on("state_changed", self._on_state_changed)
        self._bus.on("tts_complete", self._on_tts_complete)

    async def _on_speech_final(self, text: str = "", **_kw) -> None:
        self._reset()
        self._t_speech_final = time.perf_counter()
        self._current_query = text[:60] if text else ""
        self._active = True

    async def _on_intent_classified(self, intent: str = "",
                                    ms: float = 0.0, **_kw) -> None:
        if self._active:
            self._t_intent_done = time.perf_counter()

    async def _on_cursor_query(self, text: str = "", **_kw) -> None:
        """LLM path scheduled — anchor for first streamed partial latency."""
        if self._active:
            self._t_cursor_query = time.perf_counter()

    async def _on_partial_response(
        self, text: str = "", is_first: bool = False, source: str = "", **_kw,
    ) -> None:
        """First sentence/chunk from local MLX — streaming segment (not TTS partials)."""
        if source != "local":
            return
        if (
            not self._active
            or not is_first
            or self._t_cursor_query <= 0
            or self._metrics is None
        ):
            return
        dt_ms = (time.perf_counter() - self._t_cursor_query) * 1000
        self._metrics.record_latency("pipeline_llm_first_partial", dt_ms)
        self._t_cursor_query = 0.0

    async def _on_response_ready(self, text: str = "", **_kw) -> None:
        if self._active:
            self._t_action_done = time.perf_counter()

    async def _on_state_changed(self, old=None, new=None, **_kw) -> None:
        from core.state_manager import AtomState
        if self._active and new is AtomState.SPEAKING:
            self._t_tts_start = time.perf_counter()

    async def _on_tts_complete(self, **_kw) -> None:
        if not self._active:
            return
        self._t_tts_complete = time.perf_counter()
        self._log_summary()
        self._active = False

    def _log_summary(self) -> None:
        t0 = self._t_speech_final
        if t0 <= 0:
            return

        def _ms(t: float) -> float:
            return (t - t0) * 1000 if t > 0 else 0.0

        intent_ms = _ms(self._t_intent_done)
        action_ms = (_ms(self._t_action_done) - intent_ms
                     if self._t_action_done > 0 and self._t_intent_done > 0
                     else _ms(self._t_action_done))
        tts_ms = (_ms(self._t_tts_complete) - _ms(self._t_tts_start)
                  if self._t_tts_start > 0
                  else 0.0)
        total_ms = _ms(self._t_tts_complete)

        logger.info(
            "PIPELINE | Query: '%s' | Intent: %.0fms | "
            "Action: %.0fms | TTS: %.0fms | Total: %.0fms",
            self._current_query,
            intent_ms, action_ms, tts_ms, total_ms,
        )

        if self._metrics is not None:
            self._metrics.record_latency("pipeline_total", total_ms)
            self._metrics.record_latency("pipeline_intent", intent_ms)
            self._metrics.record_latency("pipeline_action", action_ms)
            self._metrics.record_latency("pipeline_tts", tts_ms)

        if tts_ms > 0.05:
            try:
                from core.observability.per_module_latency import get_latency_board

                get_latency_board().record_module_call("tts", float(tts_ms), error=False)
            except Exception:
                pass
