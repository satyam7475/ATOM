"""
ATOM -- End-to-end pipeline latency tracker (enhanced).

Tracks timing across the full voice command pipeline:
    wake_word -> speech_final -> intent -> router -> action -> tts_start -> tts_complete

Enhanced fields (v2):
  - trace_id: correlates all logs for a single command
  - decision_source: which layer answered (intent/cache/llm/cloud)
  - failure_reason: why a command failed (if it did)
  - wake_word timing: detection to STT start

Emits a structured one-line summary log per query:

    PIPELINE | id=abc123 | Wake: 0ms | Intent: 2ms | Decision: cache | Action: 4ms | TTS: 6ms | Total: 122ms

Hooks into the event bus -- zero coupling to individual modules.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.metrics import MetricsCollector

logger = logging.getLogger("atom.pipeline")


class PipelineTimer:
    """Per-query latency tracker for the ATOM voice pipeline."""

    __slots__ = (
        "_bus", "_metrics",
        "_t_wake_word", "_t_speech_final", "_t_intent_done", "_t_action_done",
        "_t_tts_start", "_t_tts_complete",
        "_t_cursor_query",
        "_current_query", "_active",
        "_trace_id", "_decision_source", "_failure_reason",
        "_intent_name",
    )

    def __init__(self, bus: "AsyncEventBus",
                 metrics: "MetricsCollector | None" = None) -> None:
        self._bus = bus
        self._metrics = metrics
        self._reset()

    def _reset(self) -> None:
        self._t_wake_word: float = 0.0
        self._t_speech_final: float = 0.0
        self._t_intent_done: float = 0.0
        self._t_action_done: float = 0.0
        self._t_tts_start: float = 0.0
        self._t_tts_complete: float = 0.0
        self._t_cursor_query: float = 0.0
        self._current_query: str = ""
        self._active: bool = False
        self._trace_id: str = ""
        self._decision_source: str = ""
        self._failure_reason: str = ""
        self._intent_name: str = ""

    def register(self) -> None:
        """Subscribe to bus events. Call once during startup."""
        self._bus.on("wake_word_detected", self._on_wake_word)
        self._bus.on("speech_final", self._on_speech_final)
        self._bus.on("intent_classified", self._on_intent_classified)
        self._bus.on("cursor_query", self._on_cursor_query)
        self._bus.on("partial_response", self._on_partial_response)
        self._bus.on("response_ready", self._on_response_ready)
        self._bus.on("state_changed", self._on_state_changed)
        self._bus.on("tts_complete", self._on_tts_complete)
        self._bus.on("command_loop_trace", self._on_command_loop_trace)

    async def _on_wake_word(self, **_kw) -> None:
        self._t_wake_word = time.perf_counter()

    async def _on_speech_final(self, text: str = "", **_kw) -> None:
        self._reset()
        self._t_speech_final = time.perf_counter()
        self._current_query = text[:60] if text else ""
        self._trace_id = uuid.uuid4().hex[:12]
        self._active = True

    async def _on_intent_classified(self, intent: str = "",
                                    ms: float = 0.0, **_kw) -> None:
        if self._active:
            self._t_intent_done = time.perf_counter()
            self._intent_name = intent
            if intent in ("fallback",):
                self._decision_source = "llm"
            elif ms < 5.0 and intent:
                self._decision_source = "intent"

    async def _on_cursor_query(self, text: str = "", source: str = "", **_kw) -> None:
        if self._active:
            self._t_cursor_query = time.perf_counter()
            if source:
                self._decision_source = source
            elif not self._decision_source:
                self._decision_source = "llm"

    async def _on_partial_response(
        self, text: str = "", is_first: bool = False, source: str = "", **_kw,
    ) -> None:
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
            if not self._decision_source:
                self._decision_source = "intent"

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

    async def _on_command_loop_trace(
        self,
        trace_id: str = "",
        stage: str = "",
        error: str = "",
        **_kw,
    ) -> None:
        if trace_id and self._active:
            self._trace_id = trace_id
        if stage == "error" and error:
            self._failure_reason = error[:200]

    def _log_summary(self) -> None:
        t0 = self._t_speech_final
        if t0 <= 0:
            return

        def _ms(t: float) -> float:
            return (t - t0) * 1000 if t > 0 else 0.0

        wake_ms = (
            (self._t_speech_final - self._t_wake_word) * 1000
            if self._t_wake_word > 0
            else 0.0
        )
        intent_ms = _ms(self._t_intent_done)
        # Sprint Ω.8 (Apr 26 2026) R10: clamp every span to >= 0. The
        # speech_final → intent_done → action_done → tts_start →
        # tts_complete sequence can re-order across event-loop hops
        # (e.g. cache hit emits action_done before intent_done from a
        # parallel handler), and atomCurrentLogs.txt L390 + L405 had
        # negative Action / TTS readings as a result. A negative span
        # in a "PIPELINE" log line is just noise; we clamp instead of
        # propagating ratio-of-totals lies into Prometheus.
        if self._t_action_done > 0 and self._t_intent_done > 0:
            action_ms = max(0.0, _ms(self._t_action_done) - intent_ms)
        else:
            action_ms = max(0.0, _ms(self._t_action_done))
        if self._t_tts_start > 0 and self._t_tts_complete > 0:
            tts_ms = max(0.0, _ms(self._t_tts_complete) - _ms(self._t_tts_start))
        else:
            tts_ms = 0.0
        total_ms = max(0.0, _ms(self._t_tts_complete))

        parts = [
            f"PIPELINE | id={self._trace_id}",
            f"Query: '{self._current_query}'",
        ]
        if wake_ms > 0:
            parts.append(f"Wake: {wake_ms:.0f}ms")
        parts.append(f"Intent: {intent_ms:.0f}ms")
        if self._decision_source:
            parts.append(f"Decision: {self._decision_source}")
        parts.append(f"Action: {action_ms:.0f}ms")
        parts.append(f"TTS: {tts_ms:.0f}ms")
        parts.append(f"Total: {total_ms:.0f}ms")
        if self._failure_reason:
            parts.append(f"FAILED: {self._failure_reason[:80]}")

        logger.info(" | ".join(parts))

        if self._metrics is not None:
            self._metrics.record_latency("pipeline_total", total_ms)
            self._metrics.record_latency("pipeline_intent", intent_ms)
            self._metrics.record_latency("pipeline_action", action_ms)
            self._metrics.record_latency("pipeline_tts", tts_ms)
            if wake_ms > 0:
                self._metrics.record_latency("pipeline_wake", wake_ms)

        if tts_ms > 0.05:
            try:
                from core.observability.per_module_latency import get_latency_board

                get_latency_board().record_module_call("tts", float(tts_ms), error=False)
            except Exception:
                logger.debug('Observability step failed', exc_info=True)
