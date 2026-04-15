"""
ATOM -- Command Loop Controller.

The single entry point for all user commands. Wraps the Router with:
  - ExecutionLock: one command at a time
  - Instant ack: sub-100ms verbal feedback before processing
  - Pipeline budget: hard latency enforcement per stage
  - State machine transitions: LISTENING -> THINKING -> SPEAKING -> IDLE
  - Pipeline tracing: every stage timed with a trace ID
  - Cancellation: interrupt handler can abort the current command
  - Voice metrics: per-command timing breakdown

Pipeline:
    [Voice/Text Input]
        -> CommandLoop.submit()
        -> ExecutionLock gate
        -> Instant ACK (within 100ms)
        -> State -> THINKING
        -> System context injection
        -> Router._route() (intent -> cache/memory -> LLM)
        -> State -> SPEAKING (via bus events)
        -> TTS output
        -> Session memory record
        -> State -> IDLE / LISTENING
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from core.execution_lock import ExecutionLock

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.router import Router
    from core.state_manager import StateManager

logger = logging.getLogger("atom.command_loop")


class CommandLoop:
    """Deterministic single-command controller wrapping the Router."""

    __slots__ = (
        "_bus", "_state", "_router", "_lock",
        "_cancelled", "_current_trace_id", "_current_task",
        "_total_commands", "_total_errors",
        "_system_state_engine", "_session_memory",
        "_ack_engine", "_pipeline_metrics",
        "_intent_continuity", "_parallel_pipeline",
        "_suggestion_engine",
    )

    def __init__(
        self,
        bus: AsyncEventBus,
        state: StateManager,
        router: Router,
        *,
        lock_timeout_s: float = 30.0,
        system_state_engine: Any = None,
        session_memory: Any = None,
    ) -> None:
        self._bus = bus
        self._state = state
        self._router = router
        self._lock = ExecutionLock(default_timeout_s=lock_timeout_s)
        self._cancelled = asyncio.Event()
        self._current_trace_id: str | None = None
        self._current_task: asyncio.Task[None] | None = None
        self._total_commands: int = 0
        self._total_errors: int = 0
        self._system_state_engine = system_state_engine
        self._session_memory = session_memory
        self._ack_engine: Any = None
        self._pipeline_metrics: Any = None
        self._intent_continuity: Any = None
        self._parallel_pipeline: Any = None
        self._suggestion_engine: Any = None

    def attach_system_state(self, engine: Any) -> None:
        self._system_state_engine = engine

    def attach_session_memory(self, memory: Any) -> None:
        self._session_memory = memory

    def attach_ack_engine(self, ack: Any) -> None:
        self._ack_engine = ack

    def attach_pipeline_metrics(self, metrics: Any) -> None:
        self._pipeline_metrics = metrics

    def attach_intent_continuity(self, continuity: Any) -> None:
        self._intent_continuity = continuity

    def attach_parallel_pipeline(self, pipeline: Any) -> None:
        self._parallel_pipeline = pipeline

    def attach_suggestion_engine(self, engine: Any) -> None:
        self._suggestion_engine = engine

    @property
    def execution_lock(self) -> ExecutionLock:
        return self._lock

    @property
    def current_trace_id(self) -> str | None:
        return self._current_trace_id

    @property
    def is_busy(self) -> bool:
        return self._lock.is_busy

    async def submit(self, text: str = "", *, priority: str = "voice", **_kw: Any) -> None:
        """Submit a command through the controlled pipeline.

        This is the bus handler for ``speech_final``.

        Args:
            priority: ``"voice"`` for normal serial commands,
                      ``"background"`` to bypass the execution lock.
        """
        text = (text or "").strip()
        if not text:
            return

        if priority == "background":
            asyncio.create_task(self._run_background(text))
            return

        trace_id = uuid.uuid4().hex[:12]
        t0 = time.perf_counter()

        from voice.pipeline_budget import PipelineBudgetTracker
        budget = PipelineBudgetTracker(trace_id=trace_id, command=text)

        if not await self._lock.acquire(command=text, timeout_s=5.0):
            logger.warning(
                "[%s] Command rejected — pipeline busy with '%s'",
                trace_id,
                self._lock.current_command or "?",
            )
            self._bus.emit_fast(
                "response_ready",
                text="I'm still working on something, Boss. One moment.",
            )
            return

        self._cancelled.clear()
        self._current_trace_id = trace_id
        self._current_task = asyncio.current_task()
        self._total_commands += 1

        try:
            self._bus.emit_fast(
                "command_loop_trace",
                trace_id=trace_id,
                stage="start",
                text=text[:120],
            )

            # ── Instant acknowledgement (sub-100ms) ──────────────────
            budget.start_stage("ack")
            if self._ack_engine is not None:
                is_follow = False
                if self._session_memory is not None:
                    is_follow = self._session_memory.is_follow_up(text)
                ack_text = self._ack_engine.get_ack(text, is_follow_up=is_follow)
                if ack_text:
                    self._bus.emit_fast("voice_ack", text=ack_text)
            budget.end_stage("ack")

            from core.state_manager import AtomState
            if self._state.current in (AtomState.LISTENING, AtomState.IDLE):
                await self._state.transition(AtomState.THINKING)

            # ── System context injection ─────────────────────────────
            if self._system_state_engine is not None:
                try:
                    ctx = self._system_state_engine.get_context()
                    refs = self._system_state_engine.resolve_reference(text)
                    hints = self._system_state_engine.context_hints(text)
                    self._bus.emit_fast(
                        "command_context",
                        trace_id=trace_id,
                        system_context=ctx,
                        references=refs,
                        context_hints=hints,
                    )
                except Exception:
                    logger.debug("System context injection failed", exc_info=True)

            # ── Consume prefetched intent from ParallelPipeline ──────
            prefetched_intent = None
            if self._parallel_pipeline is not None:
                try:
                    prefetched_intent = self._parallel_pipeline.get_prefetched(text)
                    if prefetched_intent is not None:
                        self._bus.emit_fast(
                            "command_context",
                            trace_id=trace_id,
                            prefetched_intent=prefetched_intent,
                        )
                        logger.debug(
                            "[%s] Using prefetched intent: %s",
                            trace_id, prefetched_intent,
                        )
                except Exception:
                    logger.debug("ParallelPipeline prefetch failed", exc_info=True)

            # ── Route through intent -> action/LLM ───────────────────
            budget.start_stage("full_response")
            await self._router.on_speech(text)
            response_ms = budget.end_stage("full_response")

            # ── Record to session memory ─────────────────────────────
            if self._session_memory is not None:
                try:
                    app = ""
                    if self._system_state_engine is not None:
                        app = self._system_state_engine.snapshot.active_app
                    self._session_memory.record_command(
                        text, app=app, trace_id=trace_id,
                        elapsed_ms=response_ms,
                    )
                except Exception:
                    logger.debug("Session memory record failed", exc_info=True)

            # ── Update intent continuity ──────────────────────────────
            if self._intent_continuity is not None:
                try:
                    self._intent_continuity.on_command_complete(text)
                except Exception:
                    logger.debug("Intent continuity update failed", exc_info=True)

            # ── Inline suggestion (delayed so it doesn't collide) ────
            if self._suggestion_engine is not None:
                try:
                    active_app = ""
                    if self._system_state_engine is not None:
                        active_app = self._system_state_engine.snapshot.active_app
                    suggestion = self._suggestion_engine.suggest(
                        text,
                        active_app=active_app,
                    )
                    if suggestion:
                        self._bus.emit_fast(
                            "suggestion_ready",
                            suggestions=[suggestion],
                        )
                except Exception:
                    logger.debug("Suggestion engine failed", exc_info=True)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._bus.emit_fast(
                "command_loop_trace",
                trace_id=trace_id,
                stage="done",
                elapsed_ms=round(elapsed_ms, 1),
            )

            budget.log_summary()
            if self._pipeline_metrics is not None:
                self._pipeline_metrics.record(budget)

            logger.info(
                "[%s] Command completed in %.0fms: '%s'",
                trace_id, elapsed_ms, text[:60],
            )

        except asyncio.CancelledError:
            logger.info("[%s] Command cancelled: '%s'", trace_id, text[:60])
            self._bus.emit_fast(
                "command_loop_trace",
                trace_id=trace_id,
                stage="cancelled",
            )
        except Exception as exc:
            self._total_errors += 1
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.exception(
                "[%s] Command failed in %.0fms: '%s'",
                trace_id, elapsed_ms, text[:60],
            )
            self._bus.emit_fast(
                "command_loop_trace",
                trace_id=trace_id,
                stage="error",
                error=str(exc)[:200],
                elapsed_ms=round(elapsed_ms, 1),
            )
            try:
                self._bus.emit_fast(
                    "response_ready",
                    text="Something went wrong, Boss. Try again.",
                )
                from core.state_manager import AtomState
                await self._state.on_error(source="command_loop")
            except Exception:
                logger.debug("CommandLoop error recovery failed", exc_info=True)
        finally:
            self._current_task = None
            self._current_trace_id = None
            self._lock.release()

    async def cancel_current(self) -> bool:
        """Cancel the currently running command (called by interrupt handler).

        Cancels the asyncio Task running ``submit()``, which raises
        ``CancelledError`` inside the router and releases the execution lock.
        Also signals the Gemini executor thread to stop reading if a
        streaming cloud call is in flight.
        """
        if not self._lock.is_busy:
            return False

        self._cancelled.set()

        # Signal the Gemini executor thread to abort the SSE read loop
        # immediately rather than waiting for asyncio task cancellation
        # to propagate through the event loop.
        gemini = getattr(self._router, "_gemini_client", None)
        if gemini is not None:
            cancel_fn = getattr(gemini, "cancel_streaming", None)
            if callable(cancel_fn):
                cancel_fn()

        task = self._current_task
        if task is not None and not task.done():
            task.cancel()
            logger.info(
                "CommandLoop: task cancelled for '%s'",
                self._lock.current_command or "?",
            )
        else:
            logger.info(
                "CommandLoop: cancel flag set for '%s' (no active task)",
                self._lock.current_command or "?",
            )
        return True

    async def _run_background(self, text: str) -> None:
        """Run a non-voice command without acquiring the execution lock."""
        try:
            await self._router.on_speech(text)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.debug("Background task failed for '%s'", text[:60], exc_info=True)

    def get_diagnostics(self) -> dict[str, Any]:
        diag: dict[str, Any] = {
            "total_commands": self._total_commands,
            "total_errors": self._total_errors,
            "lock": self._lock.get_diagnostics(),
            "current_trace_id": self._current_trace_id,
        }
        if self._pipeline_metrics is not None:
            diag["pipeline_metrics"] = self._pipeline_metrics.get_diagnostics()
        return diag
