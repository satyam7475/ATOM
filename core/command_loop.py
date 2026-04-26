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


def _swallow_ack_exception(task: asyncio.Task[Any]) -> None:
    """Done-callback for the C3 fire-and-forget ack task.

    Drains the exception so Python's "Task exception was never
    retrieved" warning does not pollute the logs whenever TTS
    glitches mid-ack. The ack is best-effort UX, never load-bearing.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.debug("ack overlap task raised: %s", exc, exc_info=exc)


class CommandLoop:
    """Deterministic single-command controller wrapping the Router."""

    __slots__ = (
        "_bus", "_state", "_router", "_lock",
        "_cancelled", "_current_trace_id", "_current_task",
        "_total_commands", "_total_errors",
        "_system_state_engine", "_session_memory",
        "_ack_engine", "_pipeline_metrics",
        "_parallel_pipeline",
        "_suggestion_engine",
        "_pending_turn", "_last_response_text",
        "_turn_attached", "_turn_complete_count",
        # Sprint C3: parallel ack-TTS overlap
        "_tts", "_ack_task", "_ack_overlap_count",
        # Sprint Ω.13: deferred ack — cancellation flag + submit clock so
        # we can suppress the ACK entirely whenever ``response_ready``
        # arrives within the deferral window.
        "_ack_cancelled", "_ack_submit_t", "_ack_deferral_s",
        "_ack_skipped_count",
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
        self._parallel_pipeline: Any = None
        self._suggestion_engine: Any = None
        self._pending_turn: dict[str, Any] | None = None
        self._last_response_text: str = ""
        self._turn_attached: bool = False
        self._turn_complete_count: int = 0
        # Sprint C3: parallel ack-TTS overlap. ``_tts`` is wired by
        # ``main.py`` after the TTS engine boots; when present we spawn
        # ``speak_ack`` directly (instead of going through the bus) so
        # the LLM call kicks off on the very next event-loop tick
        # instead of waiting for the bus consumer to dequeue the
        # ``voice_ack`` event. Saves ~50-200 ms of dispatch latency
        # on each turn (atomLogs.txt L343/394/511 -- "Perceived
        # latency 3.0-3.6s on simple turns").
        self._tts: Any = None
        self._ack_task: asyncio.Task[None] | None = None
        self._ack_overlap_count: int = 0
        # Sprint Ω.13 (Apr 27 2026): deferred ACK. The previous build
        # spoke "One moment." even for sub-10 ms intent fast-paths
        # (atomCurrentLogs.txt L295/L338/L365/L404/L417). We now wait
        # ``_ack_deferral_s`` before actually speaking the ack, and the
        # ``response_ready`` handler cancels the pending task if the
        # brain returns inside that window. The bus event still fires
        # immediately so dashboard/indicator subscribers stay in sync.
        self._ack_cancelled: bool = False
        self._ack_submit_t: float = 0.0
        self._ack_deferral_s: float = 0.28
        self._ack_skipped_count: int = 0

    def attach_system_state(self, engine: Any) -> None:
        self._system_state_engine = engine

    def attach_session_memory(self, memory: Any) -> None:
        self._session_memory = memory

    def attach_ack_engine(self, ack: Any) -> None:
        self._ack_engine = ack

    def attach_pipeline_metrics(self, metrics: Any) -> None:
        self._pipeline_metrics = metrics

    def attach_parallel_pipeline(self, pipeline: Any) -> None:
        self._parallel_pipeline = pipeline

    def attach_suggestion_engine(self, engine: Any) -> None:
        self._suggestion_engine = engine

    def attach_tts(self, tts: Any) -> None:
        """Wire the TTS engine for direct ack-overlap (Sprint C3).

        When provided, ``submit()`` spawns ``tts.speak_ack`` as a
        fire-and-forget task *before* awaiting any later stage so the
        ack and the LLM call run in parallel. The bus event is still
        emitted for indicator / dashboard subscribers.

        ``tts`` may be ``None`` to detach (used by tests and for
        graceful TTS-disabled fallback). The method is a no-op when
        the supplied object lacks ``speak_ack``.
        """
        if tts is not None and not hasattr(tts, "speak_ack"):
            logger.debug(
                "attach_tts: object %r lacks speak_ack; ignored",
                type(tts).__name__,
            )
            return
        self._tts = tts

    def attach_turn_emitter(self) -> None:
        """Subscribe to bus events that drive ``turn_complete``.

        Idempotent. After this call the loop will:

        * latch the response text from ``response_ready`` so that the
          per-turn payload carries both sides of the exchange;
        * emit ``turn_complete`` exactly once per user turn after the
          matching ``tts_complete`` (the reflective loop in G1 keys
          off this signal);
        * cancel a pending ``turn_complete`` when a new ``speech_final``
          arrives -- this is the *short-circuit* guard that prevents
          ATOM from advising/clarifying about a stale turn.
        """
        if self._turn_attached:
            return
        self._bus.on("response_ready", self._on_response_ready)
        self._bus.on("tts_complete", self._on_tts_complete)
        self._bus.on("speech_final", self._on_speech_final_short_circuit)
        self._turn_attached = True

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

            # ── Acknowledgement (deferred — Sprint Ω.13) ─────────────
            # Sprint C3 spawned ``speak_ack`` immediately so the audio
            # could roll while the LLM prefilled. That worked for slow
            # turns but produced a "One moment" for every sub-300 ms
            # intent fast-path (atomCurrentLogs L295/L338/L365). We now
            # wait ``_ack_deferral_s`` before speaking; the
            # ``response_ready`` handler sets ``_ack_cancelled`` and
            # cancels the task if the brain replied inside the window
            # — saving the user from a useless "On it." in front of an
            # instant answer. Bus event still fires immediately so
            # dashboard/indicator subscribers stay in sync.
            budget.start_stage("ack")
            self._ack_task = None
            self._ack_cancelled = False
            self._ack_submit_t = time.perf_counter()
            if self._ack_engine is not None:
                is_follow = False
                if self._session_memory is not None:
                    is_follow = self._session_memory.is_follow_up(text)
                ack_text = self._ack_engine.get_ack(text, is_follow_up=is_follow)
                if ack_text:
                    spoken_inline = False
                    if self._tts is not None:
                        try:
                            self._ack_task = asyncio.create_task(
                                self._deferred_ack(ack_text, trace_id),
                                name=f"ack_overlap_{trace_id}",
                            )
                            self._ack_task.add_done_callback(
                                _swallow_ack_exception,
                            )
                            self._ack_overlap_count += 1
                            spoken_inline = True
                        except Exception:
                            logger.debug(
                                "[%s] ack overlap task spawn failed",
                                trace_id, exc_info=True,
                            )
                            self._ack_task = None
                    # Always emit the bus event so dashboard /
                    # indicator subscribers see the ack. The
                    # ``spoken_inline`` flag tells ``on_voice_ack``
                    # in wiring.py NOT to double-speak when we already
                    # spawned the ack task directly.
                    self._bus.emit_fast(
                        "voice_ack",
                        text=ack_text,
                        spoken_inline=spoken_inline,
                    )
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

            # ── Inline suggestion (only after action commands, not quick/conversational) ──
            if self._suggestion_engine is not None and response_ms > 100:
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
            self._pending_turn = {
                "trace_id": trace_id,
                "user_text": text,
                "elapsed_ms": round(elapsed_ms, 1),
                "ts": time.monotonic(),
            }
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
            raise
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
            "turn_complete_count": self._turn_complete_count,
        }
        if self._pipeline_metrics is not None:
            diag["pipeline_metrics"] = self._pipeline_metrics.get_diagnostics()
        return diag

    # ── deferred ack (Sprint Ω.13) ─────────────────────────────

    async def _deferred_ack(self, ack_text: str, trace_id: str) -> None:
        """Speak the ACK only if the response hasn't already arrived.

        Sleeps ``_ack_deferral_s`` (default 280 ms). If
        :meth:`_on_response_ready` flips ``_ack_cancelled`` during the
        sleep — i.e. the brain returned faster than the deferral
        window — we skip the ACK entirely. Otherwise we speak it on
        the original schedule. ``CancelledError`` from
        ``_on_response_ready`` is swallowed so the pipeline never sees
        a stray cancellation.
        """
        if self._tts is None:
            return
        try:
            await asyncio.sleep(self._ack_deferral_s)
        except asyncio.CancelledError:
            return
        if self._ack_cancelled or self._cancelled.is_set():
            return
        try:
            await self._tts.speak_ack(ack_text)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug(
                "[%s] deferred ack speak failed", trace_id, exc_info=True,
            )

    # ── turn lifecycle (G6) ─────────────────────────────────────

    async def _on_response_ready(self, text: str = "", **_kw: Any) -> None:
        """Latch the most recent assistant utterance for ``turn_complete``.

        Sprint Ω.13: also short-circuits the deferred ACK when the brain
        returns inside the deferral window. We log
        ``PIPELINE_FAST_PATH: skip ACK`` so observability shows the
        suppression — the user feels an instant reply with no "One
        moment." preamble.
        """
        ack_task = self._ack_task
        if ack_task is not None and not ack_task.done():
            elapsed_ms = (time.perf_counter() - self._ack_submit_t) * 1000.0
            if elapsed_ms < (self._ack_deferral_s * 1000.0):
                self._ack_cancelled = True
                self._ack_skipped_count += 1
                logger.info(
                    "PIPELINE_FAST_PATH: skip ACK (response_ready in %.0fms < %dms)",
                    elapsed_ms, int(self._ack_deferral_s * 1000.0),
                )
                try:
                    ack_task.cancel()
                except Exception:
                    logger.debug(
                        "ack_task cancel raised", exc_info=True,
                    )
        if text:
            self._last_response_text = text

    async def _on_tts_complete(self, **_kw: Any) -> None:
        """Emit ``turn_complete`` once per fully-flushed turn.

        Skipped when:
          * the loop is busy (still processing the next turn),
          * there is no pending turn (the TTS belonged to a system
            insight or proactive nudge),
          * the user has already started a new turn -- the short-circuit
            guard in ``_on_speech_final_short_circuit`` cleared it.
        """
        pending = self._pending_turn
        if pending is None:
            return
        if self._lock.is_busy:
            return
        self._pending_turn = None
        self._turn_complete_count += 1
        try:
            self._bus.emit_fast(
                "turn_complete",
                trace_id=pending.get("trace_id"),
                user_text=pending.get("user_text", ""),
                response_text=self._last_response_text,
                elapsed_ms=pending.get("elapsed_ms"),
            )
        except Exception:
            logger.debug("turn_complete emit failed", exc_info=True)

    async def _on_speech_final_short_circuit(
        self, text: str = "", **_kw: Any,
    ) -> None:
        """Drop any pending turn-complete the moment a new utterance lands.

        The reflective loop also subscribes to ``speech_final`` and
        will abandon its in-flight LLM call -- this guard prevents the
        *next* tts_complete from re-emitting a now-stale turn."""
        if not text:
            return
        if self._pending_turn is not None:
            self._pending_turn = None
