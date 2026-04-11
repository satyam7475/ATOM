"""
ATOM -- Agentic Brain Controller (LLM + Tool Use + ReAct Loop).

The brain of ATOM. Runs a local LLM that can:
  1. Respond with natural language (conversation)
  2. Call tools to perform system actions (tool use)
  3. See tool results and decide next actions (ReAct loop)
  4. Chain multiple actions for complex requests

Flow:
  Query -> Build Prompt -> LLM generates response
    -> Parse for tool calls
      -> If tool calls found: execute via ActionExecutor, collect observations
         -> Feed observations back to LLM (up to MAX_REACT_STEPS)
      -> If no tool calls: emit text response for TTS

This is what makes ATOM a JARVIS-level system instead of a regex remote control.
The LLM REASONS about what to do, not just pattern-match.

Event contract:
  Emits: partial_response, cursor_response, metrics_latency, metrics_event,
         tool_executed, plan_started, plan_step_complete
  On error: response_ready, llm_error
"""

from __future__ import annotations

import asyncio
import logging
import queue
import re
import time
import uuid
from collections import deque
from functools import partial
from typing import TYPE_CHECKING, Any, Dict

from core.reasoning.tool_parser import parse_tool_calls
from core.runtime.v7_context import V7RuntimeContext

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.brain_mode_manager import BrainModeManager
    from core.reasoning.action_executor import ActionExecutor
    from cursor_bridge.structured_prompt_builder import StructuredPromptBuilder

logger = logging.getLogger("atom.local_brain")

_SENTENCE_BOUNDARY = re.compile(r"[.!?]\s")
_SENTENCE_END = re.compile(r"[.!?]$")

MAX_REACT_STEPS = 3


class LocalBrainController:
    """Agentic LLM brain with tool-use and ReAct reasoning loop.

    The LLM sees all available tools in its prompt. When it decides
    an action is needed, it outputs tool_call tags. These are parsed,
    executed through the security-gated ActionExecutor, and the results
    are fed back as observations for the next reasoning step.
    """

    def __init__(
        self,
        bus: "AsyncEventBus",
        prompt_builder: "StructuredPromptBuilder",
        config: dict,
        brain_mode_manager: "BrainModeManager | None" = None,
    ) -> None:
        self._bus = bus
        self._prompt_builder = prompt_builder
        self._config = config

        from brain.mlx_llm import MLXBrain
        self._llm = MLXBrain(config)
        if brain_mode_manager is not None:
            self._llm.set_brain_mode_manager(brain_mode_manager)

        self._action_executor: ActionExecutor | None = None

        self._total_calls = 0
        self._total_tokens_approx = 0
        self._total_tool_calls = 0
        self._total_react_loops = 0
        self._first_token_latencies: list[float] = []

        self._inference_guard: Any = None

        self._rag_engine: Any = None
        self._gpu_coord: Any = None
        _rc = config.get("rag") or {}
        self._rag_budget_ms = float(_rc.get("first_token_budget_ms", 120))

        self._timeline: Any = None
        self._mode_resolver: Any = None
        self._prefetch_engine: Any = None
        self._memory_graph: Any = None
        self._recent_queries: deque[str] = deque(maxlen=12)
        self._current_runtime_mode: str = "SMART"
        self._last_mode_info: Dict[str, Any] = {}
        self._feedback_engine: Any = None
        self._system_monitor: Any = None
        self._suggester: Any = None
        self._runtime_watchdog: Any = None
        self._prev_predictions: list[str] = []
        self._last_retrieval_source: str = ""
        self._v7_context_last: V7RuntimeContext | None = None

    def attach_feedback_engine(self, engine: Any) -> None:
        self._feedback_engine = engine

    def attach_system_monitor(self, monitor: Any) -> None:
        self._system_monitor = monitor

    def attach_suggester(self, suggester: Any) -> None:
        self._suggester = suggester

    def attach_timeline(self, timeline: Any) -> None:
        self._timeline = timeline

    def attach_mode_resolver(self, resolver: Any) -> None:
        self._mode_resolver = resolver

    def attach_prefetch_engine(self, engine: Any) -> None:
        self._prefetch_engine = engine

    def attach_memory_graph(self, graph: Any) -> None:
        self._memory_graph = graph

    def attach_rag(
        self,
        rag_engine: Any = None,
        gpu_coordinator: Any = None,
    ) -> None:
        """Wire low-latency RAG (optional). ``gpu_coordinator`` for observability snapshot."""
        self._rag_engine = rag_engine
        self._gpu_coord = gpu_coordinator

    async def _retry_with_late_rag(
        self,
        text: str,
        memory_context: list[str] | None,
        context: dict[str, str] | None,
        history: list[tuple[str, str]] | None,
        res: Any,
        trace_id: str | None,
        query_plan: Any | None = None,
        late_depth: int = 0,
    ) -> None:
        """Single follow-up generation with high-confidence RAG after budget miss."""
        await asyncio.sleep(0.06)
        await self.on_query(
            text,
            memory_context=memory_context,
            context=context,
            history=history or [],
            trace_id=trace_id,
            query_plan=query_plan,
            enriched_rag_result=res,
            _retry_from_late=True,
            _late_depth=late_depth + 1,
        )

    def attach_gpu_resource_manager(self, mgr: Any) -> None:
        """Legacy shim — use attach_inference_guard instead."""
        self._inference_guard = mgr

    def attach_inference_guard(self, guard: Any) -> None:
        self._inference_guard = guard

    def attach_runtime_watchdog(self, watchdog: Any) -> None:
        self._runtime_watchdog = watchdog

    def apply_memory_pressure(self, memory_pct: float) -> None:
        if self._rag_engine is not None:
            try:
                self._rag_engine.apply_memory_pressure(memory_pct)
            except Exception:
                logger.debug("Local brain RAG pressure hook failed", exc_info=True)
        if self._memory_graph is not None:
            try:
                self._memory_graph.apply_memory_pressure(memory_pct)
            except Exception:
                logger.debug("Local brain MemoryGraph pressure hook failed", exc_info=True)

    def set_action_executor(self, executor: "ActionExecutor") -> None:
        """Inject the ActionExecutor after Router initialization."""
        self._action_executor = executor
        logger.info("Action executor connected to brain controller")

    @property
    def available(self) -> bool:
        return self._llm.is_available()

    @property
    def is_loaded(self) -> bool:
        return self._llm.is_loaded

    def request_preempt(self) -> None:
        self._llm.request_abort_preempt()

    def unload_llm_for_power(self) -> None:
        """V7: release model memory (next query will reload)."""
        self._llm.shutdown()

    async def warm_up(
        self,
        *,
        model_role: str | None = None,
        load_all: bool = False,
    ) -> bool:
        if not self._llm.is_available():
            logger.warning(
                "Local brain not available "
                "(MLX model directory missing or mlx/mlx_lm not installed)",
            )
            return False
        role_label = "all_roles" if load_all else (model_role or "default_role")
        logger.info("Local brain: warming up MLX model (%s)...", role_label)
        loop = asyncio.get_running_loop()
        t0 = time.monotonic()
        loaded = await loop.run_in_executor(
            None,
            partial(self._llm.preload, model_role=model_role, load_all=load_all),
        )
        elapsed = (time.monotonic() - t0) * 1000
        if loaded:
            logger.info(
                "Local brain ready in %.0fms (MLX, role=%s, agentic mode)",
                elapsed,
                role_label,
            )
        else:
            logger.warning("Local brain warm-up failed (role=%s)", role_label)
        return bool(loaded)

    async def _handle_query_failure(self, source: str, exc: Exception) -> None:
        logger.exception("Local brain %s failed: %s", source, exc)
        try:
            self._bus.emit("metrics_event", counter="errors_total")
        except Exception:
            pass
        try:
            self._bus.emit("llm_error", source="local", error=str(exc)[:200])
        except Exception:
            pass
        try:
            self._bus.emit_long(
                "response_ready",
                text="Local brain hit an error, Boss. Check the log and try again.",
            )
        except Exception:
            logger.debug("Local brain fallback response failed", exc_info=True)

    async def on_query(
        self,
        text: str,
        memory_context: list[str] | None = None,
        context: dict[str, str] | None = None,
        history: list[tuple[str, str]] | None = None,
        **_kw: Any,
    ) -> None:
        try:
            await self._on_query_impl(
                text,
                memory_context=memory_context,
                context=context,
                history=history,
                **_kw,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._handle_query_failure("on_query", exc)

    async def _on_query_impl(
        self,
        text: str,
        memory_context: list[str] | None = None,
        context: dict[str, str] | None = None,
        history: list[tuple[str, str]] | None = None,
        **_kw: Any,
    ) -> None:
        """Process a query through the agentic LLM with ReAct loop.

        1. Build prompt with tools visible
        2. LLM generates response (streamed)
        3. Parse response for tool calls
        4. If tools called: execute, collect observations, re-prompt LLM
        5. Repeat up to MAX_REACT_STEPS
        6. Emit final text response for TTS
        """
        if not self._llm.is_available():
            self._bus.emit_long(
                "response_ready",
                text=(
                    "Local brain is not available. Check that the MLX model "
                    "directories exist and that mlx/mlx_lm are installed."
                ),
            )
            return

        from context.privacy_filter import redact as _redact
        logger.info("Agentic brain query: '%s'", _redact(text[:120]))

        trace_id = _kw.get("trace_id")
        if self._inference_guard is not None:
            self._inference_guard.refresh_vram()

        if self._feedback_engine is not None and self._prev_predictions:
            try:
                self._feedback_engine.evaluate_actual_vs_predictions(text, self._prev_predictions)
            except Exception:
                pass

        query_plan = _kw.get("query_plan")
        plan_mode = str(getattr(query_plan, "runtime_mode", "") or "").upper()
        if plan_mode not in {"FAST", "SMART", "DEEP", "SECURE"}:
            plan_mode = ""
        plan_model_role = str(getattr(query_plan, "model_role", "") or "").strip().lower()
        if plan_model_role not in {"fast", "primary"}:
            plan_model_role = None
        plan_use_rag = getattr(query_plan, "use_rag", None) if query_plan is not None else None
        plan_use_memory = getattr(query_plan, "use_memory", None) if query_plan is not None else None
        plan_prompt_hint = str(getattr(query_plan, "prompt_hint", "") or "").strip()
        plan_reduce_context = bool(getattr(query_plan, "reduce_context", False)) if query_plan is not None else False
        plan_memory_limit = int(getattr(query_plan, "memory_limit", 0) or 0) if query_plan is not None else 0
        plan_history_turn_limit = int(getattr(query_plan, "history_turn_limit", 0) or 0) if query_plan is not None else 0
        plan_rag_budget_ms = float(getattr(query_plan, "rag_budget_ms", 0.0) or 0.0) if query_plan is not None else 0.0
        plan_budget_tier = str(getattr(query_plan, "budget_tier", "") or "").strip().lower() if query_plan is not None else ""
        plan_requested_tier = str(getattr(query_plan, "requested_tier", "") or "").strip().lower() if query_plan is not None else ""
        plan_base_budget_ms = float(getattr(query_plan, "base_budget_ms", 0.0) or 0.0) if query_plan is not None else 0.0

        mode_override = _kw.get("runtime_mode_override") or (plan_mode or None)
        if query_plan is not None:
            try:
                logger.info(
                    "Local brain plan: tier=%s requested=%s path=%s role=%s mode=%s rag=%s think=%s base=%.0fms budget=%.0fms rag_budget=%.0fms reduce_context=%s reason=%s",
                    plan_budget_tier or "?",
                    plan_requested_tier or "?",
                    getattr(getattr(query_plan, "path", None), "value", getattr(query_plan, "path", "")),
                    plan_model_role or "primary",
                    plan_mode or "SMART",
                    getattr(query_plan, "use_rag", False),
                    getattr(query_plan, "thinking", False),
                    plan_base_budget_ms,
                    float(getattr(query_plan, "budget_ms", 0.0) or 0.0),
                    plan_rag_budget_ms,
                    plan_reduce_context,
                    getattr(query_plan, "reason", ""),
                )
            except Exception:
                logger.debug("Local brain query_plan logging failed", exc_info=True)

        gpu_util = 0.0
        if self._gpu_coord is not None:
            try:
                obs = self._gpu_coord.get_observability()
                gs = obs.get("gpu_state") or {}
                gpu_util = float(gs.get("gpu_util_pct", 0) or 0)
            except Exception:
                gpu_util = 0.0

        system_state: dict[str, Any] = {}
        if self._system_monitor is not None:
            try:
                system_state = self._system_monitor.get_system_state()
            except Exception:
                system_state = {}

        user_activity = "active"
        if self._timeline is not None:
            try:
                user_activity = (
                    "idle" if not self._timeline.user_recently_active(120.0) else "active"
                )
            except Exception:
                user_activity = "active"

        pred_acc: float | None = None
        feedback_metrics: dict[str, Any] = {}
        if self._feedback_engine is not None:
            try:
                feedback_metrics = self._feedback_engine.compute_accuracy_metrics()
                pred_acc = float(feedback_metrics.get("prediction_accuracy", 0.5))
            except Exception:
                pred_acc = None

        timeline_summary = ""
        if self._timeline is not None:
            try:
                timeline_summary = self._timeline.summary_for_prompt(
                    window_sec=600.0, max_lines=3,
                )
            except Exception:
                timeline_summary = ""

        mode_ctx = V7RuntimeContext(
            system_state=dict(system_state),
            feedback_metrics=dict(feedback_metrics),
            runtime_mode=self._current_runtime_mode,
            mode_info={},
            timeline_summary=timeline_summary,
            gpu_util_pct=gpu_util,
            prediction_accuracy=pred_acc,
            last_retrieval_source=self._last_retrieval_source,
        )

        if self._mode_resolver is not None:
            self._current_runtime_mode, self._last_mode_info = self._mode_resolver.resolve(
                text,
                gpu_util_pct=gpu_util,
                user_override=mode_override,
                system_state=system_state,
                user_activity=user_activity,
                prediction_accuracy=pred_acc,
                context=mode_ctx,
            )
        else:
            self._current_runtime_mode = mode_override or "SMART"
            self._last_mode_info = {"reason": "query_plan"} if plan_mode else {}

        v7_ctx = mode_ctx.with_mode(self._current_runtime_mode, self._last_mode_info)
        self._v7_context_last = v7_ctx

        if self._timeline is not None:
            try:
                self._timeline.append_event(
                    "llm_query",
                    {"text": text[:2000], "runtime_mode": self._current_runtime_mode},
                )
            except Exception:
                pass
        self._recent_queries.append(text.strip())

        t0_total = time.perf_counter()
        observations: list[str] = []
        all_tool_results: list[str] = []
        text_response_parts: list[str] = []
        react_step = 0
        tool_depth = 0
        MAX_TOOL_DEPTH = 3

        prompt_memory_context = memory_context if plan_use_memory is not False else None
        if prompt_memory_context and plan_memory_limit > 0:
            prompt_memory_context = list(prompt_memory_context[:plan_memory_limit])
        prompt_history = list(history or [])
        if plan_history_turn_limit > 0:
            prompt_history = prompt_history[-plan_history_turn_limit:]
        prompt_context: dict[str, str] | None = context
        if plan_prompt_hint:
            prompt_context = dict(context or {})
            prompt_context["llm_routing_hint"] = plan_prompt_hint

        rag_document_context: list[str] | None = None
        rag_enrichment: str | None = None
        enriched_in = _kw.get("enriched_rag_result")
        if enriched_in is not None:
            rag_document_context = enriched_in.document_context
            rag_enrichment = enriched_in.enrichment_block or None
        elif (
            self._rag_engine is not None
            and not _kw.get("_retry_from_late")
            and plan_use_rag is not False
        ):
            try:
                from core.rag.query_classifier import classify_query
                from core.rag.rag_engine import RagEngine, retrieve_with_time_budget

                gpu_snap: dict | None = None
                vram_p = 0.0
                if self._gpu_coord is not None:
                    try:
                        obs = self._gpu_coord.get_observability()
                        gpu_snap = obs.get("gpu_state") or {}
                        vram_p = float(gpu_snap.get("fragmentation_estimate", 0) or 0)
                    except Exception:
                        gpu_snap = None

                cx = classify_query(text)
                prefetch_hit_guess = False
                try:
                    if self._rag_engine is not None:
                        prefetch_hit_guess = self._rag_engine._caches.get_retrieval(text) is not None  # noqa: SLF001
                except Exception:
                    prefetch_hit_guess = False
                budget_ms = RagEngine.compute_budget_ms(
                    self._config,
                    cx,
                    gpu_util_pct=gpu_util,
                    vram_pressure=vram_p,
                    prefetch_hit=prefetch_hit_guess,
                )
                if self._runtime_watchdog is not None:
                    budget_ms = self._runtime_watchdog.cap_budget_ms(
                        "rag_retrieval",
                        budget_ms,
                    )
                if plan_rag_budget_ms > 0:
                    budget_ms = min(budget_ms, plan_rag_budget_ms)
                if self._current_runtime_mode == "SECURE":
                    budget_ms *= float(
                        (self._config.get("v7_intelligence") or {}).get(
                            "secure_rag_budget_factor", 0.75,
                        ),
                    )
                if plan_reduce_context:
                    budget_ms *= 0.85
                late_thr = float(
                    (self._config.get("rag") or {}).get("late_restart_confidence", 0.82),
                )
                late_depth = int(_kw.get("_late_depth", 0))
                max_pre = int(
                    ((self._config.get("v7_intelligence") or {}).get("preemption") or {}).get(
                        "max_preemptions_per_query",
                        2,
                    ),
                )

                def _late_rag(res: Any) -> None:
                    from core.cognition.preemption import should_preempt_for_late_rag

                    if late_depth >= max_pre:
                        logger.info(
                            "v7_preemption_blocked reason=max_retries depth=%d max=%d",
                            late_depth,
                            max_pre,
                        )
                        return

                    self._bus.emit_fast(
                        "rag_context_ready",
                        chunks=len(res.chunks),
                        latency_ms=res.latency_ms,
                        confidence=getattr(res, "confidence", 0.0),
                        trace_id=trace_id,
                    )
                    if getattr(res, "confidence", 0) < late_thr or len(res.chunks) < 2:
                        return
                    if not should_preempt_for_late_rag(
                        res,
                        baseline_confidence=0.0,
                        config=self._config,
                    ):
                        return
                    self._bus.emit_fast(
                        "rag_late_high_confidence",
                        confidence=res.confidence,
                        trace_id=trace_id,
                    )
                    self.request_preempt()
                    asyncio.create_task(
                        self._retry_with_late_rag(
                            text,
                            memory_context,
                            context,
                            history,
                            res,
                            trace_id,
                            query_plan=query_plan,
                            late_depth=late_depth,
                        ),
                        name="atom_rag_late_restart",
                    )

                late_cb = None if self._current_runtime_mode == "SECURE" else _late_rag

                rag_res = await retrieve_with_time_budget(
                    self._rag_engine,
                    text,
                    budget_ms,
                    memory_summaries=memory_context,
                    system_state=None,
                    gpu_snapshot=gpu_snap,
                    runtime_mode=self._current_runtime_mode,
                    on_late_result=late_cb,
                )
                try:
                    logger.info(
                        "v7_rag_retrieval mode=%s prefetch_guess=%s chunks=%d source=%s",
                        self._current_runtime_mode,
                        prefetch_hit_guess,
                        len(rag_res.chunks),
                        getattr(rag_res, "retrieval_source", ""),
                    )
                except Exception:
                    pass
                self._last_retrieval_source = str(
                    getattr(rag_res, "retrieval_source", "") or "",
                )
                if self._feedback_engine is not None:
                    try:
                        if getattr(rag_res, "prefetch_hit", False):
                            self._feedback_engine.record_prefetch_event(True)
                        else:
                            self._feedback_engine.record_prefetch_event(False)
                    except Exception:
                        pass
                if rag_res.chunks:
                    rag_document_context = rag_res.document_context
                    rag_enrichment = rag_res.enrichment_block or None
            except Exception:
                logger.debug("RAG retrieve skipped", exc_info=True)

        while react_step <= MAX_REACT_STEPS:
            prompt = self._prompt_builder.build(
                text,
                memory_summaries=prompt_memory_context,
                history=prompt_history,
                context=prompt_context,
                document_context=rag_document_context if react_step == 0 else None,
                observations=observations if observations else None,
                rag_enrichment=rag_enrichment if react_step == 0 else None,
            )

            raw_response, first_token_ms, preempted = await self._run_llm_streaming(
                prompt,
                t0_total,
                emit_partial=(react_step == 0 and not observations),
                model_role=plan_model_role,
            )

            if preempted:
                logger.info("Brain preempted at step %d (%.0fms)",
                            react_step, (time.perf_counter() - t0_total) * 1000)
                self._bus.emit("metrics_event", counter="llm_preempted")
                return

            if not raw_response:
                break

            parsed = parse_tool_calls(raw_response)

            if parsed.text_response:
                text_response_parts.append(parsed.text_response)

            if not parsed.has_tool_calls or self._action_executor is None:
                break
                
            tool_depth += 1
            if tool_depth > MAX_TOOL_DEPTH:
                logger.warning("MAX_TOOL_DEPTH (%d) exceeded, breaking ReAct loop", MAX_TOOL_DEPTH)
                text_response_parts.append("I've hit my internal limit for tool calls on this task, Boss. I'm stopping here to prevent a loop.")
                break

            react_step += 1
            self._total_react_loops += 1
            logger.info("ReAct step %d: %d tool call(s)",
                        react_step, len(parsed.tool_calls))

            step_observations: list[str] = []
            for tc in parsed.tool_calls:
                if self._runtime_watchdog is not None:
                    from core.reasoning.action_executor import ActionResult

                    tool_result = await self._runtime_watchdog.run_async(
                        "tool_execution",
                        self._action_executor.execute_async(tc),
                        default=ActionResult(
                            tool_name=tc.name,
                            success=False,
                            error="Tool execution timed out.",
                        ),
                        metadata={"tool": tc.name},
                    )
                    result = tool_result.value
                else:
                    result = await self._action_executor.execute_async(tc)
                self._total_tool_calls += 1
                step_observations.append(result.observation)
                all_tool_results.append(result.observation)

                self._bus.emit("tool_executed",
                               tool=tc.name, success=result.success,
                               elapsed_ms=result.elapsed_ms)

                if result.needs_confirmation:
                    confirm_text = (
                        f"{parsed.text_response} " if parsed.text_response else ""
                    ) + result.confirmation_prompt
                    self._bus.emit_long("response_ready", text=confirm_text)

                    self._bus.emit("pending_tool_confirmation",
                                   tool_call=tc, result=result)
                    self._emit_final_metrics(
                        t0_total, first_token_ms,
                        " ".join(text_response_parts),
                        trace_id=trace_id,
                    )
                    return

            observations.extend(step_observations)

            if react_step >= MAX_REACT_STEPS:
                logger.info("ReAct loop hit max steps (%d)", MAX_REACT_STEPS)
                break

        elapsed_total = (time.perf_counter() - t0_total) * 1000

        full_text = " ".join(text_response_parts).strip()

        if not full_text and all_tool_results:
            success_results = [r for r in all_tool_results if r.startswith("[OK]")]
            if success_results:
                full_text = ". ".join(
                    r.replace("[OK] ", "").split(": ", 1)[-1]
                    for r in success_results
                )

        if not full_text:
            logger.warning("Brain returned empty response (%.0fms)", elapsed_total)
            self._bus.emit_long(
                "response_ready",
                text="My brain couldn't process that, Boss. Try rephrasing.",
            )
            self._bus.emit("llm_error", source="local", error="empty_response")
            return

        if react_step > 0:
            self._bus.emit_long(
                "partial_response",
                text=full_text,
                is_first=True,
                is_last=True,
                source="local",
            )

        self._emit_final_metrics(t0_total, first_token_ms, full_text, trace_id=trace_id)

        self._bus.emit(
            "cursor_response",
            query=text.lower().strip(),
            response=full_text,
        )

        try:
            from core.cognition.predictor import predict_next_queries
            from core.rag.prefetch_engine import (
                RagPrefetchEngine,
                merge_prefetch_candidates,
                predict_followup_queries,
            )
            if self._rag_engine is not None and self._current_runtime_mode != "SECURE":
                v7 = self._config.get("v7_intelligence") or {}
                if bool(v7.get("prediction_prefetch_enabled", True)):
                    tsnips: list[str] = []
                    active_task: dict[str, Any] | str | None = None
                    if self._timeline is not None:
                        try:
                            tsnips = self._timeline.context_snippets_for_prediction()
                            active_task = self._timeline.get_last_active_task()
                        except Exception:
                            pass
                    last_proj = None
                    recent_ent: list[dict[str, Any]] = []
                    if self._memory_graph is not None:
                        try:
                            last_proj = self._memory_graph.get_last_active_project()
                            recent_ent = self._memory_graph.get_recent_entities(8)
                        except Exception:
                            pass
                    pred_ctx = {
                        "last_queries": list(self._recent_queries),
                        "active_task": active_task,
                        "recent_actions": tsnips,
                        "timeline_snippets": tsnips,
                        "last_project": last_proj,
                        "recent_entities": recent_ent,
                        "feedback_engine": self._feedback_engine,
                    }
                    predicted = predict_next_queries(pred_ctx)
                    legacy = predict_followup_queries(text, history or [])
                    pf_cfg = (v7.get("prefetch") or {})
                    max_pf = int(pf_cfg.get("max_prefetch_candidates", 12))
                    merged = merge_prefetch_candidates(
                        predicted, legacy, max_candidates=max_pf,
                    )
                    if self._timeline is not None:
                        try:
                            hint = self._timeline.suggest_next_from_pattern()
                            if hint:
                                merged = merge_prefetch_candidates(
                                    [hint], merged, max_candidates=max_pf,
                                )
                        except Exception:
                            pass
                    self._prev_predictions = list(merged[:12])
                    if self._feedback_engine is not None:
                        try:
                            self._feedback_engine.record_prefetch_scheduled(len(merged))
                        except Exception:
                            pass
                    eng = self._prefetch_engine or RagPrefetchEngine(
                        self._rag_engine, self._config,
                    )
                    eng.schedule_fire_and_forget(
                        merged,
                        gpu_util_pct=gpu_util,
                        prediction_accuracy=pred_acc,
                    )
        except Exception:
            pass

        try:
            if self._suggester is not None and self._timeline is not None:
                acc = 0.5
                if self._feedback_engine is not None:
                    acc = float(
                        self._feedback_engine.compute_accuracy_metrics().get(
                            "prediction_accuracy", 0.5,
                        ),
                    )
                for sug in self._suggester.produce(
                    self._timeline,
                    prediction_accuracy=acc,
                    last_query=text,
                ):
                    self._bus.emit_fast("v7_suggestion", text=sug)
        except Exception:
            pass

    async def _run_llm_streaming(
        self,
        prompt: str,
        t0_total: float,
        *,
        emit_partial: bool = True,
        model_role: str | None = None,
        _watchdog_guard: bool = False,
    ) -> tuple[str, float, bool]:
        """Run LLM inference with optional streaming to TTS.

        Returns (full_response_text, first_token_ms, was_preempted).
        When emit_partial=True, sentences are streamed to TTS in real-time.
        When False (ReAct follow-up), we collect silently.
        """
        if self._runtime_watchdog is not None and not _watchdog_guard:
            watched = await self._runtime_watchdog.run_async(
                "llm_inference",
                self._run_llm_streaming(
                    prompt,
                    t0_total,
                    emit_partial=emit_partial,
                    model_role=model_role,
                    _watchdog_guard=True,
                ),
                default=("", 0.0, True),
                metadata={"prompt_chars": len(prompt)},
            )
            return watched.value

        loop = asyncio.get_running_loop()
        token_queue: asyncio.Queue[tuple[str, bool]] = asyncio.Queue()
        first_token_time: list[float | None] = [None]

        def _on_token(token_text: str, is_done: bool) -> None:
            if first_token_time[0] is None and token_text:
                first_token_time[0] = time.perf_counter()
            # Thread-safe push to asyncio queue (eliminates polling latency)
            loop.call_soon_threadsafe(token_queue.put_nowait, (token_text, is_done))

        generate_kwargs: dict[str, Any] = {"on_token": _on_token}
        if model_role:
            generate_kwargs["model_role"] = model_role
        generate_task = asyncio.create_task(
            self._llm.generate_streaming(prompt, **generate_kwargs)
        )

        stream_id = uuid.uuid4().hex
        sentence_buffer = ""
        sentences_emitted = 0
        full_response_parts: list[str] = []
        trailing_sentence = ""
        try:
            while True:
                # Wait for the next token without polling
                get_task = asyncio.create_task(token_queue.get())
                done, pending = await asyncio.wait(
                    [get_task, generate_task],
                    return_when=asyncio.FIRST_COMPLETED
                )

                if get_task in done:
                    token_text, is_done = get_task.result()
                else:
                    # generate_task finished but queue might not be empty
                    get_task.cancel()
                    try:
                        token_text, is_done = token_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                if is_done:
                    trailing_sentence = sentence_buffer.strip()
                    break

                if not token_text:
                    continue

                sentence_buffer += token_text

                if emit_partial:
                    ready = self._extract_complete_sentence(sentence_buffer)
                    if ready:
                        sentence_text, remainder = ready
                        sentences_emitted += 1
                        self._bus.emit_long(
                            "partial_response",
                            text=sentence_text.strip(),
                            is_first=(sentences_emitted == 1),
                            is_last=False,
                            source="local",
                            stream_id=stream_id,
                        )
                        full_response_parts.append(sentence_text.strip())
                        sentence_buffer = remainder

            result = await generate_task
            answer, preempted = result

            if trailing_sentence and not preempted:
                sentences_emitted += 1
                if emit_partial:
                    self._bus.emit_long(
                        "partial_response",
                        text=trailing_sentence,
                        is_first=(sentences_emitted == 1),
                        is_last=True,
                        source="local",
                        stream_id=stream_id,
                    )
                full_response_parts.append(trailing_sentence)

            full_text = " ".join(full_response_parts) if full_response_parts else answer

            if sentences_emitted == 0 and full_text and emit_partial and not preempted:
                self._bus.emit_long(
                    "partial_response",
                    text=full_text,
                    is_first=True,
                    is_last=True,
                    source="local",
                    stream_id=stream_id,
                )

            first_token_ms = (
                (first_token_time[0] - t0_total) * 1000
                if first_token_time[0] is not None
                else 0.0
            )

            return full_text, first_token_ms, preempted
        except asyncio.CancelledError:
            generate_task.cancel()
            try:
                await generate_task
            except Exception:
                pass
            raise

    def _emit_final_metrics(
        self,
        t0: float,
        first_token_ms: float,
        full_text: str,
        trace_id: str | None = None,
    ) -> None:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._total_calls += 1
        word_count = len(full_text.split()) if full_text else 0
        self._total_tokens_approx += word_count

        if first_token_ms > 0:
            self._first_token_latencies.append(first_token_ms)

        self._bus.emit("metrics_latency", name="llm", ms=elapsed_ms)
        if first_token_ms > 0:
            self._bus.emit("metrics_latency", name="llm_first_token", ms=first_token_ms)

        try:
            from core.unified_trace import new_trace
            ut = new_trace(trace_id)
            ut.latency_ms["llm_total"] = elapsed_ms
            if first_token_ms > 0:
                ut.latency_ms["llm_first_token"] = first_token_ms
            ut.decision_path.extend(["llm_stream", "react_loop"])
            self._bus.emit("v7_unified_trace", **ut.to_dict())
        except Exception:
            pass

        logger.info(
            "Brain: %.0fms total, %.0fms first-token, %d words, %d tool calls this turn",
            elapsed_ms, first_token_ms, word_count, self._total_tool_calls,
        )

        if self._feedback_engine is not None and self._total_calls % 25 == 0:
            try:
                fm = self._feedback_engine.compute_accuracy_metrics()
                logger.info("v7_feedback_metrics %s", fm)
            except Exception:
                pass

    @staticmethod
    def _extract_complete_sentence(buffer: str) -> tuple[str, str] | None:
        """Extract the first complete sentence from the buffer.

        A sentence ends with . ! or ? followed by a space, or
        a buffer 60+ chars ending with punctuation.
        """
        match = _SENTENCE_BOUNDARY.search(buffer)
        if match:
            split_pos = match.end()
            return buffer[:split_pos].rstrip(), buffer[split_pos:]

        if len(buffer) >= 60 and _SENTENCE_END.search(buffer.rstrip()):
            return buffer.rstrip(), ""

        return None

    def close(self) -> None:
        avg_first_token = (
            sum(self._first_token_latencies) / len(self._first_token_latencies)
            if self._first_token_latencies else 0
        )
        logger.info(
            "Brain stats: %d calls, ~%d tokens, %d tool calls, "
            "%d react loops, avg first-token %.0fms",
            self._total_calls, self._total_tokens_approx,
            self._total_tool_calls, self._total_react_loops,
            avg_first_token,
        )
        close_fn = getattr(self._llm, "close", None)
        if callable(close_fn):
            close_fn()
        else:
            self._llm.shutdown()

    def get_stats(self) -> dict:
        avg_first_token = (
            sum(self._first_token_latencies) / len(self._first_token_latencies)
            if self._first_token_latencies else 0
        )
        return {
            "available": self.available,
            "loaded": self.is_loaded,
            "total_calls": self._total_calls,
            "total_tokens_approx": self._total_tokens_approx,
            "total_tool_calls": self._total_tool_calls,
            "total_react_loops": self._total_react_loops,
            "avg_first_token_ms": round(avg_first_token, 1),
        }
