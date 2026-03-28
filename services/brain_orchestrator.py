"""
ATOM V5/V6 — Cognitive Brain Orchestrator.

V6: trace_id propagation, deterministic planning seed, critical mode (no exploration),
in-process cognitive path (optional), parallel plan simulation, batched telemetry,
execution preemption, idle maintenance, warm-start.
"""

import asyncio
import json
import logging
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ipc.zmq_bus import ZmqEventBus
from core.ipc.interrupt_manager import SystemInterruptManager
from brain.goal_engine import GoalManager, GoalType, GoalStatus, PRIORITY_USER_INTERRUPT
from brain.planning_engine import PlanningEngine
from brain.execution_engine import ExecutionEngine
from brain.learning_engine import LearningEngine, Experience
from brain.reflection_engine import ReflectionEngine
from brain.skill_engine import SkillEngine
from brain.memory_graph import MemoryGraph
from brain.local_cognitive_pipeline import LocalCognitivePipeline
from core.config_manager import load_config
from core.security_policy import SecurityPolicy
from brain.simulation_engine import SimulationEngine
from brain.plan_evaluator import PlanEvaluator
from brain.exploration_engine import ExplorationEngine
from core.telemetry_engine import TelemetryEngine
from core.runtime_config import (
    get_system_mode,
    hot_path_debug,
    use_inprocess_cognitive_path,
)

logger = logging.getLogger("atom.services.brain_orchestrator")

_RUNTIME_PATH = Path(__file__).resolve().parent.parent / "config" / "atom_runtime.json"
_IDLE_INTERVAL_S = 30.0
_TELEMETRY_FLUSH_INTERVAL_S = 2.0


class BrainOrchestrator:
    def __init__(self) -> None:
        self.bus = ZmqEventBus(worker_name="brain_orchestrator")
        self._config = load_config()

        self.skill_engine = SkillEngine()
        self.security_policy = SecurityPolicy(self._config)
        self.memory_graph = MemoryGraph()

        self.goal_manager = GoalManager()
        self.planning_engine = PlanningEngine(
            skill_registry=self.skill_engine,
            security_policy=self.security_policy,
        )
        self.execution_engine = ExecutionEngine(
            event_bus=self.bus,
            skill_engine=self.skill_engine,
            planning_engine=self.planning_engine,
        )
        self.learning_engine = LearningEngine()
        self.reflection_engine = ReflectionEngine(
            learning_engine=self.learning_engine,
            interval_minutes=5,
            planning_engine=self.planning_engine,
        )
        self.simulation_engine = SimulationEngine(
            memory=self.memory_graph,
            config=self._config,
        )
        self.telemetry = TelemetryEngine(batch_interval_s=2.0, enable_batch=True)
        self.exploration = ExplorationEngine()
        self.plan_evaluator = PlanEvaluator(
            self.planning_engine,
            self.simulation_engine,
            exploration=self.exploration,
            learning=self.learning_engine,
            telemetry=self.telemetry,
        )

        self._inprocess = use_inprocess_cognitive_path()
        self._local = LocalCognitivePipeline(self.memory_graph) if self._inprocess else None

        self.interrupt_mgr = SystemInterruptManager(self.bus, "brain_orchestrator")
        self.interrupt_mgr.register_cancel_callback(self.handle_interrupt)

        self.bus.on("speech_final", self.handle_speech_final)
        self.bus.on("system_event", self.handle_system_event)
        self.bus.on("user_interrupt", self.handle_user_interrupt)

        self._idle_task: asyncio.Task | None = None
        self._telemetry_flush_task: asyncio.Task | None = None
        self._warm_start()

    def _warm_start(self) -> None:
        try:
            self.skill_engine.preload()
            self.memory_graph.query({"type": "episodic"}, limit=1)
            logger.info("V6 warm-start complete (skills + memory touch)")
        except Exception as e:
            logger.debug("Warm-start partial: %s", e)

    async def handle_interrupt(self) -> None:
        logger.info("Brain Orchestrator received global interrupt signal.")
        self.execution_engine.request_cancel()

    async def handle_user_interrupt(self, event: str, **data) -> None:
        logger.info("user_interrupt — preempting execution trace=%s", data.get("trace_id"))
        self.execution_engine.request_cancel()
        self.telemetry.record_metric("user_interrupt", 1.0)

    async def handle_system_event(self, event: str, **data):
        pass

    def _telemetry_snapshot(self, goal, plan, sim_info: dict, success: bool) -> None:
        active = self.goal_manager.active_goal_id
        g = self.goal_manager.get_goal(active) if active else None
        failing = sorted(
            self.learning_engine.state.get("skills", {}).items(),
            key=lambda kv: kv[1].get("failure", 0),
            reverse=True,
        )[:5]
        snap = {
            "active_goal_id": active,
            "active_objective": getattr(g, "objective", None),
            "trace_id": getattr(goal, "trace_id", None),
            "plan_template": getattr(plan, "template_id", None),
            "last_sim": sim_info,
            "last_success": success,
            "reflection_insights": self.reflection_engine.insights[-8:],
            "top_failing_skills": [f"{k} (fail={v.get('failure', 0)})" for k, v in failing],
            "plan_score_weights": self.learning_engine.get_plan_score_weights(),
            "system_mode": get_system_mode().value,
        }
        self.telemetry.set_snapshot(snap)
        try:
            self.telemetry.flush()
            _RUNTIME_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "telemetry_summary": self.telemetry.summary(),
                "snapshot": snap,
                "updated_at": time.time(),
            }
            with open(_RUNTIME_PATH, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            logger.debug("Could not persist runtime snapshot: %s", e)

    async def _cognitive_path(self, text: str, system_state: dict):
        """Critical path: in-process (low latency) or ZMQ REQ (distributed)."""
        if self._local:
            intent_obj = self._local.classify_intent(text)
            context = self._local.build_context(intent_obj, system_state, self.memory_graph)
            dec = self._local.evaluate_action(intent_obj, context)
            intent_dict = self._local.intent_to_dict(intent_obj)
            return intent_dict, context, dec

        intent_res = await self.bus.request("parse_intent", timeout=2.0, text=text)
        if not intent_res:
            return None, {}, None
        context_res = await self.bus.request(
            "build_context", timeout=2.0, intent=intent_res, system_state=system_state
        )
        context = context_res.get("enriched_context", {}) if context_res else {}
        decision_res = await self.bus.request(
            "evaluate_action", timeout=2.0, intent=intent_res, context=context
        )
        return intent_res, context, decision_res

    async def handle_speech_final(self, event: str, **data):
        text = data.get("text", "")
        if not text:
            return

        trace_in = data.get("trace_id") or str(uuid.uuid4())
        urgent = bool(data.get("urgent") or data.get("user_interrupt"))
        if hot_path_debug():
            logger.info(
                "speech_final trace=%s urgent=%s text=%s",
                trace_in,
                urgent,
                text[:80],
            )
        else:
            logger.debug("speech_final len=%d", len(text))

        t0 = time.time()
        system_state = {
            "system_status": "ok",
            "time": time.time(),
            "current_time": time.strftime("%H:%M:%S"),
            "trace_id": trace_in,
        }

        if urgent:
            self.execution_engine.request_cancel()
            if self.goal_manager.active_goal_id:
                self.goal_manager.update_progress(
                    self.goal_manager.active_goal_id,
                    0.0,
                    status=GoalStatus.CANCELLED,
                )

        intent_res, context, decision_res = await self._cognitive_path(text, system_state)
        if intent_res is None:
            logger.error("Intent path failure")
            return

        context = dict(context or {})
        context["trace_id"] = trace_in

        if not (decision_res and decision_res.get("approved")):
            logger.info("Action not approved or complex query. Routing to LLM.")
            self._route_to_llm(text, context)
            self.telemetry.record_metric("orchestrator_latency_ms", (time.time() - t0) * 1000.0)
            return

        action = decision_res.get("action")
        logger.info("Decision approved action=%s trace=%s", action, trace_in)

        prio = PRIORITY_USER_INTERRUPT if urgent else 0.9
        goal = self.goal_manager.create_goal(
            objective=action,
            goal_type=GoalType.REACTIVE,
            priority=prio,
            context=context,
            trace_id=trace_in,
        )
        active_goal = self.goal_manager.select_active_goal()
        if not active_goal or active_goal.id != goal.id:
            logger.info("Goal queued. Another goal is currently active.")
            self.telemetry.record_metric("orchestrator_latency_ms", (time.time() - t0) * 1000.0)
            return

        mode = get_system_mode()
        ctx_eval = dict(context)
        ctx_eval["trace_id"] = goal.trace_id

        best_entry = await self.plan_evaluator.evaluate_async(
            goal.id,
            goal.objective,
            context=ctx_eval,
            trace_id=goal.trace_id,
            deterministic_seed=goal.deterministic_seed,
            system_mode=mode,
        )
        plan = best_entry.get("plan")
        sim_info = best_entry.get("sim", {})
        if hot_path_debug():
            logger.info(
                "plan selected score=%.3f trace=%s",
                best_entry.get("score", 0.0),
                goal.trace_id,
            )

        if not self.planning_engine.validate_plan(plan):
            logger.error("Plan validation failed trace=%s", goal.trace_id)
            self.goal_manager.update_progress(goal.id, 0.0, status=GoalStatus.FAILED)
            self.learning_engine.update_plan_score_weights_from_outcome(sim_info or {}, success=False)
            self._telemetry_snapshot(goal, plan, sim_info, False)
            self.telemetry.record_metric("orchestrator_latency_ms", (time.time() - t0) * 1000.0)
            return

        exec_t0 = time.time()
        exec_ctx = dict(context)
        exec_ctx["trace_id"] = goal.trace_id
        success = await self.execution_engine.execute_plan(plan, exec_ctx)
        exec_ms = (time.time() - exec_t0) * 1000.0

        self.goal_manager.update_progress(
            goal.id,
            1.0 if success else 0.0,
            status=GoalStatus.COMPLETED if success else GoalStatus.FAILED,
        )

        outcomes = {step.action: (step.status == "completed") for step in plan.steps}
        experience = Experience(
            goal_id=goal.id,
            objective=goal.objective,
            plan_steps=[s.action for s in plan.steps],
            outcomes=outcomes,
            success_score=1.0 if success else 0.0,
        )
        self.learning_engine.record_experience(experience)
        self.learning_engine.update_plan_score_weights_from_outcome(sim_info or {}, success=success)

        exp_record = {
            "goal_id": goal.id,
            "objective": goal.objective,
            "plan": [s.action for s in plan.steps],
            "plan_signature": (sim_info or {}).get("plan_signature"),
            "outcomes": outcomes,
            "success_score": experience.success_score,
            "execution_time_s": exec_ms / 1000.0,
            "timestamp": time.time(),
            "trace_id": goal.trace_id,
        }
        self.memory_graph.index_experience(exp_record)

        if getattr(plan, "template_id", None):
            self.planning_engine.registry.record_execution(
                plan.template_id, success, execution_time_s=exec_ms / 1000.0
            )

        self.telemetry.record_metric("goal_executed", 1.0 if success else 0.0)
        self.telemetry.record_metric("execution_time_ms", exec_ms)
        self.telemetry.record_metric("orchestrator_latency_ms", (time.time() - t0) * 1000.0)
        self._telemetry_snapshot(goal, plan, sim_info, success)

        self.bus.emit("response_ready", text=f"Executed plan for {action}")

    def _route_to_llm(self, text: str, context: dict) -> None:
        req_id = f"req_{int(time.time() * 1000)}"
        self.bus.emit("llm_query_request", req_id=req_id, text=text, context=context)

    async def _idle_loop(self) -> None:
        while True:
            await asyncio.sleep(_IDLE_INTERVAL_S)
            try:
                self.memory_graph.decay_memories(half_life_hours=24.0)
                self.memory_graph.compress_memories()
            except Exception as e:
                logger.debug("Idle memory maintenance: %s", e)

    async def _telemetry_flush_loop(self) -> None:
        while True:
            await asyncio.sleep(_TELEMETRY_FLUSH_INTERVAL_S)
            try:
                self.telemetry.flush()
            except Exception:
                pass

    async def run(self) -> None:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        logger.info(
            "Starting Brain Orchestrator (inprocess_cognitive=%s)",
            self._inprocess,
        )
        self.bus.start()
        self.reflection_engine.start()
        self._idle_task = asyncio.create_task(self._idle_loop(), name="atom_idle")
        self._telemetry_flush_task = asyncio.create_task(
            self._telemetry_flush_loop(), name="atom_telemetry_flush"
        )
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            for t in (self._idle_task, self._telemetry_flush_task):
                if t and not t.done():
                    t.cancel()
            self.reflection_engine.stop()
            self.bus.stop()


if __name__ == "__main__":
    worker = BrainOrchestrator()
    asyncio.run(worker.run())
