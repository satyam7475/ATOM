import asyncio
import copy
import logging
import random
from typing import Any, Dict, List, Optional

from brain.planning_engine import Plan
from brain.simulation_engine import SimulationEngine
from brain.exploration_engine import ExplorationEngine
from brain.learning_engine import LearningEngine
from core.telemetry_engine import TelemetryEngine
from core.runtime_config import SystemMode, get_system_mode

logger = logging.getLogger("atom.brain.plan_evaluator")


class PlanEvaluator:
    """Multi-plan generation, parallel simulation, scoring with learnable weights."""

    def __init__(
        self,
        planning_engine,
        simulation: SimulationEngine,
        exploration: Optional[ExplorationEngine] = None,
        learning: Optional[LearningEngine] = None,
        telemetry: Optional[TelemetryEngine] = None,
    ):
        self.planning_engine = planning_engine
        self.simulation = simulation
        self.exploration = exploration or ExplorationEngine()
        self.learning = learning
        self.telemetry = telemetry

    def _weights(self) -> Dict[str, float]:
        if self.learning:
            return self.learning.get_plan_score_weights()
        return {
            "w_success": 0.4,
            "w_efficiency": 0.2,
            "w_context": 0.2,
            "w_risk": 0.2,
        }

    def _score(self, sim_result: Dict[str, Any]) -> float:
        w = self._weights()
        return (
            sim_result["success_probability"] * w["w_success"]
            + sim_result["efficiency"] * w["w_efficiency"]
            + sim_result["context_alignment"] * w["w_context"]
            - sim_result["risk"] * w["w_risk"]
        )

    def _apply_deterministic_seed(self, goal_id: str, objective: str, seed: int) -> None:
        random.seed(seed)

    def generate_variants(self, goal_id: str, objective: str, context: Optional[Dict[str, Any]] = None, n: int = 3) -> List[Plan]:
        base = self.planning_engine.generate_plan(goal_id=goal_id, objective=objective, context=context)
        variants: List[Plan] = [base]

        try:
            auto = copy.deepcopy(base)
            for step in auto.steps:
                if step.action == "start_backend":
                    step.action = "run_setup_script"
                    step.skill = "shell_execution"
            variants.append(auto)
        except Exception:
            pass

        while len(variants) < n:
            variants.append(copy.deepcopy(base))

        return variants[:n]

    async def evaluate_async(
        self,
        goal_id: str,
        objective: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        trace_id: Optional[str] = None,
        deterministic_seed: Optional[int] = None,
        system_mode: Optional[SystemMode] = None,
    ) -> Dict[str, Any]:
        mode = system_mode if system_mode is not None else get_system_mode()
        seed = deterministic_seed if deterministic_seed is not None else hash(f"{goal_id}:{objective}") % (2**31)
        self._apply_deterministic_seed(goal_id, objective, seed)

        plans = self.generate_variants(goal_id, objective, context=context, n=3)
        ctx = dict(context or {})
        ctx["trace_id"] = trace_id
        ctx["system_mode"] = mode.value if hasattr(mode, "value") else str(mode)

        async def sim_one(p: Plan) -> Dict[str, Any]:
            def _run():
                return self.simulation.simulate(p, ctx)

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, _run)

        sim_results = await asyncio.gather(*[sim_one(p) for p in plans])
        for s in sim_results:
            if trace_id:
                s["trace_id"] = trace_id

        scored: List[Dict[str, Any]] = []
        for p, sim in zip(plans, sim_results):
            score = self._score(sim)
            scored.append({"plan": p, "score": score, "sim": sim})
            if self.telemetry:
                self.telemetry.record_metric("plan_score", score)

        scored.sort(key=lambda x: x["score"], reverse=True)
        best = scored[0]

        if self.exploration and self.exploration.should_explore(system_mode=mode):
            if len(scored) > 1:
                choice = random.choice(scored[1:])
                logger.info("Exploration chose alternative plan trace_id=%s", trace_id)
                choice["trace_id"] = trace_id
                choice["deterministic_seed"] = seed
                return choice

        best["trace_id"] = trace_id
        best["deterministic_seed"] = seed
        return best

    def evaluate_sync(
        self,
        goal_id: str,
        objective: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Sync entry point (e.g. tests): runs evaluate_async in a new loop."""
        return asyncio.run(self.evaluate_async(goal_id, objective, context=context, **kwargs))
