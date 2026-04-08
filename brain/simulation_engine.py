"""
Experience-aware simulation: blends heuristic prediction with MemoryGraph history.

success_probability = base_prediction * blend + historical_success_rate * (1 - blend)

V7: ``SimulationMode`` selects blend — heuristic-only, hybrid, or memory-weighted.
"""

from __future__ import annotations

import hashlib
import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from brain.memory_graph import MemoryGraph
from core.profiler import profile

logger = logging.getLogger("atom.brain.simulation_engine")


class SimulationMode(str, Enum):
    """Heuristic = fast; hybrid = balanced; memory_weighted = recall-driven."""
    HEURISTIC = "heuristic"
    HYBRID = "hybrid"
    MEMORY_WEIGHTED = "memory_weighted"


def _blend_for_mode(mode: SimulationMode) -> float:
    if mode is SimulationMode.HEURISTIC:
        return 0.0
    if mode is SimulationMode.MEMORY_WEIGHTED:
        return 0.85
    return 0.5


def plan_signature(goal_id: str, plan) -> str:
    actions = "|".join(s.action for s in getattr(plan, "steps", []))
    raw = f"{goal_id}:{actions}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class SimulationEngine:
    """Simulates a plan; predictions are grounded in similar episodic memories when available."""

    def __init__(
        self,
        memory: Optional[MemoryGraph] = None,
        *,
        mode: SimulationMode | None = None,
        config: dict | None = None,
    ):
        self.memory = memory
        raw = (config or {}).get("v7_gpu", {}).get("simulation_mode") if config else None
        if mode is not None:
            self._mode = mode
        elif raw in ("heuristic", "hybrid", "memory_weighted"):
            self._mode = SimulationMode(raw)
        else:
            self._mode = SimulationMode.HYBRID

    @property
    def mode(self) -> SimulationMode:
        return self._mode

    def set_mode(self, mode: SimulationMode) -> None:
        self._mode = mode

    def _heuristic_simulate(self, plan, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        num_steps = len(getattr(plan, "steps", []))
        est_time = max(0.5, 2.0 * num_steps)

        risk = 0.0
        for step in getattr(plan, "steps", []):
            act = step.action.lower()
            if "kill" in act or "delete" in act:
                risk += 0.4
            elif "start_backend" in act or "run_script" in act or "run_setup" in act:
                risk += 0.15
            else:
                risk += 0.05
        risk = min(1.0, risk / max(1, num_steps))

        success_prob = 0.9
        success_prob = max(0.01, success_prob - (risk * 0.3))

        efficiency = max(0.0, 1.0 - (est_time / 60.0))
        context_alignment = 1.0
        if context:
            # Light alignment from context keys overlap (placeholder for richer fusion)
            context_alignment = min(1.0, 0.5 + 0.1 * min(len(context), 5))

        return {
            "success_probability": success_prob,
            "est_time_s": est_time,
            "risk": risk,
            "efficiency": efficiency,
            "context_alignment": context_alignment,
            "historical_success_rate": None,
            "memory_matches": 0,
            "plan_signature": plan_signature(getattr(plan, "goal_id", ""), plan),
            "trace_id": (context or {}).get("trace_id"),
        }

    def _query_similar_experiences(self, plan) -> List[Dict[str, Any]]:
        if not self.memory:
            return []
        text = " ".join(s.action for s in getattr(plan, "steps", []))
        if not text.strip():
            return []
        try:
            nodes = self.memory.query(
                {"text": text, "type": "episodic", "query_type": "plan"},
                limit=8,
            )
            out: List[Dict[str, Any]] = []
            for n in nodes:
                d = n.data if isinstance(n.data, dict) else {}
                if d.get("kind") == "plan_execution" or "success_score" in d:
                    out.append(d)
            return out
        except Exception as e:
            logger.debug("Memory query for simulation failed: %s", e)
            return []

    @profile("simulation")
    def simulate(self, plan, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        memory_blend = _blend_for_mode(self._mode)
        base = self._heuristic_simulate(plan, context)
        experiences = self._query_similar_experiences(plan)
        base["memory_matches"] = len(experiences)
        base["simulation_mode"] = self._mode.value

        if not experiences:
            base["historical_success_rate"] = None
            base["trace_id"] = (context or {}).get("trace_id")
            return base

        scores = [float(e.get("success_score", 0.0)) for e in experiences]
        times = [float(e.get("execution_time_s", 0.0)) for e in experiences if e.get("execution_time_s")]
        hist_rate = sum(scores) / max(1, len(scores))
        base["historical_success_rate"] = hist_rate

        if times:
            avg_t = sum(times) / len(times)
            base["est_time_s"] = 0.5 * base["est_time_s"] + 0.5 * max(0.5, avg_t)

        # Blend success probability with history
        sp = base["success_probability"]
        base["success_probability"] = (1.0 - memory_blend) * sp + memory_blend * hist_rate
        base["efficiency"] = max(0.0, 1.0 - (base["est_time_s"] / 60.0))

        # If history shows failures, slightly raise perceived risk
        if hist_rate < 0.4:
            base["risk"] = min(1.0, base["risk"] + 0.15)

        base["plan_signature"] = plan_signature(getattr(plan, "goal_id", ""), plan)
        base["trace_id"] = (context or {}).get("trace_id")
        return base
