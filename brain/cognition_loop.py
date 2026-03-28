"""
ATOM V7 — Unified cognition loop (Jarvis-style chain).

Perception → context → prediction → action evaluation → (plan / execute / reflect
are delegated to PlanningEngine / ExecutionEngine / ReflectionEngine elsewhere).

This module provides a single orchestration surface for ``LocalCognitivePipeline``
so routing and brain_orchestrator can share one narrative flow.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from brain.local_cognitive_pipeline import LocalCognitivePipeline

logger = logging.getLogger("atom.brain.cognition_loop")


class CognitionLoop:
    """One turn through intent, fused context, lightweight next-step prediction."""

    def __init__(
        self,
        memory_graph: Any = None,
        local_pipeline: LocalCognitivePipeline | None = None,
    ) -> None:
        self._local = local_pipeline or LocalCognitivePipeline(memory_graph)

    def run_turn(
        self,
        text: str,
        system_state: Dict[str, Any],
        memory: Any = None,
    ) -> Dict[str, Any]:
        decision_path: list[str] = ["perception"]
        intent = self._local.classify_intent(text)
        decision_path.append("intent_classification")

        ctx = self._local.build_context(intent, system_state, memory)
        decision_path.append("context_fusion")

        action_eval = self._local.evaluate_action(intent, ctx)
        decision_path.append("prediction")

        prediction = {
            "primary_intent": intent.type,
            "score": action_eval.get("score"),
            "approved": action_eval.get("approved"),
            "notes": "Lightweight utility/risk from ProactiveEngine",
        }

        return {
            "text": text,
            "intent": self._local.intent_to_dict(intent),
            "context": ctx,
            "prediction": prediction,
            "action_eval": action_eval,
            "decision_path": decision_path,
        }


__all__ = ["CognitionLoop"]
