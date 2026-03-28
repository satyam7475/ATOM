"""
In-process cognitive path (intent → context → decision) to avoid ZMQ latency on critical path.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from brain.intent_engine import IntentEngine
from brain.context_router import ContextRouter
from brain.behavior_model import BehaviorModel
from brain.proactive_engine import ProactiveEngine

logger = logging.getLogger("atom.brain.local_cognitive_pipeline")


class LocalCognitivePipeline:
    def __init__(self, memory_graph: Any = None):
        self.intent_engine = IntentEngine()
        self.context_router = ContextRouter()
        self.behavior_model = BehaviorModel()
        self.proactive = ProactiveEngine(
            behavior_model=self.behavior_model,
            suggestion_callback=None,
        )
        self._memory = memory_graph

    def classify_intent(self, text: str):
        return self.intent_engine.classify(text)

    def intent_to_dict(self, intent) -> Dict[str, Any]:
        return {
            "intent_type": intent.type,
            "confidence": intent.confidence,
            "entities": intent.entities,
            "urgency": getattr(intent, "urgency", "medium"),
        }

    def build_context(
        self,
        intent,
        system_state: Dict[str, Any],
        memory: Any = None,
    ) -> Dict[str, Any]:
        mem = memory if memory is not None else self._memory
        return self.context_router.build_context(intent, system_state, mem)

    def evaluate_action(self, intent, context: Dict[str, Any]) -> Dict[str, Any]:
        action = intent.type
        confidence = float(intent.confidence)
        risk = self.proactive.risk_score(action)
        utility = self.proactive._calculate_utility(action)
        context_alignment = self.proactive._calculate_context_alignment(action, context)
        uncertainty = 1.0 - confidence
        score = (confidence * utility * context_alignment) - (risk * uncertainty)
        approved = score > 0.5
        return {"action": action, "score": score, "approved": approved}
