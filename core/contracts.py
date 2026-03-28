from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ServiceContract:
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    latency_budget_ms: int

# Strict Contracts for Distributed Cognitive Graph
CONTRACTS = {
    "intent_engine": ServiceContract(
        input_schema={"text": str, "context": Optional[dict]},
        output_schema={"intent_type": str, "confidence": float, "entities": dict},
        latency_budget_ms=50
    ),
    "context_engine": ServiceContract(
        input_schema={"intent": dict, "system_state": dict},
        output_schema={"enriched_context": dict},
        latency_budget_ms=100
    ),
    "decision_engine": ServiceContract(
        input_schema={"intent": dict, "context": dict},
        output_schema={"action": str, "score": float, "approved": bool},
        latency_budget_ms=150
    ),
    "memory_engine": ServiceContract(
        input_schema={"query": str, "filters": dict},
        output_schema={"nodes": list, "scores": list},
        latency_budget_ms=200
    )
}
