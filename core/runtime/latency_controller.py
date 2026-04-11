"""
ATOM -- Dynamic latency controller.

Translates query path + live system state into concrete latency budgets and
context limits. The Cognitive Kernel remains the authority on *which* path to
take; this controller decides how much latency budget that path should get
right now.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.fast_path import LatencyBudget
from core.rag.query_classifier import QueryComplexity, classify_query


@dataclass(frozen=True)
class LatencyDecision:
    """Concrete latency and context limits for one query plan."""

    budget_ms: float
    rag_budget_ms: float = 0.0
    skip_llm: bool = False
    skip_rag: bool = False
    reduce_context: bool = False
    memory_limit: int = 0
    history_turn_limit: int = 4
    reason: str = ""

    def to_budget(self, label: str = "") -> LatencyBudget:
        return LatencyBudget(budget_ms=self.budget_ms, label=label)


class LatencyController:
    """Dynamic latency management based on query path and system state."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        cfg = (self._config.get("latency_controller") or {})
        ck = (self._config.get("cognitive_kernel") or {})

        self._base_budgets = {
            "direct": float(cfg.get("direct_budget_ms", 50.0)),
            "cache": float(cfg.get("cache_budget_ms", 100.0)),
            "quick": float(cfg.get("quick_budget_ms", 1500.0)),
            "full": float(cfg.get("full_budget_ms", 5000.0)),
            "deep": float(cfg.get("deep_budget_ms", 15000.0)),
        }
        self._simple_query_max_chars = int(
            cfg.get("simple_query_max_chars", ck.get("simple_query_max_chars", 50)),
        )
        self._memory_pressure_pct = float(
            cfg.get("memory_pressure_pct", ck.get("memory_pressure_threshold", 85.0)),
        )
        self._low_battery_pct = int(cfg.get("low_battery_pct", 20))
        self._cpu_busy_pct = float(cfg.get("cpu_busy_pct", 88.0))

        self._battery_scale = float(cfg.get("battery_scale", 0.9))
        self._low_battery_scale = float(cfg.get("low_battery_scale", 0.72))
        self._thermal_scale = float(cfg.get("thermal_scale", 0.68))
        self._memory_scale = float(cfg.get("memory_scale", 0.75))
        self._cpu_scale = float(cfg.get("cpu_scale", 0.82))
        self._simple_scale = float(cfg.get("simple_scale", 0.84))
        self._deep_scale = float(cfg.get("deep_scale", 1.08))

        self._rag_fraction_full = float(cfg.get("rag_fraction_full", 0.16))
        self._rag_fraction_deep = float(cfg.get("rag_fraction_deep", 0.24))
        self._rag_min_ms = float(cfg.get("rag_min_ms", 60.0))
        self._rag_max_ms = float(cfg.get("rag_max_ms", 650.0))

    def get_budget(
        self,
        query: str,
        *,
        path: str,
        system_state: dict[str, Any] | None = None,
        complexity: QueryComplexity | None = None,
        base_budget_ms: float | None = None,
        budget_tier: str = "",
        skip_llm: bool = False,
        use_rag: bool = False,
        use_memory: bool = False,
        thinking: bool = False,
    ) -> LatencyDecision:
        """Return dynamic latency and context limits for a query path."""
        path_key = str(path or "full").strip().lower()
        base_budget = float(
            base_budget_ms
            if base_budget_ms is not None and float(base_budget_ms) > 0
            else self._base_budgets.get(path_key, self._base_budgets["full"]),
        )
        cx = complexity or classify_query(query or "")
        qlen = len((query or "").strip())
        state = dict(system_state or {})
        tier_key = str(budget_tier or "").strip().lower()

        if skip_llm or path_key in {"direct", "cache"}:
            return LatencyDecision(
                budget_ms=base_budget,
                rag_budget_ms=0.0,
                skip_llm=skip_llm,
                skip_rag=True,
                reduce_context=False,
                memory_limit=0 if not use_memory else 1,
                history_turn_limit=2 if path_key == "direct" else 4,
                reason=f"fast_path:{tier_key or path_key}",
            )

        scale = 1.0
        reduce_context = False
        skip_rag = not use_rag
        reasons: list[str] = [tier_key or path_key]
        if tier_key and tier_key != path_key:
            reasons.append(path_key)

        cpu_pct = float(state.get("cpu_pct", state.get("cpu_percent", 0.0)) or 0.0)
        memory_pct = float(state.get("memory_pct", 0.0) or 0.0)
        battery_pct = int(state.get("battery_pct", 100) or 100)
        on_battery = bool(state.get("on_battery", False))
        thermal_pressure = str(state.get("thermal_pressure", "nominal") or "nominal").lower()
        is_throttled = bool(state.get("is_throttled", False))

        if on_battery:
            scale *= self._battery_scale
            reasons.append("battery")
            if battery_pct <= self._low_battery_pct:
                scale *= self._low_battery_scale
                reduce_context = True
                skip_rag = True
                reasons.append(f"battery_{battery_pct}")

        if is_throttled or thermal_pressure in {"serious", "critical"}:
            scale *= self._thermal_scale
            reduce_context = True
            skip_rag = True
            reasons.append(f"thermal_{thermal_pressure or 'throttled'}")

        if memory_pct >= self._memory_pressure_pct:
            scale *= self._memory_scale
            reduce_context = True
            skip_rag = True
            reasons.append(f"memory_{int(memory_pct)}")

        if cpu_pct >= self._cpu_busy_pct:
            scale *= self._cpu_scale
            reduce_context = True
            reasons.append(f"cpu_{int(cpu_pct)}")

        if cx == QueryComplexity.SIMPLE and qlen <= self._simple_query_max_chars:
            scale *= self._simple_scale
            reasons.append("simple")
        elif cx == QueryComplexity.COMPLEX and (thinking or path_key == "deep"):
            scale *= self._deep_scale
            reasons.append("deep")

        budget_ms = max(50.0, base_budget * scale)

        memory_limit = {
            "quick": 1,
            "full": 2,
            "deep": 3,
        }.get(path_key, 0)
        history_turn_limit = {
            "quick": 4,
            "full": 6,
            "deep": 8,
        }.get(path_key, 4)

        if not use_memory:
            memory_limit = 0
        elif reduce_context:
            memory_limit = max(1, memory_limit - 1)

        if reduce_context:
            history_turn_limit = max(2, history_turn_limit - 2)

        rag_budget_ms = 0.0
        if use_rag and not skip_rag:
            rag_fraction = self._rag_fraction_deep if (thinking or path_key == "deep") else self._rag_fraction_full
            rag_budget_ms = budget_ms * rag_fraction
            rag_budget_ms = max(self._rag_min_ms, min(self._rag_max_ms, rag_budget_ms))
            if reduce_context:
                rag_budget_ms = max(self._rag_min_ms, rag_budget_ms * 0.8)

        return LatencyDecision(
            budget_ms=budget_ms,
            rag_budget_ms=rag_budget_ms,
            skip_llm=False,
            skip_rag=skip_rag,
            reduce_context=reduce_context,
            memory_limit=memory_limit,
            history_turn_limit=history_turn_limit,
            reason="+".join(reasons),
        )


__all__ = ["LatencyController", "LatencyDecision"]
