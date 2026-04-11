"""Focused tests for the cognitive budget system."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cognitive_kernel import CognitiveBudgetTier, CognitiveKernel, ExecPath


class FakeBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def on(self, event: str, handler) -> None:
        pass

    def emit_fast(self, event: str, **data) -> None:
        self.events.append((event, data))


class FakeMetrics:
    def __init__(self) -> None:
        self.samples: list[tuple[str, float]] = []

    def record_latency(self, name: str, value: float) -> None:
        self.samples.append((name, value))


class FakeIntentResult:
    def __init__(
        self,
        intent: str = "fallback",
        response: str = "",
        action: str | None = None,
        action_args: dict | None = None,
    ) -> None:
        self.intent = intent
        self.response = response
        self.action = action
        self.action_args = action_args or {}


class FakeIntentEngine:
    def classify(self, text: str) -> FakeIntentResult:
        low = (text or "").strip().lower()
        if low == "open chrome":
            return FakeIntentResult(
                "open_app",
                response="Opening Chrome, Boss.",
                action="open_app",
                action_args={"app_name": "Chrome"},
            )
        if low == "what time is it":
            return FakeIntentResult("time", response="It's testing time, Boss.")
        return FakeIntentResult("fallback")


class FakeCache:
    def __init__(self, mapping: dict[str, str] | None = None) -> None:
        self._mapping = {
            str(k).strip().lower(): v
            for k, v in (mapping or {}).items()
        }

    def get(self, text: str) -> str | None:
        return self._mapping.get(str(text or "").strip().lower())


class FakeSiliconStats:
    def __init__(
        self,
        *,
        memory_pct: float = 32.0,
        cpu_pct: float = 14.0,
        battery_pct: int = 100,
        on_battery: bool = False,
        thermal_pressure: str = "nominal",
        is_throttled: bool = False,
    ) -> None:
        self.memory_pct = memory_pct
        self.cpu_pct = cpu_pct
        self.battery_pct = battery_pct
        self.on_battery = on_battery
        self.thermal_pressure = thermal_pressure
        self.is_throttled = is_throttled


class FakeSiliconGovernor:
    def __init__(self, **kwargs) -> None:
        self._stats = FakeSiliconStats(**kwargs)

    def get_stats(self) -> FakeSiliconStats:
        return self._stats


def _config() -> dict:
    return {
        "cognitive_kernel": {
            "quick_model": "qwen3-1.7b",
            "full_model": "qwen3-4b",
            "simple_query_max_chars": 48,
            "deep_query_min_chars": 120,
            "battery_degrade": True,
            "thermal_degrade": True,
            "memory_pressure_threshold": 85,
        },
        "latency_controller": {
            "direct_budget_ms": 50,
            "cache_budget_ms": 100,
            "quick_budget_ms": 1500,
            "full_budget_ms": 5000,
            "deep_budget_ms": 15000,
            "simple_query_max_chars": 48,
            "memory_pressure_pct": 85,
            "low_battery_pct": 20,
            "cpu_busy_pct": 88,
            "battery_scale": 1.0,
            "low_battery_scale": 1.0,
            "thermal_scale": 1.0,
            "memory_scale": 1.0,
            "cpu_scale": 1.0,
            "simple_scale": 1.0,
            "deep_scale": 1.0,
            "rag_fraction_full": 0.16,
            "rag_fraction_deep": 0.24,
            "rag_min_ms": 60,
            "rag_max_ms": 650,
        },
    }


def test_command_and_info_budgets() -> None:
    bus = FakeBus()
    kernel = CognitiveKernel(
        config=_config(),
        bus=bus,
        intent_engine=FakeIntentEngine(),
        cache_engine=FakeCache(),
        metrics=FakeMetrics(),
        silicon_governor=FakeSiliconGovernor(),
    )

    command_plan = kernel.route("open chrome")
    assert command_plan.path is ExecPath.DIRECT
    assert command_plan.requested_tier == CognitiveBudgetTier.COMMAND.value
    assert command_plan.budget_tier == CognitiveBudgetTier.COMMAND.value
    assert command_plan.base_budget_ms == 100.0
    assert command_plan.budget_ms == 100.0
    assert command_plan.runtime_mode == "FAST"
    assert command_plan.direct_action == "open_app"
    assert command_plan.latency_reason == "fast_path:command"

    info_plan = kernel.route("what time is it")
    assert info_plan.path is ExecPath.DIRECT
    assert info_plan.requested_tier == CognitiveBudgetTier.INFO.value
    assert info_plan.budget_tier == CognitiveBudgetTier.INFO.value
    assert info_plan.base_budget_ms == 500.0
    assert info_plan.budget_ms == 500.0
    assert info_plan.runtime_mode == "FAST"
    assert info_plan.budget_allow_memory is True
    assert info_plan.latency_reason == "fast_path:info"

    diag = kernel.get_diagnostics()
    assert diag["budget_distribution"]["command"]["count"] == 1
    assert diag["budget_distribution"]["info"]["count"] == 1
    print("  PASS: command/info budgets route correctly")


def test_simple_complex_and_creative_routing() -> None:
    kernel = CognitiveKernel(
        config=_config(),
        bus=FakeBus(),
        intent_engine=FakeIntentEngine(),
        cache_engine=FakeCache(),
        metrics=FakeMetrics(),
        silicon_governor=FakeSiliconGovernor(),
    )

    simple_plan = kernel.route("tell me a joke")
    assert simple_plan.path is ExecPath.QUICK
    assert simple_plan.requested_tier == CognitiveBudgetTier.SIMPLE.value
    assert simple_plan.budget_tier == CognitiveBudgetTier.SIMPLE.value
    assert simple_plan.model_role == "fast"
    assert simple_plan.base_budget_ms == 1500.0
    assert simple_plan.budget_ms == 1500.0
    assert simple_plan.use_memory is True
    assert simple_plan.use_rag is False

    complex_plan = kernel.route(
        "Explain the scheduler architecture and compare the tradeoffs in detail.",
    )
    assert complex_plan.path is ExecPath.FULL
    assert complex_plan.requested_tier == CognitiveBudgetTier.COMPLEX.value
    assert complex_plan.budget_tier == CognitiveBudgetTier.COMPLEX.value
    assert complex_plan.model_role == "primary"
    assert complex_plan.base_budget_ms == 5000.0
    assert complex_plan.budget_ms == 5000.0
    assert complex_plan.use_rag is True
    assert complex_plan.use_memory is True

    creative_plan = kernel.route(
        "Brainstorm three offline-first product ideas for ATOM, compare the tradeoffs, "
        "and outline a rollout plan with staged milestones.",
    )
    assert creative_plan.path is ExecPath.DEEP
    assert creative_plan.requested_tier == CognitiveBudgetTier.CREATIVE.value
    assert creative_plan.budget_tier == CognitiveBudgetTier.CREATIVE.value
    assert creative_plan.runtime_mode == "DEEP"
    assert creative_plan.thinking is True
    assert creative_plan.base_budget_ms == 10000.0
    assert creative_plan.budget_ms == 10000.0
    print("  PASS: simple/complex/creative budgets map to the right paths")


def test_creative_queries_degrade_safely_on_low_battery() -> None:
    kernel = CognitiveKernel(
        config=_config(),
        bus=FakeBus(),
        intent_engine=FakeIntentEngine(),
        cache_engine=FakeCache(),
        metrics=FakeMetrics(),
        silicon_governor=FakeSiliconGovernor(
            battery_pct=9,
            on_battery=True,
        ),
    )

    plan = kernel.route(
        "Brainstorm three offline-first product ideas for ATOM, compare the tradeoffs, "
        "and outline a rollout plan with staged milestones.",
    )

    assert plan.requested_tier == CognitiveBudgetTier.CREATIVE.value
    assert plan.budget_tier == CognitiveBudgetTier.SIMPLE.value
    assert plan.path is ExecPath.QUICK
    assert plan.runtime_mode == "FAST"
    assert plan.base_budget_ms == 1500.0
    assert plan.budget_ms == 1500.0
    assert plan.use_rag is False
    assert plan.reduce_context is True
    assert "battery" in plan.latency_reason
    print("  PASS: low battery degrades creative work to a laptop-safe budget")


if __name__ == "__main__":
    test_command_and_info_budgets()
    test_simple_complex_and_creative_routing()
    test_creative_queries_degrade_safely_on_low_battery()
    print("\ntest_cognitive_kernel: ALL PASSED")
