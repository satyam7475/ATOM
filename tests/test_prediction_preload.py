"""Focused tests for prediction-driven resource preloading."""

from __future__ import annotations

import asyncio
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cognitive.prediction_engine import PredictionEngine, PredictionResult


class FakeBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def on(self, event: str, handler) -> None:
        pass

    def emit_fast(self, event: str, **data) -> None:
        self.events.append((event, data))


class FakeBehavior:
    def __init__(self) -> None:
        self._entries: list[dict] = []


class FakeMemory:
    def __init__(self) -> None:
        self._interactions: list[dict] = []


class FakePromptBuilder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def precompile(self, query: str = "", *, prompt_hint: str = "") -> dict[str, object]:
        self.calls.append((query, prompt_hint))
        return {
            "system_prompt_hash": 123,
            "tools_cached": True,
            "query_hint": "",
            "routing_hint": prompt_hint,
            "query_chars": len(query),
        }


class FakePrefetchEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], float | None]] = []

    def schedule_fire_and_forget(
        self,
        queries: list[str],
        *,
        gpu_util_pct: float = 0.0,
        prediction_accuracy: float | None = None,
    ) -> None:
        self.calls.append((list(queries), prediction_accuracy))


class TestPredictionEngine(PredictionEngine):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.warmed_apps: list[str] = []

    async def _warm_app_target(self, target: str) -> str:
        self.warmed_apps.append(target)
        return f"/Applications/{target}.app"


def _config() -> dict:
    return {
        "cognitive": {
            "predictions_enabled": True,
            "prediction_min_confidence": 0.6,
            "max_predictions": 5,
            "prediction_preload_enabled": True,
            "prediction_preload_min_confidence": 0.8,
            "prediction_preload_max_items": 3,
            "prediction_preload_cooldown_s": 120,
            "prediction_preload_timeout_s": 2.0,
        },
    }


def test_prediction_preload_warms_multiple_resources() -> None:
    bus = FakeBus()
    prompt_builder = FakePromptBuilder()
    prefetch = FakePrefetchEngine()
    engine = TestPredictionEngine(bus, FakeBehavior(), FakeMemory(), object(), _config())
    engine.attach_prompt_builder(prompt_builder)
    engine.attach_prefetch_engine(prefetch)
    engine.attach_cognitive_kernel(
        SimpleNamespace(
            route=lambda query, allow_cache=False: (
                SimpleNamespace(
                    requested_tier="command",
                    budget_tier="command",
                    use_rag=False,
                    reduce_context=False,
                    skip_llm=True,
                    prompt_hint="",
                )
                if query.startswith("open ")
                else SimpleNamespace(
                    requested_tier="complex" if "docker" in query else "simple",
                    budget_tier="complex" if "docker" in query else "simple",
                    use_rag=("docker" in query),
                    reduce_context=False,
                    skip_llm=False,
                    prompt_hint=("Use cached project context." if "docker" in query else "Be brief."),
                )
            ),
        ),
    )

    async def _run() -> None:
        reports = await engine.preload_predicted([
            PredictionResult("llm_query", "summarize today's blockers", 0.95, "pattern", 0.9),
            PredictionResult("search", "docker architecture", 0.93, "pattern", 0.9),
            PredictionResult("open_app", "Chrome", 0.91, "pattern", 0.9),
        ])

        assert len(reports) == 3
        assert engine.warmed_apps == ["Chrome"]
        assert prefetch.calls == [(["docker architecture"], 0.93)]
        assert ("summarize today's blockers", "Be brief.") in prompt_builder.calls
        assert ("docker architecture", "Use cached project context.") in prompt_builder.calls
        assert any(event == "prediction_preload" for event, _ in bus.events)

    asyncio.run(_run())
    print("  PASS: prediction preload warms app, prompt, and RAG resources")


def test_prediction_preload_respects_cooldown_and_degradation() -> None:
    bus = FakeBus()
    prompt_builder = FakePromptBuilder()
    prefetch = FakePrefetchEngine()
    engine = TestPredictionEngine(bus, FakeBehavior(), FakeMemory(), object(), _config())
    engine.attach_prompt_builder(prompt_builder)
    engine.attach_prefetch_engine(prefetch)
    engine.attach_cognitive_kernel(
        SimpleNamespace(
            route=lambda query, allow_cache=False: SimpleNamespace(
                requested_tier="creative",
                budget_tier="simple",
                use_rag=True,
                reduce_context=True,
                skip_llm=False,
                prompt_hint="Stay concise.",
            ),
        ),
    )

    async def _run() -> None:
        pred = PredictionResult(
            "llm_query",
            "brainstorm launch ideas for ATOM",
            0.92,
            "pattern",
            0.9,
        )
        first = await engine.preload_predicted([pred])
        second = await engine.preload_predicted([pred])

        assert len(first) == 1
        assert second == []
        assert prompt_builder.calls == [("brainstorm launch ideas for ATOM", "Stay concise.")]
        assert prefetch.calls == []

    asyncio.run(_run())
    print("  PASS: prediction preload respects cooldown and skips heavy work in degraded mode")


def test_llm_query_predictions_learn_live_queries() -> None:
    engine = TestPredictionEngine(FakeBus(), FakeBehavior(), FakeMemory(), object(), _config())

    async def _run() -> None:
        await engine._on_cursor_query("summarize repo architecture")
        await engine._on_cursor_query("summarize repo architecture")
        await engine._on_cursor_query("summarize repo architecture")

    asyncio.run(_run())
    preds = engine.predict_next(max_results=5)
    matches = [p for p in preds if p.action == "llm_query"]
    assert matches
    assert matches[0].target == "summarize repo architecture"
    print("  PASS: prediction engine learns live llm_query patterns")


if __name__ == "__main__":
    test_prediction_preload_warms_multiple_resources()
    test_prediction_preload_respects_cooldown_and_degradation()
    test_llm_query_predictions_learn_live_queries()
    print("\ntest_prediction_preload: ALL PASSED")
