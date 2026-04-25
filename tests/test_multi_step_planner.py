"""Regression tests for Sprint N2 -- :class:`MultiStepPlanner`."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from core.reasoning.multi_step_planner import (
    MultiStepPlanner,
    PlannedStep,
    PlannerConfig,
)


# ── stubs ──────────────────────────────────────────────────────────────


@dataclass
class _StubTool:
    name: str
    parameters: list[Any]
    description: str = ""


class _StubRegistry:
    def __init__(self, names: list[str]) -> None:
        self._tools = {n: _StubTool(n, []) for n in names}

    def get(self, name: str) -> _StubTool | None:
        return self._tools.get(name)

    def get_all(self) -> list[_StubTool]:
        return list(self._tools.values())


@dataclass
class _StubResult:
    success: bool
    output: str = ""
    error: str = ""
    elapsed_ms: float = 1.0
    blocked: bool = False
    block_reason: str = ""


class _StubExecutor:
    def __init__(
        self,
        outcomes: dict[str, _StubResult] | None = None,
        *,
        raise_on: set[str] | None = None,
        sleep_s: float = 0.0,
    ) -> None:
        self.outcomes = outcomes or {}
        self.raise_on = raise_on or set()
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.sleep_s = sleep_s

    async def execute_async(self, tool_call: Any) -> _StubResult:
        self.calls.append(
            (tool_call.name, dict(tool_call.arguments)),
        )
        if self.sleep_s:
            await asyncio.sleep(self.sleep_s)
        if tool_call.name in self.raise_on:
            raise RuntimeError(f"boom-{tool_call.name}")
        return self.outcomes.get(
            tool_call.name,
            _StubResult(success=True, output=f"ran {tool_call.name}"),
        )


# ── parse_plan ─────────────────────────────────────────────────────────


def test_parse_plan_accepts_object_with_plan_key() -> None:
    p = MultiStepPlanner(
        _StubRegistry(["set_volume"]),
        _StubExecutor(),
    )
    blob = {
        "rationale": "Boss wanted music + volume",
        "plan": [
            {"name": "set_volume", "arguments": {"percent": 30}},
        ],
        "stop_on_error": False,
    }
    steps, rationale, stop, err = p.parse_plan(blob)
    assert err == ""
    assert len(steps) == 1
    assert steps[0].tool_name == "set_volume"
    assert steps[0].arguments == {"percent": 30}
    assert rationale.startswith("Boss wanted")
    assert stop is False


def test_parse_plan_accepts_naked_list() -> None:
    p = MultiStepPlanner(
        _StubRegistry(["mute", "unmute"]),
        _StubExecutor(),
    )
    steps, _, _, err = p.parse_plan(
        [{"name": "mute"}, {"name": "unmute"}],
    )
    assert err == ""
    assert [s.tool_name for s in steps] == ["mute", "unmute"]


def test_parse_plan_strips_markdown_fence() -> None:
    p = MultiStepPlanner(_StubRegistry(["mute"]), _StubExecutor())
    blob = '```json\n{"plan": [{"name": "mute"}]}\n```'
    steps, _, _, err = p.parse_plan(blob)
    assert err == ""
    assert steps[0].tool_name == "mute"


def test_parse_plan_rejects_unknown_tool() -> None:
    p = MultiStepPlanner(_StubRegistry(["mute"]), _StubExecutor())
    steps, _, _, err = p.parse_plan(
        [{"name": "format_disk", "arguments": {}}],
    )
    assert steps == []
    assert "format_disk" in err


def test_parse_plan_rejects_invalid_shape() -> None:
    p = MultiStepPlanner(_StubRegistry(["mute"]), _StubExecutor())
    _, _, _, err = p.parse_plan({"plan": "not a list"})
    assert err


def test_parse_plan_caps_at_max_steps() -> None:
    cfg = PlannerConfig(max_steps=2)
    p = MultiStepPlanner(_StubRegistry(["a", "b", "c", "d"]),
                         _StubExecutor(), config=cfg)
    steps, _, _, err = p.parse_plan(
        [{"name": "a"}, {"name": "b"}, {"name": "c"}, {"name": "d"}],
    )
    assert err == ""
    assert len(steps) == 2


# ── execute ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_runs_all_steps_in_order() -> None:
    reg = _StubRegistry(["mute", "set_volume", "unmute"])
    ex = _StubExecutor()
    p = MultiStepPlanner(reg, ex)

    result = await p.execute(
        [
            {"name": "mute"},
            {"name": "set_volume", "arguments": {"percent": 30}},
            {"name": "unmute"},
        ],
    )
    assert result.all_succeeded
    assert [c[0] for c in ex.calls] == ["mute", "set_volume", "unmute"]
    assert ex.calls[1][1] == {"percent": 30}


@pytest.mark.asyncio
async def test_execute_stops_on_error_by_default() -> None:
    reg = _StubRegistry(["mute", "set_volume", "unmute"])
    ex = _StubExecutor(
        outcomes={
            "mute": _StubResult(success=True),
            "set_volume": _StubResult(success=False, error="bad value"),
        },
    )
    p = MultiStepPlanner(reg, ex)
    result = await p.execute(
        [{"name": "mute"}, {"name": "set_volume"}, {"name": "unmute"}],
    )
    assert result.any_failed
    assert result.n_completed == 1
    assert any(s.skipped for s in result.steps)
    assert [c[0] for c in ex.calls] == ["mute", "set_volume"]


@pytest.mark.asyncio
async def test_execute_continues_when_stop_on_error_false() -> None:
    reg = _StubRegistry(["mute", "set_volume", "unmute"])
    ex = _StubExecutor(
        outcomes={"set_volume": _StubResult(success=False, error="x")},
    )
    p = MultiStepPlanner(reg, ex)
    result = await p.execute(
        {
            "plan": [
                {"name": "mute"},
                {"name": "set_volume"},
                {"name": "unmute"},
            ],
            "stop_on_error": False,
        },
    )
    assert [c[0] for c in ex.calls] == ["mute", "set_volume", "unmute"]
    assert result.any_failed
    assert result.n_completed == 2


@pytest.mark.asyncio
async def test_execute_handles_executor_exception() -> None:
    reg = _StubRegistry(["mute"])
    ex = _StubExecutor(raise_on={"mute"})
    p = MultiStepPlanner(reg, ex)
    result = await p.execute([{"name": "mute"}])
    assert result.steps[0].success is False
    assert "boom" in result.steps[0].error


@pytest.mark.asyncio
async def test_execute_per_step_timeout() -> None:
    reg = _StubRegistry(["mute"])
    ex = _StubExecutor(sleep_s=0.05)
    cfg = PlannerConfig(per_step_timeout_s=0.01)
    p = MultiStepPlanner(reg, ex, config=cfg)
    result = await p.execute([{"name": "mute"}])
    assert result.steps[0].success is False
    assert "timed out" in result.steps[0].error


@pytest.mark.asyncio
async def test_speak_summary_reports_partial_success() -> None:
    reg = _StubRegistry(["mute", "set_volume"])
    ex = _StubExecutor(
        outcomes={"set_volume": _StubResult(success=False, error="nope")},
    )
    p = MultiStepPlanner(reg, ex)
    result = await p.execute(
        [{"name": "mute"}, {"name": "set_volume"}],
    )
    summary = result.speak_summary()
    assert "set_volume" in summary
    assert "1 of 2" in summary or "Got 1" in summary
