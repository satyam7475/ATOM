"""Regression suite for Sprint Ω modules.

Covers:
* :class:`core.reasoning.parallel_plan_executor.ParallelPlanExecutor`
  -- parse, cycle detection, parallel scheduling, ``planner_prompt_block``.
* :class:`core.reasoning.agent_supervisor.AgentSupervisor` triage and
  end-to-end ``run`` with a stubbed LLM and stubbed dispatcher.
* ``Router.attach_supervisor`` / ``execute_dag_plan`` / ``run_supervised``
  hooks (smoke-only -- the full router has heavy dependencies, so
  these are tested via the supervisor's surface).
* ``ActionExecutor.registry`` public property (stable contract for
  Sprint Ω consumers).
* :mod:`voice.smart_turn_taker` decision logic with a mock VAD.

Owner: Satyam
"""
from __future__ import annotations

import asyncio
import json
import struct

import pytest

from core.reasoning.action_executor import ActionExecutor
from core.reasoning.agent_supervisor import (
    AgentSupervisor,
    SupervisorConfig,
)
from core.reasoning.parallel_plan_executor import (
    ParallelPlanExecutor,
    ParallelPlannerConfig,
)
from core.reasoning.tool_registry import (
    Tool,
    ToolParameter,
    ToolRegistry,
)


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tool(
        name="weather_get",
        description="Look up the current weather in a city.",
        parameters=[ToolParameter(name="city", type="string", required=True)],
        safety_level="safe",
    ))
    reg.register(Tool(
        name="calendar_today",
        description="List today's calendar events.",
        parameters=[],
        safety_level="safe",
    ))
    reg.register(Tool(
        name="compose_text",
        description="Compose a short text given a prompt.",
        parameters=[
            ToolParameter(name="prompt", type="string", required=True),
        ],
        safety_level="safe",
    ))
    return reg


class _StubSecurity:
    """Permissive security stub matching the surface ActionExecutor
    actually calls (``allow_action`` + ``audit_log``)."""

    def allow_action(self, name, params):
        return True, ""

    def audit_log(self, *args, **kwargs):
        return None


def _make_executor(registry):
    calls = []

    def dispatch(action, params):
        calls.append((action, dict(params)))
        return {"status": "ok", "action": action, "speak": f"did {action}"}

    async def adispatch(action, params):
        return dispatch(action, params)

    ex = ActionExecutor(
        dispatch_fn=dispatch,
        security=_StubSecurity(),
        registry=registry,
        async_dispatch_fn=adispatch,
    )
    return ex, calls


# ── ActionExecutor public surface ─────────────────────────────────────


def test_action_executor_registry_property_round_trips():
    reg = _make_registry()
    ex, _ = _make_executor(reg)
    assert ex.registry is reg
    # Registry seeds itself with builtin tools; we only verify our
    # three newly-registered ones are reachable through the public
    # ``registry`` property exposed by ActionExecutor.
    assert ex.registry.get("weather_get") is not None
    assert ex.registry.get("calendar_today") is not None
    assert ex.registry.get("compose_text") is not None


# ── ParallelPlanExecutor ──────────────────────────────────────────────


def test_parallel_planner_parses_flat_plan():
    reg = _make_registry()
    ex, _ = _make_executor(reg)
    pp = ParallelPlanExecutor(reg, ex, config=ParallelPlannerConfig())
    blob = json.dumps({
        "plan": [
            {"name": "weather_get", "arguments": {"city": "Delhi"}},
            {"name": "calendar_today"},
        ],
    })
    steps, rationale, stop, err = pp.parse_plan(blob)
    assert not err
    assert len(steps) == 2
    # Auto-derived step ids ("s1", "s2", ...) keep the prompt cheap;
    # the executor only requires uniqueness, not a specific scheme.
    assert all(s.id for s in steps)
    assert len({s.id for s in steps}) == 2
    assert steps[1].depends_on == ()


def test_parallel_planner_rejects_cycles():
    reg = _make_registry()
    ex, _ = _make_executor(reg)
    pp = ParallelPlanExecutor(reg, ex, config=ParallelPlannerConfig())
    blob = json.dumps({
        "plan": [
            {"id": "a", "name": "weather_get",
             "arguments": {"city": "X"}, "depends_on": ["b"]},
            {"id": "b", "name": "calendar_today",
             "depends_on": ["a"]},
        ],
    })
    steps, _r, _s, err = pp.parse_plan(blob)
    assert steps == []
    assert err is not None
    assert "cycle" in err.lower() or "depend" in err.lower()


def test_parallel_planner_rejects_unknown_tool():
    reg = _make_registry()
    ex, _ = _make_executor(reg)
    pp = ParallelPlanExecutor(
        reg, ex,
        config=ParallelPlannerConfig(allow_unknown_tools=False),
    )
    blob = json.dumps({"plan": [{"name": "nuke_everything"}]})
    _steps, _r, _s, err = pp.parse_plan(blob)
    assert err is not None
    assert "nuke_everything" in err


def test_parallel_planner_runs_independent_steps_concurrently():
    reg = _make_registry()
    ex, calls = _make_executor(reg)
    pp = ParallelPlanExecutor(
        reg, ex,
        config=ParallelPlannerConfig(max_concurrency=3),
    )
    blob = json.dumps({
        "plan": [
            {"id": "wx", "name": "weather_get",
             "arguments": {"city": "Delhi"}},
            {"id": "cal", "name": "calendar_today"},
            {"id": "sum", "name": "compose_text",
             "arguments": {"prompt": "Brief"},
             "depends_on": ["wx", "cal"]},
        ],
    })

    async def run():
        return await pp.execute(blob)

    result = asyncio.run(run())
    assert result.all_succeeded, result.speak_summary()
    # All three dispatches happened.
    actions = sorted(c[0] for c in calls)
    assert actions == ["calendar_today", "compose_text", "weather_get"]


def test_planner_prompt_block_lists_registered_tools():
    reg = _make_registry()
    ex, _ = _make_executor(reg)
    pp = ParallelPlanExecutor(reg, ex, config=ParallelPlannerConfig())
    block = pp.planner_prompt_block()
    assert "weather_get" in block
    assert "calendar_today" in block
    assert "compose_text" in block


# ── AgentSupervisor ───────────────────────────────────────────────────


def test_supervisor_triage_skips_short_intents():
    reg = _make_registry()
    ex, _ = _make_executor(reg)
    sup = AgentSupervisor(tool_registry=reg, action_executor=ex)
    assert sup.triage("hi").needs_plan is False
    assert sup.triage("play focus").needs_plan is False


def test_supervisor_triage_promotes_multi_step_hints():
    reg = _make_registry()
    ex, _ = _make_executor(reg)
    sup = AgentSupervisor(tool_registry=reg, action_executor=ex)
    t = sup.triage(
        "compare the weather and my calendar then draft a status update"
    )
    assert t.needs_plan is True
    assert t.reason in {"multi_step_hint", "multi_action_verbs", "long_action_query"}


def test_supervisor_run_executes_canned_plan_end_to_end():
    reg = _make_registry()
    ex, calls = _make_executor(reg)
    canned = json.dumps({
        "plan": [
            {"id": "wx", "name": "weather_get",
             "arguments": {"city": "Delhi"}},
            {"id": "cal", "name": "calendar_today"},
            {"id": "sum", "name": "compose_text",
             "arguments": {"prompt": "Brief"},
             "depends_on": ["wx", "cal"]},
        ],
    })

    async def stub_llm(prompt, system):
        return canned

    sup = AgentSupervisor(
        tool_registry=reg,
        action_executor=ex,
        llm_call=stub_llm,
    )
    result = asyncio.run(sup.run(
        "compare the weather and my calendar then draft a status email",
    ))
    assert result.used_plan is True
    assert result.error == ""
    assert result.plan_result is not None
    assert result.plan_result.all_succeeded
    actions = sorted(c[0] for c in calls)
    assert actions == ["calendar_today", "compose_text", "weather_get"]


def test_supervisor_run_returns_no_plan_on_short_query():
    reg = _make_registry()
    ex, _ = _make_executor(reg)

    async def stub_llm(prompt, system):  # pragma: no cover
        raise AssertionError("LLM should not be called for short queries")

    sup = AgentSupervisor(
        tool_registry=reg,
        action_executor=ex,
        llm_call=stub_llm,
    )
    result = asyncio.run(sup.run("hi"))
    assert result.used_plan is False
    assert result.error == ""


def test_supervisor_run_requires_llm_call_binding():
    reg = _make_registry()
    ex, _ = _make_executor(reg)
    sup = AgentSupervisor(tool_registry=reg, action_executor=ex)
    result = asyncio.run(sup.run(
        "compare the weather and my calendar then draft a status email",
    ))
    assert result.used_plan is False
    assert result.error == "no_llm_call_bound"


# ── SmartTurnTaker ────────────────────────────────────────────────────


def _silent_pcm16(seconds: float, sample_rate: int = 16000) -> bytes:
    return struct.pack(
        "<" + "h" * int(seconds * sample_rate),
        *([0] * int(seconds * sample_rate)),
    )


def test_smart_turn_taker_finalizes_on_silence():
    pytest.importorskip("silero_vad")
    from voice.smart_turn_taker import SmartTurnTaker, SmartTurnTakerConfig

    t = SmartTurnTaker(SmartTurnTakerConfig(enabled=True))
    assert t.preload() is True
    assert t.is_available

    decision = t.should_finalize(
        _silent_pcm16(1.0),
        silence_s=0.25,
        utterance_s=1.5,
    )
    assert decision.finalize is True
    assert decision.eot_score > 0.5


def test_smart_turn_taker_disabled_never_finalises():
    """When disabled in config the turn-taker hands every decision back
    to the legacy trailing-silence rule. We assert ``finalize`` stays
    False -- the actual reason string is an implementation detail
    (``below_min_silence`` / ``disabled`` / ``eval_skipped``) and is
    explicitly not part of the contract.
    """
    from voice.smart_turn_taker import SmartTurnTaker, SmartTurnTakerConfig
    t = SmartTurnTaker(SmartTurnTakerConfig(enabled=False))
    decision = t.should_finalize(b"", silence_s=0.1, utterance_s=0.5)
    assert decision.finalize is False
