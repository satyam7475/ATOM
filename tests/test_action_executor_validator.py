"""ATOM v3.6 -- ReAct schema-validator integration tests.

Verifies that ``ActionExecutor`` rejects malformed tool calls *before*
they reach the dispatch handler, returning an ``ActionResult`` whose
``observation`` carries the structured ``[INVALID TOOL CALL]`` prefix
the brain's ReAct loop uses to self-correct.

The bridge under test is the new ``_schema_reject`` helper that runs
``core.reasoning.tool_grammar.validate_tool_call`` after alias
normalization but before any side-effect (security/confirmation/
dispatch). Today's executor only catches missing required args and
unknown tool names; the validator extends rejection to:

  * unknown args (extra keys not declared on the Tool)
  * wrong types (e.g. a string where the schema declared integer)
  * enum violations (value outside the declared enum set)

Each rejection becomes a single, model-friendly observation
("[INVALID TOOL CALL] open_app: unknown args: wormhole") that is fed
straight back into the next ReAct step. That is the loop the user
called out as "no ReAct retry over validate_tool_call".
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.reasoning.action_executor import ActionExecutor, ActionResult
from core.reasoning.tool_parser import ToolCall
from core.reasoning.tool_registry import Tool, ToolParameter, ToolRegistry
from core.security_policy import SecurityPolicy


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_registry() -> ToolRegistry:
    """Tiny, deterministic registry with one of each schema shape."""
    reg = ToolRegistry()
    reg.register(Tool(
        name="open_app",
        description="Open a desktop application by name.",
        parameters=[
            ToolParameter(name="name", type="string", required=True),
        ],
    ))
    reg.register(Tool(
        name="set_volume",
        description="Set system volume.",
        parameters=[
            ToolParameter(name="level", type="integer", required=True),
            ToolParameter(
                name="mode", type="string", required=False,
                enum=["mute", "unmute", "set"],
            ),
        ],
    ))
    return reg


def _make_executor(
    *,
    dispatch_calls: list[tuple[str, dict]] | None = None,
) -> tuple[ActionExecutor, list[tuple[str, dict]]]:
    """Build an executor whose dispatch records every call it sees.

    The recorded list is returned alongside so tests can assert that
    rejected calls never reached the dispatcher.
    """
    calls = dispatch_calls if dispatch_calls is not None else []

    def _dispatch(name: str, args: dict) -> str:
        calls.append((name, dict(args)))
        return f"dispatched:{name}"

    async def _dispatch_async(name: str, args: dict) -> str:
        calls.append((name, dict(args)))
        return f"dispatched:{name}"

    executor = ActionExecutor(
        dispatch_fn=_dispatch,
        security=SecurityPolicy({}),
        registry=_make_registry(),
        async_dispatch_fn=_dispatch_async,
    )
    return executor, calls


# ── Sync execute() ───────────────────────────────────────────────────


def test_executor_rejects_unknown_arg_before_dispatch_sync() -> None:
    executor, calls = _make_executor()
    result = executor.execute(ToolCall(
        name="open_app",
        arguments={"name": "Chrome", "wormhole": True},
    ))
    assert isinstance(result, ActionResult)
    assert result.invalid is True
    assert result.success is False
    assert "wormhole" in result.validation_error
    assert result.observation.startswith("[INVALID TOOL CALL] open_app:")
    assert "unknown args" in result.observation
    assert calls == [], "dispatcher must not run for invalid calls"


def test_executor_rejects_wrong_type_before_dispatch_sync() -> None:
    executor, calls = _make_executor()
    result = executor.execute(ToolCall(
        name="set_volume",
        arguments={"level": "loud"},
    ))
    assert result.invalid is True
    assert "level" in result.validation_error
    assert "wrong types" in result.observation
    assert calls == []


def test_executor_rejects_enum_violation_before_dispatch_sync() -> None:
    executor, calls = _make_executor()
    result = executor.execute(ToolCall(
        name="set_volume",
        arguments={"level": 50, "mode": "explode"},
    ))
    assert result.invalid is True
    assert "mode" in result.validation_error
    assert calls == []


def test_executor_passes_well_formed_call_through_to_dispatch_sync() -> None:
    """The validator must not regress the happy path."""
    executor, calls = _make_executor()
    result = executor.execute(ToolCall(
        name="open_app",
        arguments={"name": "Chrome"},
    ))
    assert result.invalid is False
    assert result.success is True
    assert calls == [("open_app", {"name": "Chrome"})]
    assert result.observation.startswith("[OK] open_app:")


def test_executor_normalizes_aliases_before_validating_sync() -> None:
    """Alias normalization (``exe`` -> ``name``) must happen first.

    Without this guarantee the validator would see ``{"exe": "Chrome"}``,
    flag ``exe`` as unknown AND ``name`` as missing required, and
    reject perfectly recoverable calls. _validate_params runs first and
    rewrites the args, so the validator sees the canonical shape.
    """
    executor, calls = _make_executor()
    result = executor.execute(ToolCall(
        name="open_app",
        arguments={"exe": "Chrome"},
    ))
    assert result.invalid is False, result.validation_error
    assert result.success is True
    assert calls == [("open_app", {"name": "Chrome"})]


# ── Async execute_async() ────────────────────────────────────────────


def test_executor_rejects_unknown_arg_before_dispatch_async() -> None:
    executor, calls = _make_executor()
    result = asyncio.run(executor.execute_async(ToolCall(
        name="open_app",
        arguments={"name": "Chrome", "wormhole": True},
    )))
    assert result.invalid is True
    assert "wormhole" in result.validation_error
    assert result.observation.startswith("[INVALID TOOL CALL] open_app:")
    assert calls == []


def test_executor_rejects_wrong_type_before_dispatch_async() -> None:
    executor, calls = _make_executor()
    result = asyncio.run(executor.execute_async(ToolCall(
        name="set_volume",
        arguments={"level": "loud"},
    )))
    assert result.invalid is True
    assert "wrong types" in result.observation
    assert calls == []


def test_executor_passes_well_formed_call_through_to_dispatch_async() -> None:
    executor, calls = _make_executor()
    result = asyncio.run(executor.execute_async(ToolCall(
        name="open_app",
        arguments={"name": "Chrome"},
    )))
    assert result.invalid is False
    assert result.success is True
    assert calls == [("open_app", {"name": "Chrome"})]


# ── Counter / observation contract ───────────────────────────────────


def test_executor_increments_invalid_counter_per_rejection() -> None:
    """Every validator rejection bumps ``_total_invalid`` so the
    dashboard / ReAct telemetry can surface the rate without us
    threading a bus reference into the executor."""
    executor, _ = _make_executor()
    assert executor._total_invalid == 0

    executor.execute(ToolCall(
        name="open_app",
        arguments={"name": "Chrome", "wormhole": True},
    ))
    asyncio.run(executor.execute_async(ToolCall(
        name="set_volume",
        arguments={"level": "loud"},
    )))

    assert executor._total_invalid == 2


def test_invalid_observation_distinct_from_error_observation() -> None:
    """The brain MUST be able to tell ``[INVALID TOOL CALL]`` apart
    from ``[ERROR]`` so its retry policy can differ: invalid -> reword
    the call; error -> back off / report. Same field would collapse
    both into one signal and break ReAct self-correction."""
    invalid = ActionResult(
        tool_name="open_app",
        success=False,
        invalid=True,
        validation_error="unknown args: wormhole",
    )
    runtime_err = ActionResult(
        tool_name="open_app",
        success=False,
        error="Execution failed: socket timeout",
    )
    assert invalid.observation.startswith("[INVALID TOOL CALL]")
    assert runtime_err.observation.startswith("[ERROR]")
    assert invalid.observation != runtime_err.observation
