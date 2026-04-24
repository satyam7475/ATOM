"""v3 Phase 5 — tool_parser simplification + tool_grammar validator tests.

Covers:
  * The 3 surviving parser paths (canonical, simple, naked-JSON recovery).
  * The dropped legacy paths no longer false-positive on prose JSON.
  * Validator rejects unknown tools, missing required args, type mismatches.
  * Validator returns user-facing error suitable for re-prompting.
  * Prompt grammar fragment is opaque (no quotable rule text the model
    will parrot back to TTS).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.reasoning.tool_grammar import (
    ToolValidationResult,
    build_tool_call_prompt_grammar,
    validate_tool_call,
)
from core.reasoning.tool_parser import ToolCall, parse_tool_calls
from core.reasoning.tool_registry import Tool, ToolParameter, ToolRegistry


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_registry() -> ToolRegistry:
    """A tiny registry with two well-known tools we can reason about."""
    reg = ToolRegistry()
    reg.register(Tool(
        name="open_app",
        description="Open a desktop application by name.",
        parameters=[
            ToolParameter(name="app_name", type="string", required=True),
        ],
    ))
    reg.register(Tool(
        name="set_volume",
        description="Set system volume.",
        parameters=[
            ToolParameter(
                name="level", type="integer", required=True,
            ),
            ToolParameter(
                name="mode", type="string", required=False,
                enum=["mute", "unmute", "set"],
            ),
        ],
    ))
    return reg


# ── Parser: canonical <tool_call>{json}</tool_call> ──────────────────


def test_parser_extracts_canonical_tool_call() -> None:
    raw = (
        'Sure boss, opening Chrome. '
        '<tool_call>{"name": "open_app", "arguments": {"app_name": "Chrome"}}</tool_call>'
    )
    out = parse_tool_calls(raw)
    assert out.has_tool_calls
    assert out.tool_calls[0].name == "open_app"
    assert out.tool_calls[0].arguments == {"app_name": "Chrome"}
    assert "Sure boss" in out.text_response
    assert "<tool_call>" not in out.text_response


def test_parser_extracts_multiple_canonical_calls() -> None:
    raw = (
        '<tool_call>{"name": "open_app", "arguments": {"app_name": "Chrome"}}</tool_call>'
        '<tool_call>{"name": "set_volume", "arguments": {"level": 50}}</tool_call>'
    )
    out = parse_tool_calls(raw)
    assert len(out.tool_calls) == 2
    assert {tc.name for tc in out.tool_calls} == {"open_app", "set_volume"}


def test_parser_skips_invalid_canonical_json_silently() -> None:
    raw = '<tool_call>{not valid json}</tool_call> hello'
    out = parse_tool_calls(raw)
    assert not out.has_tool_calls
    # The cleaned text should still contain a recognisable fragment.
    assert "hello" in out.text_response


# ── Parser: <tool>name(args)</tool> tiny-model fallback ──────────────


def test_parser_extracts_simple_tool() -> None:
    raw = '<tool>open_app("Safari")</tool>'
    out = parse_tool_calls(raw)
    assert out.has_tool_calls
    assert out.tool_calls[0].name == "open_app"


# ── Parser: naked JSON recovery ──────────────────────────────────────


def test_parser_recovers_naked_json_tool_call() -> None:
    """A naked tool-call-shaped JSON object is recovered. This is the
    common small-model mistake -- forgetting the <tool_call> wrap."""
    raw = (
        'Opening Chrome.\n'
        '{"name": "open_app", "arguments": {"app_name": "Chrome"}}'
    )
    out = parse_tool_calls(raw)
    assert out.has_tool_calls
    assert out.tool_calls[0].name == "open_app"
    assert out.tool_calls[0].arguments == {"app_name": "Chrome"}


def test_parser_recovers_naked_json_with_nested_args() -> None:
    """Nested objects inside arguments must not break brace tracking."""
    raw = (
        '{"name": "set_volume", "arguments": {"level": 25, '
        '"meta": {"source": "user"}}}'
    )
    out = parse_tool_calls(raw)
    assert out.has_tool_calls
    assert out.tool_calls[0].name == "set_volume"
    assert out.tool_calls[0].arguments["level"] == 25


def test_parser_does_not_misinterpret_prose_json() -> None:
    """Prose JSON without name+arguments must NOT be parsed as a tool call.
    This is the regression that the dropped _JSON_TOOL_CALL_PATTERN used
    to cause -- it would happily treat {"city": "Mumbai"} as a tool."""
    raw = (
        'The forecast for today is {"city": "Mumbai", "temp": 32}. '
        'Stay hydrated, Boss.'
    )
    out = parse_tool_calls(raw)
    assert not out.has_tool_calls
    assert "Mumbai" in out.text_response


def test_parser_does_not_match_legacy_qwen_function_marker() -> None:
    """The dropped ✿FUNCTION✿ matcher must no longer fire."""
    raw = '✿FUNCTION✿: do_something\n{"foo": "bar"}'
    out = parse_tool_calls(raw)
    assert not out.has_tool_calls
    assert "do_something" in out.text_response or "FUNCTION" in out.text_response


def test_parser_handles_empty_response() -> None:
    out = parse_tool_calls("")
    assert not out.has_tool_calls
    assert out.text_response == ""


def test_parser_handles_pure_text_response() -> None:
    out = parse_tool_calls("It's 3:45 PM, Boss.")
    assert not out.has_tool_calls
    assert "3:45 PM" in out.text_response


# ── Validator ────────────────────────────────────────────────────────


def test_validator_accepts_well_formed_call() -> None:
    reg = _make_registry()
    call = ToolCall(name="open_app", arguments={"app_name": "Chrome"})
    res = validate_tool_call(call, reg)
    assert res.ok is True
    assert res.as_user_facing_error() == ""


def test_validator_rejects_unknown_tool() -> None:
    reg = _make_registry()
    call = ToolCall(name="nuke_universe", arguments={})
    res = validate_tool_call(call, reg)
    assert res.ok is False
    assert res.reason == "unknown_tool"


def test_validator_rejects_missing_required_arg() -> None:
    reg = _make_registry()
    call = ToolCall(name="open_app", arguments={})
    res = validate_tool_call(call, reg)
    assert res.ok is False
    assert "app_name" in res.missing
    assert "missing required" in res.as_user_facing_error()


def test_validator_rejects_unknown_arg() -> None:
    reg = _make_registry()
    call = ToolCall(
        name="open_app",
        arguments={"app_name": "Chrome", "wormhole": True},
    )
    res = validate_tool_call(call, reg)
    assert res.ok is False
    assert "wormhole" in res.unknown


def test_validator_rejects_wrong_type() -> None:
    reg = _make_registry()
    call = ToolCall(name="set_volume", arguments={"level": "loud"})
    res = validate_tool_call(call, reg)
    assert res.ok is False
    assert any("level" in t for t in res.type_errors)


def test_validator_enforces_enum_constraint() -> None:
    reg = _make_registry()
    call = ToolCall(
        name="set_volume",
        arguments={"level": 50, "mode": "explode"},
    )
    res = validate_tool_call(call, reg)
    assert res.ok is False
    assert any("mode" in t for t in res.type_errors)


def test_validator_rejects_non_object_arguments() -> None:
    reg = _make_registry()
    call = ToolCall(name="open_app", arguments=["Chrome"])  # type: ignore[arg-type]
    res = validate_tool_call(call, reg)
    assert res.ok is False
    assert res.reason == "arguments_not_object"


def test_validator_rejects_empty_name() -> None:
    reg = _make_registry()
    call = ToolCall(name="", arguments={})
    res = validate_tool_call(call, reg)
    assert res.ok is False
    assert res.reason == "empty_tool_name"


def test_validation_result_user_facing_error_is_concise() -> None:
    res = ToolValidationResult(
        ok=False,
        missing=("app_name",),
        unknown=("foo",),
        type_errors=(),
    )
    msg = res.as_user_facing_error()
    assert "missing required" in msg
    assert "unknown args" in msg
    assert len(msg) < 120, "re-prompting message must stay short"


# ── Prompt grammar fragment ──────────────────────────────────────────


def test_prompt_grammar_lists_registered_tools() -> None:
    reg = _make_registry()
    g = build_tool_call_prompt_grammar(reg)
    assert "open_app" in g
    assert "set_volume" in g
    assert "TOOL CALL FORMAT" in g
    # Required-arg marker should be present.
    assert "*" in g


def test_prompt_grammar_is_opaque_and_single_line() -> None:
    """The fragment must be a single sentence-style block, not a
    numbered RULE-list that Phi will parrot back to TTS."""
    reg = _make_registry()
    g = build_tool_call_prompt_grammar(reg)
    assert "RULE" not in g.upper()
    assert "FINAL-ANSWER" not in g.upper()
    assert g.count("\n") <= 1, "grammar must stay on a single line"
    assert "TOOL CALL FORMAT" in g


def test_prompt_grammar_per_tool_overhead_is_bounded() -> None:
    """The grammar should stay terse per tool. Length scales with the
    registry, but the per-tool overhead must be tight so a 60-tool
    registry doesn't blow the context budget. Exists to catch a
    regression where someone re-introduces a multi-line/per-tool
    description block."""
    reg = ToolRegistry()  # built-ins
    g = build_tool_call_prompt_grammar(reg)
    n_tools = max(1, reg.count)
    per_tool_chars = len(g) / n_tools
    assert per_tool_chars < 80, (
        f"per-tool overhead too high ({per_tool_chars:.0f} chars/tool); "
        f"someone may have added verbose descriptions"
    )


def test_prompt_grammar_handles_empty_registry() -> None:
    reg = ToolRegistry()
    # An empty registry has the built-in tools wired in __init__; that's
    # fine -- we just want this not to raise.
    g = build_tool_call_prompt_grammar(reg)
    assert isinstance(g, str)


def test_prompt_grammar_handles_none_registry() -> None:
    assert build_tool_call_prompt_grammar(None) == ""


# ── End-to-end: parse → validate → friendly rejection ────────────────


def test_parse_then_validate_round_trip() -> None:
    """Real-world flow: parse the model's text, then validate against
    the live registry; reject calls that violate the schema."""
    reg = _make_registry()
    raw = (
        '<tool_call>{"name": "open_app", "arguments": {}}</tool_call>'
    )
    parsed = parse_tool_calls(raw)
    assert parsed.has_tool_calls
    res = validate_tool_call(parsed.tool_calls[0], reg)
    assert res.ok is False
    assert "app_name" in res.missing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
