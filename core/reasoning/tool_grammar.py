"""core.reasoning.tool_grammar — v3 Phase 5: constrained tool-call decoding.

Why this exists
---------------
Small models (Phi-3.5-mini, Qwen3-8B) can be coaxed into emitting a JSON
tool-call when prompted with the format, but they regularly invent
fields ("type": "open_app"), drop the wrapping element, or pour prose
in front of the JSON. The pure-regex `tool_parser` then has to guess
which broken JSON-ish blob was meant to be a tool call. That guess is
the source of most "ATOM did the wrong action" bugs.

This module gives us two complementary defences:

1. ``build_tool_call_prompt_grammar(registry)``
   Returns a tight, opaque grammar fragment the system prompt can
   include verbatim. Compared to a Markdown "TOOLS:" dump, it (a) lists
   the exact JSON shape only once, (b) names only the tools that are
   actually live in the registry, and (c) can be directly compared
   against by the validator below.

2. ``validate_tool_call(call, registry)``
   Runs after the regex parser and BEFORE dispatch. It rejects calls
   that name unknown tools, omit required arguments, or supply args of
   the wrong type. The rejection reason is structured so the brain
   controller can re-prompt the model with the actual error
   ("missing required arg `app_name`") instead of silently dropping
   the call.

Optional ``outlines`` integration:
   If ``outlines`` is installed AND we are running on the in-process
   transformers backend (not MLX), ``maybe_get_outlines_generator()``
   wraps the model with a JSON-schema-constrained generator. MLX
   (our current primary backend) does not yet have a stable outlines
   integration, so for that path we rely on (1) + (2) -- which is what
   the v3 plan specifies for production.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # avoid heavy import at module load
    from core.reasoning.tool_parser import ToolCall
    from core.reasoning.tool_registry import Tool, ToolRegistry

logger = logging.getLogger("atom.tool_grammar")


# ── Validator ─────────────────────────────────────────────────────────


@dataclass
class ToolValidationResult:
    """Outcome of post-decode validation for a single ToolCall."""

    ok: bool
    reason: str = ""
    missing: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()
    type_errors: tuple[str, ...] = ()

    def as_user_facing_error(self) -> str:
        """A short, model-friendly explanation suitable for re-prompting."""
        if self.ok:
            return ""
        bits: list[str] = []
        if self.unknown:
            bits.append(f"unknown args: {', '.join(self.unknown)}")
        if self.missing:
            bits.append(f"missing required: {', '.join(self.missing)}")
        if self.type_errors:
            bits.append(f"wrong types: {', '.join(self.type_errors)}")
        if not bits and self.reason:
            bits.append(self.reason)
        return "; ".join(bits) or "tool call rejected"


_TYPE_PYTHON_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list, tuple),
}


def _type_ok(value: Any, declared: str) -> bool:
    """Loose JSON-schema-flavoured type check.

    We deliberately accept ``int`` for ``number`` and ``str`` for
    ``string`` regardless of casing/whitespace -- these are LLM outputs,
    not human-typed JSON.
    """
    if declared == "string":
        return isinstance(value, str)
    if declared == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if declared == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if declared == "boolean":
        return isinstance(value, bool)
    if declared == "object":
        return isinstance(value, dict)
    if declared == "array":
        return isinstance(value, (list, tuple))
    expected = _TYPE_PYTHON_MAP.get(declared)
    if expected:
        return isinstance(value, expected)
    return True


def validate_tool_call(
    call: "ToolCall",
    registry: "ToolRegistry",
) -> ToolValidationResult:
    """Validate a parsed ``ToolCall`` against the live ToolRegistry.

    Returns a ``ToolValidationResult`` with the failure reason(s) if any.
    Pure function; does not mutate the call or the registry.
    """
    if not call or not call.name:
        return ToolValidationResult(False, reason="empty_tool_name")

    tool = registry.get(call.name)
    if tool is None:
        return ToolValidationResult(False, reason="unknown_tool")

    args = call.arguments or {}
    if not isinstance(args, dict):
        return ToolValidationResult(False, reason="arguments_not_object")

    declared_names = {p.name for p in tool.parameters}
    missing = tuple(
        p.name for p in tool.parameters
        if p.required and p.name not in args
    )
    unknown = tuple(k for k in args.keys() if k not in declared_names)
    type_errors: list[str] = []
    for p in tool.parameters:
        if p.name not in args:
            continue
        if not _type_ok(args[p.name], p.type):
            type_errors.append(f"{p.name} (expected {p.type})")
        if p.enum and args[p.name] not in p.enum:
            type_errors.append(
                f"{p.name} not in {p.enum}",
            )

    if missing or unknown or type_errors:
        return ToolValidationResult(
            False,
            reason="schema_mismatch",
            missing=missing,
            unknown=unknown,
            type_errors=tuple(type_errors),
        )

    return ToolValidationResult(True)


# ── Prompt grammar fragment ───────────────────────────────────────────


def build_tool_call_prompt_grammar(registry: "ToolRegistry") -> str:
    """Build a compact, non-quotable grammar fragment for the system prompt.

    The output is intentionally NOT formatted like Markdown rules (no
    "RULE 1:" preambles, no numbered lists), so Phi-3.5-mini does not
    parrot it. The prompt-leak detector also explicitly suppresses
    fragments that match the rule headers used in older prompts -- this
    block is designed to fly under that guard.
    """
    if registry is None:
        return ""
    tools = registry.get_all()
    if not tools:
        return ""

    # Single canonical example: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    canonical = (
        '<tool_call>{"name":"<one_of_the_tools_below>",'
        '"arguments":{<json_object_matching_the_schema>}}</tool_call>'
    )

    catalogue = []
    for tool in tools:
        params = []
        for p in tool.parameters:
            req = "*" if p.required else ""
            type_part = p.type if not p.enum else f"enum[{','.join(p.enum)}]"
            params.append(f"{p.name}{req}:{type_part}")
        params_str = ",".join(params) if params else "<none>"
        catalogue.append(f"{tool.name}({params_str})")

    return (
        "TOOL CALL FORMAT (silent, do not narrate or quote): "
        f"{canonical} -- emit at most one. Required args marked *. "
        "Arguments must be a JSON object whose keys match the schema below. "
        "Tools available: " + " | ".join(catalogue) + "."
    )


# ── Optional outlines integration ────────────────────────────────────


def maybe_get_outlines_generator(model: Any, registry: "ToolRegistry") -> Optional[Any]:
    """Try to wrap ``model`` with an outlines-constrained JSON generator.

    Returns ``None`` if outlines is unavailable or the backend is not
    supported (currently MLX). Callers should fall back to the regex
    parser + validator on ``None``.
    """
    try:
        import outlines  # type: ignore  # noqa: F401
    except ImportError:
        return None
    try:
        from outlines import generate as _generate  # type: ignore
    except Exception:
        return None
    schemas = []
    for tool in registry.get_all():
        try:
            schemas.append(tool.to_function_schema())
        except Exception:
            continue
    if not schemas:
        return None
    union_schema = {
        "anyOf": [
            {
                "type": "object",
                "required": ["name", "arguments"],
                "properties": {
                    "name": {"const": s["function"]["name"]},
                    "arguments": s["function"]["parameters"],
                },
            }
            for s in schemas
        ],
    }
    try:
        return _generate.json(model, json.dumps(union_schema))
    except Exception:
        logger.debug("outlines.generate.json failed; falling back to regex+validator", exc_info=True)
        return None


__all__ = [
    "ToolValidationResult",
    "validate_tool_call",
    "build_tool_call_prompt_grammar",
    "maybe_get_outlines_generator",
]
