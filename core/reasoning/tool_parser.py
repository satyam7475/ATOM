"""
ATOM -- Tool Call Parser (LLM Response -> Action Execution).

v3 Phase 5 redesign
-------------------
The parser is now a thin recovery layer in front of the constrained
decoding path in ``core.reasoning.tool_grammar``. We accept TWO
formats only -- everything else was historical noise that produced
more false-positives than wins:

  1. Canonical:  <tool_call>{"name": "...", "arguments": {...}}</tool_call>
  2. Naked JSON: {"name": "...", "arguments": {...}}    (recovery path)

The simple ``<tool>name(args)</tool>`` form is kept for tiny models
that occasionally fall back to it. The dead ✿FUNCTION✿ Qwen-legacy
matcher and the bare ``"name":...,"arguments":..."`` regex were both
removed -- they never fired in production logs and routinely matched
prose JSON that wasn't a tool call.

Returns a ToolCallResult with:
  - tool_calls: list of parsed tool invocations
  - text_response: any non-tool text the LLM generated
  - has_tool_calls: whether any tools were invoked
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger("atom.tool_parser")


@dataclass
class ToolCall:
    """A single parsed tool invocation."""
    name: str
    arguments: dict[str, str | int | float | bool]
    raw: str = ""

    def __repr__(self) -> str:
        return f"ToolCall({self.name}, {self.arguments})"


@dataclass
class ToolCallResult:
    """Result of parsing an LLM response for tool calls."""
    tool_calls: list[ToolCall] = field(default_factory=list)
    text_response: str = ""
    raw_response: str = ""

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def first_tool(self) -> ToolCall | None:
        return self.tool_calls[0] if self.tool_calls else None


_TOOL_CALL_PATTERN = re.compile(
    r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
    re.DOTALL,
)

_SIMPLE_TOOL_PATTERN = re.compile(
    r'<tool>\s*(\w+)\s*\(([^)]*)\)\s*</tool>',
)

# v3: a tool-call-shaped naked JSON object: requires BOTH "name" and
# "arguments" keys. The check is intentionally strict so prose JSON
# (e.g. {"city": "Mumbai", "temp": 32}) does NOT get misclassified.
_NAKED_JSON_HINT_RE = re.compile(
    r'\{[^{}]*"name"\s*:\s*"\w+"[^{}]*"arguments"\s*:\s*\{',
    re.DOTALL,
)


def parse_tool_calls(response: str) -> ToolCallResult:
    """Parse an LLM response for tool call instructions.

    Order:
      1. ``<tool_call>{JSON}</tool_call>``       -- canonical
      2. ``<tool>name(args)</tool>``             -- tiny-model fallback
      3. Naked ``{"name": ..., "arguments": ...}`` -- recovery
    """
    result = ToolCallResult(raw_response=response)
    text_parts: list[str] = []
    remaining = response or ""

    canonical_matches = list(_TOOL_CALL_PATTERN.finditer(remaining))
    if canonical_matches:
        for match in canonical_matches:
            call = _parse_json_tool_blob(match.group(1), raw=match.group(0))
            if call is not None:
                result.tool_calls.append(call)
        remaining = _TOOL_CALL_PATTERN.sub("", remaining).strip()
        if remaining:
            text_parts.append(remaining)

    if not result.tool_calls:
        simple_matches = list(_SIMPLE_TOOL_PATTERN.finditer(remaining))
        if simple_matches:
            for match in simple_matches:
                name = match.group(1)
                arguments = _parse_simple_args(match.group(2).strip())
                result.tool_calls.append(ToolCall(
                    name=name, arguments=arguments, raw=match.group(0),
                ))
            remaining = _SIMPLE_TOOL_PATTERN.sub("", remaining).strip()
            if remaining:
                text_parts.append(remaining)

    if not result.tool_calls:
        for blob, raw in _scan_naked_json_blobs(remaining):
            call = _parse_json_tool_blob(blob, raw=raw)
            if call is not None:
                result.tool_calls.append(call)
                remaining = remaining.replace(raw, "", 1).strip()
        if not result.tool_calls:
            text_parts = [response or ""]
        elif remaining:
            text_parts.append(remaining)

    result.text_response = _clean_response_text(" ".join(text_parts).strip())

    if result.tool_calls:
        logger.info(
            "Parsed %d tool call(s): %s",
            len(result.tool_calls),
            ", ".join(tc.name for tc in result.tool_calls),
        )

    return result


def _parse_json_tool_blob(blob: str, *, raw: str) -> ToolCall | None:
    """Parse a JSON object that should look like {"name":..,"arguments":..}.

    Returns ``None`` when the blob is invalid JSON or lacks a name; the
    parser silently drops the candidate rather than raising. This keeps
    a malformed candidate from corrupting the rest of the response text.
    """
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        logger.debug("Tool-call JSON decode failed: %s", blob[:120])
        return None
    if not isinstance(data, dict):
        return None
    name = data.get("name", "")
    if not name or not isinstance(name, str):
        return None
    arguments = data.get("arguments", {})
    if not isinstance(arguments, dict):
        return None
    return ToolCall(name=name, arguments=arguments, raw=raw)


def _scan_naked_json_blobs(text: str) -> list[tuple[str, str]]:
    """Locate top-level JSON objects that look like tool calls.

    Walks the string with a brace-depth counter so nested objects in
    ``arguments`` don't confuse us. Returns ``[(json_blob, raw_blob)]``;
    raw_blob is identical to json_blob (kept for symmetry with the
    matched-pattern path).
    """
    if not text or "{" not in text:
        return []
    if not _NAKED_JSON_HINT_RE.search(text):
        return []
    out: list[tuple[str, str]] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                blob = text[start: i + 1]
                if '"name"' in blob and '"arguments"' in blob:
                    out.append((blob, blob))
                start = -1
    return out


def _parse_simple_args(args_str: str) -> dict:
    """Parse simple comma-separated arguments into a dict."""
    if not args_str:
        return {}

    try:
        return json.loads(f"{{{args_str}}}")
    except json.JSONDecodeError:
        pass

    parts = [p.strip().strip('"').strip("'") for p in args_str.split(",")]
    result: dict[str, str] = {}
    for i, part in enumerate(parts):
        if "=" in part:
            key, _, val = part.partition("=")
            result[key.strip()] = val.strip().strip('"').strip("'")
        else:
            if i == 0:
                result["name"] = part
            elif i == 1:
                result["target"] = part
            else:
                result[f"arg{i}"] = part
    return result


def _clean_response_text(text: str) -> str:
    """Clean up response text after tool extraction."""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^[\s,.;:]+', '', text)
    text = re.sub(r'[\s,.;:]+$', '', text)
    return text
