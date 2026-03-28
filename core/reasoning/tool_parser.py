"""
ATOM -- Tool Call Parser (LLM Response -> Action Execution).

Parses LLM responses for tool call instructions in multiple formats:
  1. ATOM native / Qwen3: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
  2. Legacy Qwen function calling: ✿FUNCTION✿ format
  3. Simple inline: <tool>tool_name(arg1, arg2)</tool>

Handles multiple tool calls in a single response.
Separates text response from tool invocations.

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

_QWEN_FUNCTION_PATTERN = re.compile(
    r'✿FUNCTION✿\s*:\s*(\w+)\s*\n(.*?)(?=✿|$)',
    re.DOTALL,
)

_JSON_TOOL_CALL_PATTERN = re.compile(
    r'\{"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}',
)


def parse_tool_calls(response: str) -> ToolCallResult:
    """Parse an LLM response for tool call instructions.

    Tries multiple formats in order of specificity:
    1. <tool_call>{JSON}</tool_call> (ATOM native / Qwen3)
    2. ✿FUNCTION✿ (legacy Qwen format)
    3. <tool>name(args)</tool> (simple format)
    4. Bare JSON tool call patterns
    """
    result = ToolCallResult(raw_response=response)
    text_parts: list[str] = []
    remaining = response

    tool_call_matches = list(_TOOL_CALL_PATTERN.finditer(remaining))
    if tool_call_matches:
        for match in tool_call_matches:
            try:
                data = json.loads(match.group(1))
                name = data.get("name", "")
                arguments = data.get("arguments", {})
                if name:
                    result.tool_calls.append(ToolCall(
                        name=name,
                        arguments=arguments if isinstance(arguments, dict) else {},
                        raw=match.group(0),
                    ))
            except json.JSONDecodeError:
                logger.debug("Failed to parse tool_call JSON: %s", match.group(1)[:100])

        remaining = _TOOL_CALL_PATTERN.sub("", remaining).strip()
        if remaining:
            text_parts.append(remaining)

    if not result.tool_calls:
        qwen_matches = list(_QWEN_FUNCTION_PATTERN.finditer(remaining))
        if qwen_matches:
            for match in qwen_matches:
                name = match.group(1).strip()
                args_str = match.group(2).strip()
                try:
                    arguments = json.loads(args_str) if args_str.startswith("{") else {}
                except json.JSONDecodeError:
                    arguments = {"raw": args_str}
                if name:
                    result.tool_calls.append(ToolCall(
                        name=name, arguments=arguments, raw=match.group(0),
                    ))
            remaining = _QWEN_FUNCTION_PATTERN.sub("", remaining).strip()
            if remaining:
                text_parts.append(remaining)

    if not result.tool_calls:
        simple_matches = list(_SIMPLE_TOOL_PATTERN.finditer(remaining))
        if simple_matches:
            for match in simple_matches:
                name = match.group(1)
                args_str = match.group(2).strip()
                arguments = _parse_simple_args(args_str)
                result.tool_calls.append(ToolCall(
                    name=name, arguments=arguments, raw=match.group(0),
                ))
            remaining = _SIMPLE_TOOL_PATTERN.sub("", remaining).strip()
            if remaining:
                text_parts.append(remaining)

    if not result.tool_calls:
        json_matches = list(_JSON_TOOL_CALL_PATTERN.finditer(remaining))
        if json_matches:
            for match in json_matches:
                name = match.group(1)
                try:
                    arguments = json.loads(match.group(2))
                except json.JSONDecodeError:
                    arguments = {}
                result.tool_calls.append(ToolCall(
                    name=name, arguments=arguments, raw=match.group(0),
                ))
            remaining = _JSON_TOOL_CALL_PATTERN.sub("", remaining).strip()
            if remaining:
                text_parts.append(remaining)

    if not result.tool_calls:
        text_parts = [response]

    result.text_response = " ".join(text_parts).strip()

    result.text_response = _clean_response_text(result.text_response)

    if result.tool_calls:
        logger.info(
            "Parsed %d tool call(s): %s",
            len(result.tool_calls),
            ", ".join(tc.name for tc in result.tool_calls),
        )

    return result


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
