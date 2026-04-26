from __future__ import annotations

from core.reasoning.tool_registry import ToolRegistry


def test_query_scoped_prompt_keeps_simple_chat_light() -> None:
    registry = ToolRegistry()

    prompt = registry.generate_prompt_tools_section(query="what time is it")

    assert "TOOL CALL FORMAT" in prompt
    assert "recall(" in prompt
    assert "open_app(" not in prompt
    assert "vision_describe(" not in prompt


def test_query_scoped_prompt_includes_relevant_category() -> None:
    registry = ToolRegistry()

    prompt = registry.generate_prompt_tools_section(query="open chrome")

    assert "open_app(" in prompt
    assert "close_app(" in prompt


def test_empty_query_preserves_full_catalogue_for_callers_without_context() -> None:
    registry = ToolRegistry()

    prompt = registry.generate_prompt_tools_section()

    assert "open_app(" in prompt
    assert "vision_describe(" in prompt
