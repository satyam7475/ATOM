"""
ATOM -- Tool Registry (Formal Tool Registration System).

Every action ATOM can perform is registered as a Tool with:
  - name, description, parameters, safety_level
  - The LLM sees this registry in its prompt and can dynamically
    choose which tools to call and with what parameters.

This replaces the hardcoded dispatch table with a formal, extensible
tool system -- the difference between a remote control and true AI.

Safety Levels:
    safe       -- No confirmation needed (time, volume, scroll)
    moderate   -- Logged but auto-executed (open_app, search)
    dangerous  -- Requires confirmation (shutdown, delete, kill)
    blocked    -- Never auto-executed (format disk, etc.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("atom.tools")


@dataclass
class ToolParameter:
    """Single parameter for a registered tool."""
    name: str
    type: str = "string"
    description: str = ""
    required: bool = False
    default: Any = None
    enum: list[str] | None = None


@dataclass
class Tool:
    """Registered tool that the LLM can invoke."""
    name: str
    description: str
    category: str = "general"
    safety_level: str = "safe"
    parameters: list[ToolParameter] = field(default_factory=list)
    handler: Callable[..., str | None] | None = None
    requires_confirmation: bool = False
    examples: list[str] = field(default_factory=list)

    def to_prompt_description(self) -> str:
        """Format this tool for LLM prompt injection."""
        params_str = ""
        if self.parameters:
            param_parts = []
            for p in self.parameters:
                req = " (required)" if p.required else ""
                param_parts.append(f"{p.name}: {p.type}{req} - {p.description}")
            params_str = "\n    Parameters: " + ", ".join(param_parts)
        return f"  - {self.name}: {self.description}{params_str}"

    def to_function_schema(self) -> dict:
        """Format as a function-calling schema (OpenAI/Qwen3 compatible)."""
        properties = {}
        required = []
        for p in self.parameters:
            prop: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            properties[p.name] = prop
            if p.required:
                required.append(p.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class ToolRegistry:
    """Central registry of all tools ATOM can use."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._categories: dict[str, list[str]] = {}
        self._register_builtin_tools()

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        cat_list = self._categories.setdefault(tool.category, [])
        if tool.name not in cat_list:
            cat_list.append(tool.name)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_all(self) -> list[Tool]:
        return list(self._tools.values())

    def get_by_category(self, category: str) -> list[Tool]:
        names = self._categories.get(category, [])
        return [self._tools[n] for n in names if n in self._tools]

    @property
    def count(self) -> int:
        return len(self._tools)

    def generate_prompt_tools_section(self) -> str:
        """Generate the tools section for the LLM prompt."""
        lines = ["AVAILABLE TOOLS (you can call these to perform actions):\n"]
        for cat in sorted(self._categories.keys()):
            tools = self.get_by_category(cat)
            if not tools:
                continue
            lines.append(f"  [{cat.upper()}]")
            for tool in tools:
                lines.append(tool.to_prompt_description())
            lines.append("")

        lines.append("TOOL CALL FORMAT:")
        lines.append('  To call a tool: <tool_call>{"name": "tool_name", "arguments": {"param": "value"}}</tool_call>')
        lines.append("  You can call multiple tools in one response.")
        lines.append("  To respond with text only, just respond normally without tool_call tags.")
        lines.append("")
        return "\n".join(lines)

    def generate_function_schemas(self) -> list[dict]:
        """Generate function-calling schemas for all tools."""
        return [t.to_function_schema() for t in self._tools.values()]

    def _register_builtin_tools(self) -> None:
        """Register all built-in ATOM tools."""
        builtins = [
            Tool("open_app", "Open an application by name", "apps", "moderate",
                 [ToolParameter("name", "string", "Application name", True)],
                 examples=["open chrome", "open notepad"]),
            Tool("close_app", "Close a running application", "apps", "dangerous",
                 [ToolParameter("name", "string", "Application name", True)],
                 requires_confirmation=True),
            Tool("list_apps", "List all running applications", "apps", "safe"),
            Tool("search", "Search the web for information", "web", "moderate",
                 [ToolParameter("query", "string", "Search query", True)]),
            Tool("open_url", "Open a URL in the browser", "web", "moderate",
                 [ToolParameter("url", "string", "URL to open", True)]),
            Tool("set_volume", "Set system volume level", "media", "safe",
                 [ToolParameter("percent", "integer", "Volume 0-100", True)]),
            Tool("mute", "Mute system audio", "media", "safe"),
            Tool("unmute", "Unmute system audio", "media", "safe"),
            Tool("play_youtube", "Play a YouTube video", "media", "moderate",
                 [ToolParameter("query", "string", "Video search query", True)],
                 requires_confirmation=True),
            Tool("stop_music", "Stop current media playback", "media", "safe"),
            Tool("screenshot", "Take a screenshot", "system", "moderate"),
            Tool("lock_screen", "Lock the workstation", "system", "moderate"),
            Tool("set_brightness", "Set screen brightness", "system", "safe",
                 [ToolParameter("percent", "integer", "Brightness 0-100", True)]),
            Tool("shutdown_pc", "Shut down the computer", "system", "dangerous",
                 requires_confirmation=True),
            Tool("restart_pc", "Restart the computer", "system", "dangerous",
                 requires_confirmation=True),
            Tool("sleep_pc", "Put computer to sleep", "system", "dangerous",
                 requires_confirmation=True),
            Tool("create_folder", "Create a new folder", "files", "dangerous",
                 [ToolParameter("name", "string", "Folder name", True),
                  ToolParameter("path", "string", "Parent path")],
                 requires_confirmation=True),
            Tool("move_path", "Move a file or folder", "files", "dangerous",
                 [ToolParameter("src", "string", "Source path", True),
                  ToolParameter("dst", "string", "Destination path", True)],
                 requires_confirmation=True),
            Tool("copy_path", "Copy a file or folder", "files", "dangerous",
                 [ToolParameter("src", "string", "Source path", True),
                  ToolParameter("dst", "string", "Destination path", True)],
                 requires_confirmation=True),
            Tool("set_reminder", "Set a timed reminder", "productivity", "safe",
                 [ToolParameter("label", "string", "Reminder text", True),
                  ToolParameter("delay_seconds", "integer", "Delay in seconds", True)]),
            Tool("show_reminders", "Show all pending reminders", "productivity", "safe"),
            Tool("scroll_down", "Scroll down on active window", "desktop", "safe",
                 [ToolParameter("clicks", "integer", "Scroll amount", False, 5)]),
            Tool("scroll_up", "Scroll up on active window", "desktop", "safe",
                 [ToolParameter("clicks", "integer", "Scroll amount", False, 5)]),
            Tool("press_key", "Press a keyboard key", "desktop", "moderate",
                 [ToolParameter("key", "string", "Key name", True)]),
            Tool("type_text", "Type text into active window", "desktop", "moderate",
                 [ToolParameter("text", "string", "Text to type", True)]),
            Tool("click_screen", "Click at mouse position", "desktop", "moderate"),
            Tool("hotkey_combo", "Press a hotkey combination", "desktop", "moderate",
                 [ToolParameter("combo", "string", "Hotkey combo like ctrl+c", True)]),
            Tool("minimize_window", "Minimize active window", "desktop", "safe"),
            Tool("maximize_window", "Maximize active window", "desktop", "safe"),
            Tool("switch_window", "Switch to next window (Alt+Tab)", "desktop", "safe"),
            Tool("kill_process", "Kill a running process", "system", "dangerous",
                 [ToolParameter("name", "string", "Process name", True)],
                 requires_confirmation=True),
            Tool("weather", "Check current weather", "info", "safe",
                 [ToolParameter("location", "string", "City name")]),
            Tool("wifi_status", "Check WiFi connection status", "info", "safe"),
            Tool("flush_dns", "Flush DNS cache", "system", "moderate"),
            Tool("empty_recycle_bin", "Empty the recycle bin", "system", "dangerous",
                 requires_confirmation=True),
            Tool("get_ip", "Get IP address", "info", "safe"),
            Tool("calculate", "Evaluate a mathematical expression safely", "utility", "safe",
                 [ToolParameter("expression", "string", "Math expression", True)]),
            Tool("remember", "Store a fact in long-term memory", "memory", "safe",
                 [ToolParameter("fact", "string", "Fact to remember", True)]),
            Tool("recall", "Recall information from memory", "memory", "safe",
                 [ToolParameter("query", "string", "What to recall", True)]),
            Tool("learn_document", "Ingest a document into knowledge base", "memory", "safe",
                 [ToolParameter("path", "string", "File path to learn", True)]),
            Tool("set_goal", "Create a new goal", "goals", "safe",
                 [ToolParameter("title", "string", "Goal description", True)]),
            Tool("show_goals", "Show active goals", "goals", "safe"),
            Tool("run_code", "Execute a Python expression safely", "utility", "moderate",
                 [ToolParameter("code", "string", "Python code to execute", True)]),
        ]

        for tool in builtins:
            self.register(tool)

        logger.info("Tool registry initialized: %d tools across %d categories",
                     self.count, len(self._categories))


_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
