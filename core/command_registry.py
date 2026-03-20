"""
ATOM v14 -- Central command registry.

Loads command definitions from config/commands.json and provides
a runtime lookup API for the router.

Benefits:
    - Single source of truth for all command metadata
    - Confirmation requirements defined in config, not code
    - Easy to add/remove commands without touching router logic
    - Introspectable: list all commands, filter by category
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom.registry")

_DEFAULT_COMMANDS_PATH = Path("config/commands.json")


class CommandDefinition:
    """Immutable metadata for a single registered command."""

    __slots__ = ("action", "category", "confirm", "description")

    def __init__(self, action: str, category: str = "other",
                 confirm: bool = False, description: str = "") -> None:
        self.action = action
        self.category = category
        self.confirm = confirm
        self.description = description

    def __repr__(self) -> str:
        return (f"CommandDefinition(action={self.action!r}, "
                f"category={self.category!r}, confirm={self.confirm})")


class CommandRegistry:
    """Loads and serves command definitions at runtime."""

    __slots__ = ("_commands", "_by_action", "_by_category")

    def __init__(self) -> None:
        self._commands: list[CommandDefinition] = []
        self._by_action: dict[str, CommandDefinition] = {}
        self._by_category: dict[str, list[CommandDefinition]] = {}

    def load(self, path: Path | None = None) -> None:
        """Load commands from JSON file."""
        path = path or _DEFAULT_COMMANDS_PATH
        if not path.exists():
            logger.warning("Commands file not found: %s -- using defaults",
                           path)
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.error("Failed to load commands from %s: %s", path, exc)
            return

        entries = data.get("commands", [])
        for entry in entries:
            if not isinstance(entry, dict) or "action" not in entry:
                continue
            cmd = CommandDefinition(
                action=entry["action"],
                category=entry.get("category", "other"),
                confirm=entry.get("confirm", False),
                description=entry.get("description", ""),
            )
            self._commands.append(cmd)
            self._by_action[cmd.action] = cmd
            self._by_category.setdefault(cmd.category, []).append(cmd)

        logger.info("Command registry loaded: %d commands from %s",
                     len(self._commands), path)

    def get(self, action: str) -> CommandDefinition | None:
        """Look up a command by action name."""
        return self._by_action.get(action)

    def requires_confirmation(self, action: str) -> bool:
        """Check if an action requires user confirmation."""
        cmd = self._by_action.get(action)
        if cmd is not None:
            return cmd.confirm
        return False

    def list_by_category(self, category: str) -> list[CommandDefinition]:
        return self._by_category.get(category, [])

    def all_commands(self) -> list[CommandDefinition]:
        return list(self._commands)

    def categories(self) -> list[str]:
        return sorted(self._by_category.keys())

    @property
    def count(self) -> int:
        return len(self._commands)

    def summary(self) -> str:
        """Human-readable summary of registered commands."""
        parts = [f"Registered commands: {self.count}"]
        for cat in self.categories():
            cmds = self._by_category[cat]
            names = ", ".join(c.action for c in cmds)
            parts.append(f"  {cat} ({len(cmds)}): {names}")
        return "\n".join(parts)


_global_registry: CommandRegistry | None = None


def get_registry() -> CommandRegistry:
    """Get or create the global command registry singleton."""
    global _global_registry
    if _global_registry is None:
        _global_registry = CommandRegistry()
        _global_registry.load()
    return _global_registry
