"""
ATOM -- Confirmation Manager (extracted from Router).

Handles all confirmation flows:
  1. ACTION CONFIRMATIONS -- dangerous actions that need "yes/no" before execution
  2. TOOL CONFIRMATIONS -- LLM-initiated tool calls that need user approval
  3. TIMEOUT HANDLING -- pending confirmations expire after 25-30 seconds

Previously these were inlined in Router._handle_confirmation, bloating
the router with flow-control logic. Now the Router delegates to this module.

Contract:
    set_pending(result) -> prompt_text      # stage a confirmation
    set_pending_tool(tool_call) -> prompt    # stage a tool confirmation
    handle(confirm_or_deny) -> ConfirmationResult  # resolve pending
    has_pending -> bool                     # check if something is waiting

Owner: Satyam
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("atom.confirmation")


@dataclass
class ConfirmationResult:
    """Outcome of a confirmation resolution."""
    resolved: bool = False
    action_result: Any = None  # the ClassifyResult to execute, or None
    tool_call: Any = None      # the ToolCall to execute, or None
    response: str = ""         # text to speak to the user
    timed_out: bool = False
    denied: bool = False


class ConfirmationManager:
    """Manages pending action and tool confirmations with timeouts."""

    __slots__ = (
        "_pending_action", "_pending_tool", "_security",
        "_action_timeout_s", "_tool_timeout_s",
    )

    def __init__(
        self,
        security: Any,
        action_timeout_s: float = 25.0,
        tool_timeout_s: float = 30.0,
    ) -> None:
        self._security = security
        self._pending_action: dict | None = None
        self._pending_tool: dict | None = None
        self._action_timeout_s = action_timeout_s
        self._tool_timeout_s = tool_timeout_s

    @property
    def has_pending(self) -> bool:
        return self._pending_action is not None or self._pending_tool is not None

    # ── Action confirmations ──────────────────────────────────────────

    def requires_confirmation(self, result: Any) -> bool:
        """Check if an action result needs confirmation before execution."""
        try:
            from core.action_safety import default_risk_for_action, risk_requires_confirmation

            action_name = getattr(result, "action", None) or ""
            if risk_requires_confirmation(default_risk_for_action(str(action_name))):
                return True
        except Exception:
            pass

        action = result.action
        if self._security.requires_extra_confirmation(action):
            return True
        try:
            from core.command_registry import get_registry
            registry = get_registry()
            if registry.count > 0:
                return registry.requires_confirmation(action)
        except Exception:
            pass
        return action in {
            "play_youtube", "create_folder", "move_path", "copy_path",
            "close_app", "shutdown_pc", "restart_pc", "logoff",
            "sleep_pc", "empty_recycle_bin", "kill_process",
            "type_text",
        }

    def set_pending_action(self, result: Any) -> str:
        """Stage an action for confirmation. Returns the prompt to speak."""
        self._pending_action = {
            "result": result,
            "created_at": time.monotonic(),
        }
        return self._build_prompt(result)

    def set_pending_tool(self, tool_call: Any) -> str:
        """Stage a tool call for confirmation. Returns the prompt to speak."""
        self._pending_tool = {
            "tool_call": tool_call,
            "created_at": time.monotonic(),
        }
        tool_name = getattr(tool_call, "name", str(tool_call))
        return f"My brain wants to run {tool_name}. Should I go ahead, Boss?"

    def handle(self, confirm_intent: str) -> ConfirmationResult:
        """Resolve a pending confirmation with confirm/deny."""

        # Tool confirmations take priority
        if self._pending_tool is not None:
            return self._resolve_tool(confirm_intent)

        if self._pending_action is None:
            if confirm_intent == "confirm":
                return ConfirmationResult(
                    resolved=True,
                    response="Nothing pending right now, boss.",
                )
            return ConfirmationResult(
                resolved=True, response="Okay boss.",
            )

        age = time.monotonic() - self._pending_action.get("created_at", 0)
        if age > self._action_timeout_s:
            self._pending_action = None
            return ConfirmationResult(
                resolved=True, timed_out=True,
                response="Pending action timed out. Please say it again.",
            )

        if confirm_intent == "deny":
            denied = self._pending_action.get("result") if self._pending_action else None
            self._pending_action = None
            self._audit_confirmation(denied, "denied")
            return ConfirmationResult(
                resolved=True, denied=True,
                response="Okay boss, cancelled.",
            )

        result = self._pending_action.get("result")
        self._pending_action = None
        self._audit_confirmation(result, "confirmed")
        return ConfirmationResult(
            resolved=True, action_result=result,
        )

    def _resolve_tool(self, confirm_intent: str) -> ConfirmationResult:
        pending = self._pending_tool
        self._pending_tool = None
        if pending is None:
            return ConfirmationResult(resolved=True)

        age = time.monotonic() - pending.get("created_at", 0)
        if age > self._tool_timeout_s:
            return ConfirmationResult(
                resolved=True, timed_out=True,
                response="That request timed out, Boss. Say it again.",
            )

        if confirm_intent == "deny":
            return ConfirmationResult(
                resolved=True, denied=True,
                response="Okay Boss, cancelled.",
            )

        tool_call = pending.get("tool_call")
        return ConfirmationResult(
            resolved=True, tool_call=tool_call,
        )

    def clear(self) -> None:
        """Clear all pending confirmations."""
        self._pending_action = None
        self._pending_tool = None

    @staticmethod
    def _audit_confirmation(result: Any, outcome: str) -> None:
        try:
            from core.action_safety import append_audit_record, default_risk_for_action, risk_label

            if result is None:
                return
            action = str(getattr(result, "action", "") or "")
            risk = default_risk_for_action(action)
            append_audit_record(
                actor="user",
                action=action,
                risk=risk_label(risk),
                reason="confirmation_gate",
                result=outcome,
            )
        except Exception:
            pass

    # ── Prompt building ───────────────────────────────────────────────

    @staticmethod
    def _build_prompt(result: Any) -> str:
        from core import adaptive_personality as personality
        args = result.action_args or {}
        action_details = {
            "play_youtube": args.get("query", "music"),
            "create_folder": args.get("name", "this folder"),
            "close_app": args.get("name", "this app"),
            "shutdown_pc": "shutdown",
            "restart_pc": "restart",
            "logoff": "log off",
            "sleep_pc": "sleep",
            "empty_recycle_bin": "empty recycle bin",
            "kill_process": args.get("name", "process"),
            "open_url": args.get("url", "link")[:120],
            "lock_screen": "lock screen",
            "screenshot": "screenshot",
            "move_path": str(args.get("src", "item"))[:80],
            "copy_path": str(args.get("src", "item"))[:80],
        }
        detail = action_details.get(result.action, "")
        return personality.confirmation_prompt(result.action, detail)
