"""
ATOM -- Action Executor (Security-Gated Tool Execution Bridge).

Bridges parsed ToolCall objects from the LLM to actual system actions.
Every tool call flows through:
    1. ToolRegistry validation (does this tool exist?)
    2. Parameter validation (are required params present?)
    3. SecurityPolicy gate (is this action allowed?)
    4. Confirmation gate (does this action need user confirmation?)
    5. Dispatch to the registered handler
    6. Result capture for LLM feedback (ReAct loop)

The ActionExecutor does NOT own the action handlers -- it receives a
dispatch function from the Router at init time. This avoids circular
dependencies while keeping the Router as the single source of truth
for action implementations.

Contract:
    execute(ToolCall) -> ActionResult
    execute_batch(list[ToolCall]) -> list[ActionResult]
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from core.reasoning.tool_parser import ToolCall
from core.reasoning.tool_registry import ToolRegistry, get_tool_registry
from core.security_policy import SecurityPolicy

logger = logging.getLogger("atom.action_executor")

DispatchFn = Callable[[str, dict], Optional[str]]


@dataclass
class ActionResult:
    """Result of executing a single tool call."""
    tool_name: str
    success: bool
    output: str = ""
    error: str = ""
    elapsed_ms: float = 0.0
    needs_confirmation: bool = False
    confirmation_prompt: str = ""
    blocked: bool = False
    block_reason: str = ""

    @property
    def observation(self) -> str:
        """Format for injection back into LLM as an observation."""
        if self.blocked:
            return f"[BLOCKED] {self.tool_name}: {self.block_reason}"
        if self.needs_confirmation:
            return f"[AWAITING CONFIRMATION] {self.tool_name}: {self.confirmation_prompt}"
        if not self.success:
            return f"[ERROR] {self.tool_name}: {self.error}"
        return f"[OK] {self.tool_name}: {self.output or 'Done.'}"


class ActionExecutor:
    """Security-gated executor that bridges LLM tool calls to system actions.

    Initialized with:
        dispatch_fn: Router._dispatch_action bound method
        security: SecurityPolicy instance
        registry: ToolRegistry (optional, uses global singleton)
    """

    def __init__(
        self,
        dispatch_fn: DispatchFn,
        security: SecurityPolicy,
        registry: ToolRegistry | None = None,
        *,
        timeline: Any = None,
        max_actions_per_turn: int = 5,
    ) -> None:
        self._dispatch = dispatch_fn
        self._security = security
        self._registry = registry or get_tool_registry()
        self._timeline = timeline
        self._max_per_turn = max_actions_per_turn
        self._total_executions = 0
        self._total_blocked = 0

    def set_registry(self, registry: ToolRegistry) -> None:
        """Update the tool registry (e.g., after full initialization)."""
        self._registry = registry

    def execute(self, tool_call: ToolCall) -> ActionResult:
        """Execute a single tool call through the security pipeline."""
        t0 = time.perf_counter()
        name = tool_call.name
        args = dict(tool_call.arguments)

        tool_def = self._registry.get(name)
        if tool_def is None:
            logger.warning("Unknown tool: %s", name)
            return ActionResult(
                tool_name=name, success=False,
                error=f"Unknown tool '{name}'. Check available tools.",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        missing = self._validate_params(tool_def, args)
        if missing:
            return ActionResult(
                tool_name=name, success=False,
                error=f"Missing required parameters: {', '.join(missing)}",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        if tool_def.safety_level == "blocked":
            self._total_blocked += 1
            return ActionResult(
                tool_name=name, success=False,
                blocked=True, block_reason="This action is permanently blocked for safety.",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        from core.execution.behavior_monitor import strip_signing_keys
        from core.security.action_signing import merge_signed_args

        signed_args = merge_signed_args(self._security, name, args)
        allowed, reason = self._security.allow_action(name, signed_args)
        if not allowed:
            self._total_blocked += 1
            logger.warning("Security BLOCKED tool call '%s': %s", name, reason)
            if self._timeline is not None:
                try:
                    self._timeline.append_event(
                        "action",
                        {"tool": name, "blocked": True, "reason": reason[:200]},
                    )
                except Exception:
                    pass
            return ActionResult(
                tool_name=name, success=False,
                blocked=True, block_reason=reason,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        if self._registry.requires_confirmation(name):
            from core import adaptive_personality as personality
            detail = self._confirmation_detail(name, args)
            prompt = personality.confirmation_prompt(name, detail)
            return ActionResult(
                tool_name=name, success=False,
                needs_confirmation=True, confirmation_prompt=prompt,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        self._security.audit_log(
            name, f"args={strip_signing_keys(signed_args)}" if signed_args else "",
        )

        try:
            response = self._dispatch(name, strip_signing_keys(signed_args))
            elapsed = (time.perf_counter() - t0) * 1000
            self._total_executions += 1
            logger.info("Executed tool '%s' in %.0fms", name, elapsed)
            if self._timeline is not None:
                try:
                    self._timeline.append_event(
                        "action",
                        {"tool": name, "success": True, "elapsed_ms": elapsed},
                    )
                except Exception:
                    pass
            return ActionResult(
                tool_name=name, success=True,
                output=response or "Action completed.",
                elapsed_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            logger.error("Tool execution failed '%s': %s", name, exc)
            if self._timeline is not None:
                try:
                    self._timeline.append_event(
                        "error",
                        {"source": "action_executor", "tool": name, "message": str(exc)[:300]},
                    )
                except Exception:
                    pass
            return ActionResult(
                tool_name=name, success=False,
                error=f"Execution failed: {str(exc)[:200]}",
                elapsed_ms=elapsed,
            )

    def execute_batch(self, tool_calls: list[ToolCall]) -> list[ActionResult]:
        """Execute multiple tool calls sequentially (respecting per-turn limit)."""
        results: list[ActionResult] = []
        for i, tc in enumerate(tool_calls[:self._max_per_turn]):
            result = self.execute(tc)
            results.append(result)
            if result.needs_confirmation:
                logger.info(
                    "Batch paused at step %d/%d: %s needs confirmation",
                    i + 1, len(tool_calls), tc.name,
                )
                break
            if result.blocked:
                logger.info("Batch stopped at step %d: %s blocked", i + 1, tc.name)
                break
        return results

    @staticmethod
    def _validate_params(tool_def, args: dict) -> list[str]:
        """Return list of missing required parameters.

        Note: normalizes parameter aliases (e.g. 'exe' -> 'name') by
        working on a copy of args to avoid mutating the caller's dict.
        The normalized args are written back after validation.
        """
        missing = []
        normalized = dict(args)  # Shallow copy to avoid side-effects
        for param in tool_def.parameters:
            if param.required and param.name not in normalized:
                alt_keys = {"name": ["exe", "app"], "query": ["q", "search"],
                            "percent": ["level", "value", "pct"],
                            "path": ["file", "dir", "folder"],
                            "src": ["source", "from"], "dst": ["dest", "to", "destination"]}
                alts = alt_keys.get(param.name, [])
                found_alt = False
                for alt in alts:
                    if alt in normalized:
                        normalized[param.name] = normalized.pop(alt)
                        found_alt = True
                        break
                if not found_alt:
                    missing.append(param.name)
        # Update the original dict with normalized keys (deliberate, after validation)
        args.clear()
        args.update(normalized)
        return missing

    @staticmethod
    def _confirmation_detail(action: str, args: dict) -> str:
        details = {
            "shutdown_pc": "shutdown",
            "restart_pc": "restart",
            "sleep_pc": "sleep",
            "logoff": "log off",
            "empty_recycle_bin": "empty recycle bin",
            "close_app": args.get("name", "app"),
            "kill_process": args.get("name", "process"),
            "play_youtube": args.get("query", "video"),
            "create_folder": args.get("name", "folder"),
            "move_path": args.get("src", "file"),
            "copy_path": args.get("src", "file"),
        }
        return details.get(action, action.replace("_", " "))

    def get_stats(self) -> dict[str, int]:
        return {
            "total_executions": self._total_executions,
            "total_blocked": self._total_blocked,
            "registered_tools": self._registry.count,
        }
