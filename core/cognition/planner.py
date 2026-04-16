"""
ATOM OS -- Planner Engine
Takes a high-level reasoning intent and decomposes it into a sequence of executable tool chains
to prevent reactive single-shot failures.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

logger = logging.getLogger("atom.planner")

# Context marker for cursor_query → LocalBrainController; must match wiring queue bypass.
ATOM_PLANNER_CURSOR_SOURCE = "atom_planner"

_LLM_PLAN_TIMEOUT_S = 10.0
_TOOL_HINT_RE = re.compile(r"\s*\[tool:([\w_]+)\]\s*$", re.IGNORECASE)
_STEP_LINE_RE = re.compile(r"^\s*(?:\d+[.)\-]|[-*•])\s*(.+)$")

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus

@dataclass
class PlanStep:
    step_num: int
    description: str
    target_tool: str
    expected_args: Dict[str, Any]
    
@dataclass
class ExecutionPlan:
    goal: str
    steps: List[PlanStep]

class PlannerEngine:
    """Multi-Step Action Planner Generator."""
    
    def __init__(self, ai_client=None, bus: Optional["AsyncEventBus"] = None):
        self.ai = ai_client
        self._bus = bus

    def _heuristic_plan(self, query: str) -> List[PlanStep]:
        q_lower = query.lower()
        if "clean" in q_lower and ("cache" in q_lower or "temp" in q_lower):
            return [
                PlanStep(1, "Scan for temp files", "find_large_files", {"min_size_mb": 10}),
                PlanStep(2, "Analyze safety", "system_analyze", {}),
                PlanStep(3, "Ask boss for confirmation before wiping", "ask_user_confirmation", {}),
                PlanStep(4, "Wipe files safely", "run_terminal_command", {"command": "rm -rf ~/.cache/*"}),
            ]
        if "close" in q_lower and "apps" in q_lower:
            return [
                PlanStep(1, "Get running apps", "get_running_apps", {}),
                PlanStep(2, "Close apps sequentially", "close_app", {"name": "target"}),
            ]
        return [PlanStep(1, "Execute inferred action", "router_dispatch", {"query": query})]

    @staticmethod
    def _parse_llm_plan_steps(llm_text: str, max_steps: int = 32) -> List[PlanStep]:
        """Parse numbered lines and optional [tool:name] suffixes into PlanStep rows."""
        out: List[PlanStep] = []
        for raw_line in llm_text.strip().split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            m = _STEP_LINE_RE.match(line)
            if not m:
                continue
            body = m.group(1).strip()
            tool_name = "router_dispatch"
            tm = _TOOL_HINT_RE.search(body)
            if tm:
                tool_name = tm.group(1).strip().lower() or "router_dispatch"
                body = _TOOL_HINT_RE.sub("", body).strip()
            if len(body) < 4:
                continue
            out.append(
                PlanStep(
                    step_num=len(out) + 1,
                    description=body[:500],
                    target_tool=tool_name,
                    expected_args={},
                )
            )
            if len(out) >= max_steps:
                break
        return out

    async def _try_llm_plan_via_bus(self, query: str, context: str) -> Optional[ExecutionPlan]:
        bus = self._bus
        if bus is None:
            return None

        task_block = query.strip()
        prompt = (
            f"Create a step-by-step plan to accomplish: {task_block}. "
            "Return each step as a numbered line with an optional [tool:name] hint."
        )
        if context and context.strip():
            prompt = f"{prompt}\n\nContext:\n{context.strip()}"

        loop = asyncio.get_running_loop()
        llm_future: asyncio.Future[str] = loop.create_future()

        async def _capture_response(event: str = "", text: str = "", **_kw: Any) -> None:
            if not llm_future.done() and text:
                llm_future.set_result(text)

        bus.on("llm_response_complete", _capture_response)
        try:
            mem: list[str] | None = [context] if (context and context.strip()) else None
            bus.emit_long(
                "cursor_query",
                text=prompt,
                memory_context=mem,
                context={
                    "source": ATOM_PLANNER_CURSOR_SOURCE,
                    "kind": "execution_plan",
                },
                history=[],
            )
            try:
                llm_text = await asyncio.wait_for(llm_future, timeout=_LLM_PLAN_TIMEOUT_S)
            except asyncio.TimeoutError:
                logger.info("LLM planning timed out after %.0fs; using heuristics", _LLM_PLAN_TIMEOUT_S)
                return None
            steps = self._parse_llm_plan_steps(llm_text)
            if not steps:
                logger.info("LLM plan response had no parseable steps; using heuristics")
                return None
            return ExecutionPlan(goal=query, steps=steps)
        except Exception:
            logger.exception("LLM planning via bus failed; using heuristics")
            return None
        finally:
            bus.off("llm_response_complete", _capture_response)
            if not llm_future.done():
                llm_future.cancel()
        
    async def generate_plan(self, query: str, context: str) -> ExecutionPlan:
        """
        Prefer an LLM-produced plan via the event bus (LocalBrainController);
        fall back to fast heuristic templates on timeout, parse failure, or no bus.
        """
        logger.info("Generating execution plan for: %s", query[:200])

        fallback = ExecutionPlan(goal=query, steps=self._heuristic_plan(query))

        if self._bus is None:
            return fallback

        llm_plan = await self._try_llm_plan_via_bus(query, context)
        if llm_plan is not None:
            return llm_plan
        return fallback
