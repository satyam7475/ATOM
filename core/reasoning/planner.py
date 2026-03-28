"""
ATOM -- Multi-Step Reasoning Planner.

For complex requests that require multiple steps, the planner:
  1. Analyzes the request to determine if planning is needed
  2. Generates a step-by-step plan using the LLM
  3. Executes steps sequentially with feedback
  4. Re-plans on failure with alternative approaches

This is what separates a command executor from JARVIS-level intelligence.
ATOM can now think through multi-step problems, not just react to commands.

Examples:
  "Set up my workspace" -> Open IDE + Open browser + Set volume + Show goals
  "Do my morning routine" -> Check weather + Show goals + Open email + Play music
  "Research and summarize X" -> Search + Read results + Summarize + Remember
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("atom.planner")


@dataclass
class PlanStep:
    """Single step in a multi-step plan."""
    index: int
    description: str
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    result: str = ""
    error: str = ""

    @property
    def is_complete(self) -> bool:
        return self.status in ("done", "skipped", "failed")


@dataclass
class Plan:
    """Multi-step execution plan."""
    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    status: str = "created"
    created_at: float = 0.0
    completed_at: float = 0.0

    @property
    def progress(self) -> float:
        if not self.steps:
            return 0.0
        done = sum(1 for s in self.steps if s.is_complete)
        return done / len(self.steps)

    @property
    def current_step(self) -> PlanStep | None:
        for s in self.steps:
            if not s.is_complete:
                return s
        return None

    def summary(self) -> str:
        done = sum(1 for s in self.steps if s.status == "done")
        total = len(self.steps)
        return f"Plan: {self.goal} [{done}/{total} steps complete]"


_MULTI_STEP_SIGNALS = [
    "and then", "after that", "also", "next", "followed by",
    "morning routine", "evening routine", "set up", "prepare",
    "research and", "find and", "do my", "start my",
    "workflow", "routine", "sequence", "chain",
]

_PLAN_TEMPLATES: dict[str, list[dict]] = {
    "morning_routine": [
        {"desc": "Check weather", "tool": "weather"},
        {"desc": "Show active goals", "tool": "show_goals"},
        {"desc": "Open email client", "tool": "open_app", "args": {"name": "outlook"}},
        {"desc": "Set comfortable volume", "tool": "set_volume", "args": {"percent": 30}},
    ],
    "workspace_setup": [
        {"desc": "Open code editor", "tool": "open_app", "args": {"name": "vscode"}},
        {"desc": "Open browser", "tool": "open_app", "args": {"name": "chrome"}},
        {"desc": "Open terminal", "tool": "open_app", "args": {"name": "terminal"}},
        {"desc": "Show active goals", "tool": "show_goals"},
    ],
    "focus_mode": [
        {"desc": "Mute notifications", "tool": "mute"},
        {"desc": "Set low volume", "tool": "set_volume", "args": {"percent": 10}},
        {"desc": "Minimize distractions", "tool": "minimize_window"},
    ],
    "end_of_day": [
        {"desc": "Show goal progress", "tool": "show_goals"},
        {"desc": "Save screenshot of work", "tool": "screenshot"},
        {"desc": "Lower brightness", "tool": "set_brightness", "args": {"percent": 30}},
    ],
}


class ReasoningPlanner:
    """Multi-step reasoning and planning engine."""

    def __init__(self, config: dict | None = None) -> None:
        self._config = (config or {}).get("reasoning", {})
        self._active_plan: Plan | None = None
        self._plan_history: list[Plan] = []
        self._max_steps = self._config.get("max_plan_steps", 10)
        self._max_retries = self._config.get("max_retries", 2)
        self._timeline: Any = None
        self._system_monitor: Any = None

    def set_timeline(self, timeline: Any) -> None:
        """Optional TimelineMemory for continuation-aware planning."""
        self._timeline = timeline

    def set_system_monitor(self, monitor: Any) -> None:
        """Optional SystemMonitor for load-aware planning hints."""
        self._system_monitor = monitor

    def timeline_hint(self) -> str:
        parts: list[str] = []
        if self._timeline is not None:
            try:
                parts.append(self._timeline.summary_for_prompt(window_sec=900.0, max_lines=4))
            except Exception:
                pass
        if self._system_monitor is not None:
            try:
                st = self._system_monitor.get_system_state()
                cpu = st.get("cpu_percent", 0)
                ram = st.get("ram_percent", 0)
                fg = (st.get("foreground_window_title") or "")[:60]
                parts.append(f"System: CPU {cpu}% RAM {ram}% foreground: {fg}")
            except Exception:
                pass
        return "\n".join(p for p in parts if p)

    def needs_planning(self, query: str) -> bool:
        """Determine if a query requires multi-step planning."""
        q = query.lower()
        if any(signal in q for signal in _MULTI_STEP_SIGNALS):
            return True
        if q.count(" and ") >= 2:
            return True
        return False

    def create_plan_from_template(self, template_key: str) -> Plan | None:
        """Create a plan from a known template."""
        template = _PLAN_TEMPLATES.get(template_key)
        if template is None:
            return None

        steps = []
        for i, step_def in enumerate(template):
            steps.append(PlanStep(
                index=i,
                description=step_def["desc"],
                tool_name=step_def.get("tool", ""),
                tool_args=step_def.get("args", {}),
            ))

        plan = Plan(
            goal=template_key.replace("_", " ").title(),
            steps=steps,
            created_at=time.time(),
        )
        self._active_plan = plan
        logger.info("Plan created from template '%s': %d steps",
                     template_key, len(steps))
        return plan

    def create_plan_from_steps(self, goal: str,
                               step_descriptions: list[str]) -> Plan:
        """Create a plan from a list of step descriptions."""
        steps = []
        for i, desc in enumerate(step_descriptions[:self._max_steps]):
            steps.append(PlanStep(index=i, description=desc))

        plan = Plan(
            goal=goal,
            steps=steps,
            status="created",
            created_at=time.time(),
        )
        self._active_plan = plan
        return plan

    def detect_template(self, query: str) -> str | None:
        """Detect if the query matches a known plan template."""
        q = query.lower()
        if any(kw in q for kw in ("morning routine", "good morning", "start my day")):
            return "morning_routine"
        if any(kw in q for kw in ("workspace", "set up", "setup", "start working")):
            return "workspace_setup"
        if any(kw in q for kw in ("focus mode", "do not disturb", "concentrate")):
            return "focus_mode"
        if any(kw in q for kw in ("end of day", "wrap up", "call it a day", "done for today")):
            return "end_of_day"
        return None

    def mark_step_done(self, result: str = "") -> PlanStep | None:
        """Mark the current step as complete and return next step."""
        if self._active_plan is None:
            return None
        current = self._active_plan.current_step
        if current is None:
            return None
        current.status = "done"
        current.result = result
        next_step = self._active_plan.current_step
        if next_step is None:
            self._active_plan.status = "completed"
            self._active_plan.completed_at = time.time()
            self._plan_history.append(self._active_plan)
            self._active_plan = None
        return next_step

    def mark_step_failed(self, error: str = "") -> None:
        if self._active_plan is None:
            return
        current = self._active_plan.current_step
        if current is not None:
            current.status = "failed"
            current.error = error

    def skip_step(self) -> PlanStep | None:
        if self._active_plan is None:
            return None
        current = self._active_plan.current_step
        if current is not None:
            current.status = "skipped"
        return self._active_plan.current_step

    @property
    def active_plan(self) -> Plan | None:
        return self._active_plan

    @property
    def has_active_plan(self) -> bool:
        return self._active_plan is not None

    def cancel_plan(self) -> str:
        if self._active_plan is not None:
            self._active_plan.status = "cancelled"
            self._plan_history.append(self._active_plan)
            self._active_plan = None
            return "Plan cancelled, Boss."
        return "No active plan to cancel."

    def get_plan_status(self) -> str:
        if self._active_plan is None:
            return "No active plan."
        plan = self._active_plan
        current = plan.current_step
        return (
            f"{plan.summary()}. "
            f"Current step: {current.description if current else 'All done'}."
        )
