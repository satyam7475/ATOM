"""
ATOM v15 -- Goal Engine: Goal-Based Intelligence + Task Graph.

Transforms ATOM from command-based to goal-based:
  User -> Goal -> Plan -> Execute -> Evaluate -> Improve

Features:
  - Create goals with deadlines and steps
  - LLM-assisted goal decomposition into step plans
  - Track progress per step with logged minutes
  - Evaluate trajectory (on_track / behind / ahead / stalled)
  - Daily briefings via TTS
  - Streak tracking and motivational suggestions

Persistence: logs/goals.json (max 20 goals).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.cognitive.second_brain import SecondBrain

logger = logging.getLogger("atom.goals")

_GOALS_FILE = Path("logs/goals.json")
_MAX_GOALS_DEFAULT = 20
_MAX_STEPS_PER_GOAL = 30


class GoalEngine:
    """Goal-based intelligence engine for ATOM OS."""

    __slots__ = (
        "_bus", "_brain", "_config", "_goals", "_max_goals",
        "_dirty", "_task", "_shutdown",
        "_eval_interval", "_last_briefing_date",
    )

    def __init__(
        self,
        bus: AsyncEventBus,
        brain: SecondBrain,
        config: dict | None = None,
    ) -> None:
        self._bus = bus
        self._brain = brain
        cfg = (config or {}).get("cognitive", {})
        self._config = cfg
        self._eval_interval: float = cfg.get("goal_evaluation_interval_s", 3600.0)
        self._max_goals: int = int(cfg.get("max_goals", _MAX_GOALS_DEFAULT))
        self._goals: dict[str, dict] = {}
        self._dirty = False
        self._task: asyncio.Task | None = None
        self._shutdown = asyncio.Event()
        self._last_briefing_date: str = ""
        self._load()

    # ── Persistence ────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if _GOALS_FILE.exists():
                data = json.loads(_GOALS_FILE.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._goals = data
                elif isinstance(data, list):
                    for g in data:
                        if "id" in g:
                            self._goals[g["id"]] = g
                logger.info("Goals loaded: %d", len(self._goals))
        except Exception:
            logger.debug("No goals file, starting fresh")

    def persist(self) -> None:
        if not self._dirty:
            return
        try:
            _GOALS_FILE.parent.mkdir(parents=True, exist_ok=True)
            _GOALS_FILE.write_text(
                json.dumps(self._goals, indent=2, default=str), encoding="utf-8",
            )
            self._dirty = False
            logger.debug("Goals persisted (%d)", len(self._goals))
        except Exception:
            logger.debug("Failed to persist goals", exc_info=True)

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        if not self._config.get("goals_enabled", True):
            logger.info("Goal engine disabled via config")
            return
        self._task = asyncio.create_task(self._run())
        logger.info("Goal engine started (eval interval=%.0fs)", self._eval_interval)

    def stop(self) -> None:
        self._shutdown.set()
        if self._task and not self._task.done():
            self._task.cancel()
            self._task = None
        self.persist()

    async def _run(self) -> None:
        await asyncio.sleep(60.0)
        while not self._shutdown.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(), timeout=self._eval_interval,
                )
                break
            except asyncio.TimeoutError:
                pass
            try:
                self._evaluate_all_goals()
                self._check_daily_briefing()
            except Exception:
                logger.exception("Goal evaluation error")

    # ── Goal CRUD ──────────────────────────────────────────────────────

    def create_goal(self, title: str, deadline_days: int | None = None) -> dict:
        """Create a new goal. Returns the goal dict."""
        if len(self._goals) >= self._max_goals:
            oldest = min(self._goals.values(), key=lambda g: g.get("created_ts", 0))
            if oldest.get("status") in ("completed", "abandoned"):
                del self._goals[oldest["id"]]
            else:
                return {"error": "Maximum goals reached. Complete or abandon a goal first."}

        goal_id = f"goal_{uuid.uuid4().hex[:8]}"
        now = time.time()
        goal = {
            "id": goal_id,
            "title": title[:200],
            "created_ts": now,
            "deadline_ts": now + (deadline_days * 86400) if deadline_days else None,
            "status": "active",
            "progress_pct": 0,
            "steps": [],
            "evaluation": {
                "trajectory": "just_started",
                "daily_target_minutes": 30,
                "streak_days": 0,
                "last_evaluated": now,
                "last_worked_date": "",
                "suggestions": [],
            },
        }
        self._goals[goal_id] = goal
        self._dirty = True
        self.persist()

        self._bus.emit_fast("goal_update", goal_id=goal_id, action="created", title=title)
        logger.info("Goal created: %s - %s", goal_id, title)
        return goal

    def add_step(
        self, goal_id: str, title: str, depends_on: list[str] | None = None,
        estimated_minutes: int = 60,
    ) -> dict | None:
        """Add a step to a goal."""
        goal = self._goals.get(goal_id)
        if not goal:
            return None
        if len(goal["steps"]) >= _MAX_STEPS_PER_GOAL:
            return None

        step_id = f"step_{uuid.uuid4().hex[:6]}"
        step = {
            "id": step_id,
            "title": title[:200],
            "depends_on": depends_on or [],
            "status": "pending",
            "estimated_minutes": max(1, estimated_minutes),
            "logged_minutes": 0,
            "success_score": 0.0,
            "last_worked": None,
            "notes": "",
        }
        goal["steps"].append(step)
        self._update_progress(goal)
        self._dirty = True
        return step

    def log_progress(self, goal_id: str, step_id: str, minutes: int) -> str:
        """Log time spent on a goal step."""
        goal = self._goals.get(goal_id)
        if not goal:
            return "Goal not found."

        for step in goal["steps"]:
            if step["id"] == step_id:
                step["logged_minutes"] += max(1, minutes)
                step["last_worked"] = time.time()
                if step["status"] == "pending":
                    step["status"] = "in_progress"
                est = step.get("estimated_minutes", 60)
                step["success_score"] = min(1.0, step["logged_minutes"] / max(1, est))
                self._update_progress(goal)
                self._update_streak(goal)
                self._dirty = True
                self.persist()
                return f"Logged {minutes} minutes on '{step['title']}'. Progress: {goal['progress_pct']}%."

        return "Step not found."

    def complete_step(self, goal_id: str, step_id: str) -> str:
        goal = self._goals.get(goal_id)
        if not goal:
            return "Goal not found."
        for step in goal["steps"]:
            if step["id"] == step_id:
                step["status"] = "completed"
                step["success_score"] = 1.0
                self._update_progress(goal)
                self._dirty = True
                self.persist()
                if goal["progress_pct"] >= 100:
                    goal["status"] = "completed"
                    return f"Step completed! Goal '{goal['title']}' is now COMPLETE!"
                return f"Step '{step['title']}' completed. Goal progress: {goal['progress_pct']}%."
        return "Step not found."

    def pause_goal(self, goal_id: str) -> str:
        goal = self._goals.get(goal_id)
        if not goal:
            return "Goal not found."
        goal["status"] = "paused"
        self._dirty = True
        return f"Goal '{goal['title']}' paused."

    def resume_goal(self, goal_id: str) -> str:
        goal = self._goals.get(goal_id)
        if not goal:
            return "Goal not found."
        goal["status"] = "active"
        self._dirty = True
        return f"Goal '{goal['title']}' resumed."

    def abandon_goal(self, goal_id: str) -> str:
        goal = self._goals.get(goal_id)
        if not goal:
            return "Goal not found."
        goal["status"] = "abandoned"
        self._dirty = True
        return f"Goal '{goal['title']}' abandoned."

    # ── Evaluation ─────────────────────────────────────────────────────

    def _update_progress(self, goal: dict) -> None:
        steps = goal.get("steps", [])
        if not steps:
            goal["progress_pct"] = 0
            return
        completed = sum(1 for s in steps if s["status"] == "completed")
        goal["progress_pct"] = int((completed / len(steps)) * 100)

    def _update_streak(self, goal: dict) -> None:
        today = datetime.now().strftime("%Y-%m-%d")
        ev = goal.get("evaluation", {})
        last_date = ev.get("last_worked_date", "")
        if last_date == today:
            return
        yesterday = datetime.now()
        from datetime import timedelta
        yesterday_str = (yesterday - timedelta(days=1)).strftime("%Y-%m-%d")
        if last_date == yesterday_str:
            ev["streak_days"] = ev.get("streak_days", 0) + 1
        elif last_date != today:
            ev["streak_days"] = 1
        ev["last_worked_date"] = today

    def _evaluate_all_goals(self) -> None:
        for goal in self._goals.values():
            if goal["status"] != "active":
                continue
            self._evaluate_goal(goal)
        if self._dirty:
            self.persist()

    def _evaluate_goal(self, goal: dict) -> None:
        ev = goal.setdefault("evaluation", {})
        steps = goal.get("steps", [])
        now = time.time()

        if not steps:
            ev["trajectory"] = "no_steps"
            ev["suggestions"] = ["Add steps to this goal to start tracking progress."]
            ev["last_evaluated"] = now
            self._dirty = True
            return

        completed = sum(1 for s in steps if s["status"] == "completed")
        total = len(steps)
        progress = completed / total if total else 0

        deadline_ts = goal.get("deadline_ts")
        if deadline_ts:
            elapsed_frac = (now - goal["created_ts"]) / max(1, deadline_ts - goal["created_ts"])
            elapsed_frac = min(1.0, max(0.0, elapsed_frac))
            if progress >= elapsed_frac + 0.1:
                ev["trajectory"] = "ahead"
            elif progress >= elapsed_frac - 0.15:
                ev["trajectory"] = "on_track"
            elif progress < elapsed_frac - 0.3:
                ev["trajectory"] = "behind"
            else:
                ev["trajectory"] = "on_track"
        else:
            last_worked = max(
                (s.get("last_worked") or 0 for s in steps), default=0,
            )
            if last_worked and (now - last_worked) > 7 * 86400:
                ev["trajectory"] = "stalled"
            elif progress > 0.5:
                ev["trajectory"] = "on_track"
            else:
                ev["trajectory"] = "in_progress"

        suggestions: list[str] = []
        if ev["trajectory"] == "behind":
            suggestions.append(f"You're behind on '{goal['title']}'. Try increasing daily time.")
        if ev["trajectory"] == "stalled":
            suggestions.append(f"'{goal['title']}' hasn't been worked on in over a week.")
        if ev.get("streak_days", 0) >= 7:
            suggestions.append(f"Great streak on '{goal['title']}': {ev['streak_days']} days!")

        ev["suggestions"] = suggestions
        ev["last_evaluated"] = now
        self._dirty = True

    # ── Daily Briefing ─────────────────────────────────────────────────

    def _check_daily_briefing(self) -> None:
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        if self._last_briefing_date == today:
            return
        if now.hour < 7 or now.hour > 10:
            return

        briefing = self.get_daily_briefing()
        if briefing:
            self._last_briefing_date = today
            self._bus.emit_fast("goal_briefing", text=briefing)

    def get_daily_briefing(self) -> str:
        """Morning summary of active goals."""
        active = [g for g in self._goals.values() if g["status"] == "active"]
        if not active:
            return ""

        parts = [f"Good morning, Boss. You have {len(active)} active goal{'s' if len(active) > 1 else ''}."]

        for goal in active[:3]:
            ev = goal.get("evaluation", {})
            trajectory = ev.get("trajectory", "unknown")
            streak = ev.get("streak_days", 0)
            parts.append(
                f"'{goal['title']}': {goal['progress_pct']}% done, {trajectory}."
            )
            if streak >= 3:
                parts.append(f"  {streak}-day streak!")
            for s in ev.get("suggestions", [])[:1]:
                parts.append(f"  {s}")

        return " ".join(parts)

    # ── LLM Goal Decomposition ─────────────────────────────────────────

    async def decompose_with_llm(self, goal_id: str) -> str:
        """Request LLM to break a goal into steps.

        Uses a dedicated event pair to avoid conflicts with the
        main cursor_query/cursor_response flow.
        """
        goal = self._goals.get(goal_id)
        if not goal:
            return "Goal not found."

        prompt = (
            f"Break this goal into 5-8 concrete, actionable steps. "
            f"Goal: {goal['title']}. "
            f"For each step, give: step title and estimated hours. "
            f"Format: 1. [title] (Xh)\n"
            f"Be practical and specific."
        )

        result_future: asyncio.Future[str] = asyncio.get_running_loop().create_future()

        async def _on_response(query: str = "", response: str = "", **_kw: Any) -> None:
            if "goal_decompose:" in query and not result_future.done():
                result_future.set_result(response)

        self._bus.on("cursor_response", _on_response)

        try:
            self._bus.emit_long(
                "cursor_query",
                text=f"goal_decompose: {prompt}",
                memory_context=[],
                context={},
                history=[],
            )

            try:
                response = await asyncio.wait_for(result_future, timeout=30.0)
            except asyncio.TimeoutError:
                return "LLM timed out. Try adding steps manually."

            steps_added = self._parse_llm_steps(goal_id, response)
            self.persist()
            return f"Added {steps_added} steps to '{goal['title']}'."
        finally:
            self._bus.off("cursor_response", _on_response)

    def _parse_llm_steps(self, goal_id: str, response: str) -> int:
        """Parse LLM response and add steps to goal."""
        import re
        pattern = re.compile(r"^\d+\.\s*(.+?)(?:\((\d+)h?\))?$", re.MULTILINE)
        count = 0
        for match in pattern.finditer(response):
            title = match.group(1).strip().rstrip("(").strip()
            hours = int(match.group(2)) if match.group(2) else 2
            if title and len(title) > 3:
                self.add_step(goal_id, title, estimated_minutes=hours * 60)
                count += 1
                if count >= 10:
                    break
        return count

    # ── Queries ────────────────────────────────────────────────────────

    def get_active_goals(self) -> list[dict]:
        return [g for g in self._goals.values() if g["status"] == "active"]

    def get_all_goals(self) -> list[dict]:
        return list(self._goals.values())

    def find_goal(self, query: str) -> dict | None:
        """Find a goal by partial title match."""
        q = query.lower()
        for goal in self._goals.values():
            if q in goal["title"].lower():
                return goal
        return None

    def format_goals_summary(self) -> str:
        active = self.get_active_goals()
        if not active:
            return "No active goals right now, Boss. Say 'set a goal' to create one."
        parts = [f"You have {len(active)} active goal{'s' if len(active) > 1 else ''}:"]
        for i, g in enumerate(active[:5], 1):
            ev = g.get("evaluation", {})
            parts.append(
                f"  {i}. {g['title']} - {g['progress_pct']}% "
                f"({ev.get('trajectory', 'unknown')})"
            )
        return "\n".join(parts)

    def get_goals_for_dashboard(self) -> list[dict]:
        """Structured goal data for WebSocket broadcast."""
        return [
            {
                "id": g["id"],
                "title": g["title"],
                "status": g["status"],
                "progress_pct": g["progress_pct"],
                "steps_total": len(g.get("steps", [])),
                "steps_done": sum(1 for s in g.get("steps", []) if s["status"] == "completed"),
                "trajectory": g.get("evaluation", {}).get("trajectory", "unknown"),
                "streak": g.get("evaluation", {}).get("streak_days", 0),
                "deadline_ts": g.get("deadline_ts"),
            }
            for g in self._goals.values()
            if g["status"] in ("active", "paused")
        ]

    @property
    def goal_count(self) -> int:
        return len(self._goals)

    @property
    def active_count(self) -> int:
        return sum(1 for g in self._goals.values() if g["status"] == "active")
