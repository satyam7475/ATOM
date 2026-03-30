"""
ATOM -- Goal Engine: Goal-Based Intelligence for Ring 6 (Cognition).

Lifecycle: Create -> Decompose (LLM) -> Track Steps -> Evaluate -> Briefing

Follows CognitiveModuleContract: start(), stop(), persist()

Max 20 goals, max 30 steps per goal.
Emits: goal_update, goal_briefing
Persists to: logs/goals.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.cognitive.second_brain import SecondBrain

logger = logging.getLogger("atom.goal_engine")

_GOALS_FILE = Path("logs/goals.json")
_MAX_GOALS = 20
_MAX_STEPS = 30
_BRIEFING_HOUR_START = 7
_BRIEFING_HOUR_END = 10


class GoalEngine:
    """Goal-based intelligence -- ATOM tracks what Boss is working towards."""

    __slots__ = (
        "_bus", "_brain", "_config",
        "_goals", "_task", "_shutdown",
        "_eval_interval", "_last_briefing_date",
        "_dirty",
    )

    def __init__(
        self,
        bus: AsyncEventBus,
        second_brain: SecondBrain,
        config: dict | None = None,
    ) -> None:
        self._bus = bus
        self._brain = second_brain
        cfg = (config or {}).get("cognitive", {})
        self._config = cfg
        self._eval_interval: float = cfg.get("goal_evaluation_interval_s", 3600.0)

        self._goals: list[dict] = []
        self._task: asyncio.Task | None = None
        self._shutdown = asyncio.Event()
        self._last_briefing_date: str = ""
        self._dirty = False
        self._load()

    # ── Persistence ────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if _GOALS_FILE.exists():
                data = json.loads(_GOALS_FILE.read_text(encoding="utf-8"))
                self._goals = data.get("goals", [])
                self._last_briefing_date = data.get("last_briefing_date", "")
                logger.info("Goal engine loaded %d goals from disk", len(self._goals))
        except Exception:
            logger.exception("Failed to load goals -- starting fresh")
            self._goals = []

    def persist(self) -> None:
        try:
            _GOALS_FILE.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "goals": self._goals,
                "last_briefing_date": self._last_briefing_date,
                "saved_at": datetime.now().isoformat(),
            }
            _GOALS_FILE.write_text(
                json.dumps(payload, indent=2, default=str),
                encoding="utf-8",
            )
            self._dirty = False
        except Exception:
            logger.exception("Failed to persist goals")

    # ── Lifecycle (CognitiveModuleContract) ────────────────────────────

    def start(self) -> None:
        if not self._config.get("goals_enabled", True):
            logger.info("Goal engine disabled via config")
            return
        self._task = asyncio.create_task(self._run())
        logger.info(
            "Goal engine started (eval_interval=%.0fs, %d goals loaded)",
            self._eval_interval, len(self._goals),
        )

    def stop(self) -> None:
        self._shutdown.set()
        if self._task and not self._task.done():
            self._task.cancel()
            self._task = None
        self.persist()

    async def _run(self) -> None:
        await asyncio.sleep(30.0)
        while not self._shutdown.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(), timeout=self._eval_interval,
                )
                break
            except asyncio.TimeoutError:
                pass
            try:
                self._evaluate_goals()
                self._maybe_briefing()
                if self._dirty:
                    self.persist()
            except Exception:
                logger.exception("Goal engine cycle error")

    # ── Goal CRUD ──────────────────────────────────────────────────────

    def create_goal(self, title: str) -> dict:
        if len(self._goals) >= _MAX_GOALS:
            return {"error": f"You already have {_MAX_GOALS} goals, Boss. Complete or abandon one first."}
        if not title.strip():
            return {"error": "Goal title can't be empty, Boss."}

        goal = {
            "id": str(uuid.uuid4())[:8],
            "title": title.strip(),
            "status": "active",
            "steps": [],
            "progress": 0.0,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "evaluation": {},
            "streak_days": 0,
            "last_progress_date": "",
            "total_minutes": 0,
        }
        self._goals.append(goal)
        self._dirty = True
        self.persist()

        self._bus.emit_fast(
            "goal_update",
            goal_id=goal["id"], action="created", title=title,
        )
        logger.info("Goal created [%s]: %s", goal["id"], title)
        return goal

    def find_goal(self, target: str) -> Optional[dict]:
        target_lower = target.lower()
        for g in self._goals:
            if g["id"] == target or target_lower in g["title"].lower():
                return g
        return None

    def get_active_goals(self) -> list[dict]:
        return [g for g in self._goals if g["status"] == "active"]

    @property
    def active_count(self) -> int:
        return sum(1 for g in self._goals if g["status"] == "active")

    def pause_goal(self, goal_id: str) -> str:
        goal = self._find_by_id(goal_id)
        if not goal:
            return "Goal not found, Boss."
        goal["status"] = "paused"
        goal["updated_at"] = datetime.now().isoformat()
        self._dirty = True
        self.persist()
        self._bus.emit_fast("goal_update", goal_id=goal_id, action="paused", title=goal["title"])
        return f"Goal '{goal['title']}' paused."

    def resume_goal(self, goal_id: str) -> str:
        goal = self._find_by_id(goal_id)
        if not goal:
            return "Goal not found, Boss."
        goal["status"] = "active"
        goal["updated_at"] = datetime.now().isoformat()
        self._dirty = True
        self.persist()
        self._bus.emit_fast("goal_update", goal_id=goal_id, action="resumed", title=goal["title"])
        return f"Goal '{goal['title']}' resumed."

    def abandon_goal(self, goal_id: str) -> str:
        goal = self._find_by_id(goal_id)
        if not goal:
            return "Goal not found, Boss."
        goal["status"] = "abandoned"
        goal["updated_at"] = datetime.now().isoformat()
        self._dirty = True
        self.persist()
        self._bus.emit_fast("goal_update", goal_id=goal_id, action="abandoned", title=goal["title"])
        return f"Goal '{goal['title']}' abandoned."

    # ── Steps ──────────────────────────────────────────────────────────

    def log_progress(self, goal_id: str, step_id: str, minutes: int = 30) -> str:
        goal = self._find_by_id(goal_id)
        if not goal:
            return "Goal not found, Boss."
        for step in goal.get("steps", []):
            if step["id"] == step_id:
                step["minutes_logged"] = step.get("minutes_logged", 0) + minutes
                step["updated_at"] = datetime.now().isoformat()
                break

        goal["total_minutes"] = goal.get("total_minutes", 0) + minutes
        today = datetime.now().strftime("%Y-%m-%d")
        if goal.get("last_progress_date") == today:
            pass
        elif goal.get("last_progress_date") == _yesterday_str():
            goal["streak_days"] = goal.get("streak_days", 0) + 1
        else:
            goal["streak_days"] = 1
        goal["last_progress_date"] = today
        goal["updated_at"] = datetime.now().isoformat()

        self._recalc_progress(goal)
        self._dirty = True
        self.persist()
        return f"Logged {minutes} minutes on '{goal['title']}'. Total: {goal['total_minutes']} min, streak: {goal['streak_days']} days."

    def complete_step(self, goal_id: str, step_id: str) -> str:
        goal = self._find_by_id(goal_id)
        if not goal:
            return "Goal not found, Boss."
        for step in goal.get("steps", []):
            if step["id"] == step_id:
                step["status"] = "completed"
                step["updated_at"] = datetime.now().isoformat()
                self._recalc_progress(goal)
                self._dirty = True
                self.persist()
                self._bus.emit_fast(
                    "goal_update",
                    goal_id=goal_id, action="step_completed", title=step["title"],
                )
                pct = int(goal["progress"] * 100)
                if goal["progress"] >= 1.0:
                    goal["status"] = "completed"
                    return f"Step completed! Goal '{goal['title']}' is now 100% done. Congratulations, Boss!"
                return f"Step '{step['title']}' done. Goal '{goal['title']}' is at {pct}%."
        return "Step not found, Boss."

    # ── LLM Decomposition ─────────────────────────────────────────────

    async def decompose_with_llm(self, goal_id: str) -> str:
        goal = self._find_by_id(goal_id)
        if not goal:
            return "Goal not found, Boss."
        if len(goal.get("steps", [])) >= _MAX_STEPS:
            return f"Goal already has {_MAX_STEPS} steps -- that's the max."

        query = f"goal_decompose:{goal['title']}"
        try:
            self._bus.emit_long(
                "cursor_query",
                text=query,
                memory_context="",
                context={},
                history=[],
            )
            steps = self._generate_default_steps(goal["title"])
            for s in steps:
                if len(goal.get("steps", [])) >= _MAX_STEPS:
                    break
                goal.setdefault("steps", []).append(s)
            self._recalc_progress(goal)
            self._dirty = True
            self.persist()
            step_list = "\n".join(f"  {i+1}. {s['title']}" for i, s in enumerate(goal["steps"]))
            return f"Broke down '{goal['title']}' into {len(goal['steps'])} steps:\n{step_list}"
        except Exception:
            logger.exception("Goal decomposition failed")
            return "Failed to decompose the goal, Boss. I'll try again later."

    @staticmethod
    def _generate_default_steps(title: str) -> list[dict]:
        templates = [
            "Research and understand the fundamentals",
            "Set up the environment and tools",
            "Create an initial plan or outline",
            "Work through the core tasks",
            "Review progress and iterate",
            "Final review and completion",
        ]
        now = datetime.now().isoformat()
        return [
            {
                "id": str(uuid.uuid4())[:8],
                "title": f"{t} for: {title[:40]}",
                "status": "pending",
                "minutes_logged": 0,
                "created_at": now,
                "updated_at": now,
            }
            for t in templates
        ]

    # ── Evaluation ─────────────────────────────────────────────────────

    def _evaluate_goals(self) -> None:
        for goal in self.get_active_goals():
            total_steps = len(goal.get("steps", []))
            if total_steps == 0:
                goal["evaluation"] = {"trajectory": "new", "note": "No steps yet"}
                continue

            completed = sum(1 for s in goal["steps"] if s.get("status") == "completed")
            ratio = completed / total_steps

            streak = goal.get("streak_days", 0)
            if ratio >= 0.8:
                trajectory = "ahead"
            elif ratio >= 0.4 or streak >= 3:
                trajectory = "on_track"
            elif goal.get("total_minutes", 0) > 0:
                trajectory = "behind"
            else:
                trajectory = "stalled"

            goal["evaluation"] = {
                "trajectory": trajectory,
                "completed_steps": completed,
                "total_steps": total_steps,
                "progress_pct": int(ratio * 100),
                "streak": streak,
            }

    # ── Briefing ───────────────────────────────────────────────────────

    def _maybe_briefing(self) -> None:
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        if self._last_briefing_date == today:
            return
        if not (_BRIEFING_HOUR_START <= now.hour < _BRIEFING_HOUR_END):
            return

        text = self.get_daily_briefing()
        if text:
            self._last_briefing_date = today
            self._dirty = True
            self._bus.emit_long("goal_briefing", text=text)

    def get_daily_briefing(self) -> Optional[str]:
        active = self.get_active_goals()
        if not active:
            return None

        lines = [f"Good morning, Boss. You have {len(active)} active goal{'s' if len(active) != 1 else ''}:"]
        for g in active:
            ev = g.get("evaluation", {})
            trajectory = ev.get("trajectory", "new")
            pct = int(g.get("progress", 0) * 100)
            streak = g.get("streak_days", 0)
            streak_str = f", {streak}-day streak" if streak > 1 else ""
            lines.append(f"  - {g['title']}: {pct}% ({trajectory}{streak_str})")

        return "\n".join(lines)

    def format_goals_summary(self) -> str:
        if not self._goals:
            return "No goals set yet, Boss. Tell me what you want to achieve."

        sections = {"active": [], "paused": [], "completed": [], "abandoned": []}
        for g in self._goals:
            sections.setdefault(g.get("status", "active"), []).append(g)

        lines = []
        for status, goals in sections.items():
            if not goals:
                continue
            lines.append(f"\n{status.upper()} ({len(goals)}):")
            for g in goals:
                pct = int(g.get("progress", 0) * 100)
                steps = len(g.get("steps", []))
                lines.append(f"  - {g['title']} [{pct}%, {steps} steps]")
        return "\n".join(lines) if lines else "No goals found."

    def get_goals_for_dashboard(self) -> list[dict]:
        return [
            {
                "id": g["id"],
                "title": g["title"],
                "status": g["status"],
                "progress": g.get("progress", 0),
                "steps": len(g.get("steps", [])),
                "completed_steps": sum(
                    1 for s in g.get("steps", []) if s.get("status") == "completed"
                ),
                "streak": g.get("streak_days", 0),
                "trajectory": g.get("evaluation", {}).get("trajectory", "new"),
            }
            for g in self._goals
        ]

    # ── Helpers ────────────────────────────────────────────────────────

    def _find_by_id(self, goal_id: str) -> Optional[dict]:
        for g in self._goals:
            if g["id"] == goal_id:
                return g
        return None

    def _recalc_progress(self, goal: dict) -> None:
        steps = goal.get("steps", [])
        if not steps:
            goal["progress"] = 0.0
            return
        completed = sum(1 for s in steps if s.get("status") == "completed")
        goal["progress"] = completed / len(steps)
        goal["updated_at"] = datetime.now().isoformat()


def _yesterday_str() -> str:
    from datetime import timedelta
    return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
