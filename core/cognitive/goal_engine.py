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
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import quote_plus

from core.persistence_manager import persistence_manager

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.cognitive.second_brain import SecondBrain

logger = logging.getLogger("atom.goal_engine")

_GOALS_FILE = Path("logs/goals.json")
_MAX_GOALS = 20
_MAX_STEPS = 30
_BRIEFING_HOUR_START = 7
_BRIEFING_HOUR_END = 10

_TOOL_BRACKETS = re.compile(
    r"\s*\[tool:([\w_]+)\](?:\s*\[args:\s*(\{[^\]]*\})\])?\s*$",
    re.IGNORECASE,
)
_ALLOWED_SUGGEST_TOOLS = frozenset({
    "open_app",
    "search",
    "spotlight_search",
    "remember",
    "set_reminder",
    "open_url",
    "learn_document",
    "screenshot",
})


def _search_url_from_text(text: str) -> str:
    low = text.lower()
    for phrase in ("search for ", "look up ", "google ", "web search "):
        if phrase in low:
            q = text[low.index(phrase) + len(phrase) :].strip()[:240]
            if q:
                return "https://www.google.com/search?q=" + quote_plus(q)
    return "https://www.google.com/search?q=" + quote_plus(text.strip()[:200])


def infer_suggested_tool(title: str) -> tuple[Optional[str], dict[str, Any]]:
    """Map free-text step title to a registry tool + args (heuristic, best-effort)."""
    raw = (title or "").strip()
    if not raw:
        return None, {}

    m = _TOOL_BRACKETS.search(raw)
    base = _TOOL_BRACKETS.sub("", raw).strip() if m else raw
    low = base.lower()

    if m:
        tname = m.group(1).strip().lower()
        if tname in _ALLOWED_SUGGEST_TOOLS:
            if m.group(2):
                try:
                    parsed = json.loads(m.group(2))
                    if isinstance(parsed, dict):
                        return tname, parsed
                except json.JSONDecodeError:
                    pass
            _, args = infer_suggested_tool(base)
            if tname == "remember" and not (args or {}).get("fact") and base.strip():
                return tname, {"fact": base.strip()[:500]}
            return tname, args or {}

    url_m = re.search(r"https?://[^\s)]+", base)
    if url_m:
        return "open_url", {"url": url_m.group(0)[:2000]}

    path_m = re.search(
        r"(?:~/|/)[\w./+\-]+\.(?:pdf|md|txt|py|json|csv|docx?|xlsx?)\b",
        base,
    )
    if path_m and any(k in low for k in ("read", "ingest", "learn", "document", "file")):
        return "learn_document", {"path": path_m.group(0)[:1024]}

    if any(k in low for k in ("spotlight", "find file", "locate file", "on disk")):
        q = base.split(":", 1)[-1].strip()[:200] or base[:200]
        return "spotlight_search", {"query": q, "limit": 15}

    if any(
        k in low
        for k in (
            "search online",
            "web search",
            "google",
            "look up online",
            "search the web",
        )
    ):
        return "search", {"url": _search_url_from_text(base)}

    if any(k in low for k in ("remind", "reminder", "schedule", "deadline alert")):
        return "set_reminder", {
            "label": base[:120],
            "delay_seconds": 900,
        }

    if any(k in low for k in ("remember", "note to self", "memorize", "don't forget")):
        return "remember", {"fact": base[:500]}

    if any(k in low for k in ("screenshot", "screen shot", "capture screen")):
        return "screenshot", {}

    for kw in ("open ", "launch ", "start "):
        if low.startswith(kw):
            name = base[len(kw) :].strip().split(",")[0].split(" for ")[0][:80]
            if name:
                return "open_app", {"name": name}
            break

    return None, {}


def _finalize_step_record(step: dict[str, Any]) -> None:
    """Normalize title (strip optional [tool:…] brackets) and attach tool hints."""
    raw_title = str(step.get("title", "")).strip()
    if not raw_title:
        return
    display = _TOOL_BRACKETS.sub("", raw_title).strip() or raw_title
    step["title"] = display[:200]
    t, a = infer_suggested_tool(raw_title)
    if t:
        step["suggested_tool"] = t
        step["suggested_args"] = a
        step["tool_link_status"] = "suggested"
    else:
        step.setdefault("suggested_tool", None)
        step.setdefault("suggested_args", {})
        step.setdefault("tool_link_status", "none")


def _args_overlap(suggested: dict[str, Any], executed: dict[str, Any]) -> bool:
    if not suggested:
        return True
    for k, v in suggested.items():
        if k not in executed:
            continue
        vs, es = str(v).strip().lower(), str(executed[k]).strip().lower()
        if vs and (vs in es or es in vs or vs == es):
            return True
    return False


class GoalEngine:
    """Goal-based intelligence -- ATOM tracks what Boss is working towards."""

    __slots__ = (
        "_bus", "_brain", "_config",
        "_goals", "_task", "_shutdown",
        "_eval_interval", "_last_briefing_date",
        "_dirty",
        # Bus-driven re-evaluation: instead of waiting the full
        # _eval_interval (default 1h) for the timer to fire, we hook
        # context_snapshot (emitted ~every 60-120s by HealthMonitor)
        # and re-evaluate when meaningful state has changed. The
        # snapshot interval (default 5min) is the floor on how often
        # snapshots can drive a re-eval -- without it, every snapshot
        # would force an eval and goal_briefing() would spam. The
        # original 1h timer stays as a deadline-style safety net for
        # the case where snapshots stop flowing.
        "_snapshot_eval_interval",
        "_last_snapshot_eval",
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
        self._snapshot_eval_interval: float = float(
            cfg.get("goal_snapshot_eval_interval_s", 300.0),
        )

        self._goals: list[dict] = []
        self._task: asyncio.Task | None = None
        self._shutdown = asyncio.Event()
        self._last_briefing_date: str = ""
        self._dirty = False
        self._last_snapshot_eval: float = 0.0
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
        self._migrate_step_tools()
        if self._dirty:
            self.persist()

    def _migrate_step_tools(self) -> None:
        """Backfill suggested_tool metadata on legacy goal steps."""
        changed = False
        for goal in self._goals:
            for step in goal.get("steps", []):
                if step.get("status") != "pending":
                    continue
                if step.get("suggested_tool"):
                    continue
                raw = str(step.get("title", "")).strip()
                if len(raw) < 4:
                    continue
                t, a = infer_suggested_tool(raw)
                if t:
                    step["suggested_tool"] = t
                    step["suggested_args"] = a
                    step["tool_link_status"] = "suggested"
                    changed = True
                else:
                    step.setdefault("tool_link_status", "none")
                    step.setdefault("suggested_args", {})
                    step.setdefault("suggested_tool", None)
        if changed:
            self._dirty = True

    def persist(self) -> None:
        try:
            payload = {
                "goals": self._goals,
                "last_briefing_date": self._last_briefing_date,
                "saved_at": datetime.now().isoformat(),
            }
            persistence_manager.register("goals", _GOALS_FILE)
            persistence_manager.save_now("goals", payload)
            self._dirty = False
        except Exception:
            logger.exception("Failed to persist goals")

    # ── Lifecycle (CognitiveModuleContract) ────────────────────────────

    def start(self) -> None:
        if not self._config.get("goals_enabled", True):
            logger.info("Goal engine disabled via config")
            return
        self._bus.on("tool_executed", self._on_tool_executed)
        self._bus.on("context_snapshot", self._on_context_snapshot)
        self._task = asyncio.create_task(self._run())
        logger.info(
            "Goal engine started (snapshot_interval=%.0fs, "
            "timer_safety_net=%.0fs, %d goals loaded)",
            self._snapshot_eval_interval, self._eval_interval, len(self._goals),
        )

    def stop(self) -> None:
        self._shutdown.set()
        try:
            self._bus.off("tool_executed", self._on_tool_executed)
        except Exception:
            logger.debug("goal_engine tool_executed off failed", exc_info=True)
        try:
            self._bus.off("context_snapshot", self._on_context_snapshot)
        except Exception:
            logger.debug("goal_engine context_snapshot off failed", exc_info=True)
        if self._task and not self._task.done():
            self._task.cancel()
            self._task = None
        self.persist()

    async def _on_context_snapshot(self, **_kw: Any) -> None:
        """Re-evaluate goals when fresh system context arrives.

        HealthMonitor emits ``context_snapshot`` every 60-120s with
        active_app, idle_minutes, time_of_day, etc. That's a much
        tighter signal than the 1h ``_eval_interval`` timer, so we
        ride it -- but rate-limit at ``_snapshot_eval_interval`` so a
        burst of snapshots can't trigger a flood of evaluations or
        repeated morning briefings.
        """
        now = time.monotonic()
        if now - self._last_snapshot_eval < self._snapshot_eval_interval:
            return
        self._last_snapshot_eval = now
        try:
            self._evaluate_goals()
            self._maybe_briefing()
            if self._dirty:
                self.persist()
        except Exception:
            logger.exception("Goal engine snapshot eval error")

    async def _run(self) -> None:
        # Safety-net timer. With the bus subscription above this is
        # almost always a no-op (the snapshot handler beat us to it),
        # but it's the deadline guarantee for the case where snapshots
        # stop flowing -- e.g. HealthMonitor wedged, fleet offline,
        # or owner runs ATOM headless without context.
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

    async def _on_tool_executed(
        self,
        tool: str = "",
        success: bool = False,
        arguments: dict[str, Any] | None = None,
        **_kw: Any,
    ) -> None:
        """Advance goals when a ReAct tool run matches the next suggested step."""
        if not success or not self._config.get("goal_tool_auto_complete", True):
            return
        try:
            if self.apply_tool_completion(str(tool), dict(arguments or {})):
                logger.info("Goal step auto-completed from tool '%s'", tool)
        except Exception:
            logger.debug("goal tool completion hook failed", exc_info=True)

    def apply_tool_completion(self, tool: str, arguments: dict[str, Any]) -> bool:
        """Complete the first pending step whose suggested tool matches *tool*."""
        strict = bool(self._config.get("goal_tool_match_strict", False))
        for goal in self.get_active_goals():
            for step in goal.get("steps", []):
                if step.get("status") != "pending":
                    continue
                sug = step.get("suggested_tool")
                if not sug or sug != tool:
                    continue
                if strict and not _args_overlap(
                    step.get("suggested_args") or {},
                    arguments,
                ):
                    continue
                step["tool_executed_at"] = datetime.now().isoformat()
                step["tool_link_status"] = "completed"
                self.complete_step(goal["id"], step["id"])
                return True
        return False

    def get_next_actionable_step(self, goal_id: str) -> Optional[dict[str, Any]]:
        goal = self._find_by_id(goal_id)
        if not goal:
            return None
        for step in goal.get("steps", []):
            if step.get("status") == "pending":
                return {
                    "step_id": step["id"],
                    "title": step.get("title", ""),
                    "suggested_tool": step.get("suggested_tool"),
                    "suggested_args": step.get("suggested_args") or {},
                }
        return None

    # ── LLM Decomposition ─────────────────────────────────────────────

    async def decompose_with_llm(self, goal_id: str) -> str:
        """Decompose a goal into steps using the LLM.

        Sends a query to the LLM brain and waits for a real response.
        Falls back to template-based decomposition if the LLM doesn't
        respond within the timeout period.
        """
        goal = self._find_by_id(goal_id)
        if not goal:
            return "Goal not found, Boss."
        if len(goal.get("steps", [])) >= _MAX_STEPS:
            return f"Goal already has {_MAX_STEPS} steps -- that's the max."

        query = (
            f"Break down this goal into 4-8 concrete, actionable steps. "
            f"Goal: {goal['title']}. "
            f"Return each step as a numbered list (1. Step description). "
            f"Optionally suffix a step with [tool:TOOL] where TOOL is one of: "
            f"open_app, search, spotlight_search, remember, set_reminder, open_url, "
            f"learn_document, screenshot — e.g. "
            f"\"3. Open Safari for testing [tool:open_app]\"."
        )

        # Set up a future to capture the LLM response
        loop = asyncio.get_running_loop()
        llm_response_future: asyncio.Future[str] = loop.create_future()

        async def _capture_response(event: str = "", text: str = "", **kw) -> None:
            """Bus callback to capture the LLM's response."""
            if not llm_response_future.done() and text:
                llm_response_future.set_result(text)

        try:
            # Register a one-shot listener for the LLM response
            self._bus.on("llm_response_complete", _capture_response)

            # Send query to the LLM brain
            self._bus.emit_long(
                "cursor_query",
                text=query,
                memory_context="",
                context={"source": "goal_decomposition", "goal_id": goal_id},
                history=[],
            )

            # Wait for LLM response with timeout
            try:
                llm_text = await asyncio.wait_for(llm_response_future, timeout=15.0)
                steps = self._parse_llm_steps(llm_text, goal["title"])
                if steps:
                    logger.info("LLM decomposed goal '%s' into %d steps", goal["title"], len(steps))
                else:
                    logger.info("LLM response didn't contain parseable steps, using defaults")
                    steps = self._generate_default_steps(goal["title"])
            except asyncio.TimeoutError:
                logger.info("LLM decomposition timed out, using default steps")
                steps = self._generate_default_steps(goal["title"])

            n_before = len(goal.get("steps", []))
            for s in steps:
                if len(goal.get("steps", [])) >= _MAX_STEPS:
                    break
                _finalize_step_record(s)
                goal.setdefault("steps", []).append(s)
            self._recalc_progress(goal)
            self._dirty = True
            self.persist()
            hints = []
            for s in goal["steps"][n_before:]:
                if s.get("suggested_tool"):
                    hints.append(f"{s['title']} → {s['suggested_tool']}")
            step_list = "\n".join(f"  {i+1}. {s['title']}" for i, s in enumerate(goal["steps"]))
            extra = ""
            if hints:
                extra = "\nSuggested tools (first runs can auto-check steps):\n  " + "\n  ".join(
                    hints[:8],
                )
            return (
                f"Broke down '{goal['title']}' into {len(goal['steps'])} steps:\n{step_list}"
                f"{extra}"
            )
        except Exception:
            logger.exception("Goal decomposition failed")
            return "Failed to decompose the goal, Boss. I'll try again later."

    @staticmethod
    def _parse_llm_steps(llm_text: str, goal_title: str) -> list[dict]:
        """Parse numbered steps from LLM response text.

        Handles formats like:
            1. Step description
            1) Step description
            - Step description
        """
        lines = llm_text.strip().split("\n")
        step_pattern = re.compile(r'^\s*(?:\d+[.)\-]|[-*•])\s*(.+)$')
        steps = []
        now = datetime.now().isoformat()

        for line in lines:
            match = step_pattern.match(line.strip())
            if match:
                title = match.group(1).strip()
                if len(title) > 5:  # Filter noise
                    steps.append({
                        "id": str(uuid.uuid4())[:8],
                        "title": title[:200],
                        "status": "pending",
                        "minutes_logged": 0,
                        "created_at": now,
                        "updated_at": now,
                        "source": "llm",
                    })

        return steps[:_MAX_STEPS]

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
        out: list[dict[str, Any]] = []
        for t in templates:
            step = {
                "id": str(uuid.uuid4())[:8],
                "title": f"{t} for: {title[:40]}",
                "status": "pending",
                "minutes_logged": 0,
                "created_at": now,
                "updated_at": now,
                "source": "template",
            }
            _finalize_step_record(step)
            out.append(step)
        return out

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
        self._evaluate_goals()
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
            nxt = self.get_next_actionable_step(g["id"])
            if nxt and nxt.get("suggested_tool"):
                lines.append(
                    f"      Next: {nxt['title'][:100]} "
                    f"(suggested tool: {nxt['suggested_tool']})",
                )

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
                line = f"  - {g['title']} [{pct}%, {steps} steps]"
                if status == "active":
                    nxt = self.get_next_actionable_step(g["id"])
                    if nxt and nxt.get("suggested_tool"):
                        line += f" → next tool: {nxt['suggested_tool']}"
                lines.append(line)
        return "\n".join(lines) if lines else "No goals found."

    def get_goals_for_dashboard(self) -> list[dict]:
        rows: list[dict[str, Any]] = []
        for g in self._goals:
            nxt = self.get_next_actionable_step(g["id"]) if g.get("status") == "active" else None
            rows.append(
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
                    "next_step_title": (nxt or {}).get("title"),
                    "next_suggested_tool": (nxt or {}).get("suggested_tool"),
                },
            )
        return rows

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
