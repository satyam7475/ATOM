"""
ATOM -- Autonomy Engine (AI OS Decision Layer).

The central brain for autonomous decision-making. Runs as a periodic
background task that:
  - Checks BehaviorTracker for high-confidence habits
  - Auto-executes trusted habits (confidence >= threshold) via SecurityPolicy
  - Suggests lower-confidence habits to the user
  - Applies rule-based decisions (CPU load, idle detection)
  - Handles user feedback to adjust habit confidence
  - Logs all autonomous decisions for auditability

Safety: destructive actions are NEVER auto-executed regardless of
confidence. All actions pass through SecurityPolicy.allow_action().
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.behavior_tracker import BehaviorTracker
    from core.health_monitor import HealthMonitor
    from core.priority_scheduler import PriorityScheduler
    from core.security_policy import SecurityPolicy

logger = logging.getLogger("atom.autonomy")

_AUTONOMY_LOG = Path("logs/autonomy.log")

_NEVER_AUTO_EXECUTE: frozenset[str] = frozenset({
    "shutdown_pc", "restart_pc", "logoff", "sleep_pc",
    "close_app", "kill_process", "empty_recycle_bin",
    "create_folder", "move_path", "copy_path",
    "type_text", "hotkey_combo", "press_key",
})


class AutonomyEngine:
    """Autonomous decision engine for ATOM OS.

    Periodically evaluates context and habits, emitting suggestions
    or auto-executing trusted actions through the event bus.
    """

    __slots__ = (
        "_bus", "_behavior", "_security", "_health",
        "_config", "_enabled", "_auto_threshold", "_suggest_threshold",
        "_check_interval", "_idle_timeout_min", "_log_all",
        "_task", "_shutdown_event", "_throttled",
        "_last_suggested", "_last_auto_executed",
        "_last_context", "_pending_suggestion",
        "_priority_sched",
    )

    def __init__(
        self,
        bus: AsyncEventBus,
        behavior: BehaviorTracker,
        security: SecurityPolicy,
        health: HealthMonitor,
        config: dict | None = None,
        priority_sched: PriorityScheduler | None = None,
    ) -> None:
        self._bus = bus
        self._behavior = behavior
        self._security = security
        self._health = health

        auto_cfg = (config or {}).get("autonomy", {})
        self._enabled: bool = auto_cfg.get("enabled", True)
        self._auto_threshold: float = auto_cfg.get("auto_execute_threshold", 0.95)
        self._suggest_threshold: float = auto_cfg.get("suggest_threshold", 0.5)
        self._check_interval: float = auto_cfg.get("check_interval_s", 90.0)
        self._idle_timeout_min: float = auto_cfg.get("idle_timeout_minutes", 10.0)
        self._log_all: bool = auto_cfg.get("log_all_decisions", True)

        self._task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()
        self._throttled: bool = False

        self._last_suggested: dict[str, float] = {}
        self._last_auto_executed: dict[str, float] = {}
        self._last_context: dict[str, Any] = {}
        self._pending_suggestion: dict | None = None
        self._priority_sched = priority_sched

        _AUTONOMY_LOG.parent.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        if not self._enabled:
            logger.info("Autonomy engine disabled via config")
            return
        self._bus.on("governor_throttle", self._on_throttle)
        self._bus.on("governor_normal", self._on_normal)
        self._bus.on("user_feedback", self._on_user_feedback)
        self._bus.on("context_snapshot", self._on_context_snapshot)
        self._task = asyncio.create_task(self._run())
        logger.info(
            "Autonomy engine started (interval=%.0fs, auto>=%.2f, suggest>=%.2f)",
            self._check_interval, self._auto_threshold, self._suggest_threshold,
        )

    def stop(self) -> None:
        self._shutdown_event.set()
        if self._task is not None:
            self._task.cancel()
            self._task = None

    # ── Event handlers ────────────────────────────────────────────────

    async def _on_throttle(self, **_kw: Any) -> None:
        self._throttled = True

    async def _on_normal(self, **_kw: Any) -> None:
        self._throttled = False

    async def _on_context_snapshot(self, **kw: Any) -> None:
        self._last_context = dict(kw)

    async def _on_user_feedback(
        self, habit_id: str = "", accepted: bool = False, **_kw: Any,
    ) -> None:
        if not habit_id:
            return
        if accepted:
            self._behavior.adjust_confidence(habit_id, +0.1)
            self._log_decision("feedback_accept", habit_id, "confidence +0.1")
        else:
            self._behavior.adjust_confidence(habit_id, -0.15)
            self._log_decision("feedback_reject", habit_id, "confidence -0.15")
        self._pending_suggestion = None

    # ── Main decision loop ────────────────────────────────────────────

    async def _run(self) -> None:
        await asyncio.sleep(30.0)

        while not self._shutdown_event.is_set():
            interval = self._effective_interval()
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(), timeout=interval,
                )
                break
            except asyncio.TimeoutError:
                pass

            try:
                await self._decision_cycle()
            except Exception:
                logger.exception("Autonomy decision cycle error")

    def _effective_interval(self) -> float:
        cpu = self._last_context.get("cpu", 30)
        if cpu < 40:
            return max(30.0, self._check_interval * 0.5)
        if cpu > 80 or self._throttled:
            return self._check_interval * 1.5
        return self._check_interval

    async def _decision_cycle(self) -> None:
        """Runs habit/rule evaluation; defers to priority queue when configured."""
        if self._priority_sched is None:
            await self._decision_cycle_inner()
            return

        from core.priority_scheduler import PRIORITY_BACKGROUND

        done = asyncio.Event()

        def coro_factory() -> object:
            async def inner() -> None:
                try:
                    await self._decision_cycle_inner()
                finally:
                    done.set()

            return inner()

        self._priority_sched.submit(
            PRIORITY_BACKGROUND, "autonomy_cycle", coro_factory,
        )
        await done.wait()

    async def _decision_cycle_inner(self) -> None:
        now = time.time()
        ctx = self._last_context
        if not ctx:
            return

        self._behavior.apply_decay()

        await self._check_rule_based_decisions(ctx)

        await self._check_habits(ctx, now)

    # ── Rule-based decisions ──────────────────────────────────────────

    async def _check_rule_based_decisions(self, ctx: dict) -> None:
        cpu = ctx.get("cpu", 0)
        idle_min = ctx.get("idle_minutes", 0)

        if cpu > 80 and not self._throttled:
            self._log_decision("rule", "high_cpu",
                               f"CPU at {cpu:.0f}%, recommending throttle")
            self._bus.emit_fast("autonomy_decision_log",
                                decision="high_cpu", detail=f"CPU {cpu:.0f}%")

        if idle_min > self._idle_timeout_min:
            self._log_decision("rule", "idle_detected",
                               f"Idle for {idle_min:.0f} min")
            self._bus.emit_fast("autonomy_decision_log",
                                decision="idle_reduce",
                                detail=f"Idle {idle_min:.0f}min")

    # ── Habit-based decisions ─────────────────────────────────────────

    def _is_reversible(self, action: str) -> bool:
        """Check if an action can be easily undone."""
        irreversible_actions = {
            "delete_file", "empty_recycle_bin", "kill_process",
            "send_email", "commit_code", "format_drive",
            "drop_table", "uninstall_app"
        }
        return action not in irreversible_actions

    async def _check_habits(self, ctx: dict, now: float) -> None:
        auto_habits = self._behavior.get_auto_habits(ctx)
        for habit in auto_habits:
            hid = habit["id"]

            if hid in _NEVER_AUTO_EXECUTE or habit["action"] in _NEVER_AUTO_EXECUTE:
                continue

            last_exec = self._last_auto_executed.get(hid, 0)
            if now - last_exec < 3600:
                continue

            # Reversibility Check
            if not self._is_reversible(habit["action"]):
                self._log_decision("auto_blocked", hid, "Action is irreversible. Requires confirmation.")
                # Fallback to suggestion instead of auto-execute
                self._suggest_habit(habit, now)
                continue

            from core.security.action_signing import merge_signed_args

            hargs = {"name": habit.get("target", "")}
            hargs = merge_signed_args(self._security, habit["action"], hargs)
            allowed, reason = self._security.allow_action(habit["action"], hargs)
            if not allowed:
                self._log_decision("auto_blocked", hid, reason)
                continue

            self._last_auto_executed[hid] = now
            self._log_decision("auto_execute", hid,
                               f"confidence={habit['confidence']:.2f}")
            self._bus.emit_fast(
                "autonomous_action",
                action=habit["action"],
                target=habit.get("target", ""),
                habit_id=hid,
                confidence=habit["confidence"],
            )
            self._bus.emit_fast(
                "autonomy_decision_log",
                decision="auto_execute",
                detail=f"{habit['action']} {habit.get('target', '')}",
                confidence=habit["confidence"],
            )
            return

        suggest_habits = self._behavior.get_active_habits(ctx)
        for habit in suggest_habits:
            if habit["confidence"] < self._suggest_threshold:
                continue
            if habit["confidence"] >= self._auto_threshold and self._is_reversible(habit["action"]):
                continue # Already handled by auto_habits
            self._suggest_habit(habit, now)
            return

    def _suggest_habit(self, habit: dict, now: float) -> None:
        hid = habit["id"]
        if hid in _NEVER_AUTO_EXECUTE or habit["action"] in _NEVER_AUTO_EXECUTE:
            return

        last_sugg = self._last_suggested.get(hid, 0)
        if now - last_sugg < 1800:
            return

        self._last_suggested[hid] = now
        suggestion_text = self._behavior.format_habit_suggestion(habit)
        if not suggestion_text:
            return

        self._pending_suggestion = habit
        self._log_decision("suggest", hid,
                           f"confidence={habit['confidence']:.2f}")
        self._bus.emit_fast(
            "habit_suggestion",
            text=suggestion_text,
            habit_id=hid,
            confidence=habit["confidence"],
        )
        self._bus.emit_fast(
            "autonomy_decision_log",
            decision="suggest",
            detail=suggestion_text,
            confidence=habit["confidence"],
        )

    # ── Dashboard data ────────────────────────────────────────────────

    def get_habits_summary(self) -> list[dict]:
        """Return active habits for dashboard display."""
        ctx = self._last_context or {}
        habits = self._behavior.get_active_habits(ctx)
        return [
            {
                "id": h["id"],
                "action": h["action"],
                "target": h.get("target", ""),
                "time_pattern": h.get("time_pattern", ""),
                "confidence": round(h["confidence"], 2),
                "auto_execute": h["confidence"] >= self._auto_threshold,
                "occurrences": h.get("occurrences", 0),
            }
            for h in habits[:20]
        ]

    # ── Logging ───────────────────────────────────────────────────────

    def _log_decision(self, dtype: str, target: str, detail: str = "") -> None:
        if not self._log_all:
            return
        try:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry = f"[{ts}] [{dtype.upper()}] {target}"
            if detail:
                entry += f" | {detail}"
            entry += "\n"
            with open(_AUTONOMY_LOG, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception:
            logger.debug("Failed to write autonomy log", exc_info=True)
