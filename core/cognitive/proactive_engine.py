"""
ATOM -- Proactive Intelligence Engine.

Works alongside JarvisCore to provide pattern-based triggers that go
beyond simple time-frequency analysis. While PredictionEngine predicts
the NEXT action, ProactiveEngine detects SITUATIONS that warrant
proactive intervention.

Trigger Categories:
    1. WORKFLOW TRIGGERS -- detects repeated multi-step sequences and
       suggests automation ("You've done open_app -> search -> copy 8 times")

    2. CONTEXT TRIGGERS -- reacts to environmental changes
       (app switch, time transitions, system state changes)

    3. BEHAVIORAL TRIGGERS -- detects anomalies in usage patterns
       (unusually long session, working outside normal hours, sudden
       change in query frequency)

    4. CONVERSATION TRIGGERS -- reacts to conversation state
       (frustration loops, topic exhaustion, long silence after error)

    5. TEMPORAL TRIGGERS -- time-based reminders and routines
       (morning routine, lunch break, end-of-day shutdown)

Each trigger produces a ProactiveInsight (same dataclass as JarvisCore)
which JarvisCore's proactive loop picks up and delivers.

Contract: CognitiveModuleContract (start, stop)
Owner: Satyam
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from datetime import datetime
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.behavior_tracker import BehaviorTracker
    from core.conversation_memory import ConversationMemory
    from core.cognitive.goal_engine import GoalEngine
    from core.jarvis_core import ProactiveInsight
    from core.owner_understanding import OwnerUnderstanding

logger = logging.getLogger("atom.proactive")


class WorkflowPattern:
    """A detected multi-step workflow pattern."""
    __slots__ = ("steps", "count", "avg_gap_s", "last_seen", "suggested")

    def __init__(self, steps: tuple[str, ...], count: int = 1) -> None:
        self.steps = steps
        self.count = count
        self.avg_gap_s: float = 0.0
        self.last_seen: float = time.time()
        self.suggested: bool = False


class ProactiveIntelligenceEngine:
    """Detects situations that warrant proactive intervention.

    Runs on a slower loop than JarvisCore (every 5 minutes), scanning
    for workflow patterns, behavioral anomalies, and contextual triggers.
    """

    __slots__ = (
        "_bus", "_behavior", "_conv_memory", "_owner", "_goals", "_config",
        "_task", "_shutdown",
        "_action_sequence", "_workflow_patterns",
        "_session_start", "_normal_hours", "_last_trigger_times",
        "_check_interval",
        "_idle_minutes", "_idle_signal_time",
        "_morning_briefing_date",
        "_last_download_insight",
        "_m5",
        "_last_scan",
        "_brain_mode_mgr",
        "_quota",
        # Bus-driven scan path: react to context_snapshot (every
        # 60-120s) and active-app changes instead of waiting the full
        # check_interval (default 15min). scan() itself already
        # debounces per-category at 600s, so calling it more often is
        # safe -- cost is just iterating internal data structures.
        "_snapshot_scan_interval",
        "_last_snapshot_scan",
        # Idle-state gate: insights emitted while ATOM is mid-turn
        # (state == thinking/speaking, or command_loop is busy) get
        # buffered and re-emitted on the next listening transition.
        # Without this, proactive notifications arrive AS the LLM
        # answer and shatter the conversation (atom_log.txt L308).
        "_state_provider",
        "_busy_provider",
        "_pending_insights",
        "_max_pending",
    )

    def __init__(
        self,
        bus: AsyncEventBus,
        config: dict | None = None,
        brain_mode_manager: Any | None = None,
        proactive_quota: Any | None = None,
    ) -> None:
        self._bus = bus
        cfg = (config or {}).get("proactive_engine", {})
        self._config = cfg
        self._check_interval = cfg.get("check_interval_s", 300.0)
        self._snapshot_scan_interval = float(
            cfg.get("snapshot_scan_interval_s", 60.0),
        )
        self._last_snapshot_scan: float = 0.0
        self._m5 = cfg.get("m5_triggers") or {}
        self._brain_mode_mgr = brain_mode_manager

        self._behavior: BehaviorTracker | None = None
        self._conv_memory: ConversationMemory | None = None
        self._owner: OwnerUnderstanding | None = None
        self._goals: GoalEngine | None = None

        self._task: asyncio.Task | None = None
        self._shutdown = asyncio.Event()

        self._action_sequence: list[tuple[str, float]] = []
        self._workflow_patterns: dict[tuple[str, ...], WorkflowPattern] = {}
        self._session_start = time.time()
        self._normal_hours: set[int] = set(range(8, 22))
        self._last_trigger_times: dict[str, float] = {}
        self._idle_minutes: float = 0.0
        self._idle_signal_time: float = 0.0
        self._morning_briefing_date: str = ""
        self._last_download_insight: float = 0.0
        self._last_scan: dict[str, Any] | None = None
        self._quota = proactive_quota

        self._state_provider: Any = None
        self._busy_provider: Any = None
        self._pending_insights: list[dict[str, Any]] = []
        self._max_pending: int = 3

    def attach_idle_gate(
        self,
        state_provider: Any,
        busy_provider: Any | None = None,
    ) -> None:
        """Wire a state-provider (StateManager-like, exposing ``current``)
        and an optional busy-provider (CommandLoop-like, exposing
        ``is_busy()``). When both report idle, buffered insights drain.
        """
        self._state_provider = state_provider
        self._busy_provider = busy_provider

    def _is_owner_idle(self) -> bool:
        sp = self._state_provider
        if sp is None:
            return True
        try:
            current = getattr(sp, "current", None)
            if callable(current):
                current = current()
            current_str = str(current or "").lower()
        except Exception:
            return True
        if current_str and current_str not in ("idle", "listening"):
            return False
        bp = self._busy_provider
        if bp is None:
            return True
        try:
            is_busy = bp.is_busy() if callable(getattr(bp, "is_busy", None)) else False
        except Exception:
            is_busy = False
        return not bool(is_busy)

    def drain_pending(self) -> int:
        """Re-emit buffered insights. Caller (state-transition wiring)
        invokes this on every listening transition. Returns count drained.
        """
        if not self._pending_insights or not self._is_owner_idle():
            return 0
        drained = 0
        pending = list(self._pending_insights)
        self._pending_insights.clear()
        for insight in pending:
            try:
                self._bus.emit_long("jarvis_insight", **insight)
                drained += 1
            except Exception:
                logger.debug("drain_pending emit failed", exc_info=True)
        return drained

    def _emit_insight(self, insight_data: dict[str, Any]) -> None:
        cat = str(insight_data.get("category", ""))
        src = str(insight_data.get("source", "proactive_intel"))
        pr_raw = insight_data.get("priority")
        try:
            p = int(pr_raw) if pr_raw is not None else None
        except (TypeError, ValueError):
            p = None
        q = self._quota
        if q is not None and not q.allow_emit(src, cat, p):
            return
        if not self._is_owner_idle():
            # Buffer instead of speaking over the active turn.
            if len(self._pending_insights) < self._max_pending:
                self._pending_insights.append(insight_data)
                logger.debug(
                    "ProactiveEngine: deferred insight (cat=%s, pending=%d)",
                    cat, len(self._pending_insights),
                )
            else:
                logger.debug(
                    "ProactiveEngine: dropped insight (buffer full, cat=%s)",
                    cat,
                )
            return
        self._bus.emit_long("jarvis_insight", **insight_data)

    def _background_enabled(self) -> bool:
        mgr = self._brain_mode_mgr
        if mgr is None:
            return True
        try:
            return bool(mgr.feature_enabled("proactive_background"))
        except Exception:
            return True

    def wire(
        self,
        behavior: BehaviorTracker | None = None,
        conv_memory: ConversationMemory | None = None,
        owner: OwnerUnderstanding | None = None,
        goals: "GoalEngine | None" = None,
    ) -> None:
        self._behavior = behavior
        self._conv_memory = conv_memory
        self._owner = owner
        self._goals = goals

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        self._bus.on("action_executed", self._on_action)
        self._bus.on("system_light_scan", self._on_system_light_scan)
        self._bus.on("idle_detected", self._on_idle_detected)
        self._bus.on("fs_event", self._on_fs_event)
        self._bus.on("system_state_update", self._on_system_state_update)
        # NEW: bus-driven scan trigger. Lets workflow / behavioral /
        # temporal / conversation / m5_context triggers all react in
        # ~1min cadence instead of every 15min.
        self._bus.on("context_snapshot", self._on_context_snapshot)

        async def _supervisor() -> None:
            while not self._shutdown.is_set():
                try:
                    await self._run()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("SUPERVISOR: ProactiveEngine crashed! (%s). Restarting in 10s...", e)
                    await asyncio.sleep(10.0)

        self._task = asyncio.create_task(_supervisor())
        logger.info(
            "Proactive Intelligence Engine started "
            "(snapshot_interval=%.0fs, timer_safety_net=%.0fs)",
            self._snapshot_scan_interval, self._check_interval,
        )

    def stop(self) -> None:
        self._shutdown.set()
        for evt, fn in (
            ("action_executed", self._on_action),
            ("system_light_scan", self._on_system_light_scan),
            ("idle_detected", self._on_idle_detected),
            ("fs_event", self._on_fs_event),
            ("system_state_update", self._on_system_state_update),
            ("context_snapshot", self._on_context_snapshot),
        ):
            try:
                self._bus.off(evt, fn)
            except Exception:
                logger.debug("proactive_engine off(%s) failed", evt, exc_info=True)
        if self._task and not self._task.done():
            self._task.cancel()

    async def _run(self) -> None:
        await asyncio.sleep(120.0)
        while not self._shutdown.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(), timeout=self._check_interval,
                )
                break
            except asyncio.TimeoutError:
                pass
            try:
                if not self._background_enabled():
                    continue
                insights = self.scan()
                for insight_data in insights:
                    self._emit_insight(insight_data)
            except Exception:
                logger.debug("Proactive scan error", exc_info=True)

    # ── Event Handlers ────────────────────────────────────────────────

    async def _on_action(self, action: str = "", **_kw: Any) -> None:
        if not action or action in ("fallback", "empty", "confirm", "deny"):
            return
        now = time.time()
        self._action_sequence.append((action, now))
        if len(self._action_sequence) > 200:
            self._action_sequence = self._action_sequence[-200:]

    async def _on_system_light_scan(self, scan: dict[str, Any] = None, **_kw: Any) -> None:
        """Zero-latency delta monitor for the 5-minute system scan."""
        if not scan:
            return

        if self._last_scan is None:
            self._last_scan = dict(scan)
            return

        old_scan = self._last_scan
        self._last_scan = dict(scan)
        
        latest_ram = scan.get("ram_percent", 0)
        old_ram = old_scan.get("ram_percent", 0)
        
        # 0ms latency mathematical delta check, keeps LLM asleep
        if latest_ram - old_ram > 20.0 and latest_ram > 75.0:
            self._emit_insight({
                "message": (
                    f"Boss, your RAM usage just jumped by {latest_ram - old_ram:.0f} "
                    f"percent. We're at {latest_ram:.0f} percent total capacity."
                ),
                "category": "system",
                "priority": 8,
                "source": "proactive_scanner",
            })
            
        latest_cpu = scan.get("cpu_percent", 0)
        old_cpu = old_scan.get("cpu_percent", 0)
        if latest_cpu - old_cpu > 40.0 and latest_cpu > 80.0:
            self._emit_insight({
                "message": (
                    f"Boss, massive CPU spike detected. We just hit {latest_cpu:.0f} "
                    f"percent. Something heavy just started."
                ),
                "category": "system",
                "priority": 8,
                "source": "proactive_scanner",
            })

    async def _on_context_snapshot(self, **_kw: Any) -> None:
        """Run lightweight scan when fresh context arrives on the bus.

        HealthMonitor emits ``context_snapshot`` each cycle (default
        60-120s) with the same shape (active_app, idle, cpu, ram,
        time_of_day) we used to wait for the 15min ``_check_interval``
        timer to consume. Riding the snapshot makes the engine feel
        live: workflow patterns get suggested within a minute of the
        third repetition, end-of-day prompts arrive at 5pm rather than
        "5pm-plus-up-to-15min", and morning briefing fires within
        ~1min of waking. Per-category 600s cooldown inside ``scan``
        prevents any of this from spamming.
        """
        now = time.time()
        if now - self._last_snapshot_scan < self._snapshot_scan_interval:
            return
        self._last_snapshot_scan = now
        try:
            if not self._background_enabled():
                return
            insights = self.scan()
            for insight_data in insights:
                self._emit_insight(insight_data)
        except Exception:
            logger.debug("Proactive snapshot-driven scan error", exc_info=True)

    async def _on_system_state_update(
        self, snapshot: dict | None = None, changed_app: bool = False, **_kw: Any,
    ) -> None:
        """React to real-time system state changes from SystemStateEngine.

        ``changed_app=True`` is ATOM's canonical "app focus changed"
        signal -- we treat it like a snapshot for scan purposes so a
        workflow trigger that just hit threshold fires the moment the
        owner switches to a relevant app, not on the next timer tick.
        """
        if not snapshot:
            return

        if changed_app:
            # Cheap way to bypass the snapshot debounce when focus
            # actually moves; the per-category cooldown inside scan()
            # still prevents spam.
            self._last_snapshot_scan = 0.0

        battery_pct = int(snapshot.get("battery_pct", 100))
        battery_plugged = bool(snapshot.get("battery_plugged", True))
        if battery_pct <= 10 and not battery_plugged:
            if self._check_cooldown("battery_critical", 600):
                self._emit_insight({
                    "message": f"Boss, battery is critically low at {battery_pct}%. You should plug in soon.",
                    "category": "system",
                    "priority": 9,
                    "source": "proactive_system_state",
                })

        active_app = str(snapshot.get("active_app", "")).lower()
        coding_apps = {"code", "cursor", "pycharm", "intellij", "xcode", "neovim", "vim"}
        if active_app and any(ca in active_app for ca in coding_apps):
            session_hours = (time.time() - self._session_start) / 3600
            if session_hours >= 2.0 and self._check_cooldown("coding_break", 3600):
                self._emit_insight({
                    "message": f"Boss, you've been coding for {session_hours:.1f} hours. Maybe take a short break?",
                    "category": "wellness",
                    "priority": 4,
                    "source": "proactive_system_state",
                })

    def _check_cooldown(self, trigger_key: str, cooldown_s: float) -> bool:
        """Return True if the trigger hasn't fired within cooldown_s."""
        now = time.time()
        last = self._last_trigger_times.get(trigger_key, 0.0)
        if now - last < cooldown_s:
            return False
        self._last_trigger_times[trigger_key] = now
        return True

    async def _on_idle_detected(self, idle_minutes: float = 0, **_kw: Any) -> None:
        """Track idle streak for goal-aware nudges (M5 Phase 6.1)."""
        try:
            self._idle_minutes = float(idle_minutes)
        except (TypeError, ValueError):
            self._idle_minutes = 0.0
        self._idle_signal_time = time.time()

    async def _on_fs_event(
        self,
        path: str = "",
        change: str = "",
        is_dir: bool = False,
        **_kw: Any,
    ) -> None:
        """Proactive download / new-file hints (debounced)."""
        kind = change or str(_kw.get("event") or "")
        if kind != "created" or is_dir or not path:
            return
        low = path.lower()
        is_downloads = "/downloads/" in low or low.rstrip("/").endswith("downloads")
        is_desktop = "/desktop/" in low or low.rstrip("/").endswith("/desktop")
        if not is_downloads and not is_desktop:
            return
        ext = low.rsplit(".", 1)[-1] if "." in low else ""
        interesting = {
            "pdf", "zip", "dmg", "pkg", "doc", "docx", "xls", "xlsx", "ppt", "pptx",
            "csv", "json", "md", "txt", "png", "jpg", "jpeg", "heic", "mp4", "mov",
        }
        if is_desktop and ext not in interesting:
            return
        now = time.time()
        if now - self._last_download_insight < 120.0:
            return
        self._last_download_insight = now
        name = path.rsplit("/", 1)[-1][:80]
        self._emit_insight({
            "message": (
                f"Boss, new file landed: {name}. "
                f"Want a quick summary or should I file it away?"
            ),
            "category": "context_download",
            "priority": 4,
            "source": "proactive_fs",
        })

    # ── Scan All Triggers ─────────────────────────────────────────────

    def scan(self) -> list[dict[str, Any]]:
        """Run all trigger scans and return insight dicts."""
        results: list[dict[str, Any]] = []
        now = time.time()

        for trigger_fn in (
            self._scan_workflow_triggers,
            self._scan_behavioral_triggers,
            self._scan_temporal_triggers,
            self._scan_conversation_triggers,
            self._scan_m5_context_triggers,
        ):
            try:
                for insight in trigger_fn(now):
                    cat = insight.get("category", "")
                    last = self._last_trigger_times.get(cat, 0)
                    if now - last > 600:
                        results.append(insight)
                        self._last_trigger_times[cat] = now
                        if cat == "temporal_morning":
                            self._morning_briefing_date = datetime.now().strftime(
                                "%Y-%m-%d",
                            )
            except Exception:
                logger.debug("Trigger scan error in %s", trigger_fn.__name__,
                             exc_info=True)

        return results

    def _disk_free_gb_boot(self) -> float | None:
        try:
            import psutil

            return round(psutil.disk_usage("/").free / (1024 ** 3), 2)
        except Exception:
            return None

    def _scan_m5_context_triggers(self, now: float) -> list[dict]:
        """Battery, memory, disk, morning briefing, idle+goals, stale projects."""
        if not bool(self._m5.get("enabled", True)):
            return []

        insights: list[dict] = []
        m5 = self._m5
        battery_low = float(m5.get("battery_low_pct", 20))
        memory_high = float(m5.get("memory_high_pct", 85))
        disk_warn_gb = float(m5.get("disk_free_gb_warn", 10))
        stale_days = float(m5.get("project_stale_days", 3))
        idle_goal_min = float(m5.get("idle_goal_nudge_minutes", 25))
        morning_hours = m5.get("morning_briefing_hours", [7, 8, 9])
        try:
            hours_set = {int(h) for h in morning_hours}
        except (TypeError, ValueError):
            hours_set = {7, 8, 9}

        scan = self._last_scan or {}
        dt = datetime.now()
        today = dt.strftime("%Y-%m-%d")

        bat = scan.get("battery") if isinstance(scan, dict) else None
        if isinstance(bat, dict):
            pct = float(bat.get("percent") or 0)
            plugged = bool(bat.get("plugged"))
            if pct > 0 and pct < battery_low and not plugged:
                insights.append({
                    "message": (
                        f"Boss, battery is at {pct:.0f}%. "
                        f"Want me to dim the screen or pause heavy jobs?"
                    ),
                    "category": "system_battery",
                    "priority": 9,
                    "source": "proactive_m5",
                })

        ram = float(scan.get("ram_percent") or 0)
        if ram >= memory_high:
            insights.append({
                "message": (
                    f"Memory pressure is high — about {ram:.0f}% RAM in use. "
                    f"Close a few tabs or want me to list top memory apps?"
                ),
                "category": "system_memory",
                "priority": 7,
                "source": "proactive_m5",
            })

        free_gb = self._disk_free_gb_boot()
        if free_gb is not None and free_gb < disk_warn_gb:
            insights.append({
                "message": (
                    f"Boss, only about {free_gb:.1f} GB free on the system disk. "
                    f"Worth clearing caches or archiving old projects."
                ),
                "category": "system_disk",
                "priority": 8,
                "source": "proactive_m5",
            })

        if (
            dt.weekday() < 5
            and dt.hour in hours_set
            and self._morning_briefing_date != today
            and (now - self._session_start) > 300
        ):
            insights.append({
                "message": (
                    "Good morning stretch, Boss. Want a quick briefing on "
                    "goals and what changed overnight?"
                ),
                "category": "temporal_morning",
                "priority": 5,
                "source": "proactive_m5",
            })

        idle_min = self._idle_minutes
        if (
            idle_min >= idle_goal_min
            and self._goals is not None
            and (now - self._idle_signal_time) < 7200
        ):
            try:
                active = self._goals.get_active_goals()
            except Exception:
                active = []
            if active:
                titles = ", ".join(
                    str(g.get("title", "")) for g in active[:3] if g.get("title")
                )
                insights.append({
                    "message": (
                        f"You've been idle ~{idle_min:.0f} min but still have "
                        f"active goals ({titles}). Want a micro-step on one of them?"
                    ),
                    "category": "goals_idle",
                    "priority": 4,
                    "source": "proactive_m5",
                })

        owner = self._owner
        if owner and owner.context.active_projects:
            top = owner.context.active_projects[0]
            last = float(top.get("last_mentioned") or top.get("first_mentioned") or 0)
            if last > 0:
                days = (now - last) / 86400.0
                if days >= stale_days:
                    name = str(top.get("name", "your project"))[:60]
                    insights.append({
                        "message": (
                            f"It's been ~{days:.0f} days since we touched '{name}'. "
                            f"Still on deck or should I park it?"
                        ),
                        "category": "context_project",
                        "priority": 3,
                        "source": "proactive_m5",
                    })

        return insights

    # ── Workflow Triggers ─────────────────────────────────────────────

    def _scan_workflow_triggers(self, now: float) -> list[dict]:
        """Detect repeated multi-step action sequences."""
        if len(self._action_sequence) < 6:
            return []

        self._detect_workflow_patterns()

        insights = []
        for pattern in self._workflow_patterns.values():
            if pattern.count >= 5 and not pattern.suggested:
                steps_str = " → ".join(
                    s.replace("_", " ") for s in pattern.steps
                )
                insights.append({
                    "message": (
                        f"Boss, I've noticed you do '{steps_str}' frequently "
                        f"({pattern.count} times). Want me to create a "
                        f"shortcut or workflow for it?"
                    ),
                    "category": "workflow",
                    "priority": 5,
                    "source": "proactive",
                })
                pattern.suggested = True

        return insights

    def _detect_workflow_patterns(self) -> None:
        """Extract recurring 2-3 step sequences from action history."""
        actions = [a for a, _ in self._action_sequence]
        if len(actions) < 4:
            return

        for window in (2, 3):
            for i in range(len(actions) - window + 1):
                seq = tuple(actions[i:i + window])
                if len(set(seq)) < 2:
                    continue
                if seq in self._workflow_patterns:
                    self._workflow_patterns[seq].count += 1
                    self._workflow_patterns[seq].last_seen = time.time()
                else:
                    self._workflow_patterns[seq] = WorkflowPattern(seq)

        if len(self._workflow_patterns) > 50:
            sorted_patterns = sorted(
                self._workflow_patterns.items(),
                key=lambda x: x[1].count,
                reverse=True,
            )
            self._workflow_patterns = dict(sorted_patterns[:30])

    # ── Behavioral Triggers ───────────────────────────────────────────

    def _scan_behavioral_triggers(self, now: float) -> list[dict]:
        """Detect anomalies in usage patterns."""
        insights = []
        session_min = (now - self._session_start) / 60
        hour = datetime.now().hour

        if hour not in self._normal_hours and session_min > 30:
            if hour >= 23 or hour < 5:
                insights.append({
                    "message": (
                        "It's pretty late, Boss. You're working outside "
                        "your normal hours. Everything okay, or should I "
                        "help you wrap up?"
                    ),
                    "category": "behavioral",
                    "priority": 5,
                    "source": "proactive",
                })

        if session_min > 360:
            insights.append({
                "message": (
                    f"You've been at it for {session_min / 60:.0f} hours "
                    f"straight, Boss. That's a marathon session. "
                    f"Consider taking a proper break."
                ),
                "category": "behavioral",
                "priority": 3,
                "source": "proactive",
            })

        # Sudden burst detection
        recent_window = [
            ts for _, ts in self._action_sequence
            if now - ts < 120
        ]
        if len(recent_window) > 15:
            insights.append({
                "message": (
                    "You're moving fast, Boss. "
                    f"{len(recent_window)} actions in the last 2 minutes. "
                    "Need me to speed anything up or automate something?"
                ),
                "category": "behavioral",
                "priority": 6,
                "source": "proactive",
            })

        return insights

    # ── Temporal Triggers ─────────────────────────────────────────────

    def _scan_temporal_triggers(self, now: float) -> list[dict]:
        """Time-based routine suggestions."""
        insights = []
        hour = datetime.now().hour
        weekday = datetime.now().weekday()
        is_weekday = weekday < 5

        if is_weekday and hour == 12 and (now - self._session_start) / 60 > 180:
            insights.append({
                "message": (
                    "It's noon, Boss. You've been going since morning. "
                    "Good time for a lunch break?"
                ),
                "category": "temporal",
                "priority": 6,
                "source": "proactive",
            })

        if is_weekday and hour == 17:
            insights.append({
                "message": (
                    "It's 5 PM, Boss. Want me to prepare an end-of-day "
                    "summary, or are you powering through?"
                ),
                "category": "temporal",
                "priority": 7,
                "source": "proactive",
            })

        return insights

    # ── Conversation Triggers ─────────────────────────────────────────

    def _scan_conversation_triggers(self, now: float) -> list[dict]:
        """React to conversation state signals."""
        insights = []

        if not self._conv_memory:
            return insights

        if self._conv_memory.is_frustrated:
            insights.append({
                "message": (
                    "I can tell we're hitting some walls, Boss. "
                    "Want me to approach this differently, or should "
                    "we switch to something else?"
                ),
                "category": "conversation",
                "priority": 3,
                "source": "proactive",
            })

        arc = self._conv_memory.sentiment_arc
        if arc == "declining" and self._conv_memory.turn_count > 5:
            insights.append({
                "message": (
                    "This conversation isn't going as smoothly as usual. "
                    "Let me know if I should adjust my approach, Boss."
                ),
                "category": "conversation",
                "priority": 5,
                "source": "proactive",
            })

        return insights

    # ── Queries ───────────────────────────────────────────────────────

    def get_workflow_suggestions(self) -> list[str]:
        """Get detected workflow patterns for dashboard display."""
        results = []
        for pattern in sorted(
            self._workflow_patterns.values(),
            key=lambda p: p.count, reverse=True,
        )[:5]:
            if pattern.count >= 3:
                steps = " → ".join(s.replace("_", " ") for s in pattern.steps)
                results.append(
                    f"{steps} ({pattern.count} times)"
                )
        return results

    @property
    def pattern_count(self) -> int:
        return len(self._workflow_patterns)
