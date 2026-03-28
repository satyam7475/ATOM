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
        "_bus", "_behavior", "_conv_memory", "_owner", "_config",
        "_task", "_shutdown",
        "_action_sequence", "_workflow_patterns",
        "_session_start", "_normal_hours", "_last_trigger_times",
        "_check_interval",
    )

    def __init__(
        self,
        bus: AsyncEventBus,
        config: dict | None = None,
    ) -> None:
        self._bus = bus
        cfg = (config or {}).get("proactive_engine", {})
        self._config = cfg
        self._check_interval = cfg.get("check_interval_s", 300.0)

        self._behavior: BehaviorTracker | None = None
        self._conv_memory: ConversationMemory | None = None
        self._owner: OwnerUnderstanding | None = None

        self._task: asyncio.Task | None = None
        self._shutdown = asyncio.Event()

        self._action_sequence: list[tuple[str, float]] = []
        self._workflow_patterns: dict[tuple[str, ...], WorkflowPattern] = {}
        self._session_start = time.time()
        self._normal_hours: set[int] = set(range(8, 22))
        self._last_trigger_times: dict[str, float] = {}

    def wire(
        self,
        behavior: BehaviorTracker | None = None,
        conv_memory: ConversationMemory | None = None,
        owner: OwnerUnderstanding | None = None,
    ) -> None:
        self._behavior = behavior
        self._conv_memory = conv_memory
        self._owner = owner

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        self._bus.on("action_executed", self._on_action)
        self._task = asyncio.create_task(self._run())
        logger.info(
            "Proactive Intelligence Engine started (interval=%.0fs)",
            self._check_interval,
        )

    def stop(self) -> None:
        self._shutdown.set()
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
                insights = self.scan()
                for insight_data in insights:
                    self._bus.emit_long("jarvis_insight", **insight_data)
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
        ):
            try:
                for insight in trigger_fn(now):
                    cat = insight.get("category", "")
                    last = self._last_trigger_times.get(cat, 0)
                    if now - last > 600:
                        results.append(insight)
                        self._last_trigger_times[cat] = now
            except Exception:
                logger.debug("Trigger scan error in %s", trigger_fn.__name__,
                             exc_info=True)

        return results

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
