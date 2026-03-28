"""
ATOM -- JARVIS Intelligence Core (The Mind of ATOM).

Upgraded from v19's shallow template-based intelligence to a true
proactive reasoning engine that fuses ALL intelligence sources:

    - ContextFusionEngine (unified situational picture)
    - OwnerUnderstanding (who Boss is, emotional state, preferences)
    - SystemScanner (hardware state, bottlenecks, health)
    - BehaviorTracker (habit patterns, recurring actions)
    - PredictionEngine (action probability models)
    - MemoryEngine (past conversations, facts)
    - ConversationMemory (current session depth and topics)

v21 Upgrades over v19:
    1. DEEP PROACTIVE INTELLIGENCE (12 insight categories vs 5)
       - Habit-aware suggestions ("You usually open VS Code now")
       - Workflow optimization ("You did X then Y 10 times, want a shortcut?")
       - Emotion-aware interventions (not just break suggestions)
       - Project momentum tracking ("You haven't touched ATOM in 3 days")
       - Learning continuity ("Want to continue learning about Docker?")
       - Environmental triggers (low disk, high CPU, battery + workload)

    2. MULTI-TIER BRIEFINGS (4 types vs 1)
       - Morning briefing (richer: habits, predictions, project status)
       - Return-from-idle summary (with system delta and missed events)
       - End-of-day review (session stats, accomplishments, suggestions)
       - Contextual micro-briefing (on significant state changes)

    3. DEEP CONTEXTUAL INFERENCE
       - Multi-source reference resolution (not just pronouns)
       - Intent disambiguation using conversation thread
       - Expertise-aware response calibration
       - Topic continuity detection

    4. CONVERSATION-AWARE INTELLIGENCE
       - Tracks conversation sentiment arc
       - Detects frustration loops (repeated failed queries)
       - Suggests escalation or approach change
       - Knows when to stay silent vs proactively help

Contract: CognitiveModuleContract (start, stop, persist)
Owner: Satyam
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.behavior_tracker import BehaviorTracker
    from core.cognitive.prediction_engine import PredictionEngine
    from core.context_fusion import ContextFusionEngine
    from core.conversation_memory import ConversationMemory
    from core.memory_engine import MemoryEngine
    from core.owner_understanding import OwnerUnderstanding
    from core.system_scanner import SystemScanner

logger = logging.getLogger("atom.jarvis")


@dataclass
class SituationalContext:
    """Complete situational awareness snapshot."""
    time_of_day: str = ""
    hour: int = 0
    is_weekend: bool = False
    owner_emotion: str = "neutral"
    owner_energy: str = "normal"
    active_app: str = ""
    active_project: str = ""
    recent_topics: list[str] = field(default_factory=list)
    system_health: int = 100
    is_on_battery: bool = False
    battery_percent: float = 100.0
    session_duration_min: float = 0.0
    idle_minutes: float = 0.0
    should_be_quiet: bool = False
    conversation_depth: int = 0


@dataclass
class ProactiveInsight:
    """A proactive suggestion or insight for the owner."""
    category: str = ""       # briefing, optimization, health, reminder, suggestion,
                             # habit, workflow, learning, project, emotion, environment
    priority: int = 5        # 1=critical, 5=low, 10=ambient
    message: str = ""
    action: str = ""
    action_args: dict = field(default_factory=dict)
    expires_at: float = 0.0
    spoken: bool = False
    source: str = ""         # which intelligence source generated this


class JarvisCore:
    """The mind of ATOM -- JARVIS-level intelligence fusion.

    v21 integrates ContextFusionEngine, BehaviorTracker, PredictionEngine,
    and ConversationMemory for deep proactive intelligence that anticipates
    needs, detects patterns, and provides contextually rich briefings.
    """

    def __init__(
        self,
        bus: AsyncEventBus,
        owner: OwnerUnderstanding,
        scanner: SystemScanner,
        config: dict | None = None,
    ) -> None:
        self._bus = bus
        self._owner = owner
        self._scanner = scanner
        self._config = config or {}

        self._task: asyncio.Task | None = None
        self._shutdown = asyncio.Event()

        self._situation = SituationalContext()
        self._pending_insights: list[ProactiveInsight] = []
        self._delivered_insights: list[str] = []
        self._delivered_categories: dict[str, float] = {}
        self._last_briefing_date: str = ""
        self._last_eod_date: str = ""
        self._last_idle_summary_time: float = 0.0
        self._session_start = time.time()
        self._conversation_count = 0
        self._silence_streak = 0
        self._failed_query_streak = 0
        self._last_emotion: str = "neutral"
        self._emotion_shift_count = 0

        self._fusion: ContextFusionEngine | None = None
        self._behavior: BehaviorTracker | None = None
        self._prediction: PredictionEngine | None = None
        self._conv_memory: ConversationMemory | None = None
        self._memory: MemoryEngine | None = None

        jcfg = self._config.get("jarvis_core", {})
        self._proactive_interval = jcfg.get("proactive_interval_s", 120.0)
        self._morning_briefing_enabled = jcfg.get("morning_briefing", True)
        self._eod_review_enabled = jcfg.get("eod_review", True)
        self._idle_summary_enabled = jcfg.get("idle_summary", True)
        self._habit_suggestions_enabled = jcfg.get("habit_suggestions", True)
        self._quiet_hours = jcfg.get("quiet_hours", [0, 1, 2, 3, 4, 5])
        self._category_cooldown_s = jcfg.get("category_cooldown_s", 600.0)

    # ── Wiring (v21: plug in additional intelligence sources) ─────

    def wire_intelligence(
        self,
        fusion: ContextFusionEngine | None = None,
        behavior: BehaviorTracker | None = None,
        prediction: PredictionEngine | None = None,
        conv_memory: ConversationMemory | None = None,
        memory: MemoryEngine | None = None,
    ) -> None:
        """Wire additional intelligence sources after construction."""
        self._fusion = fusion
        self._behavior = behavior
        self._prediction = prediction
        self._conv_memory = conv_memory
        self._memory = memory

    # ── Situational Awareness ─────────────────────────────────────

    def update_situation(self) -> SituationalContext:
        """Build a complete situational awareness snapshot."""
        now = datetime.now()
        s = self._situation

        s.hour = now.hour
        s.is_weekend = now.weekday() >= 5

        if 5 <= now.hour < 12:
            s.time_of_day = "morning"
        elif 12 <= now.hour < 17:
            s.time_of_day = "afternoon"
        elif 17 <= now.hour < 21:
            s.time_of_day = "evening"
        else:
            s.time_of_day = "night"

        s.owner_emotion = self._owner.emotion.primary
        s.owner_energy = self._owner.anticipation.current_energy_level
        s.recent_topics = self._owner.context.last_discussed_topics[:5]

        if self._owner.context.active_projects:
            s.active_project = self._owner.context.active_projects[0].get("name", "")

        scan = self._scanner.last_scan
        if scan:
            s.system_health = scan.get("health", {}).get("overall", 100)
            sys_info = scan.get("system", {})
            s.is_on_battery = (
                sys_info.get("has_battery", False)
                and not sys_info.get("is_plugged", True)
            )
            s.battery_percent = sys_info.get("battery_percent", 100)

        s.session_duration_min = (time.time() - self._session_start) / 60
        s.should_be_quiet = now.hour in self._quiet_hours
        s.conversation_depth = self._conversation_count

        if s.owner_emotion != self._last_emotion:
            self._emotion_shift_count += 1
            self._last_emotion = s.owner_emotion

        return s

    # ── Contextual Inference (v21: deeper multi-source) ───────────

    def infer_context(self, text: str) -> dict[str, Any]:
        """Analyze user input with deep contextual understanding.

        v21: uses conversation memory topics, owner expertise, and
        prediction engine to enrich inference beyond simple pronouns.
        """
        context: dict[str, Any] = {
            "original_text": text,
            "inferred_intent": None,
            "resolved_references": {},
            "suggested_tone": "normal",
            "additional_context": [],
            "expertise_level": "general",
            "conversation_depth": "shallow",
        }

        text_lower = text.lower()

        # Pronoun resolution from conversation context
        last_topics = self._owner.context.last_discussed_topics
        pronoun_refs = {
            "it": last_topics[0] if last_topics else None,
            "that": last_topics[0] if last_topics else None,
            "this": last_topics[0] if last_topics else None,
            "them": None,
        }

        for pronoun, ref in pronoun_refs.items():
            if ref and pronoun in text_lower.split():
                context["resolved_references"][pronoun] = ref
                context["additional_context"].append(
                    f"'{pronoun}' likely refers to '{ref}'"
                )

        # Active project awareness
        if self._owner.context.active_projects:
            for project in self._owner.context.active_projects:
                pname = project.get("name", "").lower()
                if pname and any(word in text_lower for word in pname.split()):
                    context["additional_context"].append(
                        f"Related to active project: {project['name']}"
                    )

        # v21: Expertise-aware calibration
        expertise = self._owner.topics.expertise_areas
        if expertise:
            for area, score in expertise.items():
                if area.lower() in text_lower and score > 5:
                    context["expertise_level"] = "expert"
                    context["additional_context"].append(
                        f"Owner is experienced with {area} -- use technical language"
                    )
                    break

        # v21: Conversation depth awareness
        if self._conv_memory:
            tc = self._conv_memory.turn_count
            if tc > 8:
                context["conversation_depth"] = "deep"
                context["additional_context"].append(
                    "Deep in conversation -- maintain context continuity"
                )
            elif tc > 3:
                context["conversation_depth"] = "medium"

            active_topics = self._conv_memory.active_topics
            if active_topics:
                context["additional_context"].append(
                    f"Active conversation topics: {', '.join(active_topics[:3])}"
                )

        # Emotion-aware tone adjustment
        adjustments = self._owner.get_personality_adjustment()
        context["suggested_tone"] = adjustments.get("tone", "normal")
        context["suggested_verbosity"] = adjustments.get("verbosity", "medium")

        emotion = self._owner.emotion.primary
        _EMOTION_GUIDANCE = {
            "frustrated": "Owner seems frustrated -- be concise, solution-focused, empathetic",
            "tired": "Owner seems tired -- keep responses brief and offer to automate",
            "stressed": "Owner is stressed -- be calm, organized, reduce cognitive load",
            "excited": "Owner is enthusiastic -- match energy, encourage exploration",
            "focused": "Owner is in flow state -- be minimal, answer precisely",
        }
        guidance = _EMOTION_GUIDANCE.get(emotion)
        if guidance:
            context["additional_context"].append(guidance)

        # v21: Failed query streak detection
        if self._failed_query_streak >= 3:
            context["additional_context"].append(
                "Owner has had multiple unsuccessful queries -- "
                "try a different approach, offer alternatives"
            )

        return context

    # ── Proactive Intelligence (v21: 12 categories) ───────────────

    def generate_proactive_insights(self) -> list[ProactiveInsight]:
        """Scan ALL intelligence sources for proactive insights.

        v21 generates insights from 12 categories:
        health, battery, system, emotion, habit, workflow,
        learning, project, prediction, briefing, environment, frustration
        """
        insights: list[ProactiveInsight] = []
        now = time.time()
        s = self._situation

        # ── Health: session duration + energy ──
        if s.session_duration_min > 120 and s.owner_energy == "low":
            insights.append(ProactiveInsight(
                category="health", priority=3, source="owner+time",
                message=(
                    f"Boss, you've been working for "
                    f"{s.session_duration_min:.0f} minutes and your energy "
                    f"seems low. A short break would recharge you."
                ),
                expires_at=now + 1800,
            ))
        elif s.session_duration_min > 180:
            insights.append(ProactiveInsight(
                category="health", priority=4, source="time",
                message=(
                    f"{s.session_duration_min / 60:.1f} hours straight, Boss. "
                    f"Even I need a cooldown cycle sometimes."
                ),
                expires_at=now + 3600,
            ))

        if self._owner.anticipation.should_suggest_break:
            insights.append(ProactiveInsight(
                category="health", priority=4, source="owner",
                message="You've been at it for a while. Want me to set a 5-minute break timer?",
                action="set_timer",
                action_args={"minutes": 5, "label": "Break"},
                expires_at=now + 3600,
            ))

        # ── Battery awareness ──
        if s.is_on_battery and s.battery_percent < 25:
            urgency = 1 if s.battery_percent < 10 else 2
            insights.append(ProactiveInsight(
                category="battery", priority=urgency, source="system",
                message=(
                    f"Battery at {s.battery_percent:.0f}%. "
                    + ("Critical -- plug in NOW, Boss." if s.battery_percent < 10
                       else "I'd recommend plugging in soon, Boss.")
                ),
                expires_at=now + 300,
            ))
        elif s.is_on_battery and s.battery_percent < 50 and s.session_duration_min > 60:
            insights.append(ProactiveInsight(
                category="battery", priority=4, source="system+time",
                message=(
                    f"Battery at {s.battery_percent:.0f}% and you've been "
                    f"going for {s.session_duration_min:.0f} minutes. "
                    f"Might want to plug in if you're planning a long session."
                ),
                expires_at=now + 1800,
            ))

        # ── System health ──
        if s.system_health < 50:
            bottlenecks = self._scanner.bottlenecks
            if bottlenecks:
                critical = [
                    b for b in bottlenecks
                    if b.severity in ("critical", "high")
                ]
                if critical:
                    bn = critical[0]
                    insights.append(ProactiveInsight(
                        category="system", priority=2, source="scanner",
                        message=(
                            f"System issue detected: {bn.description}. "
                            f"{bn.suggestion}"
                        ),
                        expires_at=now + 3600,
                    ))

        # ── Emotion-aware interventions ──
        if s.owner_emotion == "stressed" and self._conversation_count > 5:
            insights.append(ProactiveInsight(
                category="emotion", priority=4, source="owner",
                message=(
                    "I notice things are getting intense, Boss. "
                    "Want me to handle some routine tasks to free up your focus?"
                ),
                expires_at=now + 1800,
            ))

        if self._emotion_shift_count >= 4:
            insights.append(ProactiveInsight(
                category="emotion", priority=5, source="owner",
                message=(
                    "Your mood has been shifting a lot this session, Boss. "
                    "Sometimes a change of pace helps. "
                    "Want me to play some music or switch things up?"
                ),
                action="play_youtube",
                action_args={"query": "lofi focus music"},
                expires_at=now + 3600,
            ))
            self._emotion_shift_count = 0

        # ── v21: Habit-based suggestions ──
        if self._habit_suggestions_enabled and self._behavior:
            habits = self._behavior.get_active_habits({
                "time_of_day": s.time_of_day,
                "weekday": datetime.now().weekday(),
            })
            for habit in habits[:2]:
                if habit["confidence"] >= 0.6:
                    suggestion = self._behavior.format_habit_suggestion(habit)
                    if suggestion:
                        insights.append(ProactiveInsight(
                            category="habit", priority=5, source="behavior",
                            message=suggestion,
                            action=habit.get("action", ""),
                            action_args={"target": habit.get("target", "")},
                            expires_at=now + 1800,
                        ))

        # ── v21: Prediction-based anticipation ──
        if self._prediction:
            predictions = self._prediction.predict_next(max_results=1)
            for pred in predictions:
                if pred.confidence >= 0.7:
                    action_str = pred.action.replace("_", " ")
                    target_str = f" {pred.target}" if pred.target else ""
                    insights.append(ProactiveInsight(
                        category="prediction", priority=6, source="prediction",
                        message=(
                            f"Based on your patterns, you might want to "
                            f"{action_str}{target_str} soon. {pred.reason}"
                        ),
                        action=pred.action,
                        action_args={"target": pred.target} if pred.target else {},
                        expires_at=now + 1800,
                    ))

        # ── v21: Learning continuity ──
        if self._owner.topics.learning_areas:
            for area in self._owner.topics.learning_areas[:1]:
                if not any(area.lower() in t.lower() for t in s.recent_topics):
                    insights.append(ProactiveInsight(
                        category="learning", priority=7, source="owner",
                        message=(
                            f"You've been learning about {area} recently. "
                            f"Want to continue where you left off?"
                        ),
                        expires_at=now + 7200,
                    ))

        # ── v21: Project momentum ──
        if s.active_project and self._conv_memory:
            project_mentioned = any(
                s.active_project.lower() in t.lower()
                for t in (self._conv_memory.active_topics or [])
            )
            if (
                not project_mentioned
                and s.session_duration_min > 30
                and self._conversation_count > 10
            ):
                insights.append(ProactiveInsight(
                    category="project", priority=7, source="owner+conv",
                    message=(
                        f"You haven't mentioned '{s.active_project}' this session. "
                        f"Need to switch focus, or should I keep it on your radar?"
                    ),
                    expires_at=now + 3600,
                ))

        # ── v21: Frustration loop detection ──
        if self._failed_query_streak >= 3:
            insights.append(ProactiveInsight(
                category="frustration", priority=3, source="conversation",
                message=(
                    "I notice we've hit a few walls, Boss. "
                    "Want me to try a completely different approach, "
                    "or should we tackle something else and come back to this?"
                ),
                expires_at=now + 900,
            ))
            self._failed_query_streak = 0

        # ── Briefings (morning, end-of-day) ──
        now_dt = datetime.now()
        today_str = now_dt.strftime("%Y-%m-%d")

        if (self._morning_briefing_enabled
                and 7 <= now_dt.hour <= 10
                and self._last_briefing_date != today_str
                and self._conversation_count <= 2):
            briefing = self._generate_morning_briefing()
            if briefing:
                insights.append(ProactiveInsight(
                    category="briefing", priority=2, source="jarvis",
                    message=briefing,
                    expires_at=now + 7200,
                ))
                self._last_briefing_date = today_str

        if (self._eod_review_enabled
                and 21 <= now_dt.hour <= 23
                and self._last_eod_date != today_str
                and s.session_duration_min > 30):
            review = self._generate_eod_review()
            if review:
                insights.append(ProactiveInsight(
                    category="briefing", priority=4, source="jarvis",
                    message=review,
                    expires_at=now + 7200,
                ))
                self._last_eod_date = today_str

        # Filter: deduplicate, respect cooldowns, remove expired
        self._pending_insights = self._filter_insights(insights, now)
        return self._pending_insights

    def _filter_insights(
        self, insights: list[ProactiveInsight], now: float,
    ) -> list[ProactiveInsight]:
        """Deduplicate, enforce category cooldowns, remove expired/delivered."""
        filtered: list[ProactiveInsight] = []
        for i in insights:
            if i.message in self._delivered_insights:
                continue
            if i.expires_at and i.expires_at < now:
                continue
            last_cat_time = self._delivered_categories.get(i.category, 0)
            if now - last_cat_time < self._category_cooldown_s:
                continue
            filtered.append(i)
        return filtered

    def get_next_insight(self) -> ProactiveInsight | None:
        """Get the highest-priority unspoken insight."""
        for insight in sorted(self._pending_insights, key=lambda i: i.priority):
            if not insight.spoken:
                insight.spoken = True
                self._delivered_insights.append(insight.message)
                self._delivered_categories[insight.category] = time.time()
                if len(self._delivered_insights) > 100:
                    self._delivered_insights = self._delivered_insights[-50:]
                return insight
        return None

    # ── Briefings (v21: richer, multi-type) ───────────────────────

    def _generate_morning_briefing(self) -> str:
        """Generate a rich JARVIS-style morning briefing."""
        now = datetime.now()
        title = self._owner._owner_title
        parts = [
            f"Good morning, {title}. "
            f"It's {now.strftime('%A, %B %d')}."
        ]

        # System status
        scan = self._scanner.last_scan
        if scan:
            health = scan.get("health", {}).get("overall", 100)
            if health >= 80:
                parts.append("All systems nominal.")
            elif health >= 60:
                parts.append(f"System health at {health}%. Minor issues detected.")
            else:
                parts.append(
                    f"System health at {health}%. "
                    f"I recommend checking diagnostics."
                )

        # Active project
        if self._owner.context.active_projects:
            project = self._owner.context.active_projects[0]["name"]
            parts.append(f"Your active project is '{project}'.")

        # v21: Predictions for the morning
        if self._prediction:
            preds = self._prediction.predict_next(max_results=2)
            if preds:
                pred_parts = []
                for p in preds:
                    action_str = p.action.replace("_", " ")
                    target = f" {p.target}" if p.target else ""
                    pred_parts.append(f"{action_str}{target}")
                parts.append(
                    f"Based on your patterns, you'll likely want to: "
                    f"{', '.join(pred_parts)}."
                )

        # v21: Habit reminders
        if self._behavior:
            habits = self._behavior.get_active_habits({
                "time_of_day": "morning",
                "weekday": now.weekday(),
            })
            auto = [h for h in habits if h["confidence"] >= 0.8]
            if auto:
                h = auto[0]
                parts.append(
                    f"Shall I go ahead and {h['action'].replace('_', ' ')}"
                    + (f" {h['target']}?" if h['target'] else "?")
                )

        # Learning continuity
        if self._owner.topics.learning_areas:
            parts.append(
                f"You've been learning about "
                f"{self._owner.topics.learning_areas[0]}."
            )

        # Closing tone
        adjustments = self._owner.get_personality_adjustment()
        if adjustments.get("tone") == "supportive":
            parts.append("Take it easy today. I've got your back.")
        else:
            parts.append("Ready to make it a great day.")

        return " ".join(parts)

    def _generate_eod_review(self) -> str:
        """Generate an end-of-day review briefing."""
        title = self._owner._owner_title
        s = self._situation
        parts = [f"End of day review, {title}."]

        hours = s.session_duration_min / 60
        parts.append(
            f"You've been active for {hours:.1f} hours "
            f"with {self._conversation_count} interactions."
        )

        if self._conv_memory:
            topics = self._conv_memory.active_topics
            if topics:
                parts.append(f"Main topics today: {', '.join(topics[:5])}.")

        if s.active_project:
            parts.append(f"Active project: {s.active_project}.")

        if s.owner_energy == "low":
            parts.append("You seem tired. Get some rest -- I'll be here tomorrow.")
        else:
            parts.append("Great session. See you tomorrow.")

        return " ".join(parts)

    def generate_idle_summary(self) -> str | None:
        """Generate a return-from-idle summary with system delta."""
        now = time.time()
        if now - self._last_idle_summary_time < 3600:
            return None

        idle_min = self._situation.idle_minutes
        if idle_min < 15:
            return None

        self._last_idle_summary_time = now
        title = self._owner._owner_title
        parts = [f"Welcome back, {title}."]

        if idle_min > 60:
            parts.append(f"You were away for about {idle_min / 60:.0f} hours.")
        else:
            parts.append(f"You were away for about {idle_min:.0f} minutes.")

        scan = self._scanner.last_scan
        if scan:
            health = scan.get("health", {}).get("overall", 100)
            if health < 70:
                parts.append(f"System health has dropped to {health}%.")
            else:
                parts.append("All systems running smoothly.")

        # v21: Remind of active context
        if self._situation.active_project:
            parts.append(
                f"You were working on {self._situation.active_project}."
            )

        if self._conv_memory and self._conv_memory.active_topics:
            topics = self._conv_memory.active_topics
            parts.append(f"Last discussing: {', '.join(topics[:3])}.")

        return " ".join(parts)

    def generate_micro_briefing(self, trigger: str, **data: Any) -> str | None:
        """Generate a contextual micro-briefing for state changes.

        Triggers: app_switch, health_drop, emotion_shift, milestone
        """
        title = self._owner._owner_title

        if trigger == "health_drop":
            health = data.get("health", 0)
            return (
                f"Heads up, {title}. System health dropped to {health}%. "
                f"I'm monitoring it."
            )

        if trigger == "emotion_shift":
            old = data.get("old", "neutral")
            new = data.get("new", "neutral")
            if new in ("frustrated", "stressed") and old not in ("frustrated", "stressed"):
                return (
                    f"I sense a shift, {title}. "
                    f"Let me know if you need me to take anything off your plate."
                )

        if trigger == "milestone":
            count = data.get("count", 0)
            milestones = {
                100: f"That's 100 conversations, {title}. We're building something real.",
                500: f"500 conversations, {title}. I know your patterns well now.",
                1000: f"A thousand conversations, {title}. What a journey.",
            }
            return milestones.get(count)

        return None

    # ── LLM Context Enhancement ──────────────────────────────────

    def get_jarvis_context_for_llm(self) -> str:
        """Generate comprehensive JARVIS context for LLM prompt injection.

        v21: delegates to ContextFusionEngine when available for a
        unified picture. Falls back to manual assembly otherwise.
        """
        if self._fusion:
            return self._fusion.get_llm_context_block()

        s = self.update_situation()
        lines = []

        lines.append(
            f"[SITUATION] {s.time_of_day.title()}, "
            f"{'weekend' if s.is_weekend else 'weekday'} | "
            f"Session: {s.session_duration_min:.0f}min | "
            f"System health: {s.system_health}/100"
        )

        owner_ctx = self._owner.get_owner_context_for_llm()
        if owner_ctx:
            lines.append(owner_ctx)

        system_ctx = self._scanner.get_intelligence_for_llm()
        if system_ctx:
            lines.append(system_ctx)

        if s.should_be_quiet:
            lines.append("[MODE] Quiet hours -- keep responses brief")
        elif s.owner_emotion == "focused":
            lines.append("[MODE] Owner is focused -- be efficient, avoid chitchat")
        elif s.owner_emotion == "frustrated":
            lines.append(
                "[MODE] Owner is frustrated -- be helpful, empathetic, solution-focused"
            )

        if s.active_project:
            lines.append(f"[ACTIVE PROJECT] {s.active_project}")

        if s.is_on_battery:
            lines.append(f"[POWER] On battery: {s.battery_percent:.0f}%")

        return "\n".join(lines)

    def enhance_query(self, text: str) -> dict[str, Any]:
        """Enhance a user query with JARVIS-level context.

        v21: includes conversation depth, expertise hints, and
        prediction data alongside the standard context.
        """
        inference = self.infer_context(text)
        s = self.update_situation()

        enhanced: dict[str, Any] = {
            "text": text,
            "jarvis_context": self.get_jarvis_context_for_llm(),
            "inference": inference,
            "situation": {
                "time_of_day": s.time_of_day,
                "emotion": s.owner_emotion,
                "energy": s.owner_energy,
                "active_project": s.active_project,
                "session_minutes": s.session_duration_min,
                "conversation_depth": s.conversation_depth,
            },
            "personality_adjustments": self._owner.get_personality_adjustment(),
        }

        if self._fusion:
            fused = self._fusion.get_fused_context(text)
            enhanced["fused_context_quality"] = fused.quality_score()
            enhanced["should_be_brief"] = fused.should_be_brief
            enhanced["should_be_proactive"] = fused.should_be_proactive

        return enhanced

    # ── Event Handlers ────────────────────────────────────────────

    async def _on_speech(self, text: str = "", **_kw) -> None:
        self._conversation_count += 1
        self._silence_streak = 0

        if self._fusion:
            self._fusion.log_action("speech", text[:60])

        milestone = self.generate_micro_briefing(
            "milestone", count=self._conversation_count,
        )
        if milestone:
            self._bus.emit_long("jarvis_insight",
                                message=milestone, category="milestone",
                                priority=6)

    async def _on_idle_detected(self, idle_minutes: float = 0, **_kw) -> None:
        self._situation.idle_minutes = idle_minutes
        if idle_minutes > 15 and self._idle_summary_enabled:
            summary = self.generate_idle_summary()
            if summary:
                self._bus.emit_long("response_ready", text=summary)

    async def _on_context_snapshot(self, **kw) -> None:
        if kw.get("active_app"):
            self._situation.active_app = kw["active_app"]

    async def _on_action_executed(self, action: str = "", **kw) -> None:
        """Track executed actions for fusion and behavior intelligence."""
        if self._fusion:
            detail = kw.get("target", kw.get("detail", ""))
            self._fusion.log_action(action, str(detail))

    async def _on_query_failed(self, **_kw) -> None:
        """Track failed queries for frustration loop detection."""
        self._failed_query_streak += 1

    async def _on_query_succeeded(self, **_kw) -> None:
        self._failed_query_streak = 0

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        self._bus.on("speech_final", self._on_speech)
        self._bus.on("idle_detected", self._on_idle_detected)
        self._bus.on("context_snapshot", self._on_context_snapshot)
        self._bus.on("action_executed", self._on_action_executed)
        self._bus.on("query_failed", self._on_query_failed)
        self._bus.on("query_succeeded", self._on_query_succeeded)

        self._task = asyncio.ensure_future(self._proactive_loop())
        logger.info(
            "JARVIS Core v21 started (proactive_interval=%.0fs, "
            "fusion=%s, behavior=%s, prediction=%s)",
            self._proactive_interval,
            "wired" if self._fusion else "none",
            "wired" if self._behavior else "none",
            "wired" if self._prediction else "none",
        )

    def stop(self) -> None:
        self._shutdown.set()
        if self._task and not self._task.done():
            self._task.cancel()

    async def _proactive_loop(self) -> None:
        """Periodic proactive intelligence generation."""
        await asyncio.sleep(30.0)

        while not self._shutdown.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(), timeout=self._proactive_interval,
                )
                break
            except asyncio.TimeoutError:
                pass

            try:
                self.update_situation()

                if self._situation.should_be_quiet:
                    continue

                insights = self.generate_proactive_insights()
                if insights:
                    insight = self.get_next_insight()
                    if insight and insight.priority <= 5:
                        self._bus.emit_long(
                            "jarvis_insight",
                            message=insight.message,
                            category=insight.category,
                            priority=insight.priority,
                            action=insight.action,
                            action_args=insight.action_args,
                            source=insight.source,
                        )
            except Exception:
                logger.debug("JARVIS proactive loop error", exc_info=True)

    def persist(self) -> None:
        pass
