"""
ATOM -- Context Fusion Engine (The Thalamus).

In neuroscience, the thalamus relays and integrates sensory information
before it reaches the cortex. This module does the same for ATOM --
it fuses ALL intelligence sources into a single, coherent context
that the LLM and other modules can consume.

Before this module, each intelligence source operated in isolation:
  - OwnerUnderstanding knew the owner but not the system state
  - SystemScanner knew the system but not the conversation context
  - MemoryEngine recalled facts but not the owner's mood
  - ConversationMemory tracked turns but not the situation

ContextFusion creates a UNIFIED INTELLIGENCE PICTURE:

  1. SITUATION LAYER -- time, mode, energy, emotion, system health
  2. CONVERSATION LAYER -- current topic thread, depth, sentiment arc
  3. OWNER LAYER -- expertise, preferences, communication style
  4. SYSTEM LAYER -- hardware state, active app, bottlenecks
  5. MEMORY LAYER -- relevant past interactions, relevant knowledge
  6. ACTION LAYER -- recent actions taken, pending items, predictions
  7. META LAYER -- ATOM's own confidence, uncertainty indicators

The LLM prompt builder consumes this fused context instead of
manually stitching together fragments from each source.

Contract:
    get_fused_context(query) -> FusedContext  # complete picture
    get_llm_context_block() -> str            # compact for prompt injection
    get_conversation_state() -> ConversationState  # deep conversation analysis

Owner: Satyam
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.conversation_memory import ConversationMemory
    from core.jarvis_core import JarvisCore
    from core.memory_engine import MemoryEngine
    from core.owner_understanding import OwnerUnderstanding
    from core.system_scanner import SystemScanner


@dataclass
class ConversationState:
    """Deep analysis of the current conversation."""
    turn_count: int = 0
    topic_thread: list[str] = field(default_factory=list)
    sentiment_arc: str = "neutral"   # improving, neutral, declining
    depth: str = "shallow"           # shallow (1-3), medium (4-8), deep (9+)
    is_continuation: bool = False
    last_intent: str = ""
    is_multi_step: bool = False
    unresolved_question: str = ""


@dataclass
class FusedContext:
    """Complete intelligence picture for a single query."""
    # Situation
    time_of_day: str = ""
    hour: int = 0
    is_weekend: bool = False
    personality_mode: str = "work"
    session_minutes: float = 0.0

    # Owner
    emotion: str = "neutral"
    emotion_trajectory: str = "stable"
    energy: str = "normal"
    expertise_hint: str = ""
    preferred_verbosity: str = "medium"
    communication_style: str = "casual"

    # System
    system_health: int = 100
    is_on_battery: bool = False
    battery_pct: float = 100.0
    active_app: str = ""
    cpu_load: float = 0.0
    system_index_summary: str = ""
    playing_media: str = ""

    # Conversation
    conversation: ConversationState = field(default_factory=ConversationState)

    # Memory
    relevant_memories: list[str] = field(default_factory=list)
    relevant_facts: list[str] = field(default_factory=list)
    l1_cache_summary: str = ""

    # Action
    active_project: str = ""
    recent_actions: list[str] = field(default_factory=list)
    predictions: list[str] = field(default_factory=list)

    # Meta
    confidence: float = 0.8
    should_be_brief: bool = False
    should_be_proactive: bool = False
    context_richness: float = 0.5   # 0=no context, 1=fully enriched

    def quality_score(self) -> float:
        """How complete is this fused context? 0.0 to 1.0."""
        scores = [
            1.0 if self.time_of_day else 0.0,
            1.0 if self.emotion != "neutral" else 0.3,
            1.0 if self.system_health < 100 else 0.5,
            min(1.0, len(self.conversation.topic_thread) / 3),
            min(1.0, len(self.relevant_memories) / 2),
            1.0 if self.active_project else 0.0,
        ]
        return sum(scores) / len(scores)


class ContextFusionEngine:
    """Fuses all intelligence sources into a unified context.

    Call get_fused_context() to get a complete FusedContext for any query.
    Call get_llm_context_block() for a compact prompt-injectable string.
    """

    __slots__ = (
        "_bus", "_owner", "_scanner", "_memory", "_conv_memory",
        "_jarvis", "_session_start", "_last_fused_context",
        "_action_log", "_config",
    )

    def __init__(
        self,
        bus: AsyncEventBus | None = None,
        config: dict | None = None,
    ) -> None:
        self._bus = bus
        self._config = config or {}
        self._owner: OwnerUnderstanding | None = None
        self._scanner: SystemScanner | None = None
        self._memory: MemoryEngine | None = None
        self._conv_memory: ConversationMemory | None = None
        self._jarvis: JarvisCore | None = None
        self._session_start = time.time()
        self._last_fused_context: FusedContext | None = None
        self._action_log: list[dict] = []

    def wire(
        self,
        owner: OwnerUnderstanding | None = None,
        scanner: SystemScanner | None = None,
        memory: MemoryEngine | None = None,
        conv_memory: ConversationMemory | None = None,
        jarvis: JarvisCore | None = None,
    ) -> None:
        """Wire intelligence sources after initialization."""
        self._owner = owner
        self._scanner = scanner
        self._memory = memory
        self._conv_memory = conv_memory
        self._jarvis = jarvis

    def log_action(self, action: str, detail: str = "") -> None:
        self._action_log.append({
            "action": action, "detail": detail, "ts": time.time(),
        })
        if len(self._action_log) > 50:
            self._action_log = self._action_log[-50:]

    def get_fused_context(self, query: str = "") -> FusedContext:
        """Build a complete fused context from all intelligence sources."""
        ctx = FusedContext()
        now = datetime.now()

        # ── Situation layer ──
        ctx.hour = now.hour
        ctx.is_weekend = now.weekday() >= 5
        ctx.session_minutes = (time.time() - self._session_start) / 60

        if 5 <= now.hour < 12:
            ctx.time_of_day = "morning"
        elif 12 <= now.hour < 17:
            ctx.time_of_day = "afternoon"
        elif 17 <= now.hour < 21:
            ctx.time_of_day = "evening"
        else:
            ctx.time_of_day = "night"

        # ── Owner layer ──
        if self._owner:
            ctx.emotion = self._owner.emotion.primary
            ctx.emotion_trajectory = self._owner.emotion.trajectory
            ctx.energy = self._owner.anticipation.current_energy_level
            ctx.preferred_verbosity = self._owner.communication.preferred_response_length
            ctx.communication_style = (
                "formal" if self._owner.communication.formality_level > 0.6 else "casual"
            )
            if self._owner.topics.expertise_areas:
                top = sorted(
                    self._owner.topics.expertise_areas.items(),
                    key=lambda x: x[1], reverse=True,
                )[:3]
                ctx.expertise_hint = ", ".join(k for k, _ in top)

            if self._owner.context.active_projects:
                ctx.active_project = self._owner.context.active_projects[0].get("name", "")

        # ── System layer ──
        if self._scanner and self._scanner.last_scan:
            scan = self._scanner.last_scan
            ctx.system_health = scan.get("health", {}).get("overall", 100)
            sys_info = scan.get("system", {})
            ctx.is_on_battery = (
                sys_info.get("has_battery", False)
                and not sys_info.get("is_plugged", True)
            )
            ctx.battery_pct = sys_info.get("battery_percent", 100)
            
        try:
            from core.system_indexer import system_indexer
            ctx.system_index_summary = system_indexer.get_summary_for_llm()
        except ImportError:
            pass
            
        try:
            from voice.media_watcher import media_watcher
            if media_watcher.current_media.is_active:
                ctx.playing_media = media_watcher.current_media.summary()
        except ImportError:
            pass

        # ── Conversation layer ──
        conv = ctx.conversation
        if self._conv_memory:
            conv.turn_count = self._conv_memory.turn_count
            conv.topic_thread = self._conv_memory.active_topics
            conv.is_continuation = conv.turn_count > 0

            if conv.turn_count <= 3:
                conv.depth = "shallow"
            elif conv.turn_count <= 8:
                conv.depth = "medium"
            else:
                conv.depth = "deep"

        # ── Sentiment arc from emotion trajectory ──
        if self._owner:
            conv.sentiment_arc = self._owner.emotion.trajectory

        # ── Memory layer ──
        try:
            from core.l1_cache import l1_cache
            ctx.l1_cache_summary = l1_cache.get_summary_for_llm()
        except ImportError:
            pass

        # ── Action layer ──
        recent = self._action_log[-5:] if self._action_log else []
        ctx.recent_actions = [
            f"{a['action']}({a['detail']})" if a["detail"] else a["action"]
            for a in recent
        ]

        # ── Meta layer ──
        ctx.should_be_brief = (
            ctx.emotion in ("focused", "stressed")
            or ctx.preferred_verbosity == "short"
            or (ctx.hour >= 23 or ctx.hour < 5)
        )
        ctx.should_be_proactive = (
            ctx.emotion not in ("focused",)
            and ctx.energy != "low"
            and conv.turn_count < 3
        )
        ctx.context_richness = ctx.quality_score()

        self._last_fused_context = ctx
        return ctx

    def get_llm_context_block(self, query: str = "") -> str:
        """Generate a compact context block for LLM prompt injection.

        This replaces the manual stitching of context from multiple sources.
        The LLM sees a single, coherent picture of the world.
        """
        ctx = self.get_fused_context(query)
        lines: list[str] = []

        lines.append(
            f"[SITUATION] {ctx.time_of_day.title()}, "
            f"{'weekend' if ctx.is_weekend else 'weekday'} | "
            f"Session: {ctx.session_minutes:.0f}min | "
            f"System: {ctx.system_health}/100"
        )

        owner_parts = [f"Emotion: {ctx.emotion}"]
        if ctx.emotion_trajectory != "stable":
            owner_parts.append(f"trajectory: {ctx.emotion_trajectory}")
        owner_parts.append(f"Energy: {ctx.energy}")
        if ctx.expertise_hint:
            owner_parts.append(f"Expertise: {ctx.expertise_hint}")
        lines.append(f"[OWNER] {' | '.join(owner_parts)}")

        if ctx.active_project:
            lines.append(f"[PROJECT] {ctx.active_project}")

        if ctx.conversation.topic_thread:
            topics = ", ".join(ctx.conversation.topic_thread[:5])
            lines.append(f"[TOPICS] {topics}")

        if ctx.conversation.turn_count > 0:
            lines.append(
                f"[CONVERSATION] Depth: {ctx.conversation.depth} "
                f"({ctx.conversation.turn_count} turns)"
            )

        if ctx.should_be_brief:
            lines.append(
                f"[STYLE] Keep responses {ctx.preferred_verbosity}. "
                f"Owner prefers {ctx.communication_style} tone."
            )

        if ctx.is_on_battery and ctx.battery_pct < 30:
            lines.append(f"[POWER] Battery: {ctx.battery_pct:.0f}%")
            
        if ctx.system_index_summary:
            lines.append(f"[SYSTEM] {ctx.system_index_summary}")
            
        if ctx.playing_media:
            lines.append(f"[MEDIA] {ctx.playing_media}")

        if ctx.l1_cache_summary:
            lines.append(f"[FAST MEMORY] {ctx.l1_cache_summary}")

        if ctx.recent_actions:
            lines.append(f"[RECENT] {', '.join(ctx.recent_actions[-3:])}")

        return "\n".join(lines)

    def get_conversation_state(self) -> ConversationState:
        """Get deep conversation analysis for routing decisions."""
        ctx = self._last_fused_context or self.get_fused_context()
        return ctx.conversation

    @property
    def last_context(self) -> FusedContext | None:
        return self._last_fused_context
