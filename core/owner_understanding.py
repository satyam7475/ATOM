"""
ATOM -- Owner Understanding Engine (JARVIS-Level Personal Intelligence).

This is the module that makes ATOM truly JARVIS-like. It doesn't just track
habits -- it UNDERSTANDS the owner at a deep level:

    - Communication Style: How Boss talks, preferred vocabulary, sentence
      patterns, formality level, humor style
    - Emotional Intelligence: Tracks emotional trajectory over time, knows
      when Boss is stressed/happy/frustrated, adapts tone accordingly
    - Topic Expertise Map: Builds a knowledge graph of what Boss knows,
      what Boss is learning, what Boss cares about
    - Temporal Patterns: Knows work hours, break patterns, productive times,
      mood by time of day, weekend vs weekday behavior
    - Preference Learning: Automatically learns preferences from interactions
      (e.g., "Boss prefers dark mode", "Boss likes Python over Java")
    - Contextual Memory: Remembers ongoing projects, recent conversations,
      unfinished tasks, promises made
    - Relationship Intelligence: Tracks mentions of people, projects, and
      recurring themes to build contextual awareness
    - Anticipation Model: Predicts what Boss needs before they ask

This module consumes events from the entire system and builds a unified
"Owner Model" that other modules can query.

Contract: CognitiveModuleContract (start, stop, persist)
Owner: Satyam
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus

logger = logging.getLogger("atom.understanding")

_OWNER_MODEL_FILE = Path("logs/owner_model.json")
_MAX_INTERACTION_HISTORY = 500
_MAX_TOPIC_ENTRIES = 200


@dataclass
class EmotionalState:
    """Current inferred emotional state of the owner."""
    primary: str = "neutral"   # neutral, happy, stressed, frustrated, tired, focused, excited
    confidence: float = 0.5
    trajectory: str = "stable"  # improving, stable, declining
    last_updated: float = 0.0
    history: list[dict] = field(default_factory=list)


@dataclass
class CommunicationProfile:
    """How the owner communicates."""
    avg_sentence_length: float = 8.0
    formality_level: float = 0.3    # 0=very casual, 1=very formal
    uses_humor: bool = True
    common_phrases: list[str] = field(default_factory=list)
    vocabulary_richness: float = 0.5
    preferred_response_length: str = "medium"  # short, medium, long
    asks_questions_pct: float = 0.3
    gives_commands_pct: float = 0.5
    chitchat_pct: float = 0.2
    language_preferences: list[str] = field(default_factory=lambda: ["english"])


@dataclass
class TopicProfile:
    """What the owner knows and cares about."""
    expertise_areas: dict[str, float] = field(default_factory=dict)
    learning_areas: list[str] = field(default_factory=list)
    interests: list[str] = field(default_factory=list)
    mentioned_people: dict[str, int] = field(default_factory=dict)
    mentioned_projects: dict[str, int] = field(default_factory=dict)
    recent_topics: list[str] = field(default_factory=list)
    topic_frequency: dict[str, int] = field(default_factory=dict)


@dataclass
class TemporalProfile:
    """When and how the owner interacts."""
    peak_hours: list[int] = field(default_factory=list)
    avg_session_minutes: float = 30.0
    interactions_by_hour: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    interactions_by_weekday: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    typical_first_interaction: str = "09:00"
    typical_last_interaction: str = "23:00"
    weekend_active: bool = True
    break_pattern_minutes: list[float] = field(default_factory=list)


@dataclass
class ContextualMemory:
    """Ongoing context the owner hasn't closed."""
    active_projects: list[dict] = field(default_factory=list)
    unfinished_conversations: list[dict] = field(default_factory=list)
    pending_reminders: list[dict] = field(default_factory=list)
    last_discussed_topics: list[str] = field(default_factory=list)
    promises_made: list[dict] = field(default_factory=list)


@dataclass
class AnticipationModel:
    """Predictions about what the owner needs."""
    next_likely_action: str = ""
    next_likely_topic: str = ""
    current_energy_level: str = "normal"  # high, normal, low
    should_suggest_break: bool = False
    time_since_last_break_min: float = 0.0
    mood_appropriate_greeting: str = ""


class OwnerUnderstanding:
    """JARVIS-level owner intelligence engine.

    Consumes all interaction events and builds a deep, evolving model
    of the owner. Other modules query this to make ATOM feel like it
    truly knows the Boss.
    """

    def __init__(self, bus: AsyncEventBus | None = None,
                 config: dict | None = None) -> None:
        self._bus = bus
        self._config = config or {}
        self._owner_name = self._config.get("owner", {}).get("name", "Satyam")
        self._owner_title = self._config.get("owner", {}).get("title", "Boss")

        self.emotion = EmotionalState()
        self.communication = CommunicationProfile()
        self.topics = TopicProfile()
        self.temporal = TemporalProfile()
        self.context = ContextualMemory()
        self.anticipation = AnticipationModel()

        self._interaction_log: list[dict] = []
        self._session_start = time.time()
        self._last_interaction_time = time.time()
        self._total_interactions = 0
        self._word_counter: Counter = Counter()
        self._emotion_signals: list[tuple[float, str]] = []

        self._task: asyncio.Task | None = None
        self._shutdown = asyncio.Event()

        self._load_model()

    # ── Model Persistence ─────────────────────────────────────────

    def _load_model(self) -> None:
        try:
            if _OWNER_MODEL_FILE.exists():
                data = json.loads(_OWNER_MODEL_FILE.read_text(encoding="utf-8"))
                self._restore_from_dict(data)
                logger.info("Owner model loaded (%d total interactions)",
                            self._total_interactions)
        except Exception:
            logger.debug("Owner model load failed, starting fresh", exc_info=True)

    def _restore_from_dict(self, data: dict) -> None:
        self._total_interactions = data.get("total_interactions", 0)
        self._word_counter = Counter(data.get("word_frequencies", {}))

        comm = data.get("communication", {})
        self.communication.avg_sentence_length = comm.get("avg_sentence_length", 8.0)
        self.communication.formality_level = comm.get("formality_level", 0.3)
        self.communication.common_phrases = comm.get("common_phrases", [])
        self.communication.preferred_response_length = comm.get("preferred_response_length", "medium")
        self.communication.asks_questions_pct = comm.get("asks_questions_pct", 0.3)
        self.communication.gives_commands_pct = comm.get("gives_commands_pct", 0.5)

        topics = data.get("topics", {})
        self.topics.expertise_areas = topics.get("expertise_areas", {})
        self.topics.learning_areas = topics.get("learning_areas", [])
        self.topics.interests = topics.get("interests", [])
        self.topics.mentioned_people = topics.get("mentioned_people", {})
        self.topics.mentioned_projects = topics.get("mentioned_projects", {})
        self.topics.topic_frequency = topics.get("topic_frequency", {})

        temporal = data.get("temporal", {})
        self.temporal.peak_hours = temporal.get("peak_hours", [])
        self.temporal.avg_session_minutes = temporal.get("avg_session_minutes", 30.0)
        hours = temporal.get("interactions_by_hour", {})
        self.temporal.interactions_by_hour = defaultdict(int, {int(k): v for k, v in hours.items()})
        days = temporal.get("interactions_by_weekday", {})
        self.temporal.interactions_by_weekday = defaultdict(int, {int(k): v for k, v in days.items()})

        emotion = data.get("emotion", {})
        self.emotion.primary = emotion.get("primary", "neutral")
        self.emotion.trajectory = emotion.get("trajectory", "stable")

        ctx = data.get("context", {})
        self.context.active_projects = ctx.get("active_projects", [])
        self.context.last_discussed_topics = ctx.get("last_discussed_topics", [])

    def persist(self) -> None:
        try:
            _OWNER_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "total_interactions": self._total_interactions,
                "word_frequencies": dict(self._word_counter.most_common(500)),
                "communication": {
                    "avg_sentence_length": self.communication.avg_sentence_length,
                    "formality_level": self.communication.formality_level,
                    "common_phrases": self.communication.common_phrases[:50],
                    "preferred_response_length": self.communication.preferred_response_length,
                    "asks_questions_pct": round(self.communication.asks_questions_pct, 3),
                    "gives_commands_pct": round(self.communication.gives_commands_pct, 3),
                },
                "topics": {
                    "expertise_areas": dict(
                        sorted(self.topics.expertise_areas.items(),
                               key=lambda x: x[1], reverse=True)[:100]
                    ),
                    "learning_areas": self.topics.learning_areas[:30],
                    "interests": self.topics.interests[:50],
                    "mentioned_people": dict(
                        sorted(self.topics.mentioned_people.items(),
                               key=lambda x: x[1], reverse=True)[:50]
                    ),
                    "mentioned_projects": dict(
                        sorted(self.topics.mentioned_projects.items(),
                               key=lambda x: x[1], reverse=True)[:50]
                    ),
                    "topic_frequency": dict(
                        sorted(self.topics.topic_frequency.items(),
                               key=lambda x: x[1], reverse=True)[:_MAX_TOPIC_ENTRIES]
                    ),
                },
                "temporal": {
                    "peak_hours": self.temporal.peak_hours,
                    "avg_session_minutes": self.temporal.avg_session_minutes,
                    "interactions_by_hour": dict(self.temporal.interactions_by_hour),
                    "interactions_by_weekday": dict(self.temporal.interactions_by_weekday),
                },
                "emotion": {
                    "primary": self.emotion.primary,
                    "trajectory": self.emotion.trajectory,
                },
                "context": {
                    "active_projects": self.context.active_projects[:20],
                    "last_discussed_topics": self.context.last_discussed_topics[:20],
                },
                "last_persisted": time.time(),
            }
            _OWNER_MODEL_FILE.write_text(
                json.dumps(data, indent=2, default=str), encoding="utf-8",
            )
        except Exception:
            logger.debug("Owner model persist failed", exc_info=True)

    # ── Event Processing ──────────────────────────────────────────

    def process_speech(self, text: str) -> None:
        """Analyze every speech input to build the owner model."""
        if not text or len(text) < 2:
            return

        now = time.time()
        self._total_interactions += 1
        self._last_interaction_time = now

        self._update_temporal(now)
        self._update_communication(text)
        self._update_topics(text)
        self._update_emotion_from_text(text)
        self._update_context(text)
        self._update_anticipation()

        self._interaction_log.append({
            "time": now,
            "text": text[:200],
            "emotion": self.emotion.primary,
            "hour": datetime.now().hour,
        })
        if len(self._interaction_log) > _MAX_INTERACTION_HISTORY:
            self._interaction_log = self._interaction_log[-_MAX_INTERACTION_HISTORY:]

    def process_response_feedback(self, query: str, response: str,
                                   was_helpful: bool = True) -> None:
        """Learn from query-response pairs about what the owner values."""
        if was_helpful and len(response) > 50:
            words = len(response.split())
            if words > 100:
                self.communication.preferred_response_length = "long"
            elif words < 30:
                self.communication.preferred_response_length = "short"
            else:
                self.communication.preferred_response_length = "medium"

    def process_emotion_signal(self, emotion: str, confidence: float = 0.5) -> None:
        """Accept external emotion signals (from voice emotion detector etc)."""
        self._emotion_signals.append((time.time(), emotion))
        if len(self._emotion_signals) > 100:
            self._emotion_signals = self._emotion_signals[-100:]

        if confidence > self.emotion.confidence:
            old = self.emotion.primary
            self.emotion.primary = emotion
            self.emotion.confidence = confidence
            self.emotion.last_updated = time.time()

            if old != emotion:
                improving = {"frustrated": "neutral", "stressed": "neutral",
                             "tired": "neutral", "neutral": "happy",
                             "happy": "excited"}
                declining = {"excited": "happy", "happy": "neutral",
                             "neutral": "stressed", "stressed": "frustrated"}
                if improving.get(old) == emotion:
                    self.emotion.trajectory = "improving"
                elif declining.get(old) == emotion:
                    self.emotion.trajectory = "declining"
                else:
                    self.emotion.trajectory = "stable"

    # ── Internal Analysis ─────────────────────────────────────────

    def _update_temporal(self, now: float) -> None:
        dt = datetime.fromtimestamp(now)
        self.temporal.interactions_by_hour[dt.hour] += 1
        self.temporal.interactions_by_weekday[dt.weekday()] += 1

        if self.temporal.interactions_by_hour:
            sorted_hours = sorted(
                self.temporal.interactions_by_hour.items(),
                key=lambda x: x[1], reverse=True,
            )
            self.temporal.peak_hours = [h for h, _ in sorted_hours[:5]]

    def _update_communication(self, text: str) -> None:
        words = text.split()
        word_count = len(words)

        alpha = 0.05
        self.communication.avg_sentence_length = (
            (1 - alpha) * self.communication.avg_sentence_length + alpha * word_count
        )

        for w in words:
            w_lower = w.lower().strip(".,!?;:'\"")
            if len(w_lower) > 2:
                self._word_counter[w_lower] += 1

        formal_indicators = {"please", "kindly", "would", "could", "shall", "regarding"}
        casual_indicators = {"hey", "yo", "yep", "nah", "gonna", "wanna", "kinda", "lol"}
        formal_count = sum(1 for w in words if w.lower() in formal_indicators)
        casual_count = sum(1 for w in words if w.lower() in casual_indicators)
        if formal_count > casual_count:
            self.communication.formality_level = min(1.0, self.communication.formality_level + 0.01)
        elif casual_count > formal_count:
            self.communication.formality_level = max(0.0, self.communication.formality_level - 0.01)

        is_question = text.strip().endswith("?") or text.lower().startswith(
            ("what", "how", "why", "when", "where", "who", "can", "could",
             "would", "is", "are", "do", "does", "will"),
        )
        is_command = text.lower().startswith(
            ("open", "close", "set", "turn", "play", "stop", "show",
             "create", "delete", "search", "find", "run", "kill", "lock"),
        )

        n = self._total_interactions
        if is_question:
            self.communication.asks_questions_pct = (
                (self.communication.asks_questions_pct * (n - 1) + 1) / n
            )
        if is_command:
            self.communication.gives_commands_pct = (
                (self.communication.gives_commands_pct * (n - 1) + 1) / n
            )

        top_phrases = self._word_counter.most_common(20)
        self.communication.common_phrases = [w for w, c in top_phrases if c >= 3]

    def _update_topics(self, text: str) -> None:
        text_lower = text.lower()
        words = text_lower.split()

        topic_keywords = {
            "programming": {"code", "python", "javascript", "function", "class", "bug",
                           "debug", "api", "server", "database", "git", "deploy",
                           "react", "rust", "java", "typescript", "node", "django"},
            "ai_ml": {"model", "training", "neural", "llm", "gpt", "transformer",
                     "embedding", "vector", "inference", "gpu", "cuda", "gguf",
                     "machine learning", "deep learning", "ai"},
            "system_admin": {"process", "cpu", "ram", "disk", "network", "firewall",
                           "service", "driver", "registry", "permission", "admin"},
            "creative": {"design", "color", "font", "layout", "image", "video",
                        "animation", "music", "art", "creative", "ui", "ux"},
            "productivity": {"task", "goal", "schedule", "meeting", "email",
                           "calendar", "deadline", "project", "plan", "organize"},
            "gaming": {"game", "play", "steam", "fps", "rpg", "controller",
                      "graphics", "performance", "settings"},
            "research": {"research", "paper", "study", "data", "analysis",
                        "statistics", "experiment", "hypothesis"},
        }

        word_set = set(words)
        for topic, keywords in topic_keywords.items():
            overlap = word_set & keywords
            if overlap:
                score = len(overlap) / len(words) * 10
                current = self.topics.expertise_areas.get(topic, 0)
                self.topics.expertise_areas[topic] = min(10.0, current + score * 0.1)
                self.topics.topic_frequency[topic] = self.topics.topic_frequency.get(topic, 0) + 1

        for word in words:
            if len(word) > 4:
                self.topics.topic_frequency[word] = self.topics.topic_frequency.get(word, 0) + 1

        learning_markers = ["learn", "tutorial", "how to", "understand",
                           "explain", "teach me", "what is"]
        for marker in learning_markers:
            if marker in text_lower:
                topic_after = text_lower.split(marker, 1)[-1].strip()[:50]
                if topic_after and topic_after not in self.topics.learning_areas:
                    self.topics.learning_areas.append(topic_after)
                    if len(self.topics.learning_areas) > 30:
                        self.topics.learning_areas = self.topics.learning_areas[-30:]

        name_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')
        for match in name_pattern.finditer(text):
            name = match.group()
            if len(name) > 2 and name not in ("Boss", "ATOM", "Hey", "The", "This", "That"):
                self.topics.mentioned_people[name] = self.topics.mentioned_people.get(name, 0) + 1

        self.topics.recent_topics = list(
            sorted(self.topics.topic_frequency.items(),
                   key=lambda x: x[1], reverse=True)[:10]
        )

    def _update_emotion_from_text(self, text: str) -> None:
        text_lower = text.lower()

        emotion_markers = {
            "frustrated": {"damn", "ugh", "annoying", "broken", "stupid", "doesn't work",
                          "not working", "failed", "error", "wrong", "hate"},
            "stressed": {"urgent", "deadline", "hurry", "quick", "asap", "immediately",
                        "worried", "pressure", "overwhelmed"},
            "happy": {"great", "awesome", "perfect", "love", "excellent", "amazing",
                     "nice", "cool", "wonderful", "thanks", "thank you"},
            "excited": {"wow", "incredible", "fantastic", "yes!", "finally",
                       "brilliant", "epic", "lets go"},
            "tired": {"tired", "exhausted", "sleepy", "long day", "boring",
                     "yawn", "rest", "break", "enough"},
            "focused": {"focus", "concentrate", "working on", "deep work",
                       "coding", "building", "implementing"},
        }

        detected = "neutral"
        max_score = 0
        for emotion, markers in emotion_markers.items():
            score = sum(1 for m in markers if m in text_lower)
            if score > max_score:
                max_score = score
                detected = emotion

        if max_score > 0:
            self.process_emotion_signal(detected, min(0.9, 0.3 + max_score * 0.2))

    def _update_context(self, text: str) -> None:
        text_lower = text.lower()

        project_markers = ["working on", "project", "building", "developing",
                          "creating", "coding"]
        for marker in project_markers:
            if marker in text_lower:
                topic = text_lower.split(marker, 1)[-1].strip()[:60]
                if topic:
                    existing = [p for p in self.context.active_projects
                                if p.get("name", "").lower() == topic.lower()]
                    if not existing:
                        self.context.active_projects.append({
                            "name": topic,
                            "first_mentioned": time.time(),
                            "last_mentioned": time.time(),
                            "mention_count": 1,
                        })
                    else:
                        existing[0]["last_mentioned"] = time.time()
                        existing[0]["mention_count"] = existing[0].get("mention_count", 0) + 1

        if len(self.context.active_projects) > 20:
            self.context.active_projects.sort(
                key=lambda p: p.get("last_mentioned", 0), reverse=True,
            )
            self.context.active_projects = self.context.active_projects[:20]

        words = [w for w in text_lower.split() if len(w) > 4]
        for word in words[:5]:
            if word not in self.context.last_discussed_topics:
                self.context.last_discussed_topics.append(word)
        if len(self.context.last_discussed_topics) > 20:
            self.context.last_discussed_topics = self.context.last_discussed_topics[-20:]

    def _update_anticipation(self) -> None:
        dt = datetime.now()
        hour = dt.hour

        if hour < 6:
            self.anticipation.current_energy_level = "low"
        elif hour < 10:
            self.anticipation.current_energy_level = "high"
        elif hour < 14:
            self.anticipation.current_energy_level = "normal"
        elif hour < 17:
            self.anticipation.current_energy_level = "normal"
        elif hour < 21:
            self.anticipation.current_energy_level = "normal"
        else:
            self.anticipation.current_energy_level = "low"

        if self.emotion.primary in ("tired", "stressed", "frustrated"):
            self.anticipation.current_energy_level = "low"

        session_minutes = (time.time() - self._session_start) / 60
        if session_minutes > 90:
            self.anticipation.should_suggest_break = True
            self.anticipation.time_since_last_break_min = session_minutes
        else:
            self.anticipation.should_suggest_break = False

        if self.topics.recent_topics:
            self.anticipation.next_likely_topic = (
                self.topics.recent_topics[0] if isinstance(self.topics.recent_topics[0], str)
                else self.topics.recent_topics[0][0]
            )

        self.anticipation.mood_appropriate_greeting = self._generate_mood_greeting()

    def _generate_mood_greeting(self) -> str:
        dt = datetime.now()
        hour = dt.hour
        emotion = self.emotion.primary

        if hour < 12:
            time_greeting = "Good morning"
        elif hour < 17:
            time_greeting = "Good afternoon"
        elif hour < 21:
            time_greeting = "Good evening"
        else:
            time_greeting = "Working late"

        mood_additions = {
            "happy": "You seem to be in a great mood.",
            "stressed": "I can tell it's been intense. I'm here to help.",
            "frustrated": "Let me take care of the heavy lifting.",
            "tired": "Want me to handle the routine tasks?",
            "excited": "Love the energy! What are we building?",
            "focused": "I'll keep things efficient for you.",
            "neutral": "Ready when you are.",
        }

        return f"{time_greeting}, {self._owner_title}. {mood_additions.get(emotion, '')}"

    # ── Public Query API ──────────────────────────────────────────

    def get_owner_context_for_llm(self) -> str:
        """Generate a compact owner understanding context for LLM prompt injection."""
        lines = []

        lines.append(f"[OWNER STATE] Emotion: {self.emotion.primary} "
                     f"(trajectory: {self.emotion.trajectory}) | "
                     f"Energy: {self.anticipation.current_energy_level}")

        if self.topics.expertise_areas:
            top = sorted(self.topics.expertise_areas.items(),
                         key=lambda x: x[1], reverse=True)[:5]
            areas = ", ".join(f"{k}({v:.1f})" for k, v in top)
            lines.append(f"[EXPERTISE] {areas}")

        if self.topics.learning_areas:
            lines.append(f"[LEARNING] {', '.join(self.topics.learning_areas[:5])}")

        if self.context.active_projects:
            projects = ", ".join(p["name"] for p in self.context.active_projects[:5])
            lines.append(f"[ACTIVE PROJECTS] {projects}")

        if self.context.last_discussed_topics:
            lines.append(f"[RECENT TOPICS] {', '.join(self.context.last_discussed_topics[:8])}")

        lines.append(
            f"[COMMUNICATION] Prefers {self.communication.preferred_response_length} responses | "
            f"Formality: {'formal' if self.communication.formality_level > 0.6 else 'casual'} | "
            f"Questions: {self.communication.asks_questions_pct:.0%}"
        )

        if self.anticipation.should_suggest_break:
            lines.append(
                f"[WELLBEING] Working for {self.anticipation.time_since_last_break_min:.0f}min "
                f"-- consider suggesting a break"
            )

        return "\n".join(lines)

    def get_personality_adjustment(self) -> dict[str, Any]:
        """Return personality adjustments based on current owner state."""
        adjustments: dict[str, Any] = {
            "tone": "normal",
            "verbosity": self.communication.preferred_response_length,
            "formality": "casual" if self.communication.formality_level < 0.5 else "formal",
            "humor": self.communication.uses_humor,
            "proactivity": "normal",
        }

        if self.emotion.primary == "frustrated":
            adjustments["tone"] = "supportive"
            adjustments["verbosity"] = "short"
            adjustments["humor"] = False
            adjustments["proactivity"] = "high"
        elif self.emotion.primary == "stressed":
            adjustments["tone"] = "calm"
            adjustments["verbosity"] = "short"
            adjustments["proactivity"] = "high"
        elif self.emotion.primary == "tired":
            adjustments["tone"] = "gentle"
            adjustments["verbosity"] = "short"
            adjustments["proactivity"] = "low"
        elif self.emotion.primary == "happy":
            adjustments["tone"] = "enthusiastic"
            adjustments["humor"] = True
        elif self.emotion.primary == "excited":
            adjustments["tone"] = "enthusiastic"
            adjustments["humor"] = True
            adjustments["proactivity"] = "high"
        elif self.emotion.primary == "focused":
            adjustments["tone"] = "efficient"
            adjustments["verbosity"] = "short"
            adjustments["proactivity"] = "low"

        return adjustments

    def get_owner_summary(self) -> str:
        """Human-readable owner understanding summary for voice output."""
        parts = [
            f"I've learned from {self._total_interactions} interactions with you, {self._owner_title}.",
        ]

        if self.topics.expertise_areas:
            top = sorted(self.topics.expertise_areas.items(),
                         key=lambda x: x[1], reverse=True)[:3]
            areas = " and ".join(k for k, _ in top)
            parts.append(f"Your top areas are {areas}.")

        if self.temporal.peak_hours:
            peak = self.temporal.peak_hours[0]
            parts.append(f"You're most active around {peak}:00.")

        parts.append(f"Current mood: {self.emotion.primary}.")

        if self.context.active_projects:
            parts.append(
                f"You're working on {len(self.context.active_projects)} "
                f"project{'s' if len(self.context.active_projects) > 1 else ''}: "
                f"{self.context.active_projects[0]['name']}."
            )

        return " ".join(parts)

    def knows_about(self, topic: str) -> float:
        """Check how much the owner knows about a topic (0.0 - 10.0)."""
        return self.topics.expertise_areas.get(topic.lower(), 0.0)

    def get_relationship_context(self, name: str) -> str:
        """Get context about a mentioned person."""
        count = self.topics.mentioned_people.get(name, 0)
        if count == 0:
            return ""
        if count > 10:
            return f"'{name}' is frequently mentioned ({count} times)"
        if count > 3:
            return f"'{name}' has come up a few times ({count})"
        return f"'{name}' was mentioned {count} time{'s' if count > 1 else ''}"

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        if self._bus:
            self._bus.on("speech_final", self._on_speech)
            self._bus.on("cursor_response", self._on_response)
            self._bus.on("user_emotion_detected", self._on_emotion)
            self._bus.on("intent_classified", self._on_intent)

        self._task = asyncio.ensure_future(self._background_loop())
        logger.info("Owner understanding engine started (%d prior interactions)",
                     self._total_interactions)

    def stop(self) -> None:
        self._shutdown.set()
        if self._task and not self._task.done():
            self._task.cancel()
        self.persist()

    async def _on_speech(self, text: str = "", **_kw) -> None:
        if text:
            self.process_speech(text)

    async def _on_response(self, query: str = "", response: str = "", **_kw) -> None:
        if query and response:
            self.process_response_feedback(query, response)

    async def _on_emotion(self, emotion: str = "", confidence: float = 0.5, **_kw) -> None:
        if emotion:
            self.process_emotion_signal(emotion, confidence)

    async def _on_intent(self, intent: str = "", **_kw) -> None:
        if intent and intent not in ("fallback", "greeting", "thanks"):
            self.topics.topic_frequency[intent] = (
                self.topics.topic_frequency.get(intent, 0) + 1
            )

    async def _background_loop(self) -> None:
        """Periodic persistence and model maintenance."""
        while not self._shutdown.is_set():
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=300.0)
                break
            except asyncio.TimeoutError:
                pass

            try:
                self.persist()

                cutoff = time.time() - 86400 * 30
                self.context.active_projects = [
                    p for p in self.context.active_projects
                    if p.get("last_mentioned", 0) > cutoff
                ]
            except Exception:
                logger.debug("Owner understanding maintenance error", exc_info=True)
