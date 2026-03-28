"""
ATOM -- Intelligent Conversation Memory.

Upgraded from v15's simple rolling window to a conversation intelligence
module that tracks:

    1. TOPIC THREADING -- groups turns into topic threads, detects switches
    2. SENTIMENT TRACKING -- per-turn sentiment arc (improving/declining/stable)
    3. CONVERSATION STATE -- depth, engagement level, frustration detection
    4. INTENT PATTERNS -- repeated intent detection for loop avoidance
    5. TURN QUALITY -- tracks whether queries got satisfactory responses

Still lightweight (no ML), using keyword heuristics and turn metadata.

Contract: record(), active_topics, summary_for_prompt(), get_pairs()
Owner: Satyam
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

_TOPIC_KEYWORDS = re.compile(
    r"\b(deploy|debug|error|config|install|review|test|build|"
    r"backup|monitor|performance|api|database|network|meeting|"
    r"email|schedule|reminder|browser|file|folder|"
    r"docker|git|python|java|node|react|sql|kubernetes|"
    r"cpu|ram|disk|battery|process|service|"
    r"security|memory|cache|llm|brain|model|prompt|"
    r"workflow|automation|script|dashboard|log|"
    r"atom|jarvis|system|server|container|"
    r"music|youtube|weather|timer|screenshot|clipboard)\b",
    re.I,
)

_SENTIMENT_POSITIVE = re.compile(
    r"\b(thanks|thank you|great|perfect|awesome|nice|good job|"
    r"excellent|love it|well done|exactly|correct|yes)\b", re.I,
)
_SENTIMENT_NEGATIVE = re.compile(
    r"\b(wrong|bad|broken|fail|error|crash|bug|not working|"
    r"doesn't work|can't|unable|frustrated|annoying|ugh|"
    r"damn|useless|terrible|no|nope|incorrect|again)\b", re.I,
)

_MAX_TOPIC_ENTRIES = 20


@dataclass
class TopicThread:
    """A cluster of related turns around a common topic."""
    topic: str
    turn_indices: list[int] = field(default_factory=list)
    started_at: float = 0.0
    last_active: float = 0.0
    sentiment_score: float = 0.0   # -1.0 to 1.0
    is_resolved: bool = False

    @property
    def duration_s(self) -> float:
        return self.last_active - self.started_at if self.started_at else 0.0

    @property
    def depth(self) -> int:
        return len(self.turn_indices)


class ConversationTurn:
    __slots__ = (
        "query", "intent", "response_snippet", "ts", "topics",
        "sentiment", "was_successful",
    )

    def __init__(
        self,
        query: str,
        intent: str,
        response_snippet: str,
        ts: float | None = None,
        topics: list[str] | None = None,
        sentiment: float = 0.0,
        was_successful: bool = True,
    ) -> None:
        self.query = query[:200]
        self.intent = intent
        self.response_snippet = response_snippet[:120]
        self.ts = ts or time.time()
        self.topics = topics or []
        self.sentiment = sentiment
        self.was_successful = was_successful


class ConversationMemory:
    """Intelligent conversation memory with topic threading, sentiment
    tracking, and conversation state analysis.

    v21 upgrades:
    - Topic threads group related turns and track per-topic sentiment
    - Sentiment arc tracks whether the conversation is going well
    - Intent repetition detection for frustration/loop avoidance
    - Richer summary_for_prompt with thread context
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = (config or {}).get("conversation_memory", {}) or {}
        self._max_turns: int = int(cfg.get("max_turns", 20))
        self._turns: list[ConversationTurn] = []
        self._topic_map: dict[str, float] = {}

        # Topic threading
        self._threads: dict[str, TopicThread] = {}
        self._active_thread: str = ""

        # Sentiment tracking
        self._sentiment_history: list[float] = []
        self._overall_sentiment: float = 0.0

        # Intent patterns
        self._recent_intents: list[str] = []
        self._intent_repeat_count: int = 0

        # Session context
        sess = (config or {}).get("session", {}) or {}
        self._session_enabled: bool = bool(sess.get("enabled", True))
        self._max_q: int = int(sess.get("max_query_snippet_chars", 120))
        self._prev_turn: dict[str, str] = {}
        self._curr_turn: dict[str, str] = {}

    # ── Session context (prior-turn tracking) ─────────────────────────

    def on_new_user_query(self, clean_text: str) -> None:
        """Call at the start of routing a new utterance."""
        if not self._session_enabled:
            return
        if self._curr_turn.get("intent"):
            self._prev_turn = dict(self._curr_turn)
        q = (clean_text or "").strip()
        if len(q) > self._max_q:
            q = q[: self._max_q] + "…"
        self._curr_turn = {"q": q, "intent": "", "action": ""}

    def set_classified(self, intent: str, action: str | None) -> None:
        """Call after intent engine returns."""
        if not self._session_enabled:
            return
        self._curr_turn["intent"] = intent or ""
        self._curr_turn["action"] = (action or "").strip()

        # v21: Intent repetition tracking
        if intent and intent not in ("empty", "greeting", "thanks"):
            self._recent_intents.append(intent)
            if len(self._recent_intents) > 10:
                self._recent_intents = self._recent_intents[-10:]
            self._update_intent_repeat(intent)

    def _update_intent_repeat(self, intent: str) -> None:
        """Detect repeated identical intents (possible frustration loop)."""
        if len(self._recent_intents) >= 3:
            last_three = self._recent_intents[-3:]
            if all(i == intent for i in last_three):
                self._intent_repeat_count += 1
            else:
                self._intent_repeat_count = 0

    def summary_for_prompt(self) -> str:
        """Rich prior-turn + thread summary for LLM context injection."""
        if not self._session_enabled:
            return ""

        parts: list[str] = []

        if self._prev_turn.get("intent"):
            q = self._prev_turn.get("q", "")
            it = self._prev_turn.get("intent", "")
            act = self._prev_turn.get("action", "")
            tail = f" (action {act})" if act else ""
            parts.append(f"Prior turn: «{q}» → {it}{tail}.")

        # v21: Active thread context
        if self._active_thread and self._active_thread in self._threads:
            thread = self._threads[self._active_thread]
            parts.append(
                f"Current topic thread: {thread.topic} "
                f"({thread.depth} turns, "
                f"sentiment: {self._sentiment_label(thread.sentiment_score)})."
            )

        # v21: Sentiment arc
        arc = self.sentiment_arc
        if arc != "stable":
            parts.append(f"Conversation sentiment: {arc}.")

        # v21: Frustration warning
        if self._intent_repeat_count >= 2:
            parts.append(
                "WARNING: User is repeating the same intent -- "
                "likely not getting the desired result."
            )

        return " ".join(parts) if parts else ""

    # ── Turn recording (v21: with sentiment and threading) ────────────

    def record(self, query: str, intent: str, response: str) -> None:
        """Record a turn with sentiment analysis and topic threading."""
        topics = _extract_topics(query + " " + response)
        sentiment = _compute_sentiment(query + " " + response)

        turn = ConversationTurn(
            query, intent, response,
            topics=topics, sentiment=sentiment,
        )
        self._turns.append(turn)
        if len(self._turns) > self._max_turns:
            self._turns = self._turns[-self._max_turns:]

        now = time.time()
        for t in topics:
            self._topic_map[t] = now
        if len(self._topic_map) > _MAX_TOPIC_ENTRIES * 2:
            sorted_topics = sorted(
                self._topic_map.items(), key=lambda x: x[1], reverse=True,
            )
            self._topic_map = dict(sorted_topics[:_MAX_TOPIC_ENTRIES])

        # v21: Sentiment tracking
        self._sentiment_history.append(sentiment)
        if len(self._sentiment_history) > 20:
            self._sentiment_history = self._sentiment_history[-20:]
        self._overall_sentiment = (
            sum(self._sentiment_history) / len(self._sentiment_history)
        )

        # v21: Topic threading
        self._update_threads(topics, len(self._turns) - 1, sentiment, now)

    def _update_threads(
        self, topics: list[str], turn_idx: int,
        sentiment: float, now: float,
    ) -> None:
        """Update topic threads with new turn data."""
        if not topics:
            return

        primary_topic = topics[0]

        if primary_topic in self._threads:
            thread = self._threads[primary_topic]
            thread.turn_indices.append(turn_idx)
            thread.last_active = now
            thread.sentiment_score = (
                thread.sentiment_score * 0.7 + sentiment * 0.3
            )
        else:
            self._threads[primary_topic] = TopicThread(
                topic=primary_topic,
                turn_indices=[turn_idx],
                started_at=now,
                last_active=now,
                sentiment_score=sentiment,
            )

        self._active_thread = primary_topic

        # Prune old threads
        if len(self._threads) > 15:
            sorted_threads = sorted(
                self._threads.items(),
                key=lambda x: x[1].last_active,
            )
            for key, _ in sorted_threads[:5]:
                del self._threads[key]

    # ── Sentiment analysis ────────────────────────────────────────────

    @property
    def sentiment_arc(self) -> str:
        """Determine if conversation sentiment is improving, declining, or stable."""
        if len(self._sentiment_history) < 3:
            return "stable"

        recent = self._sentiment_history[-3:]
        earlier = self._sentiment_history[-6:-3] if len(self._sentiment_history) >= 6 else self._sentiment_history[:3]

        recent_avg = sum(recent) / len(recent)
        earlier_avg = sum(earlier) / len(earlier)
        delta = recent_avg - earlier_avg

        if delta > 0.15:
            return "improving"
        if delta < -0.15:
            return "declining"
        return "stable"

    @property
    def is_frustrated(self) -> bool:
        """Detect if the user appears frustrated based on sentiment + repeats."""
        return (
            self._intent_repeat_count >= 2
            or self._overall_sentiment < -0.3
            or (len(self._sentiment_history) >= 3
                and all(s < -0.1 for s in self._sentiment_history[-3:]))
        )

    @staticmethod
    def _sentiment_label(score: float) -> str:
        if score > 0.2:
            return "positive"
        if score < -0.2:
            return "negative"
        return "neutral"

    # ── Topic queries ─────────────────────────────────────────────────

    @property
    def active_topics(self) -> list[str]:
        now = time.time()
        recent = [(t, ts) for t, ts in self._topic_map.items() if now - ts < 600]
        recent.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in recent[:5]]

    @property
    def active_threads(self) -> list[TopicThread]:
        """Get active topic threads sorted by recency."""
        now = time.time()
        active = [
            t for t in self._threads.values()
            if now - t.last_active < 600
        ]
        active.sort(key=lambda t: t.last_active, reverse=True)
        return active

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def intent_repeat_count(self) -> int:
        return self._intent_repeat_count

    @property
    def overall_sentiment(self) -> float:
        return self._overall_sentiment

    # ── Summaries ─────────────────────────────────────────────────────

    def recent_summary(self, max_turns: int = 5) -> str:
        """Compact summary string for LLM context injection."""
        if not self._turns:
            return ""
        lines: list[str] = []
        for turn in self._turns[-max_turns:]:
            line = f"Q: {turn.query}"
            if turn.response_snippet:
                line += f" → A: {turn.response_snippet}"
            lines.append(line)
        topics = self.active_topics
        topic_line = f"Active topics: {', '.join(topics)}" if topics else ""

        # v21: Include sentiment arc
        sentiment_info = ""
        arc = self.sentiment_arc
        if arc != "stable":
            sentiment_info = f"Sentiment trend: {arc}"

        parts = lines
        if topic_line:
            parts = [topic_line] + parts
        if sentiment_info:
            parts = [sentiment_info] + parts
        return "\n".join(parts)

    def thread_summary(self) -> str:
        """Summary of active topic threads for deeper context."""
        threads = self.active_threads
        if not threads:
            return ""
        lines = []
        for t in threads[:3]:
            sentiment = self._sentiment_label(t.sentiment_score)
            lines.append(
                f"Thread '{t.topic}': {t.depth} turns, "
                f"sentiment {sentiment}"
            )
        return " | ".join(lines)

    def get_pairs(self) -> list[tuple[str, str]]:
        """Return (query, response) pairs for StructuredPromptBuilder compatibility."""
        return [(t.query, t.response_snippet) for t in self._turns]

    def get_conversation_state(self) -> dict[str, Any]:
        """Full conversation state for ContextFusionEngine consumption."""
        return {
            "turn_count": self.turn_count,
            "active_topics": self.active_topics,
            "active_threads": [
                {"topic": t.topic, "depth": t.depth,
                 "sentiment": self._sentiment_label(t.sentiment_score)}
                for t in self.active_threads[:3]
            ],
            "sentiment_arc": self.sentiment_arc,
            "overall_sentiment": round(self._overall_sentiment, 2),
            "is_frustrated": self.is_frustrated,
            "intent_repeat_count": self._intent_repeat_count,
            "depth": (
                "deep" if self.turn_count > 8
                else "medium" if self.turn_count > 3
                else "shallow"
            ),
        }


def _extract_topics(text: str) -> list[str]:
    found = _TOPIC_KEYWORDS.findall(text.lower())
    seen: set[str] = set()
    unique: list[str] = []
    for t in found:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique[:5]


def _compute_sentiment(text: str) -> float:
    """Lightweight sentiment score from -1.0 (negative) to 1.0 (positive)."""
    pos_matches = len(_SENTIMENT_POSITIVE.findall(text))
    neg_matches = len(_SENTIMENT_NEGATIVE.findall(text))
    total = pos_matches + neg_matches
    if total == 0:
        return 0.0
    return (pos_matches - neg_matches) / total
