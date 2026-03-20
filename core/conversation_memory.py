"""
ATOM v15 -- Unified conversation memory.

Combines short-term turn tracking, topic extraction, and prior-turn
session context (previously in session_context.py) into a single module.
"""

from __future__ import annotations

import re
import time
from typing import Any

_TOPIC_KEYWORDS = re.compile(
    r"\b(deploy|debug|error|config|install|review|test|build|"
    r"backup|monitor|performance|api|database|network|meeting|"
    r"email|schedule|reminder|browser|file|folder|"
    r"docker|git|python|java|node|react|sql|kubernetes|"
    r"cpu|ram|disk|battery|process|service)\b",
    re.I,
)

_MAX_TOPIC_ENTRIES = 12


class ConversationTurn:
    __slots__ = ("query", "intent", "response_snippet", "ts", "topics")

    def __init__(
        self,
        query: str,
        intent: str,
        response_snippet: str,
        ts: float | None = None,
        topics: list[str] | None = None,
    ) -> None:
        self.query = query[:200]
        self.intent = intent
        self.response_snippet = response_snippet[:120]
        self.ts = ts or time.time()
        self.topics = topics or []


class ConversationMemory:
    """Short-term conversation memory with topic tracking and prior-turn context."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = (config or {}).get("conversation_memory", {}) or {}
        self._max_turns: int = int(cfg.get("max_turns", 10))
        self._turns: list[ConversationTurn] = []
        self._topic_map: dict[str, float] = {}

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

    def summary_for_prompt(self) -> str:
        """One-line prior-turn summary for LLM context injection."""
        if not self._session_enabled or not self._prev_turn.get("intent"):
            return ""
        q = self._prev_turn.get("q", "")
        it = self._prev_turn.get("intent", "")
        act = self._prev_turn.get("action", "")
        tail = f" (action {act})" if act else ""
        return f"Prior turn: «{q}» → {it}{tail}."

    # ── Turn recording ────────────────────────────────────────────────

    def record(self, query: str, intent: str, response: str) -> None:
        topics = _extract_topics(query + " " + response)
        turn = ConversationTurn(query, intent, response, topics=topics)
        self._turns.append(turn)
        if len(self._turns) > self._max_turns:
            self._turns = self._turns[-self._max_turns:]
        now = time.time()
        for t in topics:
            self._topic_map[t] = now
        if len(self._topic_map) > _MAX_TOPIC_ENTRIES * 2:
            sorted_topics = sorted(self._topic_map.items(), key=lambda x: x[1], reverse=True)
            self._topic_map = dict(sorted_topics[:_MAX_TOPIC_ENTRIES])

    @property
    def active_topics(self) -> list[str]:
        now = time.time()
        recent = [(t, ts) for t, ts in self._topic_map.items() if now - ts < 600]
        recent.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in recent[:5]]

    @property
    def turn_count(self) -> int:
        return len(self._turns)

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
        parts = lines
        if topic_line:
            parts = [topic_line] + parts
        return "\n".join(parts)

    def get_pairs(self) -> list[tuple[str, str]]:
        """Return (query, response) pairs for StructuredPromptBuilder compatibility."""
        return [(t.query, t.response_snippet) for t in self._turns]


def _extract_topics(text: str) -> list[str]:
    found = _TOPIC_KEYWORDS.findall(text.lower())
    seen: set[str] = set()
    unique: list[str] = []
    for t in found:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique[:5]
