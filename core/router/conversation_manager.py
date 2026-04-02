"""
ATOM -- Conversation Manager

Handles conversation history, pronoun resolution, repetition detection,
smart acknowledgments, follow-up suggestions, and intent chaining.
"""

from __future__ import annotations

import re
import time
from typing import Any

# ── Regex patterns for conversational continuity ──────────────────────────────
_FILLER = re.compile(
    r"\b(um+|uh+|hmm+|ah+|oh+|like|actually|basically|"
    r"you know|i mean|so+|well|okay so|right so|"
    r"please|kindly)\b",
    re.I,
)
_MULTI_SPACE = re.compile(r"\s+")

_DANGLING_PRONOUN = re.compile(
    r"\b(it|that|this|there|those|these|them)\b", re.I)
_STOP_VERBS = frozenset({
    "is", "are", "was", "were", "do", "does", "did", "can", "could",
    "will", "would", "should", "has", "have", "had", "be", "been",
    "tell", "show", "give", "get", "make", "let", "know", "say",
    "explain", "search", "find", "open", "close", "check", "set",
})
_STOP_PREPS = frozenset({
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "with",
    "about", "from", "by", "as", "into", "through", "my", "your",
    "me", "i", "and", "or", "but", "so", "if", "what", "why", "how",
    "when", "where", "who", "which", "please",
})

_ACK_MAP: list[tuple[list[str], str]] = [
    (["error", "bug", "crash", "exception", "fail", "broken", "not working"],
     "Let me look into that issue."),
    (["weather", "temperature", "forecast", "rain", "humid"],
     "Checking the forecast."),
    (["code", "function", "class", "method", "variable", "syntax"],
     "Looking at the code."),
    (["explain", "what is", "what are", "meaning", "define", "difference"],
     "Let me think about that."),
    (["search", "find", "look up", "google"],
     "Let me find that for you."),
    (["how to", "how do", "steps", "guide", "tutorial"],
     "Let me work through that."),
    (["compare", "versus", "vs", "better", "which one"],
     "Weighing the options."),
    (["history", "when did", "who was", "origin"],
     "Let me recall that."),
]
_GENERIC_ACKS = [
    "On it.",
    "Working on it.",
    "One moment, Boss.",
    "Let me think...",
    "Give me a sec.",
]

_FOLLOW_UP_HINTS: list[tuple[list[str], str]] = [
    (["error", "exception", "traceback", "stack trace", "bug"],
     "Want me to read the clipboard or run a screenshot?"),
    (["install", "download", "pip install", "npm install", "setup"],
     "Should I open the terminal for you?"),
    (["documentation", "docs", "reference", "guide", "manual"],
     "Want me to search for the docs?"),
    (["reminder", "later", "tomorrow", "don't forget"],
     "Want me to set a timer for that?"),
]

_CHAIN_MAP: dict[str, str | dict[str, str]] = {
    "open_app": {
        "code": "Want me to check your git status?",
        "vscode": "Want me to check your git status?",
        "teams": "Should I check your calendar?",
        "outlook": "Want me to read your latest emails?",
        "chrome": "Need me to search for something?",
        "firefox": "Need me to search for something?",
    },
    "search": "Want me to analyze the results on your screen?",
    "screenshot": "Want me to analyze what's on screen?",
    "set_volume": "Should I also pause media?",
    "weather": "Want me to check traffic for your commute?",
    "lock_screen": "I'll keep watch. Say 'Atom' when you're back.",
}

def compress_query(text: str) -> str:
    cleaned = _FILLER.sub("", text)
    cleaned = _MULTI_SPACE.sub(" ", cleaned).strip()
    return cleaned[:1500]


class ConversationManager:
    """Manages conversational continuity, context chaining, and smart responses."""

    def __init__(self, memory: Any | None = None) -> None:
        self._conv_memory = memory
        self._last_entity: str = ""
        self._recent_queries: list[tuple[str, float]] = []
        self._conversation_window: list[tuple[str, str]] = []
        self._conv_window_max = 20

    # ── Conversational continuity (Pronouns/Entities) ───────────────────

    def extract_entity(self, text: str) -> str:
        """Extract the likely topic/entity from a query (zero-cost heuristic)."""
        words = text.lower().split()
        significant = [
            w for w in words
            if w not in _STOP_VERBS and w not in _STOP_PREPS
            and len(w) > 2 and not w.isdigit()
        ]
        if not significant:
            return ""
        return " ".join(significant[-3:])

    def resolve_pronouns(self, query: str) -> str:
        """Replace dangling pronouns with the last known entity."""
        if not self._last_entity:
            return query
        words = query.split()
        if len(words) > 8:
            return query
        if not _DANGLING_PRONOUN.search(query):
            return query
        has_noun = any(
            w.lower() not in _STOP_VERBS and w.lower() not in _STOP_PREPS
            and len(w) > 2 and not w.isdigit()
            and w.lower() not in ("it", "that", "this", "there", "those",
                                  "these", "them", "he", "she", "they")
            for w in words
        )
        if has_noun:
            return query
        resolved = _DANGLING_PRONOUN.sub(self._last_entity, query, count=1)
        return resolved

    def track_entity(self, clean_text: str) -> None:
        """Update last entity from a query."""
        entity = self.extract_entity(clean_text)
        if entity:
            self._last_entity = entity
            
    # ── Repeat query detection ──────────────────────────────────────────

    def check_repeat(self, cache_key: str) -> bool:
        """Check if querying the same thing rapidly to avoid cache."""
        now = time.monotonic()
        self._recent_queries = [
            (q, t) for q, t in self._recent_queries if now - t < 60
        ]
        for prev_q, _ in self._recent_queries:
            if prev_q == cache_key:
                return True
        self._recent_queries.append((cache_key, now))
        if len(self._recent_queries) > 5:
            self._recent_queries = self._recent_queries[-5:]
        return False

    # ── Intent chaining / Follow-ups ────────────────────────────────────

    def get_chain_suggestion(self, action: str, args: dict) -> str | None:
        """Suggest follow-ups strictly based on the tool action."""
        chain = _CHAIN_MAP.get(action)
        if chain is None:
            return None
        if isinstance(chain, dict):
            target = (args.get("name", "") or args.get("exe", "")).lower()
            for key, suggestion in chain.items():
                if key in target:
                    return suggestion
            return None
        if action == "set_volume":
            pct = int(args.get("percent", 50))
            if pct > 20:
                return None
        return chain

    def suggest_follow_up(self, query: str, response: str) -> str | None:
        """Suggest an active-listen follow up based on query+response LLM context."""
        lower_resp = response.lower()
        lower_query = query.lower()
        combined = lower_query + " " + lower_resp
        for keywords, suggestion in _FOLLOW_UP_HINTS:
            if any(kw in combined for kw in keywords):
                return suggestion
        return None

    # ── Smart Acnowledgments ────────────────────────────────────────────

    def smart_ack(self, query: str) -> str:
        q = query.lower()
        for keywords, ack in _ACK_MAP:
            if any(kw in q for kw in keywords):
                return ack
        idx = hash(query) % len(_GENERIC_ACKS)
        return _GENERIC_ACKS[idx]

    # ── History window fallbacks ────────────────────────────────────────

    def record_turn(self, query: str, response: str) -> None:
        if self._conv_memory is not None:
            self._conv_memory.record(query, "llm_response", response)
        else:
            snippet = " ".join(response.split()[:60])
            self._conversation_window.append((query[:100], snippet))
            if len(self._conversation_window) > self._conv_window_max:
                self._conversation_window = self._conversation_window[-self._conv_window_max:]

    def get_conversation_history(self) -> list[tuple[str, str]]:
        if self._conv_memory is not None and self._conv_memory.turn_count > 0:
            return self._conv_memory.get_pairs()
        return list(self._conversation_window)
