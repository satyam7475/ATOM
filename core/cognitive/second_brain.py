"""
ATOM v15 -- Second Brain: Structured Intelligence Store.

Unifies all knowledge sources into a single retrieval layer:
  - Facts learned from conversations
  - User preferences inferred from behavior
  - Learned corrections (typo/alias resolution)
  - Goal summaries (from GoalEngine)
  - Habit summaries (from BehaviorTracker)

Wraps the existing MemoryEngine with structured overlays.
No ML -- keyword overlap + tag matching.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.behavior_tracker import BehaviorTracker
    from core.memory_engine import MemoryEngine

logger = logging.getLogger("atom.brain")

_BRAIN_FILE = Path("logs/second_brain.json")
_MAX_FACTS = 500
_MAX_CORRECTIONS = 200


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"\w{3,}", text)}


class SecondBrain:
    """Structured intelligence store layered on top of MemoryEngine."""

    __slots__ = (
        "_memory", "_behavior", "_config",
        "_facts", "_preferences", "_corrections",
        "_dirty",
    )

    def __init__(
        self,
        memory: MemoryEngine,
        behavior: BehaviorTracker,
        config: dict | None = None,
    ) -> None:
        self._memory = memory
        self._behavior = behavior
        self._config = (config or {}).get("cognitive", {})

        self._facts: list[dict] = []
        self._preferences: dict[str, Any] = {}
        self._corrections: list[dict] = []
        self._dirty = False
        self._load()

    # ── Persistence ────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if _BRAIN_FILE.exists():
                data = json.loads(_BRAIN_FILE.read_text(encoding="utf-8"))
                self._facts = data.get("facts", [])[-_MAX_FACTS:]
                self._preferences = data.get("user_preferences", {})
                self._corrections = data.get("learned_corrections", [])[-_MAX_CORRECTIONS:]
                logger.info(
                    "Second brain loaded: %d facts, %d prefs, %d corrections",
                    len(self._facts), len(self._preferences), len(self._corrections),
                )
        except Exception:
            logger.debug("No second brain file, starting fresh")

    def persist(self) -> None:
        if not self._dirty:
            return
        try:
            _BRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "facts": self._facts[-_MAX_FACTS:],
                "user_preferences": self._preferences,
                "learned_corrections": self._corrections[-_MAX_CORRECTIONS:],
            }
            _BRAIN_FILE.write_text(
                json.dumps(data, indent=2, default=str), encoding="utf-8",
            )
            self._dirty = False
            logger.debug("Second brain persisted")
        except Exception:
            logger.debug("Failed to persist second brain", exc_info=True)

    # ── Learning ───────────────────────────────────────────────────────

    def learn_fact(
        self, text: str, source: str = "conversation", tags: list[str] | None = None,
    ) -> None:
        """Store a learned fact with tags for retrieval."""
        if not text or len(text) < 5:
            return
        fact = {
            "text": text[:500],
            "source": source,
            "confidence": 0.8,
            "ts": time.time(),
            "tags": list(tags or []),
            "keywords": list(_tokenize(text)),
        }
        self._facts.append(fact)
        if len(self._facts) > _MAX_FACTS:
            self._facts = self._facts[-_MAX_FACTS:]
        self._dirty = True
        logger.debug("Learned fact: %s", text[:60])

    def learn_preference(self, key: str, value: Any) -> None:
        """Store or update a user preference."""
        if not key:
            return
        self._preferences[key] = value
        self._dirty = True
        logger.debug("Learned preference: %s = %s", key, value)

    def learn_correction(self, original: str, corrected: str) -> None:
        """Learn a voice/text correction pattern."""
        if not original or not corrected:
            return
        for c in self._corrections:
            if c["original"] == original:
                c["corrected_to"] = corrected
                c["count"] = c.get("count", 0) + 1
                self._dirty = True
                return
        self._corrections.append({
            "original": original,
            "corrected_to": corrected,
            "count": 1,
        })
        if len(self._corrections) > _MAX_CORRECTIONS:
            self._corrections = self._corrections[-_MAX_CORRECTIONS:]
        self._dirty = True

    # ── Retrieval ──────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """Unified search across all knowledge sources."""
        results: list[tuple[float, str]] = []
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        for fact in self._facts:
            tokens = set(fact.get("keywords", []))
            overlap = len(query_tokens & tokens)
            if overlap > 0:
                recency_bonus = min(0.5, (time.time() - fact.get("ts", 0)) / 86400 * -0.01 + 0.5)
                score = overlap + recency_bonus
                results.append((score, f"[fact] {fact['text']}"))

        for key, value in self._preferences.items():
            key_tokens = _tokenize(key)
            overlap = len(query_tokens & key_tokens)
            if overlap > 0:
                results.append((overlap + 0.3, f"[pref] {key}: {value}"))

        memory_results = []
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            if loop.is_running():
                pass
            else:
                memory_results = loop.run_until_complete(
                    self._memory.retrieve(query, k=k)
                )
        except RuntimeError:
            pass

        for i, mem in enumerate(memory_results):
            results.append((2.0 - i * 0.1, f"[memory] {mem}"))

        results.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in results[:k]]

    def get_context_for_llm(self) -> str:
        """Build enriched context string for LLM prompts."""
        parts: list[str] = []

        if self._preferences:
            pref_items = list(self._preferences.items())[:10]
            pref_str = ", ".join(f"{k}={v}" for k, v in pref_items)
            parts.append(f"User preferences: {pref_str}")

        recent_facts = self._facts[-5:]
        if recent_facts:
            fact_strs = [f["text"][:100] for f in recent_facts]
            parts.append(f"Known facts: {'; '.join(fact_strs)}")

        behavior_prefs = self._memory.preferences
        if behavior_prefs:
            if "most_active_time" in behavior_prefs:
                parts.append(f"Most active: {behavior_prefs['most_active_time']}")
            if "top_actions" in behavior_prefs:
                parts.append(f"Common actions: {', '.join(behavior_prefs['top_actions'][:5])}")

        return " | ".join(parts) if parts else ""

    def apply_correction(self, text: str) -> str:
        """Apply learned corrections to input text."""
        result = text
        for c in self._corrections:
            if c["original"].lower() in result.lower() and c.get("count", 0) >= 2:
                result = re.sub(
                    re.escape(c["original"]),
                    c["corrected_to"],
                    result,
                    flags=re.IGNORECASE,
                    count=1,
                )
        return result

    def get_preference(self, key: str, default: Any = None) -> Any:
        return self._preferences.get(key, default)

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def fact_count(self) -> int:
        return len(self._facts)

    @property
    def preference_count(self) -> int:
        return len(self._preferences)

    @property
    def preferences(self) -> dict:
        return dict(self._preferences)
