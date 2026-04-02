"""
ATOM -- Second Brain: Vector-Enhanced Intelligence Store.

Unifies all knowledge sources into a single semantic retrieval layer:
  - Facts learned from conversations (vector-embedded)
  - User preferences inferred from behavior
  - Learned corrections (typo/alias resolution)
  - Goal summaries (from GoalEngine)
  - Habit summaries (from BehaviorTracker)
  - Document knowledge (from DocumentIngestion)

Upgrade from v15: semantic retrieval via vector store replaces
keyword-only overlap. Falls back to keyword mode gracefully.

Like JARVIS's persistent memory -- ATOM remembers everything you've
ever told it, and can recall it by meaning, not just exact words.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.persistence_manager import persistence_manager

if TYPE_CHECKING:
    from core.behavior_tracker import BehaviorTracker
    from core.memory_engine import MemoryEngine

logger = logging.getLogger("atom.brain")

_BRAIN_FILE = Path("logs/second_brain.json")
_MAX_FACTS = 1000
_MAX_CORRECTIONS = 200


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"\w{3,}", text)}


class SecondBrain:
    """Vector-enhanced intelligence store -- ATOM's long-term memory."""

    __slots__ = (
        "_memory", "_behavior", "_config",
        "_facts", "_preferences", "_corrections",
        "_dirty", "_vector_store", "_embedding_engine",
        "_vectors_ready", "_episodic_buffer",
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
        self._episodic_buffer: list[dict] = []
        self._dirty = False
        self._vector_store: Any = None
        self._embedding_engine: Any = None
        self._vectors_ready = False
        self._load()
        self._init_vectors()

    def _init_vectors(self) -> None:
        try:
            from core.embedding_engine import get_embedding_engine
            from core.vector_store import VectorStore
            self._embedding_engine = get_embedding_engine()
            self._vector_store = VectorStore()
            self._vectors_ready = True
        except Exception:
            self._vectors_ready = False

    # ── Persistence ────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if _BRAIN_FILE.exists():
                data = json.loads(_BRAIN_FILE.read_text(encoding="utf-8"))
                self._facts = data.get("facts", [])[-_MAX_FACTS:]
                self._preferences = data.get("user_preferences", {})
                self._corrections = data.get("learned_corrections", [])[-_MAX_CORRECTIONS:]
                self._episodic_buffer = data.get("episodic_buffer", [])[-50:]
                logger.info(
                    "Second brain loaded: %d facts, %d prefs, %d corrections, %d episodic",
                    len(self._facts), len(self._preferences),
                    len(self._corrections), len(self._episodic_buffer),
                )
        except Exception:
            logger.debug("No second brain file, starting fresh")

    def persist(self) -> None:
        if not self._dirty:
            return
        try:
            data = {
                "facts": self._facts[-_MAX_FACTS:],
                "user_preferences": self._preferences,
                "learned_corrections": self._corrections[-_MAX_CORRECTIONS:],
                "episodic_buffer": self._episodic_buffer[-50:],
            }
            persistence_manager.register("second_brain", _BRAIN_FILE)
            persistence_manager.save_now("second_brain", data)
            self._dirty = False
            if self._vector_store is not None:
                self._vector_store.persist()
            logger.debug("Second brain persisted")
        except Exception:
            logger.debug("Failed to persist second brain", exc_info=True)

    # ── Learning ───────────────────────────────────────────────────────

    def learn_fact(
        self, text: str, source: str = "conversation",
        tags: list[str] | None = None,
        importance: float = 0.5,
    ) -> None:
        """Store a learned fact with semantic embedding for deep recall."""
        if not text or len(text) < 5:
            return

        fact = {
            "text": text[:500],
            "source": source,
            "confidence": 0.8,
            "importance": importance,
            "ts": time.time(),
            "tags": list(tags or []),
            "keywords": list(_tokenize(text)),
        }
        self._facts.append(fact)
        if len(self._facts) > _MAX_FACTS:
            self._facts = self._facts[-_MAX_FACTS:]
        self._dirty = True

        if self._vectors_ready and self._embedding_engine is not None:
            try:
                emb = self._embedding_engine.embed_sync(text)
                self._vector_store.add(
                    "facts",
                    text=text[:500],
                    embedding=emb,
                    metadata={
                        "source": source,
                        "importance": importance,
                        "tags": ",".join(tags or []),
                    },
                )
            except Exception:
                logger.debug("Vector add for fact failed", exc_info=True)

        logger.debug("Learned fact: %s", text[:60])

    def learn_preference(self, key: str, value: Any) -> None:
        if not key:
            return
        self._preferences[key] = value
        self._dirty = True

        if self._vectors_ready and self._embedding_engine is not None:
            try:
                pref_text = f"User preference: {key} is {value}"
                emb = self._embedding_engine.embed_sync(pref_text)
                self._vector_store.add(
                    "facts",
                    text=pref_text,
                    embedding=emb,
                    metadata={"source": "preference", "key": key},
                )
            except Exception:
                pass

        logger.debug("Learned preference: %s = %s", key, value)

    def learn_correction(self, original: str, corrected: str) -> None:
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

    def add_episodic_memory(self, event: str, context: dict | None = None) -> None:
        """Store a significant episodic memory (emotional or important moment)."""
        episode = {
            "event": event[:300],
            "ts": time.time(),
            "context": context or {},
        }
        self._episodic_buffer.append(episode)
        if len(self._episodic_buffer) > 100:
            self._episodic_buffer = self._episodic_buffer[-100:]
        self._dirty = True

        if self._vectors_ready and self._embedding_engine is not None:
            try:
                emb = self._embedding_engine.embed_sync(event)
                self._vector_store.add(
                    "facts",
                    text=f"[episodic] {event}",
                    embedding=emb,
                    metadata={"source": "episodic", "importance": 0.8},
                )
            except Exception:
                pass

    # ── Retrieval ──────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        """Unified semantic search across all knowledge sources."""
        results: list[tuple[float, str]] = []
        query_tokens = _tokenize(query)

        if self._vectors_ready and self._embedding_engine is not None:
            try:
                query_emb = self._embedding_engine.embed_sync(query)
                vector_results = self._vector_store.search(
                    "facts", query_emb, k=k * 2, min_score=0.3,
                )
                for vr in vector_results:
                    results.append((vr.score * 2.0, vr.text))

                doc_results = self._vector_store.search(
                    "documents", query_emb, k=k, min_score=0.35,
                )
                for dr in doc_results:
                    results.append((dr.score * 1.8, f"[doc] {dr.text}"))
            except Exception:
                logger.debug("Vector retrieval failed, using keyword fallback", exc_info=True)

        if query_tokens:
            for fact in self._facts:
                tokens = set(fact.get("keywords", []))
                overlap = len(query_tokens & tokens)
                if overlap > 0:
                    recency_bonus = min(0.5, (time.time() - fact.get("ts", 0)) / 86400 * -0.01 + 0.5)
                    importance = fact.get("importance", 0.5)
                    score = overlap + recency_bonus + importance * 0.5
                    results.append((score, f"[fact] {fact['text']}"))

            for key, value in self._preferences.items():
                key_tokens = _tokenize(key)
                overlap = len(query_tokens & key_tokens)
                if overlap > 0:
                    results.append((overlap + 0.3, f"[pref] {key}: {value}"))

        seen: set[str] = set()
        deduped: list[tuple[float, str]] = []
        for score, text in sorted(results, key=lambda x: x[0], reverse=True):
            text_key = text[:80].lower()
            if text_key not in seen:
                seen.add(text_key)
                deduped.append((score, text))

        return [text for _, text in deduped[:k]]

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

        if self._episodic_buffer:
            recent_episodes = self._episodic_buffer[-3:]
            ep_strs = [e["event"][:80] for e in recent_episodes]
            parts.append(f"Recent events: {'; '.join(ep_strs)}")

        behavior_prefs = self._memory.preferences
        if behavior_prefs:
            if "most_active_time" in behavior_prefs:
                parts.append(f"Most active: {behavior_prefs['most_active_time']}")
            if "top_actions" in behavior_prefs:
                parts.append(f"Common actions: {', '.join(behavior_prefs['top_actions'][:5])}")

        return " | ".join(parts) if parts else ""

    def apply_correction(self, text: str) -> str:
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

    @property
    def episodic_count(self) -> int:
        return len(self._episodic_buffer)
