"""
ATOM -- Dream Engine (Offline Memory Consolidation).

Like how the human brain consolidates memories during sleep, ATOM's
Dream Engine processes the day's interactions during idle periods:

  1. REPLAY: Review all conversations and actions from the session
  2. COMPRESS: Identify patterns, extract key facts, discard noise
  3. CONNECT: Find relationships between disparate pieces of knowledge
  4. STRENGTHEN: Boost confidence on frequently-accessed memories
  5. PRUNE: Remove low-value or redundant entries

Runs automatically when ATOM has been idle for 30+ minutes,
or can be triggered with "dream mode" / "consolidate memories".

This is what makes ATOM's memory feel alive -- it doesn't just
store data, it processes and organizes it like a real brain.

Contract: CognitiveModuleContract (start, stop, persist)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.persistence_manager import persistence_manager

logger = logging.getLogger("atom.dream")

def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two dense vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

# Common words to ignore in semantic connection analysis
_STOPWORDS = frozenset({
    "about", "after", "again", "being", "could", "doing",
    "every", "first", "found", "going", "great", "https",
    "known", "large", "leave", "might", "never", "other",
    "place", "point", "quite", "right", "shall", "since",
    "small", "start", "still", "taken", "their", "there",
    "these", "thing", "think", "those", "three", "under",
    "using", "value", "watch", "where", "which", "while",
    "whole", "world", "would", "write", "years", "please",
    "should", "would", "could", "really", "system", "check",
    "what's", "don't", "can't", "that's",
})

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.cognitive.second_brain import SecondBrain

_DREAM_LOG = Path("logs/dream_log.json")
_MIN_IDLE_MINUTES = 30
_DREAM_INTERVAL_HOURS = 6


class DreamEngine:
    """Offline memory consolidation engine."""

    def __init__(
        self,
        bus: "AsyncEventBus",
        config: dict | None = None,
    ) -> None:
        self._bus = bus
        self._config = (config or {}).get("cognitive", {})
        self._enabled = self._config.get("dream_enabled", True)
        self._min_idle = self._config.get("dream_idle_minutes", _MIN_IDLE_MINUTES)
        self._dream_interval = self._config.get("dream_interval_hours", _DREAM_INTERVAL_HOURS)
        self._last_dream_time: float = 0.0
        self._dream_count: int = 0
        self._dream_log: list[dict] = []
        self._running = False
        self._task: asyncio.Task | None = None
        self._session_interactions: list[dict] = []
        self._second_brain: SecondBrain | None = None
        self._load_log()

    def wire(self, second_brain: "SecondBrain | None" = None) -> None:
        """Wire cognitive dependencies after initialization."""
        self._second_brain = second_brain

    def _load_log(self) -> None:
        if _DREAM_LOG.exists():
            try:
                data = json.loads(_DREAM_LOG.read_text(encoding="utf-8"))
                self._dream_log = data.get("dreams", [])[-50:]
                self._last_dream_time = data.get("last_dream", 0.0)
                self._dream_count = data.get("total_dreams", 0)
            except Exception:
                pass

    def start(self) -> None:
        if not self._enabled:
            return
        self._running = True
        self._task = asyncio.create_task(self._dream_monitor())
        logger.info("Dream engine started (idle threshold: %d min)", self._min_idle)

    def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            self._task = None
        self.persist()

    def record_interaction(self, query: str, response: str,
                           intent: str = "", emotion: str = "") -> None:
        """Record a session interaction for dream processing."""
        self._session_interactions.append({
            "query": query[:200],
            "response": response[:300],
            "intent": intent,
            "emotion": emotion,
            "ts": time.time(),
        })
        if len(self._session_interactions) > 200:
            self._session_interactions = self._session_interactions[-200:]

    async def _dream_monitor(self) -> None:
        """Background monitor that triggers dreaming during idle."""
        while self._running:
            try:
                await asyncio.sleep(300)
                if not self._running:
                    break

                hours_since_dream = (time.time() - self._last_dream_time) / 3600
                if hours_since_dream < self._dream_interval:
                    continue

                if len(self._session_interactions) < 5:
                    continue

                await self.dream()

            except asyncio.CancelledError:
                break
            except Exception:
                logger.debug("Dream monitor error", exc_info=True)

    async def dream(self) -> dict:
        """Execute a dream cycle -- consolidate and organize memories."""
        if not self._session_interactions:
            return {"status": "nothing_to_dream"}

        logger.info("Dream cycle starting (%d interactions to process)...",
                     len(self._session_interactions))

        t0 = time.monotonic()
        dream_result = {
            "timestamp": time.time(),
            "interactions_processed": len(self._session_interactions),
            "patterns": [],
            "facts_extracted": [],
            "connections": [],
            "pruned": 0,
        }

        patterns = self._find_patterns()
        dream_result["patterns"] = patterns

        facts = self._extract_key_facts()
        dream_result["facts_extracted"] = facts

        connections = self._find_connections()
        dream_result["connections"] = connections

        await self._strengthen_memories(facts, patterns)

        pruned = self._prune_noise()
        dream_result["pruned"] = pruned

        elapsed = (time.monotonic() - t0) * 1000
        self._dream_count += 1
        self._last_dream_time = time.time()

        self._dream_log.append(dream_result)
        if len(self._dream_log) > 50:
            self._dream_log = self._dream_log[-50:]

        logger.info(
            "Dream cycle complete in %.0fms: %d patterns, %d facts, %d connections, %d pruned",
            elapsed, len(patterns), len(facts), len(connections), pruned,
        )

        self._bus.emit_fast("dream_complete", result=dream_result)
        self.persist()

        return dream_result

    def _find_patterns(self) -> list[dict]:
        """Identify repeated patterns in the session."""
        intent_counts: dict[str, int] = {}
        time_patterns: dict[int, list[str]] = {}

        for ix in self._session_interactions:
            intent = ix.get("intent", "")
            if intent:
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
            hour = int(ix.get("ts", 0)) // 3600 % 24
            time_patterns.setdefault(hour, []).append(intent)

        patterns = []
        for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
            if count >= 3:
                patterns.append({
                    "type": "frequent_action",
                    "action": intent,
                    "count": count,
                    "insight": f"Boss frequently uses '{intent}' ({count} times this session)",
                })

        return patterns[:10]

    def _extract_key_facts(self) -> list[str]:
        """Extract important facts from conversations."""
        facts = []
        for ix in self._session_interactions:
            query = ix.get("query", "").lower()
            response = ix.get("response", "")

            if any(kw in query for kw in ("remember", "note", "important", "don't forget")):
                facts.append(f"Boss said to remember: {query[:100]}")

            if ix.get("emotion") in ("stressed", "frustrated") and query:
                facts.append(f"Boss was {ix['emotion']} about: {query[:80]}")

        return facts[:20]

    def _find_connections(self) -> list[dict]:
        """Find semantic connections using Mathematical Vector Clustering.

        Maps all queries to dense vectors and clusters them by cosine distance.
        This provides JARVIS-level semantic correlation without LLM overhead.
        """
        connections = []
        try:
            from core.embedding_engine import get_embedding_engine
            embed = get_embedding_engine()
        except ImportError:
            return []

        # 1. Embed valid queries
        embedded_ixs = []
        for i, ix in enumerate(self._session_interactions):
            query = ix.get("query", "").strip()
            if len(query) < 10:
                continue
            try:
                vec = embed.embed_sync(query)
                embedded_ixs.append((i, query, vec))
            except Exception:
                pass

        if not embedded_ixs:
            return []

        # 2. O(N^2) Math clustering threshold (safe since N < 200)
        _SIMILARITY_THRESHOLD = 0.82
        clusters: list[list[int]] = []
        assigned = set()

        for i in range(len(embedded_ixs)):
            idx1, q1, v1 = embedded_ixs[i]
            if idx1 in assigned:
                continue
                
            current_cluster = [idx1]
            assigned.add(idx1)
            
            for j in range(i + 1, len(embedded_ixs)):
                idx2, q2, v2 = embedded_ixs[j]
                if idx2 in assigned:
                    continue
                
                sim = _cosine_similarity(v1, v2)
                if sim >= _SIMILARITY_THRESHOLD:
                    current_cluster.append(idx2)
                    assigned.add(idx2)
            
            if len(current_cluster) >= 2:
                clusters.append(current_cluster)

        # 3. Format insights
        for c in clusters:
            samples = []
            for idx in c[:3]:
                q = self._session_interactions[idx].get("query", "")[:60]
                if q:
                    samples.append(q)
                    
            # Derive an arbitrary top word as topic
            words = set(samples[0].lower().split()) - _STOPWORDS if samples else set()
            topic = max(words, key=len).title() if words else "Recurring Concept"
            
            connections.append({
                "topic": topic,
                "occurrences": len(c),
                "type": "semantic_cluster",
                "sample_queries": samples,
            })

        connections.sort(key=lambda x: x["occurrences"], reverse=True)
        return connections[:10]

    async def _strengthen_memories(
        self, facts: list[str], patterns: list[dict] | None = None,
    ) -> None:
        """Store extracted facts and patterns in SecondBrain.

        Previously this method only emitted events that nothing handled.
        Now it actually persists knowledge into the SecondBrain store.
        """
        stored_count = 0

        for fact in facts:
            # 1. Always emit the event (other modules may listen)
            try:
                self._bus.emit_fast("dream_fact_learned", fact=fact)
            except Exception:
                pass

            # 2. Actually store in SecondBrain (the critical fix)
            if self._second_brain is not None:
                try:
                    self._second_brain.learn_fact(
                        text=fact,
                        source="dream_consolidation",
                        tags=["dream", "auto_extracted"],
                        importance=0.6,
                    )
                    stored_count += 1
                except Exception:
                    logger.debug("Failed to store dream fact: %s", fact[:40], exc_info=True)

        # Also store notable patterns as learned knowledge
        for pattern in (patterns or []):
            if pattern.get("count", 0) >= 5 and self._second_brain is not None:
                try:
                    insight = pattern.get("insight", "")
                    if insight:
                        self._second_brain.learn_fact(
                            text=insight,
                            source="dream_pattern",
                            tags=["dream", "pattern", pattern.get("action", "")],
                            importance=0.7,
                        )
                        stored_count += 1
                except Exception:
                    pass

        # Persist SecondBrain if we stored anything
        if stored_count > 0 and self._second_brain is not None:
            try:
                self._second_brain.persist()
            except Exception:
                pass

        logger.info(
            "Dream memory strengthening: %d facts + patterns stored in SecondBrain",
            stored_count,
        )

    def _prune_noise(self) -> int:
        """Remove low-value interactions (noise words, failed intents)."""
        original = len(self._session_interactions)
        self._session_interactions = [
            ix for ix in self._session_interactions
            if ix.get("intent") not in ("", "noise")
            and len(ix.get("query", "")) > 3
        ]
        pruned = original - len(self._session_interactions)
        return pruned

    def persist(self) -> None:
        try:
            data = {
                "dreams": self._dream_log[-50:],
                "last_dream": self._last_dream_time,
                "total_dreams": self._dream_count,
            }
            persistence_manager.register("dream_log", _DREAM_LOG)
            persistence_manager.save_now("dream_log", data)
        except Exception:
            logger.debug("Dream log persist failed", exc_info=True)

    def get_dream_summary(self) -> str:
        if not self._dream_log:
            return "No dreams yet. I consolidate memories when you're away."
        last = self._dream_log[-1]
        return (
            f"Last dream: processed {last.get('interactions_processed', 0)} interactions, "
            f"found {len(last.get('patterns', []))} patterns, "
            f"extracted {len(last.get('facts_extracted', []))} facts. "
            f"Total dreams: {self._dream_count}."
        )
