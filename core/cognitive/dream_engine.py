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
_DEFAULT_PREWARM_TOPICS = ("work", "ATOM", "project", "schedule", "personal")


def _build_pattern_summary(patterns: list[dict], connections: list[dict]) -> str:
    parts: list[str] = []
    for p in patterns[:5]:
        ins = str(p.get("insight") or "").strip()
        if ins:
            parts.append(ins)
    for c in connections[:4]:
        topic = str(c.get("topic") or "").strip()
        n = int(c.get("occurrences") or 0)
        if topic and n > 1:
            parts.append(f"Recurring theme «{topic}» ({n} related queries)")
    if not parts:
        return ""
    return "Dream summary — " + " | ".join(parts)[:900]


class DreamEngine:
    """Offline memory consolidation engine."""

    def __init__(
        self,
        bus: "AsyncEventBus",
        config: dict | None = None,
        brain_mode_manager: Any | None = None,
    ) -> None:
        self._bus = bus
        self._config = (config or {}).get("cognitive", {})
        self._enabled = self._config.get("dream_enabled", True)
        self._min_idle = self._config.get("dream_idle_minutes", _MIN_IDLE_MINUTES)
        self._dream_interval = self._config.get("dream_interval_hours", _DREAM_INTERVAL_HOURS)
        self._dream_require_idle = bool(
            self._config.get("dream_require_idle_signal", False),
        )
        self._min_interactions = max(
            1,
            int(self._config.get("dream_min_interactions", 5)),
        )
        topics = self._config.get("dream_prewarm_retrieve_topics")
        if isinstance(topics, list) and topics:
            self._prewarm_topics = tuple(str(t) for t in topics[:12] if str(t).strip())
        else:
            self._prewarm_topics = _DEFAULT_PREWARM_TOPICS
        self._last_dream_time: float = 0.0
        self._dream_count: int = 0
        self._dream_log: list[dict] = []
        self._running = False
        self._task: asyncio.Task | None = None
        self._session_interactions: list[dict] = []
        self._second_brain: SecondBrain | None = None
        self._brain_mode_mgr = brain_mode_manager
        self._idle_eligible: bool = not self._dream_require_idle
        self._last_idle_signal_ts: float = 0.0
        self._load_log()

    def _background_enabled(self) -> bool:
        mgr = self._brain_mode_mgr
        if mgr is None:
            return True
        try:
            return bool(mgr.feature_enabled("dream"))
        except Exception:
            return True

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
                logger.debug('JSON state load failed', exc_info=True)

    def start(self) -> None:
        if not self._enabled:
            return
        self._running = True
        self._bus.on("idle_detected", self._on_idle_detected)
        self._task = asyncio.create_task(self._dream_monitor())
        logger.info(
            "Dream engine started (idle threshold: %d min, require_idle_signal=%s)",
            self._min_idle,
            self._dream_require_idle,
        )

    def stop(self) -> None:
        self._running = False
        try:
            self._bus.off("idle_detected", self._on_idle_detected)
        except Exception:
            logger.debug("dream idle_detected off failed", exc_info=True)
        if self._task is not None:
            self._task.cancel()
            self._task = None
        self.persist()

    async def _on_idle_detected(self, idle_minutes: float = 0, **_kw: Any) -> None:
        """Mark system idle long enough for a dream cycle (M5)."""
        try:
            mins = float(idle_minutes)
        except (TypeError, ValueError):
            mins = 0.0
        if mins >= float(self._min_idle):
            self._idle_eligible = True
            self._last_idle_signal_ts = time.time()
            logger.debug("Dream idle gate OPEN (%.0f min idle)", mins)

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

                if not self._background_enabled():
                    continue

                hours_since_dream = (time.time() - self._last_dream_time) / 3600
                if hours_since_dream < self._dream_interval:
                    continue

                if len(self._session_interactions) < self._min_interactions:
                    continue

                if self._dream_require_idle:
                    if not self._idle_eligible:
                        continue
                    if (time.time() - self._last_idle_signal_ts) > 3600:
                        self._idle_eligible = False
                        continue

                await self.dream()
                if self._dream_require_idle:
                    self._idle_eligible = False

            except asyncio.CancelledError:
                break
            except Exception:
                logger.debug("Dream monitor error", exc_info=True)

    async def dream(self) -> dict:
        """Execute a dream cycle -- consolidate and organize memories."""
        if not self._background_enabled():
            return {"status": "mode_gated"}
        if not self._session_interactions:
            return {"status": "nothing_to_dream"}

        logger.info("Dream cycle starting (%d interactions to process)...",
                     len(self._session_interactions))

        t0 = time.monotonic()
        dream_result: dict[str, Any] = {
            "timestamp": time.time(),
            "interactions_processed": len(self._session_interactions),
            "patterns": [],
            "facts_extracted": [],
            "connections": [],
            "pruned": 0,
            "brain_pruned": 0,
            "embedding_warmups": 0,
            "pattern_summary": "",
        }

        try:
            self._bus.emit_fast(
                "dream_cycle_start",
                prefer_fast_model=True,
                min_idle_minutes=self._min_idle,
            )
        except Exception:
            logger.debug('Fast bus emit failed', exc_info=True)

        patterns = self._find_patterns()
        dream_result["patterns"] = patterns

        facts = self._extract_key_facts()
        dream_result["facts_extracted"] = facts

        connections = self._find_connections()
        dream_result["connections"] = connections

        summary = _build_pattern_summary(patterns, connections)
        dream_result["pattern_summary"] = summary
        if summary and self._second_brain is not None:
            try:
                self._second_brain.learn_fact(
                    summary,
                    source="dream_pattern",
                    tags=["dream", "session_summary"],
                    importance=0.55,
                )
            except Exception:
                logger.debug("Dream pattern summary learn failed", exc_info=True)

        await self._strengthen_memories(facts, patterns)

        pruned = self._prune_noise()
        dream_result["pruned"] = pruned

        brain_pruned = 0
        if self._second_brain is not None and self._config.get(
            "dream_prune_second_brain", True,
        ):
            try:
                brain_pruned = self._second_brain.prune_for_consolidation()
            except Exception:
                logger.debug("Dream brain prune failed", exc_info=True)
        dream_result["brain_pruned"] = brain_pruned
        if brain_pruned and self._second_brain is not None:
            try:
                self._second_brain.persist()
            except Exception:
                logger.debug("Dream brain persist after prune failed", exc_info=True)

        warmups = 0
        if self._config.get("dream_prewarm_embeddings", True):
            warmups = await self._prewarm_retrieval_embeddings()
        dream_result["embedding_warmups"] = warmups

        elapsed = (time.monotonic() - t0) * 1000
        self._dream_count += 1
        self._last_dream_time = time.time()

        self._dream_log.append(dream_result)
        if len(self._dream_log) > 50:
            self._dream_log = self._dream_log[-50:]

        logger.info(
            "Dream cycle complete in %.0fms: %d patterns, %d facts, %d connections, "
            "%d ix-pruned, %d brain-pruned, %d embed-warmups",
            elapsed, len(patterns), len(facts), len(connections), pruned,
            brain_pruned, warmups,
        )

        try:
            self._bus.emit_fast("dream_cycle_end", result=dream_result)
        except Exception:
            logger.debug('Fast bus emit failed', exc_info=True)
        self._bus.emit_fast("dream_complete", result=dream_result)
        self.persist()

        return dream_result

    async def _prewarm_retrieval_embeddings(self) -> int:
        """Warm embedding cache from SecondBrain retrieval probes (idle, low cost)."""
        if self._second_brain is None:
            return 0
        try:
            from core.embedding_engine import get_embedding_engine

            eng = get_embedding_engine()
        except Exception:
            return 0

        loop = asyncio.get_running_loop()
        count = 0

        def _embed_lines(lines: list[str]) -> int:
            n = 0
            for line in lines:
                text = (line or "").strip()
                if len(text) < 16:
                    continue
                try:
                    eng.embed_sync(text[:500])
                    n += 1
                except Exception:
                    logger.debug('Fast bus emit failed', exc_info=True)
            return n

        for topic in self._prewarm_topics:
            try:
                lines = self._second_brain.retrieve(topic, k=3)
            except Exception:
                lines = []
            if lines:
                count += await loop.run_in_executor(None, _embed_lines, lines)
        return count

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
                logger.debug('Embedding sync call failed', exc_info=True)

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
                logger.debug('Fast bus emit failed', exc_info=True)

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
                    logger.debug('Fast bus emit failed', exc_info=True)

        # Persist SecondBrain if we stored anything
        if stored_count > 0 and self._second_brain is not None:
            try:
                self._second_brain.persist()
            except Exception:
                logger.debug('Fast bus emit failed', exc_info=True)

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
        extra = ""
        ps = str(last.get("pattern_summary") or "").strip()
        if ps:
            extra = f" Summary: {ps[:220]}{'…' if len(ps) > 220 else ''}"
        return (
            f"Last dream: processed {last.get('interactions_processed', 0)} interactions, "
            f"found {len(last.get('patterns', []))} patterns, "
            f"extracted {len(last.get('facts_extracted', []))} facts, "
            f"pruned {last.get('brain_pruned', 0)} stale brain rows, "
            f"{last.get('embedding_warmups', 0)} embedding warmups.{extra} "
            f"Total dreams: {self._dream_count}."
        )
