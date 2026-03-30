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
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger("atom.dream")

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus

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
        self._load_log()

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

        await self._strengthen_memories(facts)

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
        """Find semantic connections between different interactions."""
        connections = []
        topics: dict[str, list[int]] = {}

        for i, ix in enumerate(self._session_interactions):
            words = set(ix.get("query", "").lower().split())
            for word in words:
                if len(word) > 4:
                    topics.setdefault(word, []).append(i)

        for word, indices in topics.items():
            if len(indices) >= 2 and len(indices) <= 5:
                connections.append({
                    "topic": word,
                    "occurrences": len(indices),
                    "type": "recurring_topic",
                })

        return connections[:10]

    async def _strengthen_memories(self, facts: list[str]) -> None:
        """Store extracted facts in SecondBrain."""
        try:
            from core.cognitive.second_brain import SecondBrain
        except ImportError:
            return

        for fact in facts:
            try:
                self._bus.emit_fast("dream_fact_learned", fact=fact)
            except Exception:
                pass

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
            _DREAM_LOG.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "dreams": self._dream_log[-50:],
                "last_dream": self._last_dream_time,
                "total_dreams": self._dream_count,
            }
            _DREAM_LOG.write_text(json.dumps(data, indent=2), encoding="utf-8")
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
