"""
ATOM -- Semantic Memory Engine (Vector + Keyword Hybrid).

Dual-mode memory that uses vector embeddings for semantic search when
sentence-transformers is available, falling back to keyword overlap
when it's not. This gives ATOM true "recall by meaning" -- the
difference between a notepad and a brain.

v20 optimizations over v18:
  - Inverted keyword index for O(1) keyword lookup (was O(n) scan)
  - Parallel vector + keyword retrieval (asyncio.gather)
  - Batch vector migration using add_batch
  - Cached preferences (recompute only when interactions change)
  - Pre-compiled regex patterns moved to module level
  - Compact interaction logging with timestamp-based dedup

Stores Q&A pairs with both keyword sets AND vector embeddings.
Retrieves by cosine similarity (semantic) + keyword overlap (exact).
Scores are blended: 0.7 * semantic + 0.3 * keyword overlap.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from core.persistence_manager import persistence_manager

logger = logging.getLogger("atom.memory")

_PERSIST_FILE = Path("logs/memory.json")
_INTERACTIONS_FILE = Path("logs/interactions.json")
_MAX_INTERACTIONS = 5000
_WORD_RE = re.compile(r"\w{3,}")

TECH_KEYWORD_PATTERN = re.compile(
    r"\b(process|system|cpu|memory|disk|network|battery|resource|"
    r"automate|schedule|reminder|scroll|click|desktop|browser|"
    r"api|sql|docker|deploy|pipeline|config|configure|configuration|python|java|node|git|"
    r"install|update|backup|monitor|performance|diagnostic|"
    r"spring|kafka|kubernetes|gradle|maven)\b",
    re.IGNORECASE,
)

_TOD_TABLE = ["night"] * 5 + ["morning"] * 7 + ["afternoon"] * 5 + ["evening"] * 4 + ["night"] * 3


def _time_of_day(hour: int | None = None) -> str:
    if hour is None:
        hour = datetime.now().hour
    return _TOD_TABLE[hour % 24]


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)}


def _recency_component(timestamp: float | None, now: float) -> float:
    """0..1 higher = more recent."""
    if timestamp is None:
        return 0.5
    try:
        ts = float(timestamp)
    except (TypeError, ValueError):
        return 0.5
    age_h = max(0.0, (now - ts) / 3600.0)
    return float(math.exp(-age_h / 168.0))


def score_memory_candidate(
    similarity: float,
    metadata: dict[str, Any] | None,
    *,
    now: float | None = None,
    weights: dict[str, float] | None = None,
) -> float:
    """V7 combined score: recency, importance, success_rate, similarity."""
    w = weights or {
        "recency": 0.3,
        "importance": 0.3,
        "success_rate": 0.2,
        "similarity": 0.2,
    }
    meta = metadata or {}
    t = now if now is not None else time.time()
    imp = float(meta.get("importance", 0.5))
    sr = float(meta.get("success_rate", 1.0))
    rec = _recency_component(meta.get("timestamp"), t)
    sim = max(0.0, min(1.0, float(similarity)))
    imp = max(0.0, min(1.0, imp))
    sr = max(0.0, min(1.0, sr))
    return (
        w.get("recency", 0.3) * rec
        + w.get("importance", 0.3) * imp
        + w.get("success_rate", 0.2) * sr
        + w.get("similarity", 0.2) * sim
    )


class MemoryEngine:
    """Hybrid semantic + keyword memory with inverted index and full interaction log."""

    def __init__(self, config: dict | None = None) -> None:
        mem_cfg = (config or {}).get("memory", {})
        self._max_entries: int = mem_cfg.get("max_entries", 1000)
        self._max_vector_results: int = max(1, int(mem_cfg.get("max_vector_results", 5)))
        self._default_top_k: int = max(
            1,
            min(int(mem_cfg.get("top_k", 5)), self._max_vector_results),
        )
        self._pressure_threshold_pct: float = float(
            mem_cfg.get("pressure_threshold_pct", 85.0),
        )
        self._pressure_relief_pct: float = float(
            mem_cfg.get("pressure_relief_pct", max(0.0, self._pressure_threshold_pct - 10.0)),
        )
        self._pressure_top_k: int = max(
            1,
            min(int(mem_cfg.get("pressure_top_k", 2)), self._default_top_k),
        )
        self._semantic_weight: float = mem_cfg.get("semantic_weight", 0.7)
        self._keyword_weight: float = 1.0 - self._semantic_weight
        self._entries: list[dict] = []
        self._inverted_index: dict[str, set[int]] = defaultdict(set)
        self._interactions: list[dict] = []
        self._interactions_dirty: bool = False
        self._preferences_cache: dict | None = None
        self._preferences_interaction_count: int = 0
        self._vector_store: Any = None
        self._embedding_engine: Any = None
        self._vectors_ready: bool = False
        self._migration_done: bool = False
        self._config = config or {}
        self._memory_pressure_active: bool = False
        self._lock = asyncio.Lock()
        v7 = mem_cfg.get("v7_scoring") or {}
        self._v7_scoring_enabled: bool = bool(v7.get("enabled", False))
        self._v7_w = {
            "recency": float(v7.get("recency_weight", 0.3)),
            "importance": float(v7.get("importance_weight", 0.3)),
            "success_rate": float(v7.get("success_rate_weight", 0.2)),
            "similarity": float(v7.get("similarity_weight", 0.2)),
        }
        self._load()
        self._load_interactions()
        self._init_vectors()

    def _rebuild_inverted_index(self) -> None:
        """Build keyword -> entry-index mapping for O(1) keyword retrieval."""
        self._inverted_index.clear()
        for idx, entry in enumerate(self._entries):
            for kw in entry.get("keywords", []):
                self._inverted_index[kw].add(idx)

    def _index_entry(self, idx: int, keywords: list[str]) -> None:
        for kw in keywords:
            self._inverted_index[kw].add(idx)

    def _init_vectors(self) -> None:
        """Initialize vector store and embedding engine (lazy, non-blocking)."""
        try:
            from core.embedding_engine import get_embedding_engine
            from core.vector_store import VectorStore
            self._embedding_engine = get_embedding_engine(self._config)
            self._vector_store = VectorStore(self._config)
            self._vectors_ready = True
            logger.info("Memory engine: semantic vectors enabled")
        except Exception:
            logger.info("Memory engine: keyword-only mode (vectors unavailable)")
            self._vectors_ready = False

    def _effective_top_k(self, k: int) -> int:
        limit = max(1, min(int(k), self._default_top_k, self._max_vector_results))
        if self._memory_pressure_active:
            limit = min(limit, self._pressure_top_k)
        return max(1, limit)

    def apply_memory_pressure(self, memory_pct: float) -> dict[str, Any]:
        """Drop vector-heavy behavior when unified memory gets tight."""
        active = memory_pct >= self._pressure_threshold_pct
        if self._memory_pressure_active:
            active = memory_pct > self._pressure_relief_pct

        changed = active != self._memory_pressure_active
        self._memory_pressure_active = active

        if changed and active:
            logger.warning(
                "MemoryEngine pressure mode ON at %.0f%%: vector retrieval suspended",
                memory_pct,
            )
            if self._embedding_engine is not None:
                try:
                    self._embedding_engine.shutdown()
                except Exception:
                    logger.debug("MemoryEngine embed shutdown failed", exc_info=True)
        elif changed and not active:
            logger.info(
                "MemoryEngine pressure mode OFF at %.0f%%: vector retrieval restored",
                memory_pct,
            )

        return {
            "active": self._memory_pressure_active,
            "memory_pct": round(memory_pct, 1),
            "top_k": self._pressure_top_k if self._memory_pressure_active else self._default_top_k,
        }

    async def migrate_to_vectors(self) -> None:
        """Background migration of existing keyword-only entries to vector store."""
        if self._migration_done or not self._vectors_ready or self._memory_pressure_active:
            return
        if not self._entries:
            self._migration_done = True
            return

        unmigrated = [e for e in self._entries if "embedding" not in e]
        if not unmigrated:
            self._migration_done = True
            return

        logger.info("Migrating %d memory entries to vector store...", len(unmigrated))
        texts = [f"{e['query']} {e['summary']}" for e in unmigrated]

        try:
            embeddings = await self._embedding_engine.embed_batch(texts)
            batch_texts = []
            batch_embeddings = []
            batch_metas = []
            batch_ids = []

            for entry, emb in zip(unmigrated, embeddings):
                entry["embedding"] = True
                doc_id = f"mem_{int(entry.get('timestamp', time.time()))}"
                batch_ids.append(doc_id)
                batch_texts.append(f"Q: {entry['query']} A: {entry['summary']}")
                batch_embeddings.append(emb)
                batch_metas.append({
                    "query": entry["query"][:200],
                    "summary": entry["summary"][:300],
                    "source": "migration",
                    "importance": float(entry.get("importance", 0.5)),
                    "success_rate": float(entry.get("success_rate", 1.0)),
                })

            self._vector_store.add_batch(
                "conversations", batch_texts, batch_embeddings,
                metadatas=batch_metas, doc_ids=batch_ids,
            )
            self._migration_done = True
            logger.info("Memory migration complete: %d entries vectorized", len(unmigrated))
        except Exception:
            logger.debug("Memory migration failed (will retry)", exc_info=True)

    # ── Q&A Memory ─────────────────────────────────────────────────────

    def _load(self) -> None:
        if _PERSIST_FILE.exists():
            try:
                with open(_PERSIST_FILE, "r", encoding="utf-8") as f:
                    self._entries = json.load(f)
                self._rebuild_inverted_index()
                logger.info("Memory loaded: %d entries, %d index terms",
                            len(self._entries), len(self._inverted_index))
            except Exception:
                self._entries = []

    @staticmethod
    def should_store(query: str) -> bool:
        if len(query.split()) > 5:
            return True
        if TECH_KEYWORD_PATTERN.search(query):
            return True
        return False

    async def add(self, query: str, summary: str) -> None:
        if not self.should_store(query):
            return

        from context.privacy_filter import redact as _redact
        clean_query = _redact(query)
        clean_summary = _redact(summary)

        keywords = list(_tokenize(query + " " + summary))
        entry: dict[str, Any] = {
            "query": clean_query,
            "summary": clean_summary,
            "keywords": keywords,
            "timestamp": time.time(),
            "importance": 0.5,
            "success_rate": 1.0,
        }

        if (
            self._vectors_ready
            and self._embedding_engine is not None
            and not self._memory_pressure_active
        ):
            try:
                combined = f"{clean_query} {clean_summary}"
                emb = await self._embedding_engine.embed(combined)
                doc_id = f"mem_{int(time.time())}_{len(self._entries)}"
                self._vector_store.add(
                    "conversations",
                    text=f"Q: {clean_query} A: {clean_summary}",
                    embedding=emb,
                    metadata={
                        "query": clean_query[:200],
                        "summary": clean_summary[:300],
                        "source": "conversation",
                        "importance": 0.5,
                        "success_rate": 1.0,
                    },
                    doc_id=doc_id,
                )
                entry["embedding"] = True
            except Exception:
                logger.debug("Vector add failed, keyword-only", exc_info=True)

        idx = len(self._entries)
        self._entries.append(entry)
        self._index_entry(idx, keywords)

        if len(self._entries) > self._max_entries:
            async with self._lock:
                self._entries = self._entries[-self._max_entries:]
                self._rebuild_inverted_index()

    async def _retrieve_keyword(self, query: str, k: int) -> list[tuple[float, str]]:
        """O(1) keyword lookup using inverted index instead of scanning all entries."""
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        async with self._lock:
            candidate_indices: Counter[int] = Counter()
            for token in query_tokens:
                for idx in self._inverted_index.get(token, ()):
                    candidate_indices[idx] += 1

            if not candidate_indices:
                return []

            results: list[tuple[float, str]] = []
            n_query = len(query_tokens)

            for idx, overlap_count in candidate_indices.most_common(k * 3):
                if idx >= len(self._entries):
                    continue
                entry = self._entries[idx]
                entry_tokens = set(entry.get("keywords", []))
                union = len(query_tokens | entry_tokens) or 1
                jaccard = overlap_count / union
                score = jaccard * self._keyword_weight
                results.append((score, entry["summary"]))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:k * 2]

    async def _retrieve_vector(self, query: str, k: int) -> list[tuple[float, str]]:
        """Semantic vector retrieval (optional V7 scoring over metadata)."""
        if (
            self._memory_pressure_active
            or not self._vectors_ready
            or self._embedding_engine is None
        ):
            return []
        try:
            query_emb = await self._embedding_engine.embed(query)
            vector_k = min(max(1, k * 3), self._max_vector_results)
            vector_results = self._vector_store.search(
                "conversations", query_emb, k=vector_k, min_score=0.25,
            )
            now = time.time()
            out: list[tuple[float, str]] = []
            for vr in vector_results:
                if self._v7_scoring_enabled:
                    combined = score_memory_candidate(
                        vr.score, vr.metadata, now=now, weights=self._v7_w,
                    )
                    out.append((combined, vr.text))
                else:
                    out.append((vr.score * self._semantic_weight, vr.text))
            return out
        except Exception:
            logger.debug("Vector search failed, using keyword fallback", exc_info=True)
            return []

    async def retrieve(self, query: str, k: int = 5) -> list[str]:
        """Hybrid retrieval: semantic vectors + keyword overlap, blended scoring.

        Runs vector and keyword searches in parallel for lower latency.
        """
        k = self._effective_top_k(k)
        if not self._entries and (
            not self._vectors_ready or not self._vector_store
        ):
            return []

        import asyncio
        vector_task = asyncio.create_task(self._retrieve_vector(query, k))
        keyword_task = asyncio.create_task(self._retrieve_keyword(query, k))

        vector_results, keyword_results = await asyncio.gather(
            vector_task, keyword_task, return_exceptions=True,
        )

        results: list[tuple[float, str]] = []
        if isinstance(vector_results, list):
            results.extend(vector_results)
        if isinstance(keyword_results, list):
            results.extend(keyword_results)

        if not results:
            return []

        seen: set[str] = set()
        deduped: list[tuple[float, str]] = []
        for score, text in sorted(results, key=lambda x: x[0], reverse=True):
            text_key = text[:100].lower()
            if text_key not in seen:
                seen.add(text_key)
                deduped.append((score, text))
                if len(deduped) >= k:
                    break

        return [s for _, s in deduped]

    async def retrieve_with_scores(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Retrieve memories with their relevance scores."""
        k = self._effective_top_k(k)
        if (
            self._memory_pressure_active
            or not self._vectors_ready
            or self._embedding_engine is None
        ):
            basic = await self.retrieve(query, k)
            return [(t, 0.5) for t in basic]

        try:
            query_emb = await self._embedding_engine.embed(query)
            vector_results = self._vector_store.search(
                "conversations", query_emb, k=k, min_score=0.35,
            )
            return [(vr.text, vr.score) for vr in vector_results]
        except Exception:
            basic = await self.retrieve(query, k)
            return [(t, 0.5) for t in basic]

    def persist(self) -> None:
        _PERSIST_FILE.parent.mkdir(exist_ok=True)
        try:
            save_entries = [
                {k: v for k, v in e.items() if k != "embedding_vec"}
                for e in self._entries
            ]
            persistence_manager.register("memory", _PERSIST_FILE)
            persistence_manager.save_now("memory", save_entries)
            logger.info("Memory persisted: %d entries", len(self._entries))
        except Exception:
            logger.exception("Failed to persist memory")

        self._persist_interactions()

        if self._vector_store is not None:
            try:
                self._vector_store.persist()
            except Exception:
                logger.debug("Vector store persist failed", exc_info=True)

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def vectors_ready(self) -> bool:
        return self._vectors_ready

    def get_vector_stats(self) -> dict:
        if self._vector_store is not None:
            return self._vector_store.get_stats()
        return {}

    async def warm_up_embeddings(self) -> bool:
        """Eagerly load the embedding model for the first retrieval."""
        if (
            self._memory_pressure_active
            or not self._vectors_ready
            or self._embedding_engine is None
        ):
            return False

        try:
            loop = asyncio.get_running_loop()
            t0 = time.monotonic()
            loaded = await loop.run_in_executor(
                None,
                self._embedding_engine.preload,
            )
            if loaded:
                logger.info(
                    "Memory engine: embeddings warm-up ready in %.0fms",
                    (time.monotonic() - t0) * 1000,
                )
            return bool(loaded)
        except Exception:
            logger.debug("Memory engine embeddings warm-up failed", exc_info=True)
            return False

    def get_top_commands(self, limit: int = 10) -> list[str]:
        """Return the most common successful commands from interaction history."""
        if limit <= 0 or not self._interactions:
            return []

        command_counts: Counter[str] = Counter()
        canonical: dict[str, str] = {}
        for item in self._interactions:
            command = str(item.get("command", "") or "").strip()
            result = str(item.get("result", "success") or "success").strip().lower()
            if not command or result not in {"", "success"}:
                continue
            key = command.lower()
            command_counts[key] += 1
            canonical[key] = command

        return [
            canonical[key]
            for key, _count in command_counts.most_common(limit)
            if key in canonical
        ]

    # ── Interaction Log ────────────────────────────────────────────────

    def _load_interactions(self) -> None:
        if _INTERACTIONS_FILE.exists():
            try:
                with open(_INTERACTIONS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self._interactions = data[-_MAX_INTERACTIONS:]
                logger.info("Interactions loaded: %d entries",
                            len(self._interactions))
            except Exception:
                self._interactions = []

    def log_interaction(
        self,
        command: str,
        action: str,
        system_state: dict | None = None,
        result: str = "success",
    ) -> None:
        now = datetime.now()
        hour = now.hour
        entry = {
            "command": (command or "")[:200],
            "action": action,
            "timestamp": time.time(),
            "system_state": system_state or {},
            "result": result,
            "time_of_day": _TOD_TABLE[hour],
            "hour": hour,
            "weekday": now.weekday(),
        }
        self._interactions.append(entry)
        if len(self._interactions) > _MAX_INTERACTIONS:
            self._interactions = self._interactions[-_MAX_INTERACTIONS:]
        self._interactions_dirty = True
        self._preferences_cache = None

    def _persist_interactions(self) -> None:
        if not self._interactions_dirty:
            return
        try:
            persistence_manager.register("interactions", _INTERACTIONS_FILE)
            persistence_manager.save_now("interactions", self._interactions)
            self._interactions_dirty = False
        except Exception:
            logger.debug("Failed to persist interactions", exc_info=True)

    @property
    def interaction_count(self) -> int:
        return len(self._interactions)

    # ── Preferences (cached) ──────────────────────────────────────────

    @property
    def preferences(self) -> dict:
        if not self._interactions:
            return {}

        n = len(self._interactions)
        if self._preferences_cache is not None and self._preferences_interaction_count == n:
            return self._preferences_cache

        prefs: dict = {}
        app_counts: Counter = Counter()
        action_counts: Counter = Counter()
        tod_counts: Counter = Counter()

        for ix in self._interactions:
            action = ix.get("action", "")
            action_counts[action] += 1
            tod_counts[ix.get("time_of_day", "")] += 1
            if action == "open_app":
                target = ix.get("command", "")
                if target:
                    app_counts[target] += 1

        if app_counts:
            prefs["top_apps"] = [app for app, _ in app_counts.most_common(5)]
        if action_counts:
            prefs["top_actions"] = [act for act, _ in action_counts.most_common(10)]
        if tod_counts:
            prefs["most_active_time"] = tod_counts.most_common(1)[0][0]
        prefs["total_interactions"] = n

        self._preferences_cache = prefs
        self._preferences_interaction_count = n
        return prefs
