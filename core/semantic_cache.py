"""
ATOM — Semantic Cache (Embedding-Aware Response Cache).

Extension of the existing CacheEngine that uses semantic similarity
to match queries that are worded differently but mean the same thing.

Example:
  Query 1: "What is machine learning?"
  Query 2: "Explain ML to me"
  → Semantic similarity: 0.91 → cache HIT

Uses the existing EmbeddingEngine (sentence-transformers, all-MiniLM-L6-v2)
for vector computation and cosine similarity matching.

Dramatically reduces LLM calls for rephrased questions.

Cache eviction: LRU with TTL (same as CacheEngine).

Owner: Satyam
"""

from __future__ import annotations

import logging
import time
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("atom.semantic_cache")


@dataclass
class _CacheEntry:
    """A cached query-response pair with embedding."""
    query: str
    response: str
    embedding: list[float]
    timestamp: float
    hit_count: int = 0
    source: str = "local"  # "local" or "cloud"


class SemanticCache:
    """Embedding-aware response cache.

    Usage:
        cache = SemanticCache(config)

        # Check for semantic hit:
        hit = cache.get(query)
        if hit:
            return hit  # Skip LLM entirely

        # After generating:
        cache.put(query, response, source="local")
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = (config or {}).get("semantic_cache", {})
        self._max_size = int(cfg.get("max_size", 256))
        self._ttl = float(cfg.get("ttl_seconds", 600))
        self._similarity_threshold = float(cfg.get("threshold", 0.85))
        self._enabled = bool(cfg.get("enabled", True))

        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()

        self._embedding_engine: Any = None
        self._has_embeddings = False

        self._total_hits = 0
        self._total_misses = 0
        self._total_puts = 0

        self._init_embeddings()

        logger.info(
            "SemanticCache: max=%d, ttl=%.0fs, threshold=%.2f, embeddings=%s",
            self._max_size, self._ttl, self._similarity_threshold,
            self._has_embeddings,
        )

    def _init_embeddings(self) -> None:
        """Lazy-load the embedding engine."""
        try:
            from core.embedding_engine import get_embedding_engine
            self._embedding_engine = get_embedding_engine()
            self._has_embeddings = True
        except ImportError:
            logger.info("SemanticCache: embeddings unavailable, using exact match only")
        except Exception:
            logger.debug("SemanticCache: embedding init failed", exc_info=True)

    @property
    def is_semantic(self) -> bool:
        return self._has_embeddings

    def get(self, query: str) -> str | None:
        """Look up a query in the semantic cache.

        First tries exact match, then falls back to semantic similarity.
        Returns the cached response or None.
        """
        if not self._enabled or not query or not query.strip():
            return None

        query = query.strip()
        now = time.time()

        with self._lock:
            # Step 1: Exact match (fast path)
            exact = self._cache.get(query)
            if exact and (now - exact.timestamp) < self._ttl:
                exact.hit_count += 1
                self._cache.move_to_end(query)
                self._total_hits += 1
                logger.debug("SemanticCache: exact hit for '%s'", query[:50])
                return exact.response

            # Step 2: Semantic similarity match (if embeddings available)
            if not self._has_embeddings or self._embedding_engine is None:
                self._total_misses += 1
                return None

            try:
                query_embedding = self._embedding_engine.embed_sync(query)
                if not query_embedding:
                    self._total_misses += 1
                    return None

                best_score = 0.0
                best_key: str | None = None

                # Collect candidate embeddings for batch similarity
                candidates: list[tuple[str, _CacheEntry]] = []
                for key, entry in self._cache.items():
                    if (now - entry.timestamp) < self._ttl and entry.embedding:
                        candidates.append((key, entry))

                if not candidates:
                    self._total_misses += 1
                    return None

                # Batch similarity (numpy-optimized when available)
                embeddings = [entry.embedding for _, entry in candidates]
                scores = self._embedding_engine.batch_similarity(
                    query_embedding, embeddings,
                )

                for i, sim in enumerate(scores):
                    if sim > best_score:
                        best_score = sim
                        best_key = candidates[i][0]

                if best_score >= self._similarity_threshold and best_key:
                    entry = self._cache[best_key]
                    entry.hit_count += 1
                    self._cache.move_to_end(best_key)
                    self._total_hits += 1
                    logger.info(
                        "SemanticCache: semantic hit (%.3f) '%s' → '%s'",
                        best_score, query[:40], best_key[:40],
                    )
                    return entry.response

            except Exception:
                logger.debug("SemanticCache: similarity search failed", exc_info=True)

        self._total_misses += 1
        return None

    def put(
        self,
        query: str,
        response: str,
        source: str = "local",
    ) -> None:
        """Store a query-response pair in the semantic cache."""
        if not self._enabled or not query or not response:
            return

        query = query.strip()

        # Compute embedding
        embedding: list[float] = []
        if self._has_embeddings and self._embedding_engine is not None:
            try:
                embedding = self._embedding_engine.embed_sync(query)
            except Exception:
                pass

        with self._lock:
            entry = _CacheEntry(
                query=query,
                response=response,
                embedding=embedding,
                timestamp=time.time(),
                source=source,
            )

            self._cache[query] = entry
            self._cache.move_to_end(query)
            self._total_puts += 1

            # Evict oldest entries if over capacity
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def invalidate(self, query: str) -> bool:
        """Remove a specific query from the cache."""
        with self._lock:
            if query in self._cache:
                del self._cache[query]
                return True
            return False

    def clear(self) -> None:
        """Clear the entire cache."""
        with self._lock:
            self._cache.clear()

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        now = time.time()
        removed = 0
        with self._lock:
            expired_keys = [
                k for k, v in self._cache.items()
                if (now - v.timestamp) >= self._ttl
            ]
            for k in expired_keys:
                del self._cache[k]
                removed += 1
        return removed

    # ── Diagnostics ──────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self._total_hits + self._total_misses
        if total == 0:
            return 0.0
        return self._total_hits / total

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": self._enabled,
            "is_semantic": self._has_embeddings,
            "size": self.size,
            "max_size": self._max_size,
            "total_hits": self._total_hits,
            "total_misses": self._total_misses,
            "total_puts": self._total_puts,
            "hit_rate_pct": round(self.hit_rate * 100, 1),
            "similarity_threshold": self._similarity_threshold,
        }


__all__ = ["SemanticCache"]
