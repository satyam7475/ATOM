"""
ATOM — Semantic Cache (Embedding-Aware Response Cache).

v2 upgrade: SQLite-backed persistence so cache survives restarts.

The cache has two layers:

1. **In-memory hot set** (``OrderedDict``, ~256 entries): O(1) exact-match
   hits and fast candidate iteration for semantic search. Writes are
   write-through to SQLite so nothing is ever lost on a crash.
2. **Durable SQLite store** (``data/semantic_cache.db``): keeps up to
   ``persistent_max`` entries across restarts, indexed by the normalised
   query. Warm entries are restored into memory at init.

Time-sensitive queries (weather, time/date, news, stocks, unread emails,
etc.) are auto-tagged and excluded from the cache since their answer
decays within minutes. Everything else is reusable for ``ttl_seconds``.

Backwards compatible: same class + methods (``get/put/invalidate/clear``)
as the in-memory v1, so callers don't change.

Example::

    Query 1 (Monday):   "What is machine learning?"
    Query 2 (Wednesday): "Explain ML to me"
    → semantic similarity ≥ threshold AND not expired → cache HIT in <50ms

Owner: Satyam
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import struct
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom.semantic_cache")


# Queries whose answer is time-sensitive. We never persist these to disk —
# serving yesterday's weather answer at 9am tomorrow is worse than just
# regenerating. In-memory caching inside a single session is still fine.
_VOLATILE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(weather|forecast|temperature|raining|sunny|cloudy)\b",
        r"\b(time|clock|what time|current time|what's the time)\b",
        r"\b(date|today|tomorrow|yesterday|day of the week)\b",
        r"\b(news|headlines|breaking|latest)\b",
        r"\b(stock|price|market|ticker|share price)\b",
        r"\b(unread|inbox|new email|latest email|mail count)\b",
        r"\b(score|match|live|game|result)\b",
        r"\b(battery|cpu|ram|memory|disk|storage)\s*(usage|percent|%|level)?\b",
        r"\b(meeting|calendar|next event|what's next|schedule|agenda)\b",
    )
)


def _is_volatile(query: str) -> bool:
    """Heuristic: is this a query whose answer goes stale within minutes?"""
    if not query:
        return False
    q = query.strip().lower()
    for pat in _VOLATILE_PATTERNS:
        if pat.search(q):
            return True
    return False


def _pack_embedding(vec: list[float]) -> bytes:
    """Pack a float32 vector into compact bytes for SQLite BLOB storage."""
    if not vec:
        return b""
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack_embedding(blob: bytes, dim: int) -> list[float]:
    """Unpack a BLOB produced by ``_pack_embedding`` back into a list."""
    if not blob or dim <= 0:
        return []
    try:
        count = len(blob) // 4
        if count != dim:
            return []
        return list(struct.unpack(f"{count}f", blob))
    except struct.error:
        logger.debug("SemanticCache: corrupt embedding blob, ignoring", exc_info=True)
        return []


@dataclass
class _CacheEntry:
    """A cached query-response pair with embedding."""
    query: str
    response: str
    embedding: list[float]
    timestamp: float
    hit_count: int = 0
    source: str = "local"  # "local" or "cloud:*" or "search"


class SemanticCache:
    """Embedding-aware response cache with optional on-disk persistence.

    Usage::

        cache = SemanticCache(config)
        hit = cache.get(query)
        if hit:
            return hit          # Skip LLM entirely
        cache.put(query, response, source="local")
    """

    # Default directory for the SQLite store. Overridable via
    # ``semantic_cache.db_path`` in config.
    _DEFAULT_DB_PATH = "data/semantic_cache.db"

    # Schema version. Bump this if we change the table layout so stale
    # DBs get rebuilt automatically on next boot instead of crashing at
    # read time.
    _SCHEMA_VERSION = 1

    def __init__(self, config: dict | None = None) -> None:
        cfg = (config or {}).get("semantic_cache", {})
        self._max_size = int(cfg.get("max_size", 256))
        self._ttl = float(cfg.get("ttl_seconds", 600))
        # Tightened from 0.85 — false positives at 0.85 caused ATOM to
        # answer paraphrased but distinct questions with a cached reply
        # for an unrelated query (e.g. "play the song" for "what is newton").
        self._similarity_threshold = float(cfg.get("threshold", 0.92))
        self._enabled = bool(cfg.get("enabled", True))

        # Persistence controls. Off by default for tests; production
        # config turns this on so the cache survives restarts.
        self._persistent = bool(cfg.get("persistent", True))
        self._persistent_max = int(cfg.get("persistent_max", 10_000))
        self._persistent_ttl = float(
            cfg.get("persistent_ttl_seconds", 7 * 24 * 3600.0),  # 7 days
        )
        self._db_path: Path | None = None
        self._conn: sqlite3.Connection | None = None
        self._db_lock = threading.Lock()
        self._dim: int = 0

        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()

        self._embedding_engine: Any = None
        self._has_embeddings = False

        self._total_hits = 0
        self._total_misses = 0
        self._total_puts = 0
        self._persistent_hits = 0

        self._init_embeddings()

        if self._persistent:
            db_path_str = str(cfg.get("db_path", self._DEFAULT_DB_PATH))
            self._open_db(Path(db_path_str))
            restored = self._restore_recent_entries()
            if restored:
                logger.info(
                    "SemanticCache: restored %d entries from %s",
                    restored,
                    self._db_path,
                )

        logger.info(
            "SemanticCache: max=%d, ttl=%.0fs, threshold=%.2f, embeddings=%s, persistent=%s",
            self._max_size,
            self._ttl,
            self._similarity_threshold,
            self._has_embeddings,
            self._persistent,
        )

    # ── Embedding engine wiring ──────────────────────────────────────

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

    # ── SQLite persistence ───────────────────────────────────────────

    def _open_db(self, db_path: Path) -> None:
        """Open (or create) the SQLite store. Silently falls back to
        in-memory mode on any failure so the cache is never fatal.
        """
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                str(db_path),
                check_same_thread=False,
                timeout=2.0,
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """,
            )
            cur = conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'",
            )
            row = cur.fetchone()
            current_ver = int(row[0]) if row and row[0].isdigit() else 0
            if current_ver != self._SCHEMA_VERSION:
                conn.execute("DROP TABLE IF EXISTS cache")
                conn.execute(
                    """
                    CREATE TABLE cache (
                        query TEXT PRIMARY KEY,
                        response TEXT NOT NULL,
                        embedding BLOB,
                        dim INTEGER DEFAULT 0,
                        created_at REAL NOT NULL,
                        last_access REAL NOT NULL,
                        hit_count INTEGER DEFAULT 0,
                        source TEXT DEFAULT 'local'
                    )
                    """,
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_last_access "
                    "ON cache(last_access DESC)",
                )
                conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
                    ("schema_version", str(self._SCHEMA_VERSION)),
                )
            conn.commit()
            self._conn = conn
            self._db_path = db_path
        except Exception:
            logger.warning(
                "SemanticCache: SQLite unavailable at %s, falling back to memory-only",
                db_path,
                exc_info=True,
            )
            self._conn = None
            self._persistent = False

    def _restore_recent_entries(self) -> int:
        """Load the most-recently-accessed entries into the in-memory hot
        set so the first queries after a restart still hit fast.
        """
        if self._conn is None:
            return 0
        now = time.time()
        cutoff = now - self._persistent_ttl
        try:
            with self._db_lock:
                rows = self._conn.execute(
                    """
                    SELECT query, response, embedding, dim,
                           created_at, last_access, hit_count, source
                    FROM cache
                    WHERE last_access >= ?
                    ORDER BY last_access DESC
                    LIMIT ?
                    """,
                    (cutoff, self._max_size),
                ).fetchall()
        except sqlite3.Error:
            logger.debug("SemanticCache: restore failed", exc_info=True)
            return 0

        restored = 0
        with self._lock:
            for q, resp, emb_blob, dim, created, last_access, hit, src in rows:
                if not isinstance(q, str) or not isinstance(resp, str):
                    continue
                if now - created > self._persistent_ttl:
                    continue
                dim = int(dim or 0)
                emb = _unpack_embedding(bytes(emb_blob or b""), dim) if emb_blob else []
                entry = _CacheEntry(
                    query=q,
                    response=resp,
                    embedding=emb,
                    timestamp=float(created),
                    hit_count=int(hit or 0),
                    source=str(src or "local"),
                )
                self._cache[q] = entry
                self._cache.move_to_end(q)
                if emb and self._dim == 0:
                    self._dim = len(emb)
                restored += 1
        return restored

    def _persist_put(self, entry: _CacheEntry) -> None:
        """Write (or update) an entry to SQLite. Silent on failure —
        next put retries automatically.
        """
        if self._conn is None:
            return
        blob = _pack_embedding(entry.embedding) if entry.embedding else b""
        dim = len(entry.embedding)
        try:
            with self._db_lock:
                self._conn.execute(
                    """
                    INSERT INTO cache(query, response, embedding, dim,
                                      created_at, last_access, hit_count, source)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(query) DO UPDATE SET
                        response = excluded.response,
                        embedding = excluded.embedding,
                        dim = excluded.dim,
                        last_access = excluded.last_access,
                        hit_count = cache.hit_count + 1,
                        source = excluded.source
                    """,
                    (
                        entry.query,
                        entry.response,
                        blob,
                        dim,
                        entry.timestamp,
                        entry.timestamp,
                        entry.hit_count,
                        entry.source,
                    ),
                )
                self._conn.commit()
        except sqlite3.Error:
            logger.debug("SemanticCache: persist_put failed", exc_info=True)

    def _persist_touch(self, query: str) -> None:
        """Bump last_access for a hit so LRU reflects recency."""
        if self._conn is None:
            return
        try:
            with self._db_lock:
                self._conn.execute(
                    "UPDATE cache SET last_access = ?, hit_count = hit_count + 1 "
                    "WHERE query = ?",
                    (time.time(), query),
                )
                self._conn.commit()
        except sqlite3.Error:
            logger.debug("SemanticCache: persist_touch failed", exc_info=True)

    def _persist_delete(self, query: str) -> None:
        if self._conn is None:
            return
        try:
            with self._db_lock:
                self._conn.execute(
                    "DELETE FROM cache WHERE query = ?", (query,),
                )
                self._conn.commit()
        except sqlite3.Error:
            logger.debug("SemanticCache: persist_delete failed", exc_info=True)

    def _persist_evict_if_needed(self) -> None:
        """Keep the on-disk store bounded. Runs opportunistically
        (called from cleanup_expired) rather than on every put to avoid
        hot-path overhead.
        """
        if self._conn is None:
            return
        try:
            with self._db_lock:
                cutoff_count = self._conn.execute(
                    "SELECT COUNT(*) FROM cache",
                ).fetchone()
                count = int(cutoff_count[0]) if cutoff_count else 0
                if count <= self._persistent_max:
                    return
                over = count - self._persistent_max
                self._conn.execute(
                    """
                    DELETE FROM cache WHERE query IN (
                        SELECT query FROM cache ORDER BY last_access ASC LIMIT ?
                    )
                    """,
                    (over,),
                )
                self._conn.commit()
                logger.debug("SemanticCache: evicted %d LRU entries", over)
        except sqlite3.Error:
            logger.debug("SemanticCache: eviction failed", exc_info=True)

    # ── Public API (unchanged contract) ──────────────────────────────

    def get(self, query: str) -> str | None:
        """Look up a query in the semantic cache.

        First tries exact match, then falls back to semantic similarity.
        Returns the cached response or ``None``.
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
                self._persist_touch(query)
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
                    self._persist_touch(best_key)
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
        """Store a query-response pair in the semantic cache.

        Volatile queries (weather, time, news, stock prices, etc.) are
        silently NOT stored on disk — they still live in the in-memory
        hot set for the duration of the session but never leak into the
        next boot.
        """
        if not self._enabled or not query or not response:
            return

        query = query.strip()
        if not query or not response.strip():
            return

        volatile = _is_volatile(query)

        # Compute embedding
        embedding: list[float] = []
        if self._has_embeddings and self._embedding_engine is not None:
            try:
                embedding = self._embedding_engine.embed_sync(query)
                if embedding and self._dim == 0:
                    self._dim = len(embedding)
            except Exception:
                logger.debug('Embedding sync call failed', exc_info=True)

        entry = _CacheEntry(
            query=query,
            response=response,
            embedding=embedding,
            timestamp=time.time(),
            source=source,
        )

        with self._lock:
            self._cache[query] = entry
            self._cache.move_to_end(query)
            self._total_puts += 1

            # Evict in-memory oldest entries if over capacity
            while len(self._cache) > self._max_size:
                evicted_key, _ = self._cache.popitem(last=False)
                # Don't delete from disk on in-memory eviction — the
                # hot-set is just a working window; disk eviction is
                # size-gated in _persist_evict_if_needed.
                del evicted_key  # quiet linter

        # Persist outside the hot-set lock so SQLite I/O never blocks
        # a concurrent ``get``.
        if self._persistent and not volatile:
            self._persist_put(entry)

    def invalidate(self, query: str) -> bool:
        """Remove a specific query from the cache."""
        with self._lock:
            if query in self._cache:
                del self._cache[query]
                self._persist_delete(query)
                return True
        return False

    def clear(self) -> None:
        """Clear the entire cache."""
        with self._lock:
            self._cache.clear()
        if self._conn is not None:
            try:
                with self._db_lock:
                    self._conn.execute("DELETE FROM cache")
                    self._conn.commit()
            except sqlite3.Error:
                logger.debug("SemanticCache: clear failed", exc_info=True)

    def cleanup_expired(self) -> int:
        """Remove expired entries from memory + disk. Returns count removed."""
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

        if self._conn is not None:
            try:
                with self._db_lock:
                    cutoff = now - self._persistent_ttl
                    cur = self._conn.execute(
                        "DELETE FROM cache WHERE created_at < ?",
                        (cutoff,),
                    )
                    self._conn.commit()
                    disk_removed = cur.rowcount or 0
                    if disk_removed:
                        logger.info(
                            "SemanticCache: purged %d expired disk entries",
                            disk_removed,
                        )
            except sqlite3.Error:
                logger.debug("SemanticCache: cleanup failed", exc_info=True)

        self._persist_evict_if_needed()
        return removed

    # ── Diagnostics ──────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def persistent_size(self) -> int:
        """Approximate size of the durable store. 0 if persistence off."""
        if self._conn is None:
            return 0
        try:
            with self._db_lock:
                row = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()
            return int(row[0]) if row else 0
        except sqlite3.Error:
            return 0

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
            "persistent": self._persistent,
            "persistent_size": self.persistent_size,
            "persistent_max": self._persistent_max,
            "db_path": str(self._db_path) if self._db_path else "",
            "total_hits": self._total_hits,
            "total_misses": self._total_misses,
            "total_puts": self._total_puts,
            "hit_rate_pct": round(self.hit_rate * 100, 1),
            "similarity_threshold": self._similarity_threshold,
        }

    def close(self) -> None:
        """Close the SQLite connection cleanly. Safe to call twice."""
        conn = self._conn
        self._conn = None
        if conn is not None:
            try:
                with self._db_lock:
                    conn.commit()
                    conn.close()
            except sqlite3.Error:
                logger.debug("SemanticCache: close failed", exc_info=True)


__all__ = ["SemanticCache"]
