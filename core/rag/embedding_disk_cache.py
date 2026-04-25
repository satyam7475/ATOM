"""Persistent on-disk cache for query embeddings (reduces repeat GPU/CPU embed work)."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom.rag.embed_disk")


def normalize_query_for_cache(q: str) -> str:
    """Normalize for stable keys + similar-query bucketing."""
    t = " ".join(q.lower().split())
    return t[:512] if t else ""


def similar_bucket_key(q: str) -> str:
    """Coarse bucket for near-duplicate queries (token multiset)."""
    import re
    toks = sorted(set(re.findall(r"[a-z0-9]+", q.lower())))
    return hashlib.sha256(" ".join(toks[:48]).encode()).hexdigest()[:24]


class PersistentEmbeddingCache:
    """SQLite-backed embedding vectors; thread-safe."""

    def __init__(
        self,
        path: str = "data/rag_embedding_cache.sqlite",
        *,
        namespace: str = "default",
    ) -> None:
        self._path = Path(path)
        self._namespace = namespace or "default"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_db(self) -> None:
        with self._lock:
            c = self._get_conn()
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    qkey TEXT PRIMARY KEY,
                    bucket TEXT,
                    namespace TEXT DEFAULT 'default',
                    dim INTEGER,
                    vec TEXT,
                    updated REAL
                )
                """
            )
            cols = {
                str(row[1])
                for row in c.execute("PRAGMA table_info(embeddings)").fetchall()
            }
            if "namespace" not in cols:
                c.execute(
                    "ALTER TABLE embeddings "
                    "ADD COLUMN namespace TEXT DEFAULT 'default'",
                )
            c.execute("CREATE INDEX IF NOT EXISTS idx_bucket ON embeddings(bucket)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_namespace ON embeddings(namespace)")
            c.commit()

    def _qkey(self, normalized: str) -> str:
        payload = f"{self._namespace}:{normalized}"
        return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()

    def get(self, query: str) -> list[float] | None:
        n = normalize_query_for_cache(query)
        if not n:
            return None
        k = self._qkey(n)
        with self._lock:
            try:
                cur = self._get_conn().execute(
                    "SELECT vec FROM embeddings WHERE qkey = ? AND namespace = ?",
                    (k, self._namespace),
                )
                row = cur.fetchone()
                if row:
                    return json.loads(row[0])
            except Exception:
                logger.debug("disk embed get failed", exc_info=True)
        # Similar bucket fallback
        b = similar_bucket_key(query)
        with self._lock:
            try:
                cur = self._get_conn().execute(
                    """
                    SELECT vec FROM embeddings
                    WHERE bucket = ? AND namespace = ?
                    ORDER BY updated DESC LIMIT 1
                    """,
                    (b, self._namespace),
                )
                row = cur.fetchone()
                if row:
                    return json.loads(row[0])
            except Exception:
                logger.debug('Hash digest step failed', exc_info=True)
        return None

    def put(self, query: str, vec: list[float]) -> None:
        n = normalize_query_for_cache(query)
        if not n or not vec:
            return
        k = self._qkey(n)
        b = similar_bucket_key(query)
        with self._lock:
            try:
                self._get_conn().execute(
                    """
                    INSERT OR REPLACE INTO embeddings
                        (qkey, bucket, namespace, dim, vec, updated)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (k, b, self._namespace, len(vec), json.dumps(vec), time.time()),
                )
                self._get_conn().commit()
            except Exception:
                logger.debug("disk embed put failed", exc_info=True)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            try:
                n = self._get_conn().execute("SELECT COUNT(*) FROM embeddings").fetchone()
                return {"rows": int(n[0]) if n else 0}
            except Exception:
                return {"rows": 0}
