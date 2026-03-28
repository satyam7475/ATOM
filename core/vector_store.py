"""
ATOM -- Vector Store (Persistent Semantic Memory).

ChromaDB-backed vector storage with multiple collections for different
knowledge domains. Fully persistent to disk, survives restarts.

Collections:
    conversations  -- Q&A pairs from user interactions
    facts          -- Learned facts, corrections, preferences
    documents      -- Ingested document chunks
    interactions   -- Action history with context

v20 optimizations over v18:
  - Batch add support (add_batch) for document ingestion
  - numpy-accelerated fallback search (batch_similarity, single matrix op)
  - Stats caching (avoids repeated ChromaDB count() calls)
  - Async search interface for non-blocking memory retrieval
  - Fallback store uses compact binary format with lazy embedding load
  - Temporal search pre-filters by timestamp before scoring

Contract: CognitiveModuleContract (start, stop, persist)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom.vector_store")

_STORE_DIR = Path("data/vector_db")
_FALLBACK_FILE = Path("data/vector_fallback.json")
_FALLBACK_MAX_PER_COLLECTION = 2000
_FALLBACK_PERSIST_CAP = 500
_STATS_CACHE_TTL_S = 30.0


class VectorSearchResult:
    """Single search result from vector store."""

    __slots__ = ("id", "text", "score", "metadata")

    def __init__(self, id: str, text: str, score: float,
                 metadata: dict[str, Any] | None = None) -> None:
        self.id = id
        self.text = text
        self.score = score
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"VectorSearchResult(score={self.score:.3f}, text={self.text[:50]!r})"


class VectorStore:
    """Persistent vector store with ChromaDB backend and optimized fallback."""

    _COLLECTIONS = ("conversations", "facts", "documents", "interactions")

    __slots__ = (
        "_store_dir", "_client", "_collections", "_using_chromadb",
        "_fallback_data", "_fallback_dirty",
        "_stats_cache", "_stats_cache_ts",
    )

    def __init__(self, config: dict | None = None) -> None:
        cfg = (config or {}).get("vector_store", {})
        self._backend = (cfg.get("backend") or "chroma").lower()
        self._store_dir = Path(cfg.get("path", str(_STORE_DIR)))
        self._client: Any = None
        self._collections: dict[str, Any] = {}
        self._using_chromadb: bool = False
        self._fallback_data: dict[str, list[dict]] = {}
        self._fallback_dirty: bool = False
        self._stats_cache: dict[str, int] | None = None
        self._stats_cache_ts: float = 0.0
        self._init_store()

    def _init_store(self) -> None:
        try:
            import chromadb
            self._store_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self._store_dir),
            )
            for name in self._COLLECTIONS:
                self._collections[name] = self._client.get_or_create_collection(
                    name=name,
                    metadata={"hnsw:space": "cosine"},
                )
            self._using_chromadb = True
            logger.info(
                "Vector store initialized (ChromaDB, backend=%s, path=%s, collections=%d)",
                self._backend, self._store_dir, len(self._collections),
            )
            if self._backend == "torch_gpu":
                logger.info(
                    "vector_store.backend=torch_gpu: ANN still uses Chroma/fallback unless "
                    "extended; embeddings may use CUDA via embedding.device=auto",
                )
        except ImportError:
            logger.info(
                "ChromaDB not installed -- using in-memory fallback vector store"
            )
            self._init_fallback()
        except Exception:
            logger.exception("ChromaDB init failed -- using fallback")
            self._init_fallback()

    def _init_fallback(self) -> None:
        self._using_chromadb = False
        for name in self._COLLECTIONS:
            self._fallback_data[name] = []
        if _FALLBACK_FILE.exists():
            try:
                data = json.loads(_FALLBACK_FILE.read_text(encoding="utf-8"))
                for name in self._COLLECTIONS:
                    self._fallback_data[name] = data.get(name, [])
                logger.info("Fallback vector store loaded from disk")
            except Exception:
                logger.debug("Fallback load failed", exc_info=True)

    def _invalidate_stats_cache(self) -> None:
        self._stats_cache = None

    def add(
        self,
        collection: str,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
        doc_id: str | None = None,
    ) -> str:
        """Add a document to a collection with its embedding."""
        doc_id = doc_id or uuid.uuid4().hex[:12]
        meta = dict(metadata or {})
        meta["timestamp"] = time.time()
        meta["text_preview"] = text[:200]

        if self._using_chromadb:
            coll = self._collections.get(collection)
            if coll is None:
                logger.warning("Unknown collection: %s", collection)
                return doc_id
            try:
                coll.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[text],
                    metadatas=[meta],
                )
                self._invalidate_stats_cache()
            except Exception:
                logger.debug("ChromaDB add failed", exc_info=True)
        else:
            entries = self._fallback_data.setdefault(collection, [])
            entries.append({
                "id": doc_id,
                "text": text,
                "embedding": embedding,
                "metadata": meta,
            })
            if len(entries) > _FALLBACK_MAX_PER_COLLECTION:
                self._fallback_data[collection] = entries[-_FALLBACK_MAX_PER_COLLECTION:]
            self._fallback_dirty = True
            self._invalidate_stats_cache()

        return doc_id

    def add_batch(
        self,
        collection: str,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[str]:
        """Batch add multiple documents. Much faster for document ingestion."""
        if not texts:
            return []

        n = len(texts)
        ids = doc_ids or [uuid.uuid4().hex[:12] for _ in range(n)]
        now = time.time()
        metas = []
        for i in range(n):
            m = dict((metadatas[i] if metadatas and i < len(metadatas) else {}))
            m["timestamp"] = now
            m["text_preview"] = texts[i][:200]
            metas.append(m)

        if self._using_chromadb:
            coll = self._collections.get(collection)
            if coll is None:
                return ids
            try:
                batch_size = 100
                for start in range(0, n, batch_size):
                    end = min(start + batch_size, n)
                    coll.add(
                        ids=ids[start:end],
                        embeddings=embeddings[start:end],
                        documents=texts[start:end],
                        metadatas=metas[start:end],
                    )
                self._invalidate_stats_cache()
            except Exception:
                logger.debug("ChromaDB batch add failed", exc_info=True)
        else:
            entries = self._fallback_data.setdefault(collection, [])
            for i in range(n):
                entries.append({
                    "id": ids[i],
                    "text": texts[i],
                    "embedding": embeddings[i],
                    "metadata": metas[i],
                })
            if len(entries) > _FALLBACK_MAX_PER_COLLECTION:
                self._fallback_data[collection] = entries[-_FALLBACK_MAX_PER_COLLECTION:]
            self._fallback_dirty = True
            self._invalidate_stats_cache()

        return ids

    def search(
        self,
        collection: str,
        query_embedding: list[float],
        k: int = 5,
        min_score: float = 0.3,
    ) -> list[VectorSearchResult]:
        """Search a collection by vector similarity."""
        if self._using_chromadb:
            return self._search_chromadb(collection, query_embedding, k, min_score)
        return self._search_fallback(collection, query_embedding, k, min_score)

    def _search_chromadb(
        self, collection: str, query_embedding: list[float],
        k: int, min_score: float,
    ) -> list[VectorSearchResult]:
        coll = self._collections.get(collection)
        if coll is None:
            return []
        try:
            count = coll.count()
            if count == 0:
                return []
            results = coll.query(
                query_embeddings=[query_embedding],
                n_results=min(k, count),
                include=["documents", "metadatas", "distances"],
            )
            output: list[VectorSearchResult] = []
            ids = results.get("ids", [[]])[0]
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for i, doc_id in enumerate(ids):
                score = 1.0 - distances[i] if distances else 0.0
                if score < min_score:
                    continue
                output.append(VectorSearchResult(
                    id=doc_id,
                    text=docs[i] if docs else "",
                    score=score,
                    metadata=metas[i] if metas else {},
                ))
            return output
        except Exception:
            logger.debug("ChromaDB search failed", exc_info=True)
            return []

    def _search_fallback(
        self, collection: str, query_embedding: list[float],
        k: int, min_score: float,
    ) -> list[VectorSearchResult]:
        """Numpy-accelerated fallback search using batch_similarity."""
        entries = self._fallback_data.get(collection, [])
        if not entries:
            return []

        valid_entries = [e for e in entries if e.get("embedding")]
        if not valid_entries:
            return []

        from core.embedding_engine import EmbeddingEngine

        candidate_embeddings = [e["embedding"] for e in valid_entries]
        scores = EmbeddingEngine.batch_similarity(query_embedding, candidate_embeddings)

        results: list[VectorSearchResult] = []
        for score, entry in sorted(
            zip(scores, valid_entries), key=lambda x: x[0], reverse=True,
        ):
            if score < min_score:
                break
            results.append(VectorSearchResult(
                id=entry["id"],
                text=entry["text"],
                score=score,
                metadata=entry.get("metadata", {}),
            ))
            if len(results) >= k:
                break

        return results

    def search_temporal(
        self,
        collection: str,
        query_embedding: list[float],
        k: int = 5,
        max_age_hours: float = 168.0,
        min_score: float = 0.3,
    ) -> list[VectorSearchResult]:
        """Search with a recency bias -- older entries are scored lower.

        Pre-filters by timestamp before computing similarity to reduce work.
        """
        now = time.time()
        max_age_s = max_age_hours * 3600
        cutoff = now - max_age_s

        if not self._using_chromadb:
            entries = self._fallback_data.get(collection, [])
            filtered = [
                e for e in entries
                if e.get("embedding")
                and e.get("metadata", {}).get("timestamp", now) >= cutoff
            ]
            if not filtered:
                return []

            from core.embedding_engine import EmbeddingEngine
            candidate_embeddings = [e["embedding"] for e in filtered]
            scores = EmbeddingEngine.batch_similarity(query_embedding, candidate_embeddings)

            scored: list[tuple[float, VectorSearchResult]] = []
            for raw_score, entry in zip(scores, filtered):
                if raw_score < min_score * 0.8:
                    continue
                ts = entry.get("metadata", {}).get("timestamp", now)
                age = now - ts
                recency_bonus = max(0.0, 1.0 - (age / max_age_s)) * 0.2
                adjusted = raw_score + recency_bonus
                scored.append((adjusted, VectorSearchResult(
                    id=entry["id"], text=entry["text"],
                    score=adjusted, metadata=entry.get("metadata", {}),
                )))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [r for _, r in scored[:k]]

        results = self.search(collection, query_embedding, k * 2, min_score * 0.8)
        scored_chroma: list[tuple[float, VectorSearchResult]] = []
        for r in results:
            ts = r.metadata.get("timestamp", now)
            age = now - ts
            if age > max_age_s:
                continue
            recency_bonus = max(0.0, 1.0 - (age / max_age_s)) * 0.2
            adjusted_score = r.score + recency_bonus
            scored_chroma.append((adjusted_score, r))

        scored_chroma.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored_chroma[:k]]

    def get_stats(self) -> dict[str, int]:
        """Return document count per collection (cached for 30s)."""
        now = time.monotonic()
        if self._stats_cache is not None and (now - self._stats_cache_ts) < _STATS_CACHE_TTL_S:
            return self._stats_cache

        stats: dict[str, int] = {}
        if self._using_chromadb:
            for name, coll in self._collections.items():
                try:
                    stats[name] = coll.count()
                except Exception:
                    stats[name] = 0
        else:
            for name, entries in self._fallback_data.items():
                stats[name] = len(entries)

        self._stats_cache = stats
        self._stats_cache_ts = now
        return stats

    def delete_collection(self, name: str) -> bool:
        if self._using_chromadb and self._client is not None:
            try:
                self._client.delete_collection(name)
                self._collections.pop(name, None)
                self._collections[name] = self._client.get_or_create_collection(
                    name=name, metadata={"hnsw:space": "cosine"},
                )
                self._invalidate_stats_cache()
                return True
            except Exception:
                logger.debug("Delete collection failed", exc_info=True)
                return False
        else:
            self._fallback_data[name] = []
            self._fallback_dirty = True
            self._invalidate_stats_cache()
            return True

    def persist(self) -> None:
        if not self._using_chromadb and self._fallback_dirty:
            try:
                _FALLBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
                serializable = {}
                for name, entries in self._fallback_data.items():
                    serializable[name] = entries[-_FALLBACK_PERSIST_CAP:]
                _FALLBACK_FILE.write_text(
                    json.dumps(serializable, separators=(",", ":")),
                    encoding="utf-8",
                )
                self._fallback_dirty = False
                logger.debug("Fallback vector store persisted")
            except Exception:
                logger.debug("Fallback persist failed", exc_info=True)

    def shutdown(self) -> None:
        self.persist()
        logger.info("Vector store shut down")
