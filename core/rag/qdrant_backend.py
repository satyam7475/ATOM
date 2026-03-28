"""Optional local Qdrant backend for fast vector search (graceful fallback)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom.rag.qdrant")


@dataclass
class QdrantHit:
    text: str
    score: float
    payload: dict[str, Any]


class QdrantRagBackend:
    """Thin wrapper; disabled if qdrant-client missing or init fails."""

    def __init__(self, persist_path: str = "data/qdrant_rag", collection: str = "atom_rag") -> None:
        self._path = Path(persist_path)
        self._collection = collection
        self._client: Any = None
        self._enabled = False
        self._dim: int | None = None
        self._init()

    def _init(self) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            self._path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(self._path))
            cols = self._client.get_collections().collections
            names = {c.name for c in cols}
            if self._collection not in names:
                self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )
            self._enabled = True
            self._dim = 384
            logger.info("Qdrant RAG backend ready at %s", self._path)
        except Exception:
            logger.debug("Qdrant backend unavailable", exc_info=True)
            self._client = None
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def upsert(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        if not self._enabled or not self._client or not texts:
            return
        try:
            from qdrant_client.models import PointStruct

            import uuid

            mds = metadatas or [{}] * len(texts)
            pid = ids or [uuid.uuid4().hex[:16] for _ in texts]
            points = []
            for i, (t, e, m) in enumerate(zip(texts, embeddings, mds)):
                meta = dict(m)
                meta["text"] = t
                meta["timestamp"] = time.time()
                points.append(PointStruct(id=pid[i], vector=e, payload=meta))
            self._client.upsert(collection_name=self._collection, points=points)
        except Exception:
            logger.debug("Qdrant upsert failed", exc_info=True)

    def search(
        self,
        query_embedding: list[float],
        limit: int = 8,
        score_threshold: float = 0.25,
    ) -> list[QdrantHit]:
        if not self._enabled or not self._client:
            return []
        try:
            res = self._client.search(
                collection_name=self._collection,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
            )
            out: list[QdrantHit] = []
            for r in res:
                pl = r.payload or {}
                txt = pl.get("text") or pl.get("text_preview") or ""
                out.append(QdrantHit(text=txt, score=float(r.score), payload=pl))
            return out
        except Exception:
            logger.debug("Qdrant search failed", exc_info=True)
            return []
