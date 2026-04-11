"""TTL caches for query embeddings and retrieval results (RAG)."""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class TTLCache(Generic[T]):
    """Simple thread-safe TTL cache with max entries."""

    def __init__(self, ttl_s: float, max_entries: int = 512) -> None:
        self._ttl = ttl_s
        self._max = max_entries
        self._data: dict[str, tuple[float, T]] = {}
        self._lock = threading.Lock()

    def _key(self, raw: str) -> str:
        return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()

    def get(self, raw: str) -> T | None:
        k = self._key(raw)
        now = time.monotonic()
        with self._lock:
            item = self._data.get(k)
            if item is None:
                return None
            ts, val = item
            if now - ts > self._ttl:
                del self._data[k]
                return None
            return val

    def set(self, raw: str, value: T) -> None:
        k = self._key(raw)
        now = time.monotonic()
        with self._lock:
            self._data[k] = (now, value)
            while len(self._data) > self._max:
                # drop oldest by insertion order approximation
                oldest = next(iter(self._data))
                del self._data[oldest]

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class RagCaches:
    """Embedding vector cache + retrieval result cache."""

    def __init__(
        self,
        embed_ttl_s: float = 600.0,
        retrieval_ttl_s: float = 120.0,
        max_entries: int = 512,
    ) -> None:
        self.embeddings = TTLCache[list[float]](embed_ttl_s, max_entries)
        self.retrieval = TTLCache[list[str]](retrieval_ttl_s, max_entries)

    def get_embedding(self, text: str) -> list[float] | None:
        return self.embeddings.get(text)

    def set_embedding(self, text: str, vec: list[float]) -> None:
        self.embeddings.set(text, vec)

    def get_retrieval(self, query: str) -> list[str] | None:
        return self.retrieval.get(query)

    def set_retrieval(self, query: str, chunks: list[str]) -> None:
        self.retrieval.set(query, chunks)

    def clear(self) -> None:
        self.embeddings.clear()
        self.retrieval.clear()
