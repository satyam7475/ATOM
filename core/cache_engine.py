"""
TTL-aware LRU cache for ATOM v10 with Jaccard similarity fallback.

Caches Cursor responses keyed by normalised query text.
A cache hit returns instantly -- no Cursor round-trip needed.

Two-tier lookup:
  1. O(1) exact match on normalised key (stop-word removal + stemming)
  2. Jaccard similarity scan on miss (top 32 entries, threshold 0.75)

The Jaccard layer prevents false hits: "install docker" vs "uninstall docker"
gives Jaccard = 1/3 = 0.33, correctly rejected despite stem overlap.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import OrderedDict

logger = logging.getLogger("atom.cache")

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "do", "does", "did", "has", "have", "had",
    "what", "how", "who", "where", "when", "which",
    "can", "could", "would", "should", "will",
    "i", "me", "my", "you", "your", "we", "our",
    "to", "of", "in", "on", "at", "for", "with", "about",
    "it", "its", "this", "that", "these", "those",
})


def _stem(word: str) -> str:
    """Minimal suffix stripping -- not a full stemmer, just enough for cache keys.

    Strips: -ing, -ies, -es, -s, trailing silent -e.
    This makes "configuring" / "configure" / "configured" converge to "configur".
    """
    if len(word) <= 3:
        return word
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("ied") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith("es") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    if word.endswith("e") and len(word) > 4:
        return word[:-1]
    return word


JACCARD_THRESHOLD = 0.75
JACCARD_SCAN_LIMIT = 32


class CacheEngine:
    """
    OrderedDict-based LRU with per-entry TTL and Jaccard similarity fallback.

    Using OrderedDict instead of functools.lru_cache because we need:
    - explicit TTL expiry
    - key-level invalidation
    - runtime inspection (size, keys)

    Lookup order:
      1. Exact normalised-key match (O(1))
      2. Jaccard similarity scan over the 32 most-recent entries (O(32))
    """

    __slots__ = ("_store", "_max_size", "_ttl", "_metrics", "_lock")

    def __init__(self, max_size: int = 128, ttl: float = 300.0, metrics=None) -> None:
        self._store: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl
        self._metrics = metrics
        self._lock = threading.Lock()

    @staticmethod
    def _normalise(query: str) -> str:
        words = query.lower().split()
        content = [_stem(w) for w in words if w not in _STOP_WORDS]
        return " ".join(content) if content else " ".join(words)

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        """Token-level Jaccard similarity between two normalised keys."""
        set_a = set(a.split())
        set_b = set(b.split())
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union else 0.0

    def _jaccard_lookup(self, key: str) -> str | None:
        """Scan the most-recent entries for a Jaccard match above threshold."""
        now = time.monotonic()
        best_score = 0.0
        best_value: str | None = None

        items = list(self._store.items())
        for cached_key, (value, ts) in items[-JACCARD_SCAN_LIMIT:]:
            if now - ts > self._ttl:
                continue
            score = self._jaccard(key, cached_key)
            if score > best_score:
                best_score = score
                best_value = value

        if best_score >= JACCARD_THRESHOLD and best_value is not None:
            return best_value
        return None

    def get(self, query: str) -> str | None:
        key = self._normalise(query)

        with self._lock:
            entry = self._store.get(key)
            if entry is not None:
                value, ts = entry
                if time.monotonic() - ts > self._ttl:
                    del self._store[key]
                else:
                    self._store.move_to_end(key)
                    if self._metrics:
                        self._metrics.inc("cache_hits")
                    return value

            result = self._jaccard_lookup(key)
        if result is not None:
            if self._metrics:
                self._metrics.inc("cache_hits")
        else:
            if self._metrics:
                self._metrics.inc("cache_misses")
        return result

    def put(self, query: str, response: str) -> None:
        key = self._normalise(query)
        with self._lock:
            self._store[key] = (response, time.monotonic())
            self._store.move_to_end(key)
            if len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def invalidate(self, query: str) -> None:
        key = self._normalise(query)
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def purge_expired(self) -> int:
        """Remove all expired entries. Returns count of purged entries."""
        now = time.monotonic()
        with self._lock:
            expired = [k for k, (_, ts) in self._store.items() if now - ts > self._ttl]
            for k in expired:
                del self._store[k]
        return len(expired)

    @property
    def size(self) -> int:
        return len(self._store)
