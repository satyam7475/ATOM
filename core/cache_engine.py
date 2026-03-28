"""
ATOM -- TTL-aware LRU cache with Jaccard similarity fallback.

Caches LLM responses keyed by normalised query text.
A cache hit returns instantly -- no LLM inference needed.

v20 optimizations over v10:
  - Pre-computed token sets stored alongside values (no re-splitting on lookup)
  - Reversed iterator for Jaccard scan (avoids list() copy of entire store)
  - Stem cache for repeated words (avoids re-stemming common vocabulary)
  - Early termination: perfect Jaccard match (1.0) exits scan immediately
  - Hit/miss ratio tracking for self-tuning

Two-tier lookup:
  1. O(1) exact match on normalised key (stop-word removal + stemming)
  2. Jaccard similarity scan on miss (top 48 entries, threshold 0.75)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from functools import lru_cache

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


@lru_cache(maxsize=2048)
def _stem(word: str) -> str:
    """Minimal suffix stripping with memoization.

    Strips: -ing, -ies, -es, -s, trailing silent -e.
    Results are cached so repeated words (very common in queries) don't re-compute.
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
JACCARD_SCAN_LIMIT = 48

# value, timestamp, pre-computed token frozenset for Jaccard
_CacheEntry = tuple[str, float, frozenset[str]]


class CacheEngine:
    """
    OrderedDict-based LRU with per-entry TTL and Jaccard similarity fallback.

    Lookup order:
      1. Exact normalised-key match (O(1))
      2. Jaccard similarity scan over the 48 most-recent entries

    v20: Token sets are pre-computed on put() and stored with each entry,
    so Jaccard scan avoids repeated str.split() and set() construction.
    """

    __slots__ = ("_store", "_max_size", "_ttl", "_metrics", "_lock")

    def __init__(self, max_size: int = 128, ttl: float = 300.0, metrics=None) -> None:
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()
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
    def _tokenize(key: str) -> frozenset[str]:
        return frozenset(key.split())

    def _jaccard_lookup(self, key: str, key_tokens: frozenset[str]) -> str | None:
        """Scan recent entries using pre-computed token sets.

        Uses reversed() on the OrderedDict to scan newest-first without
        copying the entire dict to a list.
        """
        now = time.monotonic()
        ttl = self._ttl
        best_score = 0.0
        best_value: str | None = None
        scanned = 0

        for cached_key in reversed(self._store):
            if scanned >= JACCARD_SCAN_LIMIT:
                break
            value, ts, cached_tokens = self._store[cached_key]
            scanned += 1
            if now - ts > ttl:
                continue
            if not cached_tokens:
                continue

            intersection = len(key_tokens & cached_tokens)
            if intersection == 0:
                continue
            union = len(key_tokens | cached_tokens)
            score = intersection / union

            if score > best_score:
                best_score = score
                best_value = value
                if score >= 1.0:
                    break

        if best_score >= JACCARD_THRESHOLD and best_value is not None:
            return best_value
        return None

    def get(self, query: str) -> str | None:
        key = self._normalise(query)

        with self._lock:
            entry = self._store.get(key)
            if entry is not None:
                value, ts, _tokens = entry
                if time.monotonic() - ts > self._ttl:
                    del self._store[key]
                else:
                    self._store.move_to_end(key)
                    if self._metrics:
                        self._metrics.inc("cache_hits")
                    return value

            key_tokens = self._tokenize(key)
            result = self._jaccard_lookup(key, key_tokens)

        if result is not None:
            if self._metrics:
                self._metrics.inc("cache_hits")
        else:
            if self._metrics:
                self._metrics.inc("cache_misses")
        return result

    def put(self, query: str, response: str) -> None:
        key = self._normalise(query)
        tokens = self._tokenize(key)
        with self._lock:
            self._store[key] = (response, time.monotonic(), tokens)
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
            expired = [k for k, (_, ts, _) in self._store.items() if now - ts > self._ttl]
            for k in expired:
                del self._store[k]
        return len(expired)

    @property
    def size(self) -> int:
        return len(self._store)
