"""
ATOM -- L1 Cache (Nano-Second Fast-Path Memory).

While SecondBrain and MemoryEngine use vector embeddings and disk I/O
(which can take 50-200ms), L1 Cache is a pure in-memory, zero-latency
key-value store for instant recall of highly relevant, recent, or
frequently accessed facts.

Architecture:
  - LRU (Least Recently Used) cache for recent context.
  - LFU (Least Frequently Used) logic for "sticky" facts (e.g., Boss's name).
  - O(1) dictionary lookup.
  - Automatically injected into ContextFusion.

This gives ATOM the "buddy feel" -- it doesn't need to "search its brain"
for things you just talked about or things it uses every day.

Owner: Satyam
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field

logger = logging.getLogger("atom.l1_cache")


@dataclass
class CacheEntry:
    key: str
    value: str
    access_count: int = 1
    last_accessed: float = field(default_factory=time.time)
    is_sticky: bool = False  # Sticky entries are never evicted


class L1Cache:
    """Zero-latency in-memory cache for instant fact recall."""

    __slots__ = ("_max_size", "_cache", "_sticky_cache", "_hits", "_misses")

    def __init__(self, max_size: int = 50) -> None:
        self._max_size = max_size
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._sticky_cache: dict[str, CacheEntry] = {}
        self._hits = 0
        self._misses = 0

    def set(self, key: str, value: str, sticky: bool = False) -> None:
        """Store a fact in the cache."""
        key = key.lower().strip()
        
        if sticky:
            self._sticky_cache[key] = CacheEntry(key=key, value=value, is_sticky=True)
            return

        if key in self._cache:
            # Update existing
            entry = self._cache[key]
            entry.value = value
            entry.access_count += 1
            entry.last_accessed = time.time()
            self._cache.move_to_end(key)
        else:
            # Add new
            if len(self._cache) >= self._max_size:
                # Evict least recently used
                self._cache.popitem(last=False)
            self._cache[key] = CacheEntry(key=key, value=value)

    def get(self, key: str) -> str | None:
        """Retrieve a fact instantly (O(1))."""
        key = key.lower().strip()
        
        if key in self._sticky_cache:
            entry = self._sticky_cache[key]
            entry.access_count += 1
            entry.last_accessed = time.time()
            self._hits += 1
            return entry.value

        if key in self._cache:
            entry = self._cache[key]
            entry.access_count += 1
            entry.last_accessed = time.time()
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.value
        
        self._misses += 1
        return None

    def search_values(self, query: str) -> list[str]:
        """Fast substring search across cached values."""
        q = query.lower()
        results = []
        
        # Check sticky first
        for entry in self._sticky_cache.values():
            if q in entry.value.lower() or q in entry.key:
                results.append(entry.value)
                entry.access_count += 1
                
        # Check LRU
        for key, entry in list(self._cache.items()):
            if q in entry.value.lower() or q in key:
                results.append(entry.value)
                entry.access_count += 1
                self._cache.move_to_end(key)
                
        return results

    def get_summary_for_llm(self) -> str:
        """Dump the most relevant cache contents for prompt injection."""
        parts = []
        
        # Always include sticky facts
        for entry in self._sticky_cache.values():
            parts.append(f"{entry.key}: {entry.value}")
            
        # Include top 5 most recently accessed non-sticky facts
        recent = list(self._cache.values())[-5:]
        for entry in reversed(recent):
            parts.append(f"{entry.key}: {entry.value}")
            
        if not parts:
            return ""
            
        return " | ".join(parts)

    def get_metrics(self) -> dict:
        """Return cache hit/miss statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": total,
            "hit_rate_percent": round(hit_rate, 1),
            "cached_entries": len(self._cache),
            "sticky_entries": len(self._sticky_cache),
            "max_size": self._max_size,
        }

    def clear(self) -> None:
        """Clear non-sticky cache."""
        self._cache.clear()

# Global singleton for instant access anywhere
l1_cache = L1Cache()
