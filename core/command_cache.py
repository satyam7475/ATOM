"""
ATOM -- Command result cache for instant repeat commands.

LRU cache that stores recent intent classification results.
If the same command text is seen again within the TTL window,
the cached IntentResult is returned instantly (~0ms) without
re-running the intent engine. Dynamic info intents stay uncached
so answers like time/cpu/ram remain fresh.

Designed for rapid-fire commands like:
    "open chrome" -> "open chrome" -> "mute" -> "mute"
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict

logger = logging.getLogger("atom.command_cache")

DEFAULT_MAX_SIZE = 64
DEFAULT_TTL_S = 120.0

_SKIP_INTENTS = frozenset({"fallback", "confirm", "deny"})
_DYNAMIC_INFO_INTENTS = frozenset({
    "time", "date", "cpu", "ram", "battery", "disk",
    "system_info", "ip", "wifi", "uptime", "top_processes",
    "resource_report", "resource_trend", "app_history",
    "show_reminders", "self_diagnostic", "system_analyze",
    "self_check", "behavior_report",
})


def _should_cache_result(result: object) -> bool:
    intent = getattr(result, "intent", None)
    return intent not in _SKIP_INTENTS and intent not in _DYNAMIC_INFO_INTENTS


class CommandCache:
    """LRU cache for IntentResult objects keyed by normalized text."""

    __slots__ = ("_store", "_max_size", "_ttl", "_hits", "_misses")

    def __init__(self, max_size: int = DEFAULT_MAX_SIZE,
                 ttl: float = DEFAULT_TTL_S) -> None:
        self._store: OrderedDict[str, tuple[float, object]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl
        self._hits = 0
        self._misses = 0

    def get(self, text: str) -> object | None:
        """Return cached IntentResult if fresh, else None."""
        key = text.lower().strip()
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None
        ts, result = entry
        if (time.monotonic() - ts) > self._ttl:
            del self._store[key]
            self._misses += 1
            return None
        self._store.move_to_end(key)
        self._hits += 1
        logger.info("Command cache HIT: '%s'", key[:40])
        return result

    def put(self, text: str, result: object, *, force: bool = False) -> None:
        """Cache an IntentResult when it is safe to reuse.

        ``force=True`` bypasses the ``_should_cache_result`` filter so
        the cold-start optimizer can pre-cache *intent classifications*
        for dynamic-info intents (``self_check``, ``time``, ...). This
        is safe because the router treats ``IntentResult`` as a pure
        classification — the actual response is rendered fresh each
        turn by the dispatch handler — so caching ``self_check`` skips
        the ~150 ms re-classify pass without serving a stale answer.
        """
        if not force and not _should_cache_result(result):
            return
        key = text.lower().strip()
        self._store[key] = (time.monotonic(), result)
        self._store.move_to_end(key)
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def put_intent_key(self, intent_key: str, result: object) -> None:
        """Cache by intent key when the result is safe to reuse."""
        if not intent_key or not _should_cache_result(result):
            return
        self._store[intent_key] = (time.monotonic(), result)
        self._store.move_to_end(intent_key)
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    @property
    def size(self) -> int:
        return len(self._store)

    def get_metrics(self) -> dict:
        """Return cache hit/miss statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": total,
            "hit_rate_percent": round(hit_rate, 1),
            "cached_entries": len(self._store),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl,
        }

    def clear(self) -> None:
        self._store.clear()


_instance: CommandCache | None = None


def get_command_cache(max_size: int = DEFAULT_MAX_SIZE,
                      ttl: float = DEFAULT_TTL_S) -> CommandCache:
    """Singleton accessor."""
    global _instance
    if _instance is None:
        _instance = CommandCache(max_size, ttl)
    return _instance
