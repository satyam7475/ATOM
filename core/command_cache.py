"""
ATOM -- Command result cache for instant repeat commands.

LRU cache that stores recent intent classification results.
If the same command text is seen again within the TTL window,
the cached IntentResult is returned instantly (~0ms) without
re-running the intent engine.

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


class CommandCache:
    """LRU cache for IntentResult objects keyed by normalized text."""

    __slots__ = ("_store", "_max_size", "_ttl")

    def __init__(self, max_size: int = DEFAULT_MAX_SIZE,
                 ttl: float = DEFAULT_TTL_S) -> None:
        self._store: OrderedDict[str, tuple[float, object]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl

    def get(self, text: str) -> object | None:
        """Return cached IntentResult if fresh, else None."""
        key = text.lower().strip()
        entry = self._store.get(key)
        if entry is None:
            return None
        ts, result = entry
        if (time.monotonic() - ts) > self._ttl:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        logger.info("Command cache HIT: '%s'", key[:40])
        return result

    def put(self, text: str, result: object) -> None:
        """Cache an IntentResult (skip fallback/confirm/deny intents)."""
        intent = getattr(result, "intent", None)
        if intent in _SKIP_INTENTS:
            return
        key = text.lower().strip()
        self._store[key] = (time.monotonic(), result)
        self._store.move_to_end(key)
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def put_intent_key(self, intent_key: str, result: object) -> None:
        """Cache by intent key for intent-based reuse (e.g. info:time, info:cpu)."""
        if not intent_key:
            return
        self._store[intent_key] = (time.monotonic(), result)
        self._store.move_to_end(intent_key)
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    @property
    def size(self) -> int:
        return len(self._store)

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
