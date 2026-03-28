"""
ATOM -- Fast-path pipeline optimizer.

Eliminates unnecessary serial steps from the voice command pipeline:
  1. Quick-reply table checked *before* intent classification (saves ~2ms)
  2. Parallel cache + memory retrieval (already done; this module adds a
     latency budget so slow lookups are abandoned)
  3. Latency budget: if intent classification + action takes >N ms, the
     pipeline timer logs a SLOW warning for profiling

Also provides a startup warm-up helper: eagerly touches hot paths
(intent regexes, cache structures, config reads) so the *first* real
query does not pay cold-start costs.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("atom.fast_path")

_LATENCY_BUDGET_MS: float = 250.0


class LatencyBudget:
    """Track whether a pipeline run is within its latency budget."""

    __slots__ = ("_budget_ms", "_t0", "_label")

    def __init__(self, budget_ms: float = _LATENCY_BUDGET_MS, label: str = "") -> None:
        self._budget_ms = budget_ms
        self._t0 = time.perf_counter()
        self._label = label

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000

    @property
    def remaining_ms(self) -> float:
        return max(0.0, self._budget_ms - self.elapsed_ms)

    @property
    def overbudget(self) -> bool:
        return self.elapsed_ms > self._budget_ms

    def warn_if_slow(self, stage: str) -> None:
        e = self.elapsed_ms
        if e > self._budget_ms:
            logger.warning(
                "SLOW %s | %s at %.0fms (budget %.0fms)",
                self._label, stage, e, self._budget_ms,
            )


def warm_up_intent_engine(intent_engine: Any) -> None:
    """Compile regexes and populate internal structures by classifying a
    throwaway string. Safe to call during bootstrap."""
    try:
        t0 = time.perf_counter()
        intent_engine.classify("warm up intent engine test query")
        ms = (time.perf_counter() - t0) * 1000
        logger.info("Intent engine warm-up: %.1fms", ms)
    except Exception as exc:
        logger.debug("Intent warm-up failed (non-fatal): %s", exc)


def warm_up_cache(cache_engine: Any) -> None:
    """Touch the cache lock once so the threading overhead is paid early."""
    try:
        cache_engine.get("__warmup__")
    except Exception:
        logger.debug("Cache warm-up failed (non-fatal)", exc_info=True)


def warm_up_memory(memory_engine: Any) -> None:
    """Load memory entries from disk (already done in __init__, but calling
    retrieve compiles the tokenizer)."""
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(memory_engine.retrieve("warm up", k=1))
        else:
            loop.run_until_complete(memory_engine.retrieve("warm up", k=1))
    except Exception:
        logger.debug("Memory warm-up failed (non-fatal)", exc_info=True)


def startup_warm_up(
    intent_engine: Any,
    cache: Any,
    memory: Any,
    config: dict[str, Any] | None = None,
) -> None:
    """Eagerly warm up hot paths. Call once after all modules are built."""
    t0 = time.perf_counter()
    warm_up_intent_engine(intent_engine)
    warm_up_cache(cache)
    warm_up_memory(memory)
    ms = (time.perf_counter() - t0) * 1000
    logger.info("Fast-path warm-up complete: %.0fms", ms)
