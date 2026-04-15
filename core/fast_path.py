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


def warm_up_system_state(system_state_engine: Any) -> None:
    """Take the first snapshot so the first command has context."""
    try:
        t0 = time.perf_counter()
        system_state_engine._capture()
        ms = (time.perf_counter() - t0) * 1000
        logger.info("System state warm-up: %.1fms", ms)
    except Exception:
        logger.debug("System state warm-up failed (non-fatal)", exc_info=True)


def startup_warm_up(
    intent_engine: Any,
    cache: Any,
    memory: Any,
    config: dict[str, Any] | None = None,
    system_state_engine: Any = None,
) -> None:
    """Eagerly warm up hot paths. Call once after all modules are built."""
    t0 = time.perf_counter()
    warm_up_intent_engine(intent_engine)
    warm_up_cache(cache)
    warm_up_memory(memory)
    if system_state_engine is not None:
        warm_up_system_state(system_state_engine)
    ms = (time.perf_counter() - t0) * 1000
    logger.info("Fast-path warm-up complete: %.0fms", ms)


class ParallelPipeline:
    """Overlap STT finalization with intent pre-classification.

    When STT emits a partial transcript with high confidence, the pipeline
    can speculatively start intent classification and cache lookup in
    parallel, so by the time ``speech_final`` fires, the intent result
    is already available.
    """

    def __init__(self, intent_engine: Any, cache: Any) -> None:
        self._intent = intent_engine
        self._cache = cache
        self._prefetched: dict[str, Any] = {}
        self._prefetch_count = 0
        self._hit_count = 0

    async def on_speech_partial(self, text: str = "", confidence: float = 0.0, **_kw: Any) -> None:
        """Called on partial STT results. Pre-classifies if confidence is high."""
        import asyncio

        text = (text or "").strip()
        if not text or len(text) < 4 or confidence < 0.7:
            return

        norm = text.lower().strip()
        if norm in self._prefetched:
            return

        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, self._intent.classify, text)
            self._prefetched[norm] = result
            self._prefetch_count += 1
            if len(self._prefetched) > 20:
                oldest = next(iter(self._prefetched))
                del self._prefetched[oldest]
        except Exception:
            pass

    def get_prefetched(self, text: str) -> Any:
        """Retrieve a pre-classified intent if available."""
        norm = text.lower().strip()
        result = self._prefetched.pop(norm, None)
        if result is not None:
            self._hit_count += 1
        return result

    def get_diagnostics(self) -> dict[str, int]:
        return {
            "prefetch_count": self._prefetch_count,
            "hit_count": self._hit_count,
            "pending": len(self._prefetched),
        }
