"""
ATOM Intent Engine -- Package entry point.

Split into categorized sub-modules for maintainability.
All external imports continue to work:

    from core.intent_engine import IntentEngine, IntentResult
    from core.intent_engine import GRAMMAR_WORDS, get_grammar_json
    from core.intent_engine import APP_MAP, CLOSE_MAP
"""

from __future__ import annotations

import logging
import time

from .base import (
    IntentResult,
    GRAMMAR_WORDS,
    get_grammar_json,
)
from .app_intents import APP_MAP, CLOSE_MAP
from . import (
    meta_intents,
    info_intents,
    app_intents,
    media_intents,
    system_intents,
    desktop_intents,
    file_intents,
    network_intents,
    os_intents,
    cognitive_intents,
    runtime_mode_intents,
    world_intents,
    memory_recall_intents,
    productivity_intents,
    routine_intents,
)

logger = logging.getLogger("atom.intent")

__all__ = [
    "IntentEngine",
    "IntentResult",
    "GRAMMAR_WORDS",
    "get_grammar_json",
    "APP_MAP",
    "CLOSE_MAP",
]


class IntentEngine:
    """Ultra-fast intent classifier. All methods are synchronous and <5ms."""

    def quick_match(self, text: str) -> str | None:
        """Fast check: does text match a known COMPLETE command?

        Returns the intent name if matched, None otherwise.
        Used by STT for early exit (skip remaining silence timeout).
        """
        t = text.strip()
        if not t:
            return None
        return (
            meta_intents.quick_match(t)
            or info_intents.quick_match(t)
            or media_intents.quick_match(t)
            or desktop_intents.quick_match(t)
            or system_intents.quick_match(t)
            or app_intents.quick_match(t)
        )

    def classify(self, text: str) -> IntentResult:
        return self._classify(text, silent=False)

    def classify_silent(self, text: str) -> IntentResult:
        """Same as classify() but suppresses the per-call INFO log line.

        Used by cold-start warmup paths that invoke the engine dozens of
        times and would otherwise flood the log with "Intent: '…' ->
        fallback/LLM" entries before ATOM has even said hello.
        """
        return self._classify(text, silent=True)

    def _classify(self, text: str, *, silent: bool) -> IntentResult:
        t0 = time.perf_counter()
        text = text.strip()
        if not text:
            return IntentResult("empty", response="I didn't catch that.")

        result = (
            meta_intents.check(text)
            or runtime_mode_intents.check(text)
            or os_intents.check_self_check(text)
            or routine_intents.check(text)
            or memory_recall_intents.check(text)
            or productivity_intents.check(text)
            or info_intents.check(text)
            or world_intents.check(text)
            or system_intents.check(text)
            or media_intents.check(text)
            or desktop_intents.check(text)
            or file_intents.check(text)
            or network_intents.check(text)
            or os_intents.check(text)
            or cognitive_intents.check(text)
            or app_intents.check(text)
        )

        elapsed = (time.perf_counter() - t0) * 1000
        if result:
            if not silent:
                logger.info("Intent: '%s' -> %s (%.1fms)", text[:60], result.intent, elapsed)
            return result

        if not silent:
            logger.info("Intent: '%s' -> fallback/LLM (%.1fms)", text[:60], elapsed)
        return IntentResult("fallback")
