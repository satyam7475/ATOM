"""
Short natural-language suggestions from timeline patterns + prediction confidence.

Read-only; emits via event bus from caller (non-blocking).
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("atom.cognition.suggester")


class SuggestionEngine:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        sc = (self._config.get("v7_intelligence") or {}).get("suggestions") or {}
        self._min_pattern_count = int(sc.get("min_pattern_count", 3))
        self._max_suggestions = int(sc.get("max_suggestions", 3))

    def produce(
        self,
        timeline: Any,
        *,
        prediction_accuracy: float = 0.0,
        last_query: str = "",
    ) -> list[str]:
        out: list[str] = []
        try:
            patterns = timeline.detect_patterns(
                window_sec=72 * 3600.0,
                min_count=self._min_pattern_count,
            ) if timeline is not None else []
        except Exception:
            patterns = []

        for p in patterns[:2]:
            label = (p.get("pattern") or "")[:100]
            if not label:
                continue
            out.append(f"You often ask about “{label}” — want a quick recap?")

        try:
            tasks = timeline.get_repeated_tasks(window_sec=48 * 3600.0) if timeline else []
        except Exception:
            tasks = []
        for t in tasks[:1]:
            out.append(f"Do you want me to continue with “{t[:80]}”?")

        if prediction_accuracy >= 0.55 and last_query:
            low = last_query.lower()
            if any(w in low for w in ("continue", "next", "then")):
                out.append("You usually do a follow-up step after this — should I prepare it?")

        result = out[: self._max_suggestions]
        if result:
            logger.info("v7_suggestion count=%d", len(result))
        return result
