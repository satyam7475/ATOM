"""
ATOM -- Context-Aware Suggestion Engine.

Generates natural-language follow-up suggestions after each command,
drawing from multiple intelligence sources:
  - TimelineMemory pattern detection
  - PredictionEngine frequency models
  - GoalEngine active goals and next steps
  - SessionMemory recent command context
  - SystemStateEngine active app and time awareness

Emits suggestions via `suggestion_ready` on the bus.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger("atom.cognition.suggester")


class SuggestionEngine:
    """Multi-source suggestion generator."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        sc = (self._config.get("v7_intelligence") or {}).get("suggestions") or {}
        self._min_pattern_count = int(sc.get("min_pattern_count", 3))
        self._max_suggestions = int(sc.get("max_suggestions", 3))
        self._min_confidence = float(sc.get("min_confidence", 0.55))

        self._prediction_engine: Any = None
        self._goal_engine: Any = None
        self._session_memory: Any = None
        self._system_state_engine: Any = None

    def wire(
        self,
        prediction_engine: Any = None,
        goal_engine: Any = None,
        session_memory: Any = None,
        system_state_engine: Any = None,
    ) -> None:
        """Late-bind intelligence sources."""
        if prediction_engine is not None:
            self._prediction_engine = prediction_engine
        if goal_engine is not None:
            self._goal_engine = goal_engine
        if session_memory is not None:
            self._session_memory = session_memory
        if system_state_engine is not None:
            self._system_state_engine = system_state_engine

    def produce(
        self,
        timeline: Any = None,
        *,
        prediction_accuracy: float = 0.0,
        last_query: str = "",
    ) -> list[str]:
        """Generate context-aware suggestions from all intelligence sources."""
        out: list[str] = []

        out.extend(self._from_timeline(timeline))
        out.extend(self._from_predictions())
        out.extend(self._from_goals())
        out.extend(self._from_session_context(last_query))
        out.extend(self._from_system_context())

        if prediction_accuracy >= self._min_confidence and last_query:
            low = last_query.lower()
            if any(w in low for w in ("continue", "next", "then")):
                out.append("You usually do a follow-up step after this -- should I prepare it?")

        seen = set()
        deduped = []
        for s in out:
            key = s.lower()[:50]
            if key not in seen:
                seen.add(key)
                deduped.append(s)

        result = deduped[:self._max_suggestions]
        if result:
            logger.info("suggestions count=%d", len(result))
        return result

    def _from_timeline(self, timeline: Any) -> list[str]:
        """Suggestions from repeated query patterns."""
        if timeline is None:
            return []
        suggestions = []
        try:
            patterns = timeline.detect_patterns(
                window_sec=72 * 3600.0,
                min_count=self._min_pattern_count,
            )
            for p in patterns[:2]:
                label = (p.get("pattern") or "")[:100]
                if label:
                    suggestions.append(
                        f'You often ask about "{label}" -- want a quick recap?',
                    )
        except Exception:
            logger.debug('core cognition suggester optional step failed', exc_info=True)

        try:
            tasks = timeline.get_repeated_tasks(window_sec=48 * 3600.0)
            for t in tasks[:1]:
                suggestions.append(f'Want me to continue with "{t[:80]}"?')
        except Exception:
            logger.debug('core cognition suggester optional step failed', exc_info=True)

        return suggestions

    def _from_predictions(self) -> list[str]:
        """Suggestions from the prediction engine's frequency models."""
        if self._prediction_engine is None:
            return []
        try:
            preds = self._prediction_engine.predict_next(max_results=1)
            suggestions = []
            for pred in preds:
                if pred.confidence >= 0.7:
                    action_str = pred.action.replace("_", " ")
                    target = f" {pred.target}" if pred.target else ""
                    suggestions.append(
                        f"Based on your patterns, want me to {action_str}{target}?",
                    )
            return suggestions
        except Exception:
            return []

    def _from_goals(self) -> list[str]:
        """Suggestions from active goals and their next steps."""
        if self._goal_engine is None:
            return []
        try:
            active = self._goal_engine.get_active_goals()
            if not active:
                return []
            top = active[0]
            title = top.get("title", "")
            steps = top.get("steps", [])
            for step in steps:
                if step.get("status") != "done":
                    step_title = step.get("title", "")
                    return [
                        f'Next step for "{title}": {step_title}. Shall I help?',
                    ]
            return []
        except Exception:
            return []

    def _from_session_context(self, last_query: str) -> list[str]:
        """Suggestions from recent session patterns."""
        if self._session_memory is None:
            return []
        try:
            rep = self._session_memory.detect_repetition()
            if rep:
                return [
                    f'You\'ve asked "{rep}" multiple times. Want me to automate it?',
                ]
        except Exception:
            logger.debug('Action prediction fusion failed', exc_info=True)
        return []

    def _from_system_context(self) -> list[str]:
        """Suggestions based on active app and time of day."""
        if self._system_state_engine is None:
            return []
        try:
            ctx = self._system_state_engine.get_live_context()
            suggestions = []

            hour = datetime.now().hour
            app = ctx.get("active_app", "")

            if app and "code" in app.lower() and 9 <= hour <= 18:
                if not self._session_memory or self._session_memory.command_count < 2:
                    suggestions.append(
                        "Want me to check git status or run your test suite?",
                    )

            return suggestions
        except Exception:
            return []

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            "max_suggestions": self._max_suggestions,
            "min_confidence": self._min_confidence,
            "has_prediction": self._prediction_engine is not None,
            "has_goals": self._goal_engine is not None,
            "has_session": self._session_memory is not None,
            "has_system_state": self._system_state_engine is not None,
        }


__all__ = ["SuggestionEngine"]
