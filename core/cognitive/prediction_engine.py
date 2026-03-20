"""
ATOM v15 -- Predictive Action Engine.

Predicts the user's next action BEFORE they speak, using:
  1. Time-slot frequency (action counts per hour-of-day)
  2. Transition probability (action A -> action B within 5 min)
  3. Day-of-week weighting (weekday vs weekend patterns)
  4. Recency decay (recent actions weighted higher)

No ML -- pure frequency analysis and conditional probability.

Predictions are emitted as events and consumed by:
  - AutonomyEngine (for proactive suggestions)
  - Dashboard (for display)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.behavior_tracker import BehaviorTracker
    from core.cognitive.behavior_model import BehaviorModel
    from core.memory_engine import MemoryEngine

logger = logging.getLogger("atom.prediction")

_REBUILD_EVERY_N = 100
_TRANSITION_WINDOW_S = 300


class PredictionResult:
    __slots__ = ("action", "target", "confidence", "reason", "time_relevance")

    def __init__(
        self, action: str, target: str, confidence: float,
        reason: str, time_relevance: float,
    ) -> None:
        self.action = action
        self.target = target
        self.confidence = confidence
        self.reason = reason
        self.time_relevance = time_relevance

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "target": self.target,
            "confidence": round(self.confidence, 2),
            "reason": self.reason,
            "time_relevance": round(self.time_relevance, 2),
        }


class PredictionEngine:
    """Predict user's next action using frequency analysis."""

    __slots__ = (
        "_bus", "_behavior", "_memory", "_bmodel", "_config",
        "_task", "_shutdown",
        "_time_freq", "_transitions", "_last_rebuild",
        "_interactions_seen", "_check_interval",
        "_min_confidence", "_max_predictions", "_last_action",
    )

    def __init__(
        self,
        bus: AsyncEventBus,
        behavior: BehaviorTracker,
        memory: MemoryEngine,
        behavior_model: BehaviorModel,
        config: dict | None = None,
    ) -> None:
        self._bus = bus
        self._behavior = behavior
        self._memory = memory
        self._bmodel = behavior_model
        cfg = (config or {}).get("cognitive", {})
        self._config = cfg
        self._check_interval: float = cfg.get("prediction_check_interval_s", 120.0)
        self._min_confidence: float = cfg.get("prediction_min_confidence", 0.6)
        self._max_predictions: int = int(cfg.get("max_predictions", 5))

        self._time_freq: dict[str, Counter] = defaultdict(Counter)
        self._transitions: dict[str, Counter] = defaultdict(Counter)
        self._last_rebuild: float = 0
        self._interactions_seen: int = 0
        self._last_action: str = ""

        self._task: asyncio.Task | None = None
        self._shutdown = asyncio.Event()

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        if not self._config.get("predictions_enabled", True):
            logger.info("Prediction engine disabled via config")
            return
        self._bus.on("intent_classified", self._on_intent)
        self._task = asyncio.create_task(self._run())
        logger.info("Prediction engine started (interval=%.0fs)", self._check_interval)

    def stop(self) -> None:
        self._shutdown.set()
        if self._task and not self._task.done():
            self._task.cancel()
            self._task = None

    async def _run(self) -> None:
        await asyncio.sleep(90.0)
        while not self._shutdown.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(), timeout=self._check_interval,
                )
                break
            except asyncio.TimeoutError:
                pass
            try:
                predictions = self.predict_next()
                if predictions:
                    self._bus.emit_fast(
                        "prediction_ready",
                        predictions=[p.to_dict() for p in predictions],
                    )
            except Exception:
                logger.exception("Prediction cycle error")

    # ── Event Handlers ─────────────────────────────────────────────────

    async def _on_intent(self, intent: str = "", **_kw: Any) -> None:
        if not intent or intent in ("empty", "fallback", "confirm", "deny",
                                     "greeting", "thanks", "status"):
            return

        now = time.time()
        hour = datetime.now().hour
        weekday = datetime.now().weekday()
        is_weekday = weekday < 5
        key = f"{hour}:{'wd' if is_weekday else 'we'}"

        self._time_freq[key][intent] += 1

        if self._last_action:
            self._transitions[self._last_action][intent] += 1
        self._last_action = intent
        self._interactions_seen += 1

        if self._interactions_seen % _REBUILD_EVERY_N == 0:
            self._rebuild_from_history()

    # ── Model Building ─────────────────────────────────────────────────

    def _rebuild_from_history(self) -> None:
        """Rebuild frequency tables from full interaction history."""
        interactions = self._memory._interactions
        if not interactions:
            return

        self._time_freq.clear()
        self._transitions.clear()

        prev_action = ""
        prev_ts = 0.0
        now = time.time()

        for ix in interactions:
            action = ix.get("action", "")
            if not action or action in ("fallback", "empty"):
                continue

            hour = ix.get("hour", 0)
            weekday = ix.get("weekday", 0)
            ts = ix.get("timestamp", 0)
            is_weekday = weekday < 5
            key = f"{hour}:{'wd' if is_weekday else 'we'}"

            recency_weight = 1
            age_days = (now - ts) / 86400
            if age_days < 7:
                recency_weight = 2

            self._time_freq[key][action] += recency_weight

            if prev_action and ts - prev_ts < _TRANSITION_WINDOW_S:
                self._transitions[prev_action][action] += recency_weight

            prev_action = action
            prev_ts = ts

        self._last_rebuild = now
        logger.debug(
            "Prediction model rebuilt: %d time slots, %d transitions",
            len(self._time_freq), len(self._transitions),
        )

    # ── Prediction ─────────────────────────────────────────────────────

    def predict_next(self, max_results: int = 3) -> list[PredictionResult]:
        """Return top predictions for the user's next action."""
        now = datetime.now()
        hour = now.hour
        is_weekday = now.weekday() < 5
        key = f"{hour}:{'wd' if is_weekday else 'we'}"

        candidates: dict[str, float] = {}
        reasons: dict[str, str] = {}

        freq = self._time_freq.get(key)
        if freq:
            total = sum(freq.values())
            for action, count in freq.most_common(10):
                prob = count / total
                if prob >= 0.15:
                    candidates[action] = prob
                    day_type = "weekdays" if is_weekday else "weekends"
                    reasons[action] = (
                        f"You do '{action.replace('_', ' ')}' at {hour}:00 on "
                        f"{day_type} ({count} times)"
                    )

        if self._last_action:
            trans = self._transitions.get(self._last_action)
            if trans:
                total_t = sum(trans.values())
                for action, count in trans.most_common(5):
                    prob = count / total_t * 0.8
                    if action in candidates:
                        candidates[action] = max(candidates[action], prob)
                    elif prob >= 0.2:
                        candidates[action] = prob
                        reasons[action] = (
                            f"After '{self._last_action.replace('_', ' ')}', "
                            f"you often do '{action.replace('_', ' ')}'"
                        )

        results: list[PredictionResult] = []
        for action, conf in sorted(candidates.items(), key=lambda x: x[1], reverse=True):
            if conf < self._min_confidence * 0.5:
                continue
            target = self._guess_target(action, hour, is_weekday)
            results.append(PredictionResult(
                action=action,
                target=target,
                confidence=min(1.0, conf),
                reason=reasons.get(action, "Pattern detected"),
                time_relevance=min(1.0, conf * 1.2),
            ))
            if len(results) >= max_results:
                break

        return results

    def _guess_target(self, action: str, hour: int, is_weekday: bool) -> str:
        """Guess the most likely target for an action."""
        if action != "open_app":
            return ""
        key = f"{hour}:{'wd' if is_weekday else 'we'}"
        entries = self._behavior._entries
        target_counts: Counter = Counter()
        for e in entries:
            if e.get("action") == action and e.get("target"):
                e_hour = e.get("hour", -1)
                if abs(e_hour - hour) <= 1:
                    target_counts[e["target"]] += 1
        if target_counts:
            return target_counts.most_common(1)[0][0]
        return ""

    # ── Queries ────────────────────────────────────────────────────────

    def get_predictions_for_dashboard(self) -> list[dict]:
        preds = self.predict_next(max_results=self._max_predictions)
        return [p.to_dict() for p in preds]

    def format_predictions(self) -> str:
        preds = self.predict_next()
        if not preds:
            return "No strong predictions right now, Boss. I'm still learning your patterns."
        parts = ["Based on your patterns, I think you might:"]
        for i, p in enumerate(preds, 1):
            action_str = p.action.replace("_", " ")
            if p.target:
                action_str += f" ({p.target})"
            parts.append(f"  {i}. {action_str} ({p.confidence:.0%} likely)")
            parts.append(f"     {p.reason}")
        return "\n".join(parts)
