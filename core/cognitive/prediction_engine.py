"""
ATOM -- Predictive Action Engine.

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
import re
import sys
import time
import urllib.parse
from collections import Counter, defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.behavior_tracker import BehaviorTracker
    from core.cognitive.behavior_model import BehaviorModel
    from core.cognitive_kernel import CognitiveKernel
    from core.memory_engine import MemoryEngine
    from core.rag.prefetch_engine import RagPrefetchEngine
    from cursor_bridge.structured_prompt_builder import StructuredPromptBuilder

logger = logging.getLogger("atom.prediction")

_REBUILD_EVERY_N = 100
_TRANSITION_WINDOW_S = 300
_OPEN_APP_TARGET = re.compile(
    r"\b(?:open|launch|start|run)\s+(.+)", re.I,
)
_SEARCH_TARGET = re.compile(
    r"\b(?:search|google|look\s+up|find\s+online)\s+(.+)", re.I,
)


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
        "_slot_targets", "_global_targets",
        "_preload_enabled", "_preload_min_confidence",
        "_preload_max_items", "_preload_cooldown_s",
        "_preload_timeout_s", "_recent_preloads",
        "_prefetch_engine", "_prompt_builder", "_cognitive_kernel",
        "_app_resolution_cache",
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
        self._preload_enabled: bool = bool(
            cfg.get("prediction_preload_enabled", True),
        )
        self._preload_min_confidence: float = float(
            cfg.get("prediction_preload_min_confidence", 0.82),
        )
        self._preload_max_items: int = int(
            cfg.get("prediction_preload_max_items", 2),
        )
        self._preload_cooldown_s: float = float(
            cfg.get("prediction_preload_cooldown_s", 120.0),
        )
        self._preload_timeout_s: float = float(
            cfg.get("prediction_preload_timeout_s", 2.5),
        )

        self._time_freq: dict[str, Counter] = defaultdict(Counter)
        self._transitions: dict[str, Counter] = defaultdict(Counter)
        self._slot_targets: dict[str, dict[str, Counter]] = defaultdict(
            lambda: defaultdict(Counter),
        )
        self._global_targets: dict[str, Counter] = defaultdict(Counter)
        self._last_rebuild: float = 0
        self._interactions_seen: int = 0
        self._last_action: str = ""
        self._recent_preloads: dict[str, float] = {}
        self._prefetch_engine: RagPrefetchEngine | None = None
        self._prompt_builder: StructuredPromptBuilder | None = None
        self._cognitive_kernel: CognitiveKernel | None = None
        self._app_resolution_cache: dict[str, str] = {}

        self._task: asyncio.Task | None = None
        self._shutdown: asyncio.Event | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        if not self._config.get("predictions_enabled", True):
            logger.info("Prediction engine disabled via config")
            return
        self._shutdown = asyncio.Event()
        self._bus.on("intent_classified", self._on_intent)
        self._bus.on("cursor_query", self._on_cursor_query)
        self._task = asyncio.create_task(self._run())
        logger.info("Prediction engine started (interval=%.0fs)", self._check_interval)

    def stop(self) -> None:
        if self._shutdown is not None:
            self._shutdown.set()
        if self._task and not self._task.done():
            self._task.cancel()
            self._task = None

    def attach_prefetch_engine(self, engine: "RagPrefetchEngine | None") -> None:
        self._prefetch_engine = engine

    def attach_prompt_builder(self, builder: "StructuredPromptBuilder | None") -> None:
        self._prompt_builder = builder

    def attach_cognitive_kernel(self, kernel: "CognitiveKernel | None") -> None:
        self._cognitive_kernel = kernel

    async def _run(self) -> None:
        if self._shutdown is None:
            self._shutdown = asyncio.Event()
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
                    await self.preload_predicted(predictions)
                    self._bus.emit_fast(
                        "prediction_ready",
                        predictions=[p.to_dict() for p in predictions],
                    )
            except Exception:
                logger.exception("Prediction cycle error")

    # ── Event Handlers ─────────────────────────────────────────────────

    async def _on_intent(self, intent: str = "", **kw: Any) -> None:
        if not intent or intent in ("empty", "fallback", "confirm", "deny",
                                     "greeting", "thanks", "status"):
            return

        key = self._current_slot_key()
        target = self._extract_target(
            intent,
            text=str(kw.get("text", "") or ""),
            action_args=kw.get("action_args", {}) or {},
        )
        self._record_observation(intent, target=target, key=key)

    async def _on_cursor_query(self, text: str = "", **_kw: Any) -> None:
        query = str(text or "").strip()
        if len(query) < 4:
            return
        self._record_observation(
            "llm_query",
            target=self._sanitize_target(query, max_len=180),
            key=self._current_slot_key(),
        )

    def _record_observation(self, action: str, *, target: str = "", key: str) -> None:
        self._time_freq[key][action] += 1
        if target:
            self._slot_targets[key][action][target] += 1
            self._global_targets[action][target] += 1

        if self._last_action:
            self._transitions[self._last_action][action] += 1
        self._last_action = action
        self._interactions_seen += 1

        if self._interactions_seen % _REBUILD_EVERY_N == 0:
            self._rebuild_from_history()

    @staticmethod
    def _current_slot_key() -> str:
        now = datetime.now()
        return f"{now.hour}:{'wd' if now.weekday() < 5 else 'we'}"

    @staticmethod
    def _sanitize_target(target: str, *, max_len: int = 120) -> str:
        cleaned = " ".join((target or "").strip().split())
        return cleaned[:max_len]

    def _extract_target(
        self,
        action: str,
        *,
        text: str = "",
        action_args: dict[str, Any] | None = None,
    ) -> str:
        args = action_args or {}
        target = ""

        if action == "open_app":
            target = str(
                args.get("name")
                or args.get("target")
                or args.get("app_name")
                or "",
            )
            if not target and text:
                match = _OPEN_APP_TARGET.search(text)
                if match:
                    target = match.group(1)
            return self._sanitize_target(target, max_len=80)

        if action == "search":
            target = str(args.get("query") or "")
            if not target:
                url = str(args.get("url") or "")
                if url:
                    try:
                        parsed = urllib.parse.urlparse(url)
                        q = urllib.parse.parse_qs(parsed.query).get("q", [])
                        target = str(q[0]) if q else ""
                    except Exception:
                        target = ""
            if not target and text:
                match = _SEARCH_TARGET.search(text)
                if match:
                    target = match.group(1)
            return self._sanitize_target(target, max_len=180)

        if action == "spotlight_search":
            target = str(args.get("query") or "")
            return self._sanitize_target(target, max_len=180)

        if action == "llm_query":
            return self._sanitize_target(text, max_len=180)

        target = str(
            args.get("target")
            or args.get("name")
            or text
            or "",
        )
        return self._sanitize_target(target)

    # ── Model Building ─────────────────────────────────────────────────

    def _rebuild_from_history(self) -> None:
        """Rebuild frequency tables from full interaction history."""
        interactions = self._memory._interactions
        if not interactions:
            return

        preserved_llm_slots = {
            key: int(counter.get("llm_query", 0))
            for key, counter in self._time_freq.items()
            if counter.get("llm_query", 0)
        }
        preserved_llm_targets = {
            key: Counter(action_map.get("llm_query", Counter()))
            for key, action_map in self._slot_targets.items()
            if action_map.get("llm_query")
        }
        preserved_llm_global = Counter(self._global_targets.get("llm_query", Counter()))

        self._time_freq.clear()
        self._transitions.clear()
        self._slot_targets.clear()
        self._global_targets.clear()

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
            target = self._extract_target(action, text=str(ix.get("command", "") or ""))
            if target:
                self._slot_targets[key][action][target] += recency_weight
                self._global_targets[action][target] += recency_weight

            if prev_action and ts - prev_ts < _TRANSITION_WINDOW_S:
                self._transitions[prev_action][action] += recency_weight

            prev_action = action
            prev_ts = ts

        for key, count in preserved_llm_slots.items():
            self._time_freq[key]["llm_query"] += count
        for key, targets in preserved_llm_targets.items():
            self._slot_targets[key]["llm_query"].update(targets)
        if preserved_llm_global:
            self._global_targets["llm_query"].update(preserved_llm_global)

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
        key = f"{hour}:{'wd' if is_weekday else 'we'}"

        slot_targets = self._slot_targets.get(key, {}).get(action)
        if slot_targets:
            return slot_targets.most_common(1)[0][0]

        global_targets = self._global_targets.get(action)
        if global_targets:
            return global_targets.most_common(1)[0][0]

        if action == "open_app":
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

    async def preload_predicted(
        self,
        predictions: list[PredictionResult],
    ) -> list[dict[str, Any]]:
        """Warm lightweight resources for the strongest predictions."""
        if not self._preload_enabled or not predictions:
            return []

        now = time.monotonic()
        selected: list[tuple[str, PredictionResult]] = []
        for pred in sorted(predictions, key=lambda item: item.confidence, reverse=True):
            if pred.confidence < self._preload_min_confidence:
                continue
            key = f"{pred.action}:{(pred.target or '').strip().lower()[:160]}"
            last = self._recent_preloads.get(key, 0.0)
            if last and (now - last) < self._preload_cooldown_s:
                continue
            selected.append((key, pred))
            if len(selected) >= self._preload_max_items:
                break

        reports: list[dict[str, Any]] = []
        for key, pred in selected:
            try:
                report = await asyncio.wait_for(
                    self._preload_prediction(pred),
                    timeout=self._preload_timeout_s,
                )
            except asyncio.TimeoutError:
                logger.debug("Prediction preload timed out for %s", key)
                continue
            except Exception:
                logger.debug("Prediction preload failed for %s", key, exc_info=True)
                continue

            if report is not None:
                reports.append(report)
                self._recent_preloads[key] = time.monotonic()

        if reports:
            self._bus.emit_fast("prediction_preload", items=reports)
        return reports

    async def _preload_prediction(
        self,
        prediction: PredictionResult,
    ) -> dict[str, Any] | None:
        query = self._query_for_prediction(prediction)
        plan = None
        if self._cognitive_kernel is not None and query:
            try:
                plan = self._cognitive_kernel.route(query, allow_cache=False)
            except Exception:
                logger.debug("Prediction preload route failed", exc_info=True)

        resources: list[dict[str, str]] = []

        if prediction.action == "open_app" and prediction.target:
            resolved = await self._warm_app_target(prediction.target)
            if resolved:
                resources.append({
                    "kind": "app",
                    "target": prediction.target,
                    "resolved": resolved,
                })

        if (
            self._prefetch_engine is not None
            and plan is not None
            and getattr(plan, "use_rag", False)
            and not getattr(plan, "reduce_context", False)
            and query
        ):
            self._prefetch_engine.schedule_fire_and_forget(
                [query],
                prediction_accuracy=prediction.confidence,
            )
            resources.append({
                "kind": "rag",
                "target": query[:180],
                "resolved": getattr(plan, "budget_tier", ""),
            })

        if (
            self._prompt_builder is not None
            and query
            and (
                prediction.action == "llm_query"
                or (plan is not None and not getattr(plan, "skip_llm", False))
            )
        ):
            info = self._prompt_builder.precompile(
                query,
                prompt_hint=str(getattr(plan, "prompt_hint", "") or ""),
            )
            resources.append({
                "kind": "prompt",
                "target": query[:180],
                "resolved": "warm" if info.get("tools_cached") else "partial",
            })

        if not resources:
            return None

        return {
            "action": prediction.action,
            "target": prediction.target,
            "confidence": round(prediction.confidence, 2),
            "query": query[:180],
            "budget_tier": str(getattr(plan, "budget_tier", "") or ""),
            "requested_tier": str(getattr(plan, "requested_tier", "") or ""),
            "resources": resources,
        }

    def _query_for_prediction(self, prediction: PredictionResult) -> str:
        target = self._sanitize_target(prediction.target, max_len=180)
        if prediction.action in {"search", "llm_query"} and target:
            return target
        if prediction.action == "open_app" and target:
            return f"open {target}"
        if target:
            return target
        return prediction.action.replace("_", " ")

    async def _warm_app_target(self, target: str) -> str:
        clean_target = self._sanitize_target(target, max_len=80)
        if not clean_target:
            return ""
        cached = self._app_resolution_cache.get(clean_target.lower())
        if cached is not None:
            return cached

        loop = asyncio.get_running_loop()
        resolved = await loop.run_in_executor(None, self._resolve_app_target, clean_target)
        self._app_resolution_cache[clean_target.lower()] = resolved
        return resolved

    def _resolve_app_target(self, target: str) -> str:
        if sys.platform != "darwin":
            return target

        try:
            import AppKit  # type: ignore[import-untyped]

            workspace = AppKit.NSWorkspace.sharedWorkspace()
            path = workspace.fullPathForApplication_(target)
            if path:
                return str(path)
        except Exception:
            logger.debug("Prediction app warm-up via NSWorkspace failed", exc_info=True)

        safe_target = re.sub(r'[^0-9A-Za-z ._+-]', "", target).strip()
        if not safe_target:
            return ""

        query = (
            'kMDItemContentType == "com.apple.application-bundle" && '
            f'(kMDItemDisplayName == "{safe_target}"c || '
            f'kMDItemFSName == "{safe_target}.app"c)'
        )
        try:
            from core.macos.spotlight_engine import SpotlightEngine

            path = SpotlightEngine().find_first_path(query, timeout=2.0)
            if path:
                return path
        except Exception:
            logger.debug("Prediction app warm-up via Spotlight failed", exc_info=True)
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
