"""
ATOM v15 -- Self-Optimization Engine.

ATOM improves its own resource usage and capabilities over time:
  - Tracks module/feature usage frequency
  - Identifies unused features and suggests disabling them
  - Identifies heavy modules and suggests reducing check intervals
  - Tracks LLM fallback patterns to suggest new intent regexes
  - Records optimization history

SAFETY: Never auto-disables features. Only suggests.
All optimizations require explicit user confirmation.

Persistence: logs/optimizer.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.metrics import MetricsCollector

logger = logging.getLogger("atom.optimizer")

_OPTIMIZER_FILE = Path("logs/optimizer.json")
_MAX_HISTORY = 100
_UNUSED_THRESHOLD_DAYS = 7


class SelfOptimizer:
    """Self-optimization engine that suggests improvements."""

    __slots__ = (
        "_bus", "_metrics", "_config",
        "_feature_usage", "_intent_counts", "_fallback_queries",
        "_suggestions_history", "_last_check",
        "_task", "_shutdown", "_check_interval",
        "_dirty",
    )

    def __init__(
        self,
        bus: AsyncEventBus,
        metrics: MetricsCollector,
        config: dict | None = None,
    ) -> None:
        self._bus = bus
        self._metrics = metrics
        cfg = (config or {}).get("cognitive", {})
        self._config = cfg
        self._check_interval: float = cfg.get("optimizer_check_interval_s", 1800.0)

        self._feature_usage: dict[str, dict] = {}
        self._intent_counts: Counter = Counter()
        self._fallback_queries: list[str] = []
        self._suggestions_history: list[dict] = []
        self._last_check: float = 0
        self._dirty = False

        self._task: asyncio.Task | None = None
        self._shutdown = asyncio.Event()
        self._load()

    # ── Persistence ────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if _OPTIMIZER_FILE.exists():
                data = json.loads(_OPTIMIZER_FILE.read_text(encoding="utf-8"))
                self._feature_usage = data.get("feature_usage", {})
                self._intent_counts = Counter(data.get("intent_counts", {}))
                self._fallback_queries = data.get("fallback_queries", [])[-200:]
                self._suggestions_history = data.get("history", [])[-_MAX_HISTORY:]
                logger.info("Optimizer data loaded")
        except Exception:
            logger.debug("No optimizer data, starting fresh")

    def persist(self) -> None:
        if not self._dirty:
            return
        try:
            _OPTIMIZER_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "feature_usage": self._feature_usage,
                "intent_counts": dict(self._intent_counts.most_common(100)),
                "fallback_queries": self._fallback_queries[-200:],
                "history": self._suggestions_history[-_MAX_HISTORY:],
            }
            _OPTIMIZER_FILE.write_text(
                json.dumps(data, indent=2, default=str), encoding="utf-8",
            )
            self._dirty = False
            logger.debug("Optimizer data persisted")
        except Exception:
            logger.debug("Failed to persist optimizer data", exc_info=True)

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        if not self._config.get("self_optimizer_enabled", True):
            logger.info("Self-optimizer disabled via config")
            return
        self._bus.on("intent_classified", self._on_intent)
        self._task = asyncio.create_task(self._run())
        logger.info("Self-optimizer started (interval=%.0fs)", self._check_interval)

    def stop(self) -> None:
        self._shutdown.set()
        if self._task and not self._task.done():
            self._task.cancel()
            self._task = None
        self.persist()

    async def _run(self) -> None:
        await asyncio.sleep(120.0)
        while not self._shutdown.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(), timeout=self._check_interval,
                )
                break
            except asyncio.TimeoutError:
                pass
            try:
                suggestions = self._analyze()
                if suggestions:
                    self._bus.emit_fast(
                        "optimization_suggestions",
                        suggestions=suggestions,
                    )
            except Exception:
                logger.exception("Self-optimizer analysis error")

    # ── Event Handlers ─────────────────────────────────────────────────

    async def _on_intent(self, intent: str = "", **_kw: Any) -> None:
        if not intent:
            return

        self._intent_counts[intent] += 1

        now = time.time()
        usage = self._feature_usage.setdefault(intent, {
            "first_seen": now, "last_used": now, "count": 0,
        })
        usage["last_used"] = now
        usage["count"] = usage.get("count", 0) + 1
        self._dirty = True

        if intent == "fallback":
            text = _kw.get("text", "")
            if text and len(text) > 5:
                self._fallback_queries.append(text[:200])
                if len(self._fallback_queries) > 200:
                    self._fallback_queries = self._fallback_queries[-200:]

    # ── Analysis ───────────────────────────────────────────────────────

    def _analyze(self) -> list[dict]:
        """Analyze usage patterns and generate suggestions."""
        now = time.time()
        suggestions: list[dict] = []

        snap = self._metrics.snapshot()
        total_queries = snap.get("queries_total", 0)
        if total_queries < 10:
            return suggestions

        for intent, usage in self._feature_usage.items():
            if intent in ("fallback", "empty", "greeting", "thanks", "status"):
                continue
            last_used = usage.get("last_used", now)
            days_unused = (now - last_used) / 86400
            if days_unused > _UNUSED_THRESHOLD_DAYS and usage.get("count", 0) < 5:
                suggestions.append({
                    "type": "unused_feature",
                    "feature": intent,
                    "days_unused": round(days_unused, 1),
                    "message": (
                        f"Feature '{intent.replace('_', ' ')}' hasn't been used in "
                        f"{days_unused:.0f} days. Consider if you still need it."
                    ),
                })

        fallback_count = self._intent_counts.get("fallback", 0)
        if total_queries > 0 and fallback_count / total_queries > 0.3:
            common_fallbacks = Counter()
            for q in self._fallback_queries[-50:]:
                words = q.lower().split()[:3]
                if words:
                    common_fallbacks[" ".join(words)] += 1

            top_patterns = common_fallbacks.most_common(3)
            if top_patterns:
                pattern_strs = [f"'{p}' ({c}x)" for p, c in top_patterns]
                suggestions.append({
                    "type": "high_fallback",
                    "fallback_pct": round(fallback_count / total_queries * 100, 1),
                    "message": (
                        f"{fallback_count / total_queries * 100:.0f}% of queries go to LLM. "
                        f"Common patterns: {', '.join(pattern_strs)}. "
                        f"Teaching ATOM these patterns could improve speed."
                    ),
                })

        cache_hit_pct = snap.get("cache_hit_rate_pct", 0)
        perceived_avg = snap.get("perceived_avg_ms", 0)
        if perceived_avg and float(perceived_avg) > 2500:
            suggestions.append({
                "type": "high_latency",
                "avg_ms": round(float(perceived_avg)),
                "message": (
                    f"Average response time is {float(perceived_avg):.0f}ms. "
                    f"Cache hit rate: {cache_hit_pct:.0f}%. "
                    f"Consider increasing cache TTL or adding local patterns."
                ),
            })

        if suggestions:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "suggestions_count": len(suggestions),
                "total_queries": total_queries,
            }
            self._suggestions_history.append(entry)
            self._dirty = True
            self.persist()

        return suggestions

    # ── Queries ────────────────────────────────────────────────────────

    def format_optimization_report(self) -> str:
        suggestions = self._analyze()
        if not suggestions:
            return "All systems are optimized, Boss. No improvements needed right now."

        parts = [f"I found {len(suggestions)} optimization opportunity{'s' if len(suggestions) > 1 else ''}:"]
        for s in suggestions[:5]:
            parts.append(f"  - {s['message']}")
        return "\n".join(parts)

    def get_feature_usage_summary(self) -> dict:
        top = self._intent_counts.most_common(15)
        return {
            "top_features": [{"feature": f, "count": c} for f, c in top],
            "total_features_used": len(self._feature_usage),
            "fallback_queries_logged": len(self._fallback_queries),
        }
