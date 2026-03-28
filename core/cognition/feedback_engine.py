"""
Feedback loop for prediction quality, prefetch effectiveness, and graph vs RAG ratio.

Thread-safe; does not touch SecurityPolicy or action execution.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("atom.cognition.feedback")


def _norm_key(text: str) -> str:
    t = (text or "").strip().lower()
    return " ".join(t.split())[:120]


@dataclass
class OutcomeRecord:
    query: str
    prediction: str | None
    result: str
    success: bool
    ts: float = field(default_factory=time.time)


class FeedbackEngine:
    """Rolling outcomes + metrics; adjusts soft weights for predictor ordering."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        v7 = self._config.get("v7_intelligence") or {}
        fc = v7.get("feedback") or {}
        self._max_records = int(fc.get("max_records", 2000))
        self._records: deque[OutcomeRecord] = deque(maxlen=self._max_records)
        self._lock = threading.RLock()

        self._win_50 = int(fc.get("rolling_window_50", 50))
        self._win_100 = int(fc.get("rolling_window_100", 100))
        self._recent_success: deque[bool] = deque(maxlen=self._win_100)

        self._prediction_hits = 0
        self._prediction_total = 0
        self._prefetch_hits = 0
        self._prefetch_total = 0
        self._prefetch_scheduled = 0
        self._override_count = 0
        self._graph_hits = 0
        self._rag_fallbacks = 0
        self._graph_misses = 0

        self._priority_weights: dict[str, float] = dict(fc.get("initial_weights") or {})
        self._learn_rate = float(fc.get("learn_rate", 0.04))
        self._min_query_chars = int(fc.get("min_query_chars", 4))
        self._learn_confidence_threshold = float(fc.get("learn_confidence_threshold", 0.35))

    def _health_cfg(self) -> dict[str, Any]:
        return (self._config.get("v7_intelligence") or {}).get("health") or {}

    def _feedback_cfg(self) -> dict[str, Any]:
        return (self._config.get("v7_intelligence") or {}).get("feedback") or {}

    def evaluate_actual_vs_predictions(
        self,
        actual_query: str,
        predicted: list[str],
    ) -> None:
        """Compare the new user query to last turn's prefetch predictions (learning loop)."""
        if not predicted:
            return
        if self._should_ignore_query_for_learning(actual_query):
            return
        p0 = predicted[0]
        hit = _prediction_matched(actual_query, p0)
        self.record_outcome(actual_query, p0, "actual", hit)

    def _should_ignore_query_for_learning(self, query: str) -> bool:
        fc = self._feedback_cfg()
        q = (query or "").strip()
        if len(q) < int(fc.get("min_query_chars", self._min_query_chars)):
            return True
        return False

    def record_outcome(
        self,
        query: str,
        prediction: str | None,
        result: str,
        success: bool,
    ) -> None:
        if self._should_ignore_query_for_learning(query or ""):
            return
        rec = OutcomeRecord(
            query=query or "",
            prediction=prediction,
            result=result or "",
            success=success,
        )
        with self._lock:
            self._records.append(rec)
            self._prediction_total += 1
            self._recent_success.append(success)
            if _prediction_matched(query, prediction):
                self._prediction_hits += 1
            pa = self._prediction_hits / max(1, self._prediction_total)
            if not success and prediction and pa >= self._learn_confidence_threshold:
                self._apply_negative_feedback(prediction)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "v7_feedback outcome success=%s pred=%s",
                success,
                (prediction or "")[:40],
            )

    def record_prefetch_event(self, hit: bool) -> None:
        with self._lock:
            self._prefetch_total += 1
            if hit:
                self._prefetch_hits += 1

    def record_prefetch_scheduled(self, n: int) -> None:
        with self._lock:
            self._prefetch_scheduled += max(0, int(n))

    def record_user_override(self, reason: str = "") -> None:
        with self._lock:
            self._override_count += 1
        logger.info("v7_feedback_metrics user_override reason=%s", reason[:80])

    def record_graph_hit(self) -> None:
        with self._lock:
            self._graph_hits += 1
        self._log_graph_ratio()

    def record_rag_fallback(self) -> None:
        with self._lock:
            self._rag_fallbacks += 1
        self._log_graph_ratio()

    def record_graph_miss(self) -> None:
        with self._lock:
            self._graph_misses += 1

    def _log_graph_ratio(self) -> None:
        with self._lock:
            g, r = self._graph_hits, self._rag_fallbacks
            tot = max(1, g + r)
            ratio = g / tot
        logger.info(
            "v7_graph_vs_rag_ratio graph_hits=%d rag_fallbacks=%d ratio=%.3f",
            g, r, ratio,
        )

    def _apply_negative_feedback(self, prediction: str) -> None:
        low = (prediction or "").lower()
        for token in _tokenize_keywords(low):
            cur = self._priority_weights.get(token, 1.0)
            self._priority_weights[token] = max(0.3, cur - self._learn_rate)

    def boost_keyword(self, keyword: str, delta: float = 0.05) -> None:
        k = (keyword or "").strip().lower()[:64]
        if not k:
            return
        with self._lock:
            cur = self._priority_weights.get(k, 1.0)
            self._priority_weights[k] = min(2.0, cur + delta)

    def reorder_predictions(self, candidates: list[str]) -> list[str]:
        """Apply soft weights; higher score first."""

        def score(c: str) -> float:
            s = 1.0
            cl = c.lower()
            with self._lock:
                weights = dict(self._priority_weights)
            for k, w in weights.items():
                if k and k in cl:
                    s *= w
            return s

        return sorted(candidates, key=score, reverse=True)

    def prediction_trend(self) -> str:
        """improving | degrading | flat — deterministic from rolling windows."""
        fc = self._feedback_cfg()
        w50 = int(fc.get("rolling_window_50", self._win_50))
        w100 = int(fc.get("rolling_window_100", self._win_100))
        flat_eps = float(fc.get("trend_flat_epsilon", 0.04))
        with self._lock:
            seq = list(self._recent_success)
        if len(seq) < max(10, w50 // 2):
            return "flat"
        first = seq[-w100:-w50] if len(seq) >= w100 else seq[:-w50]
        second = seq[-w50:]
        if not first or not second:
            return "flat"
        m1 = sum(1 for x in first if x) / max(1, len(first))
        m2 = sum(1 for x in second if x) / max(1, len(second))
        if m2 - m1 > flat_eps:
            return "improving"
        if m1 - m2 > flat_eps:
            return "degrading"
        return "flat"

    def compute_accuracy_metrics(self) -> dict[str, Any]:
        with self._lock:
            ph = self._prediction_hits
            pt = max(1, self._prediction_total)
            fh = self._prefetch_hits
            ft = max(1, self._prefetch_total)
            gh = self._graph_hits
            rf = self._rag_fallbacks
            tot_gr = max(1, gh + rf)
            oc = self._override_count
            ps = max(0, self._prefetch_scheduled)
            gm = self._graph_misses
        prefetch_miss_rate = 1.0 - (fh / ft)
        prefetch_waste_rate = (
            1.0 - (fh / max(1, ps)) if ps > 0 else prefetch_miss_rate
        )
        graph_miss_rate = gm / max(1, gh + gm)
        return {
            "prediction_accuracy": ph / pt,
            "prefetch_hit_rate": fh / ft,
            "prefetch_miss_rate": prefetch_miss_rate,
            "prefetch_waste_rate": prefetch_waste_rate,
            "prefetch_scheduled": ps,
            "graph_vs_rag_ratio": gh / tot_gr,
            "graph_hits": gh,
            "rag_fallbacks": rf,
            "graph_miss_rate": graph_miss_rate,
            "graph_misses": gm,
            "user_overrides": oc,
            "prediction_trend": self.prediction_trend(),
        }

    def get_health_status(
        self,
        system_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Deterministic labels from config thresholds only."""
        hc = self._health_cfg()
        m = self.compute_accuracy_metrics()
        cpu = float((system_state or {}).get("cpu_percent") or 0)
        ram = float((system_state or {}).get("memory_percent") or (system_state or {}).get("ram_percent") or 0)

        pa = float(m.get("prediction_accuracy", 0.5))
        good_a = float(hc.get("prediction_good_above", 0.55))
        poor_a = float(hc.get("prediction_poor_below", 0.38))
        unstable_lo = float(hc.get("prediction_unstable_low", 0.4))
        unstable_hi = float(hc.get("prediction_unstable_high", 0.52))

        if pa >= good_a:
            pq = "good"
        elif pa <= poor_a:
            pq = "poor"
        elif unstable_lo <= pa <= unstable_hi:
            pq = "unstable"
        else:
            pq = "unstable"

        pr = float(m.get("prefetch_hit_rate", 0.0))
        pf_good = float(hc.get("prefetch_good_above", 0.35))
        pf_poor = float(hc.get("prefetch_poor_below", 0.15))
        if pr >= pf_good:
            pfe = "good"
        elif pr <= pf_poor:
            pfe = "poor"
        else:
            pfe = "unstable"

        gvr = float(m.get("graph_vs_rag_ratio", 0.5))
        mr_good = float(hc.get("memory_relevance_good_above", 0.35))
        mr_poor = float(hc.get("memory_relevance_poor_below", 0.15))
        if gvr >= mr_good:
            memrel = "good"
        elif gvr <= mr_poor:
            memrel = "poor"
        else:
            memrel = "unstable"

        cpu_low = float(hc.get("system_load_cpu_low_below", 55))
        cpu_high = float(hc.get("system_load_cpu_high_above", 88))
        ram_high = float(hc.get("system_load_ram_high_above", 90))
        if cpu >= cpu_high or ram >= ram_high:
            sload = "high"
        elif cpu <= cpu_low and ram <= ram_high:
            sload = "low"
        else:
            sload = "moderate"

        return {
            "prediction_quality": pq,
            "prefetch_efficiency": pfe,
            "memory_relevance": memrel,
            "system_load": sload,
            "raw": {
                "prediction_accuracy": pa,
                "prefetch_hit_rate": pr,
                "graph_vs_rag_ratio": gvr,
                "cpu_percent": cpu,
                "memory_percent": ram,
            },
        }


def _prediction_matched(query: str, prediction: str | None) -> bool:
    if not prediction:
        return False
    a, b = _norm_key(query), _norm_key(prediction)
    if not a or not b:
        return False
    fc = (query or "").strip()
    if len(fc) < 4:
        return False
    if a == b:
        return True
    overlap = a in b or b in a or a[:48] == b[:48]
    if not overlap:
        return False
    return True


def _tokenize_keywords(text: str) -> list[str]:
    out: list[str] = []
    for w in text.replace("/", " ").split():
        w = w.strip(".,?!:;\"'")[:48]
        if len(w) >= 4:
            out.append(w)
    return out[:8]
