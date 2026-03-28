"""
ATOM -- Lightweight structured metrics collector.

Tracks counters, latencies, and uptime for production observability.
Periodic health log emits a one-line JSON summary every 60 seconds
to logs/atom_metrics.log for corporate monitoring integration.

Zero external dependencies -- stdlib only.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock

logger = logging.getLogger("atom.metrics")

HEALTH_INTERVAL_S = 60
MAX_LATENCY_SAMPLES = 100


class MetricsCollector:
    """Thread-safe metrics singleton for ATOM runtime observability."""

    __slots__ = (
        "_lock", "_start_time",
        "resume_listening_events", "cache_hits", "cache_misses",
        "llm_calls", "llm_errors", "stt_sessions",
        "errors_total", "queries_total",
        "llm_queue_coalesced", "llm_preempted", "watchdog_recoveries",
        "scheduler_jobs_submitted",
        "_llm_latencies", "_stt_confidences", "_pipeline_latencies",
        "_gauges",
    )

    def __init__(self) -> None:
        self._lock = Lock()
        self._start_time = time.monotonic()

        self.resume_listening_events: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.llm_calls: int = 0
        self.llm_errors: int = 0
        self.stt_sessions: int = 0
        self.errors_total: int = 0
        self.queries_total: int = 0
        self.llm_queue_coalesced: int = 0
        self.llm_preempted: int = 0
        self.watchdog_recoveries: int = 0
        self.scheduler_jobs_submitted: int = 0

        self._llm_latencies: deque[float] = deque(maxlen=MAX_LATENCY_SAMPLES)
        self._stt_confidences: deque[float] = deque(maxlen=MAX_LATENCY_SAMPLES)
        self._pipeline_latencies: dict[str, deque[float]] = {}
        self._gauges: dict[str, float] = {
            "scheduler_queue_depth": 0,
            "vram_used_mb": 0.0,
            "gpu_util_pct": 0.0,
            "gpu_sched_queue_depth": 0.0,
        }

    def set_gauge(self, name: str, value: int | float) -> None:
        """Set a gauge (e.g. scheduler queue depth, VRAM MB). Thread-safe."""
        with self._lock:
            v = float(value)
            if name == "scheduler_queue_depth" or name.endswith("_depth"):
                self._gauges[name] = max(0.0, v)
            else:
                self._gauges[name] = max(0.0, v)

    def inc(self, counter: str, amount: int = 1) -> None:
        """Increment a named counter. Thread-safe."""
        with self._lock:
            current = getattr(self, counter, None)
            if current is not None and isinstance(current, int):
                setattr(self, counter, current + amount)

    def record_latency(self, name: str, ms: float) -> None:
        """Record a latency sample (milliseconds)."""
        with self._lock:
            if name == "llm":
                self._llm_latencies.append(ms)
            elif name == "stt_confidence":
                self._stt_confidences.append(ms)
            elif name.startswith("pipeline_"):
                self._pipeline_latencies.setdefault(name, deque(maxlen=MAX_LATENCY_SAMPLES)).append(ms)
            elif name == "perceived":
                self._pipeline_latencies.setdefault("perceived", deque(maxlen=MAX_LATENCY_SAMPLES)).append(ms)
            elif name == "ttfa":
                self._pipeline_latencies.setdefault("ttfa", deque(maxlen=MAX_LATENCY_SAMPLES)).append(ms)
            else:
                self._pipeline_latencies.setdefault(
                    name, deque(maxlen=MAX_LATENCY_SAMPLES)
                ).append(ms)

    def snapshot(self) -> dict:
        """Return a point-in-time copy of all metrics."""
        with self._lock:
            uptime_s = time.monotonic() - self._start_time
            llm_lats = list(self._llm_latencies)
            stt_confs = list(self._stt_confidences)
            sched_depth = int(self._gauges.get("scheduler_queue_depth", 0))
            vram_mb = float(self._gauges.get("vram_used_mb", 0.0))
            gpu_u = float(self._gauges.get("gpu_util_pct", 0.0))
            gpu_q = float(self._gauges.get("gpu_sched_queue_depth", 0.0))
            pipeline_snap = {
                k: list(v) for k, v in self._pipeline_latencies.items()
            }

        avg_llm = sum(llm_lats) / len(llm_lats) if llm_lats else 0.0
        p95_llm = sorted(llm_lats)[int(len(llm_lats) * 0.95)] if len(llm_lats) >= 2 else avg_llm
        avg_conf = sum(stt_confs) / len(stt_confs) if stt_confs else 0.0

        cache_total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / cache_total * 100) if cache_total > 0 else 0.0

        result = {
            "uptime_s": round(uptime_s, 1),
            "resume_listening_events": self.resume_listening_events,
            "queries_total": self.queries_total,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate_pct": round(hit_rate, 1),
            "llm_calls": self.llm_calls,
            "llm_errors": self.llm_errors,
            "llm_avg_ms": round(avg_llm, 1),
            "llm_p95_ms": round(p95_llm, 1),
            "stt_sessions": self.stt_sessions,
            "stt_avg_confidence": round(avg_conf, 2),
            "errors_total": self.errors_total,
            "llm_queue_coalesced": self.llm_queue_coalesced,
            "llm_preempted": self.llm_preempted,
            "watchdog_recoveries": self.watchdog_recoveries,
            "scheduler_jobs_submitted": self.scheduler_jobs_submitted,
            "scheduler_queue_depth": sched_depth,
            "vram_used_mb": round(vram_mb, 1),
            "gpu_util_pct": round(gpu_u, 1),
            "gpu_sched_queue_depth": int(gpu_q),
        }

        for name, samples in pipeline_snap.items():
            if samples:
                result[f"{name}_avg_ms"] = round(
                    sum(samples) / len(samples), 1)

        return result


_metrics_logger: logging.Logger | None = None


def _get_metrics_logger() -> logging.Logger:
    """Lazy-init a dedicated logger for metrics JSON lines."""
    global _metrics_logger
    if _metrics_logger is None:
        _metrics_logger = logging.getLogger("atom.metrics.health")
        _metrics_logger.setLevel(logging.INFO)
        _metrics_logger.propagate = False
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        handler = RotatingFileHandler(
            log_dir / "atom_metrics.log",
            maxBytes=500_000,
            backupCount=1,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"))
        _metrics_logger.addHandler(handler)
    return _metrics_logger


def log_health(collector: MetricsCollector) -> None:
    """Write a single-line JSON health snapshot to the metrics log."""
    snap = collector.snapshot()
    _get_metrics_logger().info(json.dumps(snap, separators=(",", ":")))


_metrics_instance: MetricsCollector | None = None
_metrics_singleton_lock = Lock()


def get_metrics() -> MetricsCollector:
    """Process-wide metrics singleton (profiler, workers, orchestrator)."""
    global _metrics_instance
    with _metrics_singleton_lock:
        if _metrics_instance is None:
            _metrics_instance = MetricsCollector()
        return _metrics_instance
