"""
Telemetry with optional batched metric recording (V6) to reduce hot-path overhead.

V7: optional unified trace snapshots (trace_id, per-stage latency, GPU/VRAM).
"""
import statistics
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = __import__("logging").getLogger("atom.core.telemetry")


class TelemetryEngine:
    def __init__(self, batch_interval_s: float = 2.0, enable_batch: bool = True):
        self.metrics: Dict[str, List[Tuple[float, float]]] = {}
        self.lock = threading.Lock()
        self.snapshot: Dict[str, Any] = {}
        self.unified_traces: List[Dict[str, Any]] = []
        self._max_traces = 200
        self._batch_interval_s = batch_interval_s
        self._enable_batch = enable_batch
        self._pending: Dict[str, List[float]] = {}
        self._last_flush = time.time()

    def set_snapshot(self, data: Dict[str, Any]) -> None:
        with self.lock:
            self.snapshot = dict(data)

    def record_metric(self, name: str, value: float) -> None:
        if self._enable_batch:
            with self.lock:
                self._pending.setdefault(name, []).append(value)
                now = time.time()
                if now - self._last_flush >= self._batch_interval_s:
                    self._flush_locked(now)
            return
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append((time.time(), value))

    def flush(self) -> None:
        with self.lock:
            self._flush_locked(time.time())

    def _flush_locked(self, now: float) -> None:
        for name, vals in self._pending.items():
            if not vals:
                continue
            agg = sum(vals) / len(vals)
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append((now, agg))
        self._pending.clear()
        self._last_flush = now

    def get_metrics(self) -> Dict[str, Any]:
        with self.lock:
            return {k: list(v) for k, v in self.metrics.items()}

    def record_unified_trace(self, trace_dict: Dict[str, Any]) -> None:
        with self.lock:
            self.unified_traces.append(dict(trace_dict))
            if len(self.unified_traces) > self._max_traces:
                self.unified_traces = self.unified_traces[-self._max_traces :]

    def reset(self) -> None:
        with self.lock:
            self.metrics.clear()
            self.snapshot.clear()
            self.unified_traces.clear()
            self._pending.clear()

    def summary(self) -> Dict[str, Any]:
        with self.lock:
            self._flush_locked(time.time())
            snap = dict(self.snapshot)
            traces = list(self.unified_traces[-50:])
            out: Dict[str, Any] = {
                "snapshot": snap,
                "aggregates": {},
                "unified_traces": traces,
            }
            for name, series in self.metrics.items():
                vals = [v for _, v in series[-500:]]
                if not vals:
                    continue
                out["aggregates"][name] = {
                    "count": len(vals),
                    "mean": statistics.fmean(vals),
                    "last": vals[-1],
                }
                if name == "execution_time_ms" and len(vals) >= 2:
                    out["aggregates"][name]["stdev"] = statistics.pstdev(vals)
            return out
