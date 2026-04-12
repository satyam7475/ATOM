"""Per-module latency and error rollups (evolution plan observability §7.2).

Complements ``MetricsCollector`` (counters / global pipeline) with named-module
avg + p95 latencies for dashboards and future ``/v7/health`` enrichment.

Stdlib only; thread-safe for single asyncio thread + worker callbacks that
serialize through the event loop (typical ATOM pattern). For multi-thread
recorders, wrap calls in a small lock at the call site.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Optional

_board_lock = Lock()
_board_instance: Optional["ObservabilityLatencyBoard"] = None


@dataclass
class ModuleMetrics:
    name: str
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    call_count: int = 0
    error_count: int = 0
    last_call_time: float = 0.0
    _latencies: deque = field(default_factory=lambda: deque(maxlen=100))

    def record(self, latency_ms: float, error: bool = False) -> None:
        self._latencies.append(float(latency_ms))
        self.call_count += 1
        self.last_call_time = time.time()
        if error:
            self.error_count += 1

        latencies = sorted(self._latencies)
        n = len(latencies)
        if n == 0:
            self.avg_latency_ms = 0.0
            self.p95_latency_ms = 0.0
            return
        self.avg_latency_ms = sum(latencies) / n
        idx = min(n - 1, int(n * 0.95))
        self.p95_latency_ms = latencies[idx]


class ObservabilityLatencyBoard:
    """Collects rolling per-module latency samples and a short event tail."""

    TRACKED_MODULES = (
        "stt",
        "intent_engine",
        "router",
        "llm_small",
        "llm_large",
        "rag",
        "memory",
        "tts",
        "tool_executor",
        "state_manager",
    )

    def __init__(self, state_snapshot: Optional[Callable[[], dict[str, Any]]] = None) -> None:
        self._state_snapshot = state_snapshot
        self.modules: dict[str, ModuleMetrics] = {
            name: ModuleMetrics(name=name) for name in self.TRACKED_MODULES
        }
        self._event_log: deque = deque(maxlen=500)

    def record_module_call(self, module: str, latency_ms: float, error: bool = False) -> None:
        m = self.modules.get(module)
        if m is not None:
            m.record(latency_ms, error=error)

    def log_event(self, event_type: str, details: str) -> None:
        self._event_log.append(
            {"time": time.time(), "type": event_type, "details": details[:500]},
        )

    def get_dashboard_data(self) -> dict[str, Any]:
        system_state: dict[str, Any] = {}
        if self._state_snapshot is not None:
            try:
                system_state = dict(self._state_snapshot())
            except Exception:
                system_state = {"error": "state_snapshot_failed"}

        modules_out = {
            name: {
                "avg_latency_ms": round(m.avg_latency_ms, 3),
                "p95_latency_ms": round(m.p95_latency_ms, 3),
                "calls": m.call_count,
                "errors": m.error_count,
                "last_call_time": m.last_call_time,
            }
            for name, m in self.modules.items()
        }

        return {
            "system_state": system_state,
            "modules": modules_out,
            "recent_events": list(self._event_log)[-20:],
            "health": self._compute_health(),
        }

    def _compute_health(self) -> str:
        total_errors = sum(m.error_count for m in self.modules.values())
        total_calls = sum(m.call_count for m in self.modules.values())
        if total_calls == 0:
            return "idle"
        error_rate = total_errors / total_calls
        if error_rate > 0.1:
            return "degraded"
        if error_rate > 0.01:
            return "warning"
        return "healthy"


def get_latency_board() -> ObservabilityLatencyBoard:
    """Process-wide latency board (safe to call from bus handlers / router)."""
    global _board_instance
    with _board_lock:
        if _board_instance is None:
            _board_instance = ObservabilityLatencyBoard()
        return _board_instance


def reset_latency_board_for_tests() -> None:
    """Test helper: clear singleton."""
    global _board_instance
    with _board_lock:
        _board_instance = None


__all__ = [
    "ModuleMetrics",
    "ObservabilityLatencyBoard",
    "get_latency_board",
    "reset_latency_board_for_tests",
]
