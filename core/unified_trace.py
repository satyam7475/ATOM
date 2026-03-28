"""ATOM V7 — Unified trace payload for observability (latency, GPU, decision path)."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UnifiedTrace:
    trace_id: str
    started_monotonic: float = field(default_factory=time.monotonic)
    latency_ms: dict[str, float] = field(default_factory=dict)
    gpu_usage_pct: float | None = None
    vram_mb: float | None = None
    cpu_usage_pct: float | None = None
    decision_path: list[str] = field(default_factory=list)
    error: str | None = None

    def span(self, name: str, start: float) -> None:
        self.latency_ms[name] = (time.perf_counter() - start) * 1000
        self.decision_path.append(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "latency_ms": dict(self.latency_ms),
            "gpu_usage_pct": self.gpu_usage_pct,
            "vram_mb": self.vram_mb,
            "cpu_usage_pct": self.cpu_usage_pct,
            "decision_path": list(self.decision_path),
            "error": self.error,
        }


def new_trace(trace_id: str | None = None) -> UnifiedTrace:
    return UnifiedTrace(trace_id=trace_id or str(uuid.uuid4()))
