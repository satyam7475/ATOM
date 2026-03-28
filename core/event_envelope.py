"""
ATOM V7 — Intent-aware event envelope for scheduling hints.

Handlers may read optional keys ``v7_priority``, ``v7_gpu_cost``, ``v7_latency_budget_ms``
without breaking older subscribers (extra kwargs ignored).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EventEnvelope:
    """Metadata for smart routing (GPUScheduler / future PriorityEventBus)."""

    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    gpu_cost: float = 0.0
    latency_budget_ms: float = 1500.0
    trace_id: str | None = None

    def to_emit_kwargs(self) -> dict[str, Any]:
        out = dict(self.payload)
        out["v7_priority"] = self.priority
        out["v7_gpu_cost"] = self.gpu_cost
        out["v7_latency_budget_ms"] = self.latency_budget_ms
        if self.trace_id:
            out["trace_id"] = self.trace_id
        return out


def emit_envelope(bus: Any, envelope: EventEnvelope) -> None:
    """Emit on AsyncEventBus / ZmqEventBus with V7 scheduling hints."""
    fn = getattr(bus, "emit_fast", None) or getattr(bus, "emit", None)
    if fn is None:
        return
    kwargs = envelope.to_emit_kwargs()
    fn(envelope.event_type, **kwargs)


__all__ = ["EventEnvelope", "emit_envelope"]
