"""V7 observability + Sprint N4 metrics broker.

Adds a Prometheus-style ``MetricsBroker`` that any subsystem can plug
into without taking on a new client lib dependency.
"""

from core.observability.debug_snapshot import get_debug_snapshot, log_v7_debug_snapshot
from core.observability.metrics_broker import MetricsBroker, ProviderSnapshot
from core.observability.per_module_latency import (
    ModuleMetrics,
    ObservabilityLatencyBoard,
    get_latency_board,
)
from core.observability.warnings import collect_v7_warnings

__all__ = [
    "ModuleMetrics",
    "ObservabilityLatencyBoard",
    "get_latency_board",
    "collect_v7_warnings",
    "get_debug_snapshot",
    "log_v7_debug_snapshot",
    "MetricsBroker",
    "ProviderSnapshot",
]
