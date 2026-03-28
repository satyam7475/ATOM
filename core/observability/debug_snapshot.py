"""Unified V7 debug snapshot with short TTL cache (shallow reads only)."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger("atom.observability.snapshot")

_cache_lock = threading.Lock()
_cached_snapshot: dict[str, Any] | None = None
_cache_ts: float = 0.0


def get_debug_snapshot(
    config: dict[str, Any],
    *,
    runtime_mode: str,
    mode_info: dict[str, Any],
    system_state: dict[str, Any],
    feedback_metrics: dict[str, Any],
    last_retrieval_source: str,
    timeline_event_count: int,
    timeline_recent_preview: list[dict[str, Any]],
    active_project: str | None,
    preemption: dict[str, Any],
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Assemble snapshot; uses TTL cache to limit heavy aggregation."""
    v7 = config.get("v7_intelligence") or {}
    obs = v7.get("observability") or {}
    ttl = float(obs.get("debug_snapshot_cache_ttl_s", 3.0))

    now = time.time()
    with _cache_lock:
        global _cached_snapshot, _cache_ts
        if (
            not force_refresh
            and _cached_snapshot is not None
            and (now - _cache_ts) < ttl
        ):
            return dict(_cached_snapshot)

    snap = {
        "ts": now,
        "runtime_mode": runtime_mode,
        "mode_info": dict(mode_info or {}),
        "system_state": {
            "cpu_percent": system_state.get("cpu_percent"),
            "memory_percent": system_state.get("memory_percent") or system_state.get("ram_percent"),
        },
        "prediction_accuracy": feedback_metrics.get("prediction_accuracy"),
        "prefetch_hit_rate": feedback_metrics.get("prefetch_hit_rate"),
        "prefetch_miss_rate": feedback_metrics.get("prefetch_miss_rate"),
        "graph_vs_rag_ratio": feedback_metrics.get("graph_vs_rag_ratio"),
        "last_retrieval_source": last_retrieval_source,
        "preemption": dict(preemption),
        "timeline_event_count": timeline_event_count,
        "timeline_recent": timeline_recent_preview,
        "active_project": active_project,
    }

    with _cache_lock:
        _cached_snapshot = dict(snap)
        _cache_ts = now

    return snap


def log_v7_debug_snapshot(payload: dict[str, Any]) -> None:
    """Structured periodic log line."""
    try:
        logger.info("v7_debug_snapshot %s", payload)
    except Exception:
        pass
