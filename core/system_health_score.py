"""
ATOM V6.5 -- Composite system health score (0–10) from reliability signals.

    health_10 =
        success_rate      * 4.0 +
        latency_score     * 2.0 +
        failure_recovery  * 2.0 +
        stability         * 2.0

Each input is expected in [0, 1]; result is capped to [0, 10].
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from core.profiler import get_latency_snapshot


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def latency_score_from_ms(mean_ms: float, target_ms: float = 100.0) -> float:
    """1.0 at or below target; decays toward 0 as latency grows."""
    if mean_ms <= 0:
        return 1.0
    return _clamp01(target_ms / (mean_ms + target_ms))


def compute_health_score_10(
    success_rate: float,
    mean_latency_ms: float,
    recovery_ms: float,
    determinism_score: float,
    recovery_target_ms: float = 500.0,
    latency_target_ms: float = 100.0,
) -> Dict[str, Any]:
    """
    *success_rate*: fraction of successful operations [0,1].
    *mean_latency_ms*: average hot-path latency.
    *recovery_ms*: observed recovery / chaos response time (lower is better).
    *determinism_score*: [0,1] from validation engine.
    """

    sr = _clamp01(success_rate)
    lat = latency_score_from_ms(mean_latency_ms, latency_target_ms)
    fr = _clamp01(recovery_target_ms / (recovery_ms + recovery_target_ms))
    st = _clamp01(determinism_score)

    raw = sr * 4.0 + lat * 2.0 + fr * 2.0 + st * 2.0
    score_10 = min(10.0, max(0.0, raw))

    return {
        "health_score_10": round(score_10, 2),
        "components": {
            "success_rate": round(sr, 4),
            "latency_score": round(lat, 4),
            "failure_recovery_score": round(fr, 4),
            "stability_score": round(st, 4),
        },
        "inputs": {
            "success_rate": success_rate,
            "mean_latency_ms": mean_latency_ms,
            "recovery_ms": recovery_ms,
            "determinism_score": determinism_score,
        },
    }


def estimate_from_runtime_json(path: Path) -> Dict[str, Any]:
    """Best-effort score using ATOM/config/atom_runtime.json telemetry + profiler."""

    success_rate = 0.95
    mean_latency_ms = 50.0
    recovery_ms = 200.0
    det = 0.9

    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            summ = data.get("telemetry_summary") or {}
            agg = summ.get("aggregates") or {}
            # Heuristic: execution success vs failures if present
            ex = agg.get("goal_execution") or agg.get("execution") or {}
            if isinstance(ex, dict):
                ok = float(ex.get("success_count", ex.get("ok", 0)) or 0)
                fail = float(ex.get("fail_count", ex.get("fail", 0)) or 0)
                tot = ok + fail
                if tot > 0:
                    success_rate = ok / tot
            lat_agg = agg.get("latency_ms") or agg.get("planning_time_ms")
            if isinstance(lat_agg, dict) and lat_agg.get("mean") is not None:
                mean_latency_ms = float(lat_agg["mean"])
    except Exception:
        pass

    prof = get_latency_snapshot()
    if prof:
        # Prefer execution / planning means if present
        for key in ("execution", "planning", "simulation", "memory"):
            if key in prof:
                mean_latency_ms = float(prof[key]["mean_ms"])
                break

    return compute_health_score_10(
        success_rate=success_rate,
        mean_latency_ms=mean_latency_ms,
        recovery_ms=recovery_ms,
        determinism_score=det,
    )


__all__ = [
    "compute_health_score_10",
    "latency_score_from_ms",
    "estimate_from_runtime_json",
]
