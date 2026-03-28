"""V7 health warnings from config thresholds (deterministic)."""

from __future__ import annotations

from typing import Any


def collect_v7_warnings(
    config: dict[str, Any],
    *,
    feedback_metrics: dict[str, Any],
    health_status: dict[str, Any],
) -> list[dict[str, str]]:
    """Return list of {code, message} for degrading subsystems."""
    v7 = config.get("v7_intelligence") or {}
    wc = v7.get("warnings") or {}
    out: list[dict[str, str]] = []

    trend = str(feedback_metrics.get("prediction_trend") or "flat")
    if trend == "degrading" and bool(wc.get("warn_on_degrading_prediction", True)):
        out.append({
            "code": "prediction_degrading",
            "message": "Prediction trend is degrading over rolling windows.",
        })

    gmr = float(feedback_metrics.get("graph_miss_rate") or 0.0)
    gmt = float(wc.get("graph_miss_rate_above", 0.35))
    if gmr >= gmt:
        out.append({
            "code": "graph_miss_high",
            "message": f"Graph miss rate {gmr:.2f} exceeds threshold {gmt:.2f}.",
        })

    pwr = float(feedback_metrics.get("prefetch_waste_rate") or feedback_metrics.get("prefetch_miss_rate") or 0.0)
    pwt = float(wc.get("prefetch_waste_above", 0.72))
    if pwr >= pwt:
        out.append({
            "code": "prefetch_waste_high",
            "message": f"Prefetch waste rate {pwr:.2f} exceeds threshold {pwt:.2f}.",
        })

    pq = str(health_status.get("prediction_quality") or "")
    if pq == "poor":
        out.append({
            "code": "prediction_quality_poor",
            "message": "Prediction quality classified as poor.",
        })

    return out
