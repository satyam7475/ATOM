"""
Late-RAG preemption scoring: only restart generation when improvement outweighs cost.

SecurityPolicy unchanged; same session/trace as existing preempt path.
"""

from __future__ import annotations


from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger("atom.cognition.preemption")

_lock = threading.Lock()
_last_score: float | None = None
_last_threshold: float | None = None


def get_last_preemption_score() -> dict[str, float | None]:
    with _lock:
        return {"score": _last_score, "threshold": _last_threshold}


def compute_preemption_improvement_score(
    late_result: Any,
    *,
    baseline_confidence: float = 0.0,
    config: dict[str, Any] | None = None,
) -> float:
    """Return score = relevance_gain + context_gain - restart_cost.

    Higher is better; compare to threshold from config.
    """
    cfg = (config or {}).get("v7_intelligence") or {}
    pc = cfg.get("preemption") or {}
    restart_cost = float(pc.get("restart_cost", 0.45))
    rel_scale = float(pc.get("relevance_scale", 1.0))
    ctx_scale = float(pc.get("context_scale", 0.25))

    conf = float(getattr(late_result, "confidence", 0.0) or 0.0)
    relevance_gain = max(0.0, conf - max(0.0, baseline_confidence)) * rel_scale

    chunks = getattr(late_result, "chunks", None) or []
    n = len(chunks)
    total_chars = sum(len(str(c)) for c in chunks[:8])
    context_gain = min(0.85, ctx_scale * (n * 0.12 + min(1.0, total_chars / 4000.0)))

    score = relevance_gain + context_gain - restart_cost
    cfg = (config or {}).get("v7_intelligence") or {}
    thr = float((cfg.get("preemption") or {}).get("min_improvement_score", 0.25))
    global _last_score, _last_threshold
    with _lock:
        _last_score = score
        _last_threshold = thr
    try:
        logger.info(
            "v7_preemption_score conf=%.3f relevance_gain=%.3f context_gain=%.3f "
            "restart_cost=%.3f score=%.3f",
            conf, relevance_gain, context_gain, restart_cost, score,
        )
    except Exception:
        pass
    return score


def should_preempt_for_late_rag(
    late_result: Any,
    *,
    baseline_confidence: float = 0.0,
    config: dict[str, Any] | None = None,
) -> bool:
    cfg = (config or {}).get("v7_intelligence") or {}
    thr = float((cfg.get("preemption") or {}).get("min_improvement_score", 0.25))
    score = compute_preemption_improvement_score(
        late_result,
        baseline_confidence=baseline_confidence,
        config=config,
    )
    return score > thr
