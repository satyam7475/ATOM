"""Dynamic RAG first-token budget from GPU load and query complexity."""

from __future__ import annotations

from core.rag.query_classifier import QueryComplexity


def compute_adaptive_rag_budget_ms(
    base_ms: float,
    complexity: QueryComplexity,
    *,
    gpu_util_pct: float = 0.0,
    vram_pressure: float = 0.0,
    prefetch_hit: bool = False,
    cfg: dict | None = None,
) -> float:
    """Increase budget when GPU is idle; shrink under load. SIMPLE → minimal wait."""
    c = cfg or {}
    lo = float(c.get("budget_min_ms", 40))
    hi = float(c.get("budget_max_ms", 280))

    if complexity == QueryComplexity.SIMPLE:
        return lo

    ms = base_ms
    if prefetch_hit:
        ms = min(hi, ms * 0.45)

    # High GPU utilization → shorter budget (start LLM sooner)
    if gpu_util_pct >= 85:
        ms *= 0.55
    elif gpu_util_pct >= 65:
        ms *= 0.75
    elif gpu_util_pct <= 25:
        ms *= 1.25

    if vram_pressure >= 0.9:
        ms *= 0.6
    elif vram_pressure <= 0.35:
        ms *= 1.1

    return max(lo, min(hi, ms))
