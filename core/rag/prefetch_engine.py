"""Background RAG prefetch driven by cognitive loop + query prediction."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.rag.rag_engine import RagEngine

logger = logging.getLogger("atom.rag.prefetch")


def merge_prefetch_candidates(
    *groups: list[str],
    max_candidates: int | None = None,
) -> list[str]:
    """Merge and dedupe prefetch query lists (order preserved by group)."""
    cap = max(1, int(max_candidates or 18))
    seen: set[str] = set()
    out: list[str] = []
    for group in groups:
        for s in group or []:
            t = (s or "").strip()[:400]
            if not t or t in seen:
                continue
            seen.add(t)
            out.append(t)
            if len(out) >= cap:
                return out
    return out


def predict_followup_queries(last_query: str, history: list[tuple[str, str]] | None) -> list[str]:
    """Heuristic next-query candidates for prefetch (no LLM call)."""
    out: list[str] = []
    q = (last_query or "").strip()
    if len(q) < 3:
        return out

    low = q.lower()
    out.append(q)
    out.append(f"{q} details")
    out.append(f"summary of {q[:80]}")

    if history:
        prev_user = history[-1][0] if history else ""
        if prev_user and prev_user != q:
            out.append(f"{prev_user} then {q[:60]}")

    # Topic continuations
    if any(w in low for w in ("how", "what", "why", "explain")):
        m = re.search(r"(how|what|why)\s+(\w+)", low)
        if m:
            out.append(f"more about {m.group(2)}")

    seen: set[str] = set()
    uniq: list[str] = []
    for s in out:
        s = s.strip()[:300]
        if s and s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq[:6]


class RagPrefetchEngine:
    """Schedule background retrieve() for predicted queries (soft-scaling)."""

    def __init__(self, engine: "RagEngine | None", config: dict | None = None) -> None:
        self._engine = engine
        self._config = config or {}
        self._enabled = bool((self._config.get("rag") or {}).get("prefetch_enabled", True))

    def set_engine(self, engine: "RagEngine | None") -> None:
        self._engine = engine

    def _prefetch_cfg(self) -> dict[str, Any]:
        return (self._config.get("v7_intelligence") or {}).get("prefetch") or {}

    async def prefetch_queries(self, queries: list[str]) -> None:
        if not self._enabled or self._engine is None or not queries:
            return

        async def _one(q: str) -> None:
            try:
                await self._engine.retrieve(q)
                logger.debug("v7_prefetch_done query_len=%d", len(q))
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("prefetch retrieve failed for %s", q[:40], exc_info=True)

        await asyncio.gather(*(_one(q) for q in queries), return_exceptions=True)

    def schedule_fire_and_forget(
        self,
        queries: list[str],
        *,
        gpu_util_pct: float = 0.0,
        prediction_accuracy: float | None = None,
    ) -> None:
        if not queries:
            return
        pf = self._prefetch_cfg()
        max_c = int(pf.get("max_prefetch_candidates", 12))
        hard_gpu = float(pf.get("hard_abort_gpu_above", 98.0))
        soft_gpu = float(pf.get("soft_scale_gpu_above", 85.0))
        soft_factor = float(pf.get("soft_scale_factor", 0.55))
        delay_s = float(pf.get("soft_delay_s", 0.25))
        min_pred = float(pf.get("min_prediction_confidence", 0.22))
        low_conf_extra = float(pf.get("low_conf_extra_delay_s", 0.15))

        if gpu_util_pct >= hard_gpu:
            logger.info("v7_prefetch_skipped reason=gpu_hard util=%.1f", gpu_util_pct)
            return

        capped = merge_prefetch_candidates(queries, max_candidates=max_c)
        if gpu_util_pct >= soft_gpu:
            max_c = max(1, int(max_c * soft_factor))
            capped = capped[:max_c]
            delay_s += float(pf.get("gpu_soft_extra_delay_s", 0.2))
            logger.info(
                "v7_prefetch_soft_scale reason=gpu util=%.1f candidates=%d",
                gpu_util_pct,
                len(capped),
            )

        if prediction_accuracy is not None and prediction_accuracy < min_pred:
            max_c = max(1, int(max_c * soft_factor))
            capped = capped[:max_c]
            delay_s += low_conf_extra
            logger.info(
                "v7_prefetch_soft_scale reason=low_pred acc=%.3f candidates=%d",
                prediction_accuracy,
                len(capped),
            )

        if not capped:
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        queries_final = capped

        async def _delayed() -> None:
            if delay_s > 0:
                await asyncio.sleep(delay_s)
            await self.prefetch_queries(queries_final)

        loop.create_task(_delayed(), name="rag_prefetch")
