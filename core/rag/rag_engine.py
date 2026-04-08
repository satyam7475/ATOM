"""
Low-latency RAG engine (Apple Silicon).

- Smart skip (query classifier)
- TTL caches (embed + retrieval)
- Hybrid vector + keyword re-rank + recency
- Optional Qdrant + Chroma VectorStore fallback
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from core.embedding_engine import get_embedding_engine
from core.rag.adaptive_budget import compute_adaptive_rag_budget_ms
from core.rag.context_builder import build_rag_enrichment_block
from core.rag.embedding_disk_cache import PersistentEmbeddingCache
from core.rag.graph_rag import graph_snippets_for_query
from core.rag.query_classifier import QueryComplexity, classify_query
from core.rag.rag_cache import RagCaches

logger = logging.getLogger("atom.rag.engine")


def _tokenize(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _keyword_boost(query: str, text: str) -> float:
    q = _tokenize(query)
    t = _tokenize(text)
    if not q or not t:
        return 0.0
    inter = len(q & t)
    return min(1.0, inter / max(4, len(q)))


def _recency_boost(metadata: dict[str, Any], now: float, max_age_s: float = 86400 * 14) -> float:
    ts = float(metadata.get("timestamp") or now)
    age = max(0.0, now - ts)
    return max(0.0, 1.0 - min(1.0, age / max_age_s)) * 0.15


@dataclass
class RagRetrieveResult:
    chunks: list[str]
    document_context: list[str]
    enrichment_block: str
    latency_ms: float
    cache_hit: bool
    skipped_embed: bool
    complexity: QueryComplexity
    confidence: float = 0.0
    graph_snippets: list[str] = field(default_factory=list)
    prefetch_hit: bool = False
    retrieval_source: str = "unknown"
    graph_project_hint: str | None = None


class RagEngine:
    """RAG retrieval with hybrid embedding and search."""

    def __init__(
        self,
        config: dict | None = None,
        vector_store: Any = None,
        coordinator: Any = None,
    ) -> None:
        self._config = config or {}
        self._rag_cfg = self._config.get("rag", {})
        self._enabled = bool(self._rag_cfg.get("enabled", True))
        self._collections = list(
            self._rag_cfg.get("collections", ["documents", "facts", "conversations"]),
        )
        self._top_k = int(self._rag_cfg.get("top_k", 6))
        self._vec_weight = float(self._rag_cfg.get("vector_weight", 0.65))
        self._kw_weight = float(self._rag_cfg.get("keyword_weight", 0.25))
        self._rec_weight = float(self._rag_cfg.get("recency_weight", 0.10))
        self._skip_embed_util = float(self._rag_cfg.get("skip_embed_gpu_util_above", 88))
        self._batch_min = int(self._rag_cfg.get("batch_embed_min", 2))
        self._fast_mode = bool(self._rag_cfg.get("fast_mode", False))
        self._disk_cache: PersistentEmbeddingCache | None = None
        if bool(self._rag_cfg.get("persistent_embed_cache", True)):
            try:
                self._disk_cache = PersistentEmbeddingCache(
                    path=self._rag_cfg.get("embed_cache_path", "data/rag_embedding_cache.sqlite"),
                )
            except Exception:
                self._disk_cache = None

        rc = self._rag_cfg.get("cache", {})
        self._caches = RagCaches(
            embed_ttl_s=float(rc.get("embed_ttl_s", 600)),
            retrieval_ttl_s=float(rc.get("retrieval_ttl_s", 120)),
            max_entries=int(rc.get("max_entries", 512)),
        )

        self._vector_store = vector_store
        self._coord = coordinator
        self._embed = get_embedding_engine(self._config)
        self._memory_graph: Any = None
        self._feedback: Any = None
        self._skip_graph_first_once: bool = False

        self._qdrant = None
        if (self._rag_cfg.get("backend") or "chroma").lower() == "qdrant":
            try:
                from core.rag.qdrant_backend import QdrantRagBackend
                self._qdrant = QdrantRagBackend(
                    persist_path=self._rag_cfg.get("qdrant_path", "data/qdrant_rag"),
                    collection=self._rag_cfg.get("qdrant_collection", "atom_rag"),
                )
                if not self._qdrant.enabled:
                    self._qdrant = None
            except Exception:
                self._qdrant = None

    def set_vector_store(self, store: Any) -> None:
        self._vector_store = store

    def set_coordinator(self, coord: Any) -> None:
        """Legacy shim — coordinator no longer used on Apple Silicon."""
        self._coord = None

    def set_memory_graph(self, mg: Any) -> None:
        self._memory_graph = mg

    def set_feedback_engine(self, fb: Any) -> None:
        """Optional FeedbackEngine for graph vs RAG ratio metrics."""
        self._feedback = fb

    @staticmethod
    def compute_budget_ms(
        config: dict | None,
        complexity: QueryComplexity,
        gpu_util_pct: float = 0.0,
        vram_pressure: float = 0.0,
        prefetch_hit: bool = False,
    ) -> float:
        base = float((config or {}).get("rag", {}).get("first_token_budget_ms", 120))
        return compute_adaptive_rag_budget_ms(
            base,
            complexity,
            gpu_util_pct=gpu_util_pct,
            vram_pressure=vram_pressure,
            prefetch_hit=prefetch_hit,
            cfg=(config or {}).get("rag", {}).get("adaptive", {}),
        )

    def _should_skip_embed_for_gpu(self) -> bool:
        if self._coord is None:
            return False
        gs = self._coord.refresh_gpu_state()
        return gs.gpu_util_pct >= self._skip_embed_util

    async def _embed_query(self, query: str) -> tuple[list[float], bool]:
        """Returns (embedding, skipped_due_to_busy policy)."""
        cached = self._caches.get_embedding(query)
        if cached is not None:
            return cached, False

        if self._disk_cache is not None:
            disk_vec = self._disk_cache.get(query)
            if disk_vec is not None:
                self._caches.set_embedding(query, disk_vec)
                return disk_vec, False

        if self._should_skip_embed_for_gpu():
            logger.debug("RAG: sync embed path (busy policy)")
            vec = self._embed.embed_sync(query)
            self._caches.set_embedding(query, vec)
            if self._disk_cache is not None:
                self._disk_cache.put(query, vec)
            return vec, True

        vec = await self._embed.embed(query)
        self._caches.set_embedding(query, vec)
        if self._disk_cache is not None:
            self._disk_cache.put(query, vec)
        return vec, False

    async def embed_batch_gpu_aware(self, texts: list[str]) -> list[list[float]]:
        """Batch embed; uses EmbeddingEngine.embed_batch when >= batch_min."""
        if not texts:
            return []
        pending: list[str] = []
        for t in texts:
            if self._caches.get_embedding(t) is None:
                pending.append(t)

        if self._should_skip_embed_for_gpu():
            for t in pending:
                v = self._embed.embed_sync(t)
                self._caches.set_embedding(t, v)
            return [self._caches.get_embedding(t) or [] for t in texts]

        if len(pending) >= self._batch_min:
            raw = await self._embed.embed_batch(pending)
            for t, v in zip(pending, raw):
                self._caches.set_embedding(t, v)
            return [self._caches.get_embedding(t) or [] for t in texts]

        for t in pending:
            v = await self._embed.embed(t)
            self._caches.set_embedding(t, v)
        return [self._caches.get_embedding(t) or [] for t in texts]

    def _hybrid_merge(
        self,
        query: str,
        results: list[tuple[str, float, dict[str, Any]]],
        now: float,
    ) -> list[str]:
        scored: list[tuple[float, str]] = []
        for text, vec_score, meta in results:
            if not text.strip():
                continue
            kw = _keyword_boost(query, text)
            rec = _recency_boost(meta, now)
            final = (
                self._vec_weight * vec_score
                + self._kw_weight * kw
                + self._rec_weight * rec
            )
            scored.append((final, text))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[: self._top_k]]

    def _search_vector_store(
        self,
        collection: str,
        query_embedding: list[float],
    ) -> list[tuple[str, float, dict[str, Any]]]:
        if self._vector_store is None:
            return []
        try:
            from core.vector_store import VectorSearchResult

            raw = self._vector_store.search(
                collection, query_embedding, k=self._top_k * 2, min_score=0.2,
            )
            out: list[tuple[str, float, dict[str, Any]]] = []
            for r in raw:
                out.append((r.text, r.score, r.metadata or {}))
            return out
        except Exception:
            logger.debug("vector_store search failed", exc_info=True)
            return []

    async def retrieve(
        self,
        query: str,
        *,
        memory_summaries: list[str] | None = None,
        system_state: dict[str, Any] | None = None,
        gpu_snapshot: dict[str, Any] | None = None,
        runtime_mode: str | None = None,
    ) -> RagRetrieveResult:
        """Full RAG path with hybrid search and structured enrichment."""
        t0 = time.perf_counter()
        rm = (runtime_mode or "").strip().upper()
        if not self._enabled:
            cq = classify_query(query)
            return RagRetrieveResult(
                [], [], "", 0.0, False, False, cq, 0.0, [], False,
            )

        complexity = classify_query(query)
        if complexity == QueryComplexity.SIMPLE and rm != "DEEP":
            return RagRetrieveResult(
                [], [], "", (time.perf_counter() - t0) * 1000, False, False, complexity,
                0.0, [], False,
            )

        graph_snippets: list[str] = []
        graph_conf = 0.0
        last_project: str | None = None
        if self._memory_graph is not None:
            graph_snippets, graph_conf = graph_snippets_for_query(
                self._memory_graph, query, limit=min(10, self._top_k + 4),
            )
            try:
                last_project = self._memory_graph.get_last_active_project()
            except Exception:
                last_project = None
            _gf0 = self._rag_cfg.get("graph_first") or {}
            if last_project and last_project.lower() in (query or "").lower():
                graph_conf = min(1.0, graph_conf + float(_gf0.get("project_boost", 0.12)))

        effective_fast = bool(self._fast_mode or rm == "FAST")
        _gf = self._rag_cfg.get("graph_first") or {}

        skip_gf_once = self._skip_graph_first_once
        if skip_gf_once:
            self._skip_graph_first_once = False
            logger.info("v7_graph_skip_once reason=validation_fallback_to_vector")

        # Graph-first: strong MemoryGraph signal → skip vector RAG (not DEEP)
        if (
            not skip_gf_once
            and bool(_gf.get("enabled", True))
            and rm != "DEEP"
            and not effective_fast
            and graph_snippets
            and graph_conf >= float(_gf.get("min_confidence", 0.68))
            and len(graph_snippets) >= int(_gf.get("min_snippets", 2))
        ):
            chunks = list(graph_snippets[: self._top_k])
            if memory_summaries:
                for ms in memory_summaries[: self._top_k]:
                    if ms and ms not in chunks:
                        chunks.append(ms)
                chunks = chunks[: self._top_k]
            block = build_rag_enrichment_block(
                system_state=system_state,
                gpu_snapshot=gpu_snapshot,
                memory_hints=memory_summaries,
                retrieved_chunks=None,
                user_query=query,
            )
            val_min = float(_gf.get("relevance_validation_min", 0.74))
            if self._feedback is not None:
                try:
                    if graph_conf >= val_min:
                        self._feedback.record_graph_hit()
                    else:
                        self._feedback.record_graph_miss()
                        self._skip_graph_first_once = True
                        logger.info(
                            "v7_graph_relevance_low conf=%.3f validation_min=%.3f next=vector_rag",
                            graph_conf,
                            val_min,
                        )
                except Exception:
                    pass
            logger.info(
                "v7_graph_hit conf=%.3f snippets=%d project=%s",
                graph_conf,
                len(graph_snippets),
                last_project or "",
            )
            return RagRetrieveResult(
                chunks=chunks,
                document_context=chunks,
                enrichment_block=block,
                latency_ms=(time.perf_counter() - t0) * 1000,
                cache_hit=False,
                skipped_embed=True,
                complexity=complexity,
                confidence=max(graph_conf, 0.35),
                graph_snippets=graph_snippets,
                retrieval_source="graph_first",
                graph_project_hint=last_project,
            )

        # Fast mode: graph + memory only — no vector embed (real-time path)
        if effective_fast:
            chunks = list(graph_snippets)
            if memory_summaries:
                chunks.extend(memory_summaries[: self._top_k])
            chunks = chunks[: self._top_k]
            block = build_rag_enrichment_block(
                system_state=system_state,
                gpu_snapshot=gpu_snapshot,
                memory_hints=memory_summaries,
                retrieved_chunks=None,
                user_query=query,
            )
            if self._feedback is not None:
                try:
                    self._feedback.record_graph_hit()
                except Exception:
                    pass
            return RagRetrieveResult(
                chunks=chunks,
                document_context=chunks,
                enrichment_block=block,
                latency_ms=(time.perf_counter() - t0) * 1000,
                cache_hit=False,
                skipped_embed=True,
                complexity=complexity,
                confidence=max(graph_conf, 0.35),
                graph_snippets=graph_snippets,
                retrieval_source="fast_graph",
                graph_project_hint=last_project,
            )

        cached = self._caches.get_retrieval(query)
        if cached is not None:
            block = build_rag_enrichment_block(
                system_state=system_state,
                gpu_snapshot=gpu_snapshot,
                memory_hints=memory_summaries,
                retrieved_chunks=None,
                user_query=query,
            )
            merged_chunks = list(graph_snippets) + [c for c in cached if c not in graph_snippets]
            merged_chunks = merged_chunks[: self._top_k]
            return RagRetrieveResult(
                chunks=merged_chunks,
                document_context=merged_chunks,
                enrichment_block=block,
                latency_ms=(time.perf_counter() - t0) * 1000,
                cache_hit=True,
                skipped_embed=False,
                complexity=complexity,
                confidence=0.88,
                graph_snippets=graph_snippets,
                prefetch_hit=True,
                retrieval_source="rag_cache",
                graph_project_hint=last_project,
            )

        if self._feedback is not None:
            try:
                self._feedback.record_rag_fallback()
            except Exception:
                pass
        logger.info("v7_rag_fallback reason=vector_embed query=%s", (query or "")[:80])
        q_emb, skipped = await self._embed_query(query)
        now = time.time()
        merged: list[tuple[str, float, dict[str, Any]]] = []

        if self._qdrant is not None:
            for h in self._qdrant.search(q_emb, limit=self._top_k * 2):
                merged.append((h.text, h.score, h.payload))

        if self._vector_store is not None:
            for coll in self._collections:
                merged.extend(self._search_vector_store(coll, q_emb))

        seen: set[str] = set()
        deduped: list[tuple[str, float, dict[str, Any]]] = []
        for text, score, meta in merged:
            key = text[:400] if text else ""
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append((text, score, meta))

        vec_conf = 0.0
        if deduped:
            vec_conf = min(1.0, max(s for _, s, _ in deduped))

        chunks = self._hybrid_merge(query, deduped, now)
        for gs in reversed(graph_snippets):
            if gs not in chunks:
                chunks.insert(0, gs)
        if not chunks and memory_summaries:
            chunks = memory_summaries[:3]
        chunks = chunks[: self._top_k]

        confidence = max(vec_conf, graph_conf, 0.2)

        self._caches.set_retrieval(query, chunks)

        block = build_rag_enrichment_block(
            system_state=system_state,
            gpu_snapshot=gpu_snapshot,
            memory_hints=memory_summaries,
            retrieved_chunks=None,
            user_query=query,
        )

        return RagRetrieveResult(
            chunks=chunks,
            document_context=chunks,
            enrichment_block=block,
            latency_ms=(time.perf_counter() - t0) * 1000,
            cache_hit=False,
            skipped_embed=skipped,
            complexity=complexity,
            confidence=confidence,
            graph_snippets=graph_snippets,
            retrieval_source="vector_rag",
            graph_project_hint=last_project,
        )


async def retrieve_with_time_budget(
    engine: RagEngine,
    query: str,
    budget_ms: float,
    *,
    memory_summaries: list[str] | None = None,
    system_state: dict[str, Any] | None = None,
    gpu_snapshot: dict[str, Any] | None = None,
    runtime_mode: str | None = None,
    on_late_result: Callable[[RagRetrieveResult], Any] | None = None,
) -> RagRetrieveResult:
    """Wait up to budget_ms for RAG; on timeout return empty and keep retrieve running.

    Uses ``asyncio.wait`` (not ``wait_for``) so the retrieve task is not cancelled;
    ``on_late_result`` runs when retrieval finishes after the budget.
    """
    task = asyncio.create_task(
        engine.retrieve(
            query,
            memory_summaries=memory_summaries,
            system_state=system_state,
            gpu_snapshot=gpu_snapshot,
            runtime_mode=runtime_mode,
        ),
        name="rag_retrieve",
    )
    done, _pending = await asyncio.wait(
        {task},
        timeout=budget_ms / 1000.0,
        return_when=asyncio.FIRST_COMPLETED,
    )
    if task in done:
        try:
            return task.result()
        except Exception:
            logger.exception("RAG retrieve failed")
            return RagRetrieveResult(
                [], [], "", budget_ms, False, False, QueryComplexity.COMPLEX,
                0.0, [], False,
            )

    async def _late() -> None:
        try:
            res = await task
            if on_late_result:
                on_late_result(res)
        except Exception:
            logger.debug("late RAG task failed", exc_info=True)

    asyncio.create_task(_late(), name="rag_late_emit")
    return RagRetrieveResult(
        [], [], "", budget_ms, False, False, QueryComplexity.COMPLEX,
        0.0, [], False,
    )
