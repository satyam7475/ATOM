"""
Low-latency RAG engine (Apple Silicon).

- Smart skip (query classifier)
- TTL caches (embed + retrieval)
- Hybrid vector + keyword re-rank + temporal decay
- Owner-priority + usage-frequency boosts + stale-result marking
- Optional Qdrant + Chroma VectorStore fallback
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import threading
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
_STALE_PREFIX = "[Possibly outdated] "
_TIME_SENSITIVE_PATTERN = re.compile(
    r"\b("
    r"today|current|currently|latest|recent|new|now|version|versions|release|releases|"
    r"roadmap|deadline|eta|status|this\s+(?:week|month|year)|tomorrow|yesterday"
    r")\b",
    re.IGNORECASE,
)


def _tokenize(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _keyword_boost(query: str, text: str) -> float:
    q = _tokenize(query)
    t = _tokenize(text)
    if not q or not t:
        return 0.0
    inter = len(q & t)
    return min(1.0, inter / max(4, len(q)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _temporal_decay_score(
    metadata: dict[str, Any],
    now: float,
    half_life_s: float = 86400 * 7,
) -> float:
    ts = _safe_float(metadata.get("timestamp") or metadata.get("ts") or now, now)
    age = max(0.0, now - ts)
    if half_life_s <= 0:
        return 1.0
    return float(math.exp(-age / half_life_s))


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
        self._max_snippets = max(1, int(self._rag_cfg.get("max_snippets", 3)))
        self._top_k = max(
            1,
            min(int(self._rag_cfg.get("top_k", self._max_snippets)), self._max_snippets),
        )
        self._pressure_threshold_pct = float(
            self._rag_cfg.get("pressure_threshold_pct", 85.0),
        )
        self._pressure_relief_pct = float(
            self._rag_cfg.get("pressure_relief_pct", max(0.0, self._pressure_threshold_pct - 10.0)),
        )
        self._pressure_top_k = max(
            1,
            min(int(self._rag_cfg.get("pressure_top_k", 1)), self._top_k),
        )
        self._vec_weight = float(self._rag_cfg.get("vector_weight", 0.65))
        self._kw_weight = float(self._rag_cfg.get("keyword_weight", 0.25))
        self._rec_weight = float(self._rag_cfg.get("recency_weight", 0.10))
        smart_cfg = self._rag_cfg.get("smart_scoring") or {}
        owner_cfg = self._config.get("owner") or {}
        self._owner_name = str(owner_cfg.get("name", "Satyam")).strip().lower()
        self._owner_priority_multiplier = max(
            1.0,
            float(smart_cfg.get("owner_priority_multiplier", 2.0)),
        )
        self._owner_priority_sources = {
            str(src).strip().lower()
            for src in smart_cfg.get(
                "owner_priority_sources",
                ["preference", "voice", "owner", "boss", "user"],
            )
            if str(src).strip()
        }
        self._usage_boost_max = max(0.0, float(smart_cfg.get("usage_boost_max", 0.12)))
        self._usage_history_size = max(64, int(smart_cfg.get("usage_history_size", 4096)))
        self._usage_counts: dict[str, int] = {}
        self._usage_lock = threading.Lock()
        self._recency_half_life_s = max(
            3600.0,
            float(smart_cfg.get("recency_half_life_hours", 24 * 7)) * 3600.0,
        )
        self._stale_after_s = max(
            3600.0,
            float(smart_cfg.get("stale_after_hours", 24 * 45)) * 3600.0,
        )
        self._time_sensitive_stale_after_s = max(
            3600.0,
            float(smart_cfg.get("time_sensitive_stale_after_hours", 24 * 7)) * 3600.0,
        )
        self._stale_penalty = max(0.0, float(smart_cfg.get("stale_penalty", 0.18)))
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
        self._memory_pressure_active: bool = False

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
    def _normalize_chunk_text(text: str) -> str:
        chunk = (text or "").strip()
        if chunk.startswith(_STALE_PREFIX):
            chunk = chunk[len(_STALE_PREFIX):].lstrip()
        return " ".join(chunk.split())[:400]

    def _merge_unique_chunks(self, *groups: list[str] | None) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for group in groups:
            for chunk in group or []:
                norm = self._normalize_chunk_text(chunk)
                if not norm or norm in seen:
                    continue
                seen.add(norm)
                merged.append(chunk)
        return merged

    def _usage_key(self, text: str) -> str:
        return self._normalize_chunk_text(text)

    def _usage_boost(self, text: str) -> float:
        if self._usage_boost_max <= 0:
            return 0.0
        key = self._usage_key(text)
        if not key:
            return 0.0
        with self._usage_lock:
            count = int(self._usage_counts.get(key, 0))
        if count <= 0:
            return 0.0
        capped = min(count, 12)
        return (math.log1p(capped) / math.log1p(12)) * self._usage_boost_max

    def _record_chunk_usage(self, chunks: list[str]) -> None:
        with self._usage_lock:
            for chunk in chunks:
                key = self._usage_key(chunk)
                if not key:
                    continue
                if key not in self._usage_counts and len(self._usage_counts) >= self._usage_history_size:
                    oldest = next(iter(self._usage_counts))
                    del self._usage_counts[oldest]
                self._usage_counts[key] = self._usage_counts.get(key, 0) + 1

    def _is_owner_priority_match(self, metadata: dict[str, Any]) -> bool:
        owner_fields = (
            "owner", "speaker", "author", "created_by", "source", "role", "tags",
        )
        owner_tokens = {"boss", "owner", "user"}
        if self._owner_name:
            owner_tokens.add(self._owner_name)
        for field in owner_fields:
            value = metadata.get(field)
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                values = [str(v).strip().lower() for v in value if str(v).strip()]
            else:
                values = [part.strip().lower() for part in str(value).split(",") if part.strip()]
            for candidate in values:
                if candidate in self._owner_priority_sources:
                    return True
                if candidate in owner_tokens:
                    return True
                if self._owner_name and self._owner_name in candidate:
                    return True
        return False

    def _looks_time_sensitive(self, text: str, metadata: dict[str, Any]) -> bool:
        if _TIME_SENSITIVE_PATTERN.search(text or ""):
            return True
        return any(
            metadata.get(key) is not None
            for key in ("expires_at", "last_verified", "verified_at", "updated_at")
        )

    def _is_stale(self, text: str, metadata: dict[str, Any], now: float) -> bool:
        expires_at = _safe_float(metadata.get("expires_at"), 0.0)
        if expires_at > 0 and now >= expires_at:
            return True

        source = str(metadata.get("source") or "").strip().lower()
        collection = str(metadata.get("_collection") or "").strip().lower()
        if source == "preference" and not self._looks_time_sensitive(text, metadata):
            return False

        base_ts = metadata.get("last_verified") or metadata.get("verified_at") or metadata.get("updated_at")
        if base_ts is None:
            base_ts = metadata.get("timestamp") or metadata.get("ts")
        ts = _safe_float(base_ts, 0.0)
        if ts <= 0:
            return False

        age = max(0.0, now - ts)
        if self._looks_time_sensitive(text, metadata):
            return age >= self._time_sensitive_stale_after_s

        stale_sources = {
            "conversation",
            "migration",
            "llm_conversation",
            "voice",
            "dream_consolidation",
            "dream_pattern",
            "episodic",
            "web",
        }
        return (
            age >= self._stale_after_s
            and (collection in {"facts", "conversations"} or source in stale_sources)
        )

    def _format_chunk(self, text: str, metadata: dict[str, Any], now: float) -> str:
        chunk = (text or "").strip()
        if not chunk:
            return ""
        if self._is_stale(chunk, metadata, now) and not chunk.startswith(_STALE_PREFIX):
            return f"{_STALE_PREFIX}{chunk}"
        return chunk

    def _effective_top_k(self) -> int:
        if self._memory_pressure_active:
            return self._pressure_top_k
        return self._top_k

    def apply_memory_pressure(self, memory_pct: float) -> dict[str, Any]:
        """Shift RAG into minimal mode when unified memory is tight."""
        active = memory_pct >= self._pressure_threshold_pct
        if self._memory_pressure_active:
            active = memory_pct > self._pressure_relief_pct

        changed = active != self._memory_pressure_active
        self._memory_pressure_active = active

        if changed and active:
            logger.warning(
                "RAG pressure mode ON at %.0f%%: snippet budget reduced to %d",
                memory_pct,
                self._pressure_top_k,
            )
            try:
                self._caches.clear()
            except Exception:
                logger.debug("RAG cache clear failed", exc_info=True)
            try:
                self._embed.shutdown()
            except Exception:
                logger.debug("RAG embed shutdown failed", exc_info=True)
        elif changed and not active:
            logger.info(
                "RAG pressure mode OFF at %.0f%%: snippet budget restored to %d",
                memory_pct,
                self._top_k,
            )

        return {
            "active": self._memory_pressure_active,
            "memory_pct": round(memory_pct, 1),
            "top_k": self._effective_top_k(),
        }

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
        top_k: int,
    ) -> list[tuple[str, dict[str, Any]]]:
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for text, vec_score, meta in results:
            if not text.strip():
                continue
            meta = dict(meta or {})
            kw = _keyword_boost(query, text)
            rec = _temporal_decay_score(meta, now, self._recency_half_life_s)
            base_score = (
                self._vec_weight * vec_score
                + self._kw_weight * kw
                + self._rec_weight * rec
            )
            if self._is_owner_priority_match(meta):
                base_score *= self._owner_priority_multiplier
            final = base_score + self._usage_boost(text)
            if self._is_stale(text, meta, now):
                final -= self._stale_penalty
            scored.append((final, text, meta))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(t, m) for _, t, m in scored[: top_k]]

    def _search_vector_store(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        if self._vector_store is None:
            return []
        try:
            from core.vector_store import VectorSearchResult

            raw = self._vector_store.search(
                collection, query_embedding, k=top_k * 2, min_score=0.2,
            )
            out: list[tuple[str, float, dict[str, Any]]] = []
            for r in raw:
                meta = dict(r.metadata or {})
                meta["_collection"] = collection
                out.append((r.text, r.score, meta))
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
        top_k = self._effective_top_k()
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
                self._memory_graph, query, limit=min(10, top_k + 4),
            )
            try:
                last_project = self._memory_graph.get_last_active_project()
            except Exception:
                last_project = None
            _gf0 = self._rag_cfg.get("graph_first") or {}
            if last_project and last_project.lower() in (query or "").lower():
                graph_conf = min(1.0, graph_conf + float(_gf0.get("project_boost", 0.12)))

        effective_fast = bool(self._fast_mode or rm == "FAST" or self._memory_pressure_active)
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
            chunks = self._merge_unique_chunks(
                graph_snippets[: top_k],
                memory_summaries[: top_k] if memory_summaries else None,
            )[: top_k]
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
            self._record_chunk_usage(chunks)
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
            chunks = self._merge_unique_chunks(
                graph_snippets,
                memory_summaries[: top_k] if memory_summaries else None,
            )[: top_k]
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
            self._record_chunk_usage(chunks)
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
            merged_chunks = self._merge_unique_chunks(graph_snippets, cached)[: top_k]
            self._record_chunk_usage(merged_chunks)
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
            for h in self._qdrant.search(q_emb, limit=top_k * 2):
                meta = dict(h.payload or {})
                meta.setdefault("_collection", str(meta.get("collection") or "qdrant"))
                merged.append((h.text, h.score, meta))

        if self._vector_store is not None:
            for coll in self._collections:
                merged.extend(self._search_vector_store(coll, q_emb, top_k))

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

        ranked_chunks = self._hybrid_merge(query, deduped, now, top_k)
        meta_by_chunk = {
            self._normalize_chunk_text(text): metadata
            for text, metadata in ranked_chunks
        }
        raw_chunks = self._merge_unique_chunks(
            graph_snippets,
            [text for text, _meta in ranked_chunks],
        )
        if not raw_chunks and memory_summaries:
            raw_chunks = self._merge_unique_chunks(memory_summaries[:top_k])
        raw_chunks = raw_chunks[: top_k]
        chunks = [
            self._format_chunk(
                text,
                meta_by_chunk.get(self._normalize_chunk_text(text), {}),
                now,
            )
            for text in raw_chunks
        ]
        self._record_chunk_usage(chunks)

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
