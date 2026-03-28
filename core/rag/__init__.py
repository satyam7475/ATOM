"""ATOM RAG — retrieval integrated with GPU scheduling and prompt layers."""

from core.rag.adaptive_budget import compute_adaptive_rag_budget_ms
from core.rag.context_builder import build_rag_enrichment_block
from core.rag.embedding_disk_cache import PersistentEmbeddingCache, normalize_query_for_cache
from core.rag.graph_rag import graph_snippets_for_query
from core.rag.prefetch_engine import (
    RagPrefetchEngine,
    merge_prefetch_candidates,
    predict_followup_queries,
)
from core.rag.query_classifier import QueryComplexity, classify_query
from core.rag.rag_cache import RagCaches, TTLCache
from core.rag.rag_engine import RagEngine, RagRetrieveResult, retrieve_with_time_budget

__all__ = [
    "RagEngine",
    "RagRetrieveResult",
    "retrieve_with_time_budget",
    "build_rag_enrichment_block",
    "classify_query",
    "QueryComplexity",
    "RagCaches",
    "TTLCache",
    "compute_adaptive_rag_budget_ms",
    "PersistentEmbeddingCache",
    "normalize_query_for_cache",
    "graph_snippets_for_query",
    "RagPrefetchEngine",
    "merge_prefetch_candidates",
    "predict_followup_queries",
]
