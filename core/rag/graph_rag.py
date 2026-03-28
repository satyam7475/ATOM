"""Graph-augmented RAG using MemoryGraph (user → tasks → actions → files)."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from brain.memory_graph import MemoryGraph, MemoryNode

logger = logging.getLogger("atom.rag.graph")


def graph_snippets_for_query(
    memory_graph: "MemoryGraph | None",
    query: str,
    *,
    limit: int = 8,
) -> tuple[list[str], float]:
    """Retrieve structured snippets + confidence from hybrid graph query.

    Returns (snippets, confidence) where confidence is best semantic score in [0,1].
    """
    if memory_graph is None or not query.strip():
        return [], 0.0

    try:
        nodes: List["MemoryNode"] = memory_graph.query(
            {"text": query, "query_type": "knowledge"},
            limit=limit,
        )
        if not nodes:
            nodes = memory_graph.query({"type": "episodic"}, limit=min(limit, 12))
    except Exception:
        logger.debug("MemoryGraph query failed", exc_info=True)
        return [], 0.0

    if not nodes:
        return [], 0.0

    snippets: list[str] = []
    best_sim = 0.0

    for node in nodes:
        text = (node.embedding_text or "").strip()
        if not text:
            try:
                text = json.dumps(node.data, default=str)[:500]
            except Exception:
                text = str(node.data)[:500]
        rel_bits: list[str] = []
        for edge in (node.relationships or [])[:12]:
            if len(edge) >= 3:
                rel_bits.append(f"{edge[1]}:{str(edge[2])[:40]}")
        rel_str = (" | ".join(rel_bits)) if rel_bits else ""
        line = f"[{node.type}] {text[:400]}"
        if rel_str:
            line += f" :: {rel_str}"
        snippets.append(line)
        best_sim = max(best_sim, min(1.0, float(node.importance or 1.0) / 3.0))

    confidence = min(1.0, best_sim + 0.15 * min(1.0, len(nodes) / float(limit)))
    return snippets, confidence
