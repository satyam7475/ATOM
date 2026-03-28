"""Structured RAG context blocks for prompt assembly (Jarvis-style)."""

from __future__ import annotations

from typing import Any


def build_rag_enrichment_block(
    *,
    system_state: dict[str, Any] | None = None,
    gpu_snapshot: dict[str, Any] | None = None,
    memory_hints: list[str] | None = None,
    retrieved_chunks: list[str] | None = None,
    user_query: str = "",
) -> str:
    """Single structured block inserted before document layer.

    Reduces hallucination by separating signal sources.
    """
    parts: list[str] = []

    if system_state:
        lines = [f"- {k}: {v}" for k, v in list(system_state.items())[:12]]
        if lines:
            parts.append("SYSTEM STATE:\n" + "\n".join(lines))

    if gpu_snapshot:
        glines = [f"- {k}: {v}" for k, v in list(gpu_snapshot.items())[:8]]
        if glines:
            parts.append("GPU / RUNTIME:\n" + "\n".join(glines))

    if memory_hints:
        mh = "\n".join(f"- {s}" for s in memory_hints[:6])
        parts.append("MEMORY (short hints):\n" + mh)

    if retrieved_chunks:
        rh = "\n".join(f"- {c}" for c in retrieved_chunks[:8])
        parts.append("RETRIEVED (RAG):\n" + rh)

    if user_query:
        parts.append(f"USER QUERY (for grounding): {user_query[:500]}")

    if not parts:
        return ""
    return "\n\n".join(parts)
