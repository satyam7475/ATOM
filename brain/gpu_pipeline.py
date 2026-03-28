"""
ATOM V7 — In-process GPU pipeline façade (embed → memory → LLM ordering).

Reduces ad-hoc sequencing in LocalBrainController; optional parallel prefetch.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.gpu_resource_manager import GPUResourceManager

logger = logging.getLogger("atom.brain.gpu_pipeline")


@dataclass
class PipelineTrace:
    trace_id: str
    stages_ms: dict[str, float] = field(default_factory=dict)
    decision_path: list[str] = field(default_factory=list)
    error: str | None = None

    def record(self, stage: str, ms: float) -> None:
        self.stages_ms[stage] = ms
        self.decision_path.append(stage)


class GPUPipeline:
    """Single-process stages: parallel optional prefetch + ordered reasoning."""

    def __init__(
        self,
        config: dict | None = None,
        gpu_resource_manager: "GPUResourceManager | None" = None,
    ) -> None:
        self._config = config or {}
        self._gpu = gpu_resource_manager
        self._v7 = (self._config.get("v7_gpu") or {})
        self._speculative = bool(self._v7.get("speculative_response", False))

    async def run_retrieval_stage(
        self,
        query: str,
        memory_engine: Any,
        trace: PipelineTrace,
        parallel_prefetch: bool = True,
    ) -> list[str]:
        """Embedding + memory retrieve (hybrid inside MemoryEngine)."""
        t0 = time.perf_counter()
        try:
            if parallel_prefetch and hasattr(memory_engine, "retrieve"):
                memories = await memory_engine.retrieve(
                    query, k=self._config.get("memory", {}).get("top_k", 5),
                )
            else:
                memories = await memory_engine.retrieve(query, k=5)
        except Exception as e:
            trace.error = str(e)
            logger.debug("retrieve stage failed", exc_info=True)
            memories = []
        trace.record("memory_retrieve", (time.perf_counter() - t0) * 1000)
        return memories

    async def build_context_parallel(
        self,
        query: str,
        memory_engine: Any,
        intent_fn: Any,
        trace: PipelineTrace,
    ) -> tuple[list[str], Any]:
        """Parallel intent classification + memory lookup when possible."""
        t0 = time.perf_counter()

        async def _mem() -> list[str]:
            return await self.run_retrieval_stage(query, memory_engine, trace)

        async def _intent():
            if intent_fn is None:
                return None
            if asyncio.iscoroutinefunction(intent_fn):
                return await intent_fn(query)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, intent_fn, query)

        mem_task = asyncio.create_task(_mem())
        intent_task = asyncio.create_task(_intent()) if intent_fn else None

        memories = await mem_task
        intent_result = await intent_task if intent_task else None
        trace.record("parallel_intent_memory", (time.perf_counter() - t0) * 1000)
        return memories, intent_result

    def attach_gpu_manager(self, mgr: "GPUResourceManager | None") -> None:
        self._gpu = mgr

    def refresh_gpu_budget(self) -> None:
        if self._gpu is not None:
            self._gpu.refresh_vram()


__all__ = ["GPUPipeline", "PipelineTrace"]
