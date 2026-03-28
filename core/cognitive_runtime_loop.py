"""
ATOM — Continuous cognitive loop (observe → predict → decide → act).

Optional background task integrating MemoryGraph + BehaviorModel with
GPUExecutionCoordinator: heavy work is submitted as low-priority GPU tasks;
observe/decide steps stay CPU-bound and lightweight.

Does not replace BrainOrchestrator; complements it for in-process Jarvis-style ticks.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

from core.gpu_execution_coordinator import PRIORITY_BACKGROUND, TaskIntent

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.gpu_execution_coordinator import GPUExecutionCoordinator
    from brain.memory_graph import MemoryGraph
    from brain.behavior_model import BehaviorModel

logger = logging.getLogger("atom.cognitive_runtime_loop")


class CognitiveRuntimeLoop:
    """Periodic observe/predict/decide/act with optional GPU follow-up."""

    def __init__(
        self,
        bus: "AsyncEventBus",
        memory_graph: "MemoryGraph",
        behavior_model: "BehaviorModel",
        coordinator: Optional["GPUExecutionCoordinator"] = None,
        *,
        interval_s: float = 30.0,
        config: dict | None = None,
    ) -> None:
        self._bus = bus
        self._memory = memory_graph
        self._behavior = behavior_model
        self._coord = coordinator
        self._config = config or {}
        self._interval_s = float(self._config.get("cognitive_loop", {}).get("interval_s", interval_s))
        self._task: asyncio.Task | None = None
        self._shutdown = False
        self._on_proactive: Callable[[dict[str, Any]], None] | None = None

    def set_proactive_callback(self, fn: Callable[[dict[str, Any]], None] | None) -> None:
        """Optional handler for decide() output (e.g. enqueue UI notification)."""
        self._on_proactive = fn

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._shutdown = False
            self._task = asyncio.create_task(self._loop(), name="atom_cognitive_loop")

    async def shutdown(self) -> None:
        self._shutdown = True
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def tick_once(self) -> dict[str, Any]:
        """Single iteration (for tests or manual triggers)."""
        return await self._tick_body()

    async def _loop(self) -> None:
        while not self._shutdown:
            try:
                await asyncio.sleep(self._interval_s)
                await self._tick_body()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.debug("cognitive loop tick", exc_info=True)

    async def _tick_body(self) -> dict[str, Any]:
        # --- Observe ---
        obs: dict[str, Any] = {
            "t": time.time(),
            "user_state": {
                "focus": self._behavior.state.focus,
                "stress": self._behavior.state.stress,
                "mode": self._behavior.state.mode,
            },
        }
        try:
            recent = self._memory.query({"type": "episodic"}, limit=3)
            obs["recent_episodic_nodes"] = len(recent)
        except Exception:
            obs["recent_episodic_nodes"] = 0

        # --- Predict (lightweight; CPU) ---
        prediction = {
            "should_surface_briefing": self._behavior.state.focus == "high",
            "stress_elevated": self._behavior.state.stress == "high",
        }
        obs["prediction"] = prediction

        # --- Decide ---
        decision = {
            "emit_proactive": bool(prediction["should_surface_briefing"]),
            "action": "none",
        }
        if decision["emit_proactive"] and self._on_proactive:
            try:
                self._on_proactive({**obs, **decision})
            except Exception:
                logger.debug("proactive callback failed", exc_info=True)

        payload = {"observe": obs, "decide": decision}
        self._bus.emit_fast("cognitive_tick", **payload)

        # --- Act (optional GPU: memory consolidation / embed batch) ---
        if self._coord is not None:
            async def _memory_embed_act() -> None:
                # Placeholder: real impl would call VectorStore batch; keep CPU-only here
                await asyncio.sleep(0)

            await self._coord.submit_task(
                kind="embed",
                priority=PRIORITY_BACKGROUND,
                intent=TaskIntent.MEMORY_UPDATE,
                name="cognitive_loop_embed",
                run=_memory_embed_act,
                vram_required_mb=64.0,
                estimated_duration_ms=200.0,
                allow_overlap=True,
                context={"source": "cognitive_loop"},
            )

        return payload


def get_cognitive_runtime_loop(
    bus: "AsyncEventBus",
    memory_graph: "MemoryGraph",
    behavior_model: "BehaviorModel",
    coordinator: Optional["GPUExecutionCoordinator"] = None,
    config: dict | None = None,
) -> CognitiveRuntimeLoop:
    return CognitiveRuntimeLoop(
        bus, memory_graph, behavior_model,
        coordinator=coordinator,
        config=config,
    )


__all__ = ["CognitiveRuntimeLoop", "get_cognitive_runtime_loop"]
