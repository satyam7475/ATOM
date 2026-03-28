"""
ATOM V7 — Optional fused GPU cognition worker (single process).

Runs STT + LocalBrainController + shared GPUResourceManager policy in one process
to minimize cross-process VRAM churn. Enable via config ``v7_gpu.fused_gpu_worker: true``
and launch this script instead of separate stt_worker + llm_worker when desired.

Owner: Satyam
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config_manager import load_config
from core.gpu_resource_manager import GPUResourceManager
from core.ipc.zmq_bus import ZmqEventBus
from core.state_manager import StateManager
from cursor_bridge.local_brain_controller import LocalBrainController
from cursor_bridge.structured_prompt_builder import StructuredPromptBuilder
from voice.mic_manager import MicManager
from voice.stt_async import STTAsync

logger = logging.getLogger("atom.services.gpu_cognition")


class GpuCognitionWorker:
    """Single-process perception + LLM for maximum GPU locality."""

    def __init__(self) -> None:
        self.config = load_config()
        self.bus = ZmqEventBus(worker_name="gpu_cognition_worker")
        self.state = StateManager(self.bus)
        self.mic_manager = MicManager()
        self.mic_manager.profile_devices()

        self.gpu_rm = GPUResourceManager(self.bus, self.config)

        self.prompt_builder = StructuredPromptBuilder(self.config)
        self.brain = LocalBrainController(
            self.bus,
            self.prompt_builder,
            self.config,
        )
        self.brain.attach_gpu_resource_manager(self.gpu_rm)

        self.stt = STTAsync(
            self.bus,
            self.state,
            self.config,
            mic_manager=self.mic_manager,
        )

        self.bus.on("llm_query_request", self.handle_query_request)

    async def handle_query_request(self, event: str, **data) -> None:
        text = data.get("text", "")
        trace_id = data.get("trace_id")
        await self.brain.on_query(
            text,
            memory_context=data.get("memory_context"),
            context=data.get("context"),
            history=data.get("history"),
            trace_id=trace_id,
        )

    async def run(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        logger.info("GpuCognitionWorker starting (fused STT+LLM process)")
        self.bus.start()
        self.gpu_rm.start_power_task()
        await asyncio.sleep(10**9)


if __name__ == "__main__":
    w = GpuCognitionWorker()
    try:
        asyncio.run(w.run())
    except KeyboardInterrupt:
        pass
