"""
ATOM V3 -- Standalone LLM Worker Process.

This script runs the `LocalBrainController` (llama-cpp-python) in its own process.
It connects to the ZMQ Broker, listens for `llm_query_request` events,
processes the query, and streams chunks back via `llm_chunk` events.

Owner: Satyam
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path so we can import core modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config_manager import load_config
from core.ipc.zmq_bus import ZmqEventBus
from core.ipc.interrupt_manager import SystemInterruptManager
from core.gpu_resource_manager import EVENT_GPU_ACK, EVENT_GPU_UNLOAD
from cursor_bridge.local_brain_controller import LocalBrainController

logger = logging.getLogger("atom.services.llm")


class LLMWorker:
    def __init__(self) -> None:
        self.config = load_config()
        self.bus = ZmqEventBus(worker_name="llm_worker")
        
        from cursor_bridge.structured_prompt_builder import StructuredPromptBuilder
        self.prompt_builder = StructuredPromptBuilder(self.config)
        
        # Initialize the actual Brain locally in this process
        logger.info("Initializing LocalBrainController in worker process...")
        self.brain = LocalBrainController(
            bus=self.bus,
            prompt_builder=self.prompt_builder,
            config=self.config
        )
        
        # Override the brain's internal chunk emitter to route over ZMQ
        self._original_emit_chunk = self.brain._emit_chunk
        self.brain._emit_chunk = self._zmq_emit_chunk
        
        self.current_req_id = None
        
        self.interrupt_mgr = SystemInterruptManager(self.bus, "llm_worker")
        self.interrupt_mgr.register_cancel_callback(self.handle_interrupt)
        
        # Register handlers
        self.bus.on("llm_query_request", self.handle_query_request)
        self.bus.on(EVENT_GPU_UNLOAD, self.handle_v7_gpu_unload)

    def _zmq_emit_chunk(self, chunk: str) -> None:
        """Intercept chunks from the brain and send them over ZMQ."""
        if self.current_req_id:
            self.bus.emit("llm_chunk", req_id=self.current_req_id, chunk=chunk)
        # Also call original if it does local logging
        self._original_emit_chunk(chunk)

    async def handle_query_request(self, event: str, **data) -> None:
        """Handle incoming query from main process."""
        self.current_req_id = data.get("req_id", "unknown")
        text = data.get("text", "")
        memory_context = data.get("memory_context")
        context = data.get("context")
        history = data.get("history")
        
        logger.info(f"LLM Worker received query {self.current_req_id}: {text[:30]}...")
        
        try:
            # This is a blocking/heavy call, but it's in its own process now!
            # It will stream chunks via _zmq_emit_chunk
            await self.brain.on_query(
                text,
                memory_context=memory_context,
                context=context,
                history=history,
            )
        except Exception as e:
            logger.error(f"LLM Engine error: {e}")
        finally:
            # Notify main process that the full response is done
            self.bus.emit("llm_done", req_id=self.current_req_id)
            logger.debug(f"LLM Worker finished request {self.current_req_id}")
            self.current_req_id = None

    async def handle_v7_gpu_unload(self, event: str, **data) -> None:
        if data.get("slot") != "llm":
            return
        logger.info("V7 GPU: unloading LLM in worker (power policy)")
        try:
            self.brain.unload_llm_for_power()
            self.bus.emit(EVENT_GPU_ACK, slot="llm", unloaded=True)
        except Exception as e:
            logger.error("V7 GPU unload LLM failed: %s", e)

    async def handle_interrupt(self) -> None:
        """Handle global interrupt signal."""
        logger.info("LLM Worker received interrupt signal. (Interrupting generation not fully supported yet in llama-cpp)")
        self.brain.request_preempt()

    async def run(self) -> None:
        """Main worker loop."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        logger.info("Starting LLM Worker Process...")
        
        self.bus.start()
        
        try:
            # Keep process alive
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            pass
        finally:
            self.bus.stop()
            logger.info("LLM Worker Process stopped.")


if __name__ == "__main__":
    worker = LLMWorker()
    try:
        asyncio.run(worker.run())
    except KeyboardInterrupt:
        pass
