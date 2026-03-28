import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ipc.zmq_bus import ZmqEventBus
from brain.context_router import ContextRouter
from core.contracts import CONTRACTS

logger = logging.getLogger("atom.services.context")

class ContextWorker:
    def __init__(self):
        self.bus = ZmqEventBus(worker_name="context_worker")
        self.router = ContextRouter()
        self.contract = CONTRACTS["context_engine"]
        
        self.bus.register_request_handler("build_context", self.handle_build_context)

    async def handle_build_context(self, **data):
        intent = data.get("intent", {})
        system_state = data.get("system_state", {})
        
        # In a real system, memory would be fetched here or passed in
        context = self.router.build_context(intent, system_state, None)
        return {"result": {"enriched_context": context}}

    async def run(self):
        logging.basicConfig(level=logging.INFO)
        logger.info("Starting Context Worker...")
        self.bus.start()
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            self.bus.stop()

if __name__ == "__main__":
    worker = ContextWorker()
    asyncio.run(worker.run())
