import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ipc.zmq_bus import ZmqEventBus
from brain.memory_graph import MemoryGraph
from core.contracts import CONTRACTS

logger = logging.getLogger("atom.services.memory")

class MemoryWorker:
    def __init__(self):
        self.bus = ZmqEventBus(worker_name="memory_worker")
        self.memory_graph = MemoryGraph()
        self.contract = CONTRACTS["memory_engine"]
        
        self.bus.register_request_handler("query_memory", self.handle_query_memory)

    async def handle_query_memory(self, **data):
        query_params = data.get("query", {})
        context = data.get("filters", {}).get("context", {})
        limit = data.get("filters", {}).get("limit", 10)
        
        nodes = self.memory_graph.query(query_params, context, limit)
        
        # Serialize nodes
        serialized_nodes = []
        for n in nodes:
            serialized_nodes.append({
                "id": n.id,
                "type": n.type,
                "data": n.data,
                "importance": n.importance
            })
            
        return {"result": {"nodes": serialized_nodes, "scores": []}}

    async def run(self):
        logging.basicConfig(level=logging.INFO)
        logger.info("Starting Memory Worker...")
        self.bus.start()
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            self.bus.stop()

if __name__ == "__main__":
    worker = MemoryWorker()
    asyncio.run(worker.run())
