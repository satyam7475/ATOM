import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ipc.zmq_bus import ZmqEventBus
from brain.intent_engine import IntentEngine
from core.contracts import CONTRACTS

logger = logging.getLogger("atom.services.intent")

class IntentWorker:
    def __init__(self):
        self.bus = ZmqEventBus(worker_name="intent_worker")
        self.engine = IntentEngine()
        self.contract = CONTRACTS["intent_engine"]
        
        # Register REQ/REP handler for critical path
        self.bus.register_request_handler("parse_intent", self.handle_parse_intent)

    async def handle_parse_intent(self, **data):
        text = data.get("text", "")
        if not isinstance(text, self.contract.input_schema["text"]):
            return {"error": "Invalid input schema"}
            
        intent = self.engine.classify(text)
        return {
            "result": {
                "intent_type": intent.type,
                "confidence": intent.confidence,
                "entities": intent.entities
            }
        }

    async def run(self):
        logging.basicConfig(level=logging.INFO)
        logger.info("Starting Intent Worker...")
        self.bus.start()
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            self.bus.stop()

if __name__ == "__main__":
    worker = IntentWorker()
    asyncio.run(worker.run())
