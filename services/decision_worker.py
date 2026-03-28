import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ipc.zmq_bus import ZmqEventBus
from brain.proactive_engine import ProactiveEngine
from brain.behavior_model import BehaviorModel
from core.contracts import CONTRACTS

logger = logging.getLogger("atom.services.decision")

class DecisionWorker:
    def __init__(self):
        self.bus = ZmqEventBus(worker_name="decision_worker")
        self.behavior_model = BehaviorModel()
        self.engine = ProactiveEngine(behavior_model=self.behavior_model)
        self.contract = CONTRACTS["decision_engine"]
        
        self.bus.register_request_handler("evaluate_action", self.handle_evaluate_action)

    async def handle_evaluate_action(self, **data):
        intent = data.get("intent", {})
        context = data.get("context", {})
        
        action = intent.get("intent_type", "unknown")
        confidence = intent.get("confidence", 0.0)
        
        risk = self.engine.risk_score(action)
        utility = self.engine._calculate_utility(action)
        context_alignment = self.engine._calculate_context_alignment(action, context)
        uncertainty = 1.0 - confidence
        
        score = (confidence * utility * context_alignment) - (risk * uncertainty)
        approved = score > 0.5
        
        return {
            "result": {
                "action": action,
                "score": score,
                "approved": approved
            }
        }

    async def run(self):
        logging.basicConfig(level=logging.INFO)
        logger.info("Starting Decision Worker...")
        self.bus.start()
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            self.bus.stop()

if __name__ == "__main__":
    worker = DecisionWorker()
    asyncio.run(worker.run())
