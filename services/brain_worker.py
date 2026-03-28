"""
ATOM V4 -- Cognitive Brain Worker Process.

This worker acts as the central cognitive hub.
It listens for `speech_final` (from STT), processes it through the V4 Brain layers:
1. Intent Engine
2. Context Router
3. Memory Graph
4. Skill Engine (if direct action)

If LLM is needed, it forwards the request to the LLM Worker.
If no LLM is needed, it directly triggers skills and TTS.

Owner: Satyam
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ipc.zmq_bus import ZmqEventBus
from core.ipc.interrupt_manager import SystemInterruptManager
from brain.intent_engine import IntentEngine
from brain.context_router import ContextRouter
from brain.memory_graph import MemoryGraph
from brain.behavior_model import BehaviorModel
from brain.proactive_engine import ProactiveEngine
from brain.skill_engine import SkillEngine
from brain.personality_engine import PersonalityEngine

logger = logging.getLogger("atom.services.brain")

class BrainWorker:
    def __init__(self):
        self.bus = ZmqEventBus(worker_name="brain_worker")
        
        # Initialize V4 Brain Modules
        self.intent_engine = IntentEngine()
        self.context_router = ContextRouter()
        self.memory_graph = MemoryGraph()
        self.behavior_model = BehaviorModel()
        self.skill_engine = SkillEngine()
        self.personality_engine = PersonalityEngine()
        
        # Register a callback for proactive suggestions
        self.proactive_engine = ProactiveEngine(
            behavior_model=self.behavior_model,
            suggestion_callback=self._on_proactive_suggestion
        )
        
        self.interrupt_mgr = SystemInterruptManager(self.bus, "brain_worker")
        self.interrupt_mgr.register_cancel_callback(self.handle_interrupt)
        
        # Register handlers
        self.bus.on("speech_final", self.handle_speech_final)
        self.bus.on("system_event", self.handle_system_event)

    def _on_proactive_suggestion(self, suggestion: str):
        logger.info(f"Proactive Engine Suggestion: {suggestion}")
        self.bus.emit("response_ready", text=suggestion)

    async def handle_interrupt(self) -> None:
        """Handle global interrupt signal."""
        logger.info("Brain Worker received interrupt signal.")
        # Cancel any ongoing long-running skills or LLM routing

    async def handle_system_event(self, event: str, **data):
        """Track system events for behavior modeling."""
        kind = data.get("kind", "")
        if kind == "app_switch":
            app = data.get("app", "")
            self.behavior_model.track_app_usage(app, duration=0)
        elif kind == "command_executed":
            cmd = data.get("command", "")
            self.behavior_model.track_command_frequency(cmd)

    async def handle_speech_final(self, event: str, **data):
        text = data.get("text", "")
        if not text:
            return
            
        logger.info(f"Brain Worker processing input: '{text}'")
        start_time = time.time()
        
        # 1. Intent Engine (FAST path)
        intent = self.intent_engine.classify(text)
        logger.info(f"Intent classified: {intent.type} (confidence: {intent.confidence:.2f})")
        
        # 2. Parallel Fetch: System State & Memory
        async def fetch_system_state():
            # Simulate async fetch of system state
            return {
                "system_status": "ok",
                "active_tasks": [],
                "recent_chat": [],
                "current_time": time.strftime("%H:%M:%S")
            }
            
        async def fetch_memory():
            # Simulate async memory fetch
            return self.memory_graph

        current_state, memory = await asyncio.gather(
            fetch_system_state(),
            fetch_memory()
        )
        
        # 3. Context Router
        context = self.context_router.build_context(intent, current_state, memory)
        
        # 4. Decision Logic: LLM vs Direct Action
        if intent.type == "system" or intent.type == "task":
            # Check if we have a direct skill for this
            target = intent.entities.get("target")
            if target and target in self.skill_engine.skills:
                logger.info(f"Direct skill execution for: {target}")
                self.skill_engine.execute_skill(target, context)
                
                # Generate dynamic personality response
                response = self.personality_engine.generate_response(target, context)
                self.bus.emit("response_ready", text=response)
            else:
                # Fallback to LLM
                self._route_to_llm(text, context)
        else:
            # Chat or complex automation goes to LLM
            self._route_to_llm(text, context)
            
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Brain Worker processing took {elapsed:.2f}ms")

    def _route_to_llm(self, text: str, context: dict):
        req_id = f"req_{int(time.time()*1000)}"
        logger.info(f"Routing to LLM Worker (req_id: {req_id})")
        self.bus.emit("llm_query_request", req_id=req_id, text=text, context=context)

    async def run(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        logger.info("Starting V4 Brain Worker Process...")
        
        self.bus.start()
        self.proactive_engine.start()
        
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            pass
        finally:
            self.proactive_engine.stop()
            self.bus.stop()
            logger.info("Brain Worker Process stopped.")

if __name__ == "__main__":
    worker = BrainWorker()
    try:
        asyncio.run(worker.run())
    except KeyboardInterrupt:
        pass
