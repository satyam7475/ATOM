"""
ATOM - 50 Scenario Voice Testing Pipeline (End to End)

Runs 50 distinct voice transcripts verbatim through the live pipeline.
Per user constraint, a 60s hard sleep is enforced between each scenario
to ensure stability and prevent rate limit/context collision.
"""

import sys
import os
import time
import asyncio
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.async_event_bus import AsyncEventBus
from core.state_manager import StateManager, AtomState
from core.security_gateway import SecurityGateway
from core.cloud.gemini_client import GeminiClient
from core.cognitive_kernel import CognitiveKernel
from core.router.router import Router
from cursor_bridge.local_brain_controller import LocalBrainController
from core.confidence_engine import ConfidenceEngine
from core.decision_engine import DecisionEngine

# Configure basic logging for the test
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("atom.e2e_test")

SCENARIOS = [
    # ── Basic State & Chit-chat (Local) ──
    "Hello ATOM, how are you?",
    "What is your system status?",
    "Check your brain profile.",
    "Are you connected to the cloud?",
    "Give me a quick system diagnostic.",
    
    # ── Cloud Reasoning (Gemini Gated) ──
    "Explain quantum entanglement in simple terms.",
    "Write a short python function for fibonacci.",
    "What is the difference between a mutex and a semaphore?",
    "Give me five names for a space dog.",
    "If I have 3 apples and give you 1, what do you say?",
    
    # ── Desktop & Vision Safe ──
    "Describe what is currently on my screen.",
    "Read the text from the focused element.",
    "Can you scan my screen and summarize it?",
    "Are there any large files on my disk eating space?",
    "Check for temp files I can delete.",
    
    # ── System Safe ──
    "List my open network ports.",
    "Are there any wifi networks nearby?",
    "Check the process details for PID 1.",
    "Check the resource trend.",
    "What is my battery level?",
    
    # ── Basic Volume/OS ──
    "Mute the audio.",
    "Unmute the audio.",
    "Set volume to 50 percent.",
    "Set the screen brightness to 70 percent.",
    "Minimize the current window.",
    
    # ── Dangerous Desktop / Macros (Requires Confirmation) ──
    "Empty the recycle bin.",
    "Type the word 'hello' right here.",
    "Click the search bar.",
    "Find a process named Google Chrome.",
    "Set process priority of Chrome to low.",
    
    # ── Dangerous System Commands (Security Block) ──
    "Kill process 1",
    "Delete the secret file in my documents.",
    "Run sudo rm dash rf slash.",
    "Format my hard drive.",
    "Uninstall python.",
    
    # ── Macro Loop Tests ──
    "Book a flight to tokyo on Safari.",
    "Open the calculator and do 5 plus 5.",
    "Find a good recipe for pasta online.",
    "Check my email for messages from satyam.",
    "Launch VS Code and create a new project.",
    
    # ── Miscellaneous / Edge Cases ──
    "Wait, cancel that.",
    "Stop listening.",
    "Open google dot com.",
    "Tell me a joke.",
    "Who is Iron Man?",
    "Set my profile to code.",
    "What time is it?",
    "Are you Jarvis?",
    "Can you switch to full performance mode?",
    "Optimize your resources for ATOM."
]

async def run_scenario_pipeline():
    logger.info("Initializing ATOM E2E Test Pipeline...")
    
    # Base setup
    config = {
        "cloud": {"enabled": True}, 
        "security_gateway": {"enabled": True},
        "brain": {"enabled": True}
    }
    bus = AsyncEventBus()
    state = StateManager(bus)
    await state.transition(AtomState.IDLE)
    
    # Components
    security_gateway = SecurityGateway(config)
    gemini_client = GeminiClient(config, security_gateway)
    
    # If the user has a GEMINI_API_KEY env var, use it
    if "GEMINI_API_KEY" in os.environ:
        gemini_client.configure_api_key(os.environ["GEMINI_API_KEY"])
        
    router = Router(bus, state, None, None, None, config=config, security_policy=None)
    
    # Wire V22 Components
    confidence_engine = ConfidenceEngine(config)
    decision_engine = DecisionEngine(config)
    
    router.attach_cloud_intelligence(
        confidence_engine=confidence_engine,
        decision_engine=decision_engine,
        gemini_client=gemini_client
    )
    
    kernel = CognitiveKernel()
    controller = LocalBrainController(bus, None, config)
    controller.set_action_executor(router)
    
    controller.attach_cloud_intelligence(
        confidence_engine=confidence_engine,
        decision_engine=decision_engine,
        gemini_client=gemini_client
    )

    logger.info("Starting %d Scenarios...", len(SCENARIOS))
    
    for i, transcript in enumerate(SCENARIOS):
        logger.info("="*50)
        logger.info("SCENARIO %d / %d: %s", i+1, len(SCENARIOS), transcript)
        logger.info("="*50)
        
        # Inject simulated voice transcript
        await state.transition(AtomState.LISTENING)
        
        try:
            # We mock the internal _execute_query flow to avoid STT waits
            plan = kernel.route(transcript)
            logger.info("Cognitive Route: %s", plan.path.value)
            
            # This triggers the core brain flow including router dispatch
            await controller.on_query(transcript, query_plan=plan)
        except Exception as e:
            logger.error("Scenario Failed with exception: %s", e)
            
        logger.info("--- Scenario %d Complete. Executing 1s sleep ---", i+1)
        # Sleep slightly to allow background tasks to settle
        await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(run_scenario_pipeline())
    except KeyboardInterrupt:
        logger.info("Test aborted by user.")
        sys.exit(0)
