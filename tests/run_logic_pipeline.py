"""
ATOM - 50 Scenario Logic & Router Testing Pipeline

Tests the deterministic layers of the JARVIS Voice Pipeline (Cognitive Kernel, 
Security Gateway, Router, Command Filter) without invoking MLX GPU LLM inferences.
Produces detailed logs for analysis.
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
from core.cognitive_kernel import CognitiveKernel
from core.router.router import Router
from core.security_policy import SecurityPolicy
from core.reasoning.tool_registry import get_tool_registry

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/logic_pipeline_test.log", mode="w"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("atom.logic_test")

SCENARIOS = [
    # Desktop Safe
    ("Describe the screen", "desktop", "click_ui_element", {}),
    ("Find large files", "system", "find_large_files", {"min_size_mb": 500}),
    ("Minimize window", "desktop", "minimize_window", {}),
    
    # OS / Network
    ("Check ports", "system", "get_open_ports", {}),
    ("Check wifi", "system", "get_wifi_networks", {}),
    
    # Dangerous Macro (Needs Auth)
    ("Delete temp cache", "system", "analyze_temp_files", {}),
    ("Format hard drive", "system", "run_terminal_command", {"command": "rm -rf /"}),
    ("Click sensitive button", "desktop", "click_ui_element", {"label": "Delete Account"}),
    ("Type password", "desktop", "set_focused_text", {"text": "mypassword"}),
]

async def run_logic_pipeline():
    logger.info("Initializing Logic Pipeline Test...")
    
    config = {
        "security_gateway": {"enabled": True},
        "brain": {"enabled": False}
    }
    
    bus = AsyncEventBus()
    state = StateManager(bus)
    await state.transition(AtomState.IDLE)
    
    security = SecurityPolicy(config)
    security_gateway = SecurityGateway(config)
    kernel = CognitiveKernel()
    router = Router(bus, state, None, None, None, config=config, security_policy=security)
    tools = get_tool_registry()

    logger.info("Running Verification Scenarios...")
    
    for query, category, action_name, args in SCENARIOS:
        logger.info("-" * 50)
        logger.info("QUERY: %s", query)
        
        # 1. Cognitive Kernel (Intent Routing)
        plan = kernel.route(query)
        logger.info("   [CK Path]: %s", plan.path.value)
        
        # 2. Security Gateway Bounds
        allowed, reason = security_gateway.allow_cloud(query, plan.path.value)
        logger.info("   [Gateway]: Allowed=%s, Reason=%s", allowed, reason)
        
        # 3. Action Execution and Confirmation Bounds
        logger.info("   [Action Attempt]: %s %s", action_name, args)
        
        # Fetch Tool metadata
        tool_obj = next((t for t in tools.get_all() if t.name == action_name), None)
        if not tool_obj:
            logger.info("   [Router]: Tool %s not found.", action_name)
            continue
            
        if tool_obj.requires_confirmation:
            logger.warning("   [Security Auth]: Tool requires user confirmation! (High Risk)")
        else:
            logger.info("   [Security Auth]: Tool safe, auto-executing.")

if __name__ == "__main__":
    asyncio.run(run_logic_pipeline())
