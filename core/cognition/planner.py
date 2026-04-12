"""
ATOM OS -- Planner Engine
Takes a high-level reasoning intent and decomposes it into a sequence of executable tool chains
to prevent reactive single-shot failures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Dict, Any

logger = logging.getLogger("atom.planner")

@dataclass
class PlanStep:
    step_num: int
    description: str
    target_tool: str
    expected_args: Dict[str, Any]
    
@dataclass
class ExecutionPlan:
    goal: str
    steps: List[PlanStep]

class PlannerEngine:
    """Multi-Step Action Planner Generator."""
    
    def __init__(self, ai_client=None):
        self.ai = ai_client
        
    async def generate_plan(self, query: str, context: str) -> ExecutionPlan:
        """
        In production, this queries the LLM (Gemini or MLX) to decompose the query.
        For deterministic speed guarantees, we use specific heuristic templates
        if recognized.
        """
        logger.info(f"Generating execution plan for: {query}")
        
        q_lower = query.lower()
        steps = []
        
        if "clean" in q_lower and ("cache" in q_lower or "temp" in q_lower):
            steps = [
                PlanStep(1, "Scan for temp files", "find_large_files", {"min_size_mb": 10}),
                PlanStep(2, "Analyze safety", "system_analyze", {}),
                PlanStep(3, "Ask boss for confirmation before wiping", "ask_user_confirmation", {}),
                PlanStep(4, "Wipe files safely", "run_terminal_command", {"command": "rm -rf ~/.cache/*"})
            ]
        elif "close" in q_lower and "apps" in q_lower:
            steps = [
                PlanStep(1, "Get running apps", "get_running_apps", {}),
                PlanStep(2, "Close apps sequentially", "close_app", {"name": "target"})
            ]
        else:
            # Generic 1-step fallback if no complex sequence is needed
            steps = [
                PlanStep(1, "Execute inferred action", "router_dispatch", {"query": query})
            ]
            
        return ExecutionPlan(goal=query, steps=steps)
