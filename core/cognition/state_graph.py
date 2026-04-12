"""
ATOM OS -- System State Graph
Tracks temporal elements of the OS: running apps, recency, system load, user mode.
"""

from __future__ import annotations
import time
from typing import Dict, Any, List

class SystemStateGraph:
    """Singleton memory of OS state across time."""
    
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_state()
        return cls._instance
        
    def _init_state(self):
        self.active_apps: List[str] = []
        self.recent_actions: List[Dict[str, Any]] = []
        self.last_query_time: float = time.time()
        self.system_load: float = 0.0
        
    def update_active_apps(self, apps: List[str]):
        """Called by background daemon to refresh visual graph."""
        self.active_apps = apps
        
    def log_action(self, tool_name: str, args: dict, success: bool):
        self.recent_actions.append({
            "time": time.time(),
            "tool": tool_name,
            "args": args,
            "success": success
        })
        if len(self.recent_actions) > 20:
            self.recent_actions.pop(0)
            
    def get_context_summary(self) -> str:
        """Injects directly into ContextFusion."""
        state = f"Active Apps: {', '.join(self.active_apps) if self.active_apps else 'Unknown'}\n"
        if self.recent_actions:
            last = self.recent_actions[-1]
            state += f"Last Action: {last['tool']} (Success: {last['success']})\n"
        return state
