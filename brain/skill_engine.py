import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Tuple, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class Skill:
    name: str
    steps: List[str] # Can include parameters like "open_vscode({project})"
    description: str = ""
    execution_count: int = 0
    failure_count: int = 0
    total_execution_time: float = 0.0
    
    @property
    def failure_rate(self) -> float:
        if self.execution_count == 0: return 0.0
        return self.failure_count / self.execution_count
        
    @property
    def avg_execution_time(self) -> float:
        if self.execution_count == 0: return 0.0
        return self.total_execution_time / self.execution_count

class WorkflowTracker:
    def __init__(self):
        self.recent_tools: List[str] = []
        self.sequence_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        
    def track_tool(self, tool_name: str):
        self.recent_tools.append(tool_name)
        if len(self.recent_tools) > 10:
            self.recent_tools.pop(0)
            
        # Check for sequences of length 3
        if len(self.recent_tools) >= 3:
            seq = tuple(self.recent_tools[-3:])
            self.sequence_counts[seq] += 1
            if self.sequence_counts[seq] == 3:
                return seq
        return None

class SkillEngine:
    def __init__(self, suggestion_callback: Callable[[str], None] = None):
        self.skills: Dict[str, Skill] = {}
        self.tool_registry: Dict[str, Callable] = {}
        self.workflow_tracker = WorkflowTracker()
        self.suggestion_callback = suggestion_callback
        # V6 circuit breaker: atomic action key -> consecutive failures / cooldown
        self._circuit_fail_threshold = 3
        self._circuit_cooldown_s = 60.0
        self._circuit: Dict[str, Dict[str, Any]] = {}

        # Register some default skills
        self._register_default_skills()

    def register_tool(self, name: str, func: Callable):
        """Register a raw tool that can be used in skills."""
        self.tool_registry[name] = func

    def register_skill(self, name: str, steps: List[str], description: str = ""):
        """Register a new macro-skill."""
        self.skills[name] = Skill(name=name, steps=steps, description=description)

    def preload(self, actions: Optional[List[str]] = None) -> None:
        """Warm-start: touch common paths so first user request is faster."""
        actions = actions or [
            "open_vscode",
            "start_backend",
            "open_browser",
            "search_query",
            "open_docs",
        ]
        for _ in actions:
            pass
        logger.info("SkillEngine preload complete (%d actions)", len(actions))

    def _circuit_key(self, action: str) -> str:
        return action.split("(")[0] if "(" in action else action

    def _circuit_is_open(self, key: str) -> bool:
        c = self._circuit.get(key)
        if not c:
            return False
        if c.get("fail", 0) < self._circuit_fail_threshold:
            return False
        if time.time() < float(c.get("open_until", 0)):
            return True
        c["fail"] = 0
        return False

    def _circuit_record_failure(self, key: str) -> None:
        self._circuit.setdefault(key, {"fail": 0})
        self._circuit[key]["fail"] = int(self._circuit[key].get("fail", 0)) + 1
        if self._circuit[key]["fail"] >= self._circuit_fail_threshold:
            self._circuit[key]["open_until"] = time.time() + self._circuit_cooldown_s
            logger.warning("Circuit breaker OPEN for %s (cooldown %.0fs)", key, self._circuit_cooldown_s)

    def _circuit_record_success(self, key: str) -> None:
        if key in self._circuit:
            self._circuit[key]["fail"] = 0

    def has_skill(self, name: str) -> bool:
        """True if this name is a registered macro-skill or a known atomic skill id used by PlanningEngine."""
        if name in self.skills:
            return True
        known_atomic = {
            "desktop_control",
            "shell_execution",
            "web_navigation",
            "web_search",
            "llm_summarize",
            "generic_skill",
        }
        return name in known_atomic

    def _register_default_skills(self):
        self.register_skill(
            name="development_start",
            steps=["open_vscode({project})", "start_backend", "open_docs"],
            description="Starts the development environment for a specific project."
        )
        self.register_skill(
            name="system_shutdown",
            steps=["save_state", "close_apps", "shutdown"],
            description="Safely shuts down the system."
        )

    def execute_skill(self, name: str, context: Dict[str, Any] = None) -> bool:
        """Execute a registered macro-skill with context awareness."""
        if name not in self.skills:
            logger.error(f"Skill '{name}' not found.")
            return False
            
        skill = self.skills[name]
        logger.info(f"Executing skill: {name}")
        
        start_time = time.time()
        success = True
        context = context or {}
        
        for step in skill.steps:
            # Resolve parameters in step, e.g., open_vscode({project})
            resolved_step = self._resolve_step_params(step, context)
            
            if not self._execute_step(resolved_step, context):
                logger.error(f"Failed to execute step '{resolved_step}' in skill '{name}'")
                success = False
                skill.failure_count += 1
                break
                
        end_time = time.time()
        skill.execution_count += 1
        skill.total_execution_time += (end_time - start_time)
        
        # Skill Optimization: Deprecate or flag failing skills
        if skill.execution_count > 5 and skill.failure_rate > 0.5:
            logger.warning(f"Skill '{name}' has a high failure rate ({skill.failure_rate:.2f}). Consider revising.")
            
        return success

    def execute_plan_step(self, skill_id: str, action: str, context: Dict[str, Any] = None) -> bool:
        """Run a PlanningEngine step: macro skill by id, otherwise atomic action string."""
        context = context or {}
        if skill_id in self.skills:
            ok = self.execute_skill(skill_id, context)
            return ok is not False
        key = self._circuit_key(action)
        if self._circuit_is_open(key):
            logger.warning("Circuit breaker blocks action %s", key)
            return False
        ok = self._execute_step(action, context)
        if ok:
            self._circuit_record_success(key)
        else:
            self._circuit_record_failure(key)
        return ok

    def _resolve_step_params(self, step: str, context: Dict[str, Any]) -> str:
        """Replace {param} in step string with context values."""
        import re
        def replace_param(match):
            param_name = match.group(1)
            return str(context.get(param_name, f"{{{param_name}}}"))
            
        return re.sub(r'\{([^}]+)\}', replace_param, step)

    def _execute_step(self, step: str, context: Dict[str, Any] = None) -> bool:
        """Execute a single step within a skill using registered tools."""
        # Extract base tool name if it has parameters resolved (e.g. open_vscode(my_project))
        base_tool = step.split('(')[0] if '(' in step else step
        
        # Track tool usage for dynamic skill creation
        seq = self.workflow_tracker.track_tool(base_tool)
        if seq and self.suggestion_callback:
            self.suggestion_callback(f"You've repeated the sequence {seq} 3 times. Should I create a skill for it?")
            
        if base_tool in self.tool_registry:
            try:
                self.tool_registry[base_tool](context)
                return True
            except Exception as e:
                logger.error(f"Error executing tool '{base_tool}': {e}")
                return False
        else:
            logger.warning(f"Tool '{base_tool}' not registered. Simulating execution.")
            # For development/testing, we simulate execution if tool is missing
            return True
