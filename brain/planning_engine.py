import logging
from typing import List, Dict, Any, Optional

from brain.plan_registry import PlanRegistry
from core.profiler import profile

logger = logging.getLogger("atom.brain.planning_engine")

class StepStatus:
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class PlanStatus:
    DRAFT = "draft"
    VALIDATED = "validated"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"

class PlanStep:
    def __init__(self, action: str, skill: str, expected_outcome: str, dependencies: List[str] = None):
        self.action: str = action
        self.skill: str = skill
        self.expected_outcome: str = expected_outcome
        self.dependencies: List[str] = dependencies or []
        
        self.status: str = StepStatus.PENDING
        self.retry_count: int = 0

class Plan:
    def __init__(self, goal_id: str):
        self.goal_id: str = goal_id
        self.steps: List[PlanStep] = []
        self.current_step_index: int = 0
        self.status: str = PlanStatus.DRAFT
        self.template_id: Optional[str] = None

    def add_step(self, step: PlanStep):
        self.steps.append(step)

    @property
    def current_step(self) -> Optional[PlanStep]:
        if self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    @property
    def is_complete(self) -> bool:
        return self.current_step_index >= len(self.steps)

class PlanningEngine:
    """
    Converts Goals into Executable Plans.
    Supports Rule-based, Skill Graph, and LLM Planning strategies.
    """
    def __init__(self, skill_registry=None, security_policy=None, registry: Optional[PlanRegistry] = None):
        self.skill_registry = skill_registry
        self.security_policy = security_policy
        self.registry = registry or PlanRegistry()

    @profile("planning")
    def generate_plan(self, goal_id: str, objective: str, context: Dict[str, Any] = None) -> Plan:
        """Generates a plan using available strategies."""
        logger.info(f"Generating plan for Goal [{goal_id}]: {objective}")
        plan = Plan(goal_id)

        matched = self.registry.match_template(objective)
        if matched:
            tid, tmpl = matched
            plan.template_id = tid
            for s in tmpl.get("steps") or []:
                plan.add_step(
                    PlanStep(
                        action=s.get("action", ""),
                        skill=s.get("skill", "generic_skill"),
                        expected_outcome=s.get("expected_outcome", ""),
                        dependencies=s.get("dependencies") or [],
                    )
                )
            logger.info("Used plan template %s (%d steps)", tid, len(plan.steps))
            return plan

        # Strategy 1: Rule-based (fast)
        objective_lower = objective.lower()
        if "dev setup" in objective_lower or "backend" in objective_lower:
            plan.add_step(PlanStep("open_vscode", "desktop_control", "VSCode is open"))
            plan.add_step(PlanStep("start_backend", "shell_execution", "Backend server running", dependencies=["open_vscode"]))
            plan.add_step(PlanStep("open_docs", "web_navigation", "Documentation opened"))
        elif "research" in objective_lower or "search" in objective_lower:
            plan.add_step(PlanStep("open_browser", "web_navigation", "Browser opened"))
            plan.add_step(PlanStep("search_query", "web_search", "Search results retrieved", dependencies=["open_browser"]))
            plan.add_step(PlanStep("extract_info", "llm_summarize", "Information extracted and summarized", dependencies=["search_query"]))
        else:
            # Strategy 2/3 Fallback: Generic single-step execution
            # In a full implementation, this would query the Skill Graph or an LLM
            plan.add_step(PlanStep(objective, "generic_skill", "Action executed"))
            
        logger.info(f"Generated Plan with {len(plan.steps)} steps.")
        return plan

    def validate_plan(self, plan: Plan) -> bool:
        """
        Mandatory Plan Validation.
        Checks: skill exists, safe to execute, dependencies satisfied.
        """
        logger.info(f"Validating Plan for Goal [{plan.goal_id}]")
        
        executed_steps = set()
        
        for i, step in enumerate(plan.steps):
            # 1. Check if skill exists
            if self.skill_registry and not self.skill_registry.has_skill(step.skill):
                logger.error(f"Validation Failed: Skill '{step.skill}' does not exist.")
                return False
                
            # 2. Check if safe to execute
            if self.security_policy:
                allowed, reason = self.security_policy.allow_action(
                    step.action, policy_context="plan_validate",
                )
                if not allowed:
                    logger.error(f"Validation Failed: Action '{step.action}' is unsafe. Reason: {reason}")
                    return False
                    
            # 3. Check dependencies
            for dep in step.dependencies:
                # Simple check: dependency must be an action from a previous step
                if dep not in executed_steps:
                    logger.error(f"Validation Failed: Dependency '{dep}' for step '{step.action}' not satisfied.")
                    return False
                    
            executed_steps.add(step.action)
            
        plan.status = PlanStatus.VALIDATED
        logger.info("Plan validation successful.")
        return True

    def apply_adjustment(self, action: str, adjustment: dict) -> None:
        """Apply a planning template adjustment (e.g., add delay, set fallback)."""
        # Attach adjustments to the planning engine instance so future plans can reference them.
        if not hasattr(self, "template_adjustments"):
            self.template_adjustments = {}
        self.template_adjustments[action] = adjustment
        logger.info(f"Applied planning adjustment for {action}: {adjustment}")

    def get_fallback_action(self, action: str) -> Optional[str]:
        adj = getattr(self, "template_adjustments", {}).get(action) or {}
        fb = adj.get("fallback")
        return fb if isinstance(fb, str) and fb else None
