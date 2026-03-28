import hashlib
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger("atom.brain.goal_engine")

# Interrupt / preemption: higher than normal goals (0–1 scale); use values > 1.0 for urgency
PRIORITY_USER_INTERRUPT = 10.0

class GoalType:
    REACTIVE = "reactive"   # User command
    INFERRED = "inferred"   # Behavior model
    SYSTEM = "system"       # Self-maintenance

class GoalStatus:
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class Goal:
    def __init__(self, objective: str, goal_type: str, priority: float = 0.5, deadline: Optional[datetime] = None, context: Dict[str, Any] = None, trace_id: Optional[str] = None):
        self.id: str = str(uuid.uuid4())
        self.trace_id: str = trace_id or str(uuid.uuid4())
        self.objective: str = objective
        self.goal_type: str = goal_type
        self.priority: float = priority  # 0.0 to 1.0 (or >1 for interrupt urgency)
        self.deadline: Optional[datetime] = deadline
        self.status: str = GoalStatus.PENDING
        
        self.context_snapshot: Dict[str, Any] = context or {}
        if "trace_id" not in self.context_snapshot:
            self.context_snapshot["trace_id"] = self.trace_id
        self.sub_goals: List['Goal'] = []
        
        self.progress: float = 0.0  # 0.0 to 1.0
        self.confidence: float = 1.0
        
        self.created_at: datetime = datetime.now()
        self.updated_at: datetime = datetime.now()

    @property
    def deterministic_seed(self) -> int:
        """Reproducible seed for planning/exploration tied to goal identity + objective."""
        raw = f"{self.id}:{self.objective}".encode("utf-8")
        h = hashlib.sha256(raw).hexdigest()
        return int(h[:16], 16) % (2**31)

    def update_status(self, new_status: str):
        self.status = new_status
        self.updated_at = datetime.now()
        
    def update_progress(self, progress: float):
        self.progress = max(0.0, min(1.0, progress))
        self.updated_at = datetime.now()
        if self.progress >= 1.0 and self.status == GoalStatus.ACTIVE:
            self.update_status(GoalStatus.COMPLETED)

class GoalManager:
    """
    Manages the lifecycle of all goals in the system.
    Enforces the rule: Only ONE active primary goal at a time.
    """
    def __init__(self):
        self.goals: Dict[str, Goal] = {}
        self.active_goal_id: Optional[str] = None

    def create_goal(
        self,
        objective: str,
        goal_type: str,
        priority: float = 0.5,
        deadline: Optional[datetime] = None,
        context: Dict[str, Any] = None,
        trace_id: Optional[str] = None,
    ) -> Goal:
        goal = Goal(
            objective=objective,
            goal_type=goal_type,
            priority=priority,
            deadline=deadline,
            context=context,
            trace_id=trace_id,
        )
        self.goals[goal.id] = goal
        logger.info(f"Created {goal_type} Goal [{goal.id}]: {objective} (Priority: {priority})")
        self.prioritize_goals()
        return goal

    def prioritize_goals(self):
        """Sorts pending goals by priority and deadline."""
        pending_goals = [g for g in self.goals.values() if g.status == GoalStatus.PENDING]
        
        # Sort by priority (descending), then by deadline (ascending, None last)
        def sort_key(g: Goal):
            deadline_ts = g.deadline.timestamp() if g.deadline else float('inf')
            return (-g.priority, deadline_ts)
            
        pending_goals.sort(key=sort_key)
        return pending_goals

    def select_active_goal(self) -> Optional[Goal]:
        """Ensures only one active goal. Selects the highest priority pending goal if idle."""
        # Check if current active goal is still active
        if self.active_goal_id:
            current_active = self.goals.get(self.active_goal_id)
            if current_active and current_active.status == GoalStatus.ACTIVE:
                return current_active
            else:
                self.active_goal_id = None

        # Select next best goal
        pending_goals = self.prioritize_goals()
        if pending_goals:
            next_goal = pending_goals[0]
            next_goal.update_status(GoalStatus.ACTIVE)
            self.active_goal_id = next_goal.id
            logger.info(f"Selected new Active Goal [{next_goal.id}]: {next_goal.objective}")
            return next_goal
            
        return None

    def update_progress(self, goal_id: str, progress: float, status: Optional[str] = None):
        if goal_id not in self.goals:
            logger.warning(f"Attempted to update unknown goal: {goal_id}")
            return
            
        goal = self.goals[goal_id]
        goal.update_progress(progress)
        
        if status:
            goal.update_status(status)
            
        logger.info(f"Goal [{goal_id}] progress: {progress*100:.1f}%, status: {goal.status}")
        
        # If the active goal is completed/failed, free up the slot
        if goal_id == self.active_goal_id and goal.status in (GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.CANCELLED):
            self.active_goal_id = None
            self.select_active_goal()

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        return self.goals.get(goal_id)
