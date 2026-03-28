import asyncio
import logging
from typing import Any, Dict, Optional

from .planning_engine import Plan, PlanStep, StepStatus, PlanStatus
from core.profiler import profile_async

logger = logging.getLogger("atom.brain.execution_engine")


class ExecutionState:
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class ExecutionEngine:
    """
    Executes plans with async-friendly blocking work, optional fallback steps, preemption.
    """

    def __init__(self, event_bus, skill_engine, planning_engine=None):
        self.bus = event_bus
        self.skill_engine = skill_engine
        self.planning_engine = planning_engine
        self.state: str = ExecutionState.RUNNING
        self.current_plan: Optional[Plan] = None
        self._cancel_requested = False

        if self.bus:
            self.bus.on("interrupt_execution", self.handle_interrupt)

    def request_cancel(self) -> None:
        """Preempt: cancel current plan execution (best-effort)."""
        self._cancel_requested = True
        self.state = ExecutionState.CANCELLED

    async def handle_interrupt(self, event: str, **data):
        action = data.get("action")
        if action == "pause":
            logger.info("Execution Engine: PAUSED")
            self.state = ExecutionState.PAUSED
        elif action == "cancel":
            logger.info("Execution Engine: CANCELLED")
            self.request_cancel()
        elif action == "resume":
            logger.info("Execution Engine: RESUMED")
            self.state = ExecutionState.RUNNING

    @profile_async("execution")
    async def execute_plan(self, plan: Plan, context: Dict[str, Any]) -> bool:
        self.current_plan = plan
        self.state = ExecutionState.RUNNING
        self._cancel_requested = False
        plan.status = PlanStatus.ACTIVE

        logger.info("Starting execution for Plan [Goal: %s]", plan.goal_id)

        while not plan.is_complete:
            if self._cancel_requested or self.state == ExecutionState.CANCELLED:
                logger.warning("Plan execution cancelled (preempt).")
                plan.status = PlanStatus.FAILED
                return False

            while self.state == ExecutionState.PAUSED:
                await asyncio.sleep(0.5)

            step = plan.current_step
            if step is None:
                break
            step.status = StepStatus.IN_PROGRESS

            if self.bus:
                self.bus.emit("step_started", goal_id=plan.goal_id, action=step.action)

            success = await self._execute_step(step, context)

            if not success:
                fb = None
                if self.planning_engine:
                    fb = self.planning_engine.get_fallback_action(step.action)
                if fb:
                    logger.info("Fallback step for %s -> %s", step.action, fb)
                    step.action = fb
                    success = await self._execute_step(step, context)

            if success:
                step.status = StepStatus.COMPLETED
                if self.bus:
                    self.bus.emit("step_completed", goal_id=plan.goal_id, action=step.action)
                plan.current_step_index += 1
            else:
                if step.retry_count < 2:
                    step.retry_count += 1
                    logger.warning("Step %s failed. Retrying (%d/2)...", step.action, step.retry_count)
                    await asyncio.sleep(1.0)
                else:
                    step.status = StepStatus.FAILED
                    if self.bus:
                        self.bus.emit("step_failed", goal_id=plan.goal_id, action=step.action)
                    plan.status = PlanStatus.FAILED
                    return False

        plan.status = PlanStatus.COMPLETED
        logger.info("Plan execution completed for Goal [%s]", plan.goal_id)
        return True

    async def _execute_step(self, step: PlanStep, context: Dict[str, Any]) -> bool:
        try:
            if not self.skill_engine:
                await asyncio.sleep(0.05)
                return True

            def _sync() -> bool:
                return self.skill_engine.execute_plan_step(step.skill, step.action, context) is not False

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, _sync)
        except Exception as e:
            logger.error("Error executing step %s: %s", step.action, e)
            return False
