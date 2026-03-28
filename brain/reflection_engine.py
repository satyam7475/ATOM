import asyncio
import logging
import time
from typing import List

from .learning_engine import LearningEngine, Experience

logger = logging.getLogger("atom.brain.reflection_engine")

class ReflectionEngine:
    """
    Self-analysis for continuous improvement.
    Runs periodically to analyze recent experiences and generate insights.
    """
    def __init__(self, learning_engine: LearningEngine, interval_minutes: int = 5, planning_engine=None):
        self.learning_engine = learning_engine
        self.planning_engine = planning_engine
        self.interval_seconds = interval_minutes * 60
        self.running = False
        self._task = None
        self.insights: List[str] = []

    def start(self):
        if not self.running:
            self.running = True
            self._task = asyncio.create_task(self._reflection_loop())
            logger.info("Reflection Engine started.")

    def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
            logger.info("Reflection Engine stopped.")

    async def _reflection_loop(self):
        while self.running:
            await asyncio.sleep(self.interval_seconds)
            self._reflect()

    def _reflect(self):
        """Analyze recent experiences and generate insights."""
        logger.info("Starting periodic self-reflection...")
        
        # Get experiences from the last interval
        recent_experiences = self.learning_engine.get_recent_experiences(limit=20)
        now = time.time()
        relevant_experiences = [e for e in recent_experiences if (now - e.timestamp) <= self.interval_seconds]
        
        if not relevant_experiences:
            logger.info("No recent experiences to reflect upon.")
            return
            
        # 1. What failed? What succeeded?
        total_goals = len(relevant_experiences)
        failed_goals = sum(1 for e in relevant_experiences if e.success_score < 0.5)
        
        if failed_goals > 0:
            failure_rate = failed_goals / total_goals
            if failure_rate > 0.3:
                insight = f"High failure rate detected ({failure_rate*100:.1f}%). Need to adjust strategy or increase caution."
                self._add_insight(insight)
                
        # 2. Analyze specific step failures
        step_failures = {}
        for exp in relevant_experiences:
            for step, success in exp.outcomes.items():
                if not success:
                    step_failures[step] = step_failures.get(step, 0) + 1
                    
        for step, count in step_failures.items():
            if count >= 2:
                insight = f"Step '{step}' fails often. Consider adding delay or fallback mechanism."
                self._add_insight(insight)
                
        # 3. Analyze proactive behavior (mocked logic)
        # If user ignores suggestions, reduce proactive threshold
        # This would require tracking suggestion vs acceptance in Experience
        
        logger.info(f"Reflection complete. Generated {len(self.insights)} total insights.")

    def _add_insight(self, insight: str):
        if insight not in self.insights:
            self.insights.append(insight)
            logger.info(f"New Insight Generated: {insight}")
        if "Step '" in insight and "fails often" in insight:
            try:
                step = insight.split("'")[1]
                if self.learning_engine:
                    skills = self.learning_engine.state.setdefault("skills", {})
                    cur = skills.get(step, {}).get("weight")
                    if cur is not None:
                        skills.setdefault(step, {"success": 0, "failure": 0, "weight": 0.5})
                        skills[step]["weight"] = max(0.01, float(cur) - 0.05)
                        self.learning_engine._save_state()
                if self.planning_engine:
                    self.planning_engine.apply_adjustment(step, {"add_delay_s": 3, "fallback": None})
            except Exception:
                logger.exception("Failed to apply actionable insight")
