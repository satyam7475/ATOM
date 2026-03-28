import logging
import json
import os
import time
from typing import Dict, Any, List
from dataclasses import dataclass, field

logger = logging.getLogger("atom.brain.learning_engine")

@dataclass
class Experience:
    goal_id: str
    objective: str
    plan_steps: List[str]
    outcomes: Dict[str, bool]  # step_action -> success
    success_score: float       # 0.0 to 1.0
    timestamp: float = field(default_factory=time.time)

class LearningEngine:
    """
    Make system smarter over time.
    Experience -> Feedback -> Update Models
    """
    def __init__(self, storage_path: str = "atom_learning_state.json"):
        self.storage_path = storage_path
        self.state = self._load_state()
        self.experiences: List[Experience] = []
        
        # Learning rate (bounded updates)
        self.alpha = 0.1
        self.smoothing_beta = 0.3  # new = (1-beta)*old + beta*raw
        self.weight_min = 0.1
        self.weight_max = 0.9
        self._last_skill_delta: Dict[str, float] = {}

    def _load_state(self) -> Dict[str, Any]:
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load learning state: {e}")
        
        return {
            "skills": {},          # skill_name -> {"success": int, "failure": int, "weight": float}
            "preferences": {},     # pref_key -> weight
            "thresholds": {        # decision thresholds
                "autonomy": 0.85,
                "proactive": 0.70
            },
            # Learnable plan-evaluator weights (sum should stay ~1.0; we renormalize after updates)
            "plan_score_weights": {
                "w_success": 0.4,
                "w_efficiency": 0.2,
                "w_context": 0.2,
                "w_risk": 0.2,
            },
        }

    def _save_state(self):
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.state, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save learning state: {e}")

    def record_experience(self, experience: Experience):
        """Record an experience and update models."""
        self.experiences.append(experience)
        logger.info(f"Recorded Experience for Goal [{experience.goal_id}] with score {experience.success_score:.2f}")
        
        # 1. Update Skill Success Rates
        for action, success in experience.outcomes.items():
            # Assuming action name maps directly to skill for simplicity here
            skill = action 
            if skill not in self.state["skills"]:
                self.state["skills"][skill] = {"success": 0, "failure": 0, "weight": 0.5}
                
            stats = self.state["skills"][skill]
            if success:
                stats["success"] += 1
            else:
                stats["failure"] += 1

            reward = 1.0 if success else 0.0
            prediction = float(stats["weight"])
            raw_delta = self.alpha * (reward - prediction)
            # Anti-oscillation: dampen if direction flips vs last update
            if skill in self._last_skill_delta:
                if self._last_skill_delta[skill] * raw_delta < 0:
                    raw_delta *= 0.5
            self._last_skill_delta[skill] = raw_delta
            new_raw = prediction + raw_delta
            smoothed = (1.0 - self.smoothing_beta) * prediction + self.smoothing_beta * new_raw
            stats["weight"] = max(self.weight_min, min(self.weight_max, smoothed))
            
        # 2. Update Decision Thresholds
        # If overall experience is a failure, increase autonomy threshold (be more cautious)
        # If success, slightly decrease (be more confident)
        current_autonomy = float(self.state["thresholds"]["autonomy"])
        if experience.success_score < 0.5:
            self.state["thresholds"]["autonomy"] = min(0.95, current_autonomy + 0.02)
        else:
            self.state["thresholds"]["autonomy"] = max(0.70, current_autonomy - 0.005)
        self.state["thresholds"]["autonomy"] = max(0.65, min(0.98, float(self.state["thresholds"]["autonomy"])))
            
        self._save_state()
        logger.debug(f"Learning Models Updated. New Autonomy Threshold: {self.state['thresholds']['autonomy']:.3f}")

    def get_skill_weight(self, skill_name: str) -> float:
        return self.state.get("skills", {}).get(skill_name, {}).get("weight", 0.5)
        
    def get_threshold(self, key: str) -> float:
        return self.state.get("thresholds", {}).get(key, 0.8)
        
    def get_recent_experiences(self, limit: int = 10) -> List[Experience]:
        return self.experiences[-limit:]

    def get_plan_score_weights(self) -> Dict[str, float]:
        defaults = {
            "w_success": 0.4,
            "w_efficiency": 0.2,
            "w_context": 0.2,
            "w_risk": 0.2,
        }
        stored = self.state.get("plan_score_weights") or {}
        return {**defaults, **stored}

    def _renormalize_plan_weights(self, w: Dict[str, float]) -> Dict[str, float]:
        keys = ("w_success", "w_efficiency", "w_context", "w_risk")
        for k in keys:
            w[k] = max(0.05, min(0.7, float(w[k])))
        total = sum(max(0.01, w[k]) for k in keys)
        return {k: max(0.01, w[k]) / total for k in keys}

    def update_plan_score_weights_from_outcome(self, sim: Dict[str, Any], success: bool) -> None:
        """
        Calibrate scoring weights from execution feedback.
        On success: increase emphasis on dimensions that were strong in simulation.
        On failure: penalize success/context weights and increase risk sensitivity.
        """
        w = self.get_plan_score_weights()
        lr = 0.02

        sp = float(sim.get("success_probability", 0.5))
        eff = float(sim.get("efficiency", 0.5))
        ctx = float(sim.get("context_alignment", 0.5))
        risk = float(sim.get("risk", 0.5))

        if success:
            w["w_success"] += lr * sp
            w["w_efficiency"] += lr * eff
            w["w_context"] += lr * ctx
            w["w_risk"] += lr * (1.0 - risk) * 0.5
        else:
            w["w_success"] -= lr * 0.5
            w["w_context"] -= lr * 0.25
            w["w_efficiency"] -= lr * 0.15
            w["w_risk"] += lr * 0.5

        self.state["plan_score_weights"] = self._renormalize_plan_weights(w)
        self._save_state()
        logger.debug("Plan score weights updated: %s", self.state["plan_score_weights"])
