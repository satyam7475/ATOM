import time
import threading
from typing import Dict, Any, Optional, Callable, List, Tuple
from collections import defaultdict
from .behavior_model import BehaviorModel, UserState

class ProactiveEngine:
    def __init__(self, behavior_model: BehaviorModel, suggestion_callback: Callable[[str], None] = None):
        self.behavior_model = behavior_model
        self.suggestion_callback = suggestion_callback
        self.running = False
        self._thread = None
        
        # Markov chain transition matrix: state_key -> action -> count
        self.transition_matrix: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.action_history: List[str] = []
        
        # Base risk scores
        self.risk_scores: Dict[str, float] = {
            "open_vscode": 0.1,
            "start_backend": 0.3,
            "mute_notifications": 0.2,
            "kill_process": 0.9,
            "restart_system": 1.0,
            "clear_cache": 0.6
        }
        
        # Initialize some default probabilities for demonstration
        self._init_default_probabilities()

    def _init_default_probabilities(self):
        # Seed the matrix with some initial behaviors
        dev_morning_key = "development_morning"
        self.transition_matrix[dev_morning_key]["start_backend"] = 5.0
        self.transition_matrix[dev_morning_key]["open_vscode"] = 2.0
        
        stress_high_key = "high_high" # stress_focus
        self.transition_matrix[stress_high_key]["mute_notifications"] = 4.0

    def start(self):
        if not self.running:
            self.running = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join()

    def _run_loop(self):
        while self.running:
            self.tick()
            time.sleep(10)  # Loop every 10 seconds

    def tick(self):
        state = self._get_system_state()
        action, confidence = self._predict_next(state)
        
        if action:
            risk = self.risk_score(action)
            utility = self._calculate_utility(action)
            context_alignment = self._calculate_context_alignment(action, state)
            uncertainty = 1.0 - confidence
            
            # Utility-Based Decision Formula
            # score = (confidence * utility * context_alignment) - (risk * uncertainty)
            score = (confidence * utility * context_alignment) - (risk * uncertainty)
            
            # Dynamic threshold based on system learning state (if available)
            base_threshold = 0.5
            
            if score > base_threshold:
                self._suggest_or_act(action, score, risk)

    def _calculate_utility(self, action: str) -> float:
        # Placeholder for utility calculation (0.0 to 1.0)
        # In a real system, this would be based on historical benefit
        utilities = {
            "start_backend": 0.9,
            "mute_notifications": 0.8,
            "open_vscode": 0.7,
            "kill_process": 0.6
        }
        return utilities.get(action, 0.5)
        
    def _calculate_context_alignment(self, action: str, state: Dict[str, Any]) -> float:
        # Placeholder for context alignment (0.0 to 1.0)
        if action == "mute_notifications" and state["focus"] == "high":
            return 1.0
        if action == "start_backend" and state["mode"] == "development":
            return 1.0
        return 0.5

    def _get_system_state(self) -> Dict[str, Any]:
        user_state = self.behavior_model.get_current_state()
        return {
            "mode": user_state.mode,
            "focus": user_state.focus,
            "stress": user_state.stress,
            "time_of_day": self._get_time_of_day()
        }
        
    def _get_time_of_day(self) -> str:
        hour = time.localtime().tm_hour
        if 5 <= hour < 12: return "morning"
        if 12 <= hour < 17: return "afternoon"
        if 17 <= hour < 22: return "evening"
        return "night"
        
    def _get_state_key(self, state: Dict[str, Any]) -> str:
        """Create a unique key for the current state."""
        # We can use different combinations depending on context
        if state["stress"] == "high":
            return f"{state['stress']}_{state['focus']}"
        return f"{state['mode']}_{state['time_of_day']}"

    def _predict_next(self, state: Dict[str, Any]) -> Tuple[str, float]:
        """Probabilistic prediction using Markov chain."""
        state_key = self._get_state_key(state)
        transitions = self.transition_matrix.get(state_key, {})
        
        if not transitions:
            return "", 0.0
            
        total_weight = sum(transitions.values())
        if total_weight == 0:
            return "", 0.0
            
        # Find the most probable next action
        best_action = max(transitions.items(), key=lambda x: x[1])
        action = best_action[0]
        probability = best_action[1] / total_weight
        
        return action, probability
        
    def risk_score(self, action: str) -> float:
        """Evaluate the risk of an action (0.0 to 1.0)."""
        return self.risk_scores.get(action, 0.5) # Default medium risk

    def learn_from_feedback(self, state: Dict[str, Any], action: str, accepted: bool):
        """Learning loop: adjust probabilities based on user feedback."""
        state_key = self._get_state_key(state)
        
        if accepted:
            # Increase weight for this transition
            self.transition_matrix[state_key][action] += 1.0
            self.action_history.append(action)
        else:
            # Decrease weight (decay)
            self.transition_matrix[state_key][action] = max(0.0, self.transition_matrix[state_key][action] - 0.5)

    def _suggest_or_act(self, action: str, confidence: float, risk: float):
        """Decide whether to suggest or act automatically based on risk."""
        # Map action to human-readable text
        messages = {
            "start_backend": "You usually start the backend now. Should I run it?",
            "mute_notifications": "You seem to be working intensely. Should I mute notifications?",
            "open_vscode": "Want me to open VSCode?",
            "kill_process": "A process is hanging. Should I kill it?"
        }
        
        suggestion = messages.get(action, f"Should I execute {action}?")
        
        # In a real system, low risk + high confidence = auto execute
        if risk < 0.2 and confidence > 0.9:
            print(f"[Proactive Engine Auto-Action]: Executing {action}")
            # Simulate auto-accept for learning
            self.learn_from_feedback(self._get_system_state(), action, True)
        else:
            if self.suggestion_callback:
                self.suggestion_callback(suggestion)
            else:
                print(f"[Proactive Engine Suggestion]: {suggestion} (Risk: {risk:.1f}, Conf: {confidence:.2f})")
