import time
import datetime
from dataclasses import dataclass
from typing import Dict, Any, List, Callable

@dataclass
class UserState:
    focus: str  # 'high', 'medium', 'low'
    focus_score: float # 0.0 to 1.0
    stress: str  # 'high', 'medium', 'low'
    stress_score: float # 0.0 to 1.0
    mode: str    # 'development', 'casual', 'meeting'
    last_active: float
    time_of_day: str
    day_of_week: str

class BehaviorModel:
    def __init__(self, transition_callback: Callable[[str, Dict[str, Any]], None] = None):
        self.state = UserState(
            focus="medium", focus_score=0.5, 
            stress="low", stress_score=0.2, 
            mode="casual", last_active=time.time(),
            time_of_day=self._get_time_of_day(),
            day_of_week=self._get_day_of_week()
        )
        self.app_usage_history: List[Dict[str, Any]] = []
        self.command_history: List[Dict[str, Any]] = []
        self.typing_speed_history: List[Dict[str, Any]] = []
        self.idle_intervals: List[Dict[str, Any]] = []
        self.transition_callback = transition_callback

    def _get_time_of_day(self) -> str:
        hour = datetime.datetime.now().hour
        if 5 <= hour < 12: return "morning"
        if 12 <= hour < 17: return "afternoon"
        if 17 <= hour < 22: return "evening"
        return "night"

    def _get_day_of_week(self) -> str:
        day = datetime.datetime.now().weekday()
        return "weekday" if day < 5 else "weekend"

    def track_app_usage(self, app_name: str, duration: int):
        """Track app usage to infer user state."""
        self.app_usage_history.append({
            "app": app_name,
            "duration": duration,
            "timestamp": time.time()
        })
        self._fuse_signals()

    def track_command_frequency(self, command: str):
        """Track command frequency to infer user state."""
        self.command_history.append({
            "command": command,
            "timestamp": time.time()
        })
        self._fuse_signals()
        
    def track_typing_speed(self, wpm: float):
        """Track typing speed (words per minute)."""
        self.typing_speed_history.append({
            "wpm": wpm,
            "timestamp": time.time()
        })
        self._fuse_signals()
        
    def track_idle_time(self, duration: float):
        """Track idle intervals."""
        self.idle_intervals.append({
            "duration": duration,
            "timestamp": time.time()
        })
        self._fuse_signals()

    def track_session_pattern(self, pattern: Dict[str, Any]):
        """Track session patterns to infer user state."""
        # E.g., late night coding -> high focus, potential stress
        pass

    def _trigger_transition(self, event_type: str, details: Dict[str, Any]):
        if self.transition_callback:
            self.transition_callback(event_type, details)

    def _fuse_signals(self):
        """Multi-signal fusion to determine user state and detect transitions."""
        now = time.time()
        
        # Update temporal context
        self.state.time_of_day = self._get_time_of_day()
        self.state.day_of_week = self._get_day_of_week()
        
        # 1. App Usage Analysis
        dev_apps = ["vscode", "cursor", "terminal", "pycharm", "docker"]
        meeting_apps = ["zoom", "teams", "slack", "meet"]
        recent_apps = [entry["app"].lower() for entry in self.app_usage_history[-5:] if now - entry["timestamp"] < 300]
        
        mode_score = 0.5
        if any(app in recent_apps for app in dev_apps):
            self.state.mode = "development"
            mode_score = 0.9
        elif any(app in recent_apps for app in meeting_apps):
            self.state.mode = "meeting"
            mode_score = 0.8
        else:
            self.state.mode = "casual"
            mode_score = 0.4
            
        # 2. Command Frequency Analysis
        recent_commands = [c for c in self.command_history if now - c["timestamp"] < 60]
        cmd_stress_factor = min(1.0, len(recent_commands) / 15.0)
        
        # 3. Typing Speed Analysis
        recent_typing = [t["wpm"] for t in self.typing_speed_history if now - t["timestamp"] < 120]
        avg_wpm = sum(recent_typing) / len(recent_typing) if recent_typing else 40.0
        typing_focus_factor = min(1.0, avg_wpm / 80.0)
        typing_stress_factor = 0.0
        if avg_wpm > 100: # Very fast typing might indicate stress/rush
            typing_stress_factor = 0.3
            
        # 4. Idle Time Analysis
        recent_idle = [i["duration"] for i in self.idle_intervals if now - i["timestamp"] < 300]
        total_idle = sum(recent_idle)
        idle_focus_penalty = min(0.5, total_idle / 300.0)
        
        # Calculate new scores
        new_focus_score = (mode_score * 0.5) + (typing_focus_factor * 0.5) - idle_focus_penalty
        new_focus_score = max(0.0, min(1.0, new_focus_score))
        
        new_stress_score = (cmd_stress_factor * 0.7) + (typing_stress_factor * 0.3)
        new_stress_score = max(0.0, min(1.0, new_stress_score))
        
        # Detect Transitions
        if self.state.focus_score - new_focus_score > 0.3:
            self._trigger_transition("focus_drop", {"old": self.state.focus_score, "new": new_focus_score})
        elif new_focus_score - self.state.focus_score > 0.3:
            self._trigger_transition("focus_spike", {"old": self.state.focus_score, "new": new_focus_score})
            
        if new_stress_score - self.state.stress_score > 0.4:
            self._trigger_transition("stress_spike", {"old": self.state.stress_score, "new": new_stress_score})
            
        # Update State
        self.state.focus_score = new_focus_score
        self.state.stress_score = new_stress_score
        
        if new_focus_score > 0.7: self.state.focus = "high"
        elif new_focus_score > 0.4: self.state.focus = "medium"
        else: self.state.focus = "low"
        
        if new_stress_score > 0.7: self.state.stress = "high"
        elif new_stress_score > 0.4: self.state.stress = "medium"
        else: self.state.stress = "low"
        
        self.state.last_active = now

    def get_current_state(self) -> UserState:
        """Return the current inferred user state."""
        # Decay focus if inactive for a long time
        if time.time() - self.state.last_active > 300: # 5 minutes
            self.state.focus = "low"
            self.state.focus_score = max(0.0, self.state.focus_score - 0.2)
            self.state.stress = "low"
            self.state.stress_score = max(0.0, self.state.stress_score - 0.2)
            
        return self.state
