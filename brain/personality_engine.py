"""
ATOM V4 -- Personality Engine.

Generates dynamic, context-aware responses to avoid static, robotic replies.
tone = f(emotion, context, relationship, history)

Owner: Satyam
"""

import random
from typing import Dict, Any

class PersonalityEngine:
    def __init__(self):
        self.relationship_level = "professional" # casual, professional, intimate
        
    def generate_response(self, action: str, context: Dict[str, Any]) -> str:
        """Generate a dynamic response based on context and action."""
        
        # Extract context variables
        immediate = context.get("immediate", {})
        session = context.get("session", {})
        
        urgency = immediate.get("urgency", "low")
        emotion = session.get("user_emotion", "neutral")
        focus = session.get("user_focus", "medium")
        
        if action == "development_start":
            return self._dev_start_response(urgency, emotion, focus)
        elif action == "system_shutdown":
            return self._shutdown_response(urgency, emotion)
        else:
            return self._generic_ack(urgency)
            
    def _dev_start_response(self, urgency: str, emotion: str, focus: str) -> str:
        if urgency == "high":
            return "Booting dev environment immediately. Let's go."
        if focus == "high":
            return "Workspace ready. I'll keep interruptions to a minimum."
        
        responses = [
            "All set. Backend is up and VSCode is ready.",
            "Development environment is online. What are we building today?",
            "Workspace initialized. Ready when you are, Boss."
        ]
        return random.choice(responses)
        
    def _shutdown_response(self, urgency: str, emotion: str) -> str:
        if urgency == "high":
            return "Emergency shutdown initiated."
        if emotion == "stressed":
            return "Shutting down. Get some rest, you've earned it."
            
        responses = [
            "Saving state and shutting down. See you later.",
            "Systems powering down. Good work today.",
            "Going offline. Have a good one, Boss."
        ]
        return random.choice(responses)
        
    def _generic_ack(self, urgency: str) -> str:
        if urgency == "high":
            return "Done."
        return random.choice(["Got it.", "Consider it done.", "Handled."])
