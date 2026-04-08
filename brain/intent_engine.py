from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

@dataclass
class Intent:
    type: str  # 'system', 'chat', 'task', 'automation'
    confidence: float
    entities: Dict[str, Any]
    urgency: str  # 'low', 'medium', 'high'

class IntentEngine:
    def __init__(self):
        # Define keyword and regex patterns for different intents
        self.patterns = {
            "system": [
                r"\b(stop|pause|resume|shutdown|restart|status|volume|mute|unmute)\b",
                r"\b(turn off|turn on|sleep mode|wake up)\b"
            ],
            "task": [
                r"\b(create|build|run|execute|start|open|close|compile|test|deploy)\b",
                r"\b(remind me|schedule|add to my list|todo)\b"
            ],
            "automation": [
                r"\b(macro|workflow|routine|trigger|when i|every day at)\b",
                r"\b(automate|script|batch)\b"
            ]
        }
        
        # Urgency keywords
        self.urgency_keywords = {
            "high": [r"\b(urgent|now|immediately|asap|critical|emergency)\b"],
            "medium": [r"\b(soon|today|important)\b"]
        }

    def classify(self, text: str) -> Intent:
        text_lower = text.lower()
        
        # 1. Determine Intent Type and Confidence
        intent_scores = {"system": 0.0, "chat": 0.1, "task": 0.0, "automation": 0.0}
        
        for intent_type, regex_list in self.patterns.items():
            for pattern in regex_list:
                matches = re.findall(pattern, text_lower)
                if matches:
                    # Add score based on number of matches and pattern weight
                    intent_scores[intent_type] += len(matches) * 0.4

        # Normalize and find the highest scoring intent
        best_intent = "chat"
        best_score = intent_scores["chat"]
        
        for intent_type, score in intent_scores.items():
            if score > best_score:
                best_intent = intent_type
                best_score = score
                
        # Cap confidence at 1.0
        confidence = min(1.0, best_score)
        
        # 2. Extract Entities (Basic extraction for now)
        entities = self._extract_entities(text_lower)
        
        # 3. Determine Urgency
        urgency = "low"
        for level, regex_list in self.urgency_keywords.items():
            for pattern in regex_list:
                if re.search(pattern, text_lower):
                    urgency = level
                    break
            if urgency != "low":
                break

        return Intent(
            type=best_intent,
            confidence=confidence,
            entities=entities,
            urgency=urgency
        )
        
    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract basic entities like numbers, times, or specific targets."""
        entities = {}
        
        # Extract numbers
        numbers = re.findall(r'\b\d+\b', text)
        if numbers:
            entities["numbers"] = numbers
            
        # Extract potential targets (words after action verbs)
        # E.g., "open vscode" -> target: "vscode"
        target_match = re.search(r'\b(?:open|start|run|close)\s+([a-zA-Z0-9_]+)\b', text)
        if target_match:
            entities["target"] = target_match.group(1)
            
        return entities
