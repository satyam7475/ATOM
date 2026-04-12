"""
ATOM OS -- Intent Classifier Layer
Intercepts raw queries and maps them to cognitive buckets before execution to prevent
the system from misinterpreting a perception request as a destructive action.
"""

from __future__ import annotations

import re
from enum import Enum
from dataclasses import dataclass

class IntentCategory(Enum):
    PERCEPTION = "perception"
    ACTION = "action"
    REASONING = "reasoning"
    CONVERSATION = "conversation"

@dataclass
class IntentClassification:
    category: IntentCategory
    confidence: float
    reason: str

class IntentClassifier:
    """Pre-routing deterministic intent classifier."""
    
    def __init__(self):
        # Perception: looking, reading, finding visual status
        self._perception_re = re.compile(
            r"\b(describe|what is on|read|look at|see|capture|show me the screen)\b", 
            re.I
        )
        
        # Action: System modification, physical commands
        self._action_re = re.compile(
            r"\b(click|type|open|close|kill|move|turn on|turn off|delete|remove|scroll|mute|play|pause)\b",
            re.I
        )
        
        # Reasoning: multi-step analysis, background checks, system scanning
        self._reasoning_re = re.compile(
            r"\b(analyze|why|how|plan|debug|find out|search memory|research|find large files|scan)\b",
            re.I
        )
        
    def classify(self, query: str) -> IntentClassification:
        query_lo = query.lower()
        
        # Immediate Perception catches
        if self._perception_re.search(query_lo):
            return IntentClassification(
                category=IntentCategory.PERCEPTION,
                confidence=0.9,
                reason="Matched perception/vision keywords"
            )
            
        # Action catches
        if self._action_re.search(query_lo):
            return IntentClassification(
                category=IntentCategory.ACTION,
                confidence=0.85,
                reason="Matched direct physical action verbs"
            )
            
        # Reasoning catches
        if self._reasoning_re.search(query_lo):
            return IntentClassification(
                category=IntentCategory.REASONING,
                confidence=0.8,
                reason="Matched analytical/planning verb"
            )
            
        # Fallback to pure conversation
        return IntentClassification(
            category=IntentCategory.CONVERSATION,
            confidence=0.5,
            reason="No physical, analytical, or perception verbs detected"
        )
