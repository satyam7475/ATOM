from typing import Dict, Any, List, Union
from .intent_engine import Intent

class ContextRouter:
    def __init__(self):
        # Configuration for context limits to prevent token overload
        self.max_chat_history = 5
        self.max_task_history = 10

    def build_context(self, intent: Union[Intent, Dict[str, Any]], current_state: Dict[str, Any], memory_graph: Any = None) -> Dict[str, Any]:
        """
        Dynamically builds a 4-layer context payload based on intent type.
        Layers: Immediate (now), Session (last 30m), Task (current goal), Memory (long-term).
        Accepts Intent or dict (ZMQ worker / IPC compatibility).
        """
        if isinstance(intent, dict):
            intent = Intent(
                type=intent.get("intent_type", "chat"),
                confidence=float(intent.get("confidence", 0.5)),
                entities=intent.get("entities") or {},
                urgency=str(intent.get("urgency", "medium")),
            )
        # Layer 1: Immediate Context (always present)
        context = {
            "immediate": {
                "intent": intent.type,
                "confidence": intent.confidence,
                "urgency": intent.urgency,
                "entities": intent.entities,
                "system_status": current_state.get("system_status", "ok"),
                "current_time": current_state.get("current_time", "")
            },
            "session": {},
            "task": {},
            "memory": {}
        }
        
        # Route context based on intent type
        if intent.type == "system":
            pass # Minimal context is already in immediate
        elif intent.type == "task":
            context["task"] = self._get_task_context(current_state)
            context["memory"] = self._get_memory_context(memory_graph, intent.entities)
        elif intent.type == "chat":
            context["session"] = self._get_session_context(current_state)
            context["memory"] = self._get_memory_context(memory_graph, intent.entities)
        elif intent.type == "automation":
            context["task"] = self._get_automation_context(current_state)
            
        return context
        
    def _get_session_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Session context (last 30 min)."""
        return {
            "recent_chat": state.get("recent_chat", [])[-self.max_chat_history:],
            "user_emotion": state.get("user_emotion", "neutral"),
            "user_focus": state.get("user_focus", "medium")
        }
        
    def _get_task_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Task Context (current goal)."""
        return {
            "active_tasks": state.get("active_tasks", [])[:self.max_task_history],
            "current_workspace": state.get("current_workspace", "")
        }
        
    def _get_memory_context(self, memory_graph: Any, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Memory Context (long-term)."""
        if not memory_graph or "target" not in entities:
            return {}
            
        target = entities["target"]
        # Query memory graph for semantic/procedural memory related to target
        nodes = memory_graph.query({"target": target})
        return {
            "relevant_knowledge": [node.data for node in nodes] if nodes else []
        }
        
    def _get_automation_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Automation intents need workflow definitions and triggers."""
        return {
            "active_workflows": state.get("active_workflows", []),
            "recent_triggers": state.get("recent_triggers", [])
        }
