"""
Brain package: **MLX + memory graph are canonical**; several modules duplicate
``core/`` and are legacy — see ``docs/ARCHITECTURE_CANONICAL.md``.
"""
from .intent_engine import IntentEngine, Intent
from .context_router import ContextRouter
from .memory_graph import MemoryGraph, MemoryNode
from .behavior_model import BehaviorModel, UserState
from .proactive_engine import ProactiveEngine
from .skill_engine import SkillEngine, Skill

__all__ = [
    'IntentEngine', 'Intent',
    'ContextRouter',
    'MemoryGraph', 'MemoryNode',
    'BehaviorModel', 'UserState',
    'ProactiveEngine',
    'SkillEngine', 'Skill',
]
