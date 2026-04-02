"""
ATOM -- Core Services Container

Houses the main components of the ATOM engine so they can be
passed between factory, wiring, and lifecycle modules without
massive argument lists.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass
class AtomServices:
    bus: Any = None
    state: Any = None
    cache: Any = None
    memory: Any = None
    intent_engine: Any = None
    router: Any = None
    context_engine: Any = None
    
    # OS / Voice
    stt: Any = None
    tts: Any = None
    mic_manager: Any = None
    
    # Intelligence / Cognitive
    local_brain: Any = None
    brain_mode_mgr: Any = None
    assistant_mode_mgr: Any = None
    real_world_intel: Any = None
    proactive_intel: Any = None
    owner_understanding: Any = None
    jarvis_core: Any = None
    system_scanner: Any = None
    system_indexer: Any = None
    media_watcher: Any = None
    
    # Handlers & Governance
    process_mgr: Any = None
    scheduler: Any = None
    behavior: Any = None
    evolution: Any = None
    autonomy: Any = None
    health_monitor: Any = None
    system_watcher: Any = None
    gpu_governor: Any = None
    wake_word_engine: Any = None
    security_fortress: Any = None
    self_healing: Any = None
    code_introspector: Any = None
    context_fusion: Any = None
    
    # Dashboard / UI
    indicator: Any = None
    web_dashboard: Any = None
    
    # Cognitive layer
    cognitive_enabled: bool = False
    goal_engine: Any = None
    prediction_engine: Any = None
    behavior_model: Any = None
    self_optimizer: Any = None
    second_brain: Any = None
    personality_modes: Any = None
    dream_engine: Any = None
    curiosity_engine: Any = None
    
    # Workflows / Docs
    document_engine: Any = None
    workflow_engine: Any = None
    screen_reader: Any = None
    emotion_detector: Any = None
    
    # System 
    metrics: Any = None
    llm_queue: Any = None
    priority_sched: Any = None
    runtime_watchdog: Any = None
    executor: Any = None
    
    # Shutdown / State
    shutdown_event: Any = None
