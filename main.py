"""
ATOM -- Personal Cognitive AI Operating System (JARVIS-Level).

Supernatural Intelligence OS. Not a tool. Not an assistant.
A sentient-grade AI companion that knows its owner, knows the world,
anticipates needs, and acts autonomously.

Entry point. Wires all modules through the async event bus,
sets a fixed ThreadPoolExecutor, eagerly preloads STT and TTS, and
runs as an always-listening AI OS with GPU-accelerated local LLM.

Core Systems:
  - 9-layer LLM prompt architecture with fused world intelligence
  - ContextFusionEngine: unified owner/system/conversation state
  - RealWorldIntelligence: weather, news, location, temporal awareness
  - ProactiveIntelligenceEngine: workflow/behavioral/temporal triggers
  - AdaptivePersonality: context-aware, emotion-responsive expression
  - SecurityFortress: 7-gate security + VoicePrint + BehavioralAuth
  - SelfHealingEngine: failure tracking + auto-recovery
  - CodeIntrospector: self-aware codebase analysis
  - JarvisCore: proactive anticipation + contextual inference
  - Cognitive Layer: SecondBrain, GoalEngine, PredictionEngine,
    BehaviorModel, SelfOptimizer, DreamEngine, CuriosityEngine
  - Full reasoning: 40+ tools, ReAct loop, code sandbox, workflows

Pipeline: Voice Input -> Wake Word -> STT -> Intent Engine -> Router
          -> GPU LLM Brain (ReAct + Tool Use) -> True Token Streaming
          -> TTS -> Voice Output
          (with ContextFusion + RealWorldIntel enriching every query)
          (with JARVIS Core context injection at every stage)
          (with SecurityFortress gate on every action)
          (with VoicePrintAuth + BehavioralAuth on identity)
          (with ProactiveEngine anticipating needs)
          (with SelfHealingEngine capturing every failure)

Owner: Satyam
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import psutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core.boot.config_loader import load_config, set_config_overrides


logger = logging.getLogger("atom.main")
shutdown_event: asyncio.Event | None = None
_restart_requested = False


from core.boot.wiring import wire_events


async def main() -> None:
    global shutdown_event
    shutdown_event = asyncio.Event()
    import argparse
    parser = argparse.ArgumentParser(description="ATOM - Personal Cognitive AI OS")
    parser.add_argument("--v3", action="store_true", help="Run in V3 multi-process mode")
    parser.add_argument("--v4", action="store_true", help="Run in V4 Cognitive OS mode")
    args = parser.parse_args()
    distributed_mode = args.v3 or args.v4
    distributed_mode_label = "V4" if args.v4 else "V3" if args.v3 else "local"

    from core.logging_setup import setup_logging
    setup_logging()

    llm_queue = None
    runtime_watchdog = None
    priority_sched = None

    config = load_config()

    from core.config_schema import validate_and_log
    if not validate_and_log(config):
        logger.error("Invalid configuration — fix config/settings.json and restart.")
        sys.exit(1)

    from core.owner_gate import configure as _configure_owner_gate, owner_display_name
    _configure_owner_gate(config)
    try:
        from core.identity.session_manager import configure as _configure_sessions
        _configure_sessions(config)
    except Exception:
        pass
    logger.info(
        "ATOM owner binding: %s — access control via core/owner_gate.py",
        owner_display_name(),
    )

    from core.deployment_profile import (
        deployment_dashboard_badge,
        log_deployment_bootstrap,
    )
    log_deployment_bootstrap(config)

    from core.adaptive_personality import set_owner as _set_owner
    owner_cfg = config.get("owner", {})
    _set_owner(
        name=owner_cfg.get("name", "Satyam"),
        title=owner_cfg.get("title", "Boss"),
    )
    # NOTE: _set_adaptive_owner was removed — set_owner() above is the
    # correct and only API in adaptive_personality for setting owner info.

    executor = ThreadPoolExecutor(
        max_workers=config.get("executor", {}).get("max_workers", 3),
        thread_name_prefix="atom",
    )
    asyncio.get_running_loop().set_default_executor(executor)

    if distributed_mode:
        logger.info("Initializing distributed ZmqEventBus (%s)...", distributed_mode_label)
        from core.ipc.zmq_bus import ZmqEventBus
        bus = ZmqEventBus(worker_name="main_core")
    else:
        from core.async_event_bus import AsyncEventBus
        bus = AsyncEventBus()

    from core.state_manager import StateManager, AtomState
    from core.cache_engine import CacheEngine
    from core.memory_engine import MemoryEngine
    from core.intent_engine import IntentEngine
    from core.router import Router
    from context.context_engine import ContextEngine
    from voice.mic_manager import MicManager
    from core.metrics import MetricsCollector, log_health
    from core.pipeline_timer import PipelineTimer
    from core.health_monitor import HealthMonitor
    from core.command_registry import get_registry
    from core.system_watcher import SystemWatcher
    from core.behavior_tracker import BehaviorTracker
    from core.task_scheduler import TaskScheduler
    from core.process_manager import ProcessManager
    from core.self_evolution import SelfEvolutionEngine
    from core.autonomy_engine import AutonomyEngine
    from core.security_policy import SecurityPolicy
    from core.cognitive.second_brain import SecondBrain
    from core.cognitive.goal_engine import GoalEngine
    from core.cognitive.behavior_model import BehaviorModel
    from core.cognitive.prediction_engine import PredictionEngine
    from core.cognitive.self_optimizer import SelfOptimizer
    from core.personality_modes import PersonalityModes

    if distributed_mode:
        bus.start()
    state = StateManager(
        bus,
        error_recovery_hold_s=float(
            (config.get("performance", {}) or {}).get("error_recovery_hold_s", 0.35)
        ),
    )
    mic_manager = MicManager()
    metrics = MetricsCollector()

    command_registry = get_registry()
    logger.info("Command registry: %d commands loaded", command_registry.count)

    running_loop = asyncio.get_running_loop()

    cache = CacheEngine(
        max_size=config.get("cache", {}).get("max_size", 128),
        ttl=config.get("cache", {}).get("ttl_seconds", 300),
        metrics=metrics,
    )
    memory = MemoryEngine(config)
    intent_engine = IntentEngine()
    context_engine = ContextEngine(config)

    scheduler = TaskScheduler(bus)
    process_mgr = ProcessManager()
    evolution = SelfEvolutionEngine(metrics)
    behavior = BehaviorTracker(config)

    from core.brain_mode_manager import BrainModeManager
    from core.assistant_mode_manager import AssistantModeManager

    brain_mode_mgr = BrainModeManager(config)
    assistant_mode_mgr = AssistantModeManager(config)

    from core.skills_registry import SkillsRegistry
    from core.conversation_memory import ConversationMemory
    from core.memory.timeline_memory import TimelineMemory
    from core.runtime.modes import RuntimeModeResolver
    from core.cognition.feedback_engine import FeedbackEngine
    from core.cognition.suggester import SuggestionEngine
    from core.system.system_monitor import SystemMonitor

    v7i_cfg = config.get("v7_intelligence") or {}
    _tl_max = int(v7i_cfg.get("max_timeline_size") or v7i_cfg.get("timeline_max_events", 500))
    timeline_memory = TimelineMemory(
        max_events=_tl_max,
        summarize_on_prune=bool(v7i_cfg.get("timeline_summarize_on_prune", False)),
    )
    mode_resolver = RuntimeModeResolver(config)
    feedback_engine = FeedbackEngine(config)
    system_monitor = SystemMonitor(config)
    suggester_engine = SuggestionEngine(config)

    skills_reg = SkillsRegistry(config)
    conv_memory = ConversationMemory(config)

    security = SecurityPolicy(config)

    router = Router(
        bus, state, cache, memory,
        intent_engine=intent_engine, context_engine=context_engine,
        config=config, scheduler=scheduler,
        process_mgr=process_mgr, evolution=evolution,
        behavior_tracker=behavior,
        brain_mode_manager=brain_mode_mgr,
        assistant_mode_manager=assistant_mode_mgr,
        skills_registry=skills_reg,
        conversation_memory=conv_memory,
        timeline_memory=timeline_memory,
        security_policy=security,
    )
    brain_mode_mgr.attach_security(security)
    assistant_mode_mgr.attach_security(security)

    from core.fast_path import startup_warm_up
    startup_warm_up(intent_engine, cache, memory, config)

    if distributed_mode:
        logger.info("%s mode: initializing STT/TTS proxies...", distributed_mode_label)
        from core.ipc.proxies import TTSProxy, STTProxy
        stt = STTProxy(bus)
        tts = TTSProxy(bus, state)
        
        async def on_tts_done(event: str, **data) -> None:
            if state.current is AtomState.SPEAKING:
                await state.transition(AtomState.LISTENING)
        bus.on("tts_done", on_tts_done)
    else:
        from voice.stt_async import STTAsync
        stt = STTAsync(bus, state, config, mic_manager=mic_manager, intent_engine=intent_engine)

        tts_cfg = config.get("tts", {})
        tts_engine = (tts_cfg.get("engine") or "sapi").lower()

        # Auto-detect: on macOS, prefer native TTS unless explicitly set
        if tts_engine == "sapi" and sys.platform == "darwin":
            tts_engine = "macos_native"
            logger.info("macOS detected — auto-selecting native TTS")

        if tts_engine == "macos_native":
            from voice.tts_macos import MacOSTTSAsync
            tts = MacOSTTSAsync(
                bus, state,
                max_lines=tts_cfg.get("max_lines", 4),
                voice=tts_cfg.get("macos_voice", "Daniel"),
                rate=tts_cfg.get("macos_rate", 200),
            )
            logger.info("TTS: macOS Native (voice=%s, rate=%d — offline, ~5ms)",
                        tts_cfg.get("macos_voice", "Daniel"),
                        tts_cfg.get("macos_rate", 200))

        elif tts_engine == "kokoro":
            try:
                from voice.tts_kokoro import KokoroTTSAsync
                tts = KokoroTTSAsync(
                    bus, state,
                    max_lines=tts_cfg.get("max_lines", 4),
                    voice=tts_cfg.get("kokoro_voice", "af_heart")
                )
                logger.info("TTS: Kokoro Neural (offline, %s)", tts_cfg.get("kokoro_voice", "af_heart"))
            except ImportError:
                logger.warning("Kokoro TTS unavailable, falling back to Edge")
                tts_engine = "edge"
                
        if tts_engine == "edge":
            try:
                from voice.tts_edge import EdgeTTSAsync
                tts = EdgeTTSAsync(
                    bus, state,
                    max_lines=tts_cfg.get("max_lines", 4),
                    voice=tts_cfg.get("edge_voice", "en-GB-RyanNeural"),
                    rate=tts_cfg.get("edge_rate", "+15%"),
                    enable_postprocess=tts_cfg.get("edge_postprocess", True),
                    enable_ack_cache=tts_cfg.get("edge_ack_cache", True),
                )
                logger.info("TTS: Edge Neural (%s) — requires network", tts_cfg.get("edge_voice"))
            except ImportError:
                from voice.tts_async import TTSAsync
                tts = TTSAsync(
                    bus, state, max_lines=tts_cfg.get("max_lines", 4),
                    rate=tts_cfg.get("rate", 2),
                )
                logger.warning("Edge-TTS unavailable, using offline SAPI")
        elif tts_engine not in ("kokoro", "macos_native"):
            from voice.tts_async import TTSAsync
            tts = TTSAsync(
                bus, state, max_lines=tts_cfg.get("max_lines", 4),
                rate=tts_cfg.get("rate", 2),
            )
            logger.info("TTS: Windows SAPI (offline)")

    brain_enabled = config.get("brain", {}).get("enabled", False)

    from cursor_bridge.structured_prompt_builder import StructuredPromptBuilder
    prompt_builder = StructuredPromptBuilder(config)

    local_brain = None
    prefetch_eng = None
    shared_memory_graph = None
    if brain_enabled:
        if distributed_mode:
            logger.info("%s mode: initializing BrainProxy...", distributed_mode_label)
            from core.ipc.proxies import BrainProxy
            local_brain = BrainProxy(bus)
            
            async def on_llm_done(event: str, **data) -> None:
                logger.debug("LLM generation complete (V3).")
            bus.on("llm_done", on_llm_done)
        else:
            from cursor_bridge.local_brain_controller import LocalBrainController
            local_brain = LocalBrainController(
                bus, prompt_builder, config,
                brain_mode_manager=brain_mode_mgr,
            )
            local_brain.set_action_executor(router.action_executor)
            local_brain.attach_feedback_engine(feedback_engine)
            local_brain.attach_system_monitor(system_monitor)
            local_brain.attach_suggester(suggester_engine)
            local_brain.attach_timeline(timeline_memory)
            local_brain.attach_mode_resolver(mode_resolver)
            try:
                from brain.memory_graph import MemoryGraph
                from core.rag.prefetch_engine import RagPrefetchEngine
                from core.rag.rag_engine import RagEngine

                _mg_path = (config.get("memory") or {}).get(
                    "graph_db_path", "data/atom_memory.db",
                )
                shared_memory_graph = MemoryGraph(db_path=_mg_path, config=config)
                local_brain.attach_memory_graph(shared_memory_graph)
                _rag_cfg = config.get("rag") or {}
                if _rag_cfg.get("enabled", True):
                    rag_engine = RagEngine(config, vector_store=None)
                    rag_engine.set_memory_graph(shared_memory_graph)
                    rag_engine.set_feedback_engine(feedback_engine)
                    prefetch_eng = RagPrefetchEngine(rag_engine, config)
                    local_brain.attach_rag(rag_engine, None)
                    local_brain.attach_prefetch_engine(prefetch_eng)
                    logger.info(
                        "V7 intelligence: RAG + prefetch + MemoryGraph + timeline + "
                        "feedback + system awareness wired",
                    )
            except Exception as exc:
                logger.warning("V7 intelligence layer partial wiring: %s", exc)
            logger.info("Local brain ENABLED (agentic mode, tool-use, brain.enabled=true)")
    else:
        logger.info("Local brain DISABLED — enable brain.enabled for voice Q&A")

    from core.llm_inference_queue import LLMInferenceQueue
    from core.priority_scheduler import PriorityScheduler
    from core.runtime_watchdog import RuntimeWatchdog
    perf_cfg = config.get("performance", {})

    if brain_enabled and local_brain is not None:
        llm_queue = LLMInferenceQueue(bus, metrics)
        llm_queue.attach_brain(local_brain)
        logger.info("LLM inference queue enabled (single-slot, coalescing)")

    priority_sched = (
        PriorityScheduler(metrics=metrics)
        if perf_cfg.get("use_priority_scheduler", True)
        else None
    )
    if priority_sched is not None:
        logger.info("Priority scheduler ON (voice > LLM > background)")
    else:
        logger.info("Priority scheduler OFF (use_priority_scheduler=false)")

    inference_guard = None
    recovery_mgr = None
    gpu_stall_wd = None
    if (config.get("v7_gpu") or {}).get("enabled", True):
        from core.inference_guard import InferenceGuard
        from core.recovery_manager import RecoveryManager
        from core.gpu_watchdog import GPUStallWatchdog

        inference_guard = InferenceGuard(bus, config)
        recovery_mgr = RecoveryManager(bus, config)
        gpu_stall_wd = GPUStallWatchdog(bus, config)
        gpu_stall_wd.start()
        inference_guard.start_power_task()
        if brain_enabled and local_brain is not None:
            local_brain.attach_inference_guard(inference_guard)
        logger.info(
            "ATOM V7: InferenceGuard + RecoveryManager + GPUStallWatchdog (Apple Silicon)",
        )

    pipeline_timer = PipelineTimer(bus, metrics)
    pipeline_timer.register()
    perf_mode = perf_cfg.get("mode", "lite")
    logger.info("Performance mode: %s", perf_mode)

    _PERF_DEFAULTS = {
        "full":       {"health": 60,  "watcher": 10,  "maint": 120},
        "lite":       {"health": 120, "watcher": 30,  "maint": 180},
        "ultra_lite": {"health": 300, "watcher": 60,  "maint": 300},
    }
    perf_d = _PERF_DEFAULTS.get(perf_mode, _PERF_DEFAULTS["lite"])

    health_interval = perf_cfg.get("health_check_interval_s", perf_d["health"])
    watcher_interval = perf_cfg.get("system_watcher_interval_s", perf_d["watcher"])
    maint_interval = perf_cfg.get("maintenance_interval_s", perf_d["maint"])

    health_monitor = HealthMonitor(bus, state, stt=stt, tts=tts,
                                   check_interval=health_interval,
                                   config=config)

    system_watcher = SystemWatcher(bus, poll_interval=watcher_interval)

    autonomy = AutonomyEngine(
        bus, behavior, security, health_monitor, config,
        priority_sched=priority_sched,
    )

    from core.proactive_awareness import ProactiveAwareness
    proactive = ProactiveAwareness(config)

    # ── Reasoning Engine ───────────────────────────────────────────
    from core.reasoning.tool_registry import get_tool_registry
    from core.reasoning.planner import ReasoningPlanner
    from core.reasoning.code_sandbox import CodeSandbox
    from core.reasoning.workflow_engine import WorkflowEngine
    from core.document_ingestion import DocumentIngestionEngine

    tool_registry = get_tool_registry()
    tool_registry.apply_confirmation_policy(
        config=config,
        command_registry=command_registry,
    )
    reasoning_planner = ReasoningPlanner(config)
    reasoning_planner.set_timeline(timeline_memory)
    reasoning_planner.set_system_monitor(system_monitor)
    code_sandbox = CodeSandbox(config)
    workflow_engine = WorkflowEngine(config)
    document_engine = DocumentIngestionEngine(config)

    logger.info(
        "Reasoning engine initialized: %d tools, planner, sandbox, workflows, documents",
        tool_registry.count,
    )

    prompt_builder.set_tool_registry(tool_registry)
    # NOTE: prompt_builder.set_context_sources() is called later (after line ~1065)
    # when context_fusion and real_world_intel are actually instantiated.

    # ── ActionExecutor (bridges LLM tool calls -> Router dispatch) ──
    if brain_enabled and local_brain is not None and not distributed_mode:
        router.action_executor.set_registry(tool_registry)
        local_brain.set_action_executor(router.action_executor)
        logger.info("ActionExecutor connected: LLM -> security gate -> Router dispatch")

    # ── Perception Upgrade ─────────────────────────────────────────
    from voice.emotion_detector import EmotionDetector

    emotion_detector = EmotionDetector(config)

    wake_word_engine = None
    if config.get("wake_word", {}).get("enabled", False):
        from voice.wake_word import WakeWordEngine
        wake_word_engine = WakeWordEngine(bus, state, config)
        if wake_word_engine.preload():
            logger.info("Wake word engine loaded (Hey ATOM)")
        else:
            logger.info("Wake word not available (OpenWakeWord not installed)")
    else:
        logger.info("Wake word disabled in config (always-listen mode)")

    screen_reader = None
    if config.get("screen_reader", {}).get("enabled", True):
        from context.screen_reader import ScreenReader
        screen_reader = ScreenReader(config)
        logger.info(
            "Screen reader: %s (%s)",
            "OCR available" if screen_reader.is_available else "fallback mode",
            screen_reader.ocr_backend,
        )

    # ── Silicon Governor (Apple Silicon hardware monitoring) ──────
    silicon_governor = None
    if config.get("gpu", {}).get("enabled", True):
        from core.silicon_governor import SiliconGovernor
        silicon_governor = SiliconGovernor(bus, config)
        if silicon_governor.is_available:
            logger.info("Silicon Governor: monitoring active (%s)", silicon_governor.gpu_name)

    _memory_pressure_threshold = float(
        (config.get("memory") or {}).get("pressure_threshold_pct", 85.0),
    )
    _memory_pressure_relief = float(
        (config.get("memory") or {}).get(
            "pressure_relief_pct",
            max(0.0, _memory_pressure_threshold - 10.0),
        ),
    )
    _embedding_pressure_unloaded = False

    async def _on_silicon_stats_update(stats=None, **_kw) -> None:
        nonlocal _embedding_pressure_unloaded
        if not isinstance(stats, dict):
            return
        try:
            memory_pct = float(stats.get("memory_pct", 0.0) or 0.0)
        except Exception:
            return

        try:
            memory.apply_memory_pressure(memory_pct)
        except Exception:
            logger.debug("MemoryEngine pressure hook failed", exc_info=True)

        if local_brain is not None and hasattr(local_brain, "apply_memory_pressure"):
            try:
                local_brain.apply_memory_pressure(memory_pct)
            except Exception:
                logger.debug("Local brain pressure hook failed", exc_info=True)

        if inference_guard is None:
            return

        if memory_pct >= _memory_pressure_threshold:
            if not _embedding_pressure_unloaded:
                inference_guard.mark_loaded("embeddings", False)
                inference_guard.request_unload("embeddings", "memory_pressure")
                _embedding_pressure_unloaded = True
        elif memory_pct <= _memory_pressure_relief:
            _embedding_pressure_unloaded = False

    bus.on("silicon_stats_update", _on_silicon_stats_update)

    # ── Cognitive Kernel (central brain coordinator) ────────────────
    from core.cognitive_kernel import CognitiveKernel, ExecPath

    cognitive_kernel = CognitiveKernel(
        config=config,
        bus=bus,
        intent_engine=intent_engine,
        cache_engine=cache,
        metrics=metrics,
        inference_guard=inference_guard,
        silicon_governor=silicon_governor,
        state_manager=state,
    )
    router.attach_cognitive_kernel(cognitive_kernel)
    logger.info(
        "Cognitive Kernel: routing through %s paths",
        ", ".join(e.value for e in ExecPath),
    )

    # ── Security Fortress + Self-Healing + Code Introspection ──────
    from core.security_fortress import SecurityFortress
    from core.code_introspector import CodeIntrospector
    from core.self_healing import SelfHealingEngine

    security_fortress = SecurityFortress(config)
    code_introspector = CodeIntrospector()
    self_healing = SelfHealingEngine(config, introspector=code_introspector)

    self_healing.start()

    security.attach_fortress(security_fortress)

    code_introspector.scan()
    logger.info(
        "Production systems initialized: SecurityFortress(%s) + "
        "CodeIntrospector(%d files) + SelfHealingEngine",
        security_fortress.vault_backend_label,
        code_introspector.module_count,
    )

    # ── JARVIS-Level Intelligence ───────────────────────────────────
    from core.platform_adapter import get_platform_adapter
    from core.system_scanner import SystemScanner
    from core.system_indexer import system_indexer
    from core.owner_understanding import OwnerUnderstanding
    from core.system_control import SystemControl
    from voice.media_watcher import media_watcher

    platform_adapter = get_platform_adapter()
    system_scanner = SystemScanner(bus, config)
    owner_understanding = OwnerUnderstanding(bus, config)
    system_control = SystemControl(config)
    
    # Start background indexers
    system_indexer.start()
    media_watcher.start()

    # ── FSEvents File Watcher (macOS native, kernel-level) ──────
    fs_watcher = None
    if sys.platform == "darwin":
        try:
            from core.macos.fs_watcher import FSWatcher
            from core.macos.fs_watcher_config import fs_watcher_settings

            _fw = fs_watcher_settings(config)
            if _fw["enabled"]:
                fs_watcher = FSWatcher(bus)
                fs_watcher.watch(list(_fw["paths"]))
                if fs_watcher.start():
                    logger.info(
                        "FSWatcher: monitoring %s",
                        ", ".join(_fw["paths"]),
                    )
                else:
                    logger.debug("FSWatcher: could not start")
        except Exception:
            logger.debug("FSWatcher init failed", exc_info=True)

    logger.info(
        "JARVIS intelligence initialized: PlatformAdapter(%s) + "
        "SystemScanner + SystemIndexer + MediaWatcher + OwnerUnderstanding"
        + (" + FSWatcher" if fs_watcher else ""),
        platform_adapter.os_type.name,
    )

    # ── Cognitive Layer ───────────────────────────────────────────
    cognitive_enabled = config.get("cognitive", {}).get("enabled", True)
    second_brain = None
    goal_engine = None
    behavior_model = None
    prediction_engine = None
    self_optimizer = None
    personality_modes = None
    dream_engine = None
    curiosity_engine = None

    if cognitive_enabled:
        second_brain = SecondBrain(memory, behavior, config)
        goal_engine = GoalEngine(bus, second_brain, config)
        behavior_model = BehaviorModel(bus, config)
        prediction_engine = PredictionEngine(
            bus, behavior, memory, behavior_model, config,
        )
        prediction_engine.attach_prompt_builder(prompt_builder)
        prediction_engine.attach_cognitive_kernel(cognitive_kernel)
        if prefetch_eng is not None:
            prediction_engine.attach_prefetch_engine(prefetch_eng)
        self_optimizer = SelfOptimizer(bus, metrics, config)
        personality_modes = PersonalityModes(bus, behavior_model, config)

        from core.cognitive.dream_engine import DreamEngine
        from core.cognitive.curiosity_engine import CuriosityEngine

        dream_engine = DreamEngine(bus, config)
        curiosity_engine = CuriosityEngine(bus, config)
        logger.info("Cognitive layer initialized (8 modules, incl. dream + curiosity)")
    else:
        logger.info("Cognitive layer DISABLED via config")

    # ── JARVIS Core (intelligence fusion) ───────────────────────────
    from core.jarvis_core import JarvisCore

    jarvis_core = JarvisCore(
        bus,
        owner_understanding,
        system_scanner,
        personality_modes,
        config,
    )
    logger.info("JARVIS Core initialized (proactive anticipation + contextual inference)")

    # ── Context Fusion + Real World Intelligence + Proactive Engine ──
    from core.context_fusion import ContextFusionEngine
    from core.cognitive.proactive_engine import ProactiveIntelligenceEngine
    from core.real_world_intel import RealWorldIntelligence
    from core import adaptive_personality as _adaptive_personality

    context_fusion = ContextFusionEngine(bus=bus, config=config)
    context_fusion.wire(
        owner=owner_understanding,
        scanner=system_scanner,
        memory=memory,
        conv_memory=conv_memory,
        jarvis=jarvis_core,
    )

    real_world_intel = RealWorldIntelligence(config)

    proactive_intel = ProactiveIntelligenceEngine(bus=bus, config=config)
    proactive_intel.wire(
        behavior=behavior,
        conv_memory=conv_memory,
        owner=owner_understanding,
    )

    if cognitive_enabled:
        _adaptive_personality.attach_owner(owner_understanding)
        if personality_modes is not None:
            _adaptive_personality.attach_modes(personality_modes)

    # Wire context sources into prompt builder now that both objects exist.
    prompt_builder.set_context_sources(
        context_fusion=context_fusion,
        real_world_intel=real_world_intel,
    )

    logger.info(
        "Intelligence layer initialized: ContextFusion + RealWorldIntel + "
        "ProactiveEngine + AdaptivePersonality"
    )

    # ── UI ────────────────────────────────────────────────────────
    ui_cfg = config.get("ui", {})
    ui_mode = ui_cfg.get("mode", "web").lower()
    web_dashboard = None

    if ui_mode == "web":
        from ui.web_dashboard import WebDashboard
        indicator = WebDashboard(
            mic_name=stt.mic_name,
            port=ui_cfg.get("web_port", 8765),
            auto_open=ui_cfg.get("auto_open_browser", True),
            config=config,
        )
        owner_name = config.get("owner", {}).get("name", "Satyam")
        _tts_label = (
            "Kokoro (offline neural)"
            if tts_engine == "kokoro"
            else (
                f"macOS Native ({tts_cfg.get('macos_voice', 'Daniel')})"
                if tts_engine == "macos_native"
                else (
                    tts_cfg.get("edge_voice", "Edge")
                    if tts_engine == "edge"
                    else "SAPI (offline)"
                )
            )
        )
        _brain_label = "Local LLM"
        if brain_enabled and local_brain and local_brain.available:
            _brain_label = "Local: " + Path(
                config.get("brain", {}).get("model_path", "model")
            ).stem
        elif brain_enabled:
            _brain_label = "Local LLM (model not ready)"
        else:
            _brain_label = "No LLM (commands only)"
        _badge_label, _badge_show = deployment_dashboard_badge(config)
        indicator.set_init_info(
            version="ATOM",
            owner_name=owner_name,
            stt="Whisper (" + config.get("stt", {}).get("whisper_model_size", "small") + ")",
            tts=_tts_label,
            brain=_brain_label,
            perf_mode=perf_mode,
            brain_profile=brain_mode_mgr.active_profile,
            assistant_mode=assistant_mode_mgr.active,
            deployment_badge_label=_badge_label if _badge_show else "",
        )
        web_dashboard = indicator
    else:
        from ui.floating_indicator import FloatingIndicator
        indicator = FloatingIndicator(mic_name=stt.mic_name)

    if hasattr(indicator, "attach_runtime_managers"):
        indicator.attach_runtime_managers(
            brain_mode_mgr, assistant_mode_mgr, router._security,
        )

    def _ui_shutdown_callback():
        """Called from UI thread when X is clicked -- triggers full shutdown."""
        try:
            running_loop.call_soon_threadsafe(bus.emit, "shutdown_requested")
        except Exception:
            pass

    _MODE_LABELS = {
        "full": "full performance", "lite": "lite",
        "ultra_lite": "ultra lite", "auto": "auto",
    }
    _MODE_PHRASES = {
        "full": "All systems at maximum performance.",
        "lite": "Optimizing for efficiency.",
        "ultra_lite": "Entering low resource mode.",
        "auto": "Adapting to system load.",
    }

    async def _execute_mode_switch(new_mode: str) -> None:
        """Save config, speak confirmation, then trigger graceful restart."""
        global _restart_requested
        try:
            cfg_path = Path("config/settings.json")
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
            if "performance" not in cfg_data:
                cfg_data["performance"] = {}
            cfg_data["performance"]["mode"] = new_mode
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg_data, f, indent=4)
            logger.info("Performance mode updated to '%s' in settings.json", new_mode)

            label = _MODE_LABELS.get(new_mode, new_mode)
            phrase = _MODE_PHRASES.get(new_mode, "")
            msg = f"Switching to {label} mode, Boss. {phrase} Restarting now."
            bus.emit_long(
                "partial_response",
                text=msg,
                is_first=True, is_last=True,
            )
            await asyncio.sleep(3.0)

            _restart_requested = True
            shutdown_event.set()
        except Exception:
            logger.exception("Failed to update performance mode")

    def _on_mode_change(new_mode: str) -> None:
        """Called from the dashboard when user switches performance mode."""
        running_loop.call_soon_threadsafe(
            lambda: asyncio.create_task(_execute_mode_switch(new_mode))
        )

    indicator.set_shutdown_callback(_ui_shutdown_callback)
    if hasattr(indicator, "set_mode_change_callback"):
        indicator.set_mode_change_callback(_on_mode_change)

    if cognitive_enabled and personality_modes and hasattr(indicator, "set_personality_mode_callback"):
        def _on_personality_mode_from_ui(mode: str) -> None:
            running_loop.call_soon_threadsafe(
                lambda: asyncio.create_task(
                    _switch_personality_mode_async(mode)
                )
            )

        async def _switch_personality_mode_async(mode: str) -> None:
            result = personality_modes.switch_mode(mode)
            bus.emit_long("response_ready", text=result)

        indicator.set_personality_mode_callback(_on_personality_mode_from_ui)

    async def _on_bus_set_mode(mode: str = "lite", **_kw) -> None:
        await _execute_mode_switch(mode)
    bus.on("set_performance_mode", _on_bus_set_mode)

    from core.boot.cold_start import ColdStartOptimizer

    cold_start = ColdStartOptimizer(
        config=config,
        bus=bus,
        state_manager=state,
        local_brain=local_brain,
        memory_store=memory,
        conversation_memory=conv_memory,
        intent_engine=intent_engine,
        system_monitor=system_monitor,
    )

    await tts.init_voice()
    stt_preload_done = asyncio.Event()
    _bg_tasks: list[asyncio.Task] = []

    async def _background_stt_preload() -> None:
        t0 = time.monotonic()
        logger.info("STT model loading in background...")

        try:
            loop = asyncio.get_running_loop()
            if not distributed_mode:
                devices = await loop.run_in_executor(None, mic_manager.profile_devices)
                if devices:
                    best = mic_manager.get_best_device(
                        prefer_bluetooth=config.get("mic", {}).get("prefer_bluetooth", True),
                    )
                    if best:
                        mic_manager.active_device = best
                        logger.info(
                            "Audio device selected: '%s' (%s, quality=%d/100)",
                            best.name, best.device_type, best.quality_score,
                        )

            await stt.preload()
            elapsed = (time.monotonic() - t0) * 1000
            logger.info("STT pipeline ready (%.0fms: devices + model + preprocessor)", elapsed)
        except Exception:
            logger.exception("STT preload failed")
        finally:
            stt_preload_done.set()

    if config.get("stt", {}).get("preload", True):
        _bg_tasks.append(asyncio.create_task(_background_stt_preload()))
    else:
        stt_preload_done.set()

    cold_start_report = await cold_start.warm_up()
    logger.info(
        "Cold-start bootstrap: %.0fms (fast=%s embeddings=%s session=%d cache=%d)",
        cold_start_report.elapsed_ms,
        cold_start_report.fast_model_ready,
        cold_start_report.embeddings_ready,
        cold_start_report.restored_turns,
        cold_start_report.cached_commands,
    )

    _obs_v7 = (config.get("v7_intelligence") or {}).get("observability") or {}
    _snap_iv = float(_obs_v7.get("debug_snapshot_interval_s", 120.0))
    if _snap_iv > 0 and brain_enabled:
        async def _v7_periodic_snapshot() -> None:
            while True:
                await asyncio.sleep(_snap_iv)
                try:
                    from core.observability.debug_snapshot import (
                        get_debug_snapshot,
                        log_v7_debug_snapshot,
                    )
                    from core.cognition.preemption import get_last_preemption_score

                    ss2: dict = {}
                    if system_monitor is not None:
                        try:
                            ss2 = system_monitor.get_system_state()
                        except Exception:
                            pass
                    m = feedback_engine.compute_accuracy_metrics()
                    pre = get_last_preemption_score()
                    ap2 = None
                    try:
                        if local_brain is not None and getattr(
                            local_brain, "_memory_graph", None,
                        ):
                            ap2 = local_brain._memory_graph.get_last_active_project()
                    except Exception:
                        pass
                    tl_n2 = timeline_memory.event_count() if timeline_memory else 0
                    tl_p2 = timeline_memory.recent_preview(6) if timeline_memory else []
                    snap = get_debug_snapshot(
                        config,
                        runtime_mode=(
                            getattr(local_brain, "_current_runtime_mode", "SMART")
                            if local_brain
                            else "SMART"
                        ),
                        mode_info=(
                            getattr(local_brain, "_last_mode_info", {})
                            if local_brain
                            else {}
                        ),
                        system_state=ss2,
                        feedback_metrics=m,
                        last_retrieval_source=(
                            getattr(local_brain, "_last_retrieval_source", "")
                            if local_brain
                            else ""
                        ),
                        timeline_event_count=tl_n2,
                        timeline_recent_preview=tl_p2,
                        active_project=ap2,
                        preemption=pre,
                    )
                    log_v7_debug_snapshot(snap)
                except Exception:
                    logger.debug("v7 periodic snapshot failed", exc_info=True)

        _bg_tasks.append(asyncio.create_task(_v7_periodic_snapshot()))

    runtime_watchdog = RuntimeWatchdog(bus, state, config)
    runtime_watchdog.attach_local_brain(local_brain)
    router.attach_runtime_watchdog(runtime_watchdog)
    local_brain.attach_runtime_watchdog(runtime_watchdog)
    bus.on("state_changed", runtime_watchdog.on_state_changed)

    # ── Wire all event handlers (extracted for testability) ────────
    _wiring_ctx = wire_events(
        bus=bus, state=state, shutdown_event=shutdown_event, stt=stt, tts=tts, router=router,
        indicator=indicator, cache=cache, memory=memory, metrics=metrics,
        config=config, local_brain=local_brain, llm_queue=llm_queue,
        assistant_mode_mgr=assistant_mode_mgr,
        behavior=behavior,
        scheduler=scheduler, process_mgr=process_mgr, evolution=evolution,
        priority_sched=priority_sched,
        v3=args.v3, v4=args.v4,
    )
    _last_perceived_ms = _wiring_ctx["last_perceived_ms"]

    if llm_queue is not None:
        llm_queue.start()
    if priority_sched is not None:
        priority_sched.start()
    runtime_watchdog.start()
    logger.info("Runtime watchdog + priority scheduler started")

    async def _on_runtime_settings_changed(
        brain_profile: str | None = None,
        assistant_mode: str | None = None,
        **_kw,
    ) -> None:
        if web_dashboard is not None:
            await web_dashboard.broadcast_runtime_settings(
                brain_profile=brain_profile or brain_mode_mgr.active_profile,
                assistant_mode=assistant_mode or assistant_mode_mgr.active,
            )

    bus.on("runtime_settings_changed", _on_runtime_settings_changed)

    router.configure_diagnostics(
        stt=stt, tts=tts,
        metrics=metrics,
        local_brain=local_brain,
        health_monitor=health_monitor,
    )

    if web_dashboard is not None:
        async def _dashboard_unstick() -> None:
            """Dashboard UNSTICK: exit THINKING, stop TTS, clear ERROR_RECOVERY."""
            bus.emit("resume_listening")
            await asyncio.sleep(0.05)
            bus.emit("restart_listening")

        web_dashboard.set_unstick_callback(_dashboard_unstick)

        async def _on_text_input(text: str) -> None:
            """Handle typed input from dashboard — same as speech_final."""
            logger.info("Text input from dashboard: '%s'", text[:60])
            bus.emit("speech_final", text=text)

        web_dashboard.set_text_input_callback(_on_text_input)

        def _v7_health_payload() -> dict:
            from core.cognition.preemption import get_last_preemption_score
            from core.observability.debug_snapshot import get_debug_snapshot
            from core.observability.warnings import collect_v7_warnings

            ss: dict = {}
            if system_monitor is not None:
                try:
                    ss = system_monitor.get_system_state()
                except Exception:
                    ss = {}
            metrics = feedback_engine.compute_accuracy_metrics()
            health = feedback_engine.get_health_status(ss)
            warns = collect_v7_warnings(
                config, feedback_metrics=metrics, health_status=health,
            )
            pre = get_last_preemption_score()
            active_proj = None
            try:
                if local_brain is not None and getattr(local_brain, "_memory_graph", None):
                    active_proj = local_brain._memory_graph.get_last_active_project()
            except Exception:
                pass
            tl_n = timeline_memory.event_count() if timeline_memory else 0
            tl_prev = timeline_memory.recent_preview(8) if timeline_memory else []
            snap = get_debug_snapshot(
                config,
                runtime_mode=(
                    getattr(local_brain, "_current_runtime_mode", "SMART")
                    if local_brain
                    else "SMART"
                ),
                mode_info=(
                    getattr(local_brain, "_last_mode_info", {}) if local_brain else {}
                ),
                system_state=ss,
                feedback_metrics=metrics,
                last_retrieval_source=(
                    getattr(local_brain, "_last_retrieval_source", "")
                    if local_brain
                    else ""
                ),
                timeline_event_count=tl_n,
                timeline_recent_preview=tl_prev,
                active_project=active_proj,
                preemption=pre,
            )
            return {
                "health_status": health,
                "metrics": metrics,
                "warnings": warns,
                "snapshot": snap,
            }

        web_dashboard.set_v7_health_provider(_v7_health_payload)
        await web_dashboard.start()
    else:
        indicator.start()

    if local_brain and local_brain.available:
        model_name = Path(config.get("brain", {}).get("model_path", "local")).stem
        brain_label = f"Intent Engine + Agentic LLM ({model_name})"
    elif brain_enabled:
        brain_label = "Intent Engine + Local LLM (model unavailable)"
    else:
        brain_label = "Intent Engine ONLY — set brain.enabled for local LLM"
    cognitive_label = "Cognitive Layer ON (dream+curiosity)" if cognitive_enabled else "Cognitive OFF"
    logger.info("=== ATOM (Supernatural Intelligence OS) | Owner: Satyam | Mic: %s | %s | %s ===",
                stt.mic_name, brain_label, cognitive_label)
    if not brain_enabled:
        logger.warning("brain.enabled is false — voice Q&A disabled; commands still work")

    # ── Global Hotkey (Ctrl+Alt+A) ──────────────────────────────────────
    hotkey_active = False
    try:
        import keyboard

        def _hotkey_handler():
            """Toggle LISTENING state via keyboard shortcut. Also resumes from SLEEP."""
            try:
                if state.current is AtomState.SLEEP:
                    running_loop.call_soon_threadsafe(
                        bus.emit, "resume_listening",
                    )
                    logger.info("Hotkey: SLEEP -> resume listening")
                elif state.current is AtomState.LISTENING:
                    running_loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(state.transition(AtomState.IDLE))
                    )
                    logger.info("Hotkey: LISTENING -> IDLE")
                else:
                    # THINKING, SPEAKING, IDLE, ERROR_RECOVERY — unstick / resume
                    running_loop.call_soon_threadsafe(bus.emit, "resume_listening")
                    logger.info("Hotkey: resume_listening (unstick)")
            except Exception as e:
                logger.warning("Hotkey handler error: %s", e)

        keyboard.add_hotkey("ctrl+alt+a", _hotkey_handler, suppress=False)
        hotkey_active = True
        logger.info("Global hotkey registered: Ctrl+Alt+A (toggle listening)")
    except ImportError:
        logger.info("keyboard module not installed -- hotkey disabled (pip install keyboard)")
    except Exception as e:
        logger.warning("Could not register hotkey: %s", e)

    from core.power_governor import PowerGovernor
    power_governor = PowerGovernor(bus)
    power_governor.start()

    health_monitor.start()
    system_watcher.start()
    scheduler.start()
    autonomy.start()

    if cognitive_enabled:
        goal_engine.start()
        behavior_model.start()
        prediction_engine.start()
        self_optimizer.start()
        personality_modes.start()
        if dream_engine is not None:
            dream_engine.start()
        if curiosity_engine is not None:
            curiosity_engine.start()
        logger.info("Cognitive layer started (7 engines, incl. dream + curiosity)")

    # ── Start perception + governance modules ──────────────────────
    if wake_word_engine is not None and wake_word_engine.is_available:
        wake_word_engine.start(running_loop)

    if silicon_governor is not None and silicon_governor.is_available:
        silicon_governor.start()

    # ── Start JARVIS-level modules ──────────────────────────────
    system_scanner.start()
    owner_understanding.start()
    jarvis_core.start()
    real_world_intel.start()
    proactive_intel.start()
    logger.info(
        "Intelligence layer started: SystemScanner + OwnerUnderstanding + "
        "JarvisCore + RealWorldIntel + ProactiveEngine"
    )

    # ── Wire extracted event handlers ─────────────────────────────────
    from core.wiring.feature_handlers import (
        wire_documents_and_workflows,
        wire_dream_curiosity,
        wire_jarvis_and_system,
        wire_autonomy_and_governor,
    )
    from core.wiring.intelligence_handlers import (
        wire_self_healing,
        wire_voice_auth,
        wire_real_world,
    )

    wire_documents_and_workflows(
        bus=bus, router=router, security=security,
        document_engine=document_engine, workflow_engine=workflow_engine,
        screen_reader=screen_reader,
    )
    wire_dream_curiosity(
        bus=bus, dream_engine=dream_engine, curiosity_engine=curiosity_engine,
        emotion_detector=emotion_detector, cognitive_enabled=cognitive_enabled,
    )
    wire_jarvis_and_system(
        bus=bus, router=router, security=security, indicator=indicator,
        system_scanner=system_scanner, system_control=system_control,
        owner_understanding=owner_understanding,
    )
    wire_self_healing(
        bus=bus, self_healing=self_healing, code_introspector=code_introspector,
        security_fortress=security_fortress, context_engine=context_engine,
    )
    wire_voice_auth(
        bus=bus, security_fortress=security_fortress,
        context_engine=context_engine,
    )
    wire_real_world(
        bus=bus, real_world_intel=real_world_intel,
        context_fusion=context_fusion,
    )
    wire_autonomy_and_governor(
        bus=bus, router=router, indicator=indicator, memory=memory,
        autonomy=autonomy, state=state, tts=tts,
        web_dashboard=web_dashboard, emotion_detector=emotion_detector,
        wake_word_engine=wake_word_engine,
    )

    if cognitive_enabled:
        from core.wiring.cognitive_handlers import wire as wire_cognitive
        wire_cognitive(
            bus=bus, goal_engine=goal_engine,
            prediction_engine=prediction_engine,
            behavior_model=behavior_model,
            self_optimizer=self_optimizer,
            second_brain=second_brain,
            personality_modes=personality_modes,
            indicator=indicator, tts=tts,
            web_dashboard=web_dashboard,
        )

    if web_dashboard is not None:
        async def _push_habits_periodically() -> None:
            while not shutdown_event.is_set():
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=60.0)
                    break
                except asyncio.TimeoutError:
                    pass
                try:
                    habits = autonomy.get_habits_summary()
                    web_dashboard.broadcast_habits(habits)
                except Exception:
                    logger.debug("Dashboard habits broadcast failed", exc_info=True)
                if cognitive_enabled:
                    try:
                        web_dashboard.broadcast_goals(
                            goal_engine.get_goals_for_dashboard())
                        web_dashboard.broadcast_predictions(
                            prediction_engine.get_predictions_for_dashboard())
                        web_dashboard.broadcast_profile(
                            behavior_model.get_profile_for_dashboard())
                        web_dashboard.broadcast_mode(
                            personality_modes.get_mode_for_dashboard())
                    except Exception:
                        logger.debug("Dashboard cognitive broadcast failed", exc_info=True)
        _bg_tasks.append(asyncio.create_task(_push_habits_periodically()))

    await cold_start.emit_restored_context()
    state.always_listen = True
    logger.info(
        "ATOM -- Supernatural Intelligence OS | always listening | perf=%s | health=%.0fs watcher=%.0fs maint=%.0fs",
        perf_mode, health_interval, watcher_interval, maint_interval,
    )
    await state.transition(AtomState.LISTENING)

    async def _startup_greeting() -> None:
        """Speak a context-aware greeting with world intelligence."""
        mode_label = _MODE_LABELS.get(perf_mode, perf_mode)

        # Real-world awareness
        world_ctx = real_world_intel.get_world_context()
        weather_line = ""
        if not world_ctx.weather.is_stale:
            weather_line = f" Weather outside: {world_ctx.weather.summary()}."
        elif world_ctx.weather.condition != "unknown":
            weather_line = f" Last known weather: {world_ctx.weather.summary()}."

        news_line = ""
        if world_ctx.headlines:
            news_line = f" Top news: {world_ctx.headlines[0][:80]}."

        temporal = world_ctx.temporal
        holiday_line = f" Today is {temporal.holiday_name}." if temporal.is_holiday else ""

        # System awareness
        sys_summary = platform_adapter.get_system_summary()
        scan_health = ""
        if system_scanner.last_scan:
            h = system_scanner.last_scan.get("health", {}).get("overall", 0)
            scan_health = f" System health: {h} out of 100."

        # Security awareness
        security_label = "secured" if security_fortress.is_authenticated else "awaiting authentication"
        integrity_ok, _ = security_fortress.check_integrity()

        # Cognitive awareness
        cognitive_msg = ""
        if cognitive_enabled:
            active_goals = goal_engine.active_count
            if active_goals:
                cognitive_msg = f" You have {active_goals} active goal{'s' if active_goals > 1 else ''}."

        # Capability count
        cap_count = tool_registry.count

        # Build greeting using adaptive personality
        time_g = _adaptive_personality.greeting_response()

        # System Diagnostics Check
        bottleneck_msg = ""
        if system_scanner.bottlenecks:
            criticals = [b for b in system_scanner.bottlenecks if b.severity in ("critical", "high")]
            if criticals:
                bottleneck_msg = f" Note: {criticals[0].description}"

        greeting = (
            f"{time_g} "
            f"All systems online. {sys_summary}.{scan_health}{bottleneck_msg} "
            f"Security: {security_label}, {'integrity clean' if integrity_ok else 'integrity alert'}. "
            f"Running in {mode_label} mode with {cap_count} tools ready.{cognitive_msg} "
            f"{weather_line}{holiday_line}{news_line} "
            f"I know the world, I know you, and I only answer to you, Boss. What do you need?"
        )

        try:
            bat = psutil.sensors_battery()
            if bat and bat.percent < 20:
                greeting += f" Heads up, battery is at {bat.percent:.0f} percent."
        except Exception:
            logger.debug("Battery check failed", exc_info=True)

        logger.info("Startup greeting: %s", greeting[:200])

        await state.transition(AtomState.THINKING)
        bus.emit_long("partial_response", text=greeting, is_first=True, is_last=True)

        await stt_preload_done.wait()
        logger.info("STT ready -- ATOM fully operational")
        
        if distributed_mode:
            # In distributed mode, external workers own the listening loop.
            while True:
                await asyncio.sleep(1)
        else:
            await stt.start_listening()

    _bg_tasks.append(asyncio.create_task(_startup_greeting()))

    async def _auto_performance_loop() -> None:
        """Latency-driven auto mode. Ignores transient CPU spikes (LLM inference)."""
        auto_effective = "lite"
        interval = 45.0
        _COOLDOWN_S = 120.0
        _last_switch_time = 0.0
        _BRAIN_FOR_PERF = {"full": "brain", "lite": "balanced", "ultra_lite": "atom"}
        try:
            pass
        except Exception:
            logger.warning("Auto performance mode requires psutil")
            return

        while not shutdown_event.is_set():
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
            if shutdown_event.is_set():
                break
            try:
                from core.state_manager import AtomState
                if state.current in (AtomState.THINKING, AtomState.SPEAKING):
                    continue

                now = time.monotonic()
                if now - _last_switch_time < _COOLDOWN_S:
                    continue

                cpu = psutil.cpu_percent(interval=2.0)
                lat_ms = _last_perceived_ms.get("ms")

                if lat_ms is None:
                    continue

                lat_s = lat_ms / 1000.0
                if lat_s > 25.0:
                    target = "ultra_lite"
                elif lat_s > 12.0:
                    target = "lite"
                elif lat_s < 8.0 and cpu < 50:
                    target = "full"
                else:
                    target = auto_effective

                if target == auto_effective:
                    continue

                prev_l = _MODE_LABELS.get(auto_effective, auto_effective)
                new_l = _MODE_LABELS.get(target, target)
                msg = f"Boss, switching from {prev_l} to {new_l} mode. Response time was {lat_s:.0f} seconds."
                logger.info(
                    "Auto perf: latency=%.0fs CPU=%.0f%% -> %s -> %s",
                    lat_s, cpu, auto_effective, target,
                )
                _last_switch_time = now
                auto_effective = target
                bus.emit_long(
                    "partial_response", text=msg,
                    is_first=True, is_last=True,
                )
                bp = _BRAIN_FOR_PERF.get(target, "balanced")
                if brain_mode_mgr is not None:
                    ok, _ = brain_mode_mgr.set_profile(bp)
                    if ok:
                        logger.info("Auto perf: brain profile -> %s", bp)
                indicator.broadcast_perf_mode(target)
            except Exception as exc:
                logger.debug("Auto performance check error: %s", exc)

    if perf_mode == "auto":
        _bg_tasks.append(asyncio.create_task(_auto_performance_loop()))
        logger.info(
            "Auto performance mode active (CPU thresholds: %s/%s; latency bands <10s / 10-30s / >30s)",
            perf_cfg.get("auto_threshold_mid", 40),
            perf_cfg.get("auto_threshold_high", 70),
        )

    _last_ttl_change_cycle = {"v": 0}

    def _self_tune() -> None:
        """Adaptive runtime tuning based on collected metrics."""
        snap = metrics.snapshot()

        hit_rate = snap.get("cache_hit_rate_pct", 0)
        total = snap.get("cache_hits", 0) + snap.get("cache_misses", 0)
        if hit_rate > 65 and _last_ttl_change_cycle["v"] <= 0:
            new_ttl = min(600.0, cache._ttl * 1.2)
            if new_ttl != cache._ttl:
                cache._ttl = new_ttl
                _last_ttl_change_cycle["v"] = 3
                logger.info("Self-tune: cache TTL -> %.0fs (hit rate %.0f%%)",
                            new_ttl, hit_rate)
        elif hit_rate < 15 and total > 10 and _last_ttl_change_cycle["v"] <= 0:
            new_ttl = max(120.0, cache._ttl * 0.8)
            if new_ttl != cache._ttl:
                cache._ttl = new_ttl
                _last_ttl_change_cycle["v"] = 3
                logger.info("Self-tune: cache TTL -> %.0fs (hit rate %.0f%%)",
                            new_ttl, hit_rate)
        if _last_ttl_change_cycle["v"] > 0:
            _last_ttl_change_cycle["v"] -= 1

    _proactive_state = _wiring_ctx["proactive_state"]

    proactive_alerts = perf_cfg.get("proactive_alerts", perf_mode == "full")
    idle_reminder = perf_cfg.get("idle_reminder", perf_mode == "full")
    cache_purge_cycles = max(1, 1200 // maint_interval)
    tune_cycles = max(1, 600 // maint_interval)

    async def _periodic_maintenance() -> None:
        """Periodic background maintenance. Frequency adapts to performance mode."""
        cycle = 0
        while not shutdown_event.is_set():
            try:
                await asyncio.wait_for(shutdown_event.wait(),
                                       timeout=float(maint_interval))
                break
            except asyncio.TimeoutError:
                pass
            cycle += 1

            if perf_mode != "ultra_lite":
                log_health(metrics)

            if cycle % cache_purge_cycles == 0:
                cache.purge_expired()
                logger.info("Periodic maintenance: cache purged")

            if cycle % tune_cycles == 0:
                try:
                    _self_tune()
                except Exception:
                    logger.debug("Self-tune error", exc_info=True)

            if proactive_alerts and state.current.value in ("idle", "listening"):
                try:
                    bat = psutil.sensors_battery()
                    if bat and bat.percent <= 20 and not bat.power_plugged:
                        if not _proactive_state.get("low_battery_warned"):
                            _proactive_state["low_battery_warned"] = True
                            bus.emit_long("response_ready",
                                          text=f"Boss, battery is down to {bat.percent:.0f} percent. You may want to plug in.")
                            logger.info("Proactive: low battery alert (%d%%)", bat.percent)
                    elif bat and bat.percent > 30:
                        _proactive_state["low_battery_warned"] = False
                except Exception:
                    logger.debug("Battery monitoring failed", exc_info=True)

            if idle_reminder and state.current.value in ("idle", "listening"):
                idle_minutes = (time.monotonic() - _proactive_state["last_query_time"]) / 60
                if idle_minutes >= 45 and cycle % 15 == 0:
                    bus.emit_long("response_ready",
                                  text="All quiet, Boss. I'm here whenever you need me.")
                    logger.info("Proactive: idle reminder (%.0f min)", idle_minutes)

            if proactive.enabled and state.current.value in ("idle", "listening"):
                greeting = proactive.check_greeting()
                if greeting:
                    indicator.add_log("info", greeting)
                    logger.info("Proactive: greeting sent")
                idle_s = time.monotonic() - _proactive_state["last_query_time"]
                idle_hint = proactive.check_idle(idle_s)
                if idle_hint:
                    indicator.add_log("info", idle_hint)

    maintenance_task = asyncio.create_task(_periodic_maintenance())

    try:
        await shutdown_event.wait()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Interrupt received")
    finally:
        logger.info("Cleaning up...")
        power_governor.stop()
        scheduler.stop()
        system_watcher.stop()
        await health_monitor.stop()
        autonomy.stop()
        if cognitive_enabled:
            goal_engine.stop()
            behavior_model.stop()
            prediction_engine.stop()
            self_optimizer.stop()
            personality_modes.stop()
            if dream_engine is not None:
                dream_engine.stop()
            if curiosity_engine is not None:
                curiosity_engine.stop()
            second_brain.persist()
            logger.info("Cognitive layer stopped and persisted")
        if wake_word_engine is not None:
            wake_word_engine.shutdown()
        if silicon_governor is not None:
            silicon_governor.shutdown()
        if workflow_engine is not None:
            workflow_engine.persist()
        if document_engine is not None:
            document_engine.persist()
        security_fortress.shutdown()
        self_healing.persist()
        real_world_intel.shutdown()
        proactive_intel.stop()
        logger.info("Production + intelligence systems shut down")
        jarvis_core.stop()
        owner_understanding.stop()
        system_scanner.stop()
        system_scanner.persist()
        system_indexer.stop()
        media_watcher.stop()
        if fs_watcher is not None:
            fs_watcher.shutdown()
        logger.info("JARVIS intelligence modules stopped and persisted")
        maintenance_task.cancel()
        for _t in _bg_tasks:
            if not _t.done():
                _t.cancel()
        _all_cancelled = [maintenance_task] + [t for t in _bg_tasks if t.cancelled()]
        if _all_cancelled:
            await asyncio.gather(*_all_cancelled, return_exceptions=True)
        behavior.persist()
        evolution.persist()
        if hotkey_active:
            try:
                import keyboard
                keyboard.unhook_all()
            except Exception:
                pass
        if llm_queue is not None:
            await llm_queue.shutdown()
        if runtime_watchdog is not None:
            await runtime_watchdog.shutdown()
        if priority_sched is not None:
            await priority_sched.shutdown()
        bus.clear()
        stt.shutdown()
        await tts.shutdown()
        if local_brain:
            local_brain.close()
        if web_dashboard is not None:
            await web_dashboard.shutdown_async()
        else:
            indicator.shutdown()
        cold_start.persist_snapshot()
        snap = metrics.snapshot()
        logger.info(
            "SESSION_SUMMARY queries=%d cache_hit_pct=%.1f llm_calls=%d perceived_avg_ms=%s",
            snap.get("queries_total", 0),
            snap.get("cache_hit_rate_pct", 0),
            snap.get("llm_calls", 0),
            snap.get("perceived_avg_ms", "—"),
        )
        log_health(metrics)
        memory.persist()
        executor.shutdown(wait=False)
        logger.info("ATOM stopped.")


def run_atom(config_overrides: dict | None = None) -> None:
    """Launch ATOM programmatically with optional config overrides.

    This is the single entry point for embedding ATOM as a "brain".
    Pass a dict to override any settings.json value, e.g.::

        run_atom({
            "features": {"desktop_control": False},
            "control": {"lock_mode": "safe_only"},
        })

    Includes crash-guard with exponential backoff.
    Supports graceful restart when performance mode is changed via UI.
    """
    global _restart_requested
    set_config_overrides(config_overrides or {})

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    MAX_RETRIES = 5
    MAX_BACKOFF_S = 30.0
    attempt = 0

    while attempt < MAX_RETRIES:
        try:
            asyncio.run(main())
            if _restart_requested:
                _restart_requested = False
                shutdown_event.clear()
                logger.info("Graceful restart requested (mode change) -- restarting ATOM...")
                import time as _time
                _time.sleep(2.0)
                attempt = 0
                continue
            break
        except KeyboardInterrupt:
            break
        except SystemExit:
            break
        except Exception:
            attempt += 1
            backoff = min(2 ** attempt, MAX_BACKOFF_S)
            crash_logger = logging.getLogger("atom.crash_guard")
            crash_logger.exception(
                "ATOM crashed (attempt %d/%d) -- restarting in %.0fs",
                attempt, MAX_RETRIES, backoff,
            )
            if attempt >= MAX_RETRIES:
                crash_logger.critical(
                    "Max restart attempts reached (%d) -- giving up",
                    MAX_RETRIES,
                )
                break
            import time as _time
            _time.sleep(backoff)
            shutdown_event.clear()

    set_config_overrides({})


if __name__ == "__main__":
    run_atom()
