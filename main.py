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
import os
import sys
import time
from datetime import datetime
import psutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Hugging Face tokenizers fork a worker pool the first time they're
# imported. Combined with our asyncio event loop + multiprocessing
# semaphores in mlx-vlm, the default behaviour leaks one unnamed
# semaphore at process exit and prints
# ``resource_tracker: There appear to be 1 leaked semaphore objects``
# on every shutdown. Setting this env var BEFORE any tokenizer import
# (sentence-transformers, mlx-vlm, transformers) tells the tokenizer
# library to do its work in-process, which both fixes the leak and
# avoids a class of fork-after-exec deadlocks on macOS.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from core.boot.config_loader import load_config, set_config_overrides


logger = logging.getLogger("atom.main")
shutdown_event: asyncio.Event | None = None
_restart_requested = False


from core.boot.wiring import wire_events
from core.boot.cognitive_loop_wiring import wire_cognitive_loop


async def main() -> None:
    global shutdown_event
    shutdown_event = asyncio.Event()

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
        logger.debug("Session manager configure skipped or failed", exc_info=True)
    logger.info(
        "ATOM owner binding: %s — access control via core/owner_gate.py",
        owner_display_name(),
    )
    try:
        # Security: ATOM has already loaded the dashboard token into the
        # owner gate's in-memory state; everything else (HF_TOKEN, OPENAI_API_KEY,
        # GEMINI_API_KEY, ...) gets snapshotted by the secret scrub and blanked
        # in ``os.environ`` so child processes / crash dumps don't leak them.
        from core.security_secret_scrub import scrub_sensitive_env

        _secret_snapshot = scrub_sensitive_env(preserve=("ATOM_DASHBOARD_TOKEN",))
    except Exception:
        _secret_snapshot = {}
        logger.debug("Secret scrub skipped or failed", exc_info=True)

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

    from core.async_event_bus import AsyncEventBus
    bus = AsyncEventBus()
    from core.state.event_bus import AtomRuntimeStateBridge
    from core.state.ui_adapter import StateAwareIndicator

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

    atom_runtime = AtomRuntimeStateBridge(bus)
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
    try:
        memory.start_background_writers()
    except Exception:
        logger.info("Memory engine background writer failed to start", exc_info=True)
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
    from core.hardware_profile import get_hardware_profile
    from core.system.system_monitor import SystemMonitor

    v7i_cfg = config.get("v7_intelligence") or {}
    _tl_max = int(v7i_cfg.get("max_timeline_size") or v7i_cfg.get("timeline_max_events", 500))
    timeline_memory = TimelineMemory(
        max_events=_tl_max,
        summarize_on_prune=bool(v7i_cfg.get("timeline_summarize_on_prune", False)),
    )

    try:
        from core.intent_engine import memory_recall_intents as _mri
        _mri.set_timeline(timeline_memory)
    except Exception:
        logger.info("Memory-recall intent wiring failed", exc_info=True)
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

    from core.fast_path import startup_warm_up, ParallelPipeline
    startup_warm_up(intent_engine, cache, memory, config)

    from voice.voice_pipeline import VoicePipeline
    voice_pipeline = VoicePipeline(
        bus, state, config,
        mic_manager=mic_manager,
        intent_engine=intent_engine,
    )
    voice_pipeline.build()
    voice_pipeline.build_audio_intelligence()
    stt = voice_pipeline.stt
    tts = voice_pipeline.tts
    stt_runtime_label = voice_pipeline.stt_runtime_label
    stt_runtime_error = voice_pipeline.stt_runtime_error
    stt_runtime_fallbacks = voice_pipeline.stt_runtime_fallbacks
    tts_runtime_label = voice_pipeline.tts_runtime_label

    # v3 Phase 4: WhisperConfirmer for second-pass STT on suspect finals.
    # Opt-in via config["stt"]["whisper_confirm"]["enabled"]=true. Lazy
    # model load so cold boot stays fast. Wired only if the active STT
    # backend supports attach_whisper_confirmer (currently macOS native).
    try:
        _stt_cfg = config.get("stt") or {}
        if (_stt_cfg.get("whisper_confirm") or {}).get("enabled", False):
            if hasattr(stt, "attach_whisper_confirmer"):
                from voice.whisper_confirmer import WhisperConfirmer
                _wc = WhisperConfirmer(_stt_cfg)
                stt.attach_whisper_confirmer(_wc)
                logger.info(
                    "WhisperConfirmer attached to STT (model=%s, ring=%.1fs, decode=%.1fs)",
                    _wc._model_size, _wc._ring_seconds, _wc._decode_seconds,
                )
            else:
                logger.info(
                    "WhisperConfirmer enabled in config but %s STT does not "
                    "expose attach_whisper_confirmer -- skipping wiring",
                    type(stt).__name__,
                )
    except Exception:
        logger.warning("WhisperConfirmer wiring failed (non-fatal)", exc_info=True)

    # ── Boot order (runtime truth) ─────────────────────────────────
    # InferenceGuard + SiliconGovernor before CognitiveKernel so routing sees
    # VRAM/hardware state. CognitiveKernel before LocalBrain/RAG so semantic
    # stack detection (_semantic_stack_available) gates RAG without forward refs.
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
        logger.info(
            "ATOM V7: InferenceGuard + RecoveryManager + GPUStallWatchdog (Apple Silicon)",
        )

    silicon_governor = None
    if config.get("gpu", {}).get("enabled", True):
        from core.silicon_governor import SiliconGovernor
        silicon_governor = SiliconGovernor(bus, config)
        if silicon_governor.is_available:
            logger.info("Silicon Governor: monitoring active (%s)", silicon_governor.gpu_name)

    from core.cognitive_kernel import CognitiveKernel, ExecPath

    cognitive_kernel = CognitiveKernel(
        config=config,
        bus=bus,
        brain_mode_manager=brain_mode_mgr,
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

    brain_enabled = config.get("brain", {}).get("enabled", False)

    from cursor_bridge.structured_prompt_builder import StructuredPromptBuilder
    prompt_builder = StructuredPromptBuilder(config)

    local_brain = None
    prefetch_eng = None
    shared_memory_graph = None
    if brain_enabled:
        from cursor_bridge.local_brain_controller import LocalBrainController
        local_brain = LocalBrainController(
            bus, prompt_builder, config,
            brain_mode_manager=brain_mode_mgr,
        )
        local_brain.set_action_executor(router.action_executor)
        # v3 Phase 3.3: cloud-stream chunks reuse this controller's
        # sanitiser so Gemini output is cleaned with the same rules as
        # the local brain (CoT preface, prompt-leak, ChatML tokens).
        router.attach_local_brain_controller(local_brain)
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
            semantic_rag_ready = bool(getattr(cognitive_kernel, "_semantic_stack_available", False))
            if _rag_cfg.get("enabled", True) and semantic_rag_ready:
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
            elif _rag_cfg.get("enabled", True):
                logger.warning(
                    "Semantic RAG disabled at boot: vector dependencies are unavailable, so ATOM will stay in honest keyword-only memory mode.",
                )
        except Exception as exc:
            logger.warning("V7 intelligence layer partial wiring: %s", exc)
        logger.info("Local brain ENABLED (agentic mode, tool-use, brain.enabled=true)")
    else:
        logger.info("Local brain DISABLED — enable brain.enabled for voice Q&A")

    if inference_guard is not None and brain_enabled and local_brain is not None:
        local_brain.attach_inference_guard(inference_guard)

    _memory_pressure_threshold = float(
        (config.get("memory") or {}).get("pressure_threshold_pct", 85.0),
    )
    _memory_pressure_relief = float(
        (config.get("memory") or {}).get(
            "pressure_relief_pct",
            max(0.0, _memory_pressure_threshold - 10.0),
        ),
    )

    # Tiered memory-pressure orchestrator (Sprint B1):
    #  Level 1 (warn)     -> drop KV prefix cache         [cheap, fully recoverable]
    #  Level 2 (active)   -> + shrink RAG top_k / clear RAG caches
    #  Level 3 (critical) -> + unload sentence-transformer weights (keyword fallback)
    #
    # Each level uses hysteresis (10% below trigger) so we don't flap,
    # and higher tiers also imply lower tiers — the cheapest action is
    # released last when pressure clears.
    _pressure_cfg = (config.get("memory") or {}).get("pressure_tiers") or {}
    _warn_threshold = float(
        _pressure_cfg.get("warn_pct", max(0.0, _memory_pressure_threshold - 4.0)),
    )
    _warn_relief = float(
        _pressure_cfg.get("warn_relief_pct", max(0.0, _warn_threshold - 6.0)),
    )
    _critical_threshold = float(
        _pressure_cfg.get("critical_pct", min(100.0, _memory_pressure_threshold + 6.0)),
    )
    _critical_relief = float(
        _pressure_cfg.get(
            "critical_relief_pct",
            max(0.0, _critical_threshold - 8.0),
        ),
    )
    _embedding_pressure_unloaded = False
    _prompt_kv_cache_dropped = False
    _last_pressure_tier = 0
    # Thermal clamp bookkeeping (Sprint C4). We apply a multiplier to
    # the LLM's max_tokens budget so sustained heat doesn't pin the CPU
    # at 100°C. ``_last_thermal_tier`` lets us only emit/act when the
    # tier actually changes.
    _last_thermal_tier = "nominal"
    # Poll counter to require ``N`` consecutive hot samples before we
    # actually clamp — a one-off spike shouldn't trim answers mid-turn.
    _thermal_hot_streak = 0
    _THERMAL_HOT_STREAK_REQUIRED = 2

    # B3: auto-demote the brain profile under sustained memory pressure.
    # ``_pressure_hot_streak`` counts consecutive samples at tier>=2.
    # ``_pressure_clear_streak`` counts consecutive tier=0 samples.
    # ``_pre_pressure_profile`` remembers what to restore to.
    _pressure_hot_streak = 0
    _pressure_clear_streak = 0
    _pre_pressure_profile: str | None = None
    _PRESSURE_HOT_STREAK_REQUIRED = 3
    _PRESSURE_CLEAR_STREAK_REQUIRED = 4

    async def _on_silicon_stats_update(stats=None, **_kw) -> None:
        nonlocal _embedding_pressure_unloaded, _prompt_kv_cache_dropped
        nonlocal _last_pressure_tier
        nonlocal _last_thermal_tier, _thermal_hot_streak
        nonlocal _pressure_hot_streak, _pressure_clear_streak, _pre_pressure_profile
        if not isinstance(stats, dict):
            return
        try:
            memory_pct = float(stats.get("memory_pct", 0.0) or 0.0)
        except (TypeError, ValueError):
            return

        try:
            memory.apply_memory_pressure(memory_pct)
        except Exception:
            logger.info("MemoryEngine pressure hook failed", exc_info=True)

        if local_brain is not None and hasattr(local_brain, "apply_memory_pressure"):
            try:
                local_brain.apply_memory_pressure(memory_pct)
            except Exception:
                logger.info("Local brain pressure hook failed", exc_info=True)

        # Decide current tier with per-tier hysteresis. We ratchet up as
        # soon as a threshold is crossed, but only ratchet down when the
        # relief point is reached — this keeps the system from flapping
        # in and out of the critical tier while MLX is still running hot.
        prev_tier = _last_pressure_tier
        tier = prev_tier
        if memory_pct >= _critical_threshold:
            tier = 3
        elif memory_pct >= _memory_pressure_threshold:
            tier = max(tier, 2)
        elif memory_pct >= _warn_threshold:
            tier = max(tier, 1)

        if prev_tier == 3 and memory_pct <= _critical_relief:
            tier = 2 if memory_pct > _memory_pressure_relief else (
                1 if memory_pct > _warn_relief else 0
            )
        elif prev_tier == 2 and memory_pct <= _memory_pressure_relief:
            tier = 1 if memory_pct > _warn_relief else 0
        elif prev_tier == 1 and memory_pct <= _warn_relief:
            tier = 0

        if tier != prev_tier:
            logger.warning(
                "Memory pressure tier %d -> %d (memory_pct=%.1f%%)",
                prev_tier, tier, memory_pct,
            )
            try:
                bus.emit_fast(
                    "memory_pressure_tier_changed",
                    previous=prev_tier,
                    current=tier,
                    memory_pct=round(memory_pct, 1),
                )
            except Exception:
                logger.debug("pressure tier emit failed", exc_info=True)
            _last_pressure_tier = tier

        # Level 1+: drop MLX prompt-prefix KV cache. Cheapest action, 100-500MB back.
        if tier >= 1:
            if (
                not _prompt_kv_cache_dropped
                and local_brain is not None
                and hasattr(local_brain, "drop_prompt_caches")
            ):
                try:
                    local_brain.drop_prompt_caches(reason=f"memory_pressure_tier{tier}")
                    _prompt_kv_cache_dropped = True
                except Exception:
                    logger.info("Prompt cache drop failed", exc_info=True)
        else:
            _prompt_kv_cache_dropped = False

        # Level 3: unload sentence-transformer weights and fall back
        # to the keyword/warm-file path for memory + RAG. The warm
        # file keeps the hot-set answering queries, so user-visible
        # latency only grows for never-seen queries.
        if inference_guard is None:
            return

        if tier >= 3:
            if not _embedding_pressure_unloaded:
                inference_guard.mark_loaded("embeddings", False)
                inference_guard.request_unload(
                    "embeddings", f"memory_pressure_tier{tier}",
                )
                _embedding_pressure_unloaded = True
        elif tier <= 1:
            _embedding_pressure_unloaded = False

        # Sprint B3: auto-demote brain profile under sustained pressure.
        # We wait for ``_PRESSURE_HOT_STREAK_REQUIRED`` consecutive tier>=2
        # samples so brief spikes don't trigger a switch. Restoring is
        # symmetric — tier=0 must hold for ``_PRESSURE_CLEAR_STREAK_REQUIRED``
        # samples before we reinstate the prior profile. Only ever demote
        # from ``full_performance`` → ``optimal`` to keep the behaviour
        # reversible; optimal already disables the high-cost background
        # subsystems (dream, curiosity, prediction_prefetch, autonomy).
        if brain_mode_mgr is not None:
            if tier >= 2:
                _pressure_hot_streak += 1
                _pressure_clear_streak = 0
            elif tier == 0:
                _pressure_clear_streak += 1
                _pressure_hot_streak = 0
            else:
                _pressure_hot_streak = max(0, _pressure_hot_streak - 1)
                _pressure_clear_streak = 0

            try:
                active = brain_mode_mgr.active_profile
            except Exception:
                active = "optimal"

            if (
                _pressure_hot_streak >= _PRESSURE_HOT_STREAK_REQUIRED
                and active == "full_performance"
                and _pre_pressure_profile is None
            ):
                _pre_pressure_profile = active
                try:
                    ok, _msg = brain_mode_mgr.set_profile("optimal", force=True)
                    if ok:
                        logger.warning(
                            "Auto-demoted brain profile under sustained pressure: "
                            "%s -> optimal (memory_pct=%.1f%%, streak=%d)",
                            active, memory_pct, _pressure_hot_streak,
                        )
                        try:
                            bus.emit_fast(
                                "brain_profile_auto_demoted",
                                previous=active,
                                current="optimal",
                                reason="memory_pressure_sustained",
                                memory_pct=round(memory_pct, 1),
                            )
                        except Exception:
                            logger.debug("auto-demote emit failed", exc_info=True)
                except Exception:
                    logger.info("Brain profile auto-demote failed", exc_info=True)

            elif (
                _pre_pressure_profile is not None
                and _pressure_clear_streak >= _PRESSURE_CLEAR_STREAK_REQUIRED
                and active == "optimal"
            ):
                restored_target = _pre_pressure_profile
                _pre_pressure_profile = None
                try:
                    ok, _msg = brain_mode_mgr.set_profile(restored_target, force=True)
                    if ok:
                        logger.info(
                            "Brain profile auto-restored after pressure cleared: "
                            "optimal -> %s (memory_pct=%.1f%%, streak=%d)",
                            restored_target, memory_pct, _pressure_clear_streak,
                        )
                        try:
                            bus.emit_fast(
                                "brain_profile_auto_restored",
                                previous="optimal",
                                current=restored_target,
                                reason="memory_pressure_cleared",
                            )
                        except Exception:
                            logger.debug("auto-restore emit failed", exc_info=True)
                except Exception:
                    logger.info("Brain profile auto-restore failed", exc_info=True)

        # Thermal clamp (Sprint C4). Map macOS thermal pressure states
        # + throttled flag to a max_tokens multiplier and a friendly
        # reason string. Require ``_THERMAL_HOT_STREAK_REQUIRED``
        # consecutive hot samples before engaging so a one-off spike
        # doesn't trim answers mid-turn.
        thermal_pressure_raw = str(stats.get("thermal_pressure") or "").lower()
        is_throttled = bool(stats.get("is_throttled") or stats.get("throttled"))
        if thermal_pressure_raw in {"critical"} or is_throttled:
            thermal_tier_now = "critical"
        elif thermal_pressure_raw in {"heavy", "hot"}:
            thermal_tier_now = "hot"
        elif thermal_pressure_raw in {"fair", "warm"}:
            thermal_tier_now = "warm"
        else:
            thermal_tier_now = "nominal"

        if thermal_tier_now in {"hot", "critical"}:
            _thermal_hot_streak += 1
        else:
            _thermal_hot_streak = 0

        # We only flip the active clamp tier once the streak is met.
        # Cooling is immediate — once thermals drop back to nominal we
        # restore full token budget right away.
        target_tier = _last_thermal_tier
        if thermal_tier_now == "nominal":
            target_tier = "nominal"
        elif thermal_tier_now == "warm":
            target_tier = "warm"
        elif (
            thermal_tier_now in {"hot", "critical"}
            and _thermal_hot_streak >= _THERMAL_HOT_STREAK_REQUIRED
        ):
            target_tier = thermal_tier_now

        if target_tier != _last_thermal_tier:
            ratio_map = {
                "nominal": 1.0,
                "warm": 0.85,
                "hot": 0.6,
                "critical": 0.4,
            }
            reason_map = {
                "nominal": "thermal_nominal",
                "warm": "thermal_warm",
                "hot": "thermal_hot",
                "critical": "thermal_critical",
            }
            ratio = ratio_map.get(target_tier, 1.0)
            reason = reason_map.get(target_tier, "thermal")
            if local_brain is not None and hasattr(local_brain, "set_thermal_clamp"):
                try:
                    local_brain.set_thermal_clamp(ratio, reason=reason)
                except Exception:
                    logger.info("Thermal clamp dispatch failed", exc_info=True)
            try:
                bus.emit_fast(
                    "thermal_derate_update",
                    previous=_last_thermal_tier,
                    current=target_tier,
                    ratio=ratio,
                )
            except Exception:
                logger.debug("thermal derate emit failed", exc_info=True)
            _last_thermal_tier = target_tier

    bus.on("silicon_stats_update", _on_silicon_stats_update)

    try:
        from core.observability.error_rate_monitor import get_error_rate_monitor
        _obs_cfg = (config.get("observability") or {}) if isinstance(config, dict) else {}
        _error_monitor = get_error_rate_monitor(
            window_s=float(_obs_cfg.get("error_rate_window_s", 60.0)),
            threshold=int(_obs_cfg.get("error_rate_threshold", 5)),
            poll_interval_s=float(_obs_cfg.get("error_rate_poll_s", 10.0)),
        )
        _error_monitor.start(bus)
    except Exception:
        logger.info("Error-rate monitor wiring failed", exc_info=True)
        _error_monitor = None

    _last_error_burst_spoken: dict[str, float] = {"t": 0.0}

    async def _on_error_burst_detected(
        rate: int = 0,
        threshold: int = 0,
        window_s: float = 60.0,
        top_sources: list | None = None,
        **_kw: Any,
    ) -> None:
        try:
            if not shutdown_event.is_set():
                try:
                    from core.state_manager import AtomState
                    # Don't step on an active conversation.
                    if state.current in (AtomState.THINKING, AtomState.SPEAKING, AtomState.LISTENING):
                        return
                except Exception:
                    pass
            now = time.monotonic()
            # Self-alert cooldown even though the monitor also enforces one.
            if now - float(_last_error_burst_spoken.get("t", 0.0)) < 180.0:
                return
            _last_error_burst_spoken["t"] = now
            top_str = ""
            if top_sources:
                try:
                    top_str = ", ".join(str(s[0]) for s in top_sources[:2])
                except Exception:
                    top_str = ""
            msg = (
                f"Heads up, Boss — I spotted {rate} handler errors in the last minute."
                + (f" Mostly around {top_str}." if top_str else "")
                + " I'm still up, just flagging it."
            )
            speaker = tts if tts is not None and hasattr(tts, "speak_ack") else None
            if speaker is not None:
                try:
                    await speaker.speak_ack(msg)
                    return
                except Exception:
                    logger.info("error-burst TTS ack failed", exc_info=True)
            logger.warning("[error-burst] %s", msg)
        except Exception:
            logger.info("error-burst handler failed", exc_info=True)

    bus.on("atom_error_burst_detected", _on_error_burst_detected)

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

    pipeline_timer = PipelineTimer(bus, metrics)
    pipeline_timer.register()
    def _canonical_perf_mode(name: str | None) -> str:
        key = (name or "auto").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "optimal": "optimal",
            "atom": "optimal",
            "balanced": "optimal",
            "lite": "optimal",
            "ultra_lite": "optimal",
            "full_performance": "full_performance",
            "full": "full_performance",
            "brain": "full_performance",
            "auto": "auto",
        }
        return aliases.get(key, "auto")

    _PERF_DEFAULTS = {
        "optimal": {"health": 120, "watcher": 30, "maint": 180},
        "full_performance": {"health": 75, "watcher": 15, "maint": 120},
        "auto": {"health": 120, "watcher": 30, "maint": 180},
    }

    def _describe_perf_mode(mode_name: str) -> str:
        return str(mode_name or "optimal").replace("_", " ")

    def _initial_auto_mode_decision() -> tuple[str, str]:
        context_snapshot = context_engine.get_runtime_snapshot(media={"playing": False})
        activity_type = str(context_snapshot.get("activity_type") or "idle")
        try:
            battery = psutil.sensors_battery()
        except Exception:
            battery = None
        on_battery = bool(battery is not None and not getattr(battery, "power_plugged", False))
        battery_pct = float(getattr(battery, "percent", 100.0) or 100.0) if battery is not None else 100.0
        try:
            cpu_now = float(psutil.cpu_percent(interval=0.0) or 0.0)
            memory_now = float(psutil.virtual_memory().percent or 0.0)
        except Exception:
            cpu_now = 0.0
            memory_now = 0.0

        if on_battery and activity_type == "coding":
            return (
                "optimal",
                "Booted in Optimal due to battery power and an active development session.",
            )
        if (
            not on_battery
            and cpu_now <= float(perf_cfg.get("auto_threshold_mid", 40))
            and memory_now <= 70.0
        ):
            return (
                "full_performance",
                "Booted in Full Performance because power is connected and system load is low.",
            )
        if on_battery:
            return (
                "optimal",
                f"Booted in Optimal because the Mac is on battery ({battery_pct:.0f}%).",
            )
        return (
            "optimal",
            "Booted in Optimal while ATOM gathers live thermal and workload telemetry.",
        )

    perf_mode = _canonical_perf_mode(perf_cfg.get("mode", "auto"))
    perf_requested_mode = perf_mode
    perf_effective_mode = "optimal"
    initial_mode_reason = ""
    logger.info("Performance mode request: %s", perf_requested_mode)

    if perf_requested_mode == "auto":
        initial_auto_mode, initial_mode_reason = _initial_auto_mode_decision()
        brain_mode_mgr.set_profile(initial_auto_mode)
        perf_effective_mode = brain_mode_mgr.active_profile
    else:
        brain_mode_mgr.set_profile(perf_requested_mode)
        perf_effective_mode = brain_mode_mgr.active_profile
        initial_mode_reason = (
            f"Using {_describe_perf_mode(perf_effective_mode)} because it was explicitly requested."
        )

    perf_d = _PERF_DEFAULTS.get(perf_requested_mode, _PERF_DEFAULTS["optimal"])

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
        brain_mode_manager=brain_mode_mgr,
    )

    from core.proactive_awareness import ProactiveAwareness
    proactive = ProactiveAwareness(config)

    # ── Reasoning Engine ───────────────────────────────────────────
    from core.reasoning.tool_registry import get_tool_registry
    from core.reasoning.code_sandbox import CodeSandbox
    from core.reasoning.workflow_engine import WorkflowEngine
    from core.document_ingestion import DocumentIngestionEngine

    tool_registry = get_tool_registry()
    tool_registry.apply_confirmation_policy(
        config=config,
        command_registry=command_registry,
    )
    # ReasoningPlanner (core/reasoning/planner.py) is the multi-step planning library;
    # wire it from Router or LocalBrainController when orchestration should use it — see docs/ARCHITECTURE_CANONICAL.md
    code_sandbox = CodeSandbox(config)
    workflow_engine = WorkflowEngine(config)
    document_engine = DocumentIngestionEngine(config)

    logger.info(
        "Reasoning engine initialized: %d tools, sandbox, workflows, documents",
        tool_registry.count,
    )

    prompt_builder.set_tool_registry(tool_registry)
    # NOTE: prompt_builder.set_context_sources() is called later (after line ~1065)
    # when context_fusion and real_world_intel are actually instantiated.

    # ── ActionExecutor (bridges LLM tool calls -> Router dispatch) ──
    if brain_enabled and local_brain is not None:
        router.action_executor.set_registry(tool_registry)
        local_brain.set_action_executor(router.action_executor)
        logger.info("ActionExecutor connected: LLM -> security gate -> Router dispatch")

    # ── Perception Upgrade ─────────────────────────────────────────
    from voice.emotion_detector import EmotionDetector

    emotion_detector = EmotionDetector(config)

    voice_pipeline.build_wake_word()
    wake_word_engine = voice_pipeline.wake_word

    screen_reader = None
    if config.get("screen_reader", {}).get("enabled", True):
        from context.screen_reader import ScreenReader
        screen_reader = ScreenReader(config)
        logger.info(
            "Screen reader: %s (%s)",
            "OCR available" if screen_reader.is_available else "fallback mode",
            screen_reader.ocr_backend,
        )

    # ── Vision engine (camera + Apple Vision NE) ───────────────────
    # Built around AVFoundation single-frame capture + Apple Vision
    # face/object detection on the Neural Engine. No LLM/VLM is loaded
    # by this subsystem -- it stays inside the user's "one 7B model"
    # constraint while still giving ATOM eyes (built-in webcam +
    # Continuity Camera = iPhone-as-webcam).
    vision_engine = None
    # Hoisted so cold_start.warm_up can pick it up even when the vision
    # block below short-circuits (vision.enabled=false). cold_start
    # tolerates ``None`` and just skips the VLM warmup task.
    _captioner = None
    _vision_cfg = config.get("vision", {}) or {}
    if _vision_cfg.get("enabled", False):
        try:
            from core.perception.vision_engine import VisionEngine

            # Optional VLM captioner (SmolVLM-Instruct-4bit via mlx-vlm).
            # Stays disabled by default so ATOM still boots cleanly
            # when the user hasn't fetched the ~1.2 GB model yet.
            _vlm_cfg = _vision_cfg.get("vlm", {}) or {}
            if _vlm_cfg.get("enabled", False):
                try:
                    from core.perception.vlm_describe import VLMCaptioner
                    _vlm_repo = str(_vlm_cfg.get("model_repo") or "").strip()
                    _captioner = VLMCaptioner(
                        model_path=str(
                            _vlm_cfg.get("model_path")
                            or "models/smolvlm-instruct-4bit"
                        ),
                        model_repo=(_vlm_repo or None),
                        prompt=str(
                            _vlm_cfg.get("prompt")
                            or "Describe this image in one short sentence."
                        ),
                        max_tokens=int(_vlm_cfg.get("max_tokens", 48)),
                        temperature=float(_vlm_cfg.get("temperature", 0.0)),
                    )
                    _cap_reason = _captioner.disabled_reason()
                    if _cap_reason:
                        logger.warning(
                            "VLM captioner configured but offline: %s",
                            _cap_reason,
                        )
                    else:
                        logger.info(
                            "VLM captioner ready (path=%s, repo=%s)",
                            _captioner.model_path,
                            _vlm_repo or "<none>",
                        )
                except Exception:
                    logger.warning(
                        "VLM captioner init failed; vision stays "
                        "face-only", exc_info=True,
                    )
                    _captioner = None

            vision_engine = VisionEngine(
                enabled=True,
                preferred_camera=str(_vision_cfg.get("preferred_camera") or "auto"),
                explicit_uid=_vision_cfg.get("explicit_camera_uid"),
                audit_log_path=_vision_cfg.get("audit_log_path"),
                emit=getattr(bus, "emit_fast", None) or getattr(bus, "emit", None),
                min_gap_s=float(_vision_cfg.get("min_gap_s", 1.0)),
                capture_timeout_s=float(_vision_cfg.get("capture_timeout_s", 3.5)),
                captioner=_captioner,
                caption_max_age_s=float(
                    _vision_cfg.get("caption_max_age_s", 60.0),
                ),
            )
            router.attach_vision_engine(vision_engine)
            # Give the VoicePipeline the same handle so its wake-word
            # handler can fire ``look(describe=True)`` in the background
            # whenever ``vision.describe_on_wake`` is enabled.
            try:
                voice_pipeline.attach_vision_engine(vision_engine)
            except Exception:
                logger.debug(
                    "voice_pipeline.attach_vision_engine failed",
                    exc_info=True,
                )
            _vision_block_reason = vision_engine.disabled_reason()
            _vision_cams = vision_engine.list_cameras_human()
            if _vision_block_reason:
                logger.warning(
                    "Vision engine attached but offline: %s", _vision_block_reason,
                )
            else:
                logger.info(
                    "Vision engine ready: cameras=%s, preferred=%s",
                    _vision_cams or ["<none>"],
                    _vision_cfg.get("preferred_camera", "auto"),
                )
        except Exception:
            logger.warning("VisionEngine init failed; camera tools disabled", exc_info=True)
            vision_engine = None
    else:
        logger.info("Vision engine disabled (vision.enabled=false)")

    # ── v22: Hybrid Intelligence Layer (Security Gateway + Cloud + Confidence) ──
    from core.security_gateway import SecurityGateway
    from core.cloud.gemini_client import GeminiClient
    from core.confidence_engine import ConfidenceEngine
    from core.decision_engine import DecisionEngine
    from core.tools.search_tool import SearchTool
    from core.memory.preference_store import PreferenceStore
    from core.semantic_cache import SemanticCache

    security_gateway = SecurityGateway(config)
    confidence_engine = ConfidenceEngine(config)
    decision_engine = DecisionEngine(config)
    semantic_cache = SemanticCache(config)
    preference_store = PreferenceStore(config)

    # ── User Memory: unified long-term owner model ───────────────
    from core.memory.user_memory import UserMemory
    user_memory = UserMemory(
        preference_store=preference_store,
        behavior_tracker=behavior,
    )

    cloud_enabled_cfg = bool(config.get("cloud", {}).get("enabled", True))
    gemini_client: GeminiClient | None = None
    if cloud_enabled_cfg:
        gemini_client = GeminiClient(config, security_gateway=security_gateway)
        # Only probe the encrypted vault when the client hasn't already
        # resolved a key from settings.json. This avoids loading the crypto
        # stack on every boot when cloud is enabled-but-unkeyed, and keeps
        # a bad vault (missing deps, wrong master pw) off the boot path.
        if not gemini_client.is_available:
            from core.secrets_manager import get_gemini_fast_key

            _gemini_key = get_gemini_fast_key()
            if _gemini_key:
                gemini_client.configure_api_key(_gemini_key)
                logger.info("Gemini API key loaded from secure storage")
            else:
                logger.warning(
                    "Gemini API key not found in settings.json or vault. "
                    "Run: python scripts/setup_api_keys.py  (or set cloud.enabled=false)"
                )
    else:
        logger.info("Cloud/Gemini disabled in config (cloud.enabled=false) — local MLX only for LLM routing")

    search_tool = SearchTool(
        config, security_gateway=security_gateway, gemini_client=gemini_client,
    )

    # Wire cloud intelligence into Cognitive Kernel
    cognitive_kernel.attach_cloud_intelligence(
        confidence_engine=confidence_engine,
        search_tool=search_tool,
        gemini_client=gemini_client,
        semantic_cache=semantic_cache,
    )

    # Wire cloud intelligence into Router
    router.attach_cloud_intelligence(
        gemini_client=gemini_client,
        search_tool=search_tool,
        confidence_engine=confidence_engine,
        decision_engine=decision_engine,
        semantic_cache=semantic_cache,
        preference_store=preference_store,
        security_gateway=security_gateway,
    )

    # Wire cloud intelligence into LocalBrainController and PromptBuilder
    prompt_builder.set_preference_store(preference_store)
    if local_brain is not None:
        local_brain.attach_cloud_intelligence(
            confidence_engine=confidence_engine,
            decision_engine=decision_engine,
            gemini_client=gemini_client,
            semantic_cache=semantic_cache,
            preference_store=preference_store,
        )

    logger.info(
        "v22 Hybrid Intelligence: SecurityGateway + GeminiClient(%s) + "
        "ConfidenceEngine + DecisionEngine + SearchTool + PreferenceStore + "
        "SemanticCache(semantic=%s, threshold=%.2f)",
        "available" if (gemini_client and gemini_client.is_available) else "disabled",
        semantic_cache.is_semantic,
        float((config.get("semantic_cache", {}).get("threshold", 0.85))),
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

    # Wire SecurityGateway → Fortress audit trail
    try:
        security_gateway.attach_audit_trail(security_fortress._audit)
    except Exception:
        logger.info("SecurityGateway audit trail wiring skipped", exc_info=True)

    # CodeIntrospector ASTs every .py in the repo (~1.6s on M5 Air for ~360
    # files). It's only consumed by SelfHealingEngine's failure
    # root-cause analysis, which can't fire until at least the first
    # post-boot exception — moving it off the boot path makes ATOM ready
    # ~1.6s sooner without losing self-healing capability.
    async def _bg_introspect_scan() -> None:
        try:
            loop = asyncio.get_running_loop()
            count = await loop.run_in_executor(None, code_introspector.scan)
            logger.info(
                "CodeIntrospector ready: %d modules indexed (background scan)",
                count,
            )
        except Exception:
            logger.warning("CodeIntrospector background scan failed", exc_info=True)
    asyncio.create_task(_bg_introspect_scan())
    logger.info(
        "Production systems initialized: SecurityFortress(%s) + "
        "CodeIntrospector(scanning in background) + SelfHealingEngine",
        security_fortress.vault_backend_label,
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

    from core.system_profile import SystemProfile
    system_profile = SystemProfile(config, scanner=system_scanner)
    prompt_builder.set_system_profile_provider(system_profile)

    async def _on_system_intelligence_profile_refresh(*_a, **_k) -> None:
        system_profile.on_scanner_update()

    bus.on("system_intelligence", _on_system_intelligence_profile_refresh)
    bus.on("system_light_scan", _on_system_intelligence_profile_refresh)
    
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
                    logger.warning("FSWatcher: could not start")
        except Exception:
            logger.warning("FSWatcher init failed", exc_info=True)

    logger.info(
        "JARVIS intelligence initialized: PlatformAdapter(%s) + "
        "SystemScanner + SystemIndexer + MediaWatcher + OwnerUnderstanding"
        + (" + FSWatcher" if fs_watcher else ""),
        platform_adapter.os_type.name,
    )

    def _media_snapshot() -> dict[str, object]:
        info = getattr(media_watcher, "current_media", None)
        if info is None:
            return {
                "playing": False,
                "type": "",
                "source": "",
                "title": "",
                "artist": "",
                "album": "",
                "position": 0.0,
                "duration": 0.0,
                "summary": "No media playing",
            }
        source = str(getattr(info, "app_name", "") or "")
        title = str(getattr(info, "title", "") or "")
        media_type = ""
        if title:
            media_type = "video" if "youtube" in source.lower() else "audio"
        summary = (
            info.summary()
            if hasattr(info, "summary")
            else ("No media playing" if not bool(getattr(info, "is_playing", False)) else title)
        )
        return {
            "playing": bool(getattr(info, "is_playing", False)),
            "type": media_type,
            "source": source,
            "title": title,
            "artist": str(getattr(info, "artist", "") or ""),
            "album": str(getattr(info, "album", "") or ""),
            "position": float(getattr(info, "position", 0.0) or 0.0),
            "duration": float(getattr(info, "duration", 0.0) or 0.0),
            "summary": str(summary or "No media playing"),
        }

    def _current_silicon_stats():
        if silicon_governor is None or not silicon_governor.is_available:
            return None
        try:
            return silicon_governor.get_stats()
        except Exception:
            logger.debug("Silicon stats snapshot failed", exc_info=True)
            return None

    def _runtime_system_snapshot() -> dict[str, object]:
        system_state: dict[str, object] = {}
        try:
            system_state = system_monitor.get_system_state() if system_monitor is not None else {}
        except Exception:
            logger.debug("System monitor snapshot failed", exc_info=True)
            system_state = {}

        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        battery = psutil.sensors_battery()
        battery_pct = float(getattr(battery, "percent", 100.0) or 100.0) if battery is not None else 100.0
        charging = bool(getattr(battery, "power_plugged", False)) if battery is not None else False
        network = "unknown"
        try:
            stats = psutil.net_if_stats()
            network = "online" if any(v.isup for v in stats.values()) else "offline"
        except Exception:
            logger.debug("Network state probe failed", exc_info=True)

        silicon_stats = _current_silicon_stats()
        thermal_pressure = (
            str(getattr(silicon_stats, "thermal_pressure", "unknown") or "unknown")
            if silicon_stats is not None
            else "unknown"
        )

        hardware = get_hardware_profile(silicon_stats=silicon_stats)
        return {
            "cpu": round(float(system_state.get("cpu_percent", 0.0) or 0.0), 1),
            "memory_pct": round(float(system_state.get("ram_percent", 0.0) or 0.0), 1),
            "battery_pct": round(battery_pct, 1),
            "thermal_pressure": thermal_pressure,
            "disk_free_gb": round(disk.free / (1024 ** 3), 1),
            "network": network,
            "charging": charging,
            "ram_used_gb": round((mem.total - mem.available) / (1024 ** 3), 1),
            "ram_total_gb": round(mem.total / (1024 ** 3), 1),
            "ram_gb": round(float(hardware.get("ram_total_gb", 0.0) or 0.0), 1),
            "disk_total_gb": round(disk.total / (1024 ** 3), 1),
            "chip": str(hardware.get("chip") or ""),
            "power_source": str(hardware.get("battery_state") or ""),
            "top_processes": list(system_state.get("active_applications") or [])[:6],
            "hardware": hardware,
        }

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
        if local_brain is not None:
            local_brain.attach_second_brain(second_brain)
        goal_engine = GoalEngine(bus, second_brain, config)
        behavior_model = BehaviorModel(bus, config)
        prediction_engine = PredictionEngine(
            bus, behavior, memory, behavior_model, config,
            brain_mode_manager=brain_mode_mgr,
        )
        prediction_engine.attach_prompt_builder(prompt_builder)
        prediction_engine.attach_cognitive_kernel(cognitive_kernel)
        if prefetch_eng is not None:
            prediction_engine.attach_prefetch_engine(prefetch_eng)
        self_optimizer = SelfOptimizer(
            bus,
            metrics,
            config,
            brain_mode_manager=brain_mode_mgr,
        )
        personality_modes = PersonalityModes(bus, behavior_model, config)

        from core.cognitive.dream_engine import DreamEngine
        from core.cognitive.curiosity_engine import CuriosityEngine

        dream_engine = DreamEngine(bus, config, brain_mode_manager=brain_mode_mgr)
        dream_engine.wire(second_brain=second_brain)
        curiosity_engine = CuriosityEngine(
            bus,
            config,
            brain_mode_manager=brain_mode_mgr,
        )
        if local_brain is not None:
            local_brain.attach_curiosity_engine(curiosity_engine)
        logger.info("Cognitive layer initialized (8 modules, incl. dream + curiosity)")
    else:
        logger.info("Cognitive layer DISABLED via config")

    # ── JARVIS Core (intelligence fusion) ───────────────────────────
    from core.jarvis_core import JarvisCore
    from core.proactive_quota import ProactiveInsightQuota

    proactive_quota = ProactiveInsightQuota(config)

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
        timeline=timeline_memory,
        jarvis=jarvis_core,
        second_brain=second_brain if cognitive_enabled else None,
        prediction_engine=prediction_engine if cognitive_enabled else None,
    )

    real_world_intel = RealWorldIntelligence(config)

    proactive_intel = ProactiveIntelligenceEngine(
        bus=bus,
        config=config,
        brain_mode_manager=brain_mode_mgr,
        proactive_quota=proactive_quota,
    )
    proactive_intel.wire(
        behavior=behavior,
        conv_memory=conv_memory,
        owner=owner_understanding,
        goals=goal_engine,
    )

    jarvis_core.wire_intelligence(
        fusion=context_fusion,
        behavior=behavior,
        prediction=prediction_engine if cognitive_enabled else None,
        conv_memory=conv_memory,
        memory=memory,
        goals=goal_engine if cognitive_enabled else None,
        quota=proactive_quota,
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

    if ui_mode == "native":
        # ── Native macOS UI (AppKit + WKWebView, no browser/server) ──
        from ui.native_ui import NativeATOMWindow
        indicator = NativeATOMWindow(
            mic_name=stt.mic_name,
            config=config,
        )
        logger.info("UI mode: native macOS (AppKit + WKWebView — no browser, no server)")
    elif ui_mode == "web":
        from ui.web_dashboard import WebDashboard

        def _model_display_name(raw_path: str, fallback: str) -> str:
            name = Path(raw_path or fallback).name or fallback
            return name.removesuffix("-mlx").removesuffix(".mlx")

        indicator = WebDashboard(
            mic_name=stt.mic_name,
            port=ui_cfg.get("web_port", 8765),
            auto_open=ui_cfg.get("auto_open_browser", True),
            config=config,
        )
        owner_name = config.get("owner", {}).get("name", "Satyam")
        brain_cfg = config.get("brain", {})
        primary_model = _model_display_name(
            str(brain_cfg.get("mlx_primary_model", "mlx-primary")),
            "mlx-primary",
        )
        fast_model = _model_display_name(
            str(brain_cfg.get("mlx_fast_model", "mlx-fast")),
            "mlx-fast",
        )
        _brain_label = "MLX local brain"
        if brain_enabled and local_brain and local_brain.available:
            _brain_label = f"MLX dual ({fast_model} + {primary_model})"
        elif brain_enabled:
            _brain_label = f"MLX dual loading ({fast_model} + {primary_model})"
        else:
            _brain_label = "No LLM (commands only)"
        _badge_label, _badge_show = deployment_dashboard_badge(config)
        semantic_rag_ready = bool(getattr(cognitive_kernel, "_semantic_stack_available", False))
        memory_rag_status = (
            "Semantic retrieval ready"
            if semantic_rag_ready
            else "Keyword-only fallback. Semantic RAG is unavailable in this runtime."
        )
        _ui = config.get("ui") or {}
        if _ui.get("voice_only_input"):
            voice_mode = "voice_only_display"
            voice_note = (
                "Dashboard is display-only; speak to ATOM — native STT listens continuously when running."
            )
        else:
            voice_mode = "browser_voice_fallback"
            voice_note = (
                "Browser dashboard supports text plus browser-mic fallback. "
                "Launch ATOM.app for the full native macOS voice path."
            )
        if "Faster-Whisper" in stt_runtime_label and not _ui.get("voice_only_input"):
            voice_mode = "browser_voice_dev"
            voice_note = (
                "Browser voice fallback is available. The bundled ATOM.app remains "
                "the production native voice path."
            )
        indicator.set_init_info(
            version="ATOM",
            owner_name=owner_name,
            stt=stt_runtime_label,
            tts=tts_runtime_label,
            brain=_brain_label,
            perf_mode=perf_effective_mode,
            perf_mode_requested=perf_requested_mode,
            brain_profile=brain_mode_mgr.active_profile,
            assistant_mode=assistant_mode_mgr.active,
            deployment_badge_label=_badge_label if _badge_show else "",
            voice_mode=voice_mode,
            voice_note=voice_note,
            memory_rag_status=memory_rag_status,
        )
        web_dashboard = indicator
    else:
        # Fallback: use native macOS UI
        from ui.native_ui import NativeATOMWindow
        indicator = NativeATOMWindow(mic_name=stt.mic_name, config=config)
        logger.info("UI mode fallback: native macOS")

    if hasattr(indicator, "attach_runtime_managers"):
        indicator.attach_runtime_managers(
            brain_mode_mgr, assistant_mode_mgr, router._security,
        )
    indicator = StateAwareIndicator(indicator, atom_runtime)
    atom_runtime.patch_section(
        "voice",
        {
            "stt_engine": stt_runtime_label,
            "tts_engine": tts_runtime_label,
            "mic": getattr(stt, "mic_name", ""),
            "status": "idle",
            "error": stt_runtime_error or None,
            "fallback_chain": stt_runtime_fallbacks,
            "launch_mode": str(os.environ.get("ATOM_LAUNCH_MODE", "") or ""),
            "app_bundle": str(os.environ.get("ATOM_APP_BUNDLE", "") or ""),
            "voice_name": str(
                (config.get("tts") or {}).get("macos_voice")
                or (config.get("tts") or {}).get("edge_voice")
                or (config.get("tts") or {}).get("kokoro_voice")
                or ""
            ),
        },
        source="main.voice_bootstrap",
    )
    atom_runtime.patch_section(
        "mode",
        {
            "requested": perf_requested_mode,
            "effective": perf_effective_mode,
            "reason": initial_mode_reason,
            "profile": brain_mode_mgr.active_profile,
            "assistant_mode": assistant_mode_mgr.active,
            "product_tier": str(
                (config.get("deployment") or {}).get("product_tier", "") or "balanced"
            ),
            "cloud_enabled": bool(config.get("cloud", {}).get("enabled", True)),
        },
        source="main.mode_bootstrap",
    )
    atom_runtime.patch_section(
        "reasoning",
        {
            "why_this_mode": initial_mode_reason,
            "last_decision": f"Boot mode -> {perf_effective_mode}",
        },
        source="main.mode_bootstrap",
    )
    atom_runtime.patch_section(
        "execution",
        {
            "status": "idle",
            "label": "idle",
            "last_intent": "",
            "last_action": "",
        },
        source="main.execution_bootstrap",
    )
    atom_runtime.patch_section(
        "lifecycle",
        {
            "state": state.current.value,
            "label": state.current.value.upper(),
            "status": "Booting",
            "always_listen": bool(state.always_listen),
        },
        source="main.lifecycle_bootstrap",
    )

    def _ui_shutdown_callback():
        """Called from UI thread when X is clicked -- triggers full shutdown."""
        try:
            running_loop.call_soon_threadsafe(bus.emit, "shutdown_requested")
        except Exception:
            logger.debug('main optional step failed', exc_info=True)

    _MODE_LABELS = {
        "optimal": "Optimal",
        "full_performance": "Full Performance",
        "auto": "Auto",
    }
    _MODE_PHRASES = {
        "optimal": "Stable buddy mode tuned for your M5 Air.",
        "full_performance": "Deeper mode when unified memory and thermals are healthy.",
        "auto": "I'll auto-tune between Optimal and Full Performance.",
    }

    def _mode_label(mode_name: str) -> str:
        canonical = _canonical_perf_mode(mode_name)
        return _MODE_LABELS.get(canonical, canonical.replace("_", " ").title())

    def _broadcast_perf_state(reason: str = "") -> None:
        indicator.broadcast_perf_mode(
            perf_effective_mode,
            requested_mode=perf_requested_mode,
            reason=reason,
        )

    _broadcast_perf_state(initial_mode_reason)

    async def _sync_effective_mode(
        target_mode: str,
        *,
        reason: str = "",
        speak: bool = False,
    ) -> bool:
        nonlocal perf_effective_mode
        canonical = BrainModeManager.canonical_profile_name(target_mode) or "optimal"
        previous = perf_effective_mode
        if canonical == previous and brain_mode_mgr.active_profile == canonical:
            _broadcast_perf_state(reason)
            return False

        ok, _ = brain_mode_mgr.set_profile(canonical)
        if not ok:
            return False

        perf_effective_mode = brain_mode_mgr.active_profile
        bus.emit_fast(
            "runtime_settings_changed",
            brain_profile=brain_mode_mgr.active_profile,
        )
        _broadcast_perf_state(reason)

        if speak and previous != perf_effective_mode:
            msg = (
                f"Boss, switching from {_mode_label(previous)} to "
                f"{_mode_label(perf_effective_mode)}."
            )
            if reason:
                msg = f"{msg} {reason}"
            bus.emit_long(
                "partial_response",
                text=msg,
                is_first=True,
                is_last=True,
            )
        return previous != perf_effective_mode

    async def _execute_mode_switch(new_mode: str) -> None:
        """Save config, speak confirmation, then trigger graceful restart."""
        nonlocal perf_requested_mode
        global _restart_requested
        try:
            requested = _canonical_perf_mode(new_mode)
            cfg_path = Path("config/settings.json")
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
            if "performance" not in cfg_data:
                cfg_data["performance"] = {}
            cfg_data["performance"]["mode"] = requested
            cfg_data.setdefault("assistant_brain", {})
            cfg_data["assistant_brain"]["active_profile"] = (
                "optimal" if requested == "auto" else requested
            )
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg_data, f, indent=4)
            perf_requested_mode = requested
            logger.info("Performance mode updated to '%s' in settings.json", requested)

            label = _mode_label(requested)
            phrase = _MODE_PHRASES.get(requested, "")
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

    async def _on_bus_set_mode(mode: str = "optimal", **_kw) -> None:
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
        # Wired only when the user opted into the VLM (vision.vlm.enabled).
        # Letting cold_start own the load means the first wake-word fire
        # hits the hot path (~1.7s) instead of paying the ~2s cold-load
        # latency on the wake-word executor thread.
        vlm_captioner=_captioner,
        # Pre-classify every skill expansion target so the second
        # intent-engine pass after a skill match (atom_log.txt
        # L597-599) lands on a hot cache instead of paying ~150 ms.
        skills_registry=skills_reg,
    )

    await tts.init_voice()
    stt_preload_done = asyncio.Event()
    _bg_tasks: list[asyncio.Task] = []

    async def _background_stt_preload() -> None:
        t0 = time.monotonic()
        logger.info("STT model loading in background...")

        try:
            audio_intel = voice_pipeline.audio_intelligence
            if audio_intel is not None and config.get("audio_intelligence", {}).get("enabled", True):
                best = await audio_intel.boot()
                if best:
                    logger.info(
                        "Audio Intelligence: '%s' (%s, score=%.3f)",
                        best.name, best.device_type, best.quality_score,
                    )
                    if audio_intel.input_output_hardware_mismatch and hasattr(stt, "_native_voice_processing"):
                        stt._native_voice_processing = False
                        logger.info(
                            "Auto-disabled Voice Processing I/O — input (%s) and output (%s) "
                            "are on different hardware (VPIO echo cancellation causes silence)",
                            best.device_type,
                            audio_intel._selected_output.device_type if audio_intel._selected_output else "?",
                        )
                    if hasattr(stt, "set_audio_device_rebinder") and audio_intel._selected_input:
                        sel = audio_intel._selected_input
                        def _rebind() -> bool:
                            return audio_intel.apply_system_default(sel)
                        stt.set_audio_device_rebinder(_rebind, sel.name)
                        logger.info("STT device rebinder wired for '%s' (CoreAudio ID %d)", sel.name, sel.core_audio_id)
                    if hasattr(stt, "preflight_mic_check"):
                        loop = asyncio.get_running_loop()
                        mic_ok = await loop.run_in_executor(None, stt.preflight_mic_check)
                        if not mic_ok:
                            logger.warning(
                                "Pre-flight mic check: '%s' returned dead signal — "
                                "AVAudioEngine may still work (continuing)",
                                best.name,
                            )
                else:
                    logger.warning("Audio Intelligence: no suitable device found, falling back to MicManager")
                    loop = asyncio.get_running_loop()
                    devices = await loop.run_in_executor(None, mic_manager.profile_devices)
                    if devices:
                        fallback = mic_manager.get_best_device(
                            prefer_bluetooth=config.get("mic", {}).get("prefer_bluetooth", True),
                        )
                        if fallback:
                            mic_manager.active_device = fallback
            else:
                loop = asyncio.get_running_loop()
                devices = await loop.run_in_executor(None, mic_manager.profile_devices)
                if devices:
                    best_legacy = mic_manager.get_best_device(
                        prefer_bluetooth=config.get("mic", {}).get("prefer_bluetooth", True),
                    )
                    if best_legacy:
                        mic_manager.active_device = best_legacy
                        logger.info(
                            "Audio device selected (legacy): '%s' (%s, quality=%d/100)",
                            best_legacy.name, best_legacy.device_type, best_legacy.quality_score,
                        )

            if hasattr(stt, "async_preload"):
                await stt.async_preload()
            else:
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
                            logger.debug('Async task spawn failed', exc_info=True)
                    m = feedback_engine.compute_accuracy_metrics()
                    pre = get_last_preemption_score()
                    ap2 = None
                    try:
                        if local_brain is not None and getattr(
                            local_brain, "_memory_graph", None,
                        ):
                            ap2 = local_brain._memory_graph.get_last_active_project()
                    except Exception:
                        logger.debug('Memory graph project lookup failed', exc_info=True)
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

    # Periodic LLM perf snapshot. The per-turn ``MLX [profile/role]: …``
    # line is verbose; what's missing for diagnosing "is ATOM healthy?" is
    # a single rolling summary line every minute showing the lifetime
    # average decode rate, prompt-cache hit rate, and any active thermal
    # clamp. Cheap (no extra model work, just reads counters), runs only
    # when the local brain is enabled, and goes silent when no turns have
    # happened yet so an idle ATOM doesn't spam the log.
    if brain_enabled and local_brain is not None:
        async def _llm_perf_snapshot() -> None:
            await asyncio.sleep(60.0)  # first sample after warm-up
            while True:
                try:
                    snap = local_brain.get_perf_snapshot()
                    if snap and int(snap.get("turns", 0) or 0) > 0:
                        cache = snap.get("cache", {}) or {}
                        clamp = float(snap.get("thermal_clamp_ratio", 1.0) or 1.0)
                        clamp_label = (
                            f" thermal_clamp={clamp:.2f}x"
                            if clamp < 0.999
                            else ""
                        )
                        logger.info(
                            "LLM perf: turns=%d tokens=%d avg=%.1f tok/s "
                            "avg_ms=%.0f peak=%.2fGB cache=%d/%d (%.0f%%)%s",
                            int(snap.get("turns", 0) or 0),
                            int(snap.get("tokens", 0) or 0),
                            float(snap.get("avg_tok_s", 0.0) or 0.0),
                            float(snap.get("avg_ms", 0.0) or 0.0),
                            float(snap.get("peak_memory_gb", 0.0) or 0.0),
                            int(cache.get("hits", 0) or 0),
                            int(cache.get("hits", 0) or 0) + int(cache.get("misses", 0) or 0),
                            float(cache.get("hit_rate", 0.0) or 0.0) * 100.0,
                            clamp_label,
                        )
                except Exception:
                    logger.debug("LLM perf snapshot failed", exc_info=True)
                await asyncio.sleep(60.0)
        _bg_tasks.append(asyncio.create_task(_llm_perf_snapshot()))

    # ── iPhone Shortcuts bridge (Phase 1 cross_device) ─────────────
    # The bridge code (core/cross_device/iphone_bridge.py) shipped in
    # an earlier sprint but was never plugged into the boot sequence.
    # We wire it here, gated on ``cross_device.enabled`` so the
    # default-off behaviour remains: no token, no listener, no event
    # bus chatter unless the owner opts in via settings.json.
    iphone_bridge = None
    identity_engine = None
    _cross_cfg = config.get("cross_device", {}) or {}
    if _cross_cfg.get("enabled", False):
        try:
            from core.identity_engine import IdentityEngine
            from core.cross_device.iphone_bridge import IPhoneBridge
            from core.cross_device.bridge_event_wiring import wire_bridge_events

            identity_engine = IdentityEngine(config=config)
            router.attach_identity_engine(identity_engine)

            # ``IPhoneBridge.__init__`` auto-generates a ~32-char hex
            # token at ``config/bridge_token`` if one doesn't exist
            # yet, so we never need a manual setup script.
            iphone_bridge = IPhoneBridge(
                config=config,
                emit=getattr(bus, "emit_fast", None) or getattr(bus, "emit", None),
                atom_root=Path(__file__).resolve().parent,
            )

            async def _speak_iphone_hint(hint: str) -> None:
                # ProactiveAwareness hints land here; we echo them as a
                # short partial_response so they surface through the
                # same TTS path as everything else (and respect the
                # state-machine SPEAKING transition).
                try:
                    bus.emit_long("partial_response", text=hint, is_first=True, is_last=True)
                except Exception:
                    logger.debug("iphone hint emit failed", exc_info=True)

            wire_bridge_events(
                bus=bus,
                identity_engine=identity_engine,
                proactive=proactive,
                speak=_speak_iphone_hint,
            )

            async def _start_iphone_bridge() -> None:
                ok = await iphone_bridge.start()
                if not ok:
                    logger.warning(
                        "iPhone bridge failed to start; cross_device events disabled."
                    )
                    return
                port = iphone_bridge.actual_port or _cross_cfg.get("bridge_port", 8787)
                tok = iphone_bridge.token
                # Loud, single-shot setup banner. The token must be
                # entered into the iPhone Shortcuts (X-ATOM-Token
                # header) on first run; we surface it here so the user
                # never has to grep config/bridge_token by hand.
                _here = Path(__file__).resolve().parent
                logger.info("")
                logger.info("┌──────────────────────────────────────────────────────────────────────")
                logger.info("│  iPhone bridge ONLINE  ->  http://127.0.0.1:%d", port)
                logger.info("│  Token: %s", tok)
                logger.info("│  (also stored at %s/config/bridge_token)", _here)
                logger.info("│")
                logger.info("│  Next step on iPhone (one-time):")
                logger.info("│   1. Open the Shortcuts app on your iPhone 15.")
                logger.info("│   2. Create a Shortcut that does:")
                logger.info("│        Get Contents of URL")
                logger.info("│          URL:    http://<your-mac-ip>:%d/health", port)
                logger.info("│          Method: GET")
                logger.info("│          Headers: X-ATOM-Token = <paste token above>")
                logger.info("│   3. Run it once -- success registers your iPhone as the")
                logger.info("│      trusted device. Then add /faceid, /presence, /trigger.")
                logger.info("│")
                logger.info("│  See docs (or ask ATOM 'how do I connect my iPhone?')")
                logger.info("└──────────────────────────────────────────────────────────────────────")
                logger.info("")

            _bg_tasks.append(asyncio.create_task(_start_iphone_bridge()))
        except Exception:
            logger.warning(
                "iPhone bridge wiring failed; cross_device disabled this boot.",
                exc_info=True,
            )
            iphone_bridge = None
    else:
        logger.info(
            "iPhone bridge disabled (cross_device.enabled=false). "
            "Flip it to true in config/settings.json and restart to enable."
        )

    runtime_watchdog = RuntimeWatchdog(bus, state, config)
    runtime_watchdog.attach_local_brain(local_brain)
    runtime_watchdog.attach_tts(tts)
    router.attach_runtime_watchdog(runtime_watchdog)
    local_brain.attach_runtime_watchdog(runtime_watchdog)
    bus.on("state_changed", runtime_watchdog.on_state_changed)

    try:
        from core.proactive.routine_engine import RoutineEngine
        from core.intent_engine import routine_intents as _routine_intents
        routine_engine = RoutineEngine(config)
        router.attach_routine_engine(routine_engine)
        _routine_intents.set_routine_engine(routine_engine)
    except Exception:
        logger.info("Routine engine wiring failed", exc_info=True)

    try:
        _is_echo = getattr(tts, "is_echo", None)
        if callable(_is_echo):
            router.attach_tts_echo_guard(
                lambda _t: bool(_is_echo(_t, window_s=12.0)),
            )
    except Exception:
        logger.info("Router TTS echo guard wiring failed", exc_info=True)

    # ── System State Engine: real-time awareness ───────────────────
    from core.system_state_engine import SystemStateEngine
    system_state_engine = SystemStateEngine(bus, poll_interval_s=0.5)

    # ── Session Memory: short-term command history ───────────────
    from core.memory.session_memory import SessionMemory
    session_memory = SessionMemory(capacity=20)

    # ── Parallel Pipeline: overlap STT partial with intent pre-classification ──
    parallel_pipeline = ParallelPipeline(intent_engine, cache)
    bus.on("speech_partial", parallel_pipeline.on_speech_partial)

    # ── Ack Engine: instant acknowledgement before processing ──────
    from voice.ack_engine import AckEngine
    ack_engine = AckEngine()

    # ── Pipeline Metrics: aggregate voice timing diagnostics ─────
    from voice.pipeline_budget import VoicePipelineMetrics
    pipeline_metrics = VoicePipelineMetrics()

    # ── CommandLoop: single-command controller wrapping Router ──────
    from core.command_loop import CommandLoop
    command_loop = CommandLoop(
        bus, state, router,
        system_state_engine=system_state_engine,
        session_memory=session_memory,
    )
    command_loop.attach_ack_engine(ack_engine)
    command_loop.attach_pipeline_metrics(pipeline_metrics)

    # ── Wire context layer (system state + session + user memory) into Router
    router.attach_context_layer(
        system_state_engine=system_state_engine,
        session_memory=session_memory,
        user_memory=user_memory,
    )

    # ── Suggestion Engine: inline post-command follow-ups ──────────
    from core.suggestion_engine import SuggestionEngine
    suggestion_engine = SuggestionEngine()
    command_loop.attach_suggestion_engine(suggestion_engine)

    # ── Proactive idle-gate: don't let "Boss, new file landed..."
    #     interrupt mid-turn. Wire state + command_loop so insights
    #     get buffered while ATOM is thinking/speaking, then drained
    #     on the next listening transition. (atom_log.txt L308 fix.)
    try:
        proactive_intel.attach_idle_gate(state, command_loop)

        async def _drain_proactive_on_listening(
            *, old: object | None = None, new: object | None = None, **_kw: object
        ) -> None:
            new_state = getattr(new, "value", new)
            new_state = str(new_state or "").lower()
            if new_state in ("listening", "idle"):
                drained = proactive_intel.drain_pending()
                if drained:
                    logger.debug("Proactive: drained %d deferred insights", drained)

        bus.on("state_changed", _drain_proactive_on_listening)
    except Exception as exc:
        logger.warning("Proactive idle-gate wiring failed: %s", exc)

    # ── Task Manager: centralized background task tracking ─────────
    from core.task_manager import TaskManager
    task_manager = TaskManager()
    router.attach_task_manager(task_manager)

    # ── Wire all event handlers (extracted for testability) ────────
    _wiring_ctx = wire_events(
        bus=bus, state=state, state_bridge=atom_runtime, shutdown_event=shutdown_event, stt=stt, tts=tts, router=router,
        indicator=indicator, cache=cache, memory=memory, metrics=metrics,
        config=config, local_brain=local_brain, llm_queue=llm_queue,
        assistant_mode_mgr=assistant_mode_mgr,
        behavior=behavior,
        scheduler=scheduler, process_mgr=process_mgr, evolution=evolution,
        priority_sched=priority_sched,
        v3=False, v4=False,
        command_loop=command_loop,
    )
    _last_perceived_ms = _wiring_ctx["last_perceived_ms"]

    # ── Phase G: cognitive loop ──────────────────────────────────
    # Wires the always-on subsystems that turn ATOM from a chatbox
    # into a continuous OS:
    #   - turn_complete emitter on CommandLoop
    #   - ReflectiveLoop (post-TTS think pass)
    #   - PresenceSampler (camera-driven presence ticks)
    #   - SceneContextEngine (VLM captions on scene change)
    #   - MoodInferenceEngine (rules-based mood fusion)
    #   - JarvisSuggester (cadence-gated proactive nudges)
    # Each is independently togglable via config["cognitive_loop"].
    cognitive_handles = wire_cognitive_loop(
        bus=bus,
        state=state,
        command_loop=command_loop,
        config=config,
        local_brain=local_brain,
        vision_engine=vision_engine,
        captioner=_captioner,
    )
    logger.info(
        "Cognitive loop ready: %s",
        cognitive_handles.enabled_summary,
    )

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
        nonlocal perf_effective_mode
        if brain_profile:
            perf_effective_mode = (
                BrainModeManager.canonical_profile_name(brain_profile) or perf_effective_mode
            )
            _broadcast_perf_state()
        if web_dashboard is not None:
            await web_dashboard.broadcast_runtime_settings(
                brain_profile=brain_profile or brain_mode_mgr.active_profile,
                assistant_mode=assistant_mode or assistant_mode_mgr.active,
            )
        atom_runtime.patch_section(
            "mode",
            {
                "profile": brain_profile or brain_mode_mgr.active_profile,
                "assistant_mode": assistant_mode or assistant_mode_mgr.active,
                "effective": perf_effective_mode,
                "requested": perf_requested_mode,
            },
            source="main.runtime_settings_changed",
        )

    bus.on("runtime_settings_changed", _on_runtime_settings_changed)

    _latest_readiness_report: dict[str, object] = {}

    def _format_readiness_summary(readiness: dict[str, object]) -> str:
        if not readiness:
            return ""
        summary = readiness.get("summary", {})
        summary_info = summary if isinstance(summary, dict) else {}
        subsystems = readiness.get("subsystems", {})
        subsystem_info = subsystems if isinstance(subsystems, dict) else {}
        lines: list[str] = []
        if bool(readiness.get("overall_ready")):
            lines.append("All systems operational, Boss.")
        else:
            failures = int(summary_info.get("failures", 0) or 0)
            lines.append(
                f"{failures} system{'s' if failures != 1 else ''} need attention."
            )
        for name, info in subsystem_info.items():
            if not isinstance(info, dict):
                continue
            if info.get("status") == "pass":
                continue
            icon = {"warn": "WARNING", "fail": "ISSUE"}.get(str(info.get("status")), "?")
            lines.append(f"  {name}: [{icon}] {str(info.get('detail', ''))}")
        passed = int(summary_info.get("passed", 0) or 0)
        warnings_count = int(summary_info.get("warnings", 0) or 0)
        failures = int(summary_info.get("failures", 0) or 0)
        total = len(subsystem_info)
        if total and passed == total:
            lines.append(f"All {passed} subsystems passed diagnostics.")
        else:
            lines.append(
                f"{passed} passed, {warnings_count} warnings, {failures} failures."
            )
        return "\n".join(lines)

    def _voice_permissions_snapshot() -> dict[str, str]:
        return {
            "speech": str(getattr(stt, "speech_permission_status", "unknown") or "unknown"),
            "microphone": str(
                getattr(stt, "microphone_permission_status", "unknown") or "unknown"
            ),
        }

    def _publish_self_check_report(
        report: dict[str, object] | None,
        *,
        source: str = "main.self_check",
    ) -> dict[str, object]:
        payload = dict(report or {})
        if not payload:
            return {}

        warnings = [str(w) for w in payload.get("warnings", [])]
        health_status = "ok" if not warnings else "degraded"
        atom_runtime.replace_health_report(
            self_check=payload,
            score=float(payload.get("health_score", 0.0) or 0.0),
            warnings=warnings,
            status=health_status,
            source=source,
        )

        context_payload = payload.get("context", {})
        if isinstance(context_payload, dict) and context_payload:
            atom_runtime.patch_section("context", context_payload, source=source)

        voice_payload = payload.get("voice", {})
        if isinstance(voice_payload, dict) and voice_payload:
            atom_runtime.patch_section(
                "voice",
                {
                    "stt_engine": str(voice_payload.get("stt_engine") or ""),
                    "tts_engine": str(voice_payload.get("tts_engine") or ""),
                    "mic": str(voice_payload.get("mic") or ""),
                    "error": voice_payload.get("error"),
                },
                source=source,
            )

        mode_payload = payload.get("mode", {})
        if isinstance(mode_payload, dict) and mode_payload:
            atom_runtime.patch_section("mode", mode_payload, source=source)

        atom_runtime.patch_section(
            "reasoning",
            {
                "why_this_mode": str(mode_payload.get("reason") or ""),
                "last_report": str(payload.get("summary_text") or ""),
                "last_decision": str(payload.get("recommendation") or ""),
                "severity": "warning" if warnings else "info",
            },
            source=source,
        )
        atom_runtime.emit_snapshot(source=source)
        return payload

    router.configure_diagnostics(
        stt=stt,
        tts=tts,
        metrics=metrics,
        local_brain=local_brain,
        health_monitor=health_monitor,
        state_snapshot_provider=atom_runtime.snapshot,
        report_publisher=_publish_self_check_report,
        audio_intel=voice_pipeline.audio_intelligence,
    )

    async def _on_atom_readiness(report: dict[str, object] | None = None, **_kw) -> None:
        nonlocal _latest_readiness_report
        readiness = dict(report or {})
        if readiness:
            _latest_readiness_report = readiness
        summary = _format_readiness_summary(readiness)
        summary_info = readiness.get("summary", {}) if isinstance(readiness, dict) else {}
        warnings = []
        if isinstance(readiness, dict):
            subsystems = readiness.get("subsystems", {})
            if isinstance(subsystems, dict):
                warnings = [
                    f"{name}: {str(info.get('detail', ''))}"
                    for name, info in subsystems.items()
                    if isinstance(info, dict) and info.get("status") in {"warn", "fail"}
                ][:10]
        failures = int(summary_info.get("failures", 0) or 0) if isinstance(summary_info, dict) else 0
        atom_runtime.replace_health_report(
            readiness=readiness,
            score=10.0 if failures == 0 else max(2.0, 10.0 - (failures * 2.0)),
            warnings=warnings,
            status="ok" if failures == 0 else "degraded",
            readiness_summary=summary,
            scan_summary=system_scanner.get_scan_summary(),
            source="main.atom_readiness",
        )

    async def _on_context_snapshot(**kw) -> None:
        media = _media_snapshot()
        context_patch = {
            "active_app": str(kw.get("active_app") or ""),
            "window_title": str(kw.get("window_title") or ""),
            "activity_type": str(kw.get("activity_type") or "idle"),
            "confidence": float(kw.get("confidence") or 0.0),
            "time_of_day": str(kw.get("time_of_day") or ""),
            "idle_minutes": float(kw.get("idle_minutes") or 0.0),
            "is_weekday": bool(kw.get("is_weekday", True)),
            "weekday": int(kw.get("weekday") or 0),
            "frontmost_pid": int(kw.get("frontmost_pid") or 0),
            "media": media,
        }
        atom_runtime.patch_section("context", context_patch, source="main.context_snapshot")

    bus.on("atom_readiness", _on_atom_readiness)
    bus.on("context_snapshot", _on_context_snapshot)

    async def _sync_atom_world_state() -> None:
        while not shutdown_event.is_set():
            try:
                media = _media_snapshot()
                now = datetime.now()
                runtime_ctx = context_engine.get_runtime_snapshot(
                    idle_minutes=health_monitor.idle_minutes,
                    media=media,
                )
                system_snapshot = _runtime_system_snapshot()
                llm_queue_pending = bool(llm_queue is not None and llm_queue.has_pending())
                scheduler_depth = int(getattr(priority_sched, "queue_depth", 0) or 0)
                voice_error = getattr(stt, "_last_error", None) or stt_runtime_error or None
                atom_runtime.patch_section(
                    "system",
                    system_snapshot,
                    source="main.world_sync.system",
                )
                atom_runtime.patch_section(
                    "context",
                    {
                        **runtime_ctx,
                        "time_of_day": (
                            "morning"
                            if 5 <= now.hour < 12
                            else "afternoon"
                            if 12 <= now.hour < 17
                            else "evening"
                            if 17 <= now.hour < 21
                            else "night"
                        ),
                        "idle_minutes": float(health_monitor.idle_minutes),
                        "is_weekday": now.weekday() < 5,
                        "weekday": now.weekday(),
                        "media": media,
                    },
                    source="main.world_sync.context",
                )
                atom_runtime.patch_section(
                    "voice",
                    {
                        "mic": str(getattr(stt, "mic_name", "") or ""),
                        "stt_engine": str(
                            getattr(stt, "backend_name", None) or stt_runtime_label
                        ),
                        "error": voice_error,
                        "fallback_chain": list(
                            getattr(stt, "fallback_chain", None) or stt_runtime_fallbacks
                        ),
                        "permissions": _voice_permissions_snapshot(),
                        "listening": state.current.value == "listening",
                        "speaking": state.current.value == "speaking",
                        "launch_mode": str(os.environ.get("ATOM_LAUNCH_MODE", "")),
                        "app_bundle": str(os.environ.get("ATOM_APP_BUNDLE", "") or ""),
                        "perceived_latency_ms": _last_perceived_ms.get("ms"),
                    },
                    source="main.world_sync.voice",
                )
                atom_runtime.patch_section(
                    "mode",
                    {
                        "product_tier": str(
                            (config.get("deployment") or {}).get("product_tier", "")
                            or "balanced"
                        ),
                        "cloud_enabled": bool(
                            config.get("cloud", {}).get("enabled", True)
                        ),
                    },
                    source="main.world_sync.mode",
                )
                atom_runtime.patch_section(
                    "execution",
                    {
                        "status": (
                            "running"
                            if state.current.value in {"thinking", "speaking"}
                            else "idle"
                        ),
                        "active_task": str(
                            atom_runtime.store.get_section("execution").get("label", "") or ""
                        ),
                        "queue_depth": int(llm_queue_pending) + scheduler_depth,
                        "scheduler_queue_depth": scheduler_depth,
                        "llm_queue_pending": llm_queue_pending,
                        "cache_entries": int(len(getattr(cache, "_store", {}))),
                        "cache_max": int(getattr(cache, "_max_size", 0) or 0),
                    },
                    source="main.world_sync.execution",
                )
                if _latest_readiness_report:
                    atom_runtime.replace_health_report(
                        readiness=_latest_readiness_report,
                        source="main.world_sync.readiness",
                    )
            except Exception:
                logger.debug("World-state sync failed", exc_info=True)
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                continue

    _bg_tasks.append(asyncio.create_task(_sync_atom_world_state()))

    if web_dashboard is not None:
        def _can_restart_stt() -> bool:
            stt_engine = str(getattr(stt, "backend_name", "") or "").strip().lower()
            mic_name = str(getattr(stt, "mic_name", "") or "").strip().lower()
            return (
                callable(getattr(stt, "async_start_listening", None))
                or callable(getattr(stt, "start_listening", None))
            ) and stt_engine not in {"", "disabled", "unavailable"} and mic_name != "voice input unavailable"

        async def _dashboard_unstick() -> None:
            """Dashboard UNSTICK: exit THINKING, stop TTS, clear ERROR_RECOVERY."""
            bus.emit("resume_listening")
            if _can_restart_stt():
                await asyncio.sleep(0.05)
                bus.emit("restart_listening")

        web_dashboard.set_unstick_callback(_dashboard_unstick)

        async def _dashboard_stop_task() -> None:
            if local_brain is not None and hasattr(local_brain, "request_preempt"):
                local_brain.request_preempt()
            if llm_queue is not None:
                await llm_queue.clear_pending()
            try:
                if hasattr(tts, "stop"):
                    await tts.stop()
            except Exception:
                logger.debug("Dashboard stop_task TTS stop failed", exc_info=True)
            bus.emit_fast(
                "resume_listening",
                source="dashboard_stop_task",
                reason="stop_task",
                user_interrupt=True,
            )
            if _can_restart_stt():
                await asyncio.sleep(0.05)
                bus.emit("restart_listening")
            atom_runtime.patch_section(
                "execution",
                {
                    "active_task": "",
                    "status": "idle",
                    "label": "idle",
                    "llm_queue_pending": False,
                },
                source="main.dashboard_stop_task",
            )
            atom_runtime.patch_section(
                "reasoning",
                {
                    "last_decision": "Stopped the active task from the dashboard.",
                    "severity": "attention",
                },
                source="main.dashboard_stop_task",
            )

        if hasattr(web_dashboard, "set_stop_task_callback"):
            web_dashboard.set_stop_task_callback(_dashboard_stop_task)

        _voice_only_ui = bool((config.get("ui") or {}).get("voice_only_input"))

        async def _on_text_input(text: str) -> None:
            """Handle typed input from dashboard — same as speech_final (optional; voice-only disables)."""
            if _voice_only_ui:
                logger.debug("Ignoring dashboard text_input (ui.voice_only_input=true): %s", (text or "")[:48])
                return
            logger.info("Text input from dashboard: '%s'", text[:60])
            bus.emit("speech_final", text=text)

        web_dashboard.set_text_input_callback(_on_text_input)
        if hasattr(web_dashboard, "attach_atom_state"):
            web_dashboard.attach_atom_state(atom_runtime)
        if hasattr(web_dashboard, "on_state_diff"):
            bus.on("state.diff", web_dashboard.on_state_diff)
        if hasattr(web_dashboard, "on_state_snapshot"):
            bus.on("state.snapshot", web_dashboard.on_state_snapshot)

        async def _run_self_check_from_dashboard() -> dict[str, object]:
            return router._diagnostics.self_check_report()

        if hasattr(web_dashboard, "set_self_check_callback"):
            web_dashboard.set_self_check_callback(_run_self_check_from_dashboard)

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
                logger.debug('Memory graph project lookup failed', exc_info=True)
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
            try:
                from core.observability.per_module_latency import get_latency_board

                lb = get_latency_board().get_dashboard_data()
                lb["system_state"] = ss
            except Exception:
                lb = {"system_state": ss, "modules": {}, "recent_events": [], "health": "idle"}
            return {
                "health_status": health,
                "metrics": metrics,
                "warnings": warns,
                "snapshot": snap,
                "latency_board": lb,
            }

        web_dashboard.set_v7_health_provider(_v7_health_payload)

        try:
            from core.observability.health_snapshot import HealthSnapshotBuilder
            _embedding_engine = None
            try:
                if local_brain is not None:
                    _embedding_engine = getattr(local_brain, "_embedding_engine", None)
            except Exception:
                _embedding_engine = None
            health_builder = HealthSnapshotBuilder(
                bus=bus,
                state=state,
                stt=stt,
                tts=tts,
                local_brain=local_brain,
                embedding_engine=_embedding_engine,
                semantic_cache=semantic_cache,
                memory=memory,
                silicon_governor=silicon_governor,
                health_monitor=health_monitor,
                error_monitor=_error_monitor,
                mic_manager=mic_manager,
            )
            web_dashboard.set_health_provider(health_builder.build)
        except Exception:
            logger.info("Health snapshot wiring failed", exc_info=True)

        await web_dashboard.start()
    else:
        indicator.start()
    _broadcast_perf_state()

    if local_brain and local_brain.available:
        brain_cfg = config.get("brain", {})
        # Resolution order matches brain.mlx_llm._resolve_model_path so the
        # boot banner reflects the model the runtime actually loaded. The
        # ``mlx_model`` key is the post-consolidation canonical name; the
        # legacy ``primary``/``fast``/``model_path`` keys remain for
        # back-compat with older settings.json files.
        model_raw = (
            brain_cfg.get("mlx_model")
            or brain_cfg.get("mlx_primary_model")
            or brain_cfg.get("mlx_fast_model")
            or brain_cfg.get("model_path")
            or "mlx"
        )
        # Use ``Path.name`` (last path segment) instead of ``Path.stem`` —
        # the legacy model directory ``qwen2.5-7b-instruct-4bit`` would
        # otherwise be truncated to "qwen2" because ``stem`` treats every
        # dot after the first as an extension boundary. The current
        # ``qwen3-4b-instruct-4bit`` directory has no dots so it's safe
        # either way, but keeping ``.name`` future-proofs the path.
        model_name = Path(model_raw).name.replace("-mlx", "")
        brain_label = f"Intent Engine + Agentic MLX LLM ({model_name})"
    elif brain_enabled:
        brain_label = "Intent Engine + MLX LLM (model unavailable)"
    else:
        brain_label = "Intent Engine ONLY — set brain.enabled for local LLM"
    cognitive_label = "Cognitive Layer ON (dream+curiosity)" if cognitive_enabled else "Cognitive OFF"
    logger.info("=== ATOM (Supernatural Intelligence OS) | Owner: Satyam | Mic: %s | %s | %s ===",
                stt.mic_name, brain_label, cognitive_label)
    if not brain_enabled:
        logger.warning("brain.enabled is false — voice Q&A disabled; commands still work")

    # Highly-visible remediation banner when STT could not initialise (most
    # commonly because the user launched ``python main.py`` directly instead
    # of the bundle wrapper, so SFSpeechRecognizer can't see the
    # NSSpeechRecognitionUsageDescription in Info.plist). The earlier
    # WARN/ERROR lines about "_DisabledSTT" scroll past in 200+ boot lines;
    # this final block is the one log message that's almost impossible to
    # miss when looking at a fresh boot.
    _stt_label = getattr(stt, "mic_name", "")
    _stt_disabled = (
        not _stt_label
        or _stt_label.lower() in {"disabled", "voice input unavailable"}
        or "unavailable" in _stt_label.lower()
    )
    if _stt_disabled:
        _here = Path(__file__).resolve().parent
        _bundle_cmd = f'cd "{_here}" && ./Run\\ ATOM.command'
        logger.warning("")
        logger.warning("┌──────────────────────────────────────────────────────────────────────")
        logger.warning("│  VOICE INPUT IS DISABLED — ATOM cannot hear you.")
        logger.warning("│")
        logger.warning("│  Cause: SFSpeechRecognizer needs to run inside ATOM.app so macOS")
        logger.warning("│         can grant the bundle's NSSpeechRecognitionUsageDescription /")
        logger.warning("│         Microphone TCC entitlements. Running 'python main.py' from")
        logger.warning("│         a venv directly bypasses the bundle and the mic stays muted.")
        logger.warning("│")
        logger.warning("│  Fix:   Quit this process (Ctrl-C), then double-click")
        logger.warning("│         '%s/Run ATOM.command' in Finder, or run:", _here)
        logger.warning("│           %s", _bundle_cmd)
        logger.warning("│")
        logger.warning("│  TTS, dashboard chat (http://127.0.0.1:8765/), and tool execution")
        logger.warning("│  still work in this mode — only the always-on voice loop is offline.")
        logger.warning("└──────────────────────────────────────────────────────────────────────")
        logger.warning("")

    # Hotkey support removed — requires root/sudo on macOS.
    # Use the UNSTICK button in the dashboard instead.
    hotkey_active = False

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
    system_state_engine.start()
    await voice_pipeline.start_voice_loop(running_loop)

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

    # ── Morning briefing service (Sprint D1) ─────────────────────────
    morning_briefing = None
    try:
        from core.proactive.morning_briefing import MorningBriefingService
        morning_briefing = MorningBriefingService(
            config,
            bus=bus,
            real_world_intel=real_world_intel,
        )
        if morning_briefing.enabled:
            logger.info(
                "Morning briefing service armed (window %s, last=%s)",
                morning_briefing.diagnostics().get("wake_window"),
                morning_briefing.last_briefed_date or "never",
            )

            async def _on_speech_final_for_briefing(**_kw: Any) -> None:
                if morning_briefing is None:
                    return
                try:
                    await morning_briefing.maybe_trigger("speech_final")
                except Exception:
                    logger.info("morning briefing trigger failed", exc_info=True)

            bus.on("speech_final", _on_speech_final_for_briefing)

            async def _maybe_startup_briefing() -> None:
                if morning_briefing is None:
                    return
                try:
                    await asyncio.sleep(3.0)
                    await morning_briefing.maybe_trigger("startup")
                except Exception:
                    logger.info("morning briefing startup trigger failed", exc_info=True)

            asyncio.create_task(_maybe_startup_briefing())
    except Exception:
        logger.info("Morning briefing wiring failed", exc_info=True)
        morning_briefing = None

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
        context_engine=context_engine, stt=stt,
    )
    wire_real_world(
        bus=bus, real_world_intel=real_world_intel,
        context_fusion=context_fusion,
    )
    try:
        router.attach_real_world_intel(real_world_intel)
    except Exception as exc:
        logger.warning("Router real-world wiring failed: %s", exc)
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

    # ── Boot-time corpus ingestion ───────────────────────────────────
    # Walk owner-configured directories so personal knowledge (notes,
    # repo READMEs, ~/Documents) is queryable on day one. Runs ONCE
    # per boot, in the background, gated behind a 60s warmup so the
    # embedding engine + vector store finish settling before we start
    # batching. Already-ingested files are skipped via the persistent
    # dedupe cache, so this is essentially a no-op on subsequent boots
    # unless the owner edited a file or dropped a new one.
    _doc_cfg = (config.get("documents") or {})
    if _doc_cfg.get("auto_ingest_on_boot", False) and document_engine is not None:
        _ingest_paths = list(_doc_cfg.get("auto_ingest_paths") or [])
        if _ingest_paths:
            async def _auto_ingest_corpus() -> None:
                try:
                    await asyncio.sleep(60.0)
                    if not document_engine.is_ready:
                        logger.info(
                            "Auto-ingest skipped: document_engine not ready",
                        )
                        return
                    for raw_path in _ingest_paths:
                        try:
                            target = str(raw_path).strip()
                            if not target:
                                continue
                            result = await document_engine.ingest_directory(target)
                            if "error" in result:
                                logger.warning(
                                    "Auto-ingest '%s' failed: %s",
                                    target, result["error"],
                                )
                            else:
                                logger.info(
                                    "Auto-ingest '%s': %d new / %d already / "
                                    "%d errors / %d chunks in %.1fs",
                                    target,
                                    result.get("ingested", 0),
                                    result.get("skipped_already", 0),
                                    result.get("errors", 0),
                                    result.get("chunks", 0),
                                    result.get("elapsed_s", 0.0),
                                )
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            logger.exception(
                                "Auto-ingest crashed on '%s'", raw_path,
                            )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Auto-ingest supervisor crashed")
            _bg_tasks.append(asyncio.create_task(_auto_ingest_corpus()))
            logger.info(
                "Auto-ingest queued for %d path(s); will start in 60s",
                len(_ingest_paths),
            )

    await cold_start.emit_restored_context()
    state.always_listen = True
    atom_runtime.patch_section(
        "lifecycle",
        {"always_listen": True},
        source="main.always_listen",
    )
    logger.info(
        "ATOM -- Supernatural Intelligence OS | always listening | requested=%s | active=%s | health=%.0fs watcher=%.0fs maint=%.0fs",
        perf_requested_mode,
        perf_effective_mode,
        health_interval,
        watcher_interval,
        maint_interval,
    )
    # NOTE: do NOT transition to LISTENING yet. We start in THINKING so the
    # STT listen loop doesn't briefly open the mic and then have to tear it
    # back down when _startup_greeting transitions to SPEAKING.
    await state.transition(AtomState.THINKING)

    # ── Boot face check (background; non-blocking) ──────────────────
    # Snap one frame from the camera as soon as the runtime is up so
    # the greeting can include "I see you, Boss." when the owner is
    # actually in front of the lens. We deliberately schedule this in
    # a thread executor (AVCaptureSession is synchronous + warms up
    # for ~300-700ms on first call) and gate the result behind a 2.5s
    # wait inside ``_startup_greeting`` so a slow camera never holds
    # back the spoken greeting.
    boot_face_future: asyncio.Future | None = None
    if (
        vision_engine is not None
        and _vision_cfg.get("enabled", False)
        and _vision_cfg.get("boot_face_check", True)
    ):
        loop = asyncio.get_running_loop()
        def _do_boot_face_check() -> Any:
            try:
                return vision_engine.look(reason="boot_face_check")
            except Exception:
                logger.debug("boot face check raised", exc_info=True)
                return None
        boot_face_future = loop.run_in_executor(None, _do_boot_face_check)

    async def _startup_greeting() -> None:
        """Speak a context-aware greeting with world intelligence.

        Kept deliberately short: a Jarvis-style boot greeting should land in
        ~2s so the user can immediately speak.  The previous 16-word version
        ("Here, Boss. Online and warmed up. Full Performance ready. N
        active goals. What do you need?") spread over 5 TTS slices and took
        7.6s end-to-end — long enough that the user assumed the system was
        sluggish.  We now lead with the time-of-day greeting and only
        appended high-signal extras (active goals, holiday, low battery).
        Mode label and weather are dropped from the boot path because the
        user can ask explicitly for those.
        """
        world_ctx = real_world_intel.get_world_context()
        temporal = world_ctx.temporal
        time_g = _adaptive_personality.greeting_response()

        extras: list[str] = []
        if cognitive_enabled:
            active_goals = goal_engine.active_count
            if active_goals:
                extras.append(
                    f"{active_goals} active goal{'s' if active_goals != 1 else ''}."
                )

        if temporal.is_holiday:
            extras.append(f"Today is {temporal.holiday_name}.")

        tail = " What do you need?" if not time_g.rstrip().endswith("?") else ""
        if extras:
            greeting = f"{time_g} {' '.join(extras)}{tail}".strip()
        else:
            greeting = f"{time_g}{tail}".strip()
        atom_runtime.patch_section(
            "reasoning",
            {
                "last_report": greeting,
                "last_decision": "Startup greeting generated",
            },
            source="main.startup_greeting",
        )

        try:
            bat = psutil.sensors_battery()
            if bat and bat.percent < 20:
                greeting += f" Heads up, battery is at {bat.percent:.0f} percent."
        except Exception:
                    logger.info("Battery check failed", exc_info=True)

        # Wait briefly for the boot face check (started in parallel
        # before this coroutine ran). Hard 2.5s ceiling so a frozen
        # AVCapture session can never delay the greeting more than
        # the time it adds to ATOM feeling "snappy".
        if boot_face_future is not None:
            try:
                face_result = await asyncio.wait_for(
                    asyncio.shield(boot_face_future), timeout=2.5,
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                face_result = None
            except Exception:
                logger.debug("boot face check await failed", exc_info=True)
                face_result = None
            if face_result is not None:
                try:
                    if face_result.ok and face_result.faces > 0:
                        cam_label = (
                            face_result.camera.name if face_result.camera else "camera"
                        )
                        logger.info(
                            "Boot face check: detected %d face(s) via %s in %.0fms",
                            face_result.faces, cam_label, face_result.detection_ms,
                        )
                        if _vision_cfg.get("boot_face_check_announce", False):
                            # Prepend, not append -- "I see you, Boss." should
                            # be the first thing the user hears.
                            greeting = f"I see you, Boss. {greeting}".strip()
                    elif face_result.ok:
                        # Camera came up, no face in frame — the user just
                        # isn't looking at the lens. This is the common case
                        # and used to land at INFO; log at DEBUG so the boot
                        # log only shows the positive (face seen) signal.
                        logger.debug(
                            "Boot face check: camera ready (%s) but no face yet",
                            face_result.camera.name if face_result.camera else "?",
                        )
                    else:
                        # Optional feature failed (Continuity Camera dozed
                        # off, video delegate raised, AVCapture session
                        # never produced a frame). The user said it
                        # explicitly: "don't make startup noise about an
                        # optional feature." Demote to DEBUG so it shows
                        # up only when explicitly debugging vision.
                        logger.debug(
                            "Boot face check unavailable: %s",
                            face_result.error or "no detail",
                        )
                except Exception:
                    # Defensive guard so a malformed face_result (e.g.
                    # missing .ok / .camera attrs after a future refactor)
                    # cannot crash the greeting path.
                    logger.debug(
                        "Boot face check result handling raised", exc_info=True,
                    )

        logger.info("Startup greeting: %s", greeting[:200])

        # Speak the boot greeting; partial_response handler will push state
        # to SPEAKING while TTS plays, then back to LISTENING when done.
        bus.emit_long("partial_response", text=greeting, is_first=True, is_last=True)

        await stt_preload_done.wait()
        logger.info("STT ready -- ATOM fully operational")

        # Once TTS has finished the boot greeting, ensure we land in
        # LISTENING so the STT loop opens the mic. If state machine already
        # brought us to LISTENING (TTS completion path), this transition is
        # a no-op.
        if state.current is not AtomState.LISTENING:
            try:
                await state.transition(AtomState.LISTENING)
            except Exception:
                logger.debug(
                    "post-greeting transition to LISTENING failed",
                    exc_info=True,
                )
        # Do NOT await async_start_listening() here: on_state_changed already create_task()s
        # exactly one listen loop when state is LISTENING/SPEAKING.
        try:
            bus.emit("restart_listening")
        except Exception:
            logger.debug("post-preload restart_listening emit failed", exc_info=True)

    _bg_tasks.append(asyncio.create_task(_startup_greeting()))

    async def _auto_performance_loop() -> None:
        """Auto-tune between Optimal and Full Performance for the M5 Air."""
        nonlocal perf_effective_mode
        interval = 45.0
        _COOLDOWN_S = 180.0
        _last_switch_time = 0.0

        def _telemetry() -> dict[str, float | bool | str]:
            cpu = 0.0
            memory_pct = 0.0
            thermal_pressure = "nominal"
            throttled = False
            battery_pct = 100.0
            on_battery = False

            if silicon_governor is not None and silicon_governor.is_available:
                try:
                    silicon_stats = silicon_governor.get_stats()
                    cpu = float(getattr(silicon_stats, "cpu_pct", 0.0) or 0.0)
                    memory_pct = float(getattr(silicon_stats, "memory_pct", 0.0) or 0.0)
                    thermal_pressure = str(
                        getattr(silicon_stats, "thermal_pressure", "nominal") or "nominal",
                    )
                    throttled = bool(getattr(silicon_stats, "is_throttled", False))
                    battery_pct = float(getattr(silicon_stats, "battery_pct", 100.0) or 100.0)
                    on_battery = bool(getattr(silicon_stats, "on_battery", False))
                except Exception:
                    logger.info("Silicon telemetry read failed", exc_info=True)

            if cpu <= 0:
                cpu = psutil.cpu_percent(interval=1.0)
            if memory_pct <= 0:
                memory_pct = float(psutil.virtual_memory().percent)

            return {
                "cpu": cpu,
                "memory_pct": memory_pct,
                "thermal_pressure": thermal_pressure,
                "throttled": throttled,
                "battery_pct": battery_pct,
                "on_battery": on_battery,
            }

        def _mode_reason(
            telemetry: dict[str, float | bool | str],
            lat_s: float,
            *,
            activity_type: str = "idle",
            target_mode: str = "optimal",
        ) -> str:
            if (
                target_mode == "optimal"
                and bool(telemetry["on_battery"])
                and activity_type == "coding"
            ):
                return "Switched to optimal due to battery power and the active development session."
            if (
                target_mode == "full_performance"
                and not bool(telemetry["on_battery"])
                and activity_type in {"coding", "browsing", "idle"}
            ):
                return "Switched to full performance because power is connected, load is low, and thermals are healthy."
            if bool(telemetry["throttled"]):
                return "Thermal pressure is high, so I'm protecting the Air."
            if float(telemetry["memory_pct"]) >= _memory_pressure_threshold:
                return (
                    "Unified memory is under pressure, so I'm dropping back to Optimal."
                )
            if float(telemetry["cpu"]) >= float(perf_cfg.get("auto_threshold_high", 70)):
                return "CPU load is high, so I'm staying in the stable mode."
            if lat_s >= 12.0:
                return "Response latency climbed too much, so I'm trimming back."
            if bool(telemetry["on_battery"]) and float(telemetry["battery_pct"]) <= 25.0:
                return "Battery is low, so I'm reducing background load."
            return "Memory, thermals, and latency look healthy."

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

                lat_ms = _last_perceived_ms.get("ms")
                lat_s = (lat_ms / 1000.0) if lat_ms is not None else 0.0
                telemetry = _telemetry()
                runtime_context = atom_runtime.store.get_section("context")
                activity_type = str(runtime_context.get("activity_type") or "idle")
                plugged_low_load = (
                    not bool(telemetry["on_battery"])
                    and float(telemetry["cpu"]) <= float(perf_cfg.get("auto_threshold_mid", 40))
                    and float(telemetry["memory_pct"]) <= max(72.0, _memory_pressure_relief)
                    and lat_s <= 6.0
                )

                promote_ready = (
                    not bool(telemetry["throttled"])
                    and str(telemetry["thermal_pressure"]) in ("nominal", "moderate", "")
                    and float(telemetry["memory_pct"]) <= max(72.0, _memory_pressure_relief)
                    and float(telemetry["cpu"]) <= float(perf_cfg.get("auto_threshold_mid", 40))
                    and lat_s <= 6.0
                    and activity_type in {"coding", "browsing", "idle"}
                    and (
                        not bool(telemetry["on_battery"])
                        or float(telemetry["battery_pct"]) >= 35.0
                    )
                )
                must_demote = (
                    bool(telemetry["throttled"])
                    or float(telemetry["memory_pct"]) >= _memory_pressure_threshold
                    or float(telemetry["cpu"]) >= float(perf_cfg.get("auto_threshold_high", 70))
                    or lat_s >= 12.0
                    or (
                        bool(telemetry["on_battery"])
                        and float(telemetry["battery_pct"]) <= 25.0
                    )
                )

                target = perf_effective_mode
                if perf_requested_mode == "optimal":
                    target = "optimal"
                elif perf_requested_mode == "full_performance":
                    if must_demote:
                        target = "optimal"
                    elif promote_ready:
                        target = "full_performance"
                else:
                    if must_demote:
                        target = "optimal"
                    elif bool(telemetry["on_battery"]) and activity_type == "coding":
                        target = "optimal"
                    elif perf_effective_mode == "optimal" and promote_ready:
                        target = "full_performance"

                reason = _mode_reason(
                    telemetry,
                    lat_s,
                    activity_type=activity_type,
                    target_mode=target,
                )
                if target == perf_effective_mode:
                    if perf_requested_mode == "auto" and plugged_low_load:
                        _broadcast_perf_state(reason)
                    else:
                        _broadcast_perf_state(reason)
                    continue
                logger.info(
                    "M5 auto perf: requested=%s current=%s target=%s latency=%.1fs cpu=%.0f%% mem=%.0f%% thermal=%s battery=%.0f%% on_battery=%s activity=%s",
                    perf_requested_mode,
                    perf_effective_mode,
                    target,
                    lat_s,
                    float(telemetry["cpu"]),
                    float(telemetry["memory_pct"]),
                    telemetry["thermal_pressure"],
                    float(telemetry["battery_pct"]),
                    telemetry["on_battery"],
                    activity_type,
                )
                switched = await _sync_effective_mode(
                    target,
                    reason=reason,
                    speak=False,
                )
                if switched:
                    _last_switch_time = now
            except Exception as exc:
                logger.debug("Auto performance check error: %s", exc)

    if perf_requested_mode in ("auto", "full_performance"):
        _bg_tasks.append(asyncio.create_task(_auto_performance_loop()))
        logger.info(
            "M5 mode guard active (requested=%s, CPU thresholds=%s/%s, memory pressure=%.0f%%)",
            perf_requested_mode,
            perf_cfg.get("auto_threshold_mid", 40),
            perf_cfg.get("auto_threshold_high", 70),
            _memory_pressure_threshold,
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

    proactive_alerts = perf_cfg.get("proactive_alerts", perf_requested_mode == "full_performance")
    idle_reminder = perf_cfg.get("idle_reminder", perf_requested_mode == "full_performance")
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

            if perf_effective_mode == "full_performance":
                log_health(metrics)

            if cycle % cache_purge_cycles == 0:
                cache.purge_expired()
                logger.info("Periodic maintenance: cache purged")

            if cycle % tune_cycles == 0:
                try:
                    _self_tune()
                except Exception:
                    logger.debug("Self-tune error", exc_info=True)

            snapshot_cycles = max(1, 1800 // maint_interval)
            if cycle % snapshot_cycles == 0:
                try:
                    cold_start.persist_snapshot()
                except Exception:
                    logger.debug("Cold start snapshot persist failed", exc_info=True)

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

    # ── V22 Convergence: Advanced Proactive Daemon ──
    from core.background.proactive_agent import ProactiveDaemon
    from core.cognition.state_graph import SystemStateGraph
    try:
        convergence_daemon = ProactiveDaemon()
        convergence_daemon.wire(state_graph=SystemStateGraph(), tts=tts)
        convergence_daemon.start()
        logger.info("V22 Convergence Daemon started.")
    except Exception as e:
        logger.error(f"Failed to start convergence daemon: {e}")
        convergence_daemon = None

    try:
        await shutdown_event.wait()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Interrupt received")
    finally:
        logger.info("Cleaning up...")
        if convergence_daemon:
            convergence_daemon.stop()
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
        user_memory.persist()
        system_state_engine.stop()
        voice_pipeline.shutdown()
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
        try:
            system_profile.refresh_from_scanner()
            system_profile.persist()
        except Exception:
            logger.debug("System profile shutdown persist failed", exc_info=True)
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

        if llm_queue is not None:
            await llm_queue.shutdown()
        if runtime_watchdog is not None:
            await runtime_watchdog.shutdown()
        if priority_sched is not None:
            await priority_sched.shutdown()
        if iphone_bridge is not None:
            try:
                await iphone_bridge.stop()
            except Exception:
                logger.debug("iPhone bridge stop failed", exc_info=True)
        try:
            cognitive_handles.stop()
        except Exception:
            logger.debug("cognitive_handles.stop failed", exc_info=True)
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
        try:
            await memory.shutdown_writers()
        except Exception:
            logger.info("Memory engine writer shutdown failed", exc_info=True)
        memory.persist()
        try:
            _adaptive = _wiring_ctx.get("adaptive") if isinstance(_wiring_ctx, dict) else None
            if _adaptive is not None and hasattr(_adaptive, "flush"):
                _adaptive.flush()
        except Exception:
            logger.info("Adaptive engine flush failed", exc_info=True)
        executor.shutdown(wait=False)
        logger.info("ATOM stopped.")


# Boot-time permanent errors. Retrying these is pointless: an ImportError
# at module load or a SyntaxError does not self-heal across a 2-second sleep.
# Keep this tuple tight -- anything listed here will short-circuit crash_guard.
_PERMANENT_BOOT_ERRORS: tuple[type[BaseException], ...] = (
    ImportError,   # includes ModuleNotFoundError
    SyntaxError,
)


# Dependencies that must import cleanly before main() runs. Each entry is
# (module_name, install_hint). Kept minimal on purpose: only add deps that
# are (a) always required for boot and (b) have caused a crash loop before.
_HARD_DEPS: tuple[tuple[str, str], ...] = (
    ("cryptography", "pip install cryptography"),
    ("mlx_lm", "pip install mlx-lm  (Apple Silicon only)"),
)


# Files that must be present inside the primary MLX model directory for
# mlx-lm to load weights + tokenizer cleanly. Missing any of these loops
# the brain loader forever; fail fast at preflight instead.
_MLX_MODEL_REQUIRED_FILES: tuple[str, ...] = (
    "config.json",
    "tokenizer.json",
)


def _preflight_hard_deps() -> list[str]:
    """Return a list of install-hint strings for missing hard deps.

    Idempotent and side-effect free. Called once before the crash_guard
    retry loop so a missing dep produces a single diagnostic line instead
    of N identical 3-page tracebacks.
    """
    preflight_log = logging.getLogger("atom.boot.preflight")
    missing: list[str] = []
    for module_name, hint in _HARD_DEPS:
        try:
            __import__(module_name)
        except ImportError as exc:
            missing.append(f"{module_name} ({exc}) -- {hint}")
    if missing:
        for line in missing:
            preflight_log.critical("Missing hard dependency: %s", line)
    return missing


def _preflight_brain_model() -> list[str]:
    """Verify the configured MLX primary model directory is loadable.

    Returns a list of human-readable error strings; empty on success.
    Checks, in order: (1) settings.json is readable, (2) the configured
    primary-model path resolves to a directory, (3) required tokenizer
    + config files exist, (4) at least one ``*.safetensors`` weight
    file is present.

    A missing model here would otherwise crash the brain controller on
    first voice turn with a useless stacktrace and loop crash_guard.
    """
    preflight_log = logging.getLogger("atom.boot.preflight")
    errors: list[str] = []
    try:
        import json as _json
        from pathlib import Path as _Path
        repo_root = _Path(__file__).resolve().parent
        settings_path = repo_root / "config" / "settings.json"
        if not settings_path.is_file():
            errors.append(f"config/settings.json missing at {settings_path}")
            return errors
        cfg = _json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"settings.json unreadable: {exc}")
        return errors

    brain_cfg = cfg.get("brain") or {}
    if not brain_cfg.get("enabled", True):
        return []  # Brain disabled on purpose -- nothing to check.

    # Preferred key: brain.mlx_model. Older profiles used separate
    # primary/fast keys; accept them here for backwards compatibility
    # so an out-of-date settings.json still boots.
    rel = ""
    for key in ("mlx_model", "mlx_primary_model", "mlx_fast_model", "model_path"):
        val = str(brain_cfg.get(key) or "").strip()
        if val:
            rel = val
            break
    if not rel:
        errors.append(
            "brain.mlx_model missing from settings.json "
            "(no legacy mlx_primary_model / mlx_fast_model / model_path either)"
        )
        return errors

    from pathlib import Path as _Path
    model_path = (_Path(__file__).resolve().parent / rel).resolve()
    if not model_path.is_dir():
        errors.append(f"MLX model directory missing: {model_path}")
        return errors

    for name in _MLX_MODEL_REQUIRED_FILES:
        if not (model_path / name).is_file():
            errors.append(f"{name} missing inside {model_path.name}/")

    if not list(model_path.glob("*.safetensors")):
        errors.append(f"no .safetensors weights found in {model_path.name}/")

    if errors:
        for line in errors:
            preflight_log.critical("Brain model preflight: %s", line)
    return errors


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

    if _preflight_hard_deps() or _preflight_brain_model():
        logging.getLogger("atom.crash_guard").critical(
            "Preflight failed -- not starting main(). "
            "Install missing deps / restore the MLX model directory and retry."
        )
        set_config_overrides({})
        sys.exit(2)

    MAX_RETRIES = 5
    MAX_BACKOFF_S = 30.0
    attempt = 0

    while attempt < MAX_RETRIES:
        try:
            asyncio.run(main())
            if _restart_requested:
                _restart_requested = False
                if shutdown_event is not None:
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
        except _PERMANENT_BOOT_ERRORS as exc:
            crash_logger = logging.getLogger("atom.crash_guard")
            crash_logger.critical(
                "Permanent boot failure (%s: %s) -- not retrying. "
                "Fix the code/install the missing dep and relaunch.",
                type(exc).__name__, exc,
                exc_info=True,
            )
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
            if shutdown_event is not None:
                shutdown_event.clear()

    set_config_overrides({})


if __name__ == "__main__":
    run_atom()
