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

from core.boot.config_loader import load_config, set_config_overrides


logger = logging.getLogger("atom.main")
shutdown_event: asyncio.Event | None = None
_restart_requested = False


from core.boot.wiring import wire_events


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

    if False:
        bus.start()
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

    stt_cfg = config.get("stt", {})
    stt_runtime_label = "Voice input unavailable"
    stt_runtime_error = ""
    stt_runtime_fallbacks: list[str] = []
    stt_engine_pref = str(stt_cfg.get("engine", "auto") or "auto").strip().lower()

    def _build_disabled_stt(reason: str):
        class DisabledSTT:
            def __init__(self, disable_reason: str) -> None:
                self._reason = disable_reason
                self._last_error = disable_reason
                self.mic_name = "Voice input unavailable"
                self.backend_name = "Disabled"
                self.fallback_chain = [disable_reason]
                self.speech_permission_status = (
                    "bundle_missing_usage_description"
                    if "NSSpeechRecognitionUsageDescription" in disable_reason
                    else "unavailable"
                )
                self.microphone_permission_status = (
                    "dependency_missing"
                    if "PyAudio/PortAudio" in disable_reason
                    else "unknown"
                )

            async def async_preload(self) -> None:
                logger.warning("STT disabled: %s", self._reason)

            async def async_start_listening(self, **_kw) -> None:
                logger.warning("STT disabled: %s", self._reason)

            async def start_listening(self, **_kw) -> None:
                await self.async_start_listening(**_kw)

            async def on_state_changed(self, old, new, **_kw) -> None:
                return None

            def stop(self) -> None:
                return None

            def shutdown(self) -> None:
                return None

        logger.error("Voice input unavailable: %s", reason)
        return DisabledSTT(reason)

    def _build_google_stt():
        """Build Google Online STT (primary — fast, free, accurate)."""
        missing: list[str] = []
        try:
            import speech_recognition  # noqa: F401
        except ImportError:
            missing.append("SpeechRecognition")
        try:
            import pyaudio  # noqa: F401
        except ImportError:
            missing.append("PyAudio/PortAudio")

        if missing:
            return None, "Google STT dependencies missing: " + ", ".join(missing)

        from voice.stt_google import STTGoogle

        logger.info("STT: Google Online (free, fast, bilingual)")
        return STTGoogle(
            bus,
            state,
            config,
            mic_manager=mic_manager,
            intent_engine=intent_engine,
        ), None

    def _build_faster_whisper_stt():
        """Build offline faster-whisper STT (fallback if no internet)."""
        missing: list[str] = []
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            missing.append("faster-whisper")
        try:
            import speech_recognition  # noqa: F401
        except ImportError:
            missing.append("SpeechRecognition")
        try:
            import pyaudio  # noqa: F401
        except ImportError:
            missing.append("PyAudio/PortAudio")

        if missing:
            return _build_disabled_stt(
                "Offline STT dependencies missing: " + ", ".join(missing),
            )

        from voice.stt_async import STTAsync

        logger.info("STT: faster-whisper (offline fallback)")
        return STTAsync(
            bus,
            state,
            config,
            mic_manager=mic_manager,
            intent_engine=intent_engine,
        )

    def _build_native_stt():
        from voice.stt_macos import NativeSTT, native_stt_launch_supported

        native_ok, native_reason = native_stt_launch_supported()
        if not native_ok:
            return None, native_reason
        return NativeSTT(
            bus,
            state,
            config,
            mic_manager=mic_manager,
            intent_engine=intent_engine,
        ), ""

    # STT priority chain (macOS):
    #   1. macos_native → Apple SFSpeechRecognizer
    #   2. faster_whisper → offline fallback
    #   3. google_online → online fallback
    logger.info("STT engine preference: %s (platform=%s)", stt_engine_pref, sys.platform)

    if stt_engine_pref in ("auto", "macos_native") and sys.platform == "darwin":
        native_stt, native_reason = _build_native_stt()
        if native_stt is not None:
            stt = native_stt
            stt_runtime_label = "macOS Native (SFSpeechRecognizer)"
            logger.info("STT: macOS Native (SFSpeechRecognizer, on-device)")
        else:
            stt_runtime_error = native_reason
            stt_runtime_fallbacks.append(f"native unavailable: {native_reason}")
            logger.warning(
                "macOS Native STT unavailable (%s) -- falling back to Faster-Whisper",
                native_reason,
            )
            whisper_stt = _build_faster_whisper_stt()
            if type(whisper_stt).__name__ != "DisabledSTT":
                stt = whisper_stt
                stt_runtime_label = "Faster-Whisper (native fallback)"
            else:
                whisper_reason = getattr(whisper_stt, "_reason", "offline fallback unavailable")
                stt_runtime_fallbacks.append(f"offline unavailable: {whisper_reason}")
                logger.warning("Faster-Whisper unavailable (%s) -- trying Google", whisper_reason)
                google_stt, google_err = _build_google_stt()
                if google_stt is not None:
                    stt = google_stt
                    stt_runtime_label = "Google Online (third fallback)"
                else:
                    stt_runtime_fallbacks.append(f"google unavailable: {google_err}")
                    stt = _build_disabled_stt(
                        f"Native unavailable ({native_reason}); offline unavailable ({whisper_reason}); google unavailable ({google_err})"
                    )
                    stt_runtime_label = "Disabled"
    elif stt_engine_pref in ("google_online", "google"):
        google_stt, google_err = _build_google_stt()
        if google_stt is not None:
            stt = google_stt
            stt_runtime_label = "Google Online (free)"
        else:
            stt_runtime_error = google_err or ""
            stt_runtime_fallbacks.append(f"google unavailable: {google_err}")
            logger.warning("Google STT unavailable (%s) -- trying offline fallback", google_err)
            stt = _build_faster_whisper_stt()
            if type(stt).__name__ == "DisabledSTT":
                stt_runtime_label = "Disabled"
            else:
                stt_runtime_label = "Faster-Whisper (offline fallback)"
    elif stt_engine_pref == "auto":
        stt = _build_faster_whisper_stt()
        if type(stt).__name__ == "DisabledSTT":
            whisper_reason = getattr(stt, "_reason", "offline fallback unavailable")
            stt_runtime_fallbacks.append(f"offline unavailable: {whisper_reason}")
            google_stt, google_err = _build_google_stt()
            if google_stt is not None:
                stt = google_stt
                stt_runtime_label = "Google Online (auto fallback)"
            else:
                stt_runtime_fallbacks.append(f"google unavailable: {google_err}")
                stt = _build_disabled_stt(
                    f"Offline unavailable ({whisper_reason}); google unavailable ({google_err})"
                )
                stt_runtime_label = "Disabled"
        else:
            stt_runtime_label = "Faster-Whisper (auto)"
    elif stt_engine_pref == "faster_whisper":
        stt = _build_faster_whisper_stt()
        stt_runtime_label = (
            "Disabled"
            if type(stt).__name__ == "DisabledSTT"
            else "Faster-Whisper (explicit)"
        )
    else:
        stt = _build_disabled_stt(f"Unknown STT engine: {stt_engine_pref}")
        stt_runtime_label = "Disabled"

    logger.info("STT backend selected: %s", type(stt).__name__)

    tts_cfg = config.get("tts", {})
    tts_engine = (tts_cfg.get("engine") or "macos_native").lower()
    tts_runtime_label = "macOS Native"

    if tts_engine == "macos_native":
        from voice.tts_macos import MacOSTTSAsync
        tts = MacOSTTSAsync(
            bus, state,
            max_lines=tts_cfg.get("max_lines", 4),
            voice=tts_cfg.get("macos_voice", "system"),
            rate=tts_cfg.get("macos_rate", 200),
        )
        logger.info("TTS: macOS Native (voice=%s)", tts_cfg.get("macos_voice", "system"))
        tts_runtime_label = f"macOS Native ({tts_cfg.get('macos_voice', 'system')})"
    elif tts_engine == "kokoro":
        from voice.tts_kokoro import KokoroTTSAsync
        tts = KokoroTTSAsync(
            bus, state,
            max_lines=tts_cfg.get("max_lines", 4),
            voice=tts_cfg.get("kokoro_voice", "af_heart")
        )
        logger.info(
            "TTS: Kokoro Neural fallback (offline, %s)",
            tts_cfg.get("kokoro_voice", "af_heart"),
        )
        tts_runtime_label = f"Kokoro fallback ({tts_cfg.get('kokoro_voice', 'af_heart')})"
    else:
        from voice.tts_edge import EdgeTTSAsync
        tts = EdgeTTSAsync(
            bus, state,
            max_lines=tts_cfg.get("max_lines", 4),
            voice=tts_cfg.get("edge_voice", "en-GB-RyanNeural"),
            rate=tts_cfg.get("edge_rate", "+15%"),
            enable_postprocess=tts_cfg.get("edge_postprocess", True),
            enable_ack_cache=tts_cfg.get("edge_ack_cache", True),
        )
        logger.info(
            "TTS: Edge Neural fallback (%s) -- macOS native remains preferred on Apple Silicon",
            tts_cfg.get("edge_voice"),
        )
        tts_runtime_label = f"Edge fallback ({tts_cfg.get('edge_voice')})"

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
    _embedding_pressure_unloaded = False

    async def _on_silicon_stats_update(stats=None, **_kw) -> None:
        nonlocal _embedding_pressure_unloaded
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

    cloud_enabled_cfg = bool(config.get("cloud", {}).get("enabled", True))
    gemini_client: GeminiClient | None = None
    if cloud_enabled_cfg:
        gemini_client = GeminiClient(config, security_gateway=security_gateway)
        from core.secrets_manager import get_gemini_fast_key

        _gemini_key = get_gemini_fast_key()
        if _gemini_key:
            gemini_client.configure_api_key(_gemini_key)
            logger.info("Gemini API key loaded from secure storage")
        else:
            logger.warning(
                "Gemini API key not found. Run: python scripts/setup_api_keys.py"
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
            pass

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
    )

    await tts.init_voice()
    stt_preload_done = asyncio.Event()
    _bg_tasks: list[asyncio.Task] = []

    async def _background_stt_preload() -> None:
        t0 = time.monotonic()
        logger.info("STT model loading in background...")

        try:
            loop = asyncio.get_running_loop()

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
        bus=bus, state=state, state_bridge=atom_runtime, shutdown_event=shutdown_event, stt=stt, tts=tts, router=router,
        indicator=indicator, cache=cache, memory=memory, metrics=metrics,
        config=config, local_brain=local_brain, llm_queue=llm_queue,
        assistant_mode_mgr=assistant_mode_mgr,
        behavior=behavior,
        scheduler=scheduler, process_mgr=process_mgr, evolution=evolution,
        priority_sched=priority_sched,
        v3=False, v4=False,
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
        await web_dashboard.start()
    else:
        indicator.start()
    _broadcast_perf_state()

    if local_brain and local_brain.available:
        brain_cfg = config.get("brain", {})
        model_name = Path(
            brain_cfg.get("mlx_primary_model") or brain_cfg.get("mlx_fast_model") or "mlx",
        ).stem.replace("-mlx", "")
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
    await state.transition(AtomState.LISTENING)

    async def _startup_greeting() -> None:
        """Speak a context-aware greeting with world intelligence."""
        mode_label = _mode_label(perf_effective_mode)

        world_ctx = real_world_intel.get_world_context()
        temporal = world_ctx.temporal
        time_g = _adaptive_personality.greeting_response()

        greeting_bits = [f"{mode_label} ready."]
        if cognitive_enabled:
            active_goals = goal_engine.active_count
            if active_goals:
                greeting_bits.append(
                    f"{active_goals} active goal{'s' if active_goals != 1 else ''}."
                )

        if not world_ctx.weather.is_stale:
            greeting_bits.append(f"Weather: {world_ctx.weather.summary()}.")
        elif world_ctx.weather.condition != "unknown":
            greeting_bits.append(f"Last weather: {world_ctx.weather.summary()}.")

        if temporal.is_holiday:
            greeting_bits.append(f"Today is {temporal.holiday_name}.")

        greeting = f"{time_g} All systems online. {' '.join(greeting_bits)} What do you need?"
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
        logger.info("Startup greeting: %s", greeting[:200])

        await state.transition(AtomState.THINKING)
        bus.emit_long("partial_response", text=greeting, is_first=True, is_last=True)

        await stt_preload_done.wait()
        logger.info("STT ready -- ATOM fully operational")
        # Do NOT await async_start_listening() here: on_state_changed already create_task()s
        # exactly one listen loop when state is LISTENING/SPEAKING. Awaiting it duplicated the
        # loop and raced the mic with startup TTS (LISTENING→THINKING→SPEAKING→LISTENING).
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
                    speak=True,
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
        convergence_daemon = ProactiveDaemon(state_graph=SystemStateGraph(), tts_engine=tts)
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
