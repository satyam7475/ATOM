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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def _load_config() -> dict:
    cfg_path = Path("config/settings.json")
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


logger = logging.getLogger("atom.main")
shutdown_event = asyncio.Event()
_restart_requested = False


def _wire_events(
    *,
    bus,
    state,
    stt,
    tts,
    router,
    indicator,
    cache,
    memory,
    metrics,
    config: dict,
    local_brain=None,
    llm_queue=None,
    assistant_mode_mgr=None,
    behavior,
    scheduler=None,
    process_mgr=None,
    evolution=None,
    priority_sched=None,
) -> dict:
    """Wire all event bus handlers. Extracted from main() for testability.

    Returns a shared-state dict used by handlers (perceived latency tracking,
    catch counter, proactive state, stream buffer).
    """
    from core.state_manager import AtomState
    from core.metrics import log_health

    _didnt_catch_count = {"n": 0}
    _perceived = {"t_speech_final": 0.0, "logged": False}
    _last_perceived_ms = {"ms": None}
    _proactive_state = {"last_query_time": time.monotonic(), "low_battery_warned": False}
    _stream_buffer = {"text": ""}
    _ttfa_gate = {"sent": False}
    _llm_latency_history: list[float] = []
    _thinking_progress_task: dict[str, asyncio.Task | None] = {"task": None}
    _LLM_HISTORY_MAX = 10

    # Global Interrupt Manager
    from core.ipc.interrupt_manager import SystemInterruptManager
    interrupt_mgr = SystemInterruptManager(bus, "main_core")
    bus.on("state_changed", indicator.on_state_changed)
    bus.on("state_changed", stt.on_state_changed)
    if priority_sched is not None:
        from core.priority_scheduler import PRIORITY_VOICE

        async def _speech_via_priority(text: str, **kw) -> None:
            if shutdown_event.is_set():
                return
            if local_brain is not None:
                local_brain.request_preempt()

            def _factory():
                async def _job() -> None:
                    if shutdown_event.is_set():
                        return
                    await router.on_speech(text, **kw)

                return _job()

            priority_sched.submit(PRIORITY_VOICE, "speech_final", _factory)

        if not args.v4:
            bus.on("speech_final", _speech_via_priority)
    else:
        async def _speech_final_direct(text: str, **kw) -> None:
            if shutdown_event.is_set():
                return
            if local_brain is not None:
                local_brain.request_preempt()
            await router.on_speech(text, **kw)

        if not args.v4:
            bus.on("speech_final", _speech_final_direct)

    # ── Local LLM only (offline) — serial queue + fast bus handler ─
    if local_brain is not None:
        async def _local_brain_query(text: str, **kw) -> None:
            async def _run_brain() -> None:
                if shutdown_event.is_set():
                    return
                if assistant_mode_mgr is not None and not assistant_mode_mgr.allows_llm_fallback():
                    bus.emit_long(
                        "response_ready",
                        text=assistant_mode_mgr.command_only_message(),
                    )
                    return
                if not local_brain.available:
                    bus.emit_long(
                        "response_ready",
                        text=(
                            "Local brain is not ready, Boss. Check brain.model_path in "
                            "settings.json and that llama-cpp-python is installed."
                        ),
                    )
                    return
                try:
                    if llm_queue is not None:
                        await llm_queue.submit(
                            text,
                            memory_context=kw.get("memory_context"),
                            context=kw.get("context"),
                            history=kw.get("history"),
                        )
                    else:
                        await local_brain.on_query(text, **kw)
                except Exception as exc:
                    logger.exception("Local brain query failed: %s", exc)
                    bus.emit_long(
                        "response_ready",
                        text="Local brain hit an error, Boss. Check the log and try again.",
                    )

            if priority_sched is not None:
                from core.priority_scheduler import PRIORITY_LLM

                if shutdown_event.is_set():
                    return

                def _factory():
                    return _run_brain()

                priority_sched.submit(PRIORITY_LLM, "cursor_query", _factory)
                return
            await _run_brain()

        bus.on("cursor_query", _local_brain_query)

        async def _on_pending_tool_confirmation(tool_call=None, result=None, **_kw) -> None:
            """Store pending tool confirmation from agentic LLM for the Router."""
            if tool_call is not None:
                router._pending_tool_confirmation = {
                    "tool_call": tool_call,
                    "result": result,
                    "created_at": time.monotonic(),
                }

        bus.on("pending_tool_confirmation", _on_pending_tool_confirmation)

    bus.on("response_ready", tts.on_response)
    bus.on("partial_response", tts.on_partial_response)
    bus.on("tts_complete", state.on_tts_complete)
    bus.on("silence_timeout", state.on_silence_timeout)

    # ── Media / error recovery ────────────────────────────────────
    async def _on_media_started(**_kw) -> None:
        stt.on_media_started()
    bus.on("media_started", _on_media_started)

    async def on_llm_error(source: str = "local", **_kw) -> None:
        logger.error("LLM error from %s -- triggering recovery", source)
        await state.on_error(source=source)
    bus.on("llm_error", on_llm_error)

    # ── Sleep / barge-in / resume (hotkey + dashboard UNSTICK) ─
    async def on_resume_listening(**_kw) -> None:
        # Global Interrupt Handling
        await interrupt_mgr.broadcast_interrupt()
        
        if state.current is AtomState.SLEEP:
            logger.info("Leaving SLEEP via hotkey / resume")
            await state.transition(AtomState.LISTENING)
            indicator.add_log("action", "I'm back, Boss.")
            return
        if state.current is AtomState.ERROR_RECOVERY:
            logger.info("Resume during ERROR_RECOVERY -> IDLE")
            await state.transition(AtomState.IDLE)
        if state.current is AtomState.THINKING:
            logger.info("Interrupt during THINKING")
            indicator.add_log("info", "Interrupted. Go ahead, Boss.")
            if local_brain is not None:
                local_brain.request_preempt()
        if state.current is AtomState.SPEAKING:
            logger.info("Barge-in -- stopping TTS")
            tts.stop()
            
        await state.transition(AtomState.LISTENING)
    bus.on("resume_listening", on_resume_listening)

    async def _on_enter_sleep(**_kw) -> None:
        logger.info("Entering SLEEP mode -- Ctrl+Alt+A to resume listening")
        stt.stop()
        await state.transition(AtomState.SLEEP)
        indicator.add_log("action", "Silent mode. Press Ctrl+Alt+A to resume listening.")
    bus.on("enter_sleep_mode", _on_enter_sleep)

    # ── STT recovery ─────────────────────────────────────────────
    async def on_restart_listening(**_kw) -> None:
        if state.current is AtomState.LISTENING:
            await asyncio.sleep(0.1)
            if state.current is AtomState.LISTENING:
                if not (args.v3 or args.v4):
                    asyncio.create_task(stt.start_listening())
    bus.on("restart_listening", on_restart_listening)

    async def on_stt_did_not_catch(**_kw) -> None:
        _didnt_catch_count["n"] += 1
        if _didnt_catch_count["n"] <= 2:
            await state.transition(AtomState.THINKING)
            bus.emit_long("response_ready", text="I didn't catch that, Boss. Try again?")
        elif state.current is not AtomState.LISTENING:
            await state.transition(AtomState.LISTENING)

    async def on_stt_too_noisy(**_kw) -> None:
        _didnt_catch_count["n"] += 1
        if _didnt_catch_count["n"] <= 2:
            await state.transition(AtomState.THINKING)
            bus.emit_long("response_ready",
                          text="Background noise is high. Move closer or reduce noise.")
        elif state.current is not AtomState.LISTENING:
            await state.transition(AtomState.LISTENING)
    bus.on("stt_did_not_catch", on_stt_did_not_catch)
    bus.on("stt_too_noisy", on_stt_too_noisy)

    # ── UI logging ───────────────────────────────────────────────
    async def log_response(text: str, **_kw) -> None:
        _stop_thinking_progress()
        indicator.add_log("action", text)

    async def log_thinking_ack(text: str, **_kw) -> None:
        if text and _perceived["t_speech_final"] > 0 and not _ttfa_gate["sent"]:
            ttfa_ms = (time.perf_counter() - _perceived["t_speech_final"]) * 1000
            metrics.record_latency("ttfa", ttfa_ms)
            _ttfa_gate["sent"] = True
        indicator.add_log("info", text)
        if text:
            asyncio.create_task(tts.speak_ack(text))

    async def log_cursor_query(text: str, **_kw) -> None:
        indicator.add_log("action", "Thinking with local brain...")
        _start_thinking_progress()

    async def log_partial(text: str, is_first: bool = False, is_last: bool = False, **_kw) -> None:
        if is_first:
            _stream_buffer["text"] = ""
        if text.strip():
            _stream_buffer["text"] += (" " if _stream_buffer["text"] else "") + text.strip()
            indicator.add_log("speaking", _stream_buffer["text"])
        if is_last and _stream_buffer["text"]:
            indicator.add_log("action", _stream_buffer["text"])
            _stream_buffer["text"] = ""

    async def show_hearing(text: str, **_kw) -> None:
        indicator.show_hearing(text)

    def _estimate_llm_seconds() -> float:
        if _llm_latency_history:
            return sum(_llm_latency_history) / len(_llm_latency_history) / 1000.0
        return 15.0

    async def _thinking_progress_loop() -> None:
        """Emit progress updates every 2s while the LLM is thinking."""
        estimate_s = _estimate_llm_seconds()
        t0 = time.perf_counter()
        try:
            while True:
                await asyncio.sleep(2.0)
                elapsed = time.perf_counter() - t0
                if hasattr(indicator, "broadcast_thinking_progress"):
                    indicator.broadcast_thinking_progress(elapsed, estimate_s)
        except asyncio.CancelledError:
            pass

    def _start_thinking_progress() -> None:
        if _thinking_progress_task["task"] is not None:
            _thinking_progress_task["task"].cancel()
        _thinking_progress_task["task"] = asyncio.ensure_future(_thinking_progress_loop())

    def _stop_thinking_progress() -> None:
        t = _thinking_progress_task.get("task")
        if t is not None:
            t.cancel()
            _thinking_progress_task["task"] = None

    async def _measure_perceived(text: str, is_first: bool = False, **_kw) -> None:
        if is_first and _perceived["t_speech_final"] > 0 and not _perceived["logged"]:
            latency_ms = (time.perf_counter() - _perceived["t_speech_final"]) * 1000
            logger.info("PERCEIVED_LATENCY = %.0fms (speech_final -> first TTS audio)", latency_ms)
            metrics.record_latency("perceived", latency_ms)
            _last_perceived_ms["ms"] = latency_ms
            _perceived["logged"] = True
            _llm_latency_history.append(latency_ms)
            if len(_llm_latency_history) > _LLM_HISTORY_MAX:
                _llm_latency_history.pop(0)
            _stop_thinking_progress()
            if hasattr(indicator, "set_last_latency_ms"):
                indicator.set_last_latency_ms(latency_ms)

    _active_language = {"lang": "en"}

    async def _on_speech_final_consolidated(text: str, language: str = "en", **_kw) -> None:
        _perceived["t_speech_final"] = time.perf_counter()
        _perceived["logged"] = False
        _ttfa_gate["sent"] = False
        _didnt_catch_count["n"] = 0
        _proactive_state["last_query_time"] = time.monotonic()
        _active_language["lang"] = language
        indicator.clear_hearing()
        lang_label = "[HI]" if language == "hi" else "[EN]"
        indicator.add_log("heard", f"{lang_label} {text}")
        metrics.inc("queries_total")
        if hasattr(indicator, "set_last_query"):
            indicator.set_last_query(text)
        if hasattr(indicator, "set_language"):
            indicator.set_language(language)

    async def _on_intent_classified(intent: str = "", **_kw) -> None:
        if hasattr(indicator, "set_last_intent"):
            indicator.set_last_intent(intent)

    bus.on("speech_final", _on_speech_final_consolidated)
    bus.on("intent_classified", _on_intent_classified)
    bus.on("partial_response", _measure_perceived)
    bus.on("speech_partial", show_hearing)
    if hasattr(tts, "on_speech_partial"):
        bus.on("speech_partial", tts.on_speech_partial)
    async def on_text_display(text: str, **_kw) -> None:
        """Screen-only overflow text (not spoken, shown on dashboard)."""
        if text.strip():
            indicator.add_log("info", f"[screen] {text.strip()}")

    bus.on("response_ready", log_response)
    bus.on("partial_response", log_partial)
    bus.on("text_display", on_text_display)
    bus.on("thinking_ack", log_thinking_ack)
    bus.on("cursor_query", log_cursor_query)

    # ── Metrics ──────────────────────────────────────────────────
    async def metrics_on_resume_listening(**_kw) -> None:
        metrics.inc("resume_listening_events")

    async def metrics_on_counter(counter: str, **_kw) -> None:
        metrics.inc(counter)

    async def metrics_on_latency(name: str, ms: float, **_kw) -> None:
        metrics.record_latency(name, ms)
        if name == "llm":
            metrics.inc("llm_calls")
    bus.on("resume_listening", metrics_on_resume_listening)
    bus.on("metrics_event", metrics_on_counter)
    bus.on("metrics_latency", metrics_on_latency)

    # ── System events (AI OS layer) ──────────────────────────────
    async def _on_system_event(kind: str = "", app: str = "",
                               message: str = "", **kw) -> None:
        if kind == "app_switch" and process_mgr is not None:
            process_mgr.record_app_switch(app)
            return
        if kind == "resource_alert" and message:
            indicator.add_log("warning", message)
            bus.emit_long("response_ready", text=message)
            return
        if state.current not in (AtomState.IDLE, AtomState.LISTENING):
            return
        if kind == "network_lost":
            indicator.add_log("warning", "Network connection dropped.")
            bus.emit_long("response_ready",
                          text="Heads up, Boss. Your network just dropped.")
        elif kind == "network_restored":
            indicator.add_log("info", "Back online.")
        elif kind == "power_unplugged":
            level = kw.get("level", 0)
            if level < 30:
                indicator.add_log("warning",
                                  f"Unplugged at {level}% -- keep an eye on it.")
        elif kind == "battery_critical":
            level = kw.get("level", 0)
            bus.emit_long("response_ready",
                          text=f"Boss, battery is critically low at {level} percent. Plug in soon.")
        elif kind == "bt_connected":
            device = kw.get("device", "device")
            indicator.add_log("info", f"Bluetooth: {device} connected")
        elif kind == "bt_disconnected":
            device = kw.get("device", "device")
            indicator.add_log("info", f"Bluetooth: {device} disconnected")
    bus.on("system_event", _on_system_event)

    # ── Intent chaining + behavior ───────────────────────────────
    async def _on_chain_suggestion(suggestion: str = "", **_kw) -> None:
        if suggestion:
            await asyncio.sleep(1.5)
            indicator.add_log("info", suggestion)
    bus.on("intent_chain_suggestion", _on_chain_suggestion)

    async def _on_action_for_behavior(intent: str = "", **_kw) -> None:
        if intent and intent not in ("fallback", "confirm", "deny", "greeting",
                                      "thanks", "status"):
            target = _kw.get("target", "") or _kw.get("name", "")
            behavior.log(intent, target)
    bus.on("intent_classified", _on_action_for_behavior)

    # ── LLM response caching + follow-up ─────────────────────────
    async def on_cursor_response(query: str, response: str, **_kw) -> None:
        cache.put(query, response)
        await memory.add(query, response)
        router.record_turn(query, response)
        follow_up = router._suggest_follow_up(query, response)
        if follow_up:
            await asyncio.sleep(0.5)
            indicator.add_log("info", follow_up)
    bus.on("cursor_response", on_cursor_response)

    # ── AI OS: Reminder events ────────────────────────────────────
    async def _on_reminder_due(label: str = "", task_id: str = "", **_kw) -> None:
        msg = f"Boss, reminder: {label}"
        indicator.add_log("reminder", msg)
        bus.emit_long("response_ready", text=msg)
        logger.info("Reminder delivered: '%s' (id=%s)", label, task_id)
    bus.on("reminder_due", _on_reminder_due)

    # ── Shutdown + child process cleanup ─────────────────────────
    async def on_shutdown(**_kw) -> None:
        logger.info("Shutdown requested")
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
        try:
            import psutil
            current = psutil.Process()
            for child in current.children(recursive=True):
                try:
                    child.terminate()
                except Exception:
                    pass
            _, alive = psutil.wait_procs(current.children(), timeout=2)
            for p in alive:
                try:
                    p.kill()
                except Exception:
                    pass
            if alive:
                logger.info("Force-killed %d lingering child processes", len(alive))
        except Exception:
            logger.debug("Child process cleanup failed", exc_info=True)
        shutdown_event.set()
    bus.on("shutdown_requested", on_shutdown)

    # ── Mic status + auto-recover ────────────────────────────────
    async def update_mic_on_listen(old, new, **_kw) -> None:
        if new is AtomState.LISTENING:
            indicator.set_mic_name(stt.mic_name)

    async def auto_recover_to_listening(old, new, **_kw) -> None:
        if new is AtomState.IDLE and state.always_listen:
            logger.info("Always-listen recovery: IDLE -> LISTENING")
            await asyncio.sleep(1)
            if state.current is AtomState.IDLE and not shutdown_event.is_set():
                await state.transition(AtomState.LISTENING)

    async def on_mic_changed(name: str = "", **_kw) -> None:
        indicator.set_mic_name(name or stt.mic_name)
    bus.on("state_changed", update_mic_on_listen)
    bus.on("state_changed", auto_recover_to_listening)
    bus.on("mic_changed", on_mic_changed)

    return {
        "perceived": _perceived,
        "proactive_state": _proactive_state,
        "didnt_catch_count": _didnt_catch_count,
        "last_perceived_ms": _last_perceived_ms,
    }


async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="ATOM - Personal Cognitive AI OS")
    parser.add_argument("--v3", action="store_true", help="Run in V3 multi-process mode")
    parser.add_argument("--v4", action="store_true", help="Run in V4 Cognitive OS mode")
    args = parser.parse_args()

    from core.logging_setup import setup_logging
    setup_logging()

    llm_queue = None
    runtime_watchdog = None
    priority_sched = None

    config = _load_config()

    from core.config_schema import validate_and_log
    validate_and_log(config)

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
    _set_adaptive_owner(
        name=owner_cfg.get("name", "Satyam"),
        title=owner_cfg.get("title", "Boss"),
    )

    executor = ThreadPoolExecutor(
        max_workers=config.get("executor", {}).get("max_workers", 3),
        thread_name_prefix="atom",
    )
    asyncio.get_running_loop().set_default_executor(executor)

    if args.v3 or args.v4:
        logger.info("Initializing V3 ZmqEventBus...")
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

    if args.v3 or args.v4:
        bus.start()
    state = StateManager(bus)
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
    )
    brain_mode_mgr.attach_security(router._security)
    assistant_mode_mgr.attach_security(router._security)

    from core.fast_path import startup_warm_up
    startup_warm_up(intent_engine, cache, memory, config)

    if args.v3 or args.v4:
        logger.info("V3 Mode: Initializing Proxies (STT, TTS, Brain)...")
        from core.ipc.proxies import TTSProxy, STTProxy, BrainProxy
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
        if tts_engine == "kokoro":
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
        elif tts_engine != "kokoro":
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
    if brain_enabled:
        if args.v3 or args.v4:
            logger.info("V3 Mode: Initializing BrainProxy...")
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
                from core.gpu_execution_coordinator import get_gpu_execution_coordinator
                from core.rag.prefetch_engine import RagPrefetchEngine
                from core.rag.rag_engine import RagEngine

                _mg_path = (config.get("memory") or {}).get(
                    "graph_db_path", "data/atom_memory.db",
                )
                shared_memory_graph = MemoryGraph(db_path=_mg_path)
                local_brain.attach_memory_graph(shared_memory_graph)
                _rag_cfg = config.get("rag") or {}
                if _rag_cfg.get("enabled", True):
                    gpu_exec = get_gpu_execution_coordinator(bus, config, metrics)
                    rag_engine = RagEngine(
                        config, vector_store=None, coordinator=gpu_exec,
                    )
                    rag_engine.set_memory_graph(shared_memory_graph)
                    rag_engine.set_feedback_engine(feedback_engine)
                    prefetch_eng = RagPrefetchEngine(rag_engine, config)
                    local_brain.attach_rag(rag_engine, gpu_exec)
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

    gpu_rm = None
    gpu_sched_v7 = None
    recovery_mgr = None
    gpu_stall_wd = None
    if (config.get("v7_gpu") or {}).get("enabled", True):
        from core.gpu_resource_manager import GPUResourceManager
        from core.gpu_scheduler import GPUScheduler
        from core.recovery_manager import RecoveryManager
        from core.gpu_watchdog import GPUStallWatchdog

        gpu_rm = GPUResourceManager(bus, config)
        gpu_sched_v7 = GPUScheduler(priority_sched, gpu_resource_manager=gpu_rm)
        recovery_mgr = RecoveryManager(bus, config)
        gpu_stall_wd = GPUStallWatchdog(bus, config)
        gpu_stall_wd.start()
        gpu_rm.start_power_task()
        if brain_enabled and local_brain is not None:
            local_brain.attach_gpu_resource_manager(gpu_rm)
        logger.info(
            "ATOM V7: GPUResourceManager + GPUScheduler + RecoveryManager + GPUStallWatchdog",
        )
        _ = (gpu_sched_v7, recovery_mgr, gpu_stall_wd)

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
    security = SecurityPolicy(config)

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
    prompt_builder.set_context_sources(
        context_fusion=context_fusion,
        real_world_intel=real_world_intel,
    )

    # ── ActionExecutor (bridges LLM tool calls -> Router dispatch) ──
    if brain_enabled and local_brain is not None and not (args.v3 or args.v4):
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
        logger.info("Screen reader: %s", "OCR available" if screen_reader.is_available else "fallback mode")

    # ── GPU Governor ───────────────────────────────────────────────
    gpu_governor = None
    if config.get("gpu", {}).get("enabled", True):
        from core.gpu_governor import GPUGovernor
        gpu_governor = GPUGovernor(bus, config)
        if gpu_governor.is_available:
            logger.info("GPU Governor: monitoring active")
        else:
            logger.info("GPU Governor: NVML not available (pynvml not installed)")

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
        "encrypted" if security_fortress._vault.is_encrypted else "obfuscated",
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

    logger.info(
        "JARVIS intelligence initialized: PlatformAdapter(%s) + "
        "SystemScanner + SystemIndexer + MediaWatcher + OwnerUnderstanding",
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

    jarvis_core = JarvisCore(bus, owner_understanding, system_scanner, config)
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
                tts_cfg.get("edge_voice", "Edge")
                if tts_engine == "edge"
                else "SAPI (offline)"
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
            lambda: asyncio.ensure_future(_execute_mode_switch(new_mode))
        )

    indicator.set_shutdown_callback(_ui_shutdown_callback)
    if hasattr(indicator, "set_mode_change_callback"):
        indicator.set_mode_change_callback(_on_mode_change)

    if cognitive_enabled and personality_modes and hasattr(indicator, "set_personality_mode_callback"):
        def _on_personality_mode_from_ui(mode: str) -> None:
            running_loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(
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

    await tts.init_voice()

    warmup_tasks = []
    if local_brain and local_brain.available:
        logger.info("Local LLM pre-warm (background load)...")
        warmup_tasks.append(local_brain.warm_up())
    if warmup_tasks:
        await asyncio.gather(*warmup_tasks)
    else:
        logger.info("No brains to pre-warm")

    stt_preload_done = asyncio.Event()

    async def _background_stt_preload() -> None:
        t0 = time.monotonic()
        logger.info("STT model loading in background...")

        loop = asyncio.get_running_loop()
        if not (args.v3 or args.v4):
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
        stt_preload_done.set()

    _bg_tasks: list[asyncio.Task] = []

    if config.get("stt", {}).get("preload", True):
        _bg_tasks.append(asyncio.create_task(_background_stt_preload()))
    else:
        stt_preload_done.set()

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
    bus.on("state_changed", runtime_watchdog.on_state_changed)

    # ── Wire all event handlers (extracted for testability) ────────
    _wiring_ctx = _wire_events(
        bus=bus, state=state, stt=stt, tts=tts, router=router,
        indicator=indicator, cache=cache, memory=memory, metrics=metrics,
        config=config, local_brain=local_brain, llm_queue=llm_queue,
        assistant_mode_mgr=assistant_mode_mgr,
        behavior=behavior,
        scheduler=scheduler, process_mgr=process_mgr, evolution=evolution,
        priority_sched=priority_sched,
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
                        lambda: asyncio.ensure_future(state.transition(AtomState.IDLE))
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

    if gpu_governor is not None and gpu_governor.is_available:
        gpu_governor.start()

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

    # ── Wire event handlers ────────────────────────────────────────
    async def _on_document_ingest(path: str = "", **_kw) -> None:
        if document_engine.is_ready:
            result = await document_engine.ingest(path)
            status = result.get("status", result.get("error", "unknown"))
            if result.get("status") == "success":
                msg = f"Learned from '{result['name']}': {result['chunks']} knowledge chunks stored."
            elif result.get("status") == "already_ingested":
                msg = f"I already know '{result['name']}', Boss."
            else:
                msg = f"Document ingestion issue: {status}"
            bus.emit_long("response_ready", text=msg)
        else:
            bus.emit_long("response_ready", text="Document learning isn't available right now, Boss.")
    bus.on("document_ingest_request", _on_document_ingest)

    async def _on_workflow_start(name: str = "", **_kw) -> None:
        msg = workflow_engine.start_recording(name)
        bus.emit_long("response_ready", text=msg)
    bus.on("workflow_start_recording", _on_workflow_start)

    async def _on_workflow_stop(**_kw) -> None:
        msg = workflow_engine.stop_recording()
        bus.emit_long("response_ready", text=msg)
    bus.on("workflow_stop_recording", _on_workflow_stop)

    async def _on_workflow_list(**_kw) -> None:
        msg = workflow_engine.list_workflows()
        bus.emit_long("response_ready", text=msg)
    bus.on("workflow_list_request", _on_workflow_list)

    async def _on_workflow_replay(name: str = "", **_kw) -> None:
        steps = workflow_engine.get_replay_steps(name)
        if steps is None:
            bus.emit_long("response_ready", text=f"No workflow named '{name}' found, Boss.")
            return
        bus.emit_long("response_ready", text=f"Running workflow '{name}' ({len(steps)} steps).")
        for step in steps:
            try:
                router._dispatch_action(step.action, step.args)
                await asyncio.sleep(max(0.3, step.delay_ms / 1000))
            except Exception as e:
                logger.warning("Workflow step failed: %s -> %s", step.action, e)
    bus.on("workflow_replay_request", _on_workflow_replay)

    async def _on_screen_read(**_kw) -> None:
        if screen_reader is not None:
            summary = screen_reader.get_screen_summary()
            bus.emit_long("response_ready", text=summary)
        else:
            bus.emit_long("response_ready", text="Screen reading isn't available, Boss.")
    bus.on("screen_read_request", _on_screen_read)

    async def _on_dream_summary(**_kw) -> None:
        if dream_engine is not None:
            msg = dream_engine.get_dream_summary()
            bus.emit_long("response_ready", text=msg)
        else:
            bus.emit_long("response_ready", text="Dream engine isn't active, Boss.")
    bus.on("dream_summary_request", _on_dream_summary)

    if cognitive_enabled and dream_engine is not None:
        async def _on_cursor_response_for_dream(query: str = "", response: str = "", **_kw) -> None:
            emotion_state = emotion_detector.current_emotion if emotion_detector else "neutral"
            dream_engine.record_interaction(query, response, emotion=emotion_state)
        bus.on("cursor_response", _on_cursor_response_for_dream)

    if cognitive_enabled and curiosity_engine is not None:
        async def _on_intent_for_curiosity(intent: str = "", text: str = "", **_kw) -> None:
            if text:
                for word in text.lower().split():
                    if len(word) > 5:
                        curiosity_engine.track_topic(word)
        bus.on("intent_classified", _on_intent_for_curiosity)

        async def _on_curiosity_question(text: str = "", **_kw) -> None:
            bus.emit_long("response_ready", text=text)
        bus.on("curiosity_question", _on_curiosity_question)

    if workflow_engine is not None:
        async def _on_action_for_workflow(intent: str = "", **kw) -> None:
            if workflow_engine.is_recording and intent not in ("confirm", "deny", "fallback"):
                workflow_engine.record_action(intent, kw.get("action_args", {}), intent)
        bus.on("intent_classified", _on_action_for_workflow)

    # ── JARVIS Core event handlers ──────────────────────────────
    async def _on_jarvis_insight(message: str = "", category: str = "",
                                  action: str = "", action_args: dict = None,
                                  **_kw) -> None:
        if message:
            indicator.add_log("jarvis", f"[{category}] {message}")
            bus.emit_long("response_ready", text=message)
            if action and action_args:
                try:
                    router._dispatch_action(action, action_args or {})
                except Exception:
                    logger.debug("JARVIS insight action failed", exc_info=True)
    bus.on("jarvis_insight", _on_jarvis_insight)

    async def _on_system_scan_request(**_kw) -> None:
        summary = system_scanner.get_scan_summary()
        bus.emit_long("response_ready", text=summary)
    bus.on("system_scan_request", _on_system_scan_request)

    async def _on_system_control_request(action: str = "", **kw) -> None:
        result = None
        if action == "hardware_details":
            result = system_control.get_hardware_details()
        elif action == "uptime":
            result = system_control.get_system_uptime()
        elif action == "optimize":
            result = system_control.optimize_for_atom()
        elif action == "open_ports":
            result = system_control.get_open_ports()
        elif action == "wifi_scan":
            result = system_control.get_wifi_networks()
        elif action == "temp_files":
            result = system_control.analyze_temp_files()
        elif action == "startup_programs":
            result = system_control.list_startup_programs()
        elif action == "network_speed":
            result = system_control.get_network_speed()
        elif action == "full_status":
            msg = system_control.get_full_status()
            bus.emit_long("response_ready", text=msg)
            return
        elif action == "find_process":
            name = kw.get("name", "")
            result = system_control.find_process_by_name(name)
        elif action == "process_details":
            pid = kw.get("pid", 0)
            result = system_control.get_process_details(pid)
        elif action == "power_plan":
            plan = kw.get("plan", "balanced")
            result = system_control.set_power_plan(plan)

        if result:
            bus.emit_long("response_ready", text=result.message)
        else:
            bus.emit_long("response_ready",
                          text=f"Unknown system control action: {action}")
    bus.on("system_control_request", _on_system_control_request)

    async def _on_owner_summary_request(**_kw) -> None:
        summary = owner_understanding.get_owner_summary()
        bus.emit_long("response_ready", text=summary)
    bus.on("owner_summary_request", _on_owner_summary_request)

    # ── Self-Healing + Code Introspection + Security Event Handlers ──

    async def _on_diagnose_failure(intent: str = "", **kw) -> None:
        if intent == "diagnose_failure":
            msg = self_healing.diagnose_failure()
            bus.emit_long("response_ready", text=msg)

    async def _on_fix_it(intent: str = "", **kw) -> None:
        if intent == "fix_it":
            msg = self_healing.fix_last_failure()
            bus.emit_long("response_ready", text=msg)

    async def _on_fix_all(intent: str = "", **kw) -> None:
        if intent == "fix_all":
            msg = self_healing.fix_all()
            bus.emit_long("response_ready", text=msg)

    async def _on_module_health(intent: str = "", **kw) -> None:
        if intent == "module_health":
            msg = self_healing.get_health_summary()
            bus.emit_long("response_ready", text=msg)

    async def _on_read_own_code(intent: str = "", **kw) -> None:
        if intent == "read_own_code":
            if not code_introspector.is_scanned:
                code_introspector.scan()
            msg = code_introspector.explain_architecture()
            health = code_introspector.format_code_health()
            bus.emit_long("response_ready", text=f"{msg} {health}")

    async def _on_explain_module(intent: str = "", **kw) -> None:
        if intent == "explain_module":
            args = kw.get("action_args", {}) or {}
            module = args.get("module", "")
            if module:
                if not code_introspector.is_scanned:
                    code_introspector.scan()
                msg = code_introspector.explain_module(module)
                bus.emit_long("response_ready", text=msg)

    async def _on_search_code(intent: str = "", **kw) -> None:
        if intent == "search_code":
            args = kw.get("action_args", {}) or {}
            query = args.get("query", "")
            if query:
                if not code_introspector.is_scanned:
                    code_introspector.scan()
                results = code_introspector.search_code(query)
                if results:
                    lines = [f"{r['file']}:{r['line']}: {r['content']}" for r in results[:5]]
                    msg = f"Found {len(results)} matches. " + " | ".join(lines)
                else:
                    msg = f"No matches found for '{query}' in the codebase."
                bus.emit_long("response_ready", text=msg)

    async def _on_security_status(intent: str = "", **kw) -> None:
        if intent == "security_status":
            msg = security_fortress.get_security_status()
            integrity_ok, integrity_msg = security_fortress.check_integrity()
            bus.emit_long("response_ready", text=f"{msg} Integrity: {integrity_msg}")

    async def _on_security_lockdown(intent: str = "", **kw) -> None:
        if intent == "security_lockdown":
            security_fortress.log_security_event("lockdown_activated", severity="HIGH")
            bus.emit_long("response_ready",
                          text="Security lockdown activated, Boss. All sensitive operations require authentication.")

    async def _on_failure_report(intent: str = "", **kw) -> None:
        if intent in ("self_diagnostic", "self_check"):
            failure_report = self_healing.get_failure_report()
            if "No failures" not in failure_report:
                await asyncio.sleep(0.5)
                bus.emit_long("response_ready", text=failure_report)

    bus.on("intent_classified", _on_diagnose_failure)
    bus.on("intent_classified", _on_fix_it)
    bus.on("intent_classified", _on_fix_all)
    bus.on("intent_classified", _on_module_health)
    bus.on("intent_classified", _on_read_own_code)
    bus.on("intent_classified", _on_explain_module)
    bus.on("intent_classified", _on_search_code)
    bus.on("intent_classified", _on_security_status)
    bus.on("intent_classified", _on_security_lockdown)
    bus.on("intent_classified", _on_failure_report)

    # ── Voice Auth + Behavioral Auth Event Handlers ────────────────

    async def _on_voice_enroll(intent: str = "", **kw) -> None:
        if intent == "voice_enroll":
            if not security_fortress.voice_auth.is_available:
                bus.emit_long("response_ready",
                              text="Voice authentication isn't available, Boss. "
                                   "Install resemblyzer or numpy to enable it.")
                return
            bus.emit_long("response_ready",
                          text="Starting voice enrollment. Say a few natural sentences "
                               "and I'll learn your voice, Boss. I need at least 3 samples.")
            security_fortress.log_security_event(
                "voice_enroll_started", severity="INFO",
            )

    async def _on_voice_verify(intent: str = "", **kw) -> None:
        if intent == "voice_verify":
            if not security_fortress.voice_auth.is_enrolled:
                bus.emit_long("response_ready",
                              text="Voice not enrolled yet, Boss. "
                                   "Say 'enroll my voice' to set up voice authentication.")
                return
            bus.emit_long("response_ready",
                          text="Voice verification available. Your last speech will be "
                               "compared against your voice print. "
                               + security_fortress.voice_auth.get_status_message())

    async def _on_voice_auth_status(intent: str = "", **kw) -> None:
        if intent == "voice_auth_status":
            msg = security_fortress.voice_auth.get_status_message()
            bus.emit_long("response_ready", text=msg)

    async def _on_voice_reset(intent: str = "", **kw) -> None:
        if intent == "voice_reset":
            msg = security_fortress.voice_reset()
            bus.emit_long("response_ready", text=msg)

    async def _on_behavior_auth_status(intent: str = "", **kw) -> None:
        if intent == "behavior_auth_status":
            msg = security_fortress.behavior_auth.get_anomaly_report()
            bus.emit_long("response_ready", text=msg)

    async def _on_intent_for_behavior_auth(intent: str = "", **kw) -> None:
        """Feed every intent into behavioral auth for continuous verification."""
        if intent and intent not in ("confirm", "deny", "empty"):
            text = kw.get("text", "")
            target = kw.get("target", "") or kw.get("name", "")
            try:
                active_app = ""
                if hasattr(context_engine, "get_active_window"):
                    active_app = context_engine.get_active_window() or ""
            except Exception:
                active_app = ""
            security_fortress.observe_behavior(
                action=intent,
                detail=target,
                query_text=text,
                active_app=active_app,
            )

    bus.on("intent_classified", _on_voice_enroll)
    bus.on("intent_classified", _on_voice_verify)
    bus.on("intent_classified", _on_voice_auth_status)
    bus.on("intent_classified", _on_voice_reset)
    bus.on("intent_classified", _on_behavior_auth_status)
    bus.on("intent_classified", _on_intent_for_behavior_auth)

    async def _on_behavior_reauth_needed(**_kw) -> None:
        bus.emit_long("response_ready",
                      text="Boss, your usage patterns seem different from usual. "
                           "For security, please verify your identity. "
                           "Say 'verify my voice' or authenticate with your passphrase.")
        security_fortress.log_security_event(
            "behavioral_reauth_triggered",
            f"trust={security_fortress.trust_score:.2f}",
            severity="HIGH",
        )
    bus.on("reauth_required", _on_behavior_reauth_needed)

    def _reauth_callback() -> None:
        bus.emit_long("reauth_required")
    security_fortress.behavior_auth.set_reauth_callback(_reauth_callback)

    logger.info("Auth handlers wired: voice enrollment, verification, behavioral auth")
    logger.info("Security event handlers wired: self-healing, code introspection, fortress")

    # ── Real-World Intelligence Event Handlers ────────────────────
    async def _on_weather_request(intent: str = "", **_kw) -> None:
        if intent == "weather_report":
            msg = real_world_intel.get_weather_summary()
            bus.emit_long("response_ready", text=msg)

    async def _on_news_request(intent: str = "", **_kw) -> None:
        if intent == "news_headlines":
            msg = real_world_intel.get_news_summary(count=5)
            bus.emit_long("response_ready", text=msg)

    async def _on_world_clock_request(intent: str = "", **_kw) -> None:
        if intent == "world_clock":
            msg = real_world_intel.get_world_clock_summary()
            bus.emit_long("response_ready", text=msg)

    async def _on_briefing_request(intent: str = "", **_kw) -> None:
        if intent == "daily_briefing":
            msg = real_world_intel.get_briefing()
            bus.emit_long("response_ready", text=msg)

    async def _on_temporal_request(intent: str = "", **_kw) -> None:
        if intent == "temporal_info":
            msg = real_world_intel.get_temporal_summary()
            bus.emit_long("response_ready", text=msg)

    async def _on_world_status_request(intent: str = "", **_kw) -> None:
        if intent == "world_status":
            ctx = real_world_intel.get_world_context()
            parts = [real_world_intel.get_temporal_summary()]
            if not ctx.weather.is_stale:
                parts.append(real_world_intel.get_weather_summary())
            if ctx.headlines:
                parts.append(real_world_intel.get_news_summary(3))
            parts.append(real_world_intel.get_world_clock_summary())
            parts.append(f"World intelligence quality: {ctx.quality_score():.0%}")
            bus.emit_long("response_ready", text=" ".join(parts))

    async def _on_intent_for_context_fusion(intent: str = "", **kw) -> None:
        """Feed every intent into ContextFusion for action logging."""
        if intent and intent not in ("confirm", "deny", "empty"):
            context_fusion.log_action(intent, kw.get("text", "")[:60])

    bus.on("intent_classified", _on_weather_request)
    bus.on("intent_classified", _on_news_request)
    bus.on("intent_classified", _on_world_clock_request)
    bus.on("intent_classified", _on_briefing_request)
    bus.on("intent_classified", _on_temporal_request)
    bus.on("intent_classified", _on_world_status_request)
    bus.on("intent_classified", _on_intent_for_context_fusion)

    logger.info("Real-world intelligence handlers wired: weather, news, briefing, world clock")

    if emotion_detector is not None and emotion_detector.is_enabled:
        async def _on_speech_for_emotion(text: str = "", **_kw) -> None:
            result = emotion_detector.analyze_text_emotion(text)
            if result.emotion != "neutral":
                bus.emit_fast("user_emotion_detected",
                              emotion=result.emotion, confidence=result.confidence)
        bus.on("speech_final", _on_speech_for_emotion)

    if wake_word_engine is not None:
        async def _on_wake_word(**_kw) -> None:
            from core.state_manager import AtomState
            if state.current in (AtomState.IDLE, AtomState.SLEEP):
                await state.transition(AtomState.LISTENING)
                indicator.add_log("info", "Wake word detected! Listening, Boss.")
        bus.on("wake_word_detected", _on_wake_word)

    if web_dashboard is not None:
        async def _on_governor_throttle_ui(**_kw):
            web_dashboard.broadcast_governor(True)
        async def _on_governor_normal_ui(**_kw):
            web_dashboard.broadcast_governor(False)
        bus.on("governor_throttle", _on_governor_throttle_ui)
        bus.on("governor_normal", _on_governor_normal_ui)

    async def _on_governor_throttle_tts(**_kw) -> None:
        if hasattr(tts, "set_postprocess"):
            tts.set_postprocess(False)
            logger.info("Governor: TTS post-processing disabled (throttled)")
    async def _on_governor_normal_tts(**_kw) -> None:
        if hasattr(tts, "restore_postprocess"):
            tts.restore_postprocess()
            logger.info("Governor: TTS post-processing restored to config")
    bus.on("governor_throttle", _on_governor_throttle_tts)
    bus.on("governor_normal", _on_governor_normal_tts)

    # ── Autonomy event handlers ───────────────────────────────────────
    _pending_habit_id = {"id": ""}

    async def _on_habit_suggestion(text: str = "", habit_id: str = "",
                                   confidence: float = 0.0, **_kw) -> None:
        _pending_habit_id["id"] = habit_id
        indicator.add_log("info", f"[habit] {text}")
        bus.emit_long("response_ready", text=text)

    async def _on_intent_for_habit_feedback(intent: str = "", **_kw) -> None:
        hid = _pending_habit_id.get("id", "")
        if not hid:
            return
        if intent == "confirm":
            bus.emit_fast("user_feedback", habit_id=hid, accepted=True)
            _pending_habit_id["id"] = ""
        elif intent == "deny":
            bus.emit_fast("user_feedback", habit_id=hid, accepted=False)
            _pending_habit_id["id"] = ""

    bus.on("intent_classified", _on_intent_for_habit_feedback)

    async def _on_autonomous_action(action: str = "", target: str = "",
                                    habit_id: str = "",
                                    confidence: float = 0.0, **_kw) -> None:
        msg = f"Auto-executing {action.replace('_', ' ')}"
        if target:
            msg += f" for {target}"
        indicator.add_log("action", f"[auto] {msg}")
        try:
            args = {"name": target, "exe": target, "process": target}
            result = router._dispatch_action(action, args)
            response = result or (msg + ", Boss.")
            bus.emit_long("response_ready", text=response)
        except Exception as exc:
            logger.warning("Autonomous action failed: %s", exc)
            bus.emit_long("response_ready",
                          text=f"Tried to auto-execute {action}, but it failed, Boss.")

    async def _on_autonomy_decision_log(decision: str = "", detail: str = "",
                                        confidence: float = 0.0, **_kw) -> None:
        if web_dashboard is not None:
            web_dashboard.broadcast_autonomy_log(decision, detail, confidence)

    async def _on_intent_for_memory(intent: str = "", **kw) -> None:
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=0)
            ram = psutil.virtual_memory().percent
        except Exception:
            cpu, ram = 0, 0
        memory.log_interaction(
            command=kw.get("text", ""),
            action=intent,
            system_state={"cpu": cpu, "ram": ram},
        )

    bus.on("habit_suggestion", _on_habit_suggestion)
    bus.on("autonomous_action", _on_autonomous_action)
    bus.on("autonomy_decision_log", _on_autonomy_decision_log)
    bus.on("intent_classified", _on_intent_for_memory)

    # ── Cognitive Layer event handlers ─────────────────────────────
    if cognitive_enabled:
        async def _on_cognitive_intent(intent: str = "", **kw) -> None:
            """Dispatch cognitive-layer intents to the right module."""
            args = kw.get("action_args", {}) or {}

            if intent == "goal_create":
                title = args.get("title", "")
                if title:
                    result = goal_engine.create_goal(title)
                    if "error" in result:
                        bus.emit_long("response_ready", text=result["error"])
                    else:
                        msg = f"Goal set: '{title}'. Say 'break down this goal' for a step plan."
                        bus.emit_long("response_ready", text=msg)
                return

            if intent == "goal_show":
                summary = goal_engine.format_goals_summary()
                bus.emit_long("response_ready", text=summary)
                return

            if intent == "goal_progress":
                target = args.get("target", "")
                if target:
                    goal = goal_engine.find_goal(target)
                    if goal:
                        ev = goal.get("evaluation", {})
                        msg = (
                            f"Goal '{goal['title']}': {goal['progress_pct']}% done. "
                            f"Trajectory: {ev.get('trajectory', 'unknown')}. "
                            f"Streak: {ev.get('streak_days', 0)} days."
                        )
                        bus.emit_long("response_ready", text=msg)
                    else:
                        bus.emit_long("response_ready", text="I couldn't find that goal, Boss.")
                else:
                    briefing = goal_engine.get_daily_briefing()
                    bus.emit_long("response_ready", text=briefing or "No active goals to report on.")
                return

            if intent == "goal_decompose":
                active = goal_engine.get_active_goals()
                if active:
                    bus.emit_long("response_ready", text="Breaking down your latest goal with AI...")
                    result = await goal_engine.decompose_with_llm(active[-1]["id"])
                    bus.emit_long("response_ready", text=result)
                else:
                    bus.emit_long("response_ready", text="No active goals to decompose, Boss.")
                return

            if intent == "goal_log_progress":
                topic = args.get("topic", "")
                minutes = args.get("minutes", 30)
                active = goal_engine.get_active_goals()
                if active:
                    goal = active[0]
                    steps = goal.get("steps", [])
                    matched_step = None
                    for s in steps:
                        if topic.lower() in s["title"].lower():
                            matched_step = s
                            break
                    if matched_step:
                        result = goal_engine.log_progress(goal["id"], matched_step["id"], minutes)
                        bus.emit_long("response_ready", text=result)
                    elif steps:
                        result = goal_engine.log_progress(goal["id"], steps[0]["id"], minutes)
                        bus.emit_long("response_ready", text=result)
                    else:
                        bus.emit_long("response_ready",
                                      text=f"Logged {minutes} minutes. Add steps to track properly.")
                else:
                    bus.emit_long("response_ready", text="No active goals to log progress on.")
                return

            if intent == "goal_complete_step":
                step_name = args.get("step_name", "")
                active = goal_engine.get_active_goals()
                for goal in active:
                    for step in goal.get("steps", []):
                        if step_name.lower() in step["title"].lower():
                            result = goal_engine.complete_step(goal["id"], step["id"])
                            bus.emit_long("response_ready", text=result)
                            return
                bus.emit_long("response_ready", text="Couldn't find that step, Boss.")
                return

            if intent == "goal_pause":
                target = args.get("target", "")
                goal = goal_engine.find_goal(target) if target else None
                if goal:
                    result = goal_engine.pause_goal(goal["id"])
                    bus.emit_long("response_ready", text=result)
                else:
                    bus.emit_long("response_ready", text="Goal not found, Boss.")
                return

            if intent == "goal_resume":
                target = args.get("target", "")
                goal = goal_engine.find_goal(target) if target else None
                if goal:
                    result = goal_engine.resume_goal(goal["id"])
                    bus.emit_long("response_ready", text=result)
                else:
                    bus.emit_long("response_ready", text="Goal not found, Boss.")
                return

            if intent == "goal_abandon":
                target = args.get("target", "")
                goal = goal_engine.find_goal(target) if target else None
                if goal:
                    result = goal_engine.abandon_goal(goal["id"])
                    bus.emit_long("response_ready", text=result)
                else:
                    bus.emit_long("response_ready", text="Goal not found, Boss.")
                return

            if intent == "prediction":
                summary = prediction_engine.format_predictions()
                bus.emit_long("response_ready", text=summary)
                return

            if intent == "mode_switch":
                mode = args.get("mode", "work")
                result = personality_modes.switch_mode(mode)
                bus.emit_long("response_ready", text=result)
                return

            if intent == "cognitive_behavior_report":
                report = behavior_model.get_profile_summary()
                bus.emit_long("response_ready", text=report)
                return

            if intent == "scheduling_advice":
                advice = behavior_model.get_scheduling_advice()
                bus.emit_long("response_ready", text=advice)
                return

            if intent == "brain_remember":
                fact = args.get("fact", "")
                if fact:
                    second_brain.learn_fact(fact, source="voice")
                    bus.emit_long("response_ready",
                                  text=f"Got it, Boss. I'll remember: {fact[:80]}")
                return

            if intent == "brain_recall":
                query = args.get("query", "")
                if query:
                    results = second_brain.retrieve(query, k=3)
                    if results:
                        formatted = ". ".join(r for r in results)
                        bus.emit_long("response_ready",
                                      text=f"Here's what I know: {formatted}")
                    else:
                        bus.emit_long("response_ready",
                                      text="I don't have anything stored on that yet, Boss.")
                return

            if intent == "brain_preferences":
                prefs = second_brain.preferences
                if prefs:
                    items = [f"{k}: {v}" for k, v in list(prefs.items())[:8]]
                    bus.emit_long("response_ready",
                                  text=f"Your preferences: {', '.join(items)}")
                else:
                    bus.emit_long("response_ready",
                                  text="No preferences stored yet. I'm still learning, Boss.")
                return

            if intent == "self_optimize":
                report = self_optimizer.format_optimization_report()
                bus.emit_long("response_ready", text=report)
                return

        bus.on("intent_classified", _on_cognitive_intent)

        async def _on_habit_suggestion_mode_gate(text: str = "", **kw) -> None:
            """Gate habit suggestions through personality mode."""
            if personality_modes and not personality_modes.should_allow_suggestion():
                personality_modes.queue_suggestion({"text": text, **kw})
                return
        bus.on("habit_suggestion", _on_habit_suggestion_mode_gate)

        async def _on_cursor_response_for_brain(query: str = "", response: str = "", **_kw) -> None:
            if "goal_decompose:" not in query and len(response) > 20:
                second_brain.learn_fact(
                    f"Q: {query[:100]} A: {response[:200]}",
                    source="llm_conversation",
                    tags=["conversation"],
                )
        bus.on("cursor_response", _on_cursor_response_for_brain)

        async def _on_goal_briefing(text: str = "", **_kw) -> None:
            if text:
                indicator.add_log("info", f"[briefing] {text}")
                bus.emit_long("response_ready", text=text)
        bus.on("goal_briefing", _on_goal_briefing)

        async def _on_mode_changed(mode: str = "", **_kw) -> None:
            indicator.add_log("action", f"Mode: {mode.upper()}")
            if hasattr(tts, "_rate_override"):
                rate_adj = _kw.get("voice_rate_adj", 0)
                tts._rate_override = f"{rate_adj:+d}%" if rate_adj else None
            if web_dashboard is not None:
                web_dashboard.broadcast_mode(personality_modes.get_mode_for_dashboard())
        bus.on("mode_changed", _on_mode_changed)

        async def _on_prediction_ready(predictions: list = None, **_kw) -> None:
            if web_dashboard is not None and predictions:
                web_dashboard.broadcast_predictions(predictions)
        bus.on("prediction_ready", _on_prediction_ready)

        async def _on_optimization_suggestions(suggestions: list = None, **_kw) -> None:
            if suggestions:
                logger.info("Self-optimizer: %d suggestions generated", len(suggestions))
                for s in suggestions[:2]:
                    indicator.add_log("info", f"[optimize] {s.get('message', '')}")
        bus.on("optimization_suggestions", _on_optimization_suggestions)

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

        greeting = (
            f"{time_g} "
            f"All systems online. {sys_summary}.{scan_health} "
            f"Security: {security_label}, {'integrity clean' if integrity_ok else 'integrity alert'}. "
            f"Running in {mode_label} mode with {cap_count} tools ready.{cognitive_msg}"
            f"{weather_line}{holiday_line}{news_line} "
            f"I know the world, I know you, and I only answer to you, Boss. What do you need?"
        )

        try:
            import psutil
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
        
        if args.v3 or args.v4:
            # In V3, STT worker handles listening. We just keep main process alive.
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
            import psutil
        except ImportError:
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
                    import psutil
                    bat = psutil.sensors_battery()
                    if bat and bat.percent <= 20 and not bat.power_plugged:
                        if not _proactive_state["low_battery_warned"]:
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
        if gpu_governor is not None:
            gpu_governor.shutdown()
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
    _orig_load = _load_config

    def _merged_config() -> dict:
        base = _orig_load()
        if config_overrides:
            for key, val in config_overrides.items():
                if isinstance(val, dict) and isinstance(base.get(key), dict):
                    base[key].update(val)
                else:
                    base[key] = val
        return base

    globals()["_load_config"] = _merged_config

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

    globals()["_load_config"] = _orig_load


if __name__ == "__main__":
    run_atom()
