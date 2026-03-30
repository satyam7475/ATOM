"""ATOM -- Feature & subsystem event handler wiring.

Handles document ingestion, workflow engine, screen reading,
dream/curiosity engines, JARVIS insight, system scan/control,
owner understanding, autonomy actions, and governor throttle.

Extracted from main.py for testability.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.security_policy import SecurityPolicy
    from core.router import Router
    from core.state_manager import StateManager
    from core.memory_engine import MemoryEngine

logger = logging.getLogger("atom.wiring.features")


def wire_documents_and_workflows(
    *,
    bus: AsyncEventBus,
    router: Any,
    security: SecurityPolicy,
    document_engine: Any,
    workflow_engine: Any,
    screen_reader: Any | None = None,
) -> None:
    """Wire document ingestion, workflow, and screen reader handlers."""

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
            allowed, reason = security.allow_action(step.action, step.args)
            if not allowed:
                logger.warning(
                    "SecurityPolicy blocked workflow step: %s (%s)",
                    step.action, reason,
                )
                bus.emit_long(
                    "response_ready",
                    text=f"Blocked step '{step.action}' -- {reason}",
                )
                continue
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

    if workflow_engine is not None:
        async def _on_action_for_workflow(intent: str = "", **kw) -> None:
            if workflow_engine.is_recording and intent not in ("confirm", "deny", "fallback"):
                workflow_engine.record_action(intent, kw.get("action_args", {}), intent)
        bus.on("intent_classified", _on_action_for_workflow)

    logger.info("Document + workflow + screen handlers wired")


def wire_dream_curiosity(
    *,
    bus: AsyncEventBus,
    dream_engine: Any | None = None,
    curiosity_engine: Any | None = None,
    emotion_detector: Any | None = None,
    cognitive_enabled: bool = False,
) -> None:
    """Wire dream engine, curiosity engine, and emotion detector handlers."""

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

    logger.info("Dream + curiosity + emotion handlers wired")


def wire_jarvis_and_system(
    *,
    bus: AsyncEventBus,
    router: Any,
    security: SecurityPolicy,
    indicator: Any,
    system_scanner: Any,
    system_control: Any,
    owner_understanding: Any,
) -> None:
    """Wire JARVIS insight, system scan/control, and owner understanding handlers."""

    async def _on_jarvis_insight(message: str = "", category: str = "",
                                  action: str = "", action_args: dict = None,
                                  **_kw) -> None:
        if message:
            indicator.add_log("jarvis", f"[{category}] {message}")
            bus.emit_long("response_ready", text=message)
            if action and action_args:
                allowed, reason = security.allow_action(action, action_args or {})
                if not allowed:
                    logger.warning(
                        "SecurityPolicy blocked JARVIS insight action: %s (%s)",
                        action, reason,
                    )
                    return
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

    logger.info("JARVIS + system scan/control + owner understanding handlers wired")


def wire_autonomy_and_governor(
    *,
    bus: AsyncEventBus,
    router: Any,
    indicator: Any,
    memory: MemoryEngine,
    autonomy: Any,
    state: Any,
    tts: Any,
    web_dashboard: Any | None = None,
    emotion_detector: Any | None = None,
    wake_word_engine: Any | None = None,
) -> None:
    """Wire autonomy, governor throttle, wake word, and emotion handlers."""

    _pending_habit_id: dict[str, str] = {"id": ""}

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

    if emotion_detector is not None and getattr(emotion_detector, "is_enabled", False):
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

    logger.info("Autonomy + governor + wake word + emotion handlers wired")
