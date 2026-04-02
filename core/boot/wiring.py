"""
ATOM -- Core Event Wiring

Extracted event bus attachments from the main entry point.
"""
from __future__ import annotations
import asyncio
import logging
import time

logger = logging.getLogger("atom.wiring")

def wire_events(
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
    v3: bool = False,
    v4: bool = False,
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

        if not v4:
            bus.on("speech_final", _speech_via_priority)
    else:
        async def _speech_final_direct(text: str, **kw) -> None:
            if shutdown_event.is_set():
                return
            if local_brain is not None:
                local_brain.request_preempt()
            await router.on_speech(text, **kw)

        if not v4:
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
                if not (v3 or v4):
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
        _thinking_progress_task["task"] = asyncio.create_task(_thinking_progress_loop())

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


