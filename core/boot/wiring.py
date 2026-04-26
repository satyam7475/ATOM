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
    state_bridge=None,
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
    shutdown_event: asyncio.Event | None = None,
    scheduler=None,
    process_mgr=None,
    evolution=None,
    priority_sched=None,
    v3: bool = False,
    v4: bool = False,
    command_loop=None,
) -> dict:
    """Wire all event bus handlers. Extracted from main() for testability.

    Returns a shared-state dict used by handlers (perceived latency tracking,
    catch counter, proactive state, stream buffer).
    """
    from core.state_manager import AtomState
    from core.metrics import log_health

    if shutdown_event is None:
        shutdown_event = asyncio.Event()

    def _guard_handler(event: str, handler, *, source: str):
        async def _wrapped(**kw) -> None:
            try:
                await handler(**kw)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception(
                    "Wiring handler failed source=%s event=%s: %s",
                    source,
                    event,
                    exc,
                )
                try:
                    metrics.inc("errors_total")
                except Exception:
                    logger.debug('Error metric increment failed', exc_info=True)
        return _wrapped

    _didnt_catch_count = {"n": 0}
    _perceived = {"t_speech_final": 0.0, "logged": False}
    _last_perceived_ms = {"ms": None}
    _proactive_state = {
        "last_query_time": time.monotonic(),
        "low_battery_warned": False,
        "last_fs_hint": 0.0,
    }
    _stream_buffer = {"text": "", "stream_id": ""}
    _ttfa_gate = {"sent": False}
    _llm_latency_history: list[float] = []
    _thinking_progress_task: dict[str, asyncio.Task | None] = {"task": None}
    _LLM_HISTORY_MAX = 10

    # Global Interrupt Manager
    from voice.interrupt_handler import VoiceInterruptHandler

    voice_interrupt = VoiceInterruptHandler(
        bus=bus,
        state=state,
        tts=tts,
        interrupt_manager=None,
        local_brain=local_brain,
        llm_queue=llm_queue,
        indicator=indicator,
        command_loop=command_loop,
    )

    # ── Perception Engine (Phase 1) ───────────────────────────────
    from core.perception.engine import PerceptionEngine

    perception = PerceptionEngine(bus)

    async def _perception_partial_guarded(text: str = "", **kw) -> None:
        # Drop self-echo BEFORE perception decides to predict an interrupt.
        # This keeps the perception engine's session statistics honest
        # (echo "interrupts" should never count as user frustration) and
        # prevents the predicted-interrupt path from firing on our own
        # voice in the first place.
        partial = (text or "").strip()
        if partial:
            try:
                check_echo = getattr(tts, "is_echo", None)
                if callable(check_echo) and check_echo(partial):
                    return
            except Exception:
                logger.debug("perception echo guard failed", exc_info=True)
        await perception.on_speech_partial(text=text, **kw)

    bus.on(
        "speech_partial",
        _guard_handler(
            "speech_partial",
            _perception_partial_guarded,
            source="perception.on_speech_partial",
        ),
    )
    bus.on(
        "speech_final",
        _guard_handler(
            "speech_final",
            perception.on_speech_final,
            source="perception.on_speech_final",
        ),
    )

    async def _perception_tts_start(old=None, new=None, **_kw) -> None:
        from core.state_manager import AtomState
        if new is AtomState.SPEAKING:
            await perception.on_tts_speaking()

    async def _perception_tts_end(**_kw) -> None:
        await perception.on_tts_done()

    async def _on_tts_delivery_metrics(
        words_spoken: int = 0, duration_ms: float = 0.0,
        backend: str = "", **_kw,
    ) -> None:
        logger.debug(
            "TTS delivery: %d words, %.0fms, backend=%s",
            words_spoken, duration_ms, backend,
        )
        await perception.on_tts_delivery_metrics(
            words_spoken=words_spoken,
            duration_ms=duration_ms,
        )

    bus.on("state_changed", _guard_handler(
        "state_changed", _perception_tts_start,
        source="perception.tts_start",
    ))
    bus.on("tts_complete", _guard_handler(
        "tts_complete", _perception_tts_end,
        source="perception.tts_end",
    ))
    bus.on("tts_delivery_metrics", _guard_handler(
        "tts_delivery_metrics", _on_tts_delivery_metrics,
        source="perception.delivery_metrics",
    ))

    # ── SpeechController (single merge point for TTS params) ────────
    from core.speech_controller import SpeechController

    speech_ctrl = SpeechController()
    _perception_style_applied = {"locked": False}
    _adaptive_last_apply_t = {"t": 0.0}

    def _apply_merged_style() -> None:
        """Push the composed perception+adaptive params to TTS."""
        if hasattr(tts, "apply_perception_style"):
            merged = speech_ctrl.merged()
            tts.apply_perception_style(
                rate_multiplier=merged["rate_multiplier"],
                pause_multiplier=merged["pause_multiplier"],
            )
            logger.info(
                "SPEECH_PIPELINE perception=(%.2f, %.2f) adaptive=(%.2f, %.2f) -> merged=(%.3f, %.3f)",
                speech_ctrl._perception["rate_multiplier"],
                speech_ctrl._perception["pause_multiplier"],
                speech_ctrl._adaptive["rate_multiplier"],
                speech_ctrl._adaptive["pause_multiplier"],
                merged["rate_multiplier"],
                merged["pause_multiplier"],
            )

    async def _on_perception_result(
        emotion=None, urgency=None, style=None, **_kw,
    ) -> None:
        if _perception_style_applied["locked"]:
            return
        if style is None:
            return

        e_intensity = getattr(emotion, "intensity", 0.0) if emotion else 0.0
        u_score = getattr(urgency, "score", 0.0) if urgency else 0.0
        u_level = getattr(urgency, "level", "low") if urgency else "low"

        if e_intensity < 0.25 and u_score < 0.3:
            _perception_style_applied["locked"] = True
            return

        if u_level == "high":
            speech_ctrl.set_perception(rate_multiplier=1.0, pause_multiplier=1.0)
        else:
            speech_ctrl.set_perception(
                rate_multiplier=style.rate_multiplier,
                pause_multiplier=style.pause_multiplier,
            )
        _apply_merged_style()
        _perception_style_applied["locked"] = True

    async def _unlock_perception_style(**_kw) -> None:
        _perception_style_applied["locked"] = False
        speech_ctrl.reset()

    bus.on("perception_result", _guard_handler(
        "perception_result", _on_perception_result,
        source="speech_ctrl.on_perception_result",
    ))
    bus.on("tts_complete", _guard_handler(
        "tts_complete", _unlock_perception_style,
        source="speech_ctrl.unlock",
    ))

    async def _on_perception_adaptive(
        concise: bool = False, rate_boost: float = 0.0,
        session_stats: dict | None = None, **_kw,
    ) -> None:
        apply_profile = getattr(router, "apply_perception_profile", None)
        if callable(apply_profile):
            apply_profile(concise=concise, rate_boost=rate_boost)
        if session_stats:
            logger.debug(
                "Session stats: int_rate=%.2f frust=%.2f avg_dur=%.0fms",
                session_stats.get("interrupt_rate", 0),
                session_stats.get("frustration_score", 0),
                session_stats.get("avg_duration_ms", 0),
            )

    bus.on("perception_adaptive", _guard_handler(
        "perception_adaptive", _on_perception_adaptive,
        source="speech_ctrl.on_perception_adaptive",
    ))

    # Sprint N7 — mood-driven prosody. mood_changed payload is
    # {"mood": "focused"} from MoodInferenceEngine. We translate to a
    # MoodProsody profile and push it into the SpeechController's third
    # multiplier channel so it composes cleanly with perception/adaptive.
    try:
        from voice.mood_voice_profile import for_mood as _mood_prosody_for
    except Exception:  # pragma: no cover - import shim
        _mood_prosody_for = None  # type: ignore[assignment]

    async def _on_mood_changed(mood: str | None = None, **_kw) -> None:
        if _mood_prosody_for is None:
            return
        if not mood:
            return
        try:
            profile = _mood_prosody_for(str(mood))
            params = profile.to_speech_params()
            speech_ctrl.set_mood(
                rate_multiplier=params["rate_multiplier"],
                pause_multiplier=params["pause_multiplier"],
            )
            _apply_merged_style()
            logger.info(
                "MOOD prosody applied: mood=%s rate=%.2f pause=%.2f preset=%s",
                profile.mood,
                profile.rate_multiplier,
                profile.pause_multiplier,
                profile.voice_preset,
            )
        except Exception:
            logger.debug("mood prosody apply failed", exc_info=True)

    bus.on("mood_changed", _guard_handler(
        "mood_changed", _on_mood_changed,
        source="speech_ctrl.on_mood_changed",
    ))

    # Minimum word count before a perception-predicted interrupt is allowed
    # to cut off TTS. A single mis-heard token like ``Boss`` or ``mad`` is
    # almost always our own voice spilling into the mic; requiring at least
    # two real words prevents the SPEAKING -> own-voice -> LISTENING flap
    # loop that was making conversations feel broken.
    _PERCEPTION_INTERRUPT_MIN_WORDS = 2

    async def _on_interrupt_predicted(**_kw) -> None:
        # Echo guard: if the predicted-interrupt partial is just our own
        # TTS bleed, drop it. Without this gate the perception engine
        # interrupts ATOM's response as soon as the speakers leak any
        # syllable into the mic.
        partial_text = str(_kw.get("text", "") or "").strip()
        if partial_text:
            try:
                check_echo = getattr(tts, "is_echo", None)
                if callable(check_echo) and check_echo(partial_text):
                    logger.info(
                        "interrupt_predicted dropped as TTS self-echo: '%s'",
                        partial_text[:80],
                    )
                    return
            except Exception:
                logger.debug("interrupt_predicted echo check failed", exc_info=True)

            # Floor: thin partials never cut TTS off via the prediction
            # path. The burst-based path inside VoiceInterruptHandler still
            # fires when the user really is talking over us (3 partials in
            # 500ms), so genuine barge-ins still feel instant.
            try:
                if state.current is AtomState.SPEAKING:
                    word_count = len(partial_text.split())
                    if word_count < _PERCEPTION_INTERRUPT_MIN_WORDS:
                        logger.debug(
                            "interrupt_predicted ignored (only %d word(s) while speaking)",
                            word_count,
                        )
                        return
            except Exception:
                logger.debug("interrupt_predicted state check failed", exc_info=True)

        await voice_interrupt.interrupt_to_listening(
            trigger="interrupt_predicted",
            reason="perception_predicted",
            user_interrupt=True,
        )

    bus.on(
        "interrupt_predicted",
        _guard_handler(
            "interrupt_predicted",
            _on_interrupt_predicted,
            source="perception.interrupt_predicted",
        ),
    )

    # ── Adaptive Intelligence Engine (Phase 2) ────────────────────
    from core.adaptive.engine import AdaptiveEngine

    adaptive = AdaptiveEngine(bus)

    bus.on(
        "perception_result",
        _guard_handler(
            "perception_result",
            adaptive.on_perception,
            source="adaptive.on_perception",
        ),
    )
    bus.on(
        "tts_delivery_metrics",
        _guard_handler(
            "tts_delivery_metrics",
            adaptive.on_tts_delivery_metrics,
            source="adaptive.on_tts_delivery_metrics",
        ),
    )

    async def _on_adaptive_speech_update(
        rate_multiplier: float = 1.0,
        pause_multiplier: float = 1.0,
        **_kw,
    ) -> None:
        now = time.monotonic()
        if now - _adaptive_last_apply_t["t"] < 2.0:
            return
        _adaptive_last_apply_t["t"] = now
        speech_ctrl.set_adaptive(
            rate_multiplier=rate_multiplier,
            pause_multiplier=pause_multiplier,
        )
        _apply_merged_style()

    bus.on("adaptive_speech_update", _guard_handler(
        "adaptive_speech_update", _on_adaptive_speech_update,
        source="speech_ctrl.on_adaptive_speech_update",
    ))

    attach_adaptive = getattr(router, "attach_adaptive_engine", None)
    if callable(attach_adaptive):
        attach_adaptive(adaptive)
    else:
        logger.debug(
            "Router has no attach_adaptive_engine() — skipping adaptive bridge",
        )

    bus.on(
        "state_changed",
        _guard_handler(
            "state_changed",
            indicator.on_state_changed,
            source="indicator.on_state_changed",
        ),
    )
    # Defensive: STT engines without ``on_state_changed`` (older
    # backends or future drop-ins) must not break boot. Before this
    # guard, an AttributeError here propagated out of wire_events,
    # main() exited mid-init, and asyncio.run() deadlocked during
    # teardown waiting for the iPhone-bridge BG task to cancel —
    # the silent post-AdaptiveEngine stall.
    stt_on_state_changed = getattr(stt, "on_state_changed", None)
    if callable(stt_on_state_changed):
        bus.on(
            "state_changed",
            _guard_handler(
                "state_changed",
                stt_on_state_changed,
                source="stt.on_state_changed",
            ),
        )
    else:
        logger.debug(
            "STT backend %s has no on_state_changed; skipping state-bridge wire",
            type(stt).__name__,
        )
    stt_on_tts_complete = getattr(stt, "on_tts_complete", None)
    if callable(stt_on_tts_complete):
        bus.on(
            "tts_complete",
            _guard_handler(
                "tts_complete",
                stt_on_tts_complete,
                source="stt.on_tts_complete",
            ),
        )
    _speech_target = command_loop.submit if command_loop is not None else router.on_speech

    def _is_self_echo_final(text: str) -> bool:
        """True when this speech_final is the mic catching ATOM's own voice.

        ``tts.is_echo`` is the same fuzzy bag-of-words check used to suppress
        partials, but here we also require that ATOM is currently SPEAKING /
        THINKING; otherwise a legitimate user follow-up that happens to share
        words with the previous reply could be wrongly dropped.
        """
        try:
            from core.state_manager import AtomState
        except Exception:
            return False
        try:
            cur = state.current
        except Exception:
            return False
        if cur not in (AtomState.SPEAKING, AtomState.THINKING):
            return False
        try:
            check = getattr(tts, "is_echo", None)
            if not callable(check):
                return False
            return bool(check(text))
        except Exception:
            return False

    if priority_sched is not None:
        from core.priority_scheduler import PRIORITY_VOICE

        async def _speech_via_priority(text: str, **kw) -> None:
            if shutdown_event.is_set():
                return
            if _is_self_echo_final(text):
                logger.info(
                    "speech_final dropped as TTS self-echo: '%s'",
                    str(text or "")[:80],
                )
                return
            await voice_interrupt.prepare_for_new_speech(text, **kw)
            if local_brain is not None:
                local_brain.request_preempt()

            def _factory():
                async def _job() -> None:
                    if shutdown_event.is_set():
                        return
                    await _speech_target(text, **kw)

                return _job()

            priority_sched.submit(PRIORITY_VOICE, "speech_final", _factory)

        if not v4:
            bus.on(
                "speech_final",
                _guard_handler(
                    "speech_final",
                    _speech_via_priority,
                    source="command_loop.submit.priority",
                ),
            )
    else:
        async def _speech_final_direct(text: str, **kw) -> None:
            if shutdown_event.is_set():
                return
            if _is_self_echo_final(text):
                logger.info(
                    "speech_final dropped as TTS self-echo: '%s'",
                    str(text or "")[:80],
                )
                return
            await voice_interrupt.prepare_for_new_speech(text, **kw)
            if local_brain is not None:
                local_brain.request_preempt()
            await _speech_target(text, **kw)

        if not v4:
            bus.on(
                "speech_final",
                _guard_handler(
                    "speech_final",
                    _speech_final_direct,
                    source="command_loop.submit.direct",
                ),
            )

    # ── Local LLM only (offline) — serial queue + fast bus handler ─
    if local_brain is not None:
        # Attach Router's anti-hallucination vetter so action-promise
        # fabrications or low-confidence LLM output get rewritten to a
        # short clarifying question before they reach TTS.
        try:
            attach_vetter = getattr(local_brain, "attach_response_vetter", None)
            vetter = getattr(router, "vet_llm_response", None)
            if callable(attach_vetter) and callable(vetter):
                attach_vetter(vetter)
        except Exception:
            logger.debug("response vetter wiring failed", exc_info=True)

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
                            "Local brain is not ready, Boss. Check the MLX model "
                            "directories in settings and that mlx/mlx_lm are installed."
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
                            query_plan=kw.get("query_plan"),
                            repeat_hint=bool(kw.get("repeat_hint", False)),
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

        bus.on(
            "cursor_query",
            _guard_handler(
                "cursor_query",
                _local_brain_query,
                source="local_brain.on_query",
            ),
        )

        async def _on_pending_tool_confirmation(tool_call=None, result=None, **_kw) -> None:
            """Store pending tool confirmation from agentic LLM for the Router."""
            if tool_call is not None:
                router._pending_tool_confirmation = {
                    "tool_call": tool_call,
                    "result": result,
                    "created_at": time.monotonic(),
                }

        bus.on(
            "pending_tool_confirmation",
            _guard_handler(
                "pending_tool_confirmation",
                _on_pending_tool_confirmation,
                source="router.pending_tool_confirmation",
            ),
        )

    bus.on(
        "response_ready",
        _guard_handler(
            "response_ready",
            tts.on_response,
            source="tts.on_response",
        ),
    )
    bus.on(
        "partial_response",
        _guard_handler(
            "partial_response",
            tts.on_partial_response,
            source="tts.on_partial_response",
        ),
    )

    async def _on_jarvis_insight(message: str = "", **kw) -> None:
        """Route proactive insights from JarvisCore to TTS."""
        if message:
            bus.emit_long("response_ready", text=message)

    bus.on("jarvis_insight", _on_jarvis_insight)

    async def _on_suggestion_ready(suggestions: list = None, **kw) -> None:
        """Forward suggestions to dashboard only — never collide with active TTS."""
        if suggestions and len(suggestions) > 0:
            bus.emit_fast("dashboard_suggestion", suggestions=suggestions)

    bus.on("suggestion_ready", _on_suggestion_ready)

    async def _on_emotion_for_tts(emotion: str = "", **kw) -> None:
        """Forward detected emotion to TTS for dynamic prosody."""
        if emotion and hasattr(tts, "set_emotion"):
            tts.set_emotion(emotion)

    bus.on("user_emotion_detected", _on_emotion_for_tts)
    bus.on(
        "tts_complete",
        _guard_handler(
            "tts_complete",
            state.on_tts_complete,
            source="state.on_tts_complete",
        ),
    )
    bus.on(
        "silence_timeout",
        _guard_handler(
            "silence_timeout",
            state.on_silence_timeout,
            source="state.on_silence_timeout",
        ),
    )

    # ── Media / error recovery ────────────────────────────────────
    async def _on_media_started(**_kw) -> None:
        fn = getattr(stt, "on_media_started", None)
        if callable(fn):
            fn()

    async def _on_media_stopped(**_kw) -> None:
        fn = getattr(stt, "on_media_stopped", None)
        if callable(fn):
            fn()

    bus.on(
        "media_started",
        _guard_handler(
            "media_started",
            _on_media_started,
            source="stt.on_media_started",
        ),
    )
    bus.on(
        "media_stopped",
        _guard_handler(
            "media_stopped",
            _on_media_stopped,
            source="stt.on_media_stopped",
        ),
    )

    async def on_llm_error(source: str = "local", **_kw) -> None:
        logger.error("LLM error from %s -- triggering recovery", source)
        await state.on_error(source=source)
    bus.on(
        "llm_error",
        _guard_handler(
            "llm_error",
            on_llm_error,
            source="state.on_llm_error",
        ),
    )

    # ── Sleep / barge-in / resume (hotkey + dashboard UNSTICK) ─
    async def on_resume_listening(
        source: str = "",
        reason: str = "",
        partial_text: str = "",
        user_interrupt: bool = False,
        **_kw,
    ) -> None:
        await voice_interrupt.interrupt_to_listening(
            trigger=source or "resume_listening",
            reason=reason,
            partial_text=partial_text,
            user_interrupt=(bool(user_interrupt) or source == "voice_interrupt"),
        )
    bus.on(
        "resume_listening",
        _guard_handler(
            "resume_listening",
            on_resume_listening,
            source="resume_listening",
        ),
    )

    async def _on_enter_sleep(**_kw) -> None:
        logger.info("Entering SLEEP mode -- Ctrl+Alt+A to resume listening")
        stt.stop()
        await state.transition(AtomState.SLEEP)
        indicator.add_log("action", "Silent mode. Press Ctrl+Alt+A to resume listening.")
    bus.on(
        "enter_sleep_mode",
        _guard_handler(
            "enter_sleep_mode",
            _on_enter_sleep,
            source="enter_sleep_mode",
        ),
    )

    # ── STT recovery ─────────────────────────────────────────────
    async def on_restart_listening(**_kw) -> None:
        if state.current is AtomState.LISTENING:
            await asyncio.sleep(0.1)
            if state.current is AtomState.LISTENING:
                if not (v3 or v4):
                    start_listener = getattr(stt, "async_start_listening", None)
                    if not callable(start_listener):
                        start_listener = getattr(stt, "start_listening", None)
                    if callable(start_listener):
                        # Stop any in-flight listen loop so we do not stack two mic pipelines.
                        stop_fn = getattr(stt, "stop", None)
                        if callable(stop_fn):
                            try:
                                stop_fn()
                            except Exception:
                                logger.debug("restart_listening: stt.stop failed", exc_info=True)
                        await asyncio.sleep(0.08)
                        asyncio.create_task(start_listener())
                    else:
                        logger.info(
                            "Skipping restart_listening because %s has no start hook",
                            type(stt).__name__,
                        )
    bus.on(
        "restart_listening",
        _guard_handler(
            "restart_listening",
            on_restart_listening,
            source="restart_listening",
        ),
    )

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
    bus.on(
        "stt_did_not_catch",
        _guard_handler(
            "stt_did_not_catch",
            on_stt_did_not_catch,
            source="stt.did_not_catch",
        ),
    )
    bus.on(
        "stt_too_noisy",
        _guard_handler(
            "stt_too_noisy",
            on_stt_too_noisy,
            source="stt.too_noisy",
        ),
    )

    # ── UI logging ───────────────────────────────────────────────
    async def log_response(text: str, **_kw) -> None:
        _stop_thinking_progress()
        indicator.add_log("action", text)
        if state_bridge is not None:
            state_bridge.patch_section(
                "reasoning",
                {"last_report": str(text or ""), "severity": "info"},
                source="wiring.log_response",
            )

    async def log_thinking_ack(text: str, **_kw) -> None:
        if text and _perceived["t_speech_final"] > 0 and not _ttfa_gate["sent"]:
            ttfa_ms = (time.perf_counter() - _perceived["t_speech_final"]) * 1000
            metrics.record_latency("ttfa", ttfa_ms)
            _ttfa_gate["sent"] = True
        indicator.add_log("info", text)
        if state_bridge is not None and text:
            state_bridge.patch_section(
                "reasoning",
                {"last_decision": str(text), "severity": "info"},
                source="wiring.log_thinking_ack",
            )
        if text:
            asyncio.create_task(tts.speak_ack(text))

    async def log_cursor_query(text: str, **_kw) -> None:
        indicator.add_log("action", "Thinking with local brain...")
        if state_bridge is not None:
            payload = {"status": "running", "label": "Thinking with local brain..."}
            state_bridge.patch_section(
                "execution",
                payload,
                source="wiring.log_cursor_query",
            )
            state_bridge.events.emit_execution_update(
                source="wiring.log_cursor_query",
                **payload,
            )
        _start_thinking_progress()

    async def log_partial(
        text: str,
        is_first: bool = False,
        is_last: bool = False,
        stream_id: str = "",
        **_kw,
    ) -> None:
        if is_first:
            _stream_buffer["text"] = ""
            _stream_buffer["stream_id"] = stream_id or ""
        elif stream_id and _stream_buffer["stream_id"] and stream_id != _stream_buffer["stream_id"]:
            return
        if text.strip():
            _stream_buffer["text"] += (" " if _stream_buffer["text"] else "") + text.strip()
            indicator.add_log("speaking", _stream_buffer["text"])
            if state_bridge is not None:
                state_bridge.patch_section(
                    "voice",
                    {"status": "speaking"},
                    source="wiring.log_partial",
                )
        if is_last:
            if _stream_buffer["text"]:
                indicator.add_log("action", _stream_buffer["text"])
                if state_bridge is not None:
                    state_bridge.patch_section(
                        "voice",
                        {"last_final": _stream_buffer["text"], "speaking": False},
                        source="wiring.log_partial",
                    )
            _stream_buffer["text"] = ""
            _stream_buffer["stream_id"] = ""

    async def show_hearing(text: str, **_kw) -> None:
        indicator.show_hearing(text)
        if state_bridge is not None:
            state_bridge.patch_section(
                "voice",
                {"status": "listening", "last_partial": str(text or "")[:240]},
                source="wiring.show_hearing",
            )

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
            if state_bridge is not None:
                payload = {"latency_ms": float(latency_ms), "status": "idle"}
                state_bridge.patch_section(
                    "execution",
                    payload,
                    source="wiring.measure_perceived",
                )
                state_bridge.events.emit_execution_update(
                    source="wiring.measure_perceived",
                    **payload,
                )

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
        if state_bridge is not None:
            state_bridge.patch_section(
                "voice",
                {
                    "last_final": str(text or "")[:240],
                    "language": str(language or "en"),
                    "status": "processing",
                },
                source="wiring.speech_final",
            )
        metrics.inc("queries_total")
        if hasattr(indicator, "set_last_query"):
            indicator.set_last_query(text)
        if hasattr(indicator, "set_language"):
            indicator.set_language(language)

    async def _on_intent_classified(intent: str = "", **_kw) -> None:
        if hasattr(indicator, "set_last_intent"):
            indicator.set_last_intent(intent)
        if state_bridge is not None:
            payload = {"last_intent": str(intent or "")}
            state_bridge.patch_section(
                "execution",
                payload,
                source="wiring.intent_classified",
            )
            state_bridge.events.emit_execution_update(
                source="wiring.intent_classified",
                **payload,
            )

    async def _on_voice_partial_event(
        text: str = "",
        confidence: float = 0.0,
        engine: str = "",
        mic: str = "",
        **_kw,
    ) -> None:
        if state_bridge is None:
            return
        state_bridge.patch_section(
            "voice",
            {
                "last_partial": str(text or "")[:240],
                "confidence": float(confidence or 0.0),
                "mic": str(mic or ""),
                "status": "listening",
                "listening": True,
                "error": None,
            },
            source="wiring.voice_partial",
        )

    async def _on_voice_final_event(
        text: str = "",
        language: str = "en",
        confidence: float = 0.0,
        engine: str = "",
        mic: str = "",
        **_kw,
    ) -> None:
        if state_bridge is None:
            return
        patch = {
            "last_final": str(text or "")[:240],
            "language": str(language or "en"),
            "confidence": float(confidence or 0.0),
            "mic": str(mic or ""),
            "status": "processing",
            "listening": False,
            "error": None,
        }
        if engine:
            patch["stt_engine"] = str(engine)
        state_bridge.patch_section(
            "voice",
            patch,
            source="wiring.voice_final",
        )

    bus.on("speech_final", _on_speech_final_consolidated)
    bus.on("intent_classified", _on_intent_classified)
    bus.on("partial_response", _measure_perceived)
    bus.on("voice.partial", _on_voice_partial_event)
    bus.on("voice.final", _on_voice_final_event)
    bus.on(
        "speech_partial",
        _guard_handler(
            "speech_partial",
            show_hearing,
            source="indicator.show_hearing",
        ),
    )
    bus.on(
        "speech_partial",
        _guard_handler(
            "speech_partial",
            voice_interrupt.on_speech_partial,
            source="voice_interrupt.on_speech_partial",
        ),
    )
    async def on_text_display(text: str, **_kw) -> None:
        """Screen-only overflow text (not spoken, shown on dashboard)."""
        if text.strip():
            indicator.add_log("info", f"[screen] {text.strip()}")

    async def on_voice_ack(
        text: str = "",
        spoken_inline: bool = False,
        **_kw,
    ) -> None:
        # Sprint C3: when ``CommandLoop._tts`` is wired the loop spawns
        # ``speak_ack`` directly so the LLM call kicks off on the next
        # event-loop tick (no bus dispatch latency). The loop sets
        # ``spoken_inline=True`` in that case so we do NOT double-speak
        # the same ack from this subscriber.
        if spoken_inline:
            return
        if text and tts is not None:
            asyncio.create_task(tts.speak_ack(text))

    bus.on("response_ready", log_response)
    bus.on("partial_response", log_partial)
    bus.on("text_display", on_text_display)
    bus.on("thinking_ack", log_thinking_ack)
    bus.on("voice_ack", on_voice_ack)
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
            # Always update the dashboard log + state bridge so the
            # warning is visible even when we suppress the TTS.
            indicator.add_log("warning", message)
            if state_bridge is not None:
                warnings = list(state_bridge.store.get_section("health").get("warnings", []))
                if message not in warnings:
                    warnings.append(message)
                state_bridge.patch_section(
                    "health",
                    {"warnings": warnings[-10:], "status": "degraded"},
                    source="wiring.system_event",
                )
                state_bridge.events.emit_system_warning(
                    source="wiring.system_event",
                    message=message,
                    kind=kind,
                )
            # Sprint Ω.8 (Apr 26 2026) R8: do NOT speak resource alerts
            # over a user turn. atomCurrentLogs.txt L385-L390 shows the
            # exact bug — Boss asked "where is my music atom?", ATOM
            # said "Go ahead." (ack), and *while still THINKING* the
            # RAM-watcher fired "Boss, RAM is at 85%..." which got
            # spliced into the same TTS run. Result: Boss's actual
            # question never gets answered. Resource alerts fire on a
            # 5-min cooldown, so dropping the spoken alert during a
            # user turn just means it'll either re-fire next minute
            # (still 85%) or self-clear (memory came back). Either way
            # is fine; talking over Boss is not.
            if state.current in (AtomState.IDLE, AtomState.LISTENING):
                bus.emit_long("response_ready", text=message)
            else:
                logger.info(
                    "wiring.system_event: suppressed resource_alert TTS "
                    "during state=%s; warning kept in dashboard only",
                    getattr(state.current, "value", state.current),
                )
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
        record_turn = getattr(router, "record_turn", None)
        if callable(record_turn):
            record_turn(query, response)
        suggest_follow_up = getattr(router, "_suggest_follow_up", None)
        follow_up = suggest_follow_up(query, response) if callable(suggest_follow_up) else None
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
                    logger.debug('Child process terminate failed', exc_info=True)
            _, alive = psutil.wait_procs(current.children(), timeout=2)
            for p in alive:
                try:
                    p.kill()
                except Exception:
                    logger.debug('Child process terminate failed', exc_info=True)
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
            if state_bridge is not None:
                state_bridge.patch_section(
                    "voice",
                    {"mic": getattr(stt, "mic_name", ""), "status": "listening"},
                    source="wiring.update_mic_on_listen",
                )

    async def auto_recover_to_listening(old, new, **_kw) -> None:
        if new is AtomState.IDLE and state.always_listen:
            logger.info("Always-listen recovery: IDLE -> LISTENING")
            await asyncio.sleep(1)
            if state.current is AtomState.IDLE and not shutdown_event.is_set():
                await state.transition(AtomState.LISTENING)

    async def on_mic_changed(name: str = "", **_kw) -> None:
        indicator.set_mic_name(name or stt.mic_name)
        if state_bridge is not None:
            state_bridge.patch_section(
                "voice",
                {"mic": str(name or getattr(stt, "mic_name", ""))},
                source="wiring.on_mic_changed",
            )
    bus.on("state_changed", update_mic_on_listen)
    bus.on("state_changed", auto_recover_to_listening)
    bus.on("mic_changed", on_mic_changed)

    # ── FSEvents → optional proactive hints (Downloads + notable extensions) ─
    async def on_fs_event(path: str = "", change: str = "", is_dir: bool = False, **_kw) -> None:
        from core.macos.fs_watcher_config import fs_watcher_settings, notable_file_hint

        s = fs_watcher_settings(config)
        if not s["hints_enabled"]:
            return
        ev = change or str(_kw.get("event") or "")
        hint = notable_file_hint(
            path=path, event=ev, is_dir=is_dir, config=config,
        )
        if not hint:
            return
        now = time.monotonic()
        cooldown = float(s["hint_cooldown_s"])
        if now - float(_proactive_state.get("last_fs_hint", 0.0)) < cooldown:
            return
        _proactive_state["last_fs_hint"] = now
        try:
            indicator.add_log("info", hint)
        except Exception:
            logger.debug("FS hint indicator log failed", exc_info=True)
        if s["emit_voice"]:
            try:
                bus.emit_long("response_ready", text=hint)
            except Exception:
                logger.debug("FS hint voice emit failed", exc_info=True)

    bus.on(
        "fs_event",
        _guard_handler("fs_event", on_fs_event, source="wiring.on_fs_event"),
    )

    async def _on_system_state_for_personality(
        snapshot: dict | None = None, **_kw,
    ) -> None:
        if snapshot:
            from core.adaptive_personality import update_system_context
            update_system_context(snapshot)

    bus.on("system_state_update", _on_system_state_for_personality)

    return {
        "perceived": _perceived,
        "proactive_state": _proactive_state,
        "didnt_catch_count": _didnt_catch_count,
        "last_perceived_ms": _last_perceived_ms,
        "adaptive": adaptive,
    }


