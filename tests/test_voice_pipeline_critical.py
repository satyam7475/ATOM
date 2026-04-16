"""
Critical voice pipeline tests -- the most important path in ATOM.

These tests verify the core voice pipeline doesn't regress:
  1. Event bus handles both sync and async handlers
  2. STT watchdog resets timers on listening transition
  3. STT watchdog preserves partial text before restart
  4. Wake word filter detects "atom" in text
  5. Voice pipeline builds without crashing
"""

from __future__ import annotations

import asyncio
import time
import pytest


class FakeBus:
    """Minimal event bus for testing."""

    def __init__(self):
        self._handlers: dict[str, list] = {}
        self._emitted: list[tuple[str, dict]] = []

    def on(self, event: str, handler):
        self._handlers.setdefault(event, []).append(handler)

    def emit(self, event: str, **data):
        self._emitted.append((event, data))

    def emit_fast(self, event: str, **data):
        self._emitted.append((event, data))

    @property
    def emitted_events(self):
        return [(e, d) for e, d in self._emitted]


# ── Test 1: AsyncEventBus handles sync handlers ──

@pytest.mark.asyncio
async def test_event_bus_handles_sync_handler():
    """Sync handlers should not crash the event bus."""
    from core.async_event_bus import AsyncEventBus

    bus = AsyncEventBus()
    results = []

    def sync_handler(**kw):
        results.append(kw.get("text", ""))

    bus.on("test_event", sync_handler)
    bus.start()
    bus.emit("test_event", text="hello")
    await asyncio.sleep(0.3)
    await bus.stop()
    assert "hello" in results, f"Sync handler not called, results={results}"


@pytest.mark.asyncio
async def test_event_bus_handles_async_handler():
    """Async handlers should work as expected."""
    from core.async_event_bus import AsyncEventBus

    bus = AsyncEventBus()
    results = []

    async def async_handler(**kw):
        results.append(kw.get("text", ""))

    bus.on("test_event", async_handler)
    bus.start()
    bus.emit("test_event", text="world")
    await asyncio.sleep(0.3)
    await bus.stop()
    assert "world" in results, f"Async handler not called, results={results}"


# ── Test 2: STT Watchdog timer reset ──

@pytest.mark.asyncio
async def test_watchdog_resets_on_listening_transition():
    """Watchdog timers should reset when STT transitions to listening."""
    from voice.stt_watchdog import STTWatchdog

    bus = FakeBus()
    wd = STTWatchdog(bus)

    old_time = wd._last_partial_time
    wd._last_partial_time = old_time - 30.0

    class FakeSTT:
        _listening = False
        _running_async = True
        _tap_buffer_count = 0
        _last_audio_rms_db = -96.0
        _last_speech_candidate_time = 0.0
        _last_partial = ""

    stt = FakeSTT()
    wd.attach_stt(stt)

    stt._listening = True
    await wd._check_health()

    assert time.monotonic() - wd._last_partial_time < 2.0, \
        "Watchdog did not reset timers on listening transition"


# ── Test 3: Watchdog preserves partial text ──

@pytest.mark.asyncio
async def test_watchdog_salvages_partial_before_restart():
    """When restarting, watchdog should emit last partial text."""
    bus = FakeBus()

    from voice.stt_watchdog import STTWatchdog
    wd = STTWatchdog(bus)

    class FakeSTT:
        _listening = True
        _running_async = True
        _tap_buffer_count = 100
        _last_audio_rms_db = -30.0
        _last_speech_candidate_time = time.monotonic()
        _last_partial = "hey atom what time is it"
        _callback_starvation_count = 0
        _native_requires_on_device = True

        def _on_recognition_starvation(self):
            self._callback_starvation_count += 1

        def _restart_recognition_chain(self):
            pass

    stt = FakeSTT()
    wd.attach_stt(stt)
    wd._was_listening = True
    wd._last_partial_time = time.monotonic() - 20.0
    wd._last_tap_count = 50

    await wd._check_health()

    salvaged = [e for e, d in bus._emitted if e == "speech_partial"]
    assert len(salvaged) > 0, "Watchdog did not salvage partial text"
    assert bus._emitted[0][1]["text"] == "hey atom what time is it"


# ── Test 4: Wake word filter ──

def test_wake_word_filter_detects_atom():
    """WakeWordFilter should detect 'atom' in text."""
    from voice.listening_modes import WakeWordFilter

    wf = WakeWordFilter(cooldown_s=0.0)
    result = wf.check("hey atom")
    assert result is not None, "WakeWordFilter did not detect 'hey atom'"

    wf2 = WakeWordFilter(cooldown_s=0.0)
    result2 = wf2.check("tell me atom")
    assert result2 is not None, "WakeWordFilter did not detect trailing 'atom'"


def test_wake_word_filter_ignores_non_wake():
    """WakeWordFilter should not trigger on random text."""
    from voice.listening_modes import WakeWordFilter

    wf = WakeWordFilter(cooldown_s=0.0)
    result = wf.check("what time is it")
    assert result is None, "WakeWordFilter falsely triggered"


# ── Test 5: Voice pipeline builds ──

def test_voice_pipeline_imports():
    """Voice pipeline module should import without errors."""
    from voice.voice_pipeline import VoicePipeline
    assert VoicePipeline is not None


# ── Test 6: Watchdog full-restart uses atomic gate ──

@pytest.mark.asyncio
async def test_watchdog_full_restart_gates_auto_start_loop():
    """Watchdog must raise begin_full_restart gate BEFORE stop, hold it across
    the async yield, and lower it AFTER start_listening. An ``_run_async``-like
    observer that reads ``_suppress_auto_start`` during the yield window must
    see the gate up — otherwise it races the watchdog and binds a new
    recognition task to the stale recognizer.
    """
    from voice.stt_watchdog import STTWatchdog

    call_log: list[str] = []

    class FakeSTT:
        _listening = True
        _running_async = True
        _tap_buffer_count = 100
        _last_audio_rms_db = -30.0
        _last_speech_candidate_time = time.monotonic()
        _last_partial = "hello atom"
        _callback_starvation_count = 0
        _native_requires_on_device = True
        _loop = None
        _on_final = None
        _on_partial = None
        _suppress_auto_start = False

        def begin_full_restart(self):
            call_log.append("begin_full_restart")
            self._suppress_auto_start = True
            self._listening = False

        def end_full_restart(self):
            call_log.append("end_full_restart")
            self._suppress_auto_start = False

        def _recreate_recognizer(self):
            call_log.append("recreate")
            return True

        def start_listening(self, loop=None, on_final=None, on_partial=None):
            call_log.append("start_listening")
            self._listening = True
            return True

        def _on_recognition_starvation(self):
            self._callback_starvation_count += 1

        def _restart_recognition_chain(self):
            pass

    bus = FakeBus()
    wd = STTWatchdog(bus)
    stt = FakeSTT()
    wd.attach_stt(stt)

    observer_saw_gate_up: list[bool] = []

    async def observer():
        # Poll a few times during the watchdog's restart window; this
        # simulates _run_async checking the gate during asyncio.sleep(0.8).
        for _ in range(10):
            await asyncio.sleep(0.05)
            if not stt._listening:
                observer_saw_gate_up.append(bool(stt._suppress_auto_start))

    # Prime watchdog to trigger the full-restart branch on the next call.
    wd._consecutive_chain_restarts = 1  # next += 1 crosses threshold of 2
    obs_task = asyncio.create_task(observer())
    await wd._restart_stt(stt, reason="test_race")
    await obs_task

    assert "begin_full_restart" in call_log, f"begin_full_restart missing: {call_log}"
    assert "recreate" in call_log, f"recreate missing: {call_log}"
    assert "start_listening" in call_log, f"start_listening missing: {call_log}"
    assert "end_full_restart" in call_log, f"end_full_restart missing: {call_log}"
    # Ordering: gate MUST be raised before recreate/start and released after.
    assert call_log.index("begin_full_restart") < call_log.index("recreate")
    assert call_log.index("recreate") < call_log.index("start_listening")
    assert call_log.index("start_listening") < call_log.index("end_full_restart")
    # While _listening was False, the gate must have been observed as up —
    # otherwise _run_async would preempt and bind a task to the old recognizer.
    assert observer_saw_gate_up, "observer did not run during the restart window"
    assert all(observer_saw_gate_up), (
        "suppress_auto_start was False during the restart window — race re-opened"
    )


# ── Test 7: Router action-promise guardrail (Tier 2c) ──

def _make_minimal_router():
    """Construct a Router instance with only the fields vet_llm_response touches.

    vet_llm_response is a pure text transform; it only reads class-level
    constants and the optional self._conv_mgr. Bypassing __init__ avoids
    the full dependency graph (cache, memory, scheduler, ...).
    """
    from core.router.router import Router

    r = Router.__new__(Router)
    r._conv_mgr = None
    return r


def test_router_guardrail_rejects_fabricated_play_action():
    """LLM says 'Playing ...' but user never asked to play anything → clarifier."""
    r = _make_minimal_router()
    reply = "Playing your favorite song now, Boss."
    query = "What's the temperature outside?"
    vetted = r.vet_llm_response(query, reply, confidence=0.8)
    assert vetted != reply, "Fabricated action must be rewritten to a clarifier"
    assert any(marker in vetted.lower() for marker in ("catch", "rephrase", "repeat"))


def test_router_guardrail_accepts_matched_play_action():
    """Query mentions 'play' → action promise is consistent → keep as-is."""
    r = _make_minimal_router()
    reply = "Playing Lo-fi beats now, Boss."
    query = "Play some lo-fi for me"
    vetted = r.vet_llm_response(query, reply, confidence=0.8)
    assert vetted == reply, \
        f"Legit action-promise must pass through verbatim, got: {vetted!r}"


def test_router_guardrail_accepts_matched_open_action():
    r = _make_minimal_router()
    reply = "Opening Safari for you, Boss."
    query = "open safari please"
    assert r.vet_llm_response(query, reply, confidence=0.8) == reply


def test_router_guardrail_empty_inputs_are_safe():
    r = _make_minimal_router()
    assert r.vet_llm_response("", "some reply", 0.8) == "some reply"
    assert r.vet_llm_response("query", "", 0.8) == ""


def test_router_guardrail_low_confidence_rewrites_long_claim():
    """Confidence below 0.5 with a long confident-sounding reply → clarifier."""
    r = _make_minimal_router()
    reply = "The GDP of Sweden was $585 billion last year."
    query = "something vague"
    vetted = r.vet_llm_response(query, reply, confidence=0.3)
    assert vetted != reply, "Low-confidence long reply should be rewritten"


def test_router_guardrail_low_confidence_keeps_short_reply():
    """Short replies (len <= 15) are left alone even when confidence is low —
    they're usually fillers like 'On it' where a clarifier would feel worse."""
    r = _make_minimal_router()
    short = "On it, Boss."
    assert r.vet_llm_response("something", short, confidence=0.2) == short


# ── Test 8: ListeningModeController wake-flip gating (Tier 2d) ──

def test_listening_mode_default_passive_until_wake():
    from voice.listening_modes import ListeningModeController

    lm = ListeningModeController(always_active=False)
    assert lm.is_passive, "Default mode with no always-active must be PASSIVE"
    assert not lm.is_active


def test_listening_mode_activate_on_wake():
    from voice.listening_modes import ListeningModeController

    lm = ListeningModeController(always_active=False)
    changed = lm.activate(reason="wake_word_detected")
    assert changed is True
    assert lm.is_active
    # Second activate is idempotent
    assert lm.activate(reason="redundant") is False


def test_listening_mode_deactivate_back_to_passive():
    from voice.listening_modes import ListeningModeController

    lm = ListeningModeController(always_active=False)
    lm.activate(reason="wake")
    changed = lm.deactivate(reason="post_tts_idle")
    assert changed is True
    assert lm.is_passive


def test_listening_mode_always_active_ignores_deactivate():
    from voice.listening_modes import ListeningModeController

    lm = ListeningModeController(always_active=True)
    assert lm.is_active
    assert lm.deactivate(reason="post_tts_idle") is False
    assert lm.is_active, "always_active mode must stay ACTIVE"


def test_stt_attach_listening_mode_sets_ref():
    """NativeSTT.attach_listening_mode stores the controller so _on_final
    can consult it during PASSIVE-mode gating."""
    from types import SimpleNamespace

    from voice.stt_macos import NativeSTT
    from voice.listening_modes import ListeningModeController

    class _Bus:
        def on(self, *a, **kw): pass
        def emit(self, *a, **kw): pass
        def emit_fast(self, *a, **kw): pass

    state = SimpleNamespace(current=None)
    stt = NativeSTT(bus=_Bus(), state=state, config={})
    ctl = ListeningModeController(always_active=False)
    stt.attach_listening_mode(ctl)
    assert getattr(stt, "_listening_mode_ref", None) is ctl


# ── Test 9: Watchdog full-restart productivity window (Tier 1c) ──

@pytest.mark.asyncio
async def test_watchdog_productive_partial_resets_failure_counter():
    """A productive partial within the 10 s post-full-restart window must
    reset ``_full_restart_failures`` so transient hiccups don't accumulate
    into an unwarranted Whisper swap."""
    from voice.stt_watchdog import STTWatchdog

    bus = FakeBus()
    wd = STTWatchdog(bus)
    wd._full_restart_failures = 2
    wd._last_full_restart_time = time.monotonic() - 0.5

    await wd.on_speech_partial(text="hey atom, what is it")
    assert wd._full_restart_failures == 0, \
        "productive partial inside window must zero the failure counter"
    assert wd._last_full_restart_time == 0.0


@pytest.mark.asyncio
async def test_watchdog_empty_partial_does_not_reset():
    """A partial with empty text during the window must NOT count as
    productive (otherwise the cascade we're protecting against resets itself)."""
    from voice.stt_watchdog import STTWatchdog

    bus = FakeBus()
    wd = STTWatchdog(bus)
    wd._full_restart_failures = 2
    wd._last_full_restart_time = time.monotonic() - 0.5

    await wd.on_speech_partial(text="")
    assert wd._full_restart_failures == 2, \
        "empty partial must not flip productivity flag"


# ── Test 10: Earcons are a safe no-op when disabled (Tier 3c) ──

def test_earcons_disabled_noop():
    from voice.earcons import Earcons

    ec = Earcons(enabled=False)
    # Must not raise, must not try to spawn afplay.
    ec.play("wake")
    ec.play("done")
    ec.play("error")
    ec.shutdown()
    assert ec.is_enabled is False


def test_earcons_play_unknown_event_is_safe():
    from voice.earcons import Earcons

    ec = Earcons(enabled=True, volume=0.1)
    # Unknown event id → no-op without raising.
    ec.play("not-a-real-event")
    ec.shutdown()


# ── Test 11: Hardened router guardrail (P0a) ───────────────────────

def test_router_guardrail_catches_quoted_wrapper_action():
    """`The answer is "Okay, I'll play the song for you."` → clarifier."""
    r = _make_minimal_router()
    reply = 'The answer is "Okay, I\'ll play the song for you.".'
    query = "What is Newton's law?"
    vetted = r.vet_llm_response(query, reply, confidence=0.9)
    assert vetted != reply, (
        f"wrapper-style action promise should be rewritten; got {vetted!r}"
    )
    assert any(
        marker in vetted.lower() for marker in ("catch", "rephrase", "repeat")
    )


def test_router_guardrail_catches_padded_preface_action():
    """Padded preface like `Sure, I'll ...` → caught by substring scan."""
    r = _make_minimal_router()
    reply = "Sure thing, I'll play some music for you now."
    query = "What is the meaning of life?"
    vetted = r.vet_llm_response(query, reply, confidence=0.8)
    assert vetted != reply


def test_router_guardrail_wh_query_blocks_action_even_when_verb_matches():
    """On a WH query we refuse action-promise replies even if a matching
    verb sneaks into the query ('why don't you play that song?')."""
    r = _make_minimal_router()
    reply = "I'll play it now, Boss."
    query = "what is the play clock in NFL?"
    vetted = r.vet_llm_response(query, reply, confidence=0.9)
    assert vetted != reply, (
        "WH-style queries must never receive unprompted action promises"
    )


def test_router_guardrail_non_wh_with_matching_verb_passes():
    r = _make_minimal_router()
    reply = "Opening Chrome now, Boss."
    query = "open chrome for me"
    assert r.vet_llm_response(query, reply, confidence=0.8) == reply


def test_router_unwrap_reply_strips_straight_quotes_and_preface():
    from core.router.router import Router

    text = 'The answer is "Hello there"'
    assert Router._unwrap_reply(text) == "Hello there"


def test_router_unwrap_reply_leaves_plain_text_unchanged():
    from core.router.router import Router

    text = "Newton's first law states that an object in motion stays in motion."
    assert Router._unwrap_reply(text) == text


# ── Test 12: Warmer status / presence-check replies (P4) ────────────

def test_status_presence_check_drops_llm_percent():
    r = _make_minimal_router()
    r._local_queries = 5
    r._llm_queries = 5
    base = "All systems green, Boss."
    out = r._status_with_usage(base, query="can you hear me?")
    assert out == base, (
        "presence check should not append diagnostic percent"
    )


def test_status_diagnostic_query_keeps_percent():
    r = _make_minimal_router()
    r._local_queries = 6
    r._llm_queries = 4
    base = "All systems green, Boss."
    out = r._status_with_usage(base, query="system status")
    assert "percent" in out.lower()
    assert "40" in out, "should report 40 percent LLM usage for 4/10"


# ── Test 13: MLX wrapper-preface guard (P0c) ───────────────────────

def test_mlx_speaker_label_loop_on_wrapper_returns_empty():
    from brain.mlx_llm import MLXBrain

    text = 'The answer is "Okay, I\'ll play the song for you."\nAssistant: Assistant: '
    visible, reason, should_stop = MLXBrain._guard_visible_text(text, ())
    assert should_stop is True
    assert reason == "speaker_label_loop_wrapper"
    assert visible == "", "wrapper-only preface must be treated as empty"


def test_mlx_speaker_label_loop_on_real_content_keeps_it():
    from brain.mlx_llm import MLXBrain

    good = (
        "Newton's first law says an object at rest stays at rest unless "
        "a force acts on it.\nAssistant: Assistant: "
    )
    visible, reason, should_stop = MLXBrain._guard_visible_text(good, ())
    assert should_stop is True
    assert reason == "speaker_label_loop"
    assert "newton" in visible.lower()


# ── Test 14: Correction-phrase bypass in PASSIVE gate (P1c) ─────────

def test_stt_correction_phrase_matches_common_openers():
    from voice.stt_macos import NativeSTT

    for text in (
        "no, I'm not asking about song",
        "I'm not asking about the song, I am asking about what is Newton's law",
        "wait, that was wrong",
        "actually, tell me about gravity",
        "hold on, i meant calculus",
        "stop, that's not what i said",
    ):
        assert NativeSTT._is_correction_phrase(text.lower()), (
            f"correction opener should match: {text!r}"
        )


def test_stt_correction_phrase_ignores_plain_requests():
    from voice.stt_macos import NativeSTT

    for text in (
        "play some music",
        "open safari",
        "what is the weather",
        "tell me about newton's law",
    ):
        assert not NativeSTT._is_correction_phrase(text.lower()), (
            f"non-correction phrase should NOT match: {text!r}"
        )


# ── Test 15: Reply-aware confidence penalty (P0d) ──────────────────

def test_confidence_action_promise_penalty_on_wh_query():
    from core.confidence_engine import ConfidenceEngine

    eng = ConfidenceEngine()
    score_good = eng.score(
        "What is Newton's law?",
        "Newton's first law describes inertia: an object at rest stays at rest.",
    )
    score_bad = eng.score(
        "What is Newton's law?",
        "Okay, I'll play the song for you.",
    )
    assert score_bad < score_good * 0.6, (
        f"WH query with action-promise reply must score much lower: "
        f"good={score_good}, bad={score_bad}"
    )


def test_confidence_action_promise_no_penalty_on_action_query():
    from core.confidence_engine import ConfidenceEngine

    eng = ConfidenceEngine()
    score = eng.score(
        "play the song",
        "Playing the song now, Boss.",
    )
    assert score > 0.4, (
        f"legitimate action on action query must NOT be penalised, got {score}"
    )


# ── Test 16: Soft kLSR 301 restart path (P3) ───────────────────────

@pytest.mark.asyncio
async def test_watchdog_soft_chain_restart_uses_chain_only():
    """kLSRErrorDomain 301 must only call _restart_recognition_chain —
    never the full stop/recreate/start sequence."""
    from voice.stt_watchdog import STTWatchdog

    bus = FakeBus()
    wd = STTWatchdog(bus)

    calls: list[str] = []

    class _Stt:
        def _restart_recognition_chain(self):
            calls.append("chain")

        def begin_full_restart(self):
            calls.append("begin_full")

        def end_full_restart(self):
            calls.append("end_full")

        def stop_listening(self):
            calls.append("stop_listening")

        def start_listening(self, **_kw):
            calls.append("start_listening")
            return True

        def _recreate_recognizer(self):
            calls.append("recreate")

    stt = _Stt()
    await wd._soft_chain_restart(stt, reason="klsr_301_timeout")
    assert calls == ["chain"], (
        f"soft restart should only call _restart_recognition_chain, got {calls}"
    )


# ── Test 17: Voice recovery lock serialises callers (P2) ────────────

@pytest.mark.asyncio
async def test_voice_recovery_lock_serialises_callers():
    from voice.recovery_lock import voice_recovery_lock

    order: list[str] = []

    async def worker(name: str, hold_s: float) -> None:
        async with voice_recovery_lock(name, max_wait_s=3.0) as got:
            order.append(f"enter:{name}:{got}")
            await asyncio.sleep(hold_s)
            order.append(f"exit:{name}")

    await asyncio.gather(
        worker("a", 0.05),
        worker("b", 0.01),
    )
    assert len(order) == 4
    # b must have waited for a to exit before entering (or vice-versa),
    # so we should never see both 'enter' markers next to each other
    # with their matching 'exit' at the end.
    assert order[0].startswith("enter:")
    assert order[1].startswith("exit:")
    assert order[2].startswith("enter:")
    assert order[3].startswith("exit:")


@pytest.mark.asyncio
async def test_voice_recovery_lock_returns_false_on_contention():
    """Second caller must not acquire when the first is holding."""
    from voice.recovery_lock import voice_recovery_lock

    result: dict[str, bool] = {}

    async def a() -> None:
        async with voice_recovery_lock("a", max_wait_s=5.0) as got:
            result["a"] = got
            await asyncio.sleep(0.3)

    async def b() -> None:
        await asyncio.sleep(0.05)
        async with voice_recovery_lock("b", max_wait_s=0.1) as got:
            result["b"] = got

    await asyncio.gather(a(), b())
    assert result["a"] is True
    assert result["b"] is False


# ── Test 18: Passive revert timer kicks on speech_partial (P1b) ─────

@pytest.mark.asyncio
async def test_passive_revert_cancels_on_speech_partial(monkeypatch):
    """Receiving a speech_partial while the revert is pending must cancel
    and re-arm so the user isn't flipped to PASSIVE mid-utterance."""
    from voice.voice_pipeline import VoicePipeline
    from voice.listening_modes import ListeningModeController

    class _Bus:
        def __init__(self):
            self._handlers = {}

        def on(self, event, handler):
            self._handlers.setdefault(event, []).append(handler)

        def emit(self, *a, **kw):
            pass

        def emit_fast(self, *a, **kw):
            pass

        async def dispatch(self, event, **kw):
            for h in self._handlers.get(event, []):
                res = h(**kw)
                if hasattr(res, "__await__"):
                    await res

    bus = _Bus()

    # Simulate a wake-word engine being present so the controller isn't
    # forced into always-active mode (which disables the revert timer).
    class _FakeWakeWord:
        is_available = True

    pipe = VoicePipeline.__new__(VoicePipeline)
    pipe._bus = bus
    pipe._config = {"stt": {"passive_revert_delay_s": 0.3}}
    pipe._wake_word = _FakeWakeWord()
    pipe.stt = None
    pipe._listening_mode = None

    lm = pipe.build_listening_mode()
    # Force into ACTIVE so the revert timer is armed on tts_complete.
    lm.activate("wake_word_detected")
    assert lm.is_active
    assert not getattr(lm, "_always_active", False)

    await bus.dispatch("tts_complete")
    assert pipe._passive_revert_task is not None
    assert not pipe._passive_revert_task.done()

    await asyncio.sleep(0.05)
    # Partial arrives while timer running → must cancel and re-arm.
    old_task = pipe._passive_revert_task
    pipe._passive_revert_last_kick_t = 0.0
    await bus.dispatch("speech_partial", text="hello")
    # Let the event loop propagate the cancellation.
    for _ in range(5):
        if old_task.done() or old_task.cancelled():
            break
        await asyncio.sleep(0)
    assert old_task.cancelled() or old_task.done(), (
        "old revert task should be cancelled when a speech_partial kicks"
    )
    # New task must have been re-armed.
    assert pipe._passive_revert_task is not old_task
    assert pipe._passive_revert_task is not None

    if pipe._passive_revert_task and not pipe._passive_revert_task.done():
        pipe._passive_revert_task.cancel()
        try:
            await pipe._passive_revert_task
        except asyncio.CancelledError:
            pass
