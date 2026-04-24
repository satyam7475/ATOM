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


@pytest.mark.parametrize(
    "utterance",
    [
        "hey adam how are you",
        "adam are you there",
        "hey atum what time is it",
        "hey autumn open calendar",
        "hey atam tell me the news",
        "ok atom what's the status",
    ],
)
def test_wake_word_filter_tolerates_stt_mishearings(utterance: str) -> None:
    """Apple SFSpeechRecognizer (en-IN locale) routinely renders 'atom' as
    'adam', 'atum', 'autumn', 'atam'. Before the fix, every one of these
    was silently suppressed and ATOM appeared dead; the filter must now
    accept them as equivalent to 'atom'."""
    from voice.listening_modes import WakeWordFilter

    wf = WakeWordFilter(cooldown_s=0.0)
    assert wf.check(utterance) is not None, (
        f"STT-mishearing variant was not accepted: {utterance!r}"
    )


@pytest.mark.parametrize(
    "utterance",
    [
        "are you there",
        "can you hear me",
        "hello atom",
        "are you listening",
    ],
)
def test_wake_word_filter_direct_address(utterance: str) -> None:
    """Direct-address phrases should also wake ATOM from PASSIVE mode so
    a user clearly addressing the assistant never gets stranded."""
    from voice.listening_modes import WakeWordFilter

    wf = WakeWordFilter(cooldown_s=0.0)
    assert wf.check(utterance) is not None, (
        f"Direct-address phrase was not accepted: {utterance!r}"
    )


def test_wake_word_filter_contains_wake_matches_mishearings() -> None:
    """The PASSIVE final-gate uses contains_wake; it must agree with
    check() on STT-mishearing variants so a final that slipped through
    without activating the partial scanner still reaches the router."""
    from voice.listening_modes import WakeWordFilter

    assert WakeWordFilter.contains_wake("hey adam how are you")
    assert WakeWordFilter.contains_wake("adam are you there")
    assert WakeWordFilter.contains_wake("hey atum what time is it")
    assert not WakeWordFilter.contains_wake("what time is it")
    assert not WakeWordFilter.contains_wake("play that song")


def test_wake_word_engine_normalizes_configured_model_names() -> None:
    """Wake-word model names should be normalized from config so spaces,
    case, and list-vs-string forms all resolve predictably."""
    from voice.wake_word import WakeWordEngine

    assert WakeWordEngine._configured_model_names({"model": "Hey Jarvis"}) == [
        "hey_jarvis",
    ]
    assert WakeWordEngine._configured_model_names(
        {"models": [" Hey Jarvis ", "custom_atom"]},
    ) == ["hey_jarvis", "custom_atom"]


def test_tts_jarvis_preset_prefers_best_daniel_voice(monkeypatch) -> None:
    """The ``jarvis`` preset should choose the best Daniel-family voice first.

    This lets ATOM use compact Daniel immediately on a stock Mac while
    auto-upgrading to enhanced/premium Daniel if the user installs it later.
    """
    import voice.tts_macos as tts_macos

    class _FakeSynth:
        @staticmethod
        def availableVoices():
            return [
                "com.apple.voice.compact.en-US.Samantha",
                "com.apple.voice.compact.en-GB.Daniel",
                "com.apple.voice.premium.en-GB.Daniel",
            ]

        @staticmethod
        def defaultVoice():
            return None

    class _FakeAppKit:
        NSSpeechSynthesizer = _FakeSynth

    monkeypatch.setattr(tts_macos, "_HAS_NATIVE", True)
    monkeypatch.setattr(tts_macos, "_AppKit", _FakeAppKit)
    monkeypatch.setattr(
        tts_macos,
        "_spoken_content_voice_from_prefs",
        lambda _available: "",
    )

    assert (
        tts_macos._pick_best_voice("jarvis")
        == "com.apple.voice.premium.en-GB.Daniel"
    )


def test_tts_pitch_shift_keeps_daniel_neutral() -> None:
    """British Daniel should not get the generic brightness boost."""
    import voice.tts_macos as tts_macos

    assert tts_macos._preferred_pitch_shift("com.apple.voice.compact.en-GB.Daniel") == 0.0
    assert tts_macos._preferred_pitch_shift("com.apple.voice.compact.en-US.Samantha") == 2.0


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


def test_voice_pipeline_always_on_mode_bypasses_wake_word_build(monkeypatch):
    """Explicit always-on mode should skip OpenWakeWord entirely."""
    import voice.wake_word as wake_word_mod
    from voice.voice_pipeline import VoicePipeline

    called: list[str] = []

    class _FakeWakeWordEngine:
        def __init__(self, *_a, **_kw):
            called.append("init")

    monkeypatch.setattr(wake_word_mod, "WakeWordEngine", _FakeWakeWordEngine)

    pipe = VoicePipeline.__new__(VoicePipeline)
    pipe._bus = FakeBus()
    pipe._state = object()
    pipe._config = {"voice": {"activation_mode": "always_on"}}
    pipe._wake_word = "sentinel"

    assert pipe.build_wake_word() is None
    assert pipe._wake_word is None
    assert called == []


def test_voice_pipeline_always_on_mode_forces_active_listener():
    """Always-on activation must keep the listening-mode controller active
    even if a wake-word engine is technically present."""
    from voice.voice_pipeline import VoicePipeline

    class _FakeWakeWord:
        is_available = True

    pipe = VoicePipeline.__new__(VoicePipeline)
    pipe._bus = FakeBus()
    pipe._config = {
        "voice": {"activation_mode": "always_on"},
        "stt": {"passive_revert_delay_s": 0.3},
    }
    pipe._wake_word = _FakeWakeWord()
    pipe.stt = None
    pipe._listening_mode = None

    lm = pipe.build_listening_mode()
    assert lm.is_active
    assert getattr(lm, "_always_active", False) is True


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


# ── Test 18: TTS instruction-leak guards (atomlogs.txt regression) ──

def test_instruction_echo_regex_catches_quoted_user_prefix():
    """`"Dear Boss" — the user is greeting you, so respond...` is the
    classic Qwen-3 small leak we saw in production. The hardened regex
    must flag it as instruction-echo so the controller rejects/retries."""
    from cursor_bridge.local_brain_controller import _INSTRUCTION_ECHO_RE

    leaks = [
        '"Dear Boss" — the user is greeting you, so respond politely.',
        '"hey atom" - the user is asking for the weather, so reply briefly.',
        "'Dear boss' -- so respond warmly with a short greeting.",
    ]
    for text in leaks:
        assert _INSTRUCTION_ECHO_RE.search(text), (
            f"hardened instruction-echo regex missed leak: {text!r}"
        )


def test_cot_preface_stripper_peels_quoted_prefix():
    """The controller-side stripper should peel the quoted-user-text +
    narration prefix so any survivor reaching TTS only contains the real
    answer tail (or empty if the model never produced one)."""
    from cursor_bridge.local_brain_controller import _strip_cot_preface

    text = (
        '"Dear Boss" — the user is greeting you, so respond politely. '
        "Hello, Boss."
    )
    out = _strip_cot_preface(text)
    assert out.lower().startswith("hello"), (
        f"stripper failed to peel quoted+narration head: {out!r}"
    )


def test_mlx_cot_preface_stripper_peels_quoted_prefix():
    """The MLX-side stripper mirrors the controller stripper. Both must
    converge on the same outcome so guards stay aligned."""
    from brain.mlx_llm import _strip_cot_prefaces

    text = (
        '"Dear Boss" -- the user is greeting you, so respond politely. '
        "Newton's first law states that a body at rest stays at rest."
    )
    out = _strip_cot_prefaces(text)
    assert "Newton" in out
    assert "the user is greeting" not in out.lower()


# Stage-direction leak regression — exact six strings observed in
# atom_log.txt (lines 323, 456, 489, 536, 572, 655) when Qwen3-4B parroted
# the persona-adjective phrases back as a parenthetical.
_STAGE_DIRECTION_LEAKS = [
    "(in a calm, composed tone)",
    "(in a calm, composed tone).",
    "(calm, composed tone)",
    "(calm, composed tone).",
    "(in a composed, friendly tone). On it, Boss.",
    "(softly) Right away, Boss.",
    "(warmly, but professionally) Understood.",
    # New shapes proved to leak in atom_log.txt 2026-04-25 boot:
    # L357 — TTS spoke "(in a calm, composed tone" (open paren, no ')')
    "(in a calm, composed tone",
    # L554 — TTS spoke "(calm, composed tone." (open paren with inner dot)
    "(calm, composed tone.",
    # L409 — TTS spoke "(responds immediately) Sure, Boss." (narration verb)
    "(responds immediately) Sure, Boss.",
    "(responds immediately)",
    "(answers quickly) On it, Boss.",
    "(pauses briefly). Got it.",
]


def test_controller_strips_bare_stage_direction_leaks():
    """The controller-side stripper must peel any leading parenthetical
    that describes voice/tone/manner — those reached TTS six times in
    atom_log.txt and broke every conversational reply."""
    from cursor_bridge.local_brain_controller import _strip_cot_preface

    for raw in _STAGE_DIRECTION_LEAKS:
        out = _strip_cot_preface(raw)
        # Either the whole thing was a stage direction (and out is empty
        # or just trailing punctuation), or the actual reply survived.
        leftover = out.lower()
        assert "tone)" not in leftover and "calm" not in leftover and \
               "composed" not in leftover and "softly" not in leftover and \
               "warmly" not in leftover, (
            f"stage-direction leak survived stripper: {raw!r} -> {out!r}"
        )


def test_mlx_strips_bare_stage_direction_leaks():
    """MLX-side stripper mirrors the controller — both layers must catch
    the leak so we have defense-in-depth before TTS."""
    from brain.mlx_llm import _strip_cot_prefaces

    for raw in _STAGE_DIRECTION_LEAKS:
        out = _strip_cot_prefaces(raw)
        leftover = out.lower()
        assert "tone)" not in leftover and "calm" not in leftover and \
               "composed" not in leftover and "softly" not in leftover and \
               "warmly" not in leftover, (
            f"stage-direction leak survived MLX stripper: {raw!r} -> {out!r}"
        )


def test_stage_direction_strip_preserves_factual_parentheticals():
    """Length-cap + mood-keyword anchor means factual parentheticals
    like '(see line 12)' and '(2 of 3)' must pass through untouched."""
    from cursor_bridge.local_brain_controller import _strip_cot_preface
    from brain.mlx_llm import _strip_cot_prefaces

    safe_inputs = [
        "(see line 12) Boss, the file is at /tmp/foo.",
        "(2 of 3) Continuing the briefing, Boss.",
        "Boss, that's the third one (after the first two).",
        "(updated) Done, Boss.",
    ]
    for raw in safe_inputs:
        c_out = _strip_cot_preface(raw)
        m_out = _strip_cot_prefaces(raw)
        assert c_out == raw, (
            f"controller stripper false-positive on {raw!r} -> {c_out!r}"
        )
        assert m_out == raw, (
            f"mlx stripper false-positive on {raw!r} -> {m_out!r}"
        )


def test_repeat_hint_lives_in_system_layer_only():
    """Repeat-hint must NEVER be spliced into the user query. It belongs
    in the system layer where the model cannot quote it back at TTS."""
    from cursor_bridge.structured_prompt_builder import StructuredPromptBuilder

    pb = StructuredPromptBuilder({})
    base = pb.build("hello", repeat_hint=False)
    hinted = pb.build("hello", repeat_hint=True)
    assert "TURN STEER" in hinted
    assert "TURN STEER" not in base
    # The user query layer must not contain the steer or any "[SYSTEM NOTE"
    # marker -- if it did, small models could echo it during TTS.
    assert "[SYSTEM NOTE" not in hinted
    # The v3 query layer is "BOSS:\n{query}\n\nJARVIS:" (the old
    # "CURRENT USER REQUEST:" header was removed because Qwen3-8B was
    # mirroring the per-turn rule block back at TTS). The TURN STEER
    # block must still appear before the BOSS: marker so the model
    # cannot quote it as part of a reply.
    assert "BOSS:" in hinted
    boss_pos = hinted.rindex("BOSS:")
    steer_pos = hinted.index("TURN STEER")
    assert steer_pos < boss_pos, (
        "TURN STEER must live BEFORE the BOSS:/JARVIS: query layer, "
        "not inside it -- otherwise the model echoes it out loud."
    )


# ── Test 19: Wake-word/direct-address mishearing routing ──

def test_wake_word_filter_accepts_dear_boss_mishearing():
    """``dear boss`` is the recurring SFSpeech rendering of ``hey boss``.
    It must be treated as direct-address so the router short-circuits
    instead of routing the partial to the LLM. ``contains_wake`` has no
    cooldown so we use it for the back-to-back assertions."""
    from voice.listening_modes import WakeWordFilter

    assert WakeWordFilter.contains_wake("dear boss")
    assert WakeWordFilter.contains_wake("hey boss")
    assert WakeWordFilter.contains_wake("yes boss")
    # Bare owner-name (``satyam`` alone) is NOT treated as direct-address by
    # the regex — that prevents false positives when Boss is just chatting
    # about himself in third person. The bare-wake short-circuit in the
    # router still ack's a standalone "satyam" because WAKE_PHRASES has it.
    assert not WakeWordFilter.contains_wake("called my friend satyam yesterday")
    assert "satyam" in WakeWordFilter.WAKE_PHRASES
    # And the stateful checker still triggers at least once for the new term.
    wf = WakeWordFilter(cooldown_s=0.3)
    assert wf.check("dear boss") is not None


def test_router_short_circuits_bare_wake_utterance():
    """``Router._is_bare_wake_utterance`` should fire for stand-alone
    wake / direct-address tokens but NOT for genuine commands."""
    from core.router.router import Router

    is_bare = Router._is_bare_wake_utterance
    assert is_bare("atom")
    assert is_bare("hey atom")
    assert is_bare("dear boss")
    assert is_bare("hey boss")
    assert is_bare("hey boss please")
    # Bare ``boss`` alone is intentionally NOT a wake — it's a generic noun
    # too easily mid-sentence; only ``hey/dear/yes/hello boss`` qualifies.
    assert not is_bare("boss")
    # Real commands must still flow through the LLM/router.
    assert not is_bare("hey atom what time is it")
    assert not is_bare("dear boss tell me a joke")
    assert not is_bare("hey boss open chrome for me")


# ── Test 20: TTS echo guard + barge-in min-words floor ──

def test_tts_is_echo_recognises_recent_spoken_text(monkeypatch):
    """``tts.is_echo`` must answer True for partials whose words are all
    in the last few spoken slices (mic catching ATOM's own voice) and
    False for unrelated user speech."""
    import voice.tts_macos as tts_mod

    tts = tts_mod.MacOSTTSAsync.__new__(tts_mod.MacOSTTSAsync)
    import collections as _c
    tts._spoken_echo_window = _c.deque(maxlen=6)
    tts._last_spoke_t = 0.0

    tts._record_spoken("Dear Boss the user is greeting you so respond politely")
    assert tts.is_echo("Dear")  # single token while speaking → echo
    assert tts.is_echo("Dear boss")
    assert not tts.is_echo("open chrome and play music")


@pytest.mark.asyncio
async def test_interrupt_handler_drops_thin_partial_during_tts():
    """A 1-word partial like 'Dear' captured while ATOM is SPEAKING and
    matches our echo guard must NOT trigger a voice-interrupt resume."""
    from types import SimpleNamespace

    from voice.interrupt_handler import VoiceInterruptHandler
    from core.state_manager import AtomState

    bus = FakeBus()

    class _Tts:
        def is_echo(self, _t):
            return True

    state = SimpleNamespace(current=AtomState.SPEAKING)
    h = VoiceInterruptHandler(
        bus=bus,
        state=state,
        tts=_Tts(),
        emit_cooldown_s=0.0,
    )
    await h.on_speech_partial(text="Dear")
    assert not any(e == "resume_listening" for e, _ in bus.emitted_events), (
        "single-word echo partial must be silently suppressed"
    )


@pytest.mark.asyncio
async def test_interrupt_handler_requires_min_words_when_not_echo():
    """Even without echo evidence, a single-word partial during SPEAKING
    is too thin a signal to cancel TTS — wait for the burst path or for
    a richer partial."""
    from types import SimpleNamespace

    from voice.interrupt_handler import VoiceInterruptHandler
    from core.state_manager import AtomState

    bus = FakeBus()

    class _Tts:
        def is_echo(self, _t):
            return False

    state = SimpleNamespace(current=AtomState.SPEAKING)
    h = VoiceInterruptHandler(
        bus=bus,
        state=state,
        tts=_Tts(),
        emit_cooldown_s=0.0,
    )
    await h.on_speech_partial(text="Dear")
    assert not any(e == "resume_listening" for e, _ in bus.emitted_events)


# ── Test 21: Cold-start staleness gate ────────────────────────────

def test_cold_start_drops_stale_snapshot_at_load_time(monkeypatch, tmp_path):
    """Snapshots older than the TTL must be discarded inside
    ``_load_snapshot`` itself so the boot report's ``context`` flag
    reflects reality (previously the gate only ran inside
    ``_build_context_payload``)."""
    import time as _time

    from core.boot import cold_start as cs

    fake_path = tmp_path / "snap.json"

    class _FakePersistence:
        def register(self, *_a, **_kw):
            return None

        def load(self, _key):
            return {
                "saved_at": _time.time() - (cs._MAX_RESTORED_CONTEXT_AGE_S + 3600),
                "system_state": {"cpu": 5.0, "ram": 10.0},
            }

    monkeypatch.setattr(cs, "persistence_manager", _FakePersistence())

    from types import SimpleNamespace

    opt = cs.ColdStartOptimizer(
        config={},
        state_manager=SimpleNamespace(current=None),
        memory_store=SimpleNamespace(),
        intent_engine=SimpleNamespace(),
        snapshot_path=fake_path,
    )
    loaded = opt._load_snapshot()
    assert loaded == {}, (
        "stale snapshot must be discarded at load time, "
        f"got {loaded!r}"
    )


# ── JARVIS-feel tests: bare-wake ack rotation, system-note scrub, echo guard ─

def test_bare_wake_ack_rotates_warm_variants() -> None:
    """Successive bare wakes never replay the same ack twice in a row.

    JARVIS-feel: a single fixed phrase ("Yes, Boss?") sounds robotic on
    repeated calls. Router rotates through a small warm pool so even
    five back-to-back wakes feel present, not scripted.
    """
    from core.router.router import Router

    Router._BARE_WAKE_ACK_INDEX = 0
    seen: list[str] = []
    for _ in range(len(Router._BARE_WAKE_ACKS)):
        ack = Router._pick_bare_wake_ack()
        seen.append(ack)
        assert ack and isinstance(ack, str), "ack must be a non-empty string"

    assert len(set(seen)) == len(Router._BARE_WAKE_ACKS), (
        f"all rotation entries should be unique: {seen}"
    )
    next_ack = Router._pick_bare_wake_ack()
    assert next_ack == Router._BARE_WAKE_ACKS[0], "rotation must wrap"


def test_local_brain_strips_legacy_system_note_from_user_query() -> None:
    """Even if some legacy code path splices ``[SYSTEM NOTE: …]`` back into
    the user query, the controller must scrub it before logging or
    sending to the LLM. Otherwise small instruction-tuned models echo
    the bracketed text during TTS as quoted analysis.
    """
    import re as _re

    leak = (
        "Dear boss\n\n"
        "[SYSTEM NOTE: The user asked this before and wasn't satisfied "
        "with the previous answer. Provide a different answer.]"
    )
    cleaned = _re.sub(
        r"\[SYSTEM NOTE:[^\]]*\](?:[^\n]*\n?)*",
        "",
        leak,
    ).strip()
    assert "[SYSTEM NOTE" not in cleaned
    assert cleaned == "Dear boss"


def test_tts_is_echo_catches_single_token_partial_during_speech() -> None:
    """Echo guard must flag a 1-token partial like ``Dear`` when TTS just
    spoke text containing that token. Without this, NSSpeechSynthesizer
    spilling into the mic triggers a false interrupt and ATOM cuts itself
    off mid-sentence (the SPEAKING -> own-voice -> LISTENING loop).
    """
    from core.state_manager import AtomState, StateManager
    from voice.tts_macos import MacOSTTSAsync

    class _FakeBus:
        def on(self, *_a, **_k): pass
        def emit(self, *_a, **_k): pass
        def emit_fast(self, *_a, **_k): pass

    bus = _FakeBus()
    state = StateManager(bus, initial=AtomState.IDLE)
    tts = MacOSTTSAsync(bus, state)
    tts._record_spoken(
        '"Dear Boss" — the user is greeting you, so respond politely.'
    )
    assert tts.is_echo("Dear") is True
    assert tts.is_echo("dear boss") is True
    assert tts.is_echo("hello atom") is False, (
        "non-overlapping partials must NOT be flagged as echo"
    )


def test_prompt_builder_surfaces_developer_focus_and_environment() -> None:
    """JARVIS-feel: when the dev provided ``focus`` and the caller passed
    an ``active_app``/clipboard context, the LLM prompt must actually
    carry those signals. Otherwise the model produces generic answers
    detached from what Boss is looking at.
    """
    from cursor_bridge.structured_prompt_builder import StructuredPromptBuilder

    builder = StructuredPromptBuilder({
        "developer": {
            "role": "Backend engineer",
            "focus": "Python and FastAPI microservices",
            "project_name": "TestProj",
        },
    })
    prompt = builder.build(
        "explain this code",
        context={
            "active_app": "VS Code",
            "window_title": "main.py - VS Code",
            "clipboard": "def foo(): pass",
        },
    )
    assert "Python" in prompt
    assert "FastAPI" in prompt
    assert "VS Code" in prompt
    assert "def foo(): pass" in prompt
    assert "Environment:" in prompt


# ── Hard guards against the recurring "Yeah Boss" instruction-leak ───────

def test_instruction_leak_caught_in_full_sentence_form() -> None:
    """The full leak with sentence-ending period must be hard-rejected so
    TTS never speaks ``"Yeah Boss" -- the user is greeting you, respond
    politely and warmly.``
    """
    from cursor_bridge.local_brain_controller import (
        _looks_like_pure_instruction_leak,
        _strip_cot_preface,
    )

    leak = '"Yeah Boss" — the user is greeting you, respond politely and warmly.'
    assert _looks_like_pure_instruction_leak(leak) is True
    assert _strip_cot_preface(leak) == ""


def test_instruction_leak_caught_when_streamed_at_first_clause() -> None:
    """Streaming flushes the FIRST clause at the comma boundary, so the
    sanitiser sees ``"Yeah Boss" — the user is greeting you,`` with no
    sentence-ending punctuation. Without comma-tolerant stripping that
    fragment slips straight into TTS.
    """
    from cursor_bridge.local_brain_controller import (
        _looks_like_pure_instruction_leak,
        _strip_cot_preface,
    )

    fragments = [
        '"Yeah Boss" — the user is greeting you,',
        '"Yeah Boss" — the user is greeting you',
        'the user is greeting you, respond politely and warmly',
        'the user is asking what universe',
        'Boss is asking about the weather',
    ]
    for frag in fragments:
        assert _looks_like_pure_instruction_leak(frag) is True, (
            f"Leak fragment must be hard-rejected: {frag!r}"
        )
        assert _strip_cot_preface(frag) == "", (
            f"Leak fragment must strip to empty: {frag!r} -> "
            f"{_strip_cot_preface(frag)!r}"
        )


def test_genuine_responses_pass_through_leak_filter() -> None:
    """Real answers must NOT be flagged as instruction leaks. False
    positives here would silence ATOM completely on normal turns.
    """
    from cursor_bridge.local_brain_controller import (
        _looks_like_pure_instruction_leak,
        _strip_cot_preface,
    )

    real_replies = [
        "Hi Boss, how are you?",
        "I am ATOM, your personal AI buddy.",
        "The weather is sunny today.",
        "Sure, Boss. Opening Spotify now.",
        "Right here, Boss.",
        "I don't know that one yet, Boss.",
    ]
    for reply in real_replies:
        assert _looks_like_pure_instruction_leak(reply) is False, (
            f"Genuine answer wrongly flagged as leak: {reply!r}"
        )
        assert _strip_cot_preface(reply) == reply, (
            f"Genuine answer was modified by stripper: {reply!r} -> "
            f"{_strip_cot_preface(reply)!r}"
        )


def test_sanitize_emittable_text_drops_streamed_leak_fragment() -> None:
    """Integration check: the controller's emit-path sanitiser must
    return an empty string for a streamed leak clause so the bus never
    receives a partial_response with that text.
    """
    from cursor_bridge.local_brain_controller import LocalBrainController

    class _StubBus:
        def on(self, *_a, **_k): pass
        def emit(self, *_a, **_k): pass
        def emit_fast(self, *_a, **_k): pass
        def emit_long(self, *_a, **_k): pass

    class _StubBuilder:
        def build(self, *_a, **_k): return ""

    ctrl = LocalBrainController.__new__(LocalBrainController)
    ctrl._compact_text = LocalBrainController._compact_text  # type: ignore[attr-defined]

    cleaned = LocalBrainController._sanitize_emittable_text(
        ctrl,
        '"Yeah Boss" — the user is greeting you,',
    )
    assert cleaned == "", (
        f"Streamed leak must produce empty emit; got {cleaned!r}"
    )


def test_reject_low_quality_answer_keeps_paired_quotes_for_match() -> None:
    """Regression guard: don't strip leading/trailing quotes one-sided
    BEFORE the instruction-echo regex runs, otherwise the leak
    ``"Yeah Boss" — the user is greeting you, ...`` evades the rejector
    because the opening ``"`` is gone but the closing ``"`` remains.
    """
    from cursor_bridge.local_brain_controller import LocalBrainController

    ctrl = LocalBrainController.__new__(LocalBrainController)
    ctrl._compact_text = LocalBrainController._compact_text  # type: ignore[attr-defined]

    leak = '"Yeah Boss" — the user is greeting you, respond politely and warmly.'
    assert LocalBrainController._reject_low_quality_answer(ctrl, "yeah boss", leak) is True


def test_yeah_boss_is_treated_as_bare_wake_now() -> None:
    """``yeah boss`` is a conversational ack, not a real query. Without
    short-circuiting it the LLM spins up a 5s reply for a single-word
    acknowledgement.
    """
    from voice.listening_modes import WakeWordFilter

    assert "yeah boss" in WakeWordFilter.WAKE_PHRASES
    assert "yep boss" in WakeWordFilter.WAKE_PHRASES
    assert "thanks boss" in WakeWordFilter.WAKE_PHRASES
    assert "cool boss" in WakeWordFilter.WAKE_PHRASES

    from core.router.router import Router

    assert Router._is_bare_wake_utterance("yeah boss") is True
    assert Router._is_bare_wake_utterance("yeah boss?") is True
    assert Router._is_bare_wake_utterance("thanks boss.") is True
    assert Router._is_bare_wake_utterance("yeah boss please open chrome") is False


def test_perception_predicted_interrupt_drops_self_echo_partials() -> None:
    """Without this guard, a single-word echo like 'Boss' bleeding back
    through the speakers fires perception's interrupt_predicted, which
    pre-pauses TTS and drops state to LISTENING — producing the
    SPEAKING -> own-voice -> LISTENING flap loop the user reported.

    The wiring layer must now consult ``tts.is_echo`` BEFORE acting on a
    perception-predicted interrupt.
    """
    import asyncio

    from core.state_manager import AtomState

    flag = {"interrupted": False}

    class _FakeVoiceInterrupt:
        async def interrupt_to_listening(self, **_kw):
            flag["interrupted"] = True
            return True

    class _FakeTTS:
        def is_echo(self, _text, **_kw):
            return True

    class _FakeState:
        def __init__(self):
            self.current = AtomState.SPEAKING

    voice_interrupt = _FakeVoiceInterrupt()
    tts = _FakeTTS()
    state = _FakeState()
    _PERCEPTION_INTERRUPT_MIN_WORDS = 2

    async def _on_interrupt_predicted(**_kw) -> None:
        partial_text = str(_kw.get("text", "") or "").strip()
        if partial_text:
            check_echo = getattr(tts, "is_echo", None)
            if callable(check_echo) and check_echo(partial_text):
                return
            if state.current is AtomState.SPEAKING:
                if len(partial_text.split()) < _PERCEPTION_INTERRUPT_MIN_WORDS:
                    return
        await voice_interrupt.interrupt_to_listening(
            trigger="interrupt_predicted",
            reason="perception_predicted",
            user_interrupt=True,
        )

    asyncio.run(_on_interrupt_predicted(text="Boss the user"))
    assert flag["interrupted"] is False, (
        "self-echo partial must NOT trigger interrupt_to_listening"
    )
