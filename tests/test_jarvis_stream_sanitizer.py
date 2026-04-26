"""ATOM -- JARVIS-level stream-sanitizer & echo regression tests.

Locks in the third wave of "ATOM is talking to itself / sounding like a
chat-bot" fixes pulled from production atomlogs.txt:

  1. Streaming guardrail must drop declarative reasoning-leak sentences
     like 'Based on the response contract ...', 'First, looking at the
     conversation history ...', 'Now they're asking about ...',
     'I need to help the user ...', 'Just respond with the standard
     message.', 'The answer needs to be concise and in plain text.', etc.
  2. Streaming guardrail must drop short lone-quote-tail fragments like
     '" or similar.' which leak when an LLM stops mid-quote.
  3. System prompt must NOT contain the phrase "RESPONSE CONTRACT" (which
     the model was echoing back verbatim) and must carry V8/V9 rules that
     forbid internal-thought narration.
  4. TTS echo guard must allow short yes/no replies after a confirmation
     prompt ("Confirm?", "Proceed?"); Boss's "yes" / "confirm yes" / "no"
     is a real reply, not self-echo.
  5. Audio-intelligence must NOT log "STT stuck but audio flowing" on
     benign chain-rotation reasons (reactive_klsr_301, no_speech_timeout).
  6. Watchdog TTS budget default must be <= 15s so garbled streams get
     killed before a full half-minute of self-talk.
  7. Simple-voice-turn max_tokens override must cap around 96 so a FAST
     reply can't balloon into 128 tokens of CoT.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


sys.path.insert(0, str(_root()))


# ── Controller-side reasoning-leak drops ────────────────────────────


def _make_controller():
    """Instantiate a minimal LocalBrainController for _sanitize_emittable_text."""
    from cursor_bridge.local_brain_controller import LocalBrainController

    class _StubLLM:
        async def generate_streaming(self, prompt, **kwargs):
            return ("", False)

    class _StubBus:
        def emit(self, *a, **kw): pass
        def emit_fast(self, *a, **kw): pass
        def emit_long(self, *a, **kw): pass
        def on(self, *a, **kw): pass

    return LocalBrainController.__new__(LocalBrainController)  # bypass __init__


def test_sanitizer_drops_response_contract_leak() -> None:
    ctrl = _make_controller()
    text = "Based on the response contract, I should confirm my activity and ask how I can help."
    assert ctrl._sanitize_emittable_text(text) == ""


def test_sanitizer_drops_looking_at_conversation_history() -> None:
    ctrl = _make_controller()
    text = "First, looking at the conversation history, the user previously asked if I was active."
    assert ctrl._sanitize_emittable_text(text) == ""


def test_sanitizer_drops_they_re_asking_narration() -> None:
    ctrl = _make_controller()
    text = "Now they're asking about what happened to me."
    assert ctrl._sanitize_emittable_text(text) == ""


def test_sanitizer_drops_i_need_to_help_the_user() -> None:
    ctrl = _make_controller()
    text = "I need to help the user find good wallpaper examples for their MacBook Air."
    assert ctrl._sanitize_emittable_text(text) == ""


def test_sanitizer_drops_dont_need_tools_meta() -> None:
    ctrl = _make_controller()
    text = "I don't need to use any tools here since it's a straightforward confirmation."
    assert ctrl._sanitize_emittable_text(text) == ""


def test_sanitizer_drops_just_respond_meta() -> None:
    ctrl = _make_controller()
    text = "Just respond with the standard message."
    assert ctrl._sanitize_emittable_text(text) == ""


def test_sanitizer_drops_wait_looking_at_tools() -> None:
    ctrl = _make_controller()
    text = "Wait, looking at the available tools, there's the spotlight_search which can search files."
    assert ctrl._sanitize_emittable_text(text) == ""


def test_sanitizer_drops_since_i_cant_browse() -> None:
    ctrl = _make_controller()
    text = "Since I can't browse the internet directly, I should rely on my existing knowledge."
    assert ctrl._sanitize_emittable_text(text) == ""


def test_sanitizer_drops_answer_needs_to_be_concise() -> None:
    ctrl = _make_controller()
    text = "The answer needs to be concise and in plain text."
    assert ctrl._sanitize_emittable_text(text) == ""


def test_sanitizer_drops_lone_quote_tail_or_similar() -> None:
    ctrl = _make_controller()
    text = '" or similar.'
    assert ctrl._sanitize_emittable_text(text) == ""


def test_sanitizer_keeps_real_jarvis_reply() -> None:
    ctrl = _make_controller()
    text = "Yes Boss, I'm here. What do you need?"
    assert ctrl._sanitize_emittable_text(text) == text


def test_sanitizer_keeps_short_wallpaper_answer() -> None:
    ctrl = _make_controller()
    text = "I'll find some good wallpaper examples for your MacBook Air."
    assert ctrl._sanitize_emittable_text(text) == text


# ── System prompt: no RESPONSE CONTRACT verbatim ───────────────────


def test_system_prompt_does_not_contain_response_contract_header() -> None:
    from cursor_bridge.structured_prompt_builder import StructuredPromptBuilder

    # Minimal builder; only the system layer is exercised.
    b = StructuredPromptBuilder.__new__(StructuredPromptBuilder)
    b._owner_name = "Satyam"
    b._role = "buddy"
    b._project = "ATOM"
    b._focus = ""
    b._system_prompt_cache = None
    b._system_prompt_hash = 0
    b._logged_system_prompt = True
    prompt = b._build_system_layer()
    assert "RESPONSE CONTRACT" not in prompt, (
        "System prompt still contains literal 'RESPONSE CONTRACT' which "
        "the model echoes back — rename it (e.g. FINAL-ANSWER RULES)."
    )


def test_system_prompt_v3_uses_opaque_style_fingerprint() -> None:
    """v3 prompt slim: the long quoted-example RESPONSE/VOICE OUTPUT RULES
    block was the source of the verbatim parroting in production logs.
    The new prompt uses opaque, non-quotable phrasing -- assert the new
    STYLE FINGERPRINT block is present and the old quotable headers are
    gone."""
    from cursor_bridge.structured_prompt_builder import StructuredPromptBuilder

    b = StructuredPromptBuilder.__new__(StructuredPromptBuilder)
    b._owner_name = "Satyam"
    b._role = "buddy"
    b._project = "ATOM"
    b._focus = ""
    b._system_prompt_cache = None
    b._system_prompt_hash = 0
    b._logged_system_prompt = True
    prompt = b._build_system_layer()
    # New v3 STYLE FINGERPRINT block must be present.
    assert "OUTPUT STYLE" in prompt
    assert "LENGTH" in prompt
    assert "GROUNDING" in prompt
    # Old quotable rule headers must be GONE -- they were the source of
    # the verbatim parroting in production.
    assert "RESPONSE RULES:" not in prompt
    assert "VOICE OUTPUT RULES" not in prompt
    assert "RESPONSE CONTRACT" not in prompt
    # The parroted phrases from atomlogs.txt must not appear verbatim.
    assert "the final answer only" not in prompt.lower()
    assert "one short jarvis-style line" not in prompt.lower()
    assert "two short sentences max" not in prompt.lower()


def test_query_layer_v3_is_minimal() -> None:
    """v3 query layer: the per-turn FINAL-ANSWER RULES block was being
    mirrored. The new layer is just BOSS:/JARVIS: with no rule text."""
    from cursor_bridge.structured_prompt_builder import StructuredPromptBuilder

    b = StructuredPromptBuilder.__new__(StructuredPromptBuilder)
    b._owner_name = "Satyam"
    out = b._build_query_layer("are you active properly atom")
    assert "BOSS:" in out
    assert "JARVIS:" in out
    assert "are you active properly atom" in out
    # All previously-parroted scaffolding gone.
    assert "FINAL-ANSWER RULES" not in out
    assert "RESPONSE CONTRACT" not in out
    assert "the final answer only" not in out.lower()
    assert "one short" not in out.lower()


# ── TTS echo guard: confirmation-reply exception ────────────────────


def test_echo_guard_allows_yes_after_confirmation_prompt() -> None:
    from voice.tts_macos import MacOSTTSAsync

    tts = MacOSTTSAsync.__new__(MacOSTTSAsync)
    import collections
    import time as _t
    tts._spoken_echo_window = collections.deque(maxlen=6)
    tts._last_spoke_t = _t.monotonic()
    tts._last_spoken_was_confirmation = False
    # Simulate ATOM speaking "Confirm?"
    tts._record_spoken("Confirm?")
    assert tts._last_spoken_was_confirmation is True
    # A real user reply of "Confirm yes" MUST NOT be flagged as echo.
    assert tts.is_echo("Confirm yes") is False
    assert tts.is_echo("yes") is False
    assert tts.is_echo("no") is False


def test_echo_guard_still_catches_self_echo_normal_speech() -> None:
    from voice.tts_macos import MacOSTTSAsync

    tts = MacOSTTSAsync.__new__(MacOSTTSAsync)
    import collections
    import time as _t
    tts._spoken_echo_window = collections.deque(maxlen=6)
    tts._last_spoke_t = _t.monotonic()
    tts._last_spoken_was_confirmation = False
    # ATOM spoke a normal declarative sentence
    tts._record_spoken("The weather today is sunny and pleasant.")
    assert tts._last_spoken_was_confirmation is False
    # If the mic catches us saying "weather today sunny pleasant" back,
    # THAT must still be flagged as echo.
    assert tts.is_echo("weather today sunny pleasant") is True


def test_tts_cleaner_drops_internal_status_lines() -> None:
    from voice.tts_macos import _clean_for_tts

    assert _clean_for_tts("[SYSTEM] Initiating system diagnostics.") == ""
    assert _clean_for_tts("Atom.localBrain.") == ""
    assert (
        _clean_for_tts("System is degraded, Boss. Issues: 2 readiness checks.")
        == ""
    )


# ── Audio intelligence: benign-reason debounce ──────────────────────


def _run_stt_stuck(ai, reason: str, ran: list[bool]) -> None:
    """Helper: invoke _on_stt_stuck on a fresh loop and track recovery runs.

    AudioIntelligenceEngine uses __slots__, so we can't monkeypatch
    instance methods directly. Instead we wrap asyncio.create_task so we
    catch ANY follow-up _smart_stt_recovery coroutine spawned by
    _on_stt_stuck.
    """
    import asyncio
    original_create_task = asyncio.create_task

    def _spy_create_task(coro, *a, **kw):
        # If the coroutine looks like the recovery task, mark it.
        coro_name = getattr(coro, "__qualname__", "") or getattr(getattr(coro, "cr_code", None), "co_name", "")
        if "smart_stt_recovery" in str(coro_name):
            ran.append(True)
            coro.close()
            return original_create_task(asyncio.sleep(0), *a, **kw)
        return original_create_task(coro, *a, **kw)

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        # Patch on the loop so our spy sees it
        asyncio.create_task = _spy_create_task
        try:
            loop.run_until_complete(ai._on_stt_stuck(reason=reason))
        finally:
            asyncio.create_task = original_create_task
    finally:
        loop.close()


def test_on_stt_stuck_ignores_reactive_klsr_301() -> None:
    from voice.audio_intelligence import AudioIntelligenceEngine

    ai = AudioIntelligenceEngine.__new__(AudioIntelligenceEngine)
    ai._stt_restart_times = __import__("collections").deque(maxlen=10)
    ai._selected_input = None
    ai._device_memory = type("D", (), {"record_failure": lambda self, *a, **k: None})()
    ai._last_stt_recovery_t = 0.0
    ran: list[bool] = []
    _run_stt_stuck(ai, "reactive_klsr_301", ran)
    assert ran == [], "Benign reason must NOT trigger smart_stt_recovery"


def test_on_stt_stuck_ignores_no_speech_timeout() -> None:
    from voice.audio_intelligence import AudioIntelligenceEngine

    ai = AudioIntelligenceEngine.__new__(AudioIntelligenceEngine)
    ai._stt_restart_times = __import__("collections").deque(maxlen=10)
    ai._selected_input = None
    ai._device_memory = type("D", (), {"record_failure": lambda self, *a, **k: None})()
    ai._last_stt_recovery_t = 0.0
    ran: list[bool] = []
    _run_stt_stuck(ai, "no_speech_timeout", ran)
    assert ran == []


# ── Config defaults: tighter TTS / LLM watchdogs ────────────────────


def test_watchdog_tts_timeout_is_tight() -> None:
    import json

    with open(_root() / "config" / "settings.json") as f:
        cfg = json.load(f)
    perf = cfg.get("performance", {})
    assert perf.get("watchdog_tts_timeout_s", 30) <= 15, (
        "TTS watchdog must fire within 15s so a stuck/garbled stream "
        "doesn't talk for half a minute."
    )
    assert perf.get("watchdog_llm_timeout_s", 30) <= 20, (
        "LLM watchdog must be <= 20s for a voice-first JARVIS feel."
    )


# ── max_tokens override is tight for voice-first replies ────────────


def test_max_tokens_override_short_is_tight() -> None:
    from cursor_bridge.local_brain_controller import LocalBrainController
    from core.query_policy import ResponseMode

    out = LocalBrainController._max_tokens_override(
        response_mode=ResponseMode.SHORT,
        budget_tier="command",
        requested_tier="command",
    )
    assert out is not None and out <= 96, (
        f"SHORT voice turn must cap around 80 tokens, got {out}"
    )


def test_max_tokens_override_simple_is_tight() -> None:
    from cursor_bridge.local_brain_controller import LocalBrainController
    from core.query_policy import ResponseMode

    out = LocalBrainController._max_tokens_override(
        response_mode=ResponseMode.NORMAL,
        budget_tier="simple",
        requested_tier="simple",
    )
    assert out is not None and out <= 128, (
        f"SIMPLE voice turn must cap <= 128 tokens, got {out}"
    )
