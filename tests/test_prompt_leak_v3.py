"""Phase 1 v3 regression tests -- prompt leak, TTS guard, intent boot grace.

These tests pin the specific failure modes observed in atomlogs.txt
2026-04-21 02:15-02:16 session:

  1. ATOM spoke "the final answer only." and "One short line." aloud
     because Qwen3-8B regurgitated literal lines from its own system
     prompt's FINAL-ANSWER RULES block.

  2. ATOM spoke "if the question is a simple, short, or info query,
     give one short sentence when possible, two short..." -- a verbatim
     copy of the RESPONSE RULES line + the SHORT response-mode hint.

  3. The "hey what time is it" turn went to fallback/LLM (and got a
     generic greeting, not the time) because the 50 ms intent_engine
     watchdog killed the lookup on the first cold turn.

The fixes (see plan jarvis-grade-atom-v3, Phase 1):

  A. Slim the system prompt -- remove all quotable rule text.
  B. Strip per-turn FINAL-ANSWER RULES block from query layer.
  C. Add prompt-text fingerprint detector in the controller sanitiser.
  D. Mirror the same fingerprint detector inside MacOSTTSAsync as a
     final defence-in-depth audio guard.
  E. Add a configurable boot-grace window where intent_engine has no
     budget so the very first cold turn can run all regex groups.

This file is the regression suite for all of those.
"""
from __future__ import annotations

import re
import time

import pytest


# ─────────────────────────────────────────────────────────────────────
# A. System prompt no longer contains parroted phrases
# ─────────────────────────────────────────────────────────────────────


def _make_prompt_builder():
    from cursor_bridge.structured_prompt_builder import StructuredPromptBuilder

    b = StructuredPromptBuilder.__new__(StructuredPromptBuilder)
    b._owner_name = "Satyam"
    b._role = "buddy"
    b._project = "ATOM"
    b._focus = ""
    b._system_prompt_cache = None
    b._system_prompt_hash = 0
    b._logged_system_prompt = True
    return b


# Phrases that appeared verbatim in atomlogs.txt as ATOM speaking its
# own prompt out loud. None of them must remain in the prompt.
PARROTED_PHRASES_FROM_LOG = [
    "the final answer only",
    "one short jarvis-style line",
    "if the question is a simple, short, or info query",
    "give one short sentence when possible",
    "two short sentences max",
    "plain text only",
    "output only the final answer",
    "spoken = final answer",
    "if the thought feels like planning",
    "boss only hears what's spoken",
    "reply with the final answer",
    "no markdown, no bullets",
]


def test_system_prompt_strips_all_parroted_phrases():
    prompt = _make_prompt_builder()._build_system_layer().lower()
    for phrase in PARROTED_PHRASES_FROM_LOG:
        assert phrase not in prompt, (
            f"System prompt still contains parroted phrase: {phrase!r}. "
            "Qwen3-8B will regurgitate this in voice replies."
        )


def test_system_prompt_keeps_v3_style_fingerprint():
    """The slim STYLE FINGERPRINT block must replace the old rules."""
    prompt = _make_prompt_builder()._build_system_layer()
    assert "OUTPUT STYLE" in prompt
    assert "LENGTH" in prompt
    assert "GROUNDING" in prompt
    assert "LANGUAGE" in prompt


def test_system_prompt_no_old_rule_headers():
    """The 'RESPONSE RULES:' / 'VOICE OUTPUT RULES' headers were the
    source of the parroting -- they must not return."""
    prompt = _make_prompt_builder()._build_system_layer()
    assert "RESPONSE RULES:" not in prompt
    assert "VOICE OUTPUT RULES" not in prompt
    assert "RESPONSE CONTRACT" not in prompt
    assert "FINAL-ANSWER RULES" not in prompt


def test_query_layer_has_no_per_turn_rules():
    """The per-turn FINAL-ANSWER RULES block was being mirrored."""
    out = _make_prompt_builder()._build_query_layer("what time is it")
    for phrase in PARROTED_PHRASES_FROM_LOG:
        assert phrase not in out.lower(), (
            f"Query layer still contains: {phrase!r}"
        )
    assert "BOSS:" in out
    assert "JARVIS:" in out
    assert "what time is it" in out


def test_short_response_mode_hint_is_opaque():
    """The literal 'one short sentence when possible, two short sentences
    max' hint that Qwen mirrored is gone."""
    from cursor_bridge.structured_prompt_builder import _query_type_hint

    hint = _query_type_hint("what is X")  # SHORT mode classification
    assert "one short sentence when possible" not in hint.lower()
    assert "two short sentences max" not in hint.lower()


# ─────────────────────────────────────────────────────────────────────
# C. Controller sanitiser drops every prompt-leak fingerprint
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("leak_text", [
    "the final answer only.",
    "the final answer only. One short line.",
    "One short JARVIS-style line.",
    "Plain text only.",
    "if the question is a simple, short, or info query, give one short "
    "sentence when possible, two short sentences max.",
    "give one short sentence when possible, two short sentences max.",
    "two short sentences max.",
    "Output only the final answer Boss would hear from a calm, capable "
    "friend.",
    "Spoken = final answer.",
    "If the thought feels like planning, drop it.",
    "Reply with the final answer only.",
    "Respond in plain text only.",
    "No markdown, no bullets.",
    "Boss only hears what's spoken.",
])
def test_looks_like_prompt_leak_detects(leak_text):
    from cursor_bridge.local_brain_controller import _looks_like_prompt_leak

    assert _looks_like_prompt_leak(leak_text), (
        f"Detector missed prompt-leak fragment: {leak_text!r}"
    )


@pytest.mark.parametrize("good_text", [
    "It's 2:15 AM, Boss.",
    "Done.",
    "Newton's first law states an object in motion stays in motion.",
    "Opening Safari now.",
    "I don't know yet, Boss -- want me to look it up?",
    "The temperature in Delhi is 28 degrees.",
    "Right away.",
    "Got it.",
    "Sure, here's the file path.",
])
def test_looks_like_prompt_leak_no_false_positives(good_text):
    from cursor_bridge.local_brain_controller import _looks_like_prompt_leak

    assert not _looks_like_prompt_leak(good_text), (
        f"Detector falsely flagged good answer as leak: {good_text!r}"
    )


def test_sanitizer_drops_prompt_leak_fragments():
    """End-to-end: feed the literal log fragments through the controller
    sanitiser; output must be empty (suppressed before TTS)."""
    from cursor_bridge.local_brain_controller import LocalBrainController

    ctrl = LocalBrainController.__new__(LocalBrainController)

    log_leaks = [
        "the final answer only.",
        "the final answer only. One short line.",
        "if the question is a simple, short, or info query, give one "
        "short sentence when possible, two short sentences max.",
    ]
    for leak in log_leaks:
        out = ctrl._sanitize_emittable_text(leak)
        assert out == "", (
            f"Sanitizer must drop prompt-leak fragment: {leak!r} -> "
            f"got {out!r}"
        )


# ─────────────────────────────────────────────────────────────────────
# D. TTS-side fingerprint guard (defence-in-depth)
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("leak_text", [
    "the final answer only.",
    "One short JARVIS-style line.",
    "Plain text only.",
    "if the question is a simple, short, or info query, give one short "
    "sentence when possible.",
    "Output only the final answer Boss would hear.",
    "spoken = final answer.",
    "Reply with the final answer only.",
])
def test_tts_module_guard_rejects(leak_text):
    from voice.tts_macos import _is_prompt_leak

    assert _is_prompt_leak(leak_text), (
        f"TTS module guard missed: {leak_text!r}"
    )


def test_tts_module_guard_passes_real_answers():
    from voice.tts_macos import _is_prompt_leak

    for ok in [
        "It's 2:15 AM, Boss.",
        "Done.",
        "Opening Safari now.",
        "Newton's first law: an object in motion stays in motion.",
    ]:
        assert not _is_prompt_leak(ok), f"False positive on: {ok!r}"


# ─────────────────────────────────────────────────────────────────────
# E. Time intent regex matches every form from the log
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("query", [
    "hey what time is it",
    "what time is it",
    "tell me the time",
    "what's the time",
    "current time",
    "time please",
])
def test_time_intent_regex_matches(query):
    from core.intent_engine.info_intents import _TIME

    assert _TIME.search(query) is not None, (
        f"_TIME regex must match: {query!r} -- this turn is what fell "
        "to the LLM in the last log."
    )


def test_time_intent_regex_is_fast():
    """The whole reason the time intent missed is that it was killed by
    the 50ms watchdog. The regex itself must be far under 1ms so any
    boot-grace window is enough."""
    from core.intent_engine.info_intents import _TIME

    queries = [
        "hey what time is it",
        "what is the current time",
        "tell me the time please",
        "time now",
    ] * 25

    t0 = time.perf_counter()
    for q in queries:
        _TIME.search(q)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    # 100 regex matches well under the 50ms watchdog budget.
    assert elapsed_ms < 50, (
        f"_TIME regex too slow: {elapsed_ms:.2f}ms for 100 matches"
    )


# ─────────────────────────────────────────────────────────────────────
# F. Watchdog boot grace disables intent_engine budget on cold start
# ─────────────────────────────────────────────────────────────────────


def _make_watchdog():
    """Construct RuntimeWatchdog with mocked deps (no event loop)."""
    from core.runtime_watchdog import RuntimeWatchdog

    class _Bus:
        def on(self, *args, **kwargs):
            pass

        def emit(self, *args, **kwargs):
            pass

    class _State:
        pass

    cfg = {
        "performance": {
            "watchdog_intent_timeout_ms": 50,
            "watchdog_intent_boot_grace_s": 8.0,
        },
    }
    return RuntimeWatchdog(_Bus(), _State(), cfg)


def test_watchdog_intent_engine_zero_during_boot_grace():
    """Inside the boot-grace window, intent_engine returns 0.0 (= no
    budget enforced) so cold-boot regex compilation can finish."""
    wd = _make_watchdog()
    # Just constructed -- elapsed is essentially 0.
    assert wd.timeout_s("intent_engine") == 0.0


def test_watchdog_intent_engine_normal_after_grace():
    """After the grace window expires the configured 50 ms budget kicks
    back in."""
    wd = _make_watchdog()
    # Pretend boot was 10 seconds ago (past the 8s grace).
    wd._boot_time_s = time.monotonic() - 10.0
    assert wd.timeout_s("intent_engine") == pytest.approx(0.05, abs=1e-3)


def test_watchdog_grace_does_not_affect_other_stages():
    """Boot grace is intent_engine-specific. LLM/TTS/etc must keep their
    configured budgets even at boot."""
    wd = _make_watchdog()
    assert wd.timeout_s("llm_inference") > 0
    assert wd.timeout_s("tts_synthesis") > 0
    assert wd.timeout_s("tool_execution") > 0


def test_watchdog_grace_disabled_when_zero():
    """Setting watchdog_intent_boot_grace_s=0 disables the feature."""
    from core.runtime_watchdog import RuntimeWatchdog

    class _Bus:
        def on(self, *a, **kw):
            pass

        def emit(self, *a, **kw):
            pass

    cfg = {
        "performance": {
            "watchdog_intent_timeout_ms": 50,
            "watchdog_intent_boot_grace_s": 0.0,
        },
    }
    wd = RuntimeWatchdog(_Bus(), object(), cfg)
    # Even at boot, with grace disabled, the configured budget applies.
    assert wd.timeout_s("intent_engine") == pytest.approx(0.05, abs=1e-3)
