"""
ATOM -- regression suite for ``core.intent_engine.focus_intents`` (F3).

Pins:
  * Focus phrases route to ``focus_on``/``focus_off``/``focus_state``
    intents *before* the legacy "go silent" handler in ``meta_intents``
    eats them.
  * Duration suffixes ("for 30 minutes", "for an hour") populate
    ``duration_minutes`` correctly.
  * Negative cases (mute, stop music) do NOT touch focus.
"""

from __future__ import annotations

import pytest

from core.intent_engine import IntentEngine
from core.intent_engine import focus_intents


# ── focus_on ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "phrase",
    [
        "focus mode on",
        "turn on focus",
        "turn on do not disturb",
        "enable dnd",
        "start focus",
        "start deep work mode",
        "do not disturb",
        "deep work mode",
        "i'm going into deep work",
        "silence my notifications",
        "hold my notifications",
    ],
)
def test_focus_on_phrases_resolve(phrase: str) -> None:
    result = focus_intents.check(phrase)
    assert result is not None, phrase
    assert result.intent == "focus_on"
    assert result.action == "focus_on"


# ── focus_off ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "phrase",
    [
        "focus mode off",
        "turn off focus",
        "turn off do not disturb",
        "disable dnd",
        "end focus mode",
        "end deep work",
        "exit focus",
        "exit deep work",
        "do not disturb off",
        "unmute notifications",
        "resume notifications",
        "i'm done with deep work",
    ],
)
def test_focus_off_phrases_resolve(phrase: str) -> None:
    result = focus_intents.check(phrase)
    assert result is not None, phrase
    assert result.intent == "focus_off"


# ── focus_state ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "phrase",
    [
        "is focus on",
        "is do not disturb on",
        "is dnd on",
        "is focus mode enabled",
        "am i in focus mode",
        "what's my focus state",
        "focus status",
    ],
)
def test_focus_state_phrases_resolve(phrase: str) -> None:
    result = focus_intents.check(phrase)
    assert result is not None, phrase
    assert result.intent == "focus_state"


# ── duration extraction ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "phrase, expected_minutes",
    [
        ("focus mode on for 30 minutes", 30),
        ("turn on focus for 45 mins", 45),
        ("do not disturb for 2 hours", 120),
        ("deep work for an hour", 60),
        ("focus on for a minute", 1),
    ],
)
def test_duration_is_extracted(phrase: str, expected_minutes: int) -> None:
    result = focus_intents.check(phrase)
    assert result is not None, phrase
    assert result.intent == "focus_on"
    assert result.action_args is not None
    assert result.action_args.get("duration_minutes") == expected_minutes


def test_focus_on_without_duration_omits_arg() -> None:
    result = focus_intents.check("focus mode on")
    assert result is not None
    assert result.action_args is not None
    assert "duration_minutes" not in result.action_args


# ── precedence inside the engine ────────────────────────────────────


def test_engine_routes_focus_mode_to_focus_on_not_meta_silent() -> None:
    """Before F3, "focus mode" expanded to "go silent" which only put
    *ATOM* on mute. Now it must flip macOS DND."""
    engine = IntentEngine()
    out = engine.classify("focus mode on")
    assert out.intent == "focus_on"


def test_engine_routes_do_not_disturb_to_focus_on() -> None:
    engine = IntentEngine()
    out = engine.classify("do not disturb")
    assert out.intent == "focus_on"


def test_engine_keeps_mute_intent_intact() -> None:
    """Plain "mute system" is volume-mute, not focus-mode."""
    engine = IntentEngine()
    out = engine.classify("mute system")
    assert out.intent == "mute"


def test_engine_keeps_pause_music_intact() -> None:
    engine = IntentEngine()
    out = engine.classify("pause music")
    assert out.intent == "music_pause"


def test_engine_quick_match_returns_focus_intents() -> None:
    engine = IntentEngine()
    assert engine.quick_match("focus mode on") == "focus_on"
    assert engine.quick_match("turn off focus") == "focus_off"
    assert engine.quick_match("is focus on") == "focus_state"


# ── safety: no false positives ──────────────────────────────────────


@pytest.mark.parametrize(
    "phrase",
    [
        "what time is it",
        "open chrome",
        "play despacito on spotify",
        "lock the screen",
        "how do I focus better",
    ],
)
def test_non_focus_phrases_do_not_match(phrase: str) -> None:
    assert focus_intents.check(phrase) is None
