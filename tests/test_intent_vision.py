"""ATOM -- regression suite for ``core.intent_engine.vision_intents`` (C2).

Pins five behaviours so the live-log defects (atomLogs.txt L419, L453)
do not regress:

1. "Can you see me" / "look at me" / "describe me" route to
   ``vision_describe`` (camera + on-device VLM) and never fall through
   to the LLM.
2. "What do you see" routes to ``vision_look`` (face count, fastest
   path on the Neural Engine) so casual glances stay sub-300 ms.
3. "Describe my screen" / "what am I doing" / "analyze the screen"
   route to ``screen_describe`` (screenshot + VLM/Gemini) -- the new
   action wired into the router below.
4. Phrases that look musical or systemy ("play music", "what time is
   it") still route to their existing intents; vision_intents must
   not steal the room.
5. ``IntentEngine`` runs vision intents *after* music/system but
   *before* ``cognitive_intents`` so the fast camera/screen paths win
   over LLM fallback.
"""

from __future__ import annotations

import pytest

from core.intent_engine import IntentEngine
from core.intent_engine import vision_intents


# ── camera-facing ("see me") patterns ────────────────────────────


@pytest.mark.parametrize(
    "phrase",
    [
        "can you see me",
        "Can you see me right now",
        "see me",
        "look at me",
        "watch me",
        "describe me",
        "atom can you see me",
        "hey atom, can you see me",
        "do you see me",
        "am I visible",
        "am I in the frame",
        "check on me",
        "could you look at me please",
    ],
)
def test_see_me_routes_to_vision_describe(phrase: str) -> None:
    result = vision_intents.check(phrase)
    assert result is not None, f"vision_intents missed: {phrase!r}"
    assert result.intent == "vision_describe"
    assert result.action == "vision_describe"
    assert result.confidence >= 0.9


@pytest.mark.parametrize(
    "phrase",
    [
        "what do you see",
        "what can you see",
        "what's around",
        "what is in front",
        "look around",
        "glance",
        "what's in view right now",
        "atom what do you see",
    ],
)
def test_glance_routes_to_vision_look(phrase: str) -> None:
    result = vision_intents.check(phrase)
    assert result is not None, f"vision_intents missed: {phrase!r}"
    assert result.intent == "vision_look"
    assert result.action == "vision_look"
    assert result.action_args == {"focus": "general"}


# ── screen-facing patterns ───────────────────────────────────────


@pytest.mark.parametrize(
    "phrase",
    [
        "describe my screen",
        "describe the screen",
        "analyze my screen",
        "analyse my screen",
        "read my screen",
        "check my screen",
        "look at my screen",
        "inspect my screen",
        "scan my screen",
        "what's on my screen",
        "what is on my screen right now",
        "what is happening on my display",
        "what am I doing",
        "what am I working on",
        "what am I looking at",
        "what do you see on my screen",
        "atom describe the screen",
        "hey atom, what's on my display",
    ],
)
def test_screen_phrases_route_to_screen_describe(phrase: str) -> None:
    result = vision_intents.check(phrase)
    assert result is not None, f"vision_intents missed: {phrase!r}"
    assert result.intent == "screen_describe"
    assert result.action == "screen_describe"
    assert "query" in (result.action_args or {})


# ── negative cases (must not steal other intents) ────────────────


@pytest.mark.parametrize(
    "phrase",
    [
        "play music",
        "what time is it",
        "open chrome",
        "what's the weather",
        "what is my battery",
    ],
)
def test_unrelated_phrases_are_not_caught(phrase: str) -> None:
    assert vision_intents.check(phrase) is None


# ── full IntentEngine integration ────────────────────────────────


def test_engine_classifies_see_me_as_vision_describe() -> None:
    engine = IntentEngine()
    result = engine.classify("can you see me")
    assert result.action == "vision_describe"
    assert result.intent == "vision_describe"


def test_engine_classifies_screen_phrase_as_screen_describe() -> None:
    engine = IntentEngine()
    result = engine.classify("what's on my screen")
    assert result.action == "screen_describe"
    assert result.intent == "screen_describe"


def test_engine_classifies_glance_as_vision_look() -> None:
    engine = IntentEngine()
    result = engine.classify("what do you see")
    assert result.action == "vision_look"


def test_engine_does_not_mistake_play_music_for_vision() -> None:
    engine = IntentEngine()
    result = engine.classify("play some lofi music")
    assert result.intent != "vision_describe"
    assert result.intent != "screen_describe"


def test_quick_match_returns_intent_names() -> None:
    assert vision_intents.quick_match("can you see me") == "vision_describe"
    assert vision_intents.quick_match("what do you see") == "vision_look"
    assert vision_intents.quick_match("describe my screen") == "screen_describe"
    assert vision_intents.quick_match("xyz nonsense") is None
