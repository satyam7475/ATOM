"""
ATOM -- Phase G1 regression suite for the Reflective Cognitive Loop.

The loop must:
  * Stay silent on garbage / empty / unparseable LLM output.
  * Honour its 60s cooldown between reflections.
  * Short-circuit when the user starts talking again.
  * Route 'advise'/'clarify' to the bus and 'execute' to the
    execute_emitter callable.
  * Never run while the assistant is still SPEAKING/THINKING.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from core.cognitive.reflective_loop import (
    ReflectionDecision,
    ReflectiveLoop,
    VALID_DECISIONS,
    build_prompt,
    parse_decision,
)


# ── parser ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw, expected_decision, expected_text",
    [
        ('{"decision":"none"}', "none", ""),
        ('{"decision":"advise","intent":"x","text":"Try resting your eyes."}',
         "advise", "Try resting your eyes."),
        ('{"decision":"clarify","intent":"q","text":"Which playlist?"}',
         "clarify", "Which playlist?"),
        ('{"decision":"execute","intent":"music","text":"Pausing now."}',
         "execute", "Pausing now."),
    ],
)
def test_parse_decision_happy_paths(
    raw: str, expected_decision: str, expected_text: str,
) -> None:
    out = parse_decision(raw)
    assert out.decision == expected_decision
    assert out.text == expected_text


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "not json at all",
        "```json\n{not valid}\n```",
        '{"decision":"go_nuclear","text":"hi"}',  # invalid decision
        '{"decision":"advise"}',  # missing text -> none
        '{"decision":"advise","text":""}',  # empty text -> none
        '"just a string"',
    ],
)
def test_parse_decision_falls_back_to_none(raw: str) -> None:
    out = parse_decision(raw)
    assert out.decision == "none"
    assert out.text == ""


def test_parse_decision_strips_markdown_fences() -> None:
    raw = '```json\n{"decision":"advise","intent":"x","text":"Hydrate, Boss."}\n```'
    out = parse_decision(raw)
    assert out.decision == "advise"
    assert out.text == "Hydrate, Boss."


def test_parse_decision_extracts_json_from_prose() -> None:
    raw = "Sure boss! Here you go: " \
        '{"decision":"clarify","intent":"q","text":"Which one?"} thanks'
    out = parse_decision(raw)
    assert out.decision == "clarify"
    assert out.text == "Which one?"


def test_parse_decision_clamps_long_text() -> None:
    long_text = "x" * 1000
    raw = '{"decision":"advise","intent":"x","text":"' + long_text + '"}'
    out = parse_decision(raw)
    assert len(out.text) <= 240


def test_valid_decisions_constant_pinned() -> None:
    assert set(VALID_DECISIONS) == {"none", "advise", "clarify", "execute"}


# ── prompt ─────────────────────────────────────────────────────────


def test_build_prompt_contains_user_and_response_text() -> None:
    from core.cognitive.reflective_loop import _TurnSnapshot
    snap = _TurnSnapshot(
        user_text="play despacito",
        response_text="Playing on Spotify.",
        intent="music_play_specific",
        action="music_play_specific",
    )
    prompt = build_prompt(snap)
    assert "play despacito" in prompt
    assert "Playing on Spotify." in prompt
    assert "JSON" in prompt or "json" in prompt
    assert "decision" in prompt


def test_build_prompt_handles_empty_turn_safely() -> None:
    from core.cognitive.reflective_loop import _TurnSnapshot
    prompt = build_prompt(_TurnSnapshot())
    assert "(empty)" in prompt


# ── loop end-to-end ────────────────────────────────────────────────


class _FakeBus:
    """Minimal AsyncEventBus stub recording emissions and handlers."""

    def __init__(self) -> None:
        self.handlers: dict[str, list] = {}
        self.emitted_long: list[tuple[str, dict]] = []
        self.emitted_fast: list[tuple[str, dict]] = []

    def on(self, event: str, handler: Any) -> None:
        self.handlers.setdefault(event, []).append(handler)

    def off(self, event: str, handler: Any) -> None:
        try:
            self.handlers[event].remove(handler)
        except (KeyError, ValueError):
            pass

    def emit_long(self, event: str, **payload: Any) -> None:
        self.emitted_long.append((event, payload))

    def emit_fast(self, event: str, **payload: Any) -> None:
        self.emitted_fast.append((event, payload))

    async def fire(self, event: str, **payload: Any) -> None:
        for h in list(self.handlers.get(event, ())):
            await h(**payload)


def _make_loop(
    bus: _FakeBus,
    *,
    response: str = '{"decision":"none"}',
    state: str = "idle",
    cooldown_s: float = 60.0,
    response_emitter: Any = None,
    execute_emitter: Any = None,
) -> ReflectiveLoop:
    async def _llm(_prompt: str) -> tuple[str, bool]:
        return response, True

    loop = ReflectiveLoop(
        bus,
        _llm,
        cooldown_s=cooldown_s,
        min_user_chars=3,
        state_provider=lambda: state,
        response_emitter=response_emitter,
        execute_emitter=execute_emitter,
    )
    loop.attach()
    return loop


@pytest.mark.asyncio
async def test_silent_decision_emits_nothing() -> None:
    bus = _FakeBus()
    loop = _make_loop(bus, response='{"decision":"none"}')
    await bus.fire("command_loop_trace", stage="start",
                   text="what time is it", intent="time")
    await bus.fire("response_ready", text="It's 4:46 AM, Boss.")
    await bus.fire("tts_complete")
    assert bus.emitted_long == []
    assert loop.metrics["silent"] == 1
    assert loop.metrics["advise"] == 0


@pytest.mark.asyncio
async def test_advise_routes_to_bus() -> None:
    bus = _FakeBus()
    loop = _make_loop(
        bus,
        response='{"decision":"advise","intent":"hydrate",'
                 '"text":"Drink some water, Boss."}',
    )
    await bus.fire("command_loop_trace", stage="start",
                   text="been coding for 6 hours", intent="chitchat")
    await bus.fire("response_ready", text="Acknowledged.")
    await bus.fire("tts_complete")
    assert len(bus.emitted_long) == 1
    event, payload = bus.emitted_long[0]
    assert event == "response_ready"
    assert payload["text"] == "Drink some water, Boss."
    assert payload["proactive"] is True
    assert loop.metrics["advise"] == 1


@pytest.mark.asyncio
async def test_clarify_routes_to_bus() -> None:
    bus = _FakeBus()
    loop = _make_loop(
        bus,
        response='{"decision":"clarify","intent":"music",'
                 '"text":"Which playlist, Boss?"}',
    )
    await bus.fire("command_loop_trace", stage="start",
                   text="play something", intent="music_play")
    await bus.fire("response_ready", text="Sure.")
    await bus.fire("tts_complete")
    assert len(bus.emitted_long) == 1
    assert bus.emitted_long[0][1]["text"] == "Which playlist, Boss?"
    assert loop.metrics["clarify"] == 1


@pytest.mark.asyncio
async def test_execute_uses_execute_emitter_when_provided() -> None:
    bus = _FakeBus()
    captured: list[str] = []
    loop = _make_loop(
        bus,
        response='{"decision":"execute","intent":"music_pause",'
                 '"text":"Pausing your music."}',
        execute_emitter=captured.append,
    )
    await bus.fire("command_loop_trace", stage="start",
                   text="i need to focus now", intent="focus_on")
    await bus.fire("response_ready", text="Focus on.")
    await bus.fire("tts_complete")
    assert captured == ["Pausing your music."]
    assert bus.emitted_long == []
    assert loop.metrics["execute"] == 1


@pytest.mark.asyncio
async def test_execute_falls_back_to_bus_when_no_emitter() -> None:
    bus = _FakeBus()
    loop = _make_loop(
        bus,
        response='{"decision":"execute","intent":"x","text":"Doing it."}',
        execute_emitter=None,
    )
    await bus.fire("command_loop_trace", stage="start",
                   text="hello atom", intent="chitchat")
    await bus.fire("response_ready", text="Hi.")
    await bus.fire("tts_complete")
    assert len(bus.emitted_long) == 1
    assert bus.emitted_long[0][1]["text"] == "Doing it."


@pytest.mark.asyncio
async def test_cooldown_blocks_back_to_back_reflections() -> None:
    bus = _FakeBus()
    loop = _make_loop(
        bus,
        response='{"decision":"advise","intent":"x","text":"Stretch, Boss."}',
        cooldown_s=300.0,
    )

    async def _one_turn() -> None:
        await bus.fire("command_loop_trace", stage="start",
                       text="been coding for hours", intent="chitchat")
        await bus.fire("response_ready", text="Yes.")
        await bus.fire("tts_complete")

    await _one_turn()
    await _one_turn()
    await _one_turn()
    assert len(bus.emitted_long) == 1
    assert loop.metrics["advise"] == 1


@pytest.mark.asyncio
async def test_skips_when_state_is_speaking() -> None:
    bus = _FakeBus()
    loop = _make_loop(
        bus,
        response='{"decision":"advise","intent":"x","text":"Stretch, Boss."}',
        state="speaking",
    )
    await bus.fire("command_loop_trace", stage="start",
                   text="hi", intent="chitchat")
    await bus.fire("response_ready", text="Hi.")
    await bus.fire("tts_complete")
    assert bus.emitted_long == []
    assert loop.metrics["attempts"] == 0


@pytest.mark.asyncio
async def test_short_user_turn_is_skipped() -> None:
    bus = _FakeBus()
    loop = _make_loop(
        bus,
        response='{"decision":"advise","intent":"x","text":"Stretch."}',
    )
    await bus.fire("command_loop_trace", stage="start", text="hi", intent="x")
    await bus.fire("response_ready", text="Hi.")
    await bus.fire("tts_complete")
    assert bus.emitted_long == []
    assert loop.metrics["attempts"] == 0


@pytest.mark.asyncio
async def test_speech_final_resets_in_flight_guard() -> None:
    """Real-world race: user starts speaking again mid-reflection.

    The loop must let go of the in-flight flag so the *next*
    tts_complete can reflect on the new turn."""
    bus = _FakeBus()
    loop = _make_loop(
        bus,
        response='{"decision":"advise","intent":"x","text":"Tip, Boss."}',
        cooldown_s=0.0,
    )
    loop._in_flight = True
    await bus.fire("speech_final", text="atom what's the time")
    assert loop._in_flight is False


@pytest.mark.asyncio
async def test_llm_failure_does_not_emit() -> None:
    bus = _FakeBus()

    async def _broken_llm(_prompt: str) -> tuple[str, bool]:
        raise RuntimeError("MLX exploded")

    loop = ReflectiveLoop(
        bus, _broken_llm, cooldown_s=0.0, min_user_chars=3,
        state_provider=lambda: "idle",
    )
    loop.attach()

    await bus.fire("command_loop_trace", stage="start",
                   text="long enough sentence", intent="x")
    await bus.fire("response_ready", text="Done.")
    await bus.fire("tts_complete")
    assert bus.emitted_long == []
    assert loop.metrics["attempts"] == 1


@pytest.mark.asyncio
async def test_attach_is_idempotent() -> None:
    bus = _FakeBus()
    loop = _make_loop(bus)
    loop.attach()
    loop.attach()
    assert len(bus.handlers["tts_complete"]) == 1


@pytest.mark.asyncio
async def test_detach_removes_subscribers() -> None:
    bus = _FakeBus()
    loop = _make_loop(bus)
    loop.detach()
    assert bus.handlers.get("tts_complete", []) == []


# ── llm provider helper ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_make_default_llm_provider_forwards_to_mlx() -> None:
    from core.cognitive.reflective_loop import make_default_llm_provider

    mlx = MagicMock()
    captured: dict[str, Any] = {}

    async def _gen(prompt: str, *, model_role: str, max_tokens_override: int):
        captured["prompt"] = prompt
        captured["model_role"] = model_role
        captured["max_tokens"] = max_tokens_override
        return "ok", True

    mlx.generate = _gen
    provider = make_default_llm_provider(mlx, model_role="fast", max_tokens=180)
    out, ok = await provider("hello")
    assert ok is True
    assert out == "ok"
    assert captured["model_role"] == "fast"
    assert captured["max_tokens"] == 180


@pytest.mark.asyncio
async def test_make_default_llm_provider_swallows_exceptions() -> None:
    from core.cognitive.reflective_loop import make_default_llm_provider

    mlx = MagicMock()

    async def _broken(*_a, **_kw):
        raise RuntimeError("boom")

    mlx.generate = _broken
    provider = make_default_llm_provider(mlx)
    out, ok = await provider("hello")
    assert ok is False
    assert out == ""
