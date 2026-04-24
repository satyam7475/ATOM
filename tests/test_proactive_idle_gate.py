"""Regression test for ProactiveIntelligenceEngine idle gate.

Without this gate, "Boss, new file landed: X" arrives mid-LLM-turn and
plays as the answer to the user's actual question (atom_log.txt L308,
L584, L619). The gate must buffer insights while the owner is mid-turn
and re-emit them on the next listening transition.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core.cognitive.proactive_engine import ProactiveIntelligenceEngine


class _FakeBus:
    def __init__(self) -> None:
        self.long_emits: list[tuple[str, dict[str, Any]]] = []

    def emit_long(self, event: str, **kwargs) -> None:
        self.long_emits.append((event, dict(kwargs)))

    def on(self, *args, **kwargs) -> None:
        pass


class _FakeState:
    def __init__(self, current: str = "listening") -> None:
        self.current = current


class _FakeCommandLoop:
    def __init__(self, busy: bool = False) -> None:
        self._busy = busy

    def is_busy(self) -> bool:
        return self._busy


@pytest.fixture
def engine_with_gate():
    bus = _FakeBus()
    engine = ProactiveIntelligenceEngine(bus=bus, config={"proactive_engine": {}})
    state = _FakeState(current="listening")
    cl = _FakeCommandLoop(busy=False)
    engine.attach_idle_gate(state, cl)
    return engine, bus, state, cl


def _insight(name: str = "test_event", category: str = "context_download") -> dict[str, Any]:
    return {
        "message": f"Boss, {name}.",
        "category": category,
        "priority": 4,
        "source": "proactive_fs",
    }


def test_emit_passes_through_when_idle(engine_with_gate):
    engine, bus, _state, _cl = engine_with_gate
    engine._emit_insight(_insight("file landed: foo.pdf"))
    assert len(bus.long_emits) == 1
    assert bus.long_emits[0][0] == "jarvis_insight"


def test_emit_buffers_when_state_thinking(engine_with_gate):
    engine, bus, state, _cl = engine_with_gate
    state.current = "thinking"
    engine._emit_insight(_insight("file landed: bar.pdf"))
    assert len(bus.long_emits) == 0
    assert len(engine._pending_insights) == 1


def test_emit_buffers_when_state_speaking(engine_with_gate):
    engine, bus, state, _cl = engine_with_gate
    state.current = "speaking"
    engine._emit_insight(_insight("file landed: baz.pdf"))
    assert len(bus.long_emits) == 0
    assert len(engine._pending_insights) == 1


def test_emit_buffers_when_command_loop_busy(engine_with_gate):
    engine, bus, _state, cl = engine_with_gate
    cl._busy = True
    engine._emit_insight(_insight("file landed: qux.pdf"))
    assert len(bus.long_emits) == 0
    assert len(engine._pending_insights) == 1


def test_drain_pending_only_when_idle(engine_with_gate):
    engine, bus, state, _cl = engine_with_gate
    state.current = "thinking"
    for i in range(2):
        engine._emit_insight(_insight(f"file {i}"))
    assert len(engine._pending_insights) == 2

    drained = engine.drain_pending()
    assert drained == 0, "must not drain while still busy"

    state.current = "listening"
    drained = engine.drain_pending()
    assert drained == 2
    assert len(engine._pending_insights) == 0
    assert len(bus.long_emits) == 2


def test_buffer_caps_at_max_pending(engine_with_gate):
    engine, bus, state, _cl = engine_with_gate
    state.current = "thinking"
    for i in range(10):
        engine._emit_insight(_insight(f"file {i}"))
    assert len(engine._pending_insights) == engine._max_pending == 3


def test_no_state_provider_means_always_idle():
    """If no state provider is wired (legacy callers), default to
    immediate emit so we don't silently break anyone."""
    bus = _FakeBus()
    engine = ProactiveIntelligenceEngine(bus=bus, config={"proactive_engine": {}})
    engine._emit_insight(_insight("legacy"))
    assert len(bus.long_emits) == 1
