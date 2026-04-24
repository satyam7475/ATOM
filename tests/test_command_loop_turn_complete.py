"""Regression suite for the Phase G6 ``turn_complete`` wiring.

The CommandLoop is the single owner of the per-turn lifecycle. With
the G6 wiring in place it must:

  * latch the most recent assistant utterance (``response_ready``);
  * emit ``turn_complete`` once per fully-flushed turn after
    ``tts_complete``;
  * skip the emission when the lock is busy (back-to-back turns);
  * abandon the pending turn the moment a new ``speech_final``
    arrives -- the *short-circuit guard* the reflective loop relies on.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core.command_loop import CommandLoop


# ── fakes ──────────────────────────────────────────────────────────


class _FakeBus:
    def __init__(self) -> None:
        self.subs: dict[str, list] = {}
        self.fast_emits: list[tuple[str, dict[str, Any]]] = []
        self.long_emits: list[tuple[str, dict[str, Any]]] = []

    def on(self, event: str, handler) -> None:
        self.subs.setdefault(event, []).append(handler)

    def off(self, event: str, handler) -> None:
        if event in self.subs and handler in self.subs[event]:
            self.subs[event].remove(handler)

    def emit_fast(self, event: str, **data: Any) -> None:
        self.fast_emits.append((event, data))

    def emit_long(self, event: str, **data: Any) -> None:
        self.long_emits.append((event, data))

    async def fire(self, event: str, **data: Any) -> None:
        for fn in list(self.subs.get(event, [])):
            await fn(**data)


class _FakeState:
    def __init__(self) -> None:
        self.current = None


class _FakeRouter:
    async def on_speech(self, *_a: Any, **_kw: Any) -> None:
        return None


def _make_loop() -> tuple[CommandLoop, _FakeBus]:
    bus = _FakeBus()
    loop = CommandLoop(bus, _FakeState(), _FakeRouter())
    loop.attach_turn_emitter()
    return loop, bus


# ── attach idempotency ────────────────────────────────────────────


def test_attach_turn_emitter_is_idempotent() -> None:
    loop, bus = _make_loop()
    loop.attach_turn_emitter()
    loop.attach_turn_emitter()
    assert len(bus.subs.get("response_ready", [])) == 1
    assert len(bus.subs.get("tts_complete", [])) == 1
    assert len(bus.subs.get("speech_final", [])) == 1


# ── happy path ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_turn_complete_emitted_after_tts_complete() -> None:
    loop, bus = _make_loop()
    loop._pending_turn = {  # type: ignore[attr-defined]
        "trace_id": "abc123",
        "user_text": "what's the weather",
        "elapsed_ms": 412.0,
        "ts": 0.0,
    }
    await bus.fire("response_ready", text="Sunny and warm, Boss.")
    await bus.fire("tts_complete")

    turn_events = [e for e in bus.fast_emits if e[0] == "turn_complete"]
    assert len(turn_events) == 1
    payload = turn_events[0][1]
    assert payload["trace_id"] == "abc123"
    assert payload["user_text"] == "what's the weather"
    assert payload["response_text"] == "Sunny and warm, Boss."
    assert payload["elapsed_ms"] == 412.0
    assert loop.get_diagnostics()["turn_complete_count"] == 1


@pytest.mark.asyncio
async def test_turn_complete_skipped_when_no_pending_turn() -> None:
    loop, bus = _make_loop()
    await bus.fire("tts_complete")
    assert all(e[0] != "turn_complete" for e in bus.fast_emits)


# ── short-circuit guard ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_speech_final_short_circuits_pending_turn() -> None:
    loop, bus = _make_loop()
    loop._pending_turn = {  # type: ignore[attr-defined]
        "trace_id": "stale", "user_text": "old turn",
        "elapsed_ms": 0.0, "ts": 0.0,
    }
    await bus.fire("speech_final", text="new question right after")
    await bus.fire("tts_complete")
    assert all(e[0] != "turn_complete" for e in bus.fast_emits)
    assert loop.get_diagnostics()["turn_complete_count"] == 0


@pytest.mark.asyncio
async def test_empty_speech_final_does_not_clear_pending() -> None:
    loop, bus = _make_loop()
    loop._pending_turn = {  # type: ignore[attr-defined]
        "trace_id": "keep", "user_text": "keep me",
        "elapsed_ms": 1.0, "ts": 0.0,
    }
    await bus.fire("speech_final", text="")
    await bus.fire("tts_complete")
    assert any(e[0] == "turn_complete" for e in bus.fast_emits)


# ── busy gating ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_turn_complete_skipped_while_lock_busy() -> None:
    loop, bus = _make_loop()
    loop._pending_turn = {  # type: ignore[attr-defined]
        "trace_id": "busy", "user_text": "x",
        "elapsed_ms": 0.0, "ts": 0.0,
    }
    acquired = await loop.execution_lock.acquire(command="busy", timeout_s=1.0)
    try:
        assert acquired is True
        await bus.fire("tts_complete")
        assert all(e[0] != "turn_complete" for e in bus.fast_emits)
    finally:
        loop.execution_lock.release()


# ── response latch ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_response_text_latch_uses_latest_response() -> None:
    loop, bus = _make_loop()
    loop._pending_turn = {  # type: ignore[attr-defined]
        "trace_id": "latch", "user_text": "ask",
        "elapsed_ms": 1.0, "ts": 0.0,
    }
    await bus.fire("response_ready", text="first draft")
    await bus.fire("response_ready", text="final answer")
    await bus.fire("tts_complete")
    payload = next(e[1] for e in bus.fast_emits if e[0] == "turn_complete")
    assert payload["response_text"] == "final answer"


@pytest.mark.asyncio
async def test_blank_response_does_not_overwrite_latch() -> None:
    loop, bus = _make_loop()
    loop._pending_turn = {  # type: ignore[attr-defined]
        "trace_id": "latch2", "user_text": "ask",
        "elapsed_ms": 1.0, "ts": 0.0,
    }
    await bus.fire("response_ready", text="real answer")
    await bus.fire("response_ready", text="")
    await bus.fire("tts_complete")
    payload = next(e[1] for e in bus.fast_emits if e[0] == "turn_complete")
    assert payload["response_text"] == "real answer"


# ── single-fire guarantee ────────────────────────────────────────


@pytest.mark.asyncio
async def test_turn_complete_fires_only_once_per_turn() -> None:
    loop, bus = _make_loop()
    loop._pending_turn = {  # type: ignore[attr-defined]
        "trace_id": "once", "user_text": "x",
        "elapsed_ms": 1.0, "ts": 0.0,
    }
    await bus.fire("tts_complete")
    await bus.fire("tts_complete")
    fires = [e for e in bus.fast_emits if e[0] == "turn_complete"]
    assert len(fires) == 1


# ── diagnostics counter ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_turn_complete_count_advances() -> None:
    loop, bus = _make_loop()
    for i in range(3):
        loop._pending_turn = {  # type: ignore[attr-defined]
            "trace_id": f"t{i}", "user_text": f"q{i}",
            "elapsed_ms": 1.0, "ts": 0.0,
        }
        await bus.fire("tts_complete")
    assert loop.get_diagnostics()["turn_complete_count"] == 3
