"""Wire-up tests: iphone.* events -> identity / proactive / trigger dispatch.

Uses the real :py:class:`core.async_event_bus.AsyncEventBus` so the
priority-queue + async-dispatch path is exercised end-to-end.

Run: ``python3 -m pytest tests/test_bridge_event_wiring.py -v``
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.async_event_bus import AsyncEventBus  # noqa: E402
from core.cross_device.bridge_event_wiring import wire_bridge_events  # noqa: E402
from core.identity_engine import IdentityEngine  # noqa: E402
from core.proactive_awareness import ProactiveAwareness  # noqa: E402


# ────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────

async def _drain_bus(bus: AsyncEventBus, *, limit_s: float = 0.5) -> None:
    """Let the priority worker + active subscriber tasks finish.

    We wait both for the queue to empty and for the WeakSet of handler
    tasks to settle. 500 ms is plenty for synchronous handlers."""
    deadline = time.monotonic() + limit_s
    while time.monotonic() < deadline:
        await asyncio.sleep(0.02)
        q_empty = (bus._queue is None) or bus._queue.empty()  # type: ignore[attr-defined]
        tasks = [t for t in bus._active_tasks if not t.done()]  # type: ignore[attr-defined]
        if q_empty and not tasks:
            return


def _build() -> tuple[AsyncEventBus, IdentityEngine, ProactiveAwareness, list, list]:
    bus = AsyncEventBus()
    ie = IdentityEngine(config={"cross_device": {"faceid_freshness_s": 60}})
    pa = ProactiveAwareness(config={"features": {"proactive_awareness": True}})
    spoken: list[str] = []
    triggered: list[tuple[str, dict]] = []

    async def speak(text: str) -> None:
        spoken.append(text)

    async def on_trigger(name: str, args: dict) -> None:
        triggered.append((name, dict(args)))

    wire_bridge_events(
        bus=bus,
        identity_engine=ie,
        proactive=pa,
        speak=speak,
        on_trigger=on_trigger,
    )
    return bus, ie, pa, spoken, triggered


# ────────────────────────────────────────────
# faceid
# ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_faceid_event_updates_identity_engine() -> None:
    bus, ie, _pa, _spoken, _trig = _build()
    bus.start()
    try:
        bus.emit(
            "iphone.faceid.verified",
            device_id="iphone-boss",
            verified=True,
            timestamp=time.time(),
            label="Boss iPhone",
        )
        await _drain_bus(bus)
        assert ie.is_owner_verified() is True
        info = ie.faceid_freshness_info()
        assert info["label"] == "Boss iPhone"
    finally:
        await bus.stop()


@pytest.mark.asyncio
async def test_faceid_failed_event_does_not_grant_freshness() -> None:
    bus, ie, _pa, _spoken, _trig = _build()
    bus.start()
    try:
        bus.emit(
            "iphone.faceid.verified",
            device_id="iphone-boss",
            verified=False,
            timestamp=time.time(),
        )
        await _drain_bus(bus)
        assert ie.is_owner_verified() is False
    finally:
        await bus.stop()


# ────────────────────────────────────────────
# presence
# ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_presence_event_speaks_hint() -> None:
    bus, _ie, _pa, spoken, _trig = _build()
    bus.start()
    try:
        bus.emit(
            "iphone.presence.changed",
            device_id="iphone-boss",
            state="at_desk",
            timestamp=time.time(),
        )
        await _drain_bus(bus)
        assert len(spoken) == 1
        assert "Boss" in spoken[0]
    finally:
        await bus.stop()


@pytest.mark.asyncio
async def test_presence_unknown_state_speaks_nothing() -> None:
    bus, _ie, _pa, spoken, _trig = _build()
    bus.start()
    try:
        bus.emit("iphone.presence.changed", state="moonwalking", device_id="a")
        await _drain_bus(bus)
        assert spoken == []
    finally:
        await bus.stop()


# ────────────────────────────────────────────
# triggers
# ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_trigger_event_dispatches_and_acks() -> None:
    bus, _ie, _pa, spoken, triggered = _build()
    bus.start()
    try:
        bus.emit(
            "iphone.trigger.fired",
            device_id="iphone-boss",
            name="morning_routine",
            args={"include_weather": False},
        )
        await _drain_bus(bus)
        assert len(triggered) == 1
        name, args = triggered[0]
        assert name == "morning_routine"
        assert args == {"include_weather": False}
        assert len(spoken) == 1
        assert "morning" in spoken[0].lower()
    finally:
        await bus.stop()


@pytest.mark.asyncio
async def test_trigger_empty_name_skipped() -> None:
    bus, _ie, _pa, spoken, triggered = _build()
    bus.start()
    try:
        bus.emit("iphone.trigger.fired", device_id="a", name="", args={})
        await _drain_bus(bus)
        assert triggered == []
        assert spoken == []
    finally:
        await bus.stop()


# ────────────────────────────────────────────
# Robustness
# ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_wiring_with_none_speak_still_updates_state() -> None:
    """speak=None must not crash the wiring -- hint is swallowed silently."""
    bus = AsyncEventBus()
    ie = IdentityEngine(config={"cross_device": {"faceid_freshness_s": 60}})
    pa = ProactiveAwareness(config={"features": {"proactive_awareness": True}})
    wire_bridge_events(bus=bus, identity_engine=ie, proactive=pa, speak=None)
    bus.start()
    try:
        bus.emit("iphone.presence.changed", device_id="a", state="at_desk")
        await _drain_bus(bus)
        # pa internal state should still have advanced even though
        # nobody was listening for the hint.
        assert pa.handle_iphone_presence("at_desk") is None
    finally:
        await bus.stop()


@pytest.mark.asyncio
async def test_wiring_with_none_bus_is_noop() -> None:
    """wire_bridge_events(bus=None) must log + return without raising."""
    wire_bridge_events(
        bus=None,
        identity_engine=None,
        proactive=None,
    )


@pytest.mark.asyncio
async def test_faulty_speak_fn_does_not_kill_subscriber() -> None:
    bus = AsyncEventBus()
    pa = ProactiveAwareness(config={"features": {"proactive_awareness": True}})
    ie = IdentityEngine(config={})
    calls: list[str] = []

    async def boom(text: str) -> None:
        calls.append(text)
        raise RuntimeError("tts is unhappy")

    wire_bridge_events(
        bus=bus,
        identity_engine=ie,
        proactive=pa,
        speak=boom,
    )
    bus.start()
    try:
        bus.emit("iphone.presence.changed", device_id="a", state="at_desk")
        await _drain_bus(bus)
        assert calls, "speak fn was still invoked"
        # Second event must still process despite prior exception.
        bus.emit("iphone.presence.changed", device_id="a", state="leaving")
        await _drain_bus(bus)
        assert len(calls) == 2
    finally:
        await bus.stop()
