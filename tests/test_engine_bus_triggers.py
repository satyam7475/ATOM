"""Tests for v3.7 bus-driven proactivity.

Goal engine and proactive engine used to wait on multi-minute timers
to re-evaluate. v3.7 hooks ``context_snapshot`` (emitted ~every minute
by HealthMonitor) so they react in near-real-time. These tests verify:

  * subscribe / unsubscribe lifecycle is symmetric (no event leaks)
  * the snapshot handler actually triggers the inner work
  * the per-engine snapshot interval debounces a flood of events
  * ``changed_app=True`` on system_state_update bypasses the
    proactive snapshot debounce (live workflow trigger latency)

Note: both engines use ``__slots__`` and call ``asyncio.create_task``
in ``start()``, so we drive lifecycle through ``asyncio.run`` and use
class-level monkeypatching where we need to spy on bound methods.

Run: python3 -m pytest tests/test_engine_bus_triggers.py -q
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest


def _fake_bus():
    bus = MagicMock()
    bus.on = MagicMock()
    bus.off = MagicMock()
    bus.emit_fast = MagicMock()
    bus.emit_long = MagicMock()
    return bus


# ── Goal engine ──────────────────────────────────────────────────────


def test_goal_engine_subscribes_to_context_snapshot():
    from core.cognitive.goal_engine import GoalEngine

    bus = _fake_bus()

    async def _flow():
        eng = GoalEngine(bus, MagicMock(), {"cognitive": {"goals_enabled": True}})
        eng.start()
        await asyncio.sleep(0)  # let the create_task settle
        subs = {call.args[0] for call in bus.on.call_args_list}
        eng.stop()
        await asyncio.sleep(0)
        return subs

    subscribed = asyncio.run(_flow())
    assert "context_snapshot" in subscribed, (
        "goal_engine must subscribe to context_snapshot in v3.7+"
    )
    assert "tool_executed" in subscribed


def test_goal_engine_unsubscribes_on_stop():
    from core.cognitive.goal_engine import GoalEngine

    bus = _fake_bus()

    async def _flow():
        eng = GoalEngine(bus, MagicMock(), {"cognitive": {"goals_enabled": True}})
        eng.start()
        await asyncio.sleep(0)
        eng.stop()
        await asyncio.sleep(0)
        return {call.args[0] for call in bus.off.call_args_list}

    unsubscribed = asyncio.run(_flow())
    assert "context_snapshot" in unsubscribed
    assert "tool_executed" in unsubscribed


def test_goal_engine_snapshot_handler_calls_evaluate():
    """Side-effect witness: ``_evaluate_goals`` mutates each active
    goal's ``evaluation`` dict in-place. We populate one goal so the
    method has something observable to do."""
    from core.cognitive.goal_engine import GoalEngine

    bus = _fake_bus()
    eng = GoalEngine(
        bus, MagicMock(),
        {"cognitive": {
            "goals_enabled": True,
            "goal_snapshot_eval_interval_s": 0.0,  # no debounce
        }},
    )
    eng._goals = [{
        "id": "g1", "title": "ship v3.7", "status": "active",
        "steps": [{
            "id": "s1", "title": "code change", "status": "pending",
            "minutes_logged": 0, "created_at": "", "updated_at": "",
        }],
        "progress": 0.0, "created_at": "", "updated_at": "",
        "evaluation": {}, "streak_days": 0, "last_progress_date": "",
        "total_minutes": 0,
    }]

    asyncio.run(eng._on_context_snapshot(active_app="cursor"))

    assert eng._goals[0]["evaluation"], (
        "snapshot handler did not run _evaluate_goals "
        "(evaluation dict still empty)"
    )
    # _last_snapshot_eval is the canonical witness for the handler firing.
    assert eng._last_snapshot_eval > 0.0


def test_goal_engine_snapshot_debounces_floods():
    """A burst of snapshots within the interval must collapse to one eval."""
    from core.cognitive.goal_engine import GoalEngine

    bus = _fake_bus()
    eng = GoalEngine(
        bus, MagicMock(),
        {"cognitive": {
            "goals_enabled": True,
            "goal_snapshot_eval_interval_s": 60.0,
        }},
    )

    async def _burst():
        for _ in range(10):
            await eng._on_context_snapshot()
            await asyncio.sleep(0)

    first_ts_capture = []
    original = eng.__class__._on_context_snapshot

    async def _spy(self, **kw):
        before = self._last_snapshot_eval
        await original(self, **kw)
        if self._last_snapshot_eval != before:
            first_ts_capture.append(self._last_snapshot_eval)

    eng.__class__._on_context_snapshot = _spy
    try:
        asyncio.run(_burst())
    finally:
        eng.__class__._on_context_snapshot = original

    assert len(first_ts_capture) == 1, (
        f"snapshot debounce broken: expected 1 eval, "
        f"got {len(first_ts_capture)}"
    )


# ── Proactive engine ─────────────────────────────────────────────────


def test_proactive_engine_subscribes_to_context_snapshot():
    from core.cognitive.proactive_engine import ProactiveIntelligenceEngine

    bus = _fake_bus()

    async def _flow():
        eng = ProactiveIntelligenceEngine(bus, {"proactive_engine": {}})
        eng.start()
        await asyncio.sleep(0)
        subs = {call.args[0] for call in bus.on.call_args_list}
        eng.stop()
        await asyncio.sleep(0)
        return subs

    subscribed = asyncio.run(_flow())
    assert "context_snapshot" in subscribed
    assert "system_state_update" in subscribed
    assert "action_executed" in subscribed


def test_proactive_engine_unsubscribes_on_stop():
    from core.cognitive.proactive_engine import ProactiveIntelligenceEngine

    bus = _fake_bus()

    async def _flow():
        eng = ProactiveIntelligenceEngine(bus, {"proactive_engine": {}})
        eng.start()
        await asyncio.sleep(0)
        eng.stop()
        await asyncio.sleep(0)
        return {call.args[0] for call in bus.off.call_args_list}

    unsubscribed = asyncio.run(_flow())
    for evt in (
        "action_executed", "system_light_scan", "idle_detected",
        "fs_event", "system_state_update", "context_snapshot",
    ):
        assert evt in unsubscribed, f"missing off({evt!r}) on stop()"


def test_proactive_snapshot_handler_runs_scan():
    """``_last_snapshot_scan`` is the witness -- starts at 0, set to
    monotonic time on each accepted snapshot."""
    from core.cognitive.proactive_engine import ProactiveIntelligenceEngine

    bus = _fake_bus()
    eng = ProactiveIntelligenceEngine(
        bus,
        {"proactive_engine": {"snapshot_scan_interval_s": 0.0}},
    )
    assert eng._last_snapshot_scan == 0.0

    asyncio.run(eng._on_context_snapshot(active_app="terminal"))
    assert eng._last_snapshot_scan > 0.0


def test_proactive_snapshot_debounces_floods():
    from core.cognitive.proactive_engine import ProactiveIntelligenceEngine

    bus = _fake_bus()
    eng = ProactiveIntelligenceEngine(
        bus,
        {"proactive_engine": {"snapshot_scan_interval_s": 60.0}},
    )

    async def _burst():
        for _ in range(8):
            await eng._on_context_snapshot()
            await asyncio.sleep(0)

    asyncio.run(_burst())
    # Even after 8 snapshot events, the scan timestamp should only
    # have been written once (at the very first one).
    # We can't easily count without monkeypatching slots, but we can
    # assert that bus.emit_long (which scan uses to publish insights)
    # was called at most once if any insight was produced.
    emit_long_calls = [
        c for c in bus.emit_long.call_args_list
        if c.args and c.args[0] == "jarvis_insight"
    ]
    assert len(emit_long_calls) <= 1, (
        f"snapshot debounce broken: emit_long(jarvis_insight) "
        f"called {len(emit_long_calls)} times for 8 snapshots"
    )


def test_proactive_app_focus_change_bypasses_debounce():
    """system_state_update with changed_app=True must let the next
    context_snapshot through immediately, even within the 60s window."""
    from core.cognitive.proactive_engine import ProactiveIntelligenceEngine

    bus = _fake_bus()
    eng = ProactiveIntelligenceEngine(
        bus,
        {"proactive_engine": {"snapshot_scan_interval_s": 600.0}},
    )

    async def _flow():
        await eng._on_context_snapshot()
        first_ts = eng._last_snapshot_scan
        assert first_ts > 0.0

        await eng._on_context_snapshot()
        assert eng._last_snapshot_scan == first_ts, (
            "second snapshot in 600s window should be debounced"
        )

        await eng._on_system_state_update(
            snapshot={"active_app": "code"}, changed_app=True,
        )
        # changed_app resets the debounce floor, so next snapshot fires
        await eng._on_context_snapshot()
        assert eng._last_snapshot_scan != first_ts, (
            "changed_app=True should let the next snapshot through"
        )

    asyncio.run(_flow())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
