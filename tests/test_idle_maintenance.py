"""Tests for IdleMaintenance (Sprint Ω.12, Apr 27 2026).

Drives :class:`core.idle_maintenance.IdleMaintenance` with a synthetic
event bus + monkeypatched monotonic clock so the freeze, idle-tick,
cooldown, and busy-skip behaviours are deterministic without relying
on real wall clock or a live MLX runtime.
"""

from __future__ import annotations

import gc
import os
import sys
import time
import unittest
from typing import Any
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.idle_maintenance import IdleMaintenance


# ──────────────────────────────────────────────────────────────────────
# Test doubles
# ──────────────────────────────────────────────────────────────────────


class FakeBus:
    """Captures ``on``/``off`` subscriptions and lets tests fire events."""

    def __init__(self) -> None:
        self.subs: dict[str, list] = {}
        self.unsubs: list[tuple[str, object]] = []

    def on(self, event: str, handler) -> None:
        self.subs.setdefault(event, []).append(handler)

    def off(self, event: str, handler) -> None:
        self.unsubs.append((event, handler))
        try:
            self.subs[event].remove(handler)
        except (KeyError, ValueError):
            pass

    def fire(self, event: str, **payload) -> None:
        for h in list(self.subs.get(event, [])):
            h(**payload)


class _ClockSpy:
    """Tiny callable that returns a controllable monotonic clock value
    plus tracks how many MLX clears were invoked.
    """

    def __init__(self) -> None:
        self.now = 1_000.0  # arbitrary non-zero start
        self.mlx_clears = 0

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def __call__(self) -> float:
        return self.now

    def clear_metal(self) -> None:
        self.mlx_clears += 1


def _config(**overrides) -> dict:
    base = {
        "enabled": True,
        "freeze_after_boot_s": 60,
        "idle_threshold_s": 120,
        "tick_interval_s": 30,
        "min_action_interval_s": 60,
        "gc_threshold_gen0": 2000,
        "gc_threshold_gen1": 25,
        "gc_threshold_gen2": 25,
        "clear_mlx_cache_on_idle": True,
        "log_actions": False,  # keep test output quiet
    }
    base.update(overrides)
    return {"idle_maintenance": base}


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────


class IdleMaintenanceLifecycleTest(unittest.TestCase):
    """``start()`` is idempotent and applies gc thresholds; ``stop()``
    unsubscribes from the bus."""

    def test_start_applies_gc_thresholds_and_subscribes(self) -> None:
        prior = gc.get_threshold()
        try:
            bus = FakeBus()
            idle = IdleMaintenance(config=_config(), bus=bus)
            idle.start()
            self.assertTrue(idle._started)
            self.assertEqual(gc.get_threshold(), (2000, 25, 25))
            # subscribed to all activity events
            for event in (
                "speech_final", "partial_response", "response_ready",
                "turn_started", "turn_complete",
            ):
                self.assertIn(event, bus.subs)
                self.assertEqual(len(bus.subs[event]), 1)
        finally:
            gc.set_threshold(*prior)

    def test_start_is_idempotent(self) -> None:
        prior = gc.get_threshold()
        try:
            bus = FakeBus()
            idle = IdleMaintenance(config=_config(), bus=bus)
            idle.start()
            idle.start()
            for event in ("speech_final", "response_ready"):
                self.assertEqual(len(bus.subs[event]), 1)
        finally:
            gc.set_threshold(*prior)

    def test_disabled_config_short_circuits(self) -> None:
        bus = FakeBus()
        idle = IdleMaintenance(
            config=_config(enabled=False), bus=bus,
        )
        idle.start()
        # No bus subscriptions, no actions taken.
        self.assertEqual(bus.subs, {})
        self.assertEqual(idle.maybe_tick().get("action"), "skip")

    def test_stop_unsubscribes(self) -> None:
        prior = gc.get_threshold()
        try:
            bus = FakeBus()
            idle = IdleMaintenance(config=_config(), bus=bus)
            idle.start()
            idle.stop()
            # 5 events × 1 handler each = 5 unsubscribes.
            self.assertEqual(len(bus.unsubs), 5)
            for event, _h in bus.unsubs:
                self.assertEqual(bus.subs.get(event, []), [])
        finally:
            gc.set_threshold(*prior)

    def test_invalid_gc_thresholds_fall_back_to_defaults(self) -> None:
        prior = gc.get_threshold()
        try:
            idle = IdleMaintenance(
                config=_config(gc_threshold_gen0=0),
            )
            idle.start()
            # Bad value -> falls back to defaults (2000, 25, 25).
            self.assertEqual(gc.get_threshold(), (2000, 25, 25))
        finally:
            gc.set_threshold(*prior)


class IdleMaintenanceActivityTest(unittest.TestCase):
    """Activity events from the bus should reset the idle clock."""

    def test_bus_event_resets_idle_clock(self) -> None:
        clock = _ClockSpy()
        with patch("time.monotonic", clock):
            bus = FakeBus()
            idle = IdleMaintenance(config=_config(), bus=bus)
            idle.start()
            clock.advance(500.0)
            bus.fire("speech_final", text="hello")
            diag = idle.diagnostics()
            # idle_s reset to ~0 because activity just fired.
            self.assertLess(diag["idle_s"], 1.0)


class IdleMaintenanceTickTest(unittest.TestCase):
    """``maybe_tick()`` decision tree: idle threshold, cooldown, busy."""

    def test_tick_skips_when_not_idle(self) -> None:
        clock = _ClockSpy()
        with patch("time.monotonic", clock):
            bus = FakeBus()
            idle = IdleMaintenance(config=_config(), bus=bus)
            idle.start()
            clock.advance(60.0)  # below idle_threshold_s=120
            result = idle.maybe_tick()
            self.assertEqual(result["action"], "skip")
            self.assertEqual(result["reason"], "not_idle")

    def test_tick_runs_after_idle_threshold(self) -> None:
        clock = _ClockSpy()
        with patch("time.monotonic", clock):
            bus = FakeBus()
            idle = IdleMaintenance(
                config=_config(),
                bus=bus,
                clear_metal_cache=clock.clear_metal,
            )
            idle.start()
            clock.advance(150.0)  # idle 150 s > 120 s threshold
            result = idle.maybe_tick()
            self.assertEqual(result["action"], "ran")
            self.assertIn("mlx_clear", result["actions"])
            # gc1 string includes a count
            self.assertTrue(any(a.startswith("gc1=") for a in result["actions"]))
            self.assertEqual(clock.mlx_clears, 1)
            self.assertEqual(idle.diagnostics()["mlx_clears_total"], 1)
            self.assertEqual(idle.diagnostics()["gen1_collections_total"], 1)

    def test_tick_respects_min_action_interval(self) -> None:
        clock = _ClockSpy()
        with patch("time.monotonic", clock):
            bus = FakeBus()
            idle = IdleMaintenance(
                config=_config(),
                bus=bus,
                clear_metal_cache=clock.clear_metal,
            )
            idle.start()
            clock.advance(150.0)
            r1 = idle.maybe_tick()
            self.assertEqual(r1["action"], "ran")
            # 30 s later - still idle, but cooldown not yet elapsed.
            clock.advance(30.0)
            r2 = idle.maybe_tick()
            self.assertEqual(r2["action"], "skip")
            self.assertEqual(r2["reason"], "cooldown")
            # 60 s after the action - cooldown expired.
            clock.advance(31.0)
            r3 = idle.maybe_tick()
            self.assertEqual(r3["action"], "ran")
            self.assertEqual(clock.mlx_clears, 2)

    def test_tick_skips_when_busy(self) -> None:
        clock = _ClockSpy()
        with patch("time.monotonic", clock):
            bus = FakeBus()
            idle = IdleMaintenance(
                config=_config(),
                bus=bus,
                clear_metal_cache=clock.clear_metal,
                is_busy=lambda: True,
            )
            idle.start()
            clock.advance(150.0)
            result = idle.maybe_tick()
            self.assertEqual(result["action"], "skip")
            self.assertEqual(result["reason"], "busy")
            self.assertEqual(clock.mlx_clears, 0)

    def test_tick_swallows_mlx_callback_exception(self) -> None:
        clock = _ClockSpy()

        def boom() -> None:
            raise RuntimeError("metal driver flaky")

        with patch("time.monotonic", clock):
            bus = FakeBus()
            idle = IdleMaintenance(
                config=_config(),
                bus=bus,
                clear_metal_cache=boom,
            )
            idle.start()
            clock.advance(150.0)
            # Must not raise.
            result = idle.maybe_tick()
            # Action still recorded as "ran" because gc.collect succeeded.
            self.assertEqual(result["action"], "ran")
            # mlx_clear is NOT in actions because the callback raised.
            self.assertNotIn("mlx_clear", result["actions"])
            self.assertEqual(idle.diagnostics()["mlx_clears_total"], 0)


class IdleMaintenanceFreezeTest(unittest.TestCase):
    """``freeze_now()`` is one-shot and observable via diagnostics."""

    def test_freeze_now_is_idempotent(self) -> None:
        # We cannot un-freeze in Python, so isolate by spawning the
        # freeze logic with a fresh IdleMaintenance instance and reset
        # at the end. ``gc.freeze()`` is global state — the second
        # call from the same process is harmless because IdleMaintenance
        # tracks ``self._frozen`` and short-circuits.
        prior = gc.get_threshold()
        try:
            idle = IdleMaintenance(config=_config())
            idle.start()
            count1 = idle.freeze_now()
            count2 = idle.freeze_now()
            self.assertGreaterEqual(count1, 0)
            # Second call short-circuits; returns 0 because already frozen.
            self.assertEqual(count2, 0)
            self.assertTrue(idle.diagnostics()["frozen"])
        finally:
            gc.set_threshold(*prior)
            try:
                gc.unfreeze()
            except Exception:
                pass

    def test_schedule_freeze_zero_delay_runs_inline(self) -> None:
        prior = gc.get_threshold()
        try:
            idle = IdleMaintenance(config=_config())
            idle.start()
            idle.schedule_freeze_after_boot(delay_s=0)
            self.assertTrue(idle.diagnostics()["frozen"])
        finally:
            gc.set_threshold(*prior)
            try:
                gc.unfreeze()
            except Exception:
                pass

    def test_schedule_freeze_uses_daemon_timer(self) -> None:
        """A small positive delay schedules a Timer; we just verify
        the scheduler call returns immediately and freezes shortly."""
        prior = gc.get_threshold()
        try:
            idle = IdleMaintenance(config=_config())
            idle.start()
            t0 = time.monotonic()
            idle.schedule_freeze_after_boot(delay_s=0.05)
            # Scheduling itself returns ~instantly.
            self.assertLess(time.monotonic() - t0, 0.05)
            # Wait briefly for the daemon timer to fire.
            for _ in range(20):
                if idle.diagnostics()["frozen"]:
                    break
                time.sleep(0.05)
            self.assertTrue(idle.diagnostics()["frozen"])
        finally:
            gc.set_threshold(*prior)
            try:
                gc.unfreeze()
            except Exception:
                pass


class IdleMaintenanceDiagnosticsTest(unittest.TestCase):
    def test_diagnostics_shape(self) -> None:
        clock = _ClockSpy()
        with patch("time.monotonic", clock):
            bus = FakeBus()
            idle = IdleMaintenance(config=_config(), bus=bus)
            idle.start()
            d: dict[str, Any] = idle.diagnostics()
            for key in (
                "enabled", "started", "frozen", "frozen_object_count",
                "idle_s", "since_last_action_s",
                "gen1_collections_total", "mlx_clears_total",
                "last_action_kind", "thresholds",
            ):
                self.assertIn(key, d)
            self.assertEqual(d["thresholds"]["gc_thresholds"], [2000, 25, 25])


if __name__ == "__main__":
    unittest.main()
