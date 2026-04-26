"""Tests for the per-role memory governor (Sprint Ω.4.C, Apr 26 2026).

Drives :class:`core.memory_governor.MemoryGovernor` with synthetic memory
percentages instead of a live :class:`SiliconGovernor` so the eviction
order, tier classification, hysteresis, and bus integration are all
testable without a real M-series box.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory_governor import MemoryGovernor


# ──────────────────────────────────────────────────────────────────────
# Test doubles
# ──────────────────────────────────────────────────────────────────────


class FakeBus:
    """Captures ``on``/``off`` subscriptions and ``emit_fast`` events."""

    def __init__(self) -> None:
        self.subscriptions: dict[str, list] = {}
        self.unsubscriptions: list[tuple[str, object]] = []
        self.emitted: list[tuple[str, dict]] = []

    def on(self, event: str, handler) -> None:
        self.subscriptions.setdefault(event, []).append(handler)

    def off(self, event: str, handler) -> None:
        self.unsubscriptions.append((event, handler))
        try:
            self.subscriptions[event].remove(handler)
        except (KeyError, ValueError):
            pass

    def emit_fast(self, event: str, **data) -> None:
        self.emitted.append((event, data))


class _Spy:
    """Tiny callable spy: counts calls + records side-effects."""

    def __init__(self, name: str = "", raises: bool = False) -> None:
        self.name = name
        self.calls = 0
        self.raises = raises

    def __call__(self) -> None:
        self.calls += 1
        if self.raises:
            raise RuntimeError(f"intentional failure in {self.name!r}")


# ──────────────────────────────────────────────────────────────────────
# Configuration helpers
# ──────────────────────────────────────────────────────────────────────


def _config(**overrides) -> dict:
    """Build a memory_governor config block with default thresholds."""
    base = {
        "enabled": True,
        "tier1_threshold_pct": 80,
        "tier2_threshold_pct": 86,
        "tier3_threshold_pct": 92,
        "rewarm_hysteresis_pct": 6,
        "eviction_order": [
            "smolvlm",
            "whisper_confirmer",
            "draft_model",
            "embeddings_warm_cache",
            "persona_kv_cache",
        ],
    }
    base.update(overrides)
    return {"memory_governor": base}


# ──────────────────────────────────────────────────────────────────────
# Tier classification + escalation
# ──────────────────────────────────────────────────────────────────────


class TierClassificationTests(unittest.TestCase):
    """Walking through pressure curves should land in the right tier."""

    def test_tier_zero_below_tier1_threshold(self) -> None:
        gov = MemoryGovernor(bus=None, config=_config())
        spy = _Spy("smolvlm")
        gov.register("smolvlm", evict=spy)
        self.assertEqual(gov.on_stats(50.0), 0)
        self.assertEqual(gov.on_stats(79.9), 0)
        self.assertEqual(spy.calls, 0)

    def test_tier_one_evicts_first_third(self) -> None:
        # 5 roles → ceil(5/3) = 2 evictions at tier 1
        gov = MemoryGovernor(bus=None, config=_config())
        spies = [_Spy(name) for name in (
            "smolvlm", "whisper_confirmer", "draft_model",
            "embeddings_warm_cache", "persona_kv_cache",
        )]
        for name, spy in zip(gov.eviction_order, spies):
            gov.register(name, evict=spy)
        gov.on_stats(82.0)
        self.assertEqual(gov.current_tier, 1)
        self.assertEqual([s.calls for s in spies], [1, 1, 0, 0, 0])

    def test_tier_two_evicts_first_two_thirds(self) -> None:
        # 5 roles → ceil(2*5/3) = ceil(3.33) = 4 evictions at tier 2
        gov = MemoryGovernor(bus=None, config=_config())
        spies = [_Spy(name) for name in (
            "smolvlm", "whisper_confirmer", "draft_model",
            "embeddings_warm_cache", "persona_kv_cache",
        )]
        for name, spy in zip(gov.eviction_order, spies):
            gov.register(name, evict=spy)
        gov.on_stats(88.0)
        self.assertEqual(gov.current_tier, 2)
        self.assertEqual([s.calls for s in spies], [1, 1, 1, 1, 0])

    def test_tier_three_evicts_everything_including_sacred(self) -> None:
        gov = MemoryGovernor(bus=None, config=_config())
        spies = [_Spy(name) for name in (
            "smolvlm", "whisper_confirmer", "draft_model",
            "embeddings_warm_cache", "persona_kv_cache",
        )]
        for name, spy in zip(gov.eviction_order, spies):
            gov.register(name, evict=spy)
        gov.on_stats(95.0)
        self.assertEqual(gov.current_tier, 3)
        # All 5 evicted -- including the sacred persona_kv_cache.
        self.assertEqual([s.calls for s in spies], [1, 1, 1, 1, 1])

    def test_steady_state_does_not_re_fire_evict(self) -> None:
        gov = MemoryGovernor(bus=None, config=_config())
        spy = _Spy("smolvlm")
        gov.register("smolvlm", evict=spy)
        # Three consecutive tier-1 events at 82 % memory pressure.
        for _ in range(3):
            gov.on_stats(82.0)
        self.assertEqual(spy.calls, 1)


# ──────────────────────────────────────────────────────────────────────
# Hysteresis
# ──────────────────────────────────────────────────────────────────────


class HysteresisTests(unittest.TestCase):
    """Once a tier triggers, dropping just below threshold keeps the tier."""

    def test_pressure_dipping_below_threshold_holds_tier(self) -> None:
        # tier1=80, hysteresis=6 → must drop below 74 to clear tier 1.
        gov = MemoryGovernor(bus=None, config=_config())
        spy = _Spy("smolvlm")
        gov.register("smolvlm", evict=spy)
        gov.on_stats(82.0)
        self.assertEqual(gov.current_tier, 1)
        # Dipping to 78 (just below tier1=80) should stay at tier 1.
        gov.on_stats(78.0)
        self.assertEqual(gov.current_tier, 1)

    def test_pressure_dropping_below_hysteresis_relaxes_tier(self) -> None:
        gov = MemoryGovernor(bus=None, config=_config())
        spy = _Spy("smolvlm")
        gov.register("smolvlm", evict=spy)
        gov.on_stats(82.0)
        self.assertEqual(gov.current_tier, 1)
        # Drop below tier1 - hysteresis = 80 - 6 = 74.
        gov.on_stats(70.0)
        self.assertEqual(gov.current_tier, 0)

    def test_relax_clears_evicted_flag_so_re_escalation_fires_again(
        self,
    ) -> None:
        gov = MemoryGovernor(bus=None, config=_config())
        spy = _Spy("smolvlm")
        gov.register("smolvlm", evict=spy)
        gov.on_stats(82.0)
        self.assertEqual(spy.calls, 1)
        gov.on_stats(50.0)
        self.assertEqual(gov.current_tier, 0)
        gov.on_stats(82.0)
        self.assertEqual(spy.calls, 2)

    def test_rewarm_callback_fires_on_relax(self) -> None:
        gov = MemoryGovernor(bus=None, config=_config())
        evict = _Spy("smolvlm-evict")
        rewarm = _Spy("smolvlm-rewarm")
        gov.register("smolvlm", evict=evict, rewarm=rewarm)
        gov.on_stats(82.0)
        self.assertEqual(rewarm.calls, 0)
        gov.on_stats(40.0)
        self.assertEqual(rewarm.calls, 1)


# ──────────────────────────────────────────────────────────────────────
# Bus integration
# ──────────────────────────────────────────────────────────────────────


class BusIntegrationTests(unittest.TestCase):
    """``start()`` subscribes to silicon_stats_update; events reach evict."""

    def test_start_subscribes_and_stop_unsubscribes(self) -> None:
        bus = FakeBus()
        gov = MemoryGovernor(bus=bus, config=_config())
        gov.start()
        self.assertEqual(
            len(bus.subscriptions.get("silicon_stats_update", [])), 1,
        )
        gov.stop()
        self.assertEqual(
            len(bus.subscriptions.get("silicon_stats_update", [])), 0,
        )
        self.assertTrue(any(
            evt == "silicon_stats_update" for evt, _ in bus.unsubscriptions
        ))

    def test_handle_event_routes_memory_pct_into_on_stats(self) -> None:
        bus = FakeBus()
        gov = MemoryGovernor(bus=bus, config=_config())
        spy = _Spy("smolvlm")
        gov.register("smolvlm", evict=spy)
        gov.start()
        handler = bus.subscriptions["silicon_stats_update"][0]
        handler(stats={"memory_pct": 82.0})
        self.assertEqual(gov.current_tier, 1)
        self.assertEqual(spy.calls, 1)

    def test_handle_event_swallows_missing_stats_payload(self) -> None:
        bus = FakeBus()
        gov = MemoryGovernor(bus=bus, config=_config())
        gov.start()
        handler = bus.subscriptions["silicon_stats_update"][0]
        handler()
        handler(stats=None)
        handler(stats={"cpu_pct": 90.0})
        self.assertEqual(gov.current_tier, 0)

    def test_eviction_emits_memory_governor_event(self) -> None:
        bus = FakeBus()
        gov = MemoryGovernor(bus=bus, config=_config())
        gov.register("smolvlm", evict=_Spy("smolvlm"))
        gov.register("whisper_confirmer", evict=_Spy("whisper_confirmer"))
        gov.start()
        gov.on_stats(82.0)
        evicted_events = [
            data for evt, data in bus.emitted if evt == "memory_governor_evicted"
        ]
        self.assertEqual(len(evicted_events), 1)
        self.assertEqual(evicted_events[0]["tier"], 1)
        self.assertEqual(evicted_events[0]["roles"], ["smolvlm", "whisper_confirmer"])


# ──────────────────────────────────────────────────────────────────────
# Robustness
# ──────────────────────────────────────────────────────────────────────


class RobustnessTests(unittest.TestCase):
    """Bad config, raising callbacks, and edge cases must not break boot."""

    def test_disabled_config_never_evicts(self) -> None:
        gov = MemoryGovernor(bus=None, config=_config(enabled=False))
        spy = _Spy("smolvlm")
        gov.register("smolvlm", evict=spy)
        gov.on_stats(95.0)
        self.assertEqual(gov.current_tier, 0)
        self.assertEqual(spy.calls, 0)

    def test_unsorted_thresholds_fall_back_to_defaults(self) -> None:
        # tier1 > tier2 is invalid → governor should reset to defaults.
        gov = MemoryGovernor(bus=None, config=_config(
            tier1_threshold_pct=90, tier2_threshold_pct=85, tier3_threshold_pct=92,
        ))
        # Default tier1=80; pressure of 81 should now classify as tier 1.
        spy = _Spy("smolvlm")
        gov.register("smolvlm", evict=spy)
        self.assertEqual(gov.on_stats(81.0), 1)

    def test_role_outside_eviction_order_is_never_evicted(self) -> None:
        gov = MemoryGovernor(bus=None, config=_config())
        ghost = _Spy("ghost")
        gov.register("ghost", evict=ghost)
        gov.on_stats(95.0)
        self.assertEqual(ghost.calls, 0)

    def test_evict_exception_does_not_stop_walker(self) -> None:
        gov = MemoryGovernor(bus=None, config=_config())
        first = _Spy("smolvlm", raises=True)
        second = _Spy("whisper_confirmer")
        gov.register("smolvlm", evict=first)
        gov.register("whisper_confirmer", evict=second)
        gov.on_stats(82.0)
        self.assertEqual(first.calls, 1)
        self.assertEqual(second.calls, 1)

    def test_register_requires_callable(self) -> None:
        gov = MemoryGovernor(bus=None, config=_config())
        with self.assertRaises(TypeError):
            gov.register("oops", evict="not a callable")  # type: ignore[arg-type]

    def test_unregister_removes_role(self) -> None:
        gov = MemoryGovernor(bus=None, config=_config())
        spy = _Spy("smolvlm")
        gov.register("smolvlm", evict=spy)
        gov.unregister("smolvlm")
        gov.on_stats(95.0)
        self.assertEqual(spy.calls, 0)

    def test_diagnostics_snapshot_shape(self) -> None:
        gov = MemoryGovernor(bus=None, config=_config())
        spy = _Spy("smolvlm")
        gov.register("smolvlm", evict=spy)
        gov.on_stats(82.0)
        snap = gov.diagnostics()
        self.assertTrue(snap["enabled"])
        self.assertEqual(snap["current_tier"], 1)
        self.assertIn("smolvlm", snap["registered_roles"])
        self.assertIn("smolvlm", snap["evicted_roles"])
        self.assertEqual(snap["evictions_total"], 1)
        self.assertEqual(snap["thresholds"]["tier1"], 80.0)


# ──────────────────────────────────────────────────────────────────────
# Custom eviction orders
# ──────────────────────────────────────────────────────────────────────


class CustomOrderTests(unittest.TestCase):
    """User-supplied eviction orders (subset of defaults, custom names)."""

    def test_custom_three_role_order_walks_correctly(self) -> None:
        # 3 roles → tier 1: ceil(3/3)=1, tier 2: ceil(6/3)=2, tier 3: 3
        gov = MemoryGovernor(bus=None, config=_config(
            eviction_order=["alpha", "beta", "gamma"],
        ))
        spies = {name: _Spy(name) for name in ("alpha", "beta", "gamma")}
        for name, spy in spies.items():
            gov.register(name, evict=spy)
        gov.on_stats(82.0)
        self.assertEqual(spies["alpha"].calls, 1)
        self.assertEqual(spies["beta"].calls, 0)
        gov.on_stats(88.0)
        self.assertEqual(spies["beta"].calls, 1)
        self.assertEqual(spies["gamma"].calls, 0)
        gov.on_stats(95.0)
        self.assertEqual(spies["gamma"].calls, 1)

    def test_single_role_order_is_only_evicted_at_tier_three(self) -> None:
        # n=1: ceil(1/3)=1 → tier 1 also evicts the single role.
        # That is intentional under the new contract: order is the only
        # protection level, and a 1-role config has no sacred slot.
        gov = MemoryGovernor(bus=None, config=_config(
            eviction_order=["only_role"],
        ))
        spy = _Spy("only_role")
        gov.register("only_role", evict=spy)
        gov.on_stats(82.0)
        self.assertEqual(spy.calls, 1)


if __name__ == "__main__":
    unittest.main()
