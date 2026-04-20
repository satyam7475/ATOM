"""Focused tests for Sprint B3: auto-demote brain profile under sustained pressure."""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field


@dataclass
class _State:
    hot_streak: int = 0
    clear_streak: int = 0
    pre_pressure_profile: str | None = None
    HOT_REQUIRED: int = 3
    CLEAR_REQUIRED: int = 4
    active_profile: str = "full_performance"
    events: list[tuple[str, str]] = field(default_factory=list)


def _simulate_sample(state: _State, tier: int) -> None:
    """One orchestrator tick — mirrors the B3 block in main.py."""
    if tier >= 2:
        state.hot_streak += 1
        state.clear_streak = 0
    elif tier == 0:
        state.clear_streak += 1
        state.hot_streak = 0
    else:
        state.hot_streak = max(0, state.hot_streak - 1)
        state.clear_streak = 0

    if (
        state.hot_streak >= state.HOT_REQUIRED
        and state.active_profile == "full_performance"
        and state.pre_pressure_profile is None
    ):
        state.pre_pressure_profile = state.active_profile
        state.active_profile = "optimal"
        state.events.append(("demote", state.pre_pressure_profile))
    elif (
        state.pre_pressure_profile is not None
        and state.clear_streak >= state.CLEAR_REQUIRED
        and state.active_profile == "optimal"
    ):
        target = state.pre_pressure_profile
        state.pre_pressure_profile = None
        state.active_profile = target
        state.events.append(("restore", target))


class PressureProfileTests(unittest.TestCase):
    def test_sustained_pressure_demotes_to_optimal(self) -> None:
        s = _State()
        for _ in range(s.HOT_REQUIRED):
            _simulate_sample(s, tier=2)
        self.assertEqual(s.active_profile, "optimal")
        self.assertEqual(s.pre_pressure_profile, "full_performance")
        self.assertEqual(s.events[-1], ("demote", "full_performance"))

    def test_single_spike_does_not_demote(self) -> None:
        s = _State()
        _simulate_sample(s, tier=2)
        _simulate_sample(s, tier=0)
        _simulate_sample(s, tier=0)
        self.assertEqual(s.active_profile, "full_performance")
        self.assertFalse(any(ev[0] == "demote" for ev in s.events))

    def test_restore_after_sustained_clear(self) -> None:
        s = _State()
        for _ in range(s.HOT_REQUIRED):
            _simulate_sample(s, tier=3)
        self.assertEqual(s.active_profile, "optimal")

        for _ in range(s.CLEAR_REQUIRED):
            _simulate_sample(s, tier=0)
        self.assertEqual(s.active_profile, "full_performance")
        self.assertIsNone(s.pre_pressure_profile)
        self.assertEqual(s.events[-1][0], "restore")

    def test_optimal_start_never_demotes(self) -> None:
        s = _State(active_profile="optimal")
        for _ in range(s.HOT_REQUIRED + 2):
            _simulate_sample(s, tier=3)
        self.assertEqual(s.active_profile, "optimal")
        self.assertIsNone(s.pre_pressure_profile)
        self.assertEqual(s.events, [])


if __name__ == "__main__":
    unittest.main()
