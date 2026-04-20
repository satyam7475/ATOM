"""
ATOM -- Sprint D5 focused tests: BehaviorMemory persistence round-trip.

These tests verify that:
    1. The learned profile is serialized to disk on flush().
    2. A new BehaviorMemory instance picks up the saved profile.
    3. Corrupted / partial files fall back to defaults safely.
    4. ``persist_path=None`` disables all disk I/O.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from core.adaptive.behavior_memory import BehaviorMemory, _DEFAULT_PROFILE


def test_profile_round_trips_to_disk() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "profile.json"
        mem = BehaviorMemory(persist_path=path)
        for _ in range(6):
            mem.record({
                "words_spoken": 40,
                "duration_ms": 3000,
                "interrupt_count": 2,
            })
        mem.update_from_metrics()
        mem.flush()

        assert path.exists(), "profile should be written on flush"
        raw = json.loads(path.read_text())
        assert "profile" in raw
        saved_prof = raw["profile"]
        live_prof = mem.get_profile()
        for key in _DEFAULT_PROFILE:
            assert key in saved_prof
            assert abs(float(saved_prof[key]) - live_prof[key]) < 1e-6

        mem2 = BehaviorMemory(persist_path=path)
        restored = mem2.get_profile()
        for key in _DEFAULT_PROFILE:
            assert abs(restored[key] - live_prof[key]) < 1e-6, (
                f"profile key {key} did not restore correctly: "
                f"{restored[key]} vs {live_prof[key]}"
            )


def test_missing_file_falls_back_to_defaults() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "does_not_exist.json"
        mem = BehaviorMemory(persist_path=path)
        prof = mem.get_profile()
        for key, default_val in _DEFAULT_PROFILE.items():
            assert prof[key] == default_val


def test_corrupted_file_falls_back_to_defaults() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "broken.json"
        path.write_text("not-json-at-all")
        mem = BehaviorMemory(persist_path=path)
        prof = mem.get_profile()
        for key, default_val in _DEFAULT_PROFILE.items():
            assert prof[key] == default_val


def test_persist_path_none_disables_io() -> None:
    mem = BehaviorMemory(persist_path=None)
    for _ in range(4):
        mem.record({
            "words_spoken": 20,
            "duration_ms": 2000,
            "interrupt_count": 0,
        })
    mem.update_from_metrics()
    mem.flush()
    assert mem._persist_enabled is False


if __name__ == "__main__":
    test_profile_round_trips_to_disk()
    test_missing_file_falls_back_to_defaults()
    test_corrupted_file_falls_back_to_defaults()
    test_persist_path_none_disables_io()
    print("[D5] All BehaviorMemory persistence tests passed.")
