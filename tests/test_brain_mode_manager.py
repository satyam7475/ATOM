from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.brain_mode_manager import BrainModeManager


def _config() -> dict:
    return {
        "brain": {
            "n_ctx": 6144,
            "n_threads": 4,
            "n_gpu_layers": -1,
            "n_batch": 384,
            "max_tokens": 384,
            "timeout_seconds": 24,
        },
        "assistant_brain": {
            "active_profile": "optimal",
            "restore_persisted_profile": False,
            "persist_active_profile": False,
            "profiles": {
                "optimal": {
                    "n_ctx": 4096,
                    "max_tokens": 320,
                    "timeout_seconds": 18,
                },
                "full_performance": {
                    "n_ctx": 12288,
                    "max_tokens": 640,
                    "timeout_seconds": 32,
                },
            },
        },
    }


def test_legacy_aliases_map_to_canonical_profiles() -> None:
    mgr = BrainModeManager(_config())

    ok, _ = mgr.set_profile("brain")
    assert ok is True
    assert mgr.active_profile == "full_performance"

    ok, _ = mgr.set_profile("balanced")
    assert ok is True
    assert mgr.active_profile == "optimal"


def test_feature_flags_and_effective_params_follow_profile() -> None:
    mgr = BrainModeManager(_config())

    assert mgr.feature_enabled("dream") is False
    assert mgr.effective_params()["n_ctx"] == 4096

    ok, _ = mgr.set_profile("full performance")
    assert ok is True
    assert mgr.feature_enabled("dream") is True
    assert mgr.feature_enabled("prediction_prefetch") is True
    assert mgr.effective_params()["max_tokens"] == 640
