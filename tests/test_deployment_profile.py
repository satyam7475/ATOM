"""Tests for deployment profile audit and dashboard badge helpers."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.deployment_profile import (  # noqa: E402
    audit_corporate_alignment,
    deployment_dashboard_badge,
    PROFILE_CORPORATE_LAPTOP,
)


def test_audit_flags_cloud_stt() -> None:
    cfg = {
        "deployment": {"profile": PROFILE_CORPORATE_LAPTOP},
        "stt": {"engine": "google"},
    }
    msgs = audit_corporate_alignment(cfg)
    assert any("google" in m.lower() for m in msgs)


def test_audit_clean_baseline() -> None:
    cfg = {
        "deployment": {"profile": PROFILE_CORPORATE_LAPTOP},
        "features": {
            "web_research": False,
            "online_weather": False,
        },
        "vision": {"enabled": False},
        "brain": {"n_gpu_layers": 0},
        "stt": {"engine": "vosk"},
        "tts": {"engine": "sapi"},
        "security": {"mode": "strict"},
        "control": {"assistant_mode": "hybrid"},
    }
    msgs = audit_corporate_alignment(cfg)
    assert msgs == []


def test_dashboard_badge_respects_toggle() -> None:
    cfg = {
        "deployment": {
            "profile": PROFILE_CORPORATE_LAPTOP,
            "dashboard_badge": False,
        },
    }
    assert deployment_dashboard_badge(cfg) == ("", False)

    cfg["deployment"]["dashboard_badge"] = True
    label, show = deployment_dashboard_badge(cfg)
    assert show and label == "CORPORATE"


if __name__ == "__main__":
    test_audit_flags_cloud_stt()
    test_audit_clean_baseline()
    test_dashboard_badge_respects_toggle()
    print("test_deployment_profile: ok")
