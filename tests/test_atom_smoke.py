"""Smoke tests: config validity and brain model paths (no full ATOM boot)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_ATOM_ROOT = Path(__file__).resolve().parents[1]


def test_settings_json_validates() -> None:
    from core.config_schema import validate_config

    raw = (_ATOM_ROOT / "config" / "settings.json").read_text(encoding="utf-8")
    cfg = json.loads(raw)
    errors = validate_config(cfg)
    assert not errors, "settings.json schema errors:\n" + "\n".join(errors)


def test_mlx_model_directories_exist() -> None:
    """Lean profile uses qwen3-1.7b-mlx for both roles — paths must exist."""
    raw = (_ATOM_ROOT / "config" / "settings.json").read_text(encoding="utf-8")
    cfg = json.loads(raw)
    brain = cfg.get("brain", {})
    for key in ("mlx_primary_model", "mlx_fast_model"):
        rel = str(brain.get(key) or "").strip()
        assert rel, f"brain.{key} missing"
        p = (_ATOM_ROOT / rel).resolve()
        assert p.is_dir(), f"MLX model directory missing: {p}"


def test_mlx_brain_has_distinct_tier_paths() -> None:
    """Fast and primary tiers should point to distinct model directories.

    Earlier config kept both pointing at qwen3-4b-mlx, which made the
    cognitive-kernel's fast-path tier pointless. The settings now route
    fast traffic at qwen3-1.7b-mlx while primary stays on the 4b model.
    """
    from brain.mlx_llm import MLXBrain

    raw = (_ATOM_ROOT / "config" / "settings.json").read_text(encoding="utf-8")
    cfg = json.loads(raw)
    b = MLXBrain(cfg)
    assert Path(b._primary_path).resolve() != Path(b._fast_path).resolve()
    assert Path(b._primary_path).is_dir()
    assert Path(b._fast_path).is_dir()
