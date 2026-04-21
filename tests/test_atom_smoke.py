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
    """Single-model MLX profile uses phi-3.5-mini-mlx-4bit for both roles."""
    raw = (_ATOM_ROOT / "config" / "settings.json").read_text(encoding="utf-8")
    cfg = json.loads(raw)
    brain = cfg.get("brain", {})
    for key in ("mlx_primary_model", "mlx_fast_model"):
        rel = str(brain.get(key) or "").strip()
        assert rel, f"brain.{key} missing"
        p = (_ATOM_ROOT / rel).resolve()
        assert p.is_dir(), f"MLX model directory missing: {p}"


def test_mlx_brain_shares_single_model_path() -> None:
    """Fast and primary roles reuse the same Phi-3.5-mini directory.

    ATOM v3 runs ONE local model (Phi-3.5-mini-MLX-4bit) for both
    latency tiers. MLXBrain still exposes ``fast`` and ``primary``
    roles for routing, but both resolve to the same on-disk model path
    and share one in-memory load. Heavy reasoning is delegated to
    Gemini cloud via cognitive_kernel Path 2.65.
    """
    from brain.mlx_llm import MLXBrain

    raw = (_ATOM_ROOT / "config" / "settings.json").read_text(encoding="utf-8")
    cfg = json.loads(raw)
    b = MLXBrain(cfg)
    assert Path(b._primary_path).resolve() == Path(b._fast_path).resolve()
    assert Path(b._primary_path).is_dir()
    assert Path(b._fast_path).is_dir()


def test_voice_defaults_pin_reliable_stt_and_jarvis_preset() -> None:
    """Voice defaults should favor reliable always-on handling on this Mac.

    ``en-US`` avoids the recurring en-IN ``atom -> adam`` misrecognition, and
    the ``jarvis`` TTS preset resolves to the best British voice available on
    the host (Daniel compact today, premium Daniel later if installed). The
    voice pipeline now defaults to always-on activation with duplex/barge-in
    enabled for a more Jarvis-like interaction loop.
    """
    raw = (_ATOM_ROOT / "config" / "settings.json").read_text(encoding="utf-8")
    cfg = json.loads(raw)
    assert cfg["stt"]["locale"] == "en-US"
    assert cfg["stt"]["barge_in_during_speak"] is True
    assert cfg["tts"]["macos_voice"] == "jarvis"
    assert cfg["voice"]["activation_mode"] == "always_on"
