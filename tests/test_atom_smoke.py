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


def test_mlx_model_directory_exists() -> None:
    """ATOM v3.2 runs a single MLX model declared at brain.mlx_model."""
    raw = (_ATOM_ROOT / "config" / "settings.json").read_text(encoding="utf-8")
    cfg = json.loads(raw)
    brain = cfg.get("brain", {})
    rel = str(brain.get("mlx_model") or "").strip()
    assert rel, "brain.mlx_model missing from settings.json"
    p = (_ATOM_ROOT / rel).resolve()
    assert p.is_dir(), f"MLX model directory missing: {p}"


def test_mlx_brain_has_single_model_path() -> None:
    """ATOM v3.2 runs ONE local MLX model (Qwen2.5-7B-Instruct-4bit by
    default). MLXBrain still tags each request with a role label
    (``primary`` | ``fast``) for telemetry, but both resolve to the
    same on-disk path and the same in-memory tensors. Heavy reasoning
    is delegated to Gemini cloud via cognitive_kernel Path 2.65.
    """
    from brain.mlx_llm import MLXBrain

    raw = (_ATOM_ROOT / "config" / "settings.json").read_text(encoding="utf-8")
    cfg = json.loads(raw)
    b = MLXBrain(cfg)
    assert Path(b._model_path).is_dir()
    assert b._path_for_role("primary") == b._path_for_role("fast")
    assert b._path_for_role("primary") == b._model_path


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
