"""Sprint C1 — regression tests for MLXBrain.pin_prompt_prefix and
repin_persona_if_changed.

We can't load the real Qwen-3-4B-MLX weights in CI, so every test
patches ``MLXBrain._ensure_loaded`` + ``MLXBrain._generate_sync`` to
spy on the prefill call and verifies the bookkeeping (mtime watch,
diagnostics surface, role normalisation) without touching the GPU.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mlx_brain(monkeypatch):
    """Construct an MLXBrain with the heavy MLX bits stubbed out."""
    from brain.mlx_llm import MLXBrain

    config = {"brain": {"enabled": True, "mlx_model": "/tmp/atom-fake-model"}}
    brain = MLXBrain(config)

    # Pretend MLX, the model directory, and both roles are healthy.
    monkeypatch.setattr(
        brain, "_ensure_loaded", lambda role=None: True,
    )
    monkeypatch.setattr(
        brain, "is_available", lambda: True,
    )
    # Stub ``_generate_sync`` so we don't actually call MLX.
    spy = MagicMock(return_value=("", False))
    monkeypatch.setattr(brain, "_generate_sync", spy)

    # A fake tokenizer for the FAST role -- only ``encode`` is used.
    fake_tokenizer = MagicMock()
    fake_tokenizer.encode = MagicMock(return_value=list(range(123)))
    fake_tokenizer.bos_token = None
    brain._tokenizers["fast"] = fake_tokenizer
    brain._tokenizers["primary"] = fake_tokenizer
    brain._loaded_roles["fast"] = True
    brain._loaded_roles["primary"] = True

    # Force the _HAS_MLX flag (module-level) so pin_prompt_prefix's
    # availability gate passes. We patch the attribute on the module.
    import brain.mlx_llm as mod
    monkeypatch.setattr(mod, "_HAS_MLX", True, raising=False)

    return brain, spy


def test_pin_prompt_prefix_runs_one_token_prefill(mlx_brain):
    brain, spy = mlx_brain
    result = brain.pin_prompt_prefix(
        "You are ATOM. Boss is Satyam.",
        model_role="fast",
    )
    assert result["ok"] is True, f"pin failed: {result}"
    assert spy.call_count == 1, "pin should call _generate_sync exactly once"
    _, kwargs = spy.call_args
    assert kwargs.get("max_tokens_override") == 1, \
        "pin must use max_tokens=1 to keep the prefill cheap"
    assert kwargs.get("model_role") == "fast"


def test_pin_prompt_prefix_records_token_count(mlx_brain):
    brain, _ = mlx_brain
    result = brain.pin_prompt_prefix("persona text here")
    assert result["tokens"] == 123, \
        f"expected the fake tokenizer to report 123 tokens, got {result}"
    info = brain.pinned_persona_info
    assert info["tokens"] == 123
    assert info["role"] == "fast"


def test_pin_prompt_prefix_rejects_empty_prefix(mlx_brain):
    brain, spy = mlx_brain
    result = brain.pin_prompt_prefix("")
    assert result["ok"] is False
    assert spy.call_count == 0


def test_pin_prompt_prefix_records_source_path_mtime(mlx_brain, tmp_path):
    brain, _ = mlx_brain
    persona = tmp_path / "atom_persona.md"
    persona.write_text("PERSONA v1", encoding="utf-8")
    original_mtime = persona.stat().st_mtime

    result = brain.pin_prompt_prefix(
        persona.read_text(encoding="utf-8"),
        source_path=str(persona),
    )
    assert result["ok"] is True

    info = brain.pinned_persona_info
    assert info["path"] == str(persona)
    assert info["mtime"] == pytest.approx(original_mtime, abs=1e-3)


def test_repin_persona_if_changed_skips_when_mtime_unchanged(
    mlx_brain, tmp_path,
):
    brain, spy = mlx_brain
    persona = tmp_path / "atom_persona.md"
    persona.write_text("PERSONA v1", encoding="utf-8")

    brain.pin_prompt_prefix(
        persona.read_text(encoding="utf-8"),
        source_path=str(persona),
    )
    spy.reset_mock()

    rep = brain.repin_persona_if_changed()
    assert rep is False, "no mtime change should skip the re-pin"
    assert spy.call_count == 0


def test_repin_persona_if_changed_repins_when_mtime_moves(
    mlx_brain, tmp_path,
):
    brain, spy = mlx_brain
    persona = tmp_path / "atom_persona.md"
    persona.write_text("PERSONA v1", encoding="utf-8")
    brain.pin_prompt_prefix(
        persona.read_text(encoding="utf-8"),
        source_path=str(persona),
    )
    spy.reset_mock()

    # Bump the mtime forward.
    new_mtime = persona.stat().st_mtime + 5.0
    persona.write_text("PERSONA v2 -- updated", encoding="utf-8")
    import os
    os.utime(persona, (new_mtime, new_mtime))

    rep = brain.repin_persona_if_changed()
    assert rep is True
    assert spy.call_count == 1, \
        "an mtime move should run a fresh pin prefill"


def test_repin_persona_if_changed_returns_false_without_pin(mlx_brain):
    brain, spy = mlx_brain
    assert brain.repin_persona_if_changed() is False
    assert spy.call_count == 0


def test_pin_prompt_prefix_bails_when_role_not_loaded(monkeypatch):
    from brain.mlx_llm import MLXBrain

    config = {"brain": {"enabled": True, "mlx_model": "/tmp/atom-fake-model"}}
    brain = MLXBrain(config)
    monkeypatch.setattr(brain, "is_available", lambda: True)
    monkeypatch.setattr(brain, "_ensure_loaded", lambda role=None: False)
    spy = MagicMock(return_value=("", False))
    monkeypatch.setattr(brain, "_generate_sync", spy)

    import brain.mlx_llm as mod
    monkeypatch.setattr(mod, "_HAS_MLX", True, raising=False)

    res = brain.pin_prompt_prefix("anything")
    assert res["ok"] is False
    assert "not loaded" in res["reason"]
    assert spy.call_count == 0


def test_pin_prompt_prefix_bails_when_mlx_unavailable(monkeypatch):
    from brain.mlx_llm import MLXBrain

    config = {"brain": {"enabled": True, "mlx_model": "/tmp/atom-fake-model"}}
    brain = MLXBrain(config)
    monkeypatch.setattr(brain, "is_available", lambda: False)

    res = brain.pin_prompt_prefix("anything")
    assert res["ok"] is False


def test_pin_prompt_prefix_swallows_prefill_failure(mlx_brain):
    brain, spy = mlx_brain
    spy.side_effect = RuntimeError("boom")
    res = brain.pin_prompt_prefix("persona")
    assert res["ok"] is False
    assert "raised" in res["reason"]


def test_pinned_persona_info_initial_state(monkeypatch):
    from brain.mlx_llm import MLXBrain

    brain = MLXBrain({"brain": {"enabled": True, "mlx_model": "/tmp/m"}})
    info = brain.pinned_persona_info
    assert info == {
        "path": None,
        "mtime": 0.0,
        "role": "fast",
        "tokens": 0,
    }
