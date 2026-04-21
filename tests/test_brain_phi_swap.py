"""Phase 2 v3 regression tests -- brain swap to Phi-3.5-mini-only.

Pins:
  1. Brain config points at the Phi-3.5-mini-MLX-4bit model directory
     for BOTH primary and fast roles.
  2. Heavy "deep" reasoning is no longer a separate on-device model --
     deep queries route to cloud (Gemini) via cognitive_kernel
     Path 2.65; if cloud is unavailable, ``deep`` collapses to the
     primary Phi model so callers still get an answer.
  3. MLXBrain still accepts "deep" as a valid role -- it just resolves
     to the primary path when ``mlx_deep_model`` is unset.
  4. _max_tokens_override caps are tuned for Phi's tighter generation
     behaviour (SHORT 96, NORMAL 160, DETAIL 256, REPORT None).
  5. The Phi model directory exists on disk (download succeeded).

The actual Phi smoke generation is exercised in
``tests/test_brain_phi_smoke.py`` (separate file, marked slow / skipped
in headless CI) so this file stays fast and import-safe.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def _settings() -> dict:
    with (REPO_ROOT / "config" / "settings.json").open() as fh:
        return json.load(fh)


# ─────────────────────────────────────────────────────────────────────
# Config swap: primary/fast = Phi, deep = Qwen
# ─────────────────────────────────────────────────────────────────────


def test_brain_primary_is_phi_3_5_mini():
    cfg = _settings()
    primary = cfg["brain"]["mlx_primary_model"]
    assert "phi-3.5-mini" in primary.lower(), (
        f"Brain primary model must be Phi-3.5-mini after v3 swap. Got {primary!r}"
    )


def test_brain_fast_is_phi_3_5_mini():
    cfg = _settings()
    fast = cfg["brain"]["mlx_fast_model"]
    assert "phi-3.5-mini" in fast.lower(), (
        f"Brain fast model must be Phi-3.5-mini after v3 swap. Got {fast!r}"
    )


def test_brain_deep_is_unset_collapses_to_phi():
    """v3 cleanup: heavy on-device deep model has been retired.

    Deep reasoning now routes to Gemini cloud (cognitive_kernel
    Path 2.65). When cloud is unreachable AND a deep query still gets
    queued, ``mlx_brain._path_for_role('deep')`` falls back to the
    primary Phi path so callers always get an answer.
    """
    cfg = _settings()
    assert "mlx_deep_model" not in cfg["brain"], (
        "v3 cleanup expects mlx_deep_model to be removed from settings.json; "
        "deep reasoning should route to cloud, not a 4 GB on-device model."
    )


def test_phi_model_directory_exists():
    cfg = _settings()
    p = REPO_ROOT / cfg["brain"]["mlx_primary_model"]
    assert p.is_dir(), f"Phi-3.5-mini model directory not found at {p}"
    # Sanity: must have weight + tokenizer files.
    safetensors = list(p.glob("*.safetensors"))
    assert safetensors, f"No .safetensors weights found in {p}"
    assert (p / "tokenizer.json").exists(), f"Missing tokenizer.json in {p}"
    assert (p / "config.json").exists(), f"Missing config.json in {p}"


# ─────────────────────────────────────────────────────────────────────
# MLXBrain accepts "deep" as a role
# ─────────────────────────────────────────────────────────────────────


def test_mlx_brain_valid_roles_includes_deep():
    from brain.mlx_llm import MLXBrain

    assert "deep" in MLXBrain._VALID_ROLES
    assert "primary" in MLXBrain._VALID_ROLES
    assert "fast" in MLXBrain._VALID_ROLES


def test_mlx_brain_path_for_role_deep_collapses_to_phi():
    """With ``mlx_deep_model`` unset (v3 cleanup), the deep role must
    collapse to the primary Phi path, not crash."""
    from brain.mlx_llm import MLXBrain

    cfg = _settings()
    brain = MLXBrain(cfg)
    assert brain._path_for_role("deep").endswith("phi-3.5-mini-mlx-4bit")
    assert brain._path_for_role("primary").endswith("phi-3.5-mini-mlx-4bit")
    assert brain._path_for_role("fast").endswith("phi-3.5-mini-mlx-4bit")


def test_mlx_brain_path_for_role_unknown_falls_back_to_primary():
    from brain.mlx_llm import MLXBrain

    cfg = _settings()
    brain = MLXBrain(cfg)
    # Unknown role -> primary fallback (Phi).
    assert brain._path_for_role("nonsense").endswith("phi-3.5-mini-mlx-4bit")


def test_mlx_brain_deep_path_falls_back_to_primary_when_unset():
    from brain.mlx_llm import MLXBrain

    cfg = {"brain": {
        "mlx_primary_model": "models/phi-3.5-mini-mlx-4bit",
        "mlx_fast_model": "models/phi-3.5-mini-mlx-4bit",
        # No mlx_deep_model key.
    }}
    brain = MLXBrain(cfg)
    # No deep model configured -> falls back to primary (so caller still
    # gets a sensible answer).
    assert brain._path_for_role("deep").endswith("phi-3.5-mini-mlx-4bit")


def test_mlx_brain_dicts_have_deep_slot():
    """All per-role state dicts must include a 'deep' slot so the new
    role doesn't KeyError on first access."""
    from brain.mlx_llm import MLXBrain

    cfg = _settings()
    brain = MLXBrain(cfg)
    assert "deep" in brain._models
    assert "deep" in brain._tokenizers
    assert "deep" in brain._fingerprints
    assert "deep" in brain._loaded_roles
    assert "deep" in brain._load_failed
    assert "deep" in brain._role_last_used


# ─────────────────────────────────────────────────────────────────────
# Idle-unload guard for the heavy deep model
# ─────────────────────────────────────────────────────────────────────


def test_deep_model_idle_unload_threshold_is_sensible():
    """The heavy deep model holds ~5GB; we want it dropped after a few
    minutes of inactivity, not seconds (would thrash) and not hours
    (would block other models)."""
    from brain.mlx_llm import MLXBrain

    threshold = MLXBrain._DEEP_IDLE_UNLOAD_S
    assert 60 <= threshold <= 1800, (
        f"_DEEP_IDLE_UNLOAD_S should be between 1 min and 30 min, got {threshold}"
    )


def test_maybe_unload_idle_deep_skips_when_not_loaded():
    """No-op when nothing is loaded -- safe to call any time."""
    from brain.mlx_llm import MLXBrain

    cfg = _settings()
    brain = MLXBrain(cfg)
    # Should not raise, should not log loudly -- nothing loaded yet.
    brain._maybe_unload_idle_deep()
    assert not brain._loaded_roles["deep"]


# ─────────────────────────────────────────────────────────────────────
# max_tokens override caps tuned for Phi
# ─────────────────────────────────────────────────────────────────────


def test_max_tokens_override_short_for_phi():
    from cursor_bridge.local_brain_controller import LocalBrainController
    from core.query_policy import ResponseMode

    cap = LocalBrainController._max_tokens_override(
        response_mode=ResponseMode.SHORT,
        budget_tier="command",
        requested_tier="command",
    )
    # Phi-tuned: 96 (was 72 for Qwen). Anything above 110 is the model
    # starting to narrate -- still tight.
    assert cap == 96


def test_max_tokens_override_simple_for_phi():
    from cursor_bridge.local_brain_controller import LocalBrainController
    from core.query_policy import ResponseMode

    cap = LocalBrainController._max_tokens_override(
        response_mode=ResponseMode.NORMAL,
        budget_tier="simple",
        requested_tier="simple",
    )
    assert cap == 128


def test_max_tokens_override_detail_for_phi():
    from cursor_bridge.local_brain_controller import LocalBrainController
    from core.query_policy import ResponseMode

    cap = LocalBrainController._max_tokens_override(
        response_mode=ResponseMode.DETAIL,
        budget_tier="complex",
        requested_tier="complex",
    )
    # Phi is more concise per token -> lifted to 256 (was 192).
    assert cap == 256


def test_max_tokens_override_normal_default_for_phi():
    from cursor_bridge.local_brain_controller import LocalBrainController
    from core.query_policy import ResponseMode

    cap = LocalBrainController._max_tokens_override(
        response_mode=ResponseMode.NORMAL,
        budget_tier="standard",
        requested_tier="standard",
    )
    assert cap == 160


def test_max_tokens_override_report_unbounded():
    """Report mode keeps the unbounded path (None) so long-form answers
    aren't truncated. Both Qwen and Phi use the default model max
    when the cap is None."""
    from cursor_bridge.local_brain_controller import LocalBrainController
    from core.query_policy import ResponseMode

    cap = LocalBrainController._max_tokens_override(
        response_mode=ResponseMode.REPORT,
        budget_tier="creative",
        requested_tier="creative",
    )
    assert cap is None
