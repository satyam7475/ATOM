"""v3.3 regression tests -- single-model brain (Qwen3-4B-Instruct-2507-4bit).

ATOM was on Qwen2.5-7B-Instruct-MLX-4bit until 2026-04-24; the v3.3
JARVIS-grade rewrite swapped to Qwen3-4B-Instruct-2507-4bit because the
7B was the dominant cause of:
  - 6.3 GB warm RAM on a 16 GB Apple Silicon machine
  - 14 tok/s steady-state (3-4 s first-token latency)
  - frequent ``memory_pressure`` degrade trips when Cursor + Chrome ran
    alongside ATOM

Qwen3-4B-Instruct-2507-4bit lands at:
  - 2.1 GB on disk (was 4.0 GB)
  - ~2.4 GB warm RAM (was ~4.5 GB)
  - higher steady-state tok/s with first-token < 1.5 s on the smoke run
  - same single-model alias model: every role still resolves to one set
    of weights, no extra memory for the 'fast' alias

Pins:
  1. Brain config points at the Qwen3-4B-Instruct-2507-4bit directory
     via the single ``brain.mlx_model`` key.
  2. ATOM is single-model: legacy ``mlx_primary_model`` /
     ``mlx_fast_model`` / ``mlx_deep_model`` / ``mlx_default_role`` /
     ``model_path`` keys are gone from the default settings.json but the
     loader still accepts them for backwards compatibility.
  3. MLXBrain tags every request with a role label (``primary`` |
     ``fast``) for observability, but both resolve to the same weights
     and the same in-memory tensors via the alias-on-first-use path.
  4. The Qwen model directory exists on disk.

Live generation is exercised in ``scripts/smoke_metal_warmup.py`` (real
MLX + real torch.mps + real cold-start) so this file stays fast and
import-safe.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
# Updated 2026-04-24 (v3.3): swapped 7B -> 4B for JARVIS-grade
# responsiveness. Both names are kept here as constants because the
# back-compat tests below have to reference the *current* on-disk model
# directory (the legacy 7B dir is removed after the swap is verified).
_QWEN_DIRNAME = "qwen3-4b-instruct-4bit"


def _settings() -> dict:
    with (REPO_ROOT / "config" / "settings.json").open() as fh:
        return json.load(fh)


# ─────────────────────────────────────────────────────────────────────
# Config: single mlx_model key
# ─────────────────────────────────────────────────────────────────────


def test_brain_mlx_model_is_qwen3_4b():
    cfg = _settings()
    model = cfg["brain"]["mlx_model"]
    assert _QWEN_DIRNAME in model.lower(), (
        f"Brain must load Qwen3-4B-Instruct-2507 via brain.mlx_model after "
        f"the v3.3 lightweight rewrite. Got {model!r}"
    )


def test_brain_legacy_keys_removed():
    """The single-model cleanup drops the role-specific keys. The loader
    still reads them if present (for upgrading users) but the shipped
    settings.json should not carry the dead weight."""
    cfg = _settings()
    legacy_keys = (
        "mlx_primary_model",
        "mlx_fast_model",
        "mlx_deep_model",
        "mlx_default_role",
        "model_path",
    )
    present = [k for k in legacy_keys if k in cfg["brain"]]
    assert not present, (
        f"Legacy brain keys should be removed after v3.2 single-model "
        f"cleanup: {present}"
    )


def test_qwen_model_directory_exists():
    cfg = _settings()
    p = REPO_ROOT / cfg["brain"]["mlx_model"]
    assert p.is_dir(), f"Qwen3-4B-Instruct model directory not found at {p}"
    safetensors = list(p.glob("*.safetensors"))
    assert safetensors, f"No .safetensors weights found in {p}"
    assert (p / "tokenizer.json").exists(), f"Missing tokenizer.json in {p}"
    assert (p / "config.json").exists(), f"Missing config.json in {p}"


def test_brain_watchdog_budgets_match_4b_class():
    """v3.3: tighter watchdog/latency budgets for the 4B brain.

    The 7B's worst-case first-token was ~3-4 s, which forced a 28 s
    LLM watchdog. The 4B replies in <1.5 s on the smoke and gets a
    20 s brain timeout + 14 s LLM watchdog, which is what makes ATOM
    feel snappier without false-positive recovery trips.
    """
    cfg = _settings()
    assert cfg["brain"]["timeout_seconds"] <= 22, (
        f"4B should not need the 7B's 28 s timeout; got "
        f"{cfg['brain']['timeout_seconds']}"
    )
    perf = cfg["performance"]
    assert perf["watchdog_llm_timeout_s"] <= 16, (
        f"watchdog_llm_timeout_s should tighten with the smaller brain; "
        f"got {perf['watchdog_llm_timeout_s']}"
    )
    lc = cfg["latency_controller"]
    assert lc["quick_budget_ms"] <= 1100, (
        f"quick_budget_ms should drop with the faster brain; got "
        f"{lc['quick_budget_ms']}"
    )


# ─────────────────────────────────────────────────────────────────────
# MLXBrain: single-model role resolution
# ─────────────────────────────────────────────────────────────────────


def test_mlx_brain_roles_are_primary_and_fast():
    from brain.mlx_llm import MLXBrain

    assert MLXBrain._ROLES == ("primary", "fast"), (
        f"v3.2 cleanup: only primary + fast roles remain; got {MLXBrain._ROLES!r}"
    )


def test_mlx_brain_path_for_role_is_single_path():
    from brain.mlx_llm import MLXBrain

    cfg = _settings()
    brain = MLXBrain(cfg)
    # Every role -- including a historical "deep" label that callers
    # might still pass -- must resolve to the same on-disk model path.
    assert brain._path_for_role("primary").endswith(_QWEN_DIRNAME)
    assert brain._path_for_role("fast").endswith(_QWEN_DIRNAME)
    assert brain._path_for_role("deep").endswith(_QWEN_DIRNAME)
    assert brain._path_for_role("nonsense").endswith(_QWEN_DIRNAME)


def test_mlx_brain_single_model_path_attribute():
    """The internal state has one path, not three."""
    from brain.mlx_llm import MLXBrain

    cfg = _settings()
    brain = MLXBrain(cfg)
    assert hasattr(brain, "_model_path")
    assert _QWEN_DIRNAME in brain._model_path.lower()
    assert not hasattr(brain, "_fast_path"), (
        "single-model cleanup should have removed _fast_path"
    )
    assert not hasattr(brain, "_deep_path"), (
        "single-model cleanup should have removed _deep_path"
    )


def test_mlx_brain_dicts_only_have_two_role_slots():
    """Per-role state dicts drop the 'deep' slot after the single-model
    cleanup."""
    from brain.mlx_llm import MLXBrain

    cfg = _settings()
    brain = MLXBrain(cfg)
    for d in (
        brain._models,
        brain._tokenizers,
        brain._fingerprints,
        brain._loaded_roles,
        brain._load_failed,
        brain._role_last_used,
    ):
        assert set(d.keys()) == {"primary", "fast"}, (
            f"Per-role dict keys drifted: {set(d.keys())}"
        )


# ─────────────────────────────────────────────────────────────────────
# Back-compat: legacy settings.json still boots
# ─────────────────────────────────────────────────────────────────────


def test_mlx_brain_accepts_legacy_mlx_primary_model():
    """A pre-v3.2 config with only ``mlx_primary_model`` must still
    resolve to a usable path -- we don't break users mid-upgrade."""
    from brain.mlx_llm import MLXBrain

    cfg = {"brain": {"mlx_primary_model": f"models/{_QWEN_DIRNAME}"}}
    brain = MLXBrain(cfg)
    assert brain._model_path.endswith(_QWEN_DIRNAME)


def test_mlx_brain_accepts_legacy_model_path():
    """Even older configs (GGUF era) used ``model_path`` -- still honoured."""
    from brain.mlx_llm import MLXBrain

    cfg = {"brain": {"model_path": f"models/{_QWEN_DIRNAME}"}}
    brain = MLXBrain(cfg)
    assert brain._model_path.endswith(_QWEN_DIRNAME)


def test_mlx_brain_falls_back_to_default_when_all_keys_missing():
    """No keys at all -> the built-in default path, so a stubbed
    config dict still yields a functional MLXBrain instance."""
    from brain.mlx_llm import MLXBrain

    brain = MLXBrain({"brain": {}})
    assert brain._model_path.endswith(_QWEN_DIRNAME)


# ─────────────────────────────────────────────────────────────────────
# max_tokens override caps -- tuned to prevent CoT leakage on voice turns
# ─────────────────────────────────────────────────────────────────────


def test_max_tokens_override_short():
    from cursor_bridge.local_brain_controller import LocalBrainController
    from core.query_policy import ResponseMode

    cap = LocalBrainController._max_tokens_override(
        response_mode=ResponseMode.SHORT,
        budget_tier="command",
        requested_tier="command",
    )
    assert cap == 96


def test_max_tokens_override_simple():
    from cursor_bridge.local_brain_controller import LocalBrainController
    from core.query_policy import ResponseMode

    cap = LocalBrainController._max_tokens_override(
        response_mode=ResponseMode.NORMAL,
        budget_tier="simple",
        requested_tier="simple",
    )
    assert cap == 128


def test_max_tokens_override_detail():
    from cursor_bridge.local_brain_controller import LocalBrainController
    from core.query_policy import ResponseMode

    cap = LocalBrainController._max_tokens_override(
        response_mode=ResponseMode.DETAIL,
        budget_tier="complex",
        requested_tier="complex",
    )
    assert cap == 256


def test_max_tokens_override_normal_default():
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
    aren't truncated. Voice paths always carry a tier != 'creative'."""
    from cursor_bridge.local_brain_controller import LocalBrainController
    from core.query_policy import ResponseMode

    cap = LocalBrainController._max_tokens_override(
        response_mode=ResponseMode.REPORT,
        budget_tier="creative",
        requested_tier="creative",
    )
    assert cap is None
