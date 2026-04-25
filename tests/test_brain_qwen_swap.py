"""Single-model brain regression tests -- Qwen3 family.

History:
  - v3.3 (2026-04-24): swapped Qwen2.5-7B → Qwen3-4B-Instruct-2507-4bit
    for JARVIS-grade responsiveness on the 16 GB Apple Silicon machines.
  - Sprint Ω (2026-04-25): swapped Qwen3-4B → Qwen3-8B-4bit as the
    primary brain on the M5 (10 GPU cores). The 4B is now the explicit
    fallback (``brain.mlx_model_fallback``) for thermally constrained
    or RAM-limited machines.

Both Qwen3-8B-4bit and Qwen3-4B-Instruct-2507-4bit are accepted as the
configured ``brain.mlx_model``; the loader transparently aliases the
``fast`` role to the same weights as ``primary`` and pins the system
prompt into the KV prefix on warmup.

Live generation is exercised in ``scripts/smoke_metal_warmup.py`` (real
MLX + real torch.mps + real cold-start) so this file stays fast and
import-safe.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
# Sprint Ω: 8B is the production primary on the M5; 4B remains a
# supported fallback for thermal/RAM-constrained machines. Either is
# acceptable as the configured brain.mlx_model.
_QWEN_DIRNAMES = ("qwen3-8b-4bit", "qwen3-4b-instruct-4bit")
_QWEN_DIRNAME = _QWEN_DIRNAMES[0]  # default for back-compat fixtures


def _is_supported_qwen(model_path: str) -> bool:
    needle = model_path.lower()
    return any(name in needle for name in _QWEN_DIRNAMES)


def _settings() -> dict:
    with (REPO_ROOT / "config" / "settings.json").open() as fh:
        return json.load(fh)


# ─────────────────────────────────────────────────────────────────────
# Config: single mlx_model key
# ─────────────────────────────────────────────────────────────────────


def test_brain_mlx_model_is_supported_qwen3():
    cfg = _settings()
    model = cfg["brain"]["mlx_model"]
    assert _is_supported_qwen(model), (
        f"Brain must load a supported Qwen3 variant ({_QWEN_DIRNAMES!r}). "
        f"Sprint Ω promoted Qwen3-8B-4bit to primary; Qwen3-4B-Instruct-2507-4bit "
        f"remains an accepted fallback. Got {model!r}."
    )


def test_brain_mlx_model_fallback_is_supported_qwen3():
    """Sprint Ω introduced an explicit fallback brain for thermally
    constrained machines. If present, it must also be a supported
    Qwen3 variant -- not a mismatched legacy model."""
    cfg = _settings()
    fb = cfg.get("brain", {}).get("mlx_model_fallback")
    if not fb:
        pytest.skip("brain.mlx_model_fallback not configured (optional)")
    assert _is_supported_qwen(fb), (
        f"brain.mlx_model_fallback must be a supported Qwen3 variant. "
        f"Got {fb!r}."
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
    assert p.is_dir(), f"Qwen3 model directory not found at {p}"
    safetensors = list(p.glob("*.safetensors"))
    assert safetensors, f"No .safetensors weights found in {p}"
    assert (p / "tokenizer.json").exists(), f"Missing tokenizer.json in {p}"
    assert (p / "config.json").exists(), f"Missing config.json in {p}"


def test_brain_watchdog_budgets_match_qwen3_class():
    """Watchdog/latency budgets stay within the Qwen3 envelope.

    Sprint Ω's 8B sits between the 4B's <1.5 s first-token and the
    legacy 7B's 3-4 s, so the brain timeout + LLM watchdog need to
    accommodate the 8B's slightly heavier prefill while still staying
    well below the 7B-era settings.
    """
    cfg = _settings()
    assert cfg["brain"]["timeout_seconds"] <= 30, (
        f"Qwen3 (4B or 8B) should not exceed a 30 s brain timeout; got "
        f"{cfg['brain']['timeout_seconds']}"
    )
    perf = cfg["performance"]
    assert perf["watchdog_llm_timeout_s"] <= 20, (
        f"watchdog_llm_timeout_s should stay well below 7B-era values; "
        f"got {perf['watchdog_llm_timeout_s']}"
    )
    lc = cfg["latency_controller"]
    assert lc["quick_budget_ms"] <= 1500, (
        f"quick_budget_ms should remain in the snappy range; got "
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
    primary = brain._path_for_role("primary")
    assert _is_supported_qwen(primary), primary
    assert brain._path_for_role("fast") == primary
    assert brain._path_for_role("deep") == primary
    assert brain._path_for_role("nonsense") == primary


def test_mlx_brain_single_model_path_attribute():
    """The internal state has one path, not three."""
    from brain.mlx_llm import MLXBrain

    cfg = _settings()
    brain = MLXBrain(cfg)
    assert hasattr(brain, "_model_path")
    assert _is_supported_qwen(brain._model_path)
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
    assert _is_supported_qwen(brain._model_path), brain._model_path


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
