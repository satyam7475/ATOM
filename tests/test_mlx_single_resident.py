"""Sprint Ω.7 (Apr 26 2026): single-resident invariant tests.

The contract is simple but the bug it prevents is expensive: on a 16 GB
Apple Silicon laptop, two MLX chat models (e.g. 4B + 8B) co-resident
will spike unified memory past the thermal headroom and either trigger
sustained throttling or push other warm caches (embeddings, persona KV)
out of RAM. ATOM keeps EXACTLY ONE chat-model weight bundle in memory
at a time when ``brain.single_resident`` is true.

These tests deliberately use synthetic stand-in objects in place of
real MLX tensors -- the goal is to lock the *eviction policy*, not to
exercise mlx-lm. The invariants:

1. When the requested role's path differs from a sibling's loaded
   path, the sibling is unloaded *before* the new load.
2. When the requested role's path matches the sibling's loaded path
   (single-model profile), no eviction happens -- tensors are aliased.
3. Speculative decoding refuses the draft load while single_resident
   is on, regardless of the speculative_decoding.enabled flag.
4. ``preload(load_all=True)`` honours the invariant and warms only
   the requested role.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


_ATOM_ROOT = Path(__file__).resolve().parents[1]


def _config_with(brain_overrides: dict | None = None) -> dict:
    raw = (_ATOM_ROOT / "config" / "settings.json").read_text(encoding="utf-8")
    cfg = json.loads(raw)
    if brain_overrides:
        cfg.setdefault("brain", {}).update(brain_overrides)
    return cfg


def test_settings_has_single_resident_enabled() -> None:
    """The shipped config opts in to single-resident on the M5."""
    cfg = _config_with()
    assert cfg["brain"].get("single_resident") is True, (
        "config/settings.json must ship with brain.single_resident=true "
        "to keep one chat model in RAM at a time"
    )


def test_settings_disables_speculative_decoding() -> None:
    """Speculative decoding requires target+draft co-resident, which is
    the exact pattern single_resident exists to prevent. Ship with
    speculative_decoding.enabled=false."""
    cfg = _config_with()
    spec = cfg["brain"].get("speculative_decoding", {})
    assert spec.get("enabled") is False, (
        "speculative_decoding.enabled must be false on the shipped "
        "config -- it is structurally incompatible with single_resident"
    )


def test_settings_disables_whisper_confirm() -> None:
    """WhisperKit (CoreML/ANE) is the live STT engine. The
    faster-whisper tiny confirmer is a redundant second decoder that
    competes for CPU/RAM with no quality benefit on a healthy
    WhisperKit run."""
    cfg = _config_with()
    wc = cfg.get("stt", {}).get("whisper_confirm", {})
    assert wc.get("enabled") is False, (
        "stt.whisper_confirm.enabled must be false -- "
        "WhisperKit is the live engine, second-pass decoder is redundant"
    )


def test_evict_other_roles_skips_aliased_paths() -> None:
    """When two roles share the same loaded path (the shipped
    single-model profile), the eviction helper must NOT unload either
    role -- doing so would poison the alias."""
    from brain.mlx_llm import MLXBrain

    cfg = _config_with({"single_resident": True})
    brain = MLXBrain(cfg)

    sentinel = object()
    same_path = brain._model_path
    for role in brain._ROLES:
        brain._models[role] = sentinel
        brain._tokenizers[role] = sentinel
        brain._fingerprints[role] = same_path
        brain._loaded_roles[role] = True

    evicted = brain._evict_other_roles_unlocked("primary", same_path)
    assert evicted == 0, "aliased same-path siblings must not be evicted"
    for role in brain._ROLES:
        assert brain._loaded_roles[role] is True
        assert brain._models[role] is sentinel


def test_evict_other_roles_drops_divergent_paths() -> None:
    """When the requested role's path differs from a sibling's loaded
    path, the sibling must be unloaded so only the requested tier is
    resident -- the core single-resident contract."""
    from brain.mlx_llm import MLXBrain

    cfg = _config_with({"single_resident": True})
    brain = MLXBrain(cfg)

    sentinel = object()
    other_path = "models/qwen3-8b-4bit"
    requested_path = "models/qwen3-4b-instruct-4bit"

    brain._models["fast"] = sentinel
    brain._tokenizers["fast"] = sentinel
    brain._fingerprints["fast"] = other_path
    brain._loaded_roles["fast"] = True

    evicted = brain._evict_other_roles_unlocked("primary", requested_path)
    assert evicted == 1
    assert brain._loaded_roles["fast"] is False
    assert brain._models["fast"] is None
    assert brain._fingerprints["fast"] is None


def test_evict_clears_role_prompt_cache() -> None:
    """Per-role prompt KV caches reference the model tensors. When the
    role is evicted, the cache must be dropped too -- otherwise a swap
    back leaves stale references in the LRU."""
    from brain.mlx_llm import MLXBrain

    cfg = _config_with({"single_resident": True})
    brain = MLXBrain(cfg)

    brain._models["fast"] = object()
    brain._fingerprints["fast"] = "models/some-other-path"
    brain._loaded_roles["fast"] = True
    brain._prompt_caches["fast"] = object()

    brain._evict_other_roles_unlocked("primary", brain._model_path)
    assert "fast" not in brain._prompt_caches


def test_speculative_draft_refused_when_single_resident() -> None:
    """The draft load is refused as long as single_resident is true,
    even when speculative_decoding.enabled is also true. This is
    defense in depth -- the shipped config disables speculative, but
    a future operator who flips it on must not silently re-introduce
    a co-resident pair."""
    from brain.mlx_llm import MLXBrain

    cfg = _config_with({
        "single_resident": True,
        "speculative_decoding": {
            "enabled": True,
            "draft_model_path": "models/qwen3-8b-4bit",
            "num_draft_tokens": 3,
        },
    })
    brain = MLXBrain(cfg)
    assert brain._ensure_draft_loaded() is False
    assert brain._draft_load_failed is True
    assert brain._draft_loaded is False


def test_speculative_draft_skipped_when_master_flag_off() -> None:
    """If the master speculative flag is off, the draft must stay
    untouched regardless of single_resident. The skip path returns
    early without flipping ``_draft_load_failed`` so a later flag
    flip would be retried."""
    from brain.mlx_llm import MLXBrain

    cfg = _config_with({
        "single_resident": True,
        "speculative_decoding": {
            "enabled": False,
            "draft_model_path": "models/qwen3-8b-4bit",
            "num_draft_tokens": 3,
        },
    })
    brain = MLXBrain(cfg)
    assert brain._ensure_draft_loaded() is False
    assert brain._draft_load_failed is False


def test_role_switch_min_interval_attribute_default() -> None:
    """The hysteresis floor is configurable but ships with a sensible
    default so the kernel cannot trigger a model swap every turn even
    if single_resident=true is the only flag set."""
    from brain.mlx_llm import MLXBrain

    cfg = _config_with({"single_resident": True})
    brain = MLXBrain(cfg)
    assert brain._role_switch_min_interval > 0.0
    assert brain._role_switch_min_interval >= 5.0
