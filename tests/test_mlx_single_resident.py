"""Sprint Ω.7 (Apr 26 2026) → Ω.10 (Apr 27 2026): multi-tier eviction tests.

History:
  * Ω.7 introduced ``brain.single_resident=true`` as a hard invariant
    that kept exactly one chat-model bundle in unified memory at a
    time. That made sense when "primary" and "fast" pointed at the
    same 4B weights -- aliasing was free, swapping was expensive.
  * Ω.10 (this sprint) flipped ``single_resident`` to ``false`` after
    we proved 0.6B (ultra) + 4B (primary/fast) co-resident fits the
    16 GB / 24 GB M-class budget *and* unlocks ~400 ms wins on the
    "quick reply" path. The eviction policy itself is unchanged --
    when an operator opts back into single_resident, the helpers in
    ``MLXBrain`` still drop divergent siblings before loading the
    requested role.

These tests deliberately use synthetic stand-in objects in place of
real MLX tensors -- the goal is to lock the *eviction policy*, not to
exercise mlx-lm. The invariants on the *shipped* config (Ω.10):

1. ``brain.single_resident`` ships as ``false`` -- ultra (0.6B) and
   primary (4B) live side-by-side so the kernel can route quick
   replies onto the small brain without a swap penalty.
2. ``brain.speculative_decoding.enabled`` ships as ``false``. The
   shipped Qwen3 0.6B/4B pair regressed sustained tok/s in the live
   audit (10.3 wps vs 17.5 wps baseline), so we keep the feature off
   until a higher-acceptance draft is found.
3. ``stt.whisper_confirm.enabled`` stays ``false`` -- WhisperKit is
   the live STT engine; the second-pass confirmer is redundant CPU.

Plus the eviction policy invariants (still active when an operator
flips ``single_resident=true`` for a smaller-RAM rig):

4. When the requested role's path differs from a sibling's loaded
   path, the sibling is unloaded before the new load.
5. When the requested role's path matches the sibling's loaded path,
   no eviction happens -- tensors are aliased.
6. Per-role prompt caches are dropped along with the role's weights
   (otherwise the LRU would hold dangling references).
7. The role-switch hysteresis floor stays > 0 so we cannot thrash a
   swap every turn.
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


def test_settings_disables_single_resident_for_multi_tier() -> None:
    """Sprint Ω.10 (Apr 27 2026): the shipped config now ships with
    ``single_resident=false`` so the 0.6B "ultra" tier can stay warm
    next to the 4B primary/fast model. The eviction helpers still
    work when an operator flips it back to ``true`` (see the policy
    tests below) — this guards the *shipped* default for the M5 rig.
    """
    cfg = _config_with()
    assert cfg["brain"].get("single_resident") is False, (
        "config/settings.json must ship with brain.single_resident=false "
        "(Sprint Ω.10) so ultra (0.6B) + primary/fast (4B) can be co-resident"
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


def test_speculative_draft_refuses_missing_path() -> None:
    """Sprint Ω.10 (Apr 27 2026): the legacy ``single_resident`` block
    that structurally refused a draft load was removed (the multi-tier
    refactor needs ultra + primary/fast co-resident, so co-residency
    of target+draft is no longer banned by definition). The draft
    loader must still refuse cleanly when the configured draft path
    does not exist on disk — that's the layer of safety we keep
    asserting so a typo in ``draft_model_path`` flips the failure
    flag instead of silently 500-ing on the first generate.
    """
    from brain.mlx_llm import MLXBrain

    cfg = _config_with({
        "single_resident": True,
        "speculative_decoding": {
            "enabled": True,
            "draft_model_path": "models/this-path-definitely-does-not-exist",
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
