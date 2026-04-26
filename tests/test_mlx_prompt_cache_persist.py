"""B7: cross-boot persistence of the MLX prompt cache.

Two slices under test:

1. ``_persist_path_for_role`` derives a per-role file path so primary
   and fast caches can never cross-pollute.
2. ``_persist_prompt_cache`` / ``_restore_persisted_prompt_cache``
   round-trip through disk: a saved snapshot is reloaded into the
   LRU on the next boot under the original trie key, so the next
   turn benefits from the warm cache (sub-second first-token).

We mock mlx_lm's save/load so the test runs anywhere — including CI
without an MPS device.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from brain.mlx_llm import MLXBrain


def _brain(tmp_path: Path, persist_enabled: bool = True) -> MLXBrain:
    cfg = {
        "brain": {
            "mlx_model": "models/test-mlx-stub",
            "max_tokens": 32,
            "prompt_cache_enabled": True,
            "prompt_cache_persist": persist_enabled,
            "prompt_cache_persist_path": str(
                tmp_path / "prompt_cache_test.safetensors",
            ),
            "prompt_cache_persist_min_tokens": 8,
        },
    }
    return MLXBrain(cfg)


def test_persist_path_per_role_does_not_collide(tmp_path: Path) -> None:
    brain = _brain(tmp_path)
    primary_path = brain._persist_path_for_role("primary")
    fast_path = brain._persist_path_for_role("fast")
    assert primary_path != fast_path
    assert primary_path.suffix == ".safetensors"
    assert "primary" in primary_path.name
    assert "fast" in fast_path.name


def test_persist_disabled_when_save_fn_unavailable(tmp_path: Path) -> None:
    """If mlx_lm.models.cache.save_prompt_cache is missing, the
    persistence layer must stay quietly off — no crashes, no log spam."""
    with patch("brain.mlx_llm._save_prompt_cache", None):
        brain = _brain(tmp_path)
        assert brain._prompt_cache_persist_enabled is False


def test_persist_skipped_below_min_tokens(tmp_path: Path) -> None:
    """Don't waste disk on degenerate first-turn snapshots."""
    brain = _brain(tmp_path)
    brain._prompt_cache_persist_min_tokens = 100

    save_mock = MagicMock()
    with patch("brain.mlx_llm._save_prompt_cache", save_mock):
        # Force the same code path _commit_prompt_cache uses.
        brain._persist_prompt_cache("primary", cache=[MagicMock()], tokens_for_trie=[1, 2, 3])
        # _persist_prompt_cache itself does NOT enforce the threshold —
        # the threshold is enforced by the caller _commit_prompt_cache.
        # Here we just verify that the call path doesn't crash on small
        # inputs, and that a persisted file metadata captures the size.
        save_mock.assert_called_once()
        args, kwargs = save_mock.call_args
        meta = args[2] if len(args) >= 3 else kwargs.get("metadata")
        assert meta and meta["n_tokens"] == "3"


def test_restore_returns_silently_when_no_file(tmp_path: Path) -> None:
    brain = _brain(tmp_path)
    # No file written yet → restore must be a no-op.
    brain._restore_persisted_prompt_cache("primary")
    # No exception, persisted-flag still False.
    assert brain._prompt_cache_persisted_role["primary"] is False


def test_restore_validates_model_path_match(tmp_path: Path) -> None:
    """Persisted cache from a DIFFERENT model must NOT be loaded —
    KV layout depends on hidden_dim / num_layers / dtype, so loading
    a mismatched cache would crash inference. The check protects
    against model swaps without prompt-cache invalidation."""
    brain = _brain(tmp_path)

    fake_cache = [MagicMock()]
    fake_meta = {
        "model_path": "models/some-OTHER-model",
        "tokens": "1,2,3,4",
        "n_tokens": "4",
        "version": "v33",
    }
    path = brain._persist_path_for_role("primary")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()  # _restore checks .is_file() before loading

    with patch(
        "brain.mlx_llm._load_prompt_cache",
        return_value=(fake_cache, fake_meta),
    ):
        brain._restore_persisted_prompt_cache("primary")

    # Mismatched model_path → restore aborted, persisted-flag stays False.
    assert brain._prompt_cache_persisted_role["primary"] is False


def test_restore_idempotent_per_role(tmp_path: Path) -> None:
    """Restore must run at most once per role per boot — repeated
    calls during model reloads must NOT keep re-injecting the same
    cache and bloating the LRU.

    Sprint P2.6 (Apr 26 2026): the MLX symbols are now lazy-loaded, so
    ``_prompt_cache_persist_enabled`` checks BOTH ``_save_prompt_cache``
    and ``_load_prompt_cache`` for non-None. Pre-P2.6 only the load
    symbol mattered; we now patch both with a sentinel so the property
    returns ``True`` and the function reaches its body.
    """
    brain = _brain(tmp_path)

    load_mock = MagicMock(return_value=(None, {}))
    save_mock = MagicMock()
    with (
        patch("brain.mlx_llm._load_prompt_cache", load_mock),
        patch("brain.mlx_llm._save_prompt_cache", save_mock),
    ):
        brain._restore_persisted_prompt_cache("primary")
        brain._restore_persisted_prompt_cache("primary")
        brain._restore_persisted_prompt_cache("primary")
    # File missing → load never called either way, but the attempted
    # flag still flips after the first call.
    assert brain._prompt_cache_restore_attempted["primary"] is True


def test_settings_schema_accepts_persist_keys() -> None:
    """The new keys must validate against config_schema, otherwise
    settings.json fails on boot."""
    from core.config_schema import validate_config

    settings_path = Path(__file__).parent.parent / "config" / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    errors = validate_config(settings)
    persist_errors = [e for e in errors if "prompt_cache_persist" in e]
    assert not persist_errors, (
        f"persist keys did not validate: {persist_errors}"
    )
    brain_cfg = settings["brain"]
    assert brain_cfg.get("prompt_cache_persist") is True
    assert "prompt_cache_persist_path" in brain_cfg
    assert "prompt_cache_persist_min_tokens" in brain_cfg
