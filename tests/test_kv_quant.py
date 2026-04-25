"""ATOM -- regression tests for KV cache quantisation (Sprint C5).

Pins three behaviours so we don't lose the ~10-15% throughput win
from ``kv_bits=8`` on the FAST/DEEP MLX paths:

1. ``MLXBrain.__init__`` reads ``kv_bits`` / ``kv_group_size`` /
   ``kv_quant_warmup_tokens`` from ``config["brain"]`` with
   sensible defaults.
2. Out-of-range ``kv_bits`` (anything other than 0/4/8) is clamped
   to 0 with a warning -- mlx-lm only supports those values.
3. The streaming inner loop forwards the params to
   ``stream_generate(... kv_bits=, kv_group_size=,
   quantized_kv_start=)`` ONLY when quantisation is enabled
   (``kv_bits != 0``).
4. Schema validation accepts the new keys and rejects bad values.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from brain import mlx_llm


# ── __init__ defaults & validation ───────────────────────────────


def _make_bare_brain(brain_cfg: dict) -> mlx_llm.MLXBrain:
    """Construct an MLXBrain bypassing model load."""
    return mlx_llm.MLXBrain({"brain": brain_cfg})


def test_default_kv_bits_is_8() -> None:
    brain = _make_bare_brain({})
    assert brain._kv_bits == 8
    assert brain._kv_group_size == 64
    assert brain._kv_quant_warmup == 512


def test_explicit_kv_bits_zero_disables_quantisation() -> None:
    brain = _make_bare_brain({"kv_bits": 0})
    assert brain._kv_bits == 0


def test_invalid_kv_bits_falls_back_to_zero(caplog) -> None:
    """kv_bits not in (0, 4, 8) must log a warning and fall back."""
    with caplog.at_level("WARNING", logger="atom.brain.mlx"):
        brain = _make_bare_brain({"kv_bits": 5})
    assert brain._kv_bits == 0
    assert any("kv_bits" in rec.message for rec in caplog.records), (
        f"Expected a warning about kv_bits=5; got: {caplog.records!r}"
    )


def test_kv_bits_4_is_accepted() -> None:
    brain = _make_bare_brain({"kv_bits": 4})
    assert brain._kv_bits == 4


def test_kv_group_size_and_warmup_tokens_overridable() -> None:
    brain = _make_bare_brain({
        "kv_bits": 8,
        "kv_group_size": 32,
        "kv_quant_warmup_tokens": 256,
    })
    assert brain._kv_group_size == 32
    assert brain._kv_quant_warmup == 256


# ── stream_generate kwargs forwarding ────────────────────────────


def _make_stub_brain_for_stream(kv_bits: int) -> mlx_llm.MLXBrain:
    brain = mlx_llm.MLXBrain.__new__(mlx_llm.MLXBrain)
    brain._abort_generation = 0  # type: ignore[attr-defined]
    brain._kv_bits = kv_bits  # type: ignore[attr-defined]
    brain._kv_group_size = 64  # type: ignore[attr-defined]
    brain._kv_quant_warmup = 512  # type: ignore[attr-defined]
    brain._make_sampler = lambda *a, **k: None  # type: ignore[attr-defined]
    brain._make_logits_processors = lambda *a, **k: None  # type: ignore[attr-defined]
    brain._prepare_prompt_cache = lambda *a, **k: (None, "p", 1)  # type: ignore[attr-defined]
    brain._role_last_used = {"fast": 0.0}  # type: ignore[attr-defined]
    brain._perf_lock = threading.Lock()  # type: ignore[attr-defined]
    brain._role_perf = {}  # type: ignore[attr-defined]
    return brain


def _drive_inner_loop(brain, eff, monkeypatch) -> dict:
    captured: dict = {}

    def _fake_stream_generate(model, tokenizer, prompt, **kwargs):
        captured.update(kwargs)
        return iter(())

    monkeypatch.setattr(mlx_llm, "stream_generate", _fake_stream_generate)
    monkeypatch.setattr(mlx_llm, "_HAS_MLX", False)

    try:
        brain._generate_sync_streaming_inner(
            "fast", eff, MagicMock(), MagicMock(), "prompt",
            on_token=None,
            max_tokens_override=None,
            extra_stop_sequences=None,
        )
    except AttributeError:
        # downstream perf-bookkeeping touches attrs not stubbed here
        pass
    return captured


def test_stream_generate_receives_kv_quant_params_when_enabled(
    monkeypatch,
) -> None:
    brain = _make_stub_brain_for_stream(kv_bits=8)
    eff = {
        "model_role": "fast",
        "profile": "fast",
        "max_tokens": 96,
        "temperature": 0.7,
        "top_p": 0.9,
        "repeat_penalty": 1.0,
        "extra_stop_sequences": [],
    }
    captured = _drive_inner_loop(brain, eff, monkeypatch)

    assert captured.get("kv_bits") == 8
    assert captured.get("kv_group_size") == 64
    assert captured.get("quantized_kv_start") == 512


def test_stream_generate_skips_kv_quant_params_when_disabled(
    monkeypatch,
) -> None:
    brain = _make_stub_brain_for_stream(kv_bits=0)
    eff = {
        "model_role": "fast",
        "profile": "fast",
        "max_tokens": 96,
        "temperature": 0.7,
        "top_p": 0.9,
        "repeat_penalty": 1.0,
        "extra_stop_sequences": [],
    }
    captured = _drive_inner_loop(brain, eff, monkeypatch)

    assert "kv_bits" not in captured
    assert "kv_group_size" not in captured
    assert "quantized_kv_start" not in captured


def test_stream_generate_kv_bits_4_path(monkeypatch) -> None:
    brain = _make_stub_brain_for_stream(kv_bits=4)
    brain._kv_quant_warmup = 128  # type: ignore[attr-defined]
    eff = {
        "model_role": "fast",
        "profile": "fast",
        "max_tokens": 96,
        "temperature": 0.7,
        "top_p": 0.9,
        "repeat_penalty": 1.0,
        "extra_stop_sequences": [],
    }
    captured = _drive_inner_loop(brain, eff, monkeypatch)

    assert captured.get("kv_bits") == 4
    assert captured.get("quantized_kv_start") == 128


# ── schema acceptance ───────────────────────────────────────────


def test_config_schema_accepts_new_kv_keys() -> None:
    import json
    from pathlib import Path
    from core.config_schema import validate_config

    cfg = json.loads(Path("config/settings.json").read_text())
    errors = validate_config(cfg) or []
    assert errors == [], f"Live config has schema errors: {errors!r}"


def test_config_schema_rejects_bad_kv_bits_value() -> None:
    from core.config_schema import validate_config

    cfg = {
        "brain": {
            "enabled": True,
            "mlx_model": "models/qwen3-4b-instruct-4bit",
            "n_ctx": 4096,
            "n_threads": 4,
            "n_gpu_layers": -1,
            "n_batch": 256,
            "max_tokens": 320,
            "temperature": 0.6,
            "top_p": 0.85,
            "repeat_penalty": 1.1,
            "timeout_seconds": 20,
            "kv_bits": 5,  # invalid -- enum is [0, 4, 8]
        },
    }
    errors = validate_config(cfg) or []
    assert any("kv_bits" in str(e) for e in errors), (
        "Schema must reject kv_bits=5; got errors: " + repr(errors)
    )
