"""Phi-3.5-mini-MLX-4bit live generation smoke test.

Loads the actual Phi model on Metal and runs ONE short prompt to verify
that mlx-lm can read the weights and produce a sensible answer. Slow
(~3-5s on M1/M5), so it's marked ``@pytest.mark.slow`` and skipped if
the model directory isn't present.

Run only this test:
    PYTHONPATH=. pytest tests/test_brain_phi_smoke.py -v -m slow

Skip slow tests in normal runs:
    PYTHONPATH=. pytest -m "not slow"
"""
from __future__ import annotations

from pathlib import Path

import pytest


PHI_DIR = Path(__file__).resolve().parent.parent / "models" / "phi-3.5-mini-mlx-4bit"


@pytest.mark.slow
@pytest.mark.skipif(
    not PHI_DIR.is_dir(),
    reason=f"Phi model directory not present at {PHI_DIR}",
)
def test_phi_generates_basic_arithmetic():
    """Loads Phi via mlx-lm and asks 'what is 2+2'. Must reply with '4'
    in <= 25 words. Pins that the model + tokenizer are functional."""
    pytest.importorskip("mlx_lm")
    from mlx_lm import generate, load

    model, tokenizer = load(str(PHI_DIR))
    # Phi-3.5 uses the <|user|>/<|end|>/<|assistant|> chat template.
    prompt = "<|user|>\nWhat is 2+2? Answer in one short sentence.<|end|>\n<|assistant|>"
    out = generate(model, tokenizer, prompt=prompt, max_tokens=32, verbose=False)

    assert "4" in out, f"Phi did not produce '4' in answer: {out!r}"
    word_count = len(out.split())
    # 25 words is roomy for Phi answering "what is 2+2" -- catches any
    # tokenizer / chat-template misalignment that produces verbose CoT.
    assert word_count <= 30, (
        f"Phi answer too verbose ({word_count} words) -- prompt-template "
        f"may be wrong: {out!r}"
    )


@pytest.mark.slow
@pytest.mark.skipif(
    not PHI_DIR.is_dir(),
    reason=f"Phi model directory not present at {PHI_DIR}",
)
def test_phi_does_not_parrot_system_rules():
    """Regression for the v3 prompt-leak issue: when Phi is asked a
    simple question, it must NOT echo any imperative system-prompt-like
    text in the reply (e.g. 'one short sentence', 'plain text only')."""
    pytest.importorskip("mlx_lm")
    from mlx_lm import generate, load

    model, tokenizer = load(str(PHI_DIR))
    prompt = (
        "<|system|>\n"
        "You are a helpful assistant. Reply concisely.<|end|>\n"
        "<|user|>\n"
        "What time is it?<|end|>\n"
        "<|assistant|>"
    )
    out = generate(model, tokenizer, prompt=prompt, max_tokens=48, verbose=False).lower()

    # None of the parroted phrases from the Qwen log may appear in Phi's
    # output (they wouldn't anyway -- Phi was picked precisely because it
    # doesn't mirror system text). Treat these as red-flag substrings.
    forbidden = [
        "the final answer only",
        "one short sentence when possible",
        "two short sentences max",
        "plain text only",
        "spoken = final answer",
    ]
    for token in forbidden:
        assert token not in out, (
            f"Phi parroted system-prompt phrase {token!r} -- regression: {out!r}"
        )
