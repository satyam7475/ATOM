"""Qwen2.5-7B-Instruct-MLX-4bit live generation smoke test.

Loads the actual model on Metal and runs ONE short prompt to verify
that mlx-lm can read the weights and produce a sensible answer. Slow
(~5-8s on M5 first call, ~1-2s warm), so it's marked ``@pytest.mark.slow``
and skipped if the model directory isn't present.

Run only this test:
    PYTHONPATH=. pytest tests/test_brain_qwen_smoke.py -v -m slow

Skip slow tests in normal runs:
    PYTHONPATH=. pytest -m "not slow"
"""
from __future__ import annotations

from pathlib import Path

import pytest


QWEN_DIR = (
    Path(__file__).resolve().parent.parent / "models" / "qwen2.5-7b-instruct-4bit"
)


@pytest.mark.slow
@pytest.mark.skipif(
    not QWEN_DIR.is_dir(),
    reason=f"Qwen2.5-7B-Instruct model directory not present at {QWEN_DIR}",
)
def test_qwen_generates_basic_arithmetic():
    """Loads Qwen via mlx-lm and asks 'what is 2+2'. Must reply with '4'
    in <= 40 words. Pins that the model + tokenizer are functional."""
    pytest.importorskip("mlx_lm")
    from mlx_lm import generate, load

    model, tokenizer = load(str(QWEN_DIR))
    # Qwen2.5 uses the ChatML template. Let the tokenizer apply it so
    # we inherit the exact formatting the training data expects.
    messages = [
        {"role": "user", "content": "What is 2+2? Answer in one short sentence."}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    out = generate(model, tokenizer, prompt=prompt, max_tokens=48, verbose=False)

    assert "4" in out, f"Qwen did not produce '4' in answer: {out!r}"
    word_count = len(out.split())
    assert word_count <= 40, (
        f"Qwen answer too verbose ({word_count} words) -- prompt-template "
        f"may be wrong: {out!r}"
    )


@pytest.mark.slow
@pytest.mark.skipif(
    not QWEN_DIR.is_dir(),
    reason=f"Qwen2.5-7B-Instruct model directory not present at {QWEN_DIR}",
)
def test_qwen_does_not_parrot_system_rules():
    """Regression for the v3 prompt-leak issue: when asked a simple
    question, Qwen must NOT echo imperative system-prompt text
    (e.g. 'one short sentence', 'plain text only')."""
    pytest.importorskip("mlx_lm")
    from mlx_lm import generate, load

    model, tokenizer = load(str(QWEN_DIR))
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Reply concisely."},
        {"role": "user", "content": "What time is it?"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    out = generate(
        model, tokenizer, prompt=prompt, max_tokens=64, verbose=False,
    ).lower()

    forbidden = [
        "the final answer only",
        "one short sentence when possible",
        "two short sentences max",
        "plain text only",
        "spoken = final answer",
    ]
    for token in forbidden:
        assert token not in out, (
            f"Qwen parroted system-prompt phrase {token!r} -- regression: {out!r}"
        )
