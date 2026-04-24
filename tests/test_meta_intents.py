"""Regression tests for ATOM meta-intent confirm/deny matching.

The original `_CONFIRM` regex in core/intent_engine/meta_intents.py only
recognised "yes confirm" word-order, so STT artifacts like "Confirm yes"
or "Confirm confirm yes" (observed verbatim in atom_log.txt L508+L641)
fell through to the LLM and play_youtube never executed. These tests
lock in the confirm-dominant matcher + reverse-order regex variants.
"""
from __future__ import annotations

import pytest

from core.intent_engine.meta_intents import (
    _is_confirm_dominant,
    _is_deny_dominant,
    check,
    quick_match,
)


# Exact STT outputs observed in atom_log.txt that previously failed.
_LIVE_LOG_CONFIRMS = [
    "Confirm yes",
    "Confirm confirm yes",
    "confirm yes",
    "confirm confirm yes",
]

# Classic confirm phrases.
_CLASSIC_CONFIRMS = [
    "yes",
    "yeah",
    "yep",
    "sure",
    "ok",
    "okay",
    "go",
    "go ahead",
    "do it",
    "confirm",
    "haan",
    "ha",
    "han",
    "theek hai",
    "yes please",
    "yes yes",
    "ok ok",
]

# STT-stutter / dictation artifacts that should still be confirms.
_STUTTERED_CONFIRMS = [
    "yes ok",
    "yes sure ok",
    "ok ok yes",
    "yeah yeah sure",
    "haan haan",
    "confirm please",
    "confirm go ahead",
]

# Things that LOOK like they start with "yes" but are real queries.
_NOT_CONFIRMS = [
    "yes can you tell me the weather",
    "yes but actually wait",
    "okay so what about that file",
    "sure but also play music",
    "ok let me think about it more",
]


@pytest.mark.parametrize("text", _LIVE_LOG_CONFIRMS + _CLASSIC_CONFIRMS + _STUTTERED_CONFIRMS)
def test_confirm_phrases_classify_as_confirm(text: str) -> None:
    result = check(text)
    assert result is not None, f"check() returned None for {text!r}"
    assert result.intent == "confirm", (
        f"check() returned intent={result.intent!r} for {text!r}"
    )
    assert quick_match(text) == "confirm", (
        f"quick_match() failed for {text!r}"
    )


@pytest.mark.parametrize("text", _NOT_CONFIRMS)
def test_real_queries_starting_with_yes_are_not_confirms(text: str) -> None:
    """Long-form queries that start with a confirm token must NOT be
    classified as confirm — they belong on the LLM path."""
    result = check(text)
    assert result is None or result.intent != "confirm", (
        f"false-positive confirm on real query {text!r}: {result}"
    )


def test_confirm_dominant_caps_at_5_tokens() -> None:
    """Helper must reject anything longer than 5 tokens, even if every
    word is a confirm vocab word, to prevent runaway false-positives."""
    assert _is_confirm_dominant("yes yes yes yes yes") is True
    assert _is_confirm_dominant("yes yes yes yes yes yes") is False
    assert _is_confirm_dominant("") is False
    assert _is_confirm_dominant("   ") is False


def test_deny_phrases_classify_as_deny() -> None:
    for text in ["no", "cancel", "stop", "nope", "nahi", "no no", "deny",
                 "abort", "cancel it"]:
        result = check(text)
        assert result is not None and result.intent == "deny", (
            f"check() failed for deny {text!r}: {result}"
        )


def test_deny_dominant_caps_at_5_tokens() -> None:
    assert _is_deny_dominant("no no no no no") is True
    assert _is_deny_dominant("no no no no no no") is False
    assert _is_deny_dominant("no I want to keep going actually") is False
