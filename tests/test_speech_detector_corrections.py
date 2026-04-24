"""Regression suite for ``voice.speech_detector.correct_text``.

The atom_log.txt boot dated 2026-04-25 captured a music command being
mangled by an unconditional STT correction:

    L320  STT promoted to final: 'Play some music for me item'
    L325  Input normalized:    'play some music for me item' -> 'play some music for me atom'
    L327  Intent: ... -> fallback/LLM   (the dedicated music intent never matched)

Removing the bare ``("item", "atom")`` substitution and replacing it
with a head-anchored wake-corrector restores both the wake misrecognition
fix AND the integrity of body-position tokens.
"""
from __future__ import annotations

from voice.speech_detector import (
    _correct_leading_wake_token,
    correct_text,
)


def test_trailing_item_preserved_in_music_command() -> None:
    """The exact regression from atom_log.txt L320-L325. Trailing 'item'
    must reach the router untouched so dedicated music intents (added in
    Phase F2) can match the verb-object shape."""
    out = correct_text("Play some music for me item")
    assert "atom" not in out, (
        f"trailing 'item' must NOT become 'atom' in body position: {out!r}"
    )
    assert out == "play some music for me item"


def test_trailing_item_preserved_in_arbitrary_command() -> None:
    out = correct_text("Add this item to my list")
    assert out == "add this item to my list"


def test_leading_item_promoted_to_atom() -> None:
    """First-token confusables ARE wake-context — promote them. This
    preserves the original wake-misrecognition fix without breaking body
    occurrences."""
    out = correct_text("item what time is it")
    assert out.startswith("atom "), out


def test_bare_item_promoted_to_atom() -> None:
    """A single-token 'item' utterance is almost certainly a bare wake."""
    out = correct_text("item")
    assert out == "atom"


def test_leading_atum_promoted_to_atom() -> None:
    out = _correct_leading_wake_token("atum, what's the time")
    assert out.startswith("atom"), out


def test_body_atum_preserved() -> None:
    out = _correct_leading_wake_token("ping the atum service")
    assert "atum" in out, out


def test_existing_wake_phrase_corrections_unchanged() -> None:
    """Phrase-level corrections like ('hey item', 'hey atom') still work
    because they are explicit two-token rewrites, not bare-token swaps."""
    assert correct_text("hey item") == "hey atom"
    assert correct_text("hello item") == "hello atom"
    assert correct_text("good morning item") == "good morning atom"


def test_existing_unrelated_corrections_intact() -> None:
    assert correct_text("open the room") == "open chrome"
    assert correct_text("hey adam") == "hey atom"
