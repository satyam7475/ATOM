"""Regression suite for ``ConversationManager.resolve_pronouns``.

atom_log.txt 2026-04-25 captured a high-impact pronoun-resolution leak:

    L483  STT promoted to final: 'How do you know that'
    L488  Pronoun resolved:      'how do that' -> 'how do you chicken wings'
    L504  TTS spoke:             "I'm not a chicken wing expert..."

The resolver took the user's perfectly valid query, lost "you know" to a
filler regex, then substituted the bare "that" pronoun with a polluted
``_last_entity`` ("you chicken wings") carried over from an earlier turn.

The new net-token-count guard must reject any resolution that injects
more than ``_MAX_RESOLVED_NEW_TOKENS`` new content tokens not present
in the user's original query.
"""
from __future__ import annotations

from core.router.conversation_manager import ConversationManager


def _seed_entity(cm: ConversationManager, entity: str) -> None:
    """Bypass extract_entity heuristics; pin a literal _last_entity for
    tests that need a specific multi-word context."""
    cm._last_entity = entity


def test_pronoun_resolver_blocks_multiword_entity_dump() -> None:
    cm = ConversationManager()
    _seed_entity(cm, "you chicken wings")

    out = cm.resolve_pronouns("how do that")

    assert out == "how do that", (
        f"Resolver must reject multi-word entity dump that would have "
        f"produced 'how do you chicken wings'. Got: {out!r}"
    )


def test_pronoun_resolver_allows_short_entity() -> None:
    cm = ConversationManager()
    _seed_entity(cm, "newton")

    out = cm.resolve_pronouns("explain that")

    assert out == "explain newton"


def test_pronoun_resolver_allows_two_token_entity() -> None:
    cm = ConversationManager()
    _seed_entity(cm, "machine learning")

    out = cm.resolve_pronouns("explain that")

    assert out == "explain machine learning"


def test_pronoun_resolver_blocks_three_token_entity() -> None:
    cm = ConversationManager()
    _seed_entity(cm, "neural network architecture overview")

    out = cm.resolve_pronouns("show that")

    assert out == "show that", (
        f"Four-token entity injection must be rejected. Got: {out!r}"
    )


def test_pronoun_resolver_no_op_without_pronoun() -> None:
    cm = ConversationManager()
    _seed_entity(cm, "anything")

    out = cm.resolve_pronouns("what is the capital of france")
    assert out == "what is the capital of france"


def test_pronoun_resolver_no_op_with_concrete_noun() -> None:
    """Existing has_noun guard: query already has a real noun, so no
    pronoun substitution should fire even if a pronoun is present."""
    cm = ConversationManager()
    _seed_entity(cm, "newton")

    out = cm.resolve_pronouns("tell me about quantum that physics")
    assert out == "tell me about quantum that physics"


def test_pronoun_resolver_no_op_without_entity() -> None:
    cm = ConversationManager()
    out = cm.resolve_pronouns("what is that")
    assert out == "what is that"
