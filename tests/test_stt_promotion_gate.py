"""Regression tests for the short-promotion gate in stt_macos.

Live log evidence (atom_log.txt L427/L461/L587):
    Bare 1-3 char STT outputs like "Ali", "Tom", "ok" arrived as
    isFinal=True from SFSpeechRecognizer and were routed to the LLM,
    burning a turn on a noise spike. The fix adds a centralized
    ``_should_block_short_promotion(text)`` helper that rejects
    short finals unless a wake phrase was matched in the last 3s.

These tests pin the helper and its grace-window logic without booting
the full AVAudioEngine; we instantiate ``MacOSSTTEngine`` with stub
config / state / bus / mic_manager and exercise the promotion-gate
methods directly.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from voice.stt_macos import NativeSTT


class _FakeBus:
    def emit(self, *args: Any, **kwargs: Any) -> None: ...
    def emit_long(self, *args: Any, **kwargs: Any) -> None: ...
    def on(self, *args: Any, **kwargs: Any) -> None: ...


class _FakeState:
    def __init__(self) -> None:
        self.current = None


class _FakeMicManager:
    def acquire(self, *_a: Any, **_kw: Any) -> bool:
        return True

    def release(self, *_a: Any, **_kw: Any) -> None: ...


def _make_engine(stt_config: dict | None = None) -> NativeSTT:
    return NativeSTT(
        bus=_FakeBus(),
        state=_FakeState(),
        config={"stt": stt_config or {"locale": "en-US"}},
        mic_manager=_FakeMicManager(),
        intent_engine=None,
    )


@pytest.fixture
def engine() -> NativeSTT:
    return _make_engine()


# ---------------------------------------------------------------------------
# _should_block_short_promotion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Ali",
        "Tom",
        "uh",
        "ok",
        "no",
        "go",
        "x",
        "hi",
    ],
)
def test_blocks_short_text_without_wake_context(
    engine: NativeSTT, text: str,
) -> None:
    """Short transcripts with no recent wake phrase MUST be blocked."""
    assert engine._last_wake_seen_t == 0.0
    assert engine._should_block_short_promotion(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "what's the weather",
        "play music on YouTube",
        "atom",
        "hello",
        "summary please",
        "stop",
    ],
)
def test_allows_long_or_threshold_text(
    engine: NativeSTT, text: str,
) -> None:
    """Anything at or above the min-alpha threshold passes regardless
    of wake context (longer text is presumed intentional)."""
    assert engine._should_block_short_promotion(text) is False


def test_wake_grace_lets_short_replies_through(
    engine: NativeSTT,
) -> None:
    """Within ``_promotion_wake_grace_s`` after a wake match, even a
    one-syllable reply ("yes", "no") is allowed — the user is clearly
    talking to ATOM."""
    engine._last_wake_seen_t = time.monotonic()

    for short in ("yes", "no", "go", "ok"):
        assert engine._should_block_short_promotion(short) is False


def test_wake_grace_expires(engine: NativeSTT) -> None:
    """Past the grace window, the gate re-engages."""
    engine._last_wake_seen_t = time.monotonic() - (
        engine._promotion_wake_grace_s + 0.5
    )

    assert engine._should_block_short_promotion("ok") is True


def test_empty_text_returns_false(engine: NativeSTT) -> None:
    """Empty / whitespace inputs are not the gate's job — drop happens
    earlier in the trivial-final guard."""
    assert engine._should_block_short_promotion("") is False
    assert engine._should_block_short_promotion("   ") is False


def test_punctuation_only_returns_true(engine: NativeSTT) -> None:
    """Punctuation-only payloads have zero alpha-numeric chars (well
    below the threshold) and no wake context, so the gate blocks them.
    Upstream still has its own trivial-final drop, but defense-in-depth
    is the goal."""
    assert engine._should_block_short_promotion("...") is True


def test_min_alpha_threshold_respected(engine: NativeSTT) -> None:
    """The default threshold is 4 alpha-numeric characters."""
    assert engine._promotion_min_alpha_chars == 4

    # 3-char alphas are blocked.
    assert engine._should_block_short_promotion("Ali") is True
    assert engine._should_block_short_promotion("Tom") is True

    # 4-char alphas pass (defended by min threshold, not wake context).
    assert engine._should_block_short_promotion("Adam") is False
    assert engine._should_block_short_promotion("atom") is False


def test_threshold_overridable_by_config() -> None:
    """``promotion_min_alpha_chars`` can be tuned per deployment via
    config — owners on noisier mics may want 5+ chars."""
    eng = _make_engine({"locale": "en-US", "promotion_min_alpha_chars": 6})
    assert eng._promotion_min_alpha_chars == 6
    assert eng._should_block_short_promotion("hello") is True   # 5 < 6
    assert eng._should_block_short_promotion("summary") is False  # 7 >= 6


def test_grace_window_overridable_by_config() -> None:
    eng = _make_engine({"locale": "en-US", "promotion_wake_grace_s": 10.0})
    assert eng._promotion_wake_grace_s == 10.0


# ---------------------------------------------------------------------------
# _is_recent_wake_context
# ---------------------------------------------------------------------------


def test_recent_wake_context_default_false(engine: NativeSTT) -> None:
    """Until the recognizer matches a wake phrase, the recent-wake
    helper must return False — otherwise every boot would treat its
    first noise spike as an intentional reply."""
    assert engine._is_recent_wake_context() is False


def test_recent_wake_context_true_inside_window(
    engine: NativeSTT,
) -> None:
    engine._last_wake_seen_t = time.monotonic()
    assert engine._is_recent_wake_context() is True


def test_recent_wake_context_false_outside_window(
    engine: NativeSTT,
) -> None:
    engine._last_wake_seen_t = time.monotonic() - (
        engine._promotion_wake_grace_s + 1.0
    )
    assert engine._is_recent_wake_context() is False
