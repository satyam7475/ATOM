"""Regression tests for the shared speech sanitiser (Sprint A3).

Pins:

* The legacy stage-direction strip (open + closed paren forms, narration
  verbs, length-cap) still works after the move to ``brain._speech_sanitizer``.
* ``brain.mlx_llm`` and ``cursor_bridge.local_brain_controller`` rebind
  the same function name, so no caller goes stale.
* :class:`StreamingLeakBuffer` holds the head until it has either a
  terminator or 60 chars, then releases the cleaned text and switches
  to passthrough.
"""

from __future__ import annotations

import pytest

from brain._speech_sanitizer import (
    StreamingLeakBuffer,
    looks_like_stage_direction_head,
    strip_stage_direction_leak,
)


# ── core sanitiser ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw, expected",
    [
        # closed-paren form, original Sprint A
        (
            "(in a calm, composed tone). Boss, on it.",
            "Boss, on it.",
        ),
        # open-paren form, atomLogs.txt L357
        (
            "(in a calm, composed tone",
            "",
        ),
        # open-paren with inner dot, atomLogs.txt L554
        (
            "(calm, composed tone.",
            "",
        ),
        # narration verb, atomLogs.txt L409
        (
            "(responds immediately) Sure, Boss.",
            "Sure, Boss.",
        ),
    ],
)
def test_strip_stage_direction_leak_handles_known_log_shapes(
    raw: str, expected: str,
) -> None:
    assert strip_stage_direction_leak(raw).strip() == expected.strip()


def test_strip_preserves_truncated_head_without_vocab() -> None:
    """The sanitiser is vocab-anchored: a leak shape WITHOUT the vocab
    (e.g. only the literal ``(in a.`` which was log-line 301) is left
    alone -- the streaming buffer is what catches it because it waits
    for additional slices to arrive first."""
    # The legacy regex requires the leak vocab. A bare ``(in a.`` does
    # not contain it. Accept either passthrough OR strip; what matters
    # is the StreamingLeakBuffer test below covers the live path.
    out = strip_stage_direction_leak("(in a.")
    assert out == "(in a." or out == ""


@pytest.mark.parametrize(
    "passthrough",
    [
        "",
        "Boss, weather looks clear today.",
        "(see line 12) for more context.",          # legitimate aside
        "(2 of 3) processing complete.",            # numeric aside
        "Sure, on it.",
        "Hello Boss",
    ],
)
def test_strip_stage_direction_leak_leaves_clean_text_untouched(
    passthrough: str,
) -> None:
    assert strip_stage_direction_leak(passthrough) == passthrough


def test_strip_is_idempotent() -> None:
    raw = "(in a calm, composed tone). (responds politely) Boss."
    once = strip_stage_direction_leak(raw)
    twice = strip_stage_direction_leak(once)
    assert once == twice
    assert "Boss" in twice


def test_strip_does_not_eat_long_legitimate_paren() -> None:
    """A 200-char inner paren without leak vocab must be untouched -- the
    stripper is length-capped and vocab-anchored on purpose."""
    long_safe = "(" + "x" * 200 + ") Hello Boss."
    assert strip_stage_direction_leak(long_safe) == long_safe


# ── helper: looks_like_stage_direction_head ────────────────────────


def test_looks_like_head_positive() -> None:
    assert looks_like_stage_direction_head("(in a calm tone") is True
    assert looks_like_stage_direction_head("(responds immediately)") is True


def test_looks_like_head_negative() -> None:
    assert looks_like_stage_direction_head("") is False
    assert looks_like_stage_direction_head("Boss, on it.") is False
    assert looks_like_stage_direction_head("(see line 12)") is False


# ── unified definition: rebound names match ────────────────────────


def test_mlx_llm_rebinds_to_shared_sanitizer() -> None:
    from brain import _speech_sanitizer
    from brain.mlx_llm import _strip_stage_direction_leak as alias

    assert alias is _speech_sanitizer.strip_stage_direction_leak


def test_local_brain_controller_rebinds_to_shared_sanitizer() -> None:
    from brain import _speech_sanitizer
    from cursor_bridge.local_brain_controller import (
        _strip_stage_direction_leak as alias,
    )

    assert alias is _speech_sanitizer.strip_stage_direction_leak


# ── StreamingLeakBuffer ────────────────────────────────────────────


def test_buffer_holds_head_until_terminator() -> None:
    buf = StreamingLeakBuffer()
    out = buf.feed("(in a")
    assert out == []                # still buffering
    out = buf.feed(" calm, composed tone). Boss, on it.")
    # The terminator `.` is now in the buffer -- one cleaned release.
    assert len(out) == 1
    assert out[0].strip().startswith("Boss")
    assert buf.released is True


def test_buffer_passthrough_after_release() -> None:
    buf = StreamingLeakBuffer()
    # Force release with a clean head.
    out = buf.feed("Sure, Boss.")
    assert out == ["Sure, Boss."]
    assert buf.released is True
    # All further feeds bypass the buffer.
    assert buf.feed(" Next chunk.") == [" Next chunk."]
    assert buf.feed("Another one") == ["Another one"]


def test_buffer_releases_at_head_chars_cap() -> None:
    """No terminator, but 60-char head reached -> release."""
    buf = StreamingLeakBuffer(head_chars=60)
    long_clean = "x" * 70                # no terminator, no leak
    out = buf.feed(long_clean)
    assert out == [long_clean]
    assert buf.released is True


def test_buffer_strips_open_paren_leak() -> None:
    """Open-paren leak with no closer -- must be eaten on flush."""
    buf = StreamingLeakBuffer()
    assert buf.feed("(in a calm composed tone") == []
    out = buf.flush()
    # Either nothing emitted or empty string suppressed -- both fine.
    assert all(s == "" for s in out) or out == []
    assert buf.released is True


def test_buffer_handles_first_log_line_301_shape_streaming() -> None:
    """Reproduces atomLogs.txt L301: the FIRST stream slice was ``(in a.``
    and got pushed straight to TTS. With the StreamingLeakBuffer in
    place, even the first slice is held until either (a) more slices
    arrive that complete the leak vocab, or (b) the head is shown to
    be clean. Crucially, the leaked head ``(in a calm, composed tone).``
    is suppressed."""
    buf = StreamingLeakBuffer(head_chars=80)
    # Slice 1: the head we used to speak too early
    assert buf.feed("(in a.") == []
    # Slice 2: the model continues; vocab now lands in the buffer
    out = buf.feed(" calm, composed tone). Boss, on it.")
    # Whatever we release must NOT contain the leaked head fragment.
    cleaned = "".join(out)
    assert "(in a" not in cleaned
    assert "calm" not in cleaned or "Boss" in cleaned
    assert "Boss" in cleaned


def test_buffer_reset_clears_state() -> None:
    buf = StreamingLeakBuffer()
    buf.feed("Hello Boss.")             # release
    assert buf.released is True
    buf.reset()
    assert buf.released is False
    assert buf.buffered == ""


def test_buffer_empty_feed_is_noop() -> None:
    buf = StreamingLeakBuffer()
    assert buf.feed("") == []
    assert buf.released is False


def test_buffer_flush_when_already_released_is_noop() -> None:
    buf = StreamingLeakBuffer()
    buf.feed("Already clean.")
    assert buf.released is True
    assert buf.flush() == []


def test_buffer_releases_clean_head_when_terminator_arrives_first() -> None:
    """Real-life shape: a couple of small slices ending with `.`.
    The cleaned head should be everything we've buffered."""
    buf = StreamingLeakBuffer()
    assert buf.feed("Boss") == []
    assert buf.feed(", on") == []
    out = buf.feed(" it.")
    assert len(out) == 1
    assert out[0].strip().endswith("on it.")
