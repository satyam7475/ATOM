"""Shared speech-output sanitiser.

Single source of truth for stripping the chain-of-thought preface and
stage-direction parentheticals that smaller instruction-tuned models
(Qwen3-4B, in particular) leak into the head of the response. The
underlying regex was originally duplicated in ``brain/mlx_llm.py`` and
``cursor_bridge/local_brain_controller.py``; both now defer here so the
batch-LLM path, the streaming-LLM path and the streaming-TTS path use
the exact same definition of "is this a leak?".

Two public surfaces:

* :func:`strip_stage_direction_leak` -- single-shot strip (idempotent,
  safe on empty input). Use after the model has produced a complete
  utterance / paragraph.
* :class:`StreamingLeakBuffer` -- holds the head of a streaming TTS
  response until either (a) we have enough chars to know the leak is
  not at the front, or (b) a sentence boundary closes the head.
  Prevents the first audio slice from speaking the leaked parenthetical
  before the batch sanitiser had a chance to fire.
"""

from __future__ import annotations

import re

# ── 1.  Vocabulary anchor ──────────────────────────────────────────
# Adjectives + verbs that almost always indicate a stage direction
# rather than a legitimate aside like "(see line 12)" or "(2 of 3)".
_STAGE_LEAK_VOCAB = (
    r"\b(?:tone|voice|manner|composed|composedly|calm(?:ly)?|softly|"
    r"warmly|gently|firmly|politely|brief(?:ly)?|professional(?:ly)?|"
    r"quietly|quickly|slowly|immediately|confidently|cheerful(?:ly)?|"
    r"cheery|crisp(?:ly)?|relaxed|respectful(?:ly)?|measured|steady|"
    r"steadily|chief\s+of\s+staff|friday[-\s]?style|jarvis[-\s]?style|"
    r"respond(?:s|ed|ing)?|reply(?:ies|ied|ying)?|answer(?:s|ed|ing)?|"
    r"pause(?:s|d|ing)?|nod(?:s|ded|ding)?|smile(?:s|d|ing)?|"
    r"chuckle(?:s|d|ing)?|sigh(?:s|ed|ing)?|breathe(?:s|d|ing)?|"
    r"speaks?|speaking|in\s+a\s+(?:tone|voice|manner))\b"
)

# Closed-paren leak: "(in a calm, composed tone). Boss, ..."
_STAGE_DIRECTION_LEAK_RE = re.compile(
    r"""
    ^\s*
    \(\s*
    [^()\n]{0,80}?
    """ + _STAGE_LEAK_VOCAB + r"""
    [^()\n]{0,80}?
    \)
    \s*[\.,;:\-\u2013\u2014]?\s*
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Open-paren leak: "(in a calm, composed tone" -- model truncated mid-
# clause. Terminates at end-of-string or newline.
_STAGE_DIRECTION_OPEN_LEAK_RE = re.compile(
    r"""
    ^\s*
    \(\s*
    [^()\n]{0,160}?
    """ + _STAGE_LEAK_VOCAB + r"""
    [^()\n]*?
    (?:$|\n)
    """,
    re.IGNORECASE | re.VERBOSE,
)


# Terminator chars that close the "head" we want to inspect for a leak.
# When we see one of these (or `)` for closed leaks) we know the leak
# slot is fully buffered and we can run the regex.
_HEAD_TERMINATORS: tuple[str, ...] = (
    ".", "!", "?", "\n", ")",
)


def strip_stage_direction_leak(text: str) -> str:
    """Peel a leading bare-parenthetical stage direction.

    Safe on empty input. Idempotent. Length-capped at 160 chars from the
    head so legitimate citations or footnotes pass through untouched.
    """
    if not text or "(" not in text[:160]:
        return text
    out = text
    for _ in range(2):
        new = _STAGE_DIRECTION_LEAK_RE.sub("", out, count=1).lstrip()
        if new == out:
            new = _STAGE_DIRECTION_OPEN_LEAK_RE.sub("", out, count=1).lstrip()
        if new == out:
            break
        out = new
    return out


def looks_like_stage_direction_head(text: str) -> bool:
    """True iff the first non-space char of *text* is ``(`` and the head
    matches the leak vocab. Used by the streaming buffer to decide
    whether to keep buffering or release."""
    if not text:
        return False
    head = text.lstrip()
    if not head.startswith("("):
        return False
    return strip_stage_direction_leak(head) != head


class StreamingLeakBuffer:
    """Holds the first ``head_chars`` of a streaming reply until we are
    confident the head is leak-free.

    Used by the TTS streamer to prevent the very first audio slice from
    pronouncing a leaked parenthetical before the batch-level sanitiser
    has a chance to peel it. The underlying defect appears in
    ``atomLogs.txt`` lines 301 / 401 / 521 -- TTS spoke ``"(in a."`` and
    ``"(bypassing the current content...)"`` because they were the very
    first stream slices and slipped past the batch guard.

    Lifecycle::

        buf = StreamingLeakBuffer()
        for slice_text in llm_stream():
            for ready_text in buf.feed(slice_text):
                tts.speak(ready_text)
        for ready_text in buf.flush():
            tts.speak(ready_text)
    """

    __slots__ = ("_buf", "_released", "_head_chars")

    def __init__(self, head_chars: int = 60) -> None:
        # 60 chars is enough for every leak shape we have seen so far
        # ("(in a calm, composed, professional tone)." == 41 chars) and
        # short enough that the perceived TTS-start latency is bounded
        # at one sentence-boundary worth of buffering.
        self._buf = ""
        self._released = False
        self._head_chars = max(8, int(head_chars))

    @property
    def buffered(self) -> str:
        return self._buf

    @property
    def released(self) -> bool:
        return self._released

    def reset(self) -> None:
        self._buf = ""
        self._released = False

    def feed(self, chunk: str) -> list[str]:
        """Push another stream slice. Returns zero or more sanitised
        chunks the caller may safely speak."""
        if not chunk:
            return []
        if self._released:
            return [chunk]
        self._buf += chunk
        head = self._buf.lstrip()
        # When the head starts with `(`, treat the `)` as the only
        # safe sentence-equivalent terminator -- a `.` or `\n` could
        # easily land *inside* an unclosed stage direction (atomLogs
        # L301 is the canonical failure: the model emitted ``(in a.``
        # as its first slice and the old buffer released on the dot).
        if head.startswith("("):
            terminators = (")",)
        else:
            terminators = _HEAD_TERMINATORS
        if (
            len(self._buf) >= self._head_chars
            or any(t in self._buf for t in terminators)
        ):
            cleaned = strip_stage_direction_leak(self._buf)
            self._buf = ""
            self._released = True
            return [cleaned] if cleaned else []
        return []

    def flush(self) -> list[str]:
        """Stream finished. Drain whatever's left through the
        sanitiser."""
        if self._released:
            return []
        cleaned = strip_stage_direction_leak(self._buf)
        self._buf = ""
        self._released = True
        return [cleaned] if cleaned else []


__all__ = (
    "strip_stage_direction_leak",
    "looks_like_stage_direction_head",
    "StreamingLeakBuffer",
)
