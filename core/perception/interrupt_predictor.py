"""
ATOM -- Interrupt prediction engine.

Complements the existing ``VoiceInterruptHandler`` burst detection
(3 partials in 500ms) with a forward-looking signal: predicts that
the user *will* interrupt based on voice activity during TTS playback.

Emits ``interrupt_predicted`` on the bus so the interrupt handler
can pre-pause TTS before the user finishes speaking their new command.
"""

from __future__ import annotations

import time


_MIN_PARTIAL_LEN = 3
_REQUIRED_PARTIALS = 2
_CONFIRM_WINDOW_S = 0.8


class InterruptPredictor:
    """Predict user interrupts during active TTS playback.

    Requires 2+ non-trivial partials within 800ms to avoid false
    triggers from breath noise, mic bumps, or single-phoneme blips.
    """

    __slots__ = (
        "_speaking",
        "_partial_count",
        "_first_partial_t",
        "_fired",
    )

    def __init__(self) -> None:
        self._speaking = False
        self._partial_count = 0
        self._first_partial_t = 0.0
        self._fired = False

    def on_tts_start(self) -> None:
        self._speaking = True
        self._partial_count = 0
        self._first_partial_t = 0.0
        self._fired = False

    def on_tts_end(self) -> None:
        self._speaking = False
        self._partial_count = 0
        self._first_partial_t = 0.0
        self._fired = False

    def on_partial(self, text: str) -> bool:
        """Return True when accumulated partials confirm a real interrupt.

        Only fires once per TTS session to avoid repeated signals.
        """
        if not self._speaking or self._fired:
            return False

        stripped = (text or "").strip()
        if len(stripped) < _MIN_PARTIAL_LEN:
            return False

        now = time.monotonic()

        if self._partial_count == 0:
            self._first_partial_t = now

        if now - self._first_partial_t > _CONFIRM_WINDOW_S:
            self._partial_count = 1
            self._first_partial_t = now
        else:
            self._partial_count += 1

        if self._partial_count >= _REQUIRED_PARTIALS:
            self._fired = True
            return True

        return False
