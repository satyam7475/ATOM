"""
ATOM v14 -- Microphone Ownership Manager.

Ensures only ONE component can own the microphone at a time.
Uses threading.Condition for efficient wait/notify semantics so
the acquiring thread blocks until the mic is free rather than polling.

Ownership protocol:
    1. Call acquire(owner_name) before opening a PyAudio stream
    2. Call release(owner_name) after closing the stream
    3. acquire() blocks until the current owner releases

This prevents PortAudioError on Windows where exclusive-mode audio
devices reject concurrent stream opens.
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger("atom.mic")


class MicManager:
    """
    Central microphone lock for coordinating exclusive mic users (e.g. STTAsync).

    Thread-safe: uses threading.Condition internally. Safe to call from
    asyncio executor threads and any background audio consumers.
    """

    __slots__ = ("_lock", "_condition", "_owner")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._owner: str | None = None

    def acquire(self, owner: str, timeout: float = 5.0) -> bool:
        """
        Acquire microphone ownership, blocking until it's free.

        Returns True if acquired, False if timed out.
        The caller MUST call release() when done with the mic.
        """
        deadline = time.monotonic() + timeout

        with self._condition:
            while self._owner is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning(
                        "Mic acquire timed out for '%s' (held by '%s')",
                        owner, self._owner,
                    )
                    return False
                self._condition.wait(timeout=remaining)

            self._owner = owner
            return True

    def release(self, owner: str) -> None:
        """
        Release microphone ownership. Only the current owner can release.

        Wakes up any thread waiting in acquire().
        """
        with self._condition:
            if self._owner == owner:
                self._owner = None
                self._condition.notify_all()
            elif self._owner is None:
                pass
            else:
                logger.warning(
                    "Mic release rejected: '%s' tried to release but '%s' owns it",
                    owner, self._owner,
                )

    @property
    def owner(self) -> str | None:
        """Current mic owner name, or None if free. Lock-free read (GIL safe)."""
        return self._owner

    @property
    def is_free(self) -> bool:
        return self._owner is None
