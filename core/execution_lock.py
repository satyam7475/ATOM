"""
ATOM -- Execution Lock (single-command mutex).

Ensures only one user command runs at a time through the ATOM pipeline.
Background/async tasks use a separate queue and are not gated here.

Features:
  - Async lock with configurable timeout (auto-releases stuck commands)
  - Tracks the current command label and start time for diagnostics
  - Context-manager interface for clean acquire/release
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger("atom.execution_lock")


class ExecutionLock:
    """Async mutex that guarantees one active command at a time."""

    __slots__ = (
        "_lock", "_current_command", "_started_at",
        "_default_timeout_s", "_total_acquired", "_total_rejected",
    )

    def __init__(self, default_timeout_s: float = 30.0) -> None:
        self._lock = asyncio.Lock()
        self._current_command: str | None = None
        self._started_at: float = 0.0
        self._default_timeout_s = max(1.0, float(default_timeout_s))
        self._total_acquired: int = 0
        self._total_rejected: int = 0

    @property
    def is_busy(self) -> bool:
        return self._lock.locked()

    @property
    def current_command(self) -> str | None:
        return self._current_command

    @property
    def elapsed_s(self) -> float:
        if not self._started_at:
            return 0.0
        return time.monotonic() - self._started_at

    async def acquire(
        self,
        command: str = "",
        timeout_s: float | None = None,
    ) -> bool:
        """Try to acquire the lock within *timeout_s* seconds.

        Returns True if acquired, False if the lock is already held and
        the timeout expired.
        """
        timeout = timeout_s if timeout_s is not None else self._default_timeout_s
        try:
            acquired = await asyncio.wait_for(
                self._lock.acquire(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            self._total_rejected += 1
            logger.warning(
                "ExecutionLock timeout (%.1fs) — '%s' rejected while '%s' is running (%.1fs)",
                timeout,
                command[:60],
                self._current_command or "?",
                self.elapsed_s,
            )
            return False

        if acquired:
            self._current_command = command[:120] if command else None
            self._started_at = time.monotonic()
            self._total_acquired += 1
            logger.debug("ExecutionLock acquired for '%s'", self._current_command or "?")
        return acquired

    def release(self) -> None:
        """Release the lock and clear command metadata."""
        elapsed = self.elapsed_s
        cmd = self._current_command
        self._current_command = None
        self._started_at = 0.0
        try:
            self._lock.release()
        except RuntimeError:
            logger.debug("ExecutionLock.release called when not locked")
            return
        logger.debug("ExecutionLock released for '%s' (%.0fms)", cmd or "?", elapsed * 1000)

    async def force_release(self) -> None:
        """Force-release on interrupt/cancel — safe even if not held."""
        if self._lock.locked():
            self.release()

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            "busy": self.is_busy,
            "current_command": self._current_command,
            "elapsed_s": round(self.elapsed_s, 2),
            "total_acquired": self._total_acquired,
            "total_rejected": self._total_rejected,
        }
