"""
ATOM -- shared voice recovery lock.

Two independent mechanisms can tear down the microphone / recognition
pipeline while trying to recover from a hang:

  * STTWatchdog -> NativeSTT._restart_recognition_chain / full restart
  * AudioIntelligenceEngine -> seamless_switch / _smart_stt_recovery

If they both fire at the same time (e.g. "STT stuck for 22s" and
"audio still flowing, switch device"), two threads race to stop() /
close() the PortAudio stream. CoreAudio surfaces that as::

    PaMacCore (AUHAL) Error on line 2523: err='-50'

and the mic goes mute for 10s+ while both paths fumble. Both recovery
paths now funnel through :class:`VoiceRecoveryLock` so only ONE runs at
a time and the other waits with a short timeout before giving up.

Usage::

    async with voice_recovery_lock("stt_watchdog.full_restart"):
        await do_heavy_restart()

The lock is best-effort: if the primary holder wedges we release after
``max_wait_s`` and log a warning so observability still catches it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time as _time
from typing import Any, AsyncIterator

logger = logging.getLogger("atom.voice.recovery")


class VoiceRecoveryLock:
    """Process-wide coordinator between STT watchdog restarts and audio
    intelligence device switches.

    Lazily acquires a fresh :class:`asyncio.Lock` bound to the currently
    running loop, which matters because ATOM re-enters the event loop
    during shutdown / hot-reload tests. Also exposes a synchronous
    :class:`threading.Lock` for callers not in async context (rare but
    needed for Objective-C thread callbacks).
    """

    _async_lock: asyncio.Lock | None = None
    _async_lock_loop: asyncio.AbstractEventLoop | None = None
    _sync_lock: threading.Lock = threading.Lock()
    _current_holder: str | None = None
    _holder_start_t: float = 0.0

    DEFAULT_MAX_WAIT_S: float = 2.0
    STALE_HOLDER_WARN_S: float = 5.0

    @classmethod
    def _get_async_lock(cls) -> asyncio.Lock:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        if cls._async_lock is None or cls._async_lock_loop is not loop:
            cls._async_lock = asyncio.Lock()
            cls._async_lock_loop = loop
        return cls._async_lock

    @classmethod
    async def _try_acquire_async(cls, holder: str, max_wait_s: float) -> bool:
        lock = cls._get_async_lock()
        start = _time.monotonic()
        if lock.locked():
            existing = cls._current_holder or "unknown"
            held_for = _time.monotonic() - cls._holder_start_t
            logger.info(
                "Voice recovery lock busy (holder=%s, %.1fs) — %s waiting up to %.1fs",
                existing, held_for, holder, max_wait_s,
            )
        try:
            await asyncio.wait_for(lock.acquire(), timeout=max_wait_s)
        except asyncio.TimeoutError:
            existing = cls._current_holder or "unknown"
            held_for = _time.monotonic() - cls._holder_start_t
            logger.warning(
                "Voice recovery lock timeout after %.1fs (holder=%s held=%.1fs) — "
                "%s will skip recovery this cycle",
                _time.monotonic() - start, existing, held_for, holder,
            )
            return False
        cls._current_holder = holder
        cls._holder_start_t = _time.monotonic()
        return True

    @classmethod
    def _release_async(cls, holder: str) -> None:
        lock = cls._async_lock
        if lock is None:
            return
        if cls._current_holder == holder:
            cls._current_holder = None
            cls._holder_start_t = 0.0
        try:
            lock.release()
        except RuntimeError:
            logger.debug("Voice recovery lock already released", exc_info=True)


@contextlib.asynccontextmanager
async def voice_recovery_lock(
    holder: str,
    *,
    max_wait_s: float = VoiceRecoveryLock.DEFAULT_MAX_WAIT_S,
) -> AsyncIterator[bool]:
    """Async context manager returning ``True`` if the caller acquired the
    recovery lock. ``False`` means another path is already recovering and
    the caller should skip this cycle (logged as a warning).

    The caller MUST honour the boolean — blindly running recovery after a
    ``False`` defeats the whole point of serialisation.
    """
    got_lock = await VoiceRecoveryLock._try_acquire_async(holder, max_wait_s)
    try:
        yield got_lock
    finally:
        if got_lock:
            VoiceRecoveryLock._release_async(holder)


async def stream_drain_delay(ms: int = 400) -> None:
    """CoreAudio needs a short grace window between ``sd.stop()`` and the
    next ``sd.start()`` on the same device, otherwise PortAudio surfaces
    ``err=-50``. Use this between tearing down the old input and opening
    the new one during device switches."""
    delay_s = max(0.05, float(ms) / 1000.0)
    await asyncio.sleep(delay_s)


__all__ = [
    "VoiceRecoveryLock",
    "voice_recovery_lock",
    "stream_drain_delay",
]
