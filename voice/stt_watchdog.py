"""
ATOM -- STT Self-Healing Watchdog.

Monitors the STT engine for stuck/silent states and automatically
restarts it. Detects:

  1. No partials for > N seconds while mic should be active
  2. Recognition chain stuck (task completed but not restarted)
  3. Audio engine stopped unexpectedly
  4. Repeated errors without recovery

Runs as a lightweight async task alongside the STT loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger("atom.stt_watchdog")

_SILENT_TIMEOUT_S = 8.0
_STUCK_TIMEOUT_S = 15.0
_MAX_RESTARTS_PER_WINDOW = 5
_RESTART_WINDOW_S = 300.0
_CHECK_INTERVAL_S = 2.0
_CHAIN_RESTARTS_BEFORE_FULL = 3


class STTWatchdog:
    """Monitors STT health and triggers auto-recovery."""

    def __init__(
        self,
        bus: Any,
        *,
        silent_timeout_s: float = _SILENT_TIMEOUT_S,
        stuck_timeout_s: float = _STUCK_TIMEOUT_S,
    ) -> None:
        self._bus = bus
        self._silent_timeout = max(3.0, float(silent_timeout_s))
        self._stuck_timeout = max(5.0, float(stuck_timeout_s))

        self._last_partial_time: float = time.monotonic()
        self._last_final_time: float = time.monotonic()
        self._last_tap_count: int = 0
        self._stt_ref: Any = None

        self._restart_times: list[float] = []
        self._total_restarts: int = 0
        self._total_silent_detections: int = 0
        self._consecutive_chain_restarts: int = 0

        self._task: asyncio.Task | None = None
        self._shutdown = asyncio.Event()

    def attach_stt(self, stt: Any) -> None:
        self._stt_ref = stt

    def on_speech_partial(self, text: str = "", **_kw: Any) -> None:
        """Called from bus on every STT partial -- updates liveness."""
        self._last_partial_time = time.monotonic()
        self._consecutive_chain_restarts = 0

    def on_speech_final(self, text: str = "", **_kw: Any) -> None:
        """Called from bus on every STT final -- updates liveness."""
        self._last_final_time = time.monotonic()
        self._last_partial_time = time.monotonic()

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._shutdown.clear()
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("STT Watchdog started (silent=%.0fs stuck=%.0fs)",
                     self._silent_timeout, self._stuck_timeout)

    def stop(self) -> None:
        self._shutdown.set()
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None

    async def _monitor_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(), timeout=_CHECK_INTERVAL_S,
                )
                break
            except asyncio.TimeoutError:
                pass

            try:
                await self._check_health()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.debug("STT Watchdog check error", exc_info=True)

    async def _check_health(self) -> None:
        stt = self._stt_ref
        if stt is None:
            return

        now = time.monotonic()
        is_listening = getattr(stt, "_listening", False)
        is_running = getattr(stt, "_running_async", False)

        if not is_running:
            return

        state_mgr = getattr(stt, "_state", None)
        if state_mgr is not None:
            try:
                from core.state_manager import AtomState
                cur = state_mgr.current
                if cur is AtomState.SPEAKING:
                    return
            except Exception:
                pass

        since_partial = now - self._last_partial_time
        since_final = now - self._last_final_time

        tap_count = getattr(stt, "_tap_buffer_count", 0)
        audio_flowing = tap_count > self._last_tap_count
        self._last_tap_count = tap_count
        last_audio_rms_db = float(getattr(stt, "_last_audio_rms_db", -96.0))
        last_speech_candidate = float(getattr(stt, "_last_speech_candidate_time", 0.0) or 0.0)
        speech_likely = (now - last_speech_candidate) < 3.0 if last_speech_candidate > 0 else False

        if is_listening and since_partial > self._silent_timeout and audio_flowing and speech_likely:
            self._total_silent_detections += 1
            logger.warning(
                "STT Watchdog: no partials for %.1fs despite speech-like audio "
                "(tap_count=%d, rms=%.1f dB) -- recognizer likely starved",
                since_partial, tap_count, last_audio_rms_db,
            )
            starvation_fn = getattr(stt, "_on_recognition_starvation", None)
            if callable(starvation_fn):
                try:
                    starvation_fn()
                except Exception:
                    logger.debug("STT Watchdog: recognition starvation hook failed", exc_info=True)
            if self._can_restart():
                await self._restart_stt(stt, "recognition_starved")
        elif is_listening and since_partial > self._silent_timeout and audio_flowing:
            logger.debug(
                "STT Watchdog: audio is flowing but no speech-like activity yet "
                "(tap_count=%d, rms=%.1f dB) -- suppressing restart",
                tap_count, last_audio_rms_db,
            )

        if is_listening and since_final > self._stuck_timeout and since_partial > self._stuck_timeout:
            last_error = getattr(stt, "_last_error", None)
            if last_error:
                logger.warning(
                    "STT Watchdog: stuck for %.1fs with error '%s' -- restarting",
                    since_final, str(last_error)[:80],
                )
                if self._can_restart():
                    await self._restart_stt(stt, "stuck_with_error")

    def _can_restart(self) -> bool:
        now = time.monotonic()
        self._restart_times = [
            t for t in self._restart_times
            if now - t < _RESTART_WINDOW_S
        ]
        return len(self._restart_times) < _MAX_RESTARTS_PER_WINDOW

    async def _restart_stt(self, stt: Any, reason: str) -> None:
        self._total_restarts += 1
        self._consecutive_chain_restarts += 1
        self._restart_times.append(time.monotonic())

        needs_full_restart = self._consecutive_chain_restarts >= _CHAIN_RESTARTS_BEFORE_FULL

        if needs_full_restart:
            logger.warning(
                "STT Watchdog: %d consecutive chain restarts with no partials — "
                "escalating to full engine restart with device rebinding (#%d)",
                self._consecutive_chain_restarts, self._total_restarts,
            )
            self._consecutive_chain_restarts = 0
        else:
            logger.info("STT Watchdog: restarting STT (%s, restart #%d, chain #%d/%d)",
                        reason, self._total_restarts,
                        self._consecutive_chain_restarts, _CHAIN_RESTARTS_BEFORE_FULL)

        if not needs_full_restart:
            restart_fn = getattr(stt, "_restart_recognition_chain", None)
            if callable(restart_fn):
                try:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, restart_fn)
                    self._last_partial_time = time.monotonic()
                    logger.info("STT Watchdog: recognition chain restarted successfully")
                    self._bus.emit_fast("stt_watchdog_restart", reason=reason, restart_count=self._total_restarts)
                    return
                except Exception:
                    logger.warning("STT Watchdog: chain restart failed, trying full stop/start", exc_info=True)

        rebind_fn = getattr(stt, "_rebind_audio_device", None)
        if callable(rebind_fn):
            try:
                rebind_fn()
                logger.info("STT Watchdog: re-asserted CoreAudio device before full restart")
            except Exception:
                logger.debug("STT Watchdog: device rebind failed", exc_info=True)

        try:
            stt.stop_listening()
            await asyncio.sleep(0.8)
            start_fn = getattr(stt, "start_listening", None)
            if callable(start_fn):
                loop = getattr(stt, "_loop", None)
                on_final = getattr(stt, "_on_final", None)
                on_partial = getattr(stt, "_on_partial", None)
                start_fn(loop=loop, on_final=on_final, on_partial=on_partial)
                self._last_partial_time = time.monotonic()
                logger.info("STT Watchdog: full STT restart completed (engine rebuilt)")
        except Exception:
            logger.exception("STT Watchdog: full restart failed")

        self._bus.emit_fast("stt_watchdog_restart", reason=reason, restart_count=self._total_restarts)

    def get_diagnostics(self) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "total_restarts": self._total_restarts,
            "total_silent_detections": self._total_silent_detections,
            "since_last_partial_s": round(now - self._last_partial_time, 1),
            "since_last_final_s": round(now - self._last_final_time, 1),
            "recent_restarts": len(self._restart_times),
        }


__all__ = ["STTWatchdog"]
