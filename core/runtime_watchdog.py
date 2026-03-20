"""
ATOM — Stuck-state watchdog + lightweight STT/TTS recovery supervisor.

* THINKING / SPEAKING held too long → resume_listening + restart_listening
* Cooldown prevents recovery storms
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.state_manager import StateManager

logger = logging.getLogger("atom.watchdog")


class RuntimeWatchdog:
    """Polls state dwell time; emits the same recovery events as UNSTICK."""

    __slots__ = (
        "_bus", "_state", "_config", "_state_entered", "_task",
        "_shutdown", "_cooldown_s", "_last_recovery", "_think_s", "_speak_s",
        "_poll_interval",
    )

    def __init__(
        self,
        bus: "AsyncEventBus",
        state: "StateManager",
        config: dict,
    ) -> None:
        self._bus = bus
        self._state = state
        self._config = config
        perf = config.get("performance", {}) or {}
        self._think_s = float(perf.get("watchdog_thinking_timeout_s", 120))
        self._speak_s = float(perf.get("watchdog_speaking_timeout_s", 300))
        self._cooldown_s = float(perf.get("supervisor_restart_cooldown_s", 8))
        self._poll_interval = float(perf.get("watchdog_poll_interval_s", 2.0))
        self._state_entered = time.monotonic()
        self._task: asyncio.Task | None = None
        self._shutdown = False
        self._last_recovery = 0.0

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._shutdown = False
            self._task = asyncio.create_task(
                self._loop(), name="atom_runtime_watchdog"
            )
            logger.info(
                "RuntimeWatchdog started (think=%.0fs speak=%.0fs cooldown=%.0fs)",
                self._think_s, self._speak_s, self._cooldown_s,
            )

    async def shutdown(self) -> None:
        self._shutdown = True
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def on_state_changed(
        self,
        old: Any = None,
        new: Any = None,
        **_kw: Any,
    ) -> None:
        self._state_entered = time.monotonic()

    def _maybe_recover(self, reason: str) -> None:
        now = time.monotonic()
        if now - self._last_recovery < self._cooldown_s:
            logger.debug("Watchdog skip %s (cooldown)", reason)
            return
        self._last_recovery = now
        logger.warning("Watchdog recovery: %s", reason)
        self._bus.emit("metrics_event", counter="watchdog_recoveries")
        self._bus.emit("resume_listening")
        # restart_listening scheduled shortly after (matches dashboard UNSTICK)
        asyncio.get_running_loop().call_later(
            0.05, lambda: self._bus.emit("restart_listening")
        )

    async def _loop(self) -> None:
        from core.state_manager import AtomState

        while not self._shutdown:
            try:
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                break
            if self._shutdown:
                break
            st = self._state.current
            elapsed = time.monotonic() - self._state_entered
            if st is AtomState.THINKING and elapsed > self._think_s:
                self._maybe_recover(f"THINKING stuck {elapsed:.0f}s")
            elif st is AtomState.SPEAKING and elapsed > self._speak_s:
                self._maybe_recover(f"SPEAKING stuck {elapsed:.0f}s")
