"""
ATOM -- Adaptive Power Governor.

Monitors host battery state to optimize ATOM's background intelligence daemons.
Dynamically throttles non-essential loops when unplugged to preserve laptop battery.
"""

from __future__ import annotations

import asyncio
import logging
import psutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus

logger = logging.getLogger("atom.power")

class PowerGovernor:
    """Manages system power states and optimizes ATOM background operations."""
    
    __slots__ = ("_bus", "_is_plugged_in", "_shutdown", "_task", "_check_interval")

    def __init__(self, bus: AsyncEventBus) -> None:
        self._bus = bus
        self._shutdown = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._check_interval = 60.0
        self._is_plugged_in = self._check_power_state()

    def _check_power_state(self) -> bool:
        """Polls psutil for true power state. Assumes plugged in if unavailable."""
        try:
            bat = psutil.sensors_battery()
            if bat is None:
                return True
            return bat.power_plugged
        except Exception:
            return True

    def start(self) -> None:
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("Adaptive Power Governor active (Plugged In: %s)", self._is_plugged_in)

    def stop(self) -> None:
        self._shutdown.set()
        if self._task and not self._task.done():
            self._task.cancel()

    async def _monitor_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=self._check_interval)
                break
            except asyncio.TimeoutError:
                pass
            
            try:
                current_state = self._check_power_state()
                if current_state != self._is_plugged_in:
                    self._is_plugged_in = current_state
                    mode = "plugged_in" if current_state else "battery"
                    logger.warning("Power Governor: Switched to %s mode", mode.upper())
                    self._bus.emit_fast("power_state_changed", state=mode)
            except Exception:
                logger.debug("Power Governor check failed", exc_info=True)
