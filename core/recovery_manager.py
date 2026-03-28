"""
ATOM V7 — Recovery manager: worker crash hooks, event ring, degradation hints.

Does not replace ServiceWatchdog; complements process-level restarts.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Callable

from core.event_ring import EventRingBuffer
from core.runtime_config import DegradationMode, set_degradation_mode

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus

logger = logging.getLogger("atom.recovery_manager")


class RecoveryManager:
    def __init__(
        self,
        bus: "AsyncEventBus | None" = None,
        config: dict | None = None,
    ) -> None:
        self._bus = bus
        self._v7 = (config or {}).get("v7_gpu") or {}
        self._ring = EventRingBuffer(
            max_events=int(self._v7.get("event_replay_max", 32)),
        )
        self._on_restart: list[Callable[[str], None]] = []

    def record_event(self, name: str, payload: dict | None = None) -> None:
        self._ring.push(name, payload)

    def on_worker_crash(self, worker_name: str, exit_code: int | None = None) -> None:
        logger.error("RecoveryManager: worker %s crashed (code=%s)", worker_name, exit_code)
        if self._bus:
            self._bus.emit_fast(
                "v7_worker_crash",
                worker=worker_name,
                exit_code=exit_code,
                t=time.time(),
            )
        set_degradation_mode(DegradationMode.LIMITED)
        for cb in self._on_restart:
            try:
                cb(worker_name)
            except Exception:
                logger.debug("recovery callback failed", exc_info=True)

    def replay_recent(self, handler: Callable[[str, dict], None]) -> int:
        """Replay ring to handler (caller must filter idempotent events)."""
        n = 0
        for ev in self._ring.recent():
            try:
                handler(ev.name, ev.payload)
                n += 1
            except Exception:
                logger.debug("replay failed for %s", ev.name, exc_info=True)
        return n

    def register_restart_callback(self, cb: Callable[[str], None]) -> None:
        self._on_restart.append(cb)
