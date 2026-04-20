"""
ATOM — Error Rate Monitor (Sprint C6).

Tracks handler errors across the event bus (and optionally other
subsystems) in a rolling 60-second window and emits a
``atom_error_burst_detected`` bus event when the rate crosses a
threshold. The orchestrator subscribes to that event and can:

  * log a spoken warning to the user ("something's glitchy, Boss")
  * surface the issue on the web dashboard
  * trip a shallow "safe mode" that shrinks work in flight

This module avoids pulling the event bus as a dependency so the bus
can import us without a cycle. Consumers pass the bus at wire-up
time.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from typing import Any, Deque, Iterable

logger = logging.getLogger("atom.obs.error_rate")


class ErrorRateMonitor:
    """Global rolling-window error counter + alerter."""

    def __init__(
        self,
        *,
        window_s: float = 60.0,
        threshold: int = 5,
        poll_interval_s: float = 10.0,
        alert_cooldown_s: float = 60.0,
    ) -> None:
        self._window_s = float(window_s)
        self._threshold = int(threshold)
        self._poll_interval_s = float(poll_interval_s)
        self._alert_cooldown_s = float(alert_cooldown_s)

        self._events: Deque[tuple[float, str, str]] = deque(maxlen=1024)
        self._lock = threading.Lock()
        self._last_alert_t: float = 0.0
        self._total_errors: int = 0
        self._task: asyncio.Task | None = None
        self._shutdown: asyncio.Event | None = None
        self._bus: Any = None

    # ── Public API ───────────────────────────────────────────────

    def record(self, source: str, detail: str = "") -> None:
        """Record a single error. Safe from any thread."""
        now = time.monotonic()
        with self._lock:
            self._events.append((now, source or "unknown", detail[:200] if detail else ""))
            self._total_errors += 1

    def rate(self, window_s: float | None = None) -> int:
        """Count errors within the last ``window_s`` seconds (default: configured window)."""
        w = float(window_s) if window_s is not None else self._window_s
        cutoff = time.monotonic() - w
        with self._lock:
            return sum(1 for t, _, _ in self._events if t >= cutoff)

    def top_sources(self, window_s: float | None = None, n: int = 3) -> list[tuple[str, int]]:
        w = float(window_s) if window_s is not None else self._window_s
        cutoff = time.monotonic() - w
        counts: dict[str, int] = {}
        with self._lock:
            for t, src, _ in self._events:
                if t < cutoff:
                    continue
                counts[src] = counts.get(src, 0) + 1
        return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:n]

    def diagnostics(self) -> dict[str, Any]:
        return {
            "threshold": self._threshold,
            "window_s": self._window_s,
            "total_errors": self._total_errors,
            "recent_rate": self.rate(),
            "top_sources_recent": self.top_sources(),
        }

    # ── Lifecycle ────────────────────────────────────────────────

    def start(self, bus: Any) -> None:
        self._bus = bus
        if self._task is not None and not self._task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("ErrorRateMonitor.start() requires a running loop")
            return
        self._shutdown = asyncio.Event()
        self._task = loop.create_task(self._monitor_loop())
        logger.info(
            "Error-rate monitor started (threshold=%d/%.0fs, poll=%.0fs)",
            self._threshold, self._window_s, self._poll_interval_s,
        )

    def stop(self) -> None:
        evt = self._shutdown
        if evt is not None:
            evt.set()
        task = self._task
        if task is not None and not task.done():
            task.cancel()
        self._task = None
        self._shutdown = None

    # ── Internals ────────────────────────────────────────────────

    async def _monitor_loop(self) -> None:
        evt = self._shutdown
        while evt is not None and not evt.is_set():
            try:
                await asyncio.wait_for(evt.wait(), timeout=self._poll_interval_s)
                break
            except asyncio.TimeoutError:
                pass
            try:
                self._maybe_alert()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.debug("error-rate monitor check failed", exc_info=True)

    def _maybe_alert(self) -> None:
        rate = self.rate()
        if rate < self._threshold:
            return
        now = time.monotonic()
        if now - self._last_alert_t < self._alert_cooldown_s:
            return
        top = self.top_sources(n=3)
        self._last_alert_t = now
        logger.warning(
            "Error burst detected: %d errors in the last %.0fs (top: %s)",
            rate, self._window_s,
            ", ".join(f"{s}={c}" for s, c in top) or "unknown",
        )
        bus = self._bus
        if bus is None:
            return
        try:
            bus.emit_fast(
                "atom_error_burst_detected",
                rate=rate,
                threshold=self._threshold,
                window_s=self._window_s,
                top_sources=top,
            )
        except Exception:
            logger.debug("error-rate emit failed", exc_info=True)


_instance: ErrorRateMonitor | None = None
_instance_lock = threading.Lock()


def get_error_rate_monitor(
    *,
    window_s: float = 60.0,
    threshold: int = 5,
    poll_interval_s: float = 10.0,
) -> ErrorRateMonitor:
    global _instance
    if _instance is not None:
        return _instance
    with _instance_lock:
        if _instance is None:
            _instance = ErrorRateMonitor(
                window_s=window_s,
                threshold=threshold,
                poll_interval_s=poll_interval_s,
            )
        return _instance


def record_error(source: str, detail: str = "") -> None:
    """Module-level helper for subsystems that don't hold the instance."""
    try:
        get_error_rate_monitor().record(source, detail)
    except Exception:
        logger.debug("record_error failed", exc_info=True)


__all__ = ["ErrorRateMonitor", "get_error_rate_monitor", "record_error"]
