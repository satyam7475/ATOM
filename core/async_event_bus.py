"""
ATOM -- Async Event Bus (lightweight pub/sub backbone).

All inter-module communication flows through this bus.
Subscribers are invoked as asyncio.Tasks so emitters never block.

v20 optimizations over v14:
  - Cached loop reference (avoids get_running_loop() on every emit)
  - Tuple-based handler storage (avoid list copy on every emit)
  - Single-handler fast-path (skip task creation overhead)
  - Handler count tracking per event for metrics
  - Batch emit support for correlated events

Thread-safety hardening (v10):
  - Handler timeout (10s) prevents hung tasks from accumulating
  - Active task tracking via WeakSet for observability
  - Slow handler warnings (>5s)
"""

from __future__ import annotations

import asyncio
import logging
import time
import weakref
from collections import defaultdict
from typing import Any, Callable, Coroutine

logger = logging.getLogger("atom.event_bus")

EventHandler = Callable[..., Coroutine[Any, Any, None]]

HANDLER_TIMEOUT_S = 10.0
SLOW_HANDLER_WARN_S = 5.0
LONG_HANDLER_TIMEOUT_S = 60.0


class PriorityEventBus:
    """Fire-and-forget async event bus with priority queues and error isolation.
    
    Uses an asyncio.PriorityQueue (similar to Apple's GCD dispatch queues)
    to ensure critical voice/UI events preempt background tasks instantly,
    without the overhead of polling or task cancellation.
    """

    __slots__ = (
        "_subscribers", "_handler_snapshot", "_loop",
        "_active_tasks", "_emit_counts", "_worker_task",
        "_queue"
    )

    def __init__(self) -> None:
        self._subscribers: dict[str, list[EventHandler]] = defaultdict(list)
        self._handler_snapshot: dict[str, tuple[EventHandler, ...]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._active_tasks: weakref.WeakSet[asyncio.Task] = weakref.WeakSet()
        self._emit_counts: dict[str, int] = defaultdict(int)
        
        self._queue: asyncio.PriorityQueue | None = None
        self._worker_task: asyncio.Task | None = None

class AsyncEventBus(PriorityEventBus):
    """Alias for backwards compatibility."""
    pass

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        loop = self._loop
        if loop is None or loop.is_closed():
            loop = asyncio.get_running_loop()
            self._loop = loop
        return loop

    def _invalidate_snapshot(self, event: str) -> None:
        self._handler_snapshot.pop(event, None)

    def _get_handlers(self, event: str) -> tuple[EventHandler, ...]:
        snap = self._handler_snapshot.get(event)
        if snap is not None:
            return snap
        handlers = self._subscribers.get(event)
        if not handlers:
            return ()
        snap = tuple(handlers)
        self._handler_snapshot[event] = snap
        return snap

    def on(self, event: str, handler: EventHandler) -> None:
        """Register an async handler for *event*."""
        subs = self._subscribers[event]
        if handler not in subs:
            subs.append(handler)
            self._invalidate_snapshot(event)

    def off(self, event: str, handler: EventHandler) -> None:
        """Unregister a previously registered handler."""
        try:
            self._subscribers[event].remove(handler)
            self._invalidate_snapshot(event)
        except ValueError:
            pass

    def start(self) -> None:
        """Start the priority worker task."""
        if self._worker_task is not None:
            return
        loop = self._get_loop()
        self._queue = asyncio.PriorityQueue()
        self._worker_task = loop.create_task(self._priority_worker())

    async def stop(self) -> None:
        """Stop the worker and flush queues."""
        if self._worker_task:
            self._worker_task.cancel()
            self._worker_task = None
        
        # Cancel all active tasks
        for task in list(self._active_tasks):
            if not task.done():
                task.cancel()
        
        # Flush queue
        if self._queue:
            while not self._queue.empty():
                self._queue.get_nowait()

    async def _priority_worker(self) -> None:
        """Process events by priority to prevent inversion."""
        while True:
            try:
                # PriorityQueue returns the lowest integer first
                priority, _, event, data, emit_type = await self._queue.get()
                self._dispatch(event, data, emit_type)
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Priority worker error: %s", e)
                await asyncio.sleep(0.1)

    def _get_priority(self, event: str) -> int:
        """Determine priority level (0=High, 1=Medium, 2=Low)."""
        high_events = {"speech_final", "speech_partial", "interrupt", "wake_word", "silence_timeout"}
        low_events = {"system_scan", "media_update", "log", "metrics"}
        
        if any(event.startswith(h) or event == h for h in high_events):
            return 0
        if any(event.startswith(l) or event == l for l in low_events):
            return 2
        return 1

    def emit(self, event: str, **data: Any) -> None:
        """Queue event for normal processing."""
        if not self._queue:
            self.start()
        priority = self._get_priority(event)
        self._queue.put_nowait((priority, time.monotonic(), event, data, "normal"))

    def emit_fast(self, event: str, **data: Any) -> None:
        """Queue event for fast processing."""
        if not self._queue:
            self.start()
        priority = self._get_priority(event)
        self._queue.put_nowait((priority, time.monotonic(), event, data, "fast"))

    def emit_long(self, event: str, **data: Any) -> None:
        """Queue event for long processing."""
        if not self._queue:
            self.start()
        priority = self._get_priority(event)
        self._queue.put_nowait((priority, time.monotonic(), event, data, "long"))

    @staticmethod
    async def _long_call(handler: EventHandler, event: str,
                         data: dict[str, Any]) -> None:
        try:
            await asyncio.wait_for(
                handler(**data), timeout=LONG_HANDLER_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.error(
                "Long handler %s TIMED OUT on '%s' (>%.0fs)",
                handler.__qualname__, event, LONG_HANDLER_TIMEOUT_S,
            )
        except Exception:
            logger.exception("Handler %s failed on event '%s'",
                             handler.__qualname__, event)

    @staticmethod
    async def _fast_call(handler: EventHandler, event: str,
                         data: dict[str, Any]) -> None:
        try:
            await handler(**data)
        except Exception:
            logger.exception("Handler %s failed on event '%s'",
                             handler.__qualname__, event)

    @staticmethod
    async def _fast_call_single(handler: EventHandler, event: str,
                                data: dict[str, Any]) -> None:
        """Optimized path for single-handler fast events."""
        try:
            await handler(**data)
        except Exception:
            logger.exception("Handler %s failed on event '%s'",
                             handler.__qualname__, event)

    @staticmethod
    async def _safe_call(handler: EventHandler, event: str, data: dict[str, Any]) -> None:
        t0 = time.perf_counter()
        try:
            await asyncio.wait_for(handler(**data), timeout=HANDLER_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.error(
                "Handler %s TIMED OUT on event '%s' (>%.0fs) -- cancelled",
                handler.__qualname__, event, HANDLER_TIMEOUT_S,
            )
        except Exception:
            logger.exception("Handler %s failed on event '%s'", handler.__qualname__, event)
        finally:
            elapsed = time.perf_counter() - t0
            if elapsed > SLOW_HANDLER_WARN_S:
                logger.warning(
                    "Slow handler %s on '%s': %.1fs",
                    handler.__qualname__, event, elapsed,
                )

    @staticmethod
    def _task_done(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                "Unhandled exception in event bus task: %s: %s",
                type(exc).__name__, exc,
                exc_info=exc,
            )

    @property
    def pending_count(self) -> int:
        """Number of currently active handler tasks (for metrics/debugging)."""
        return len(self._active_tasks)

    @property
    def total_emits(self) -> int:
        """Total events emitted since startup."""
        return sum(self._emit_counts.values())

    def get_event_stats(self) -> dict[str, int]:
        """Per-event emission counts for diagnostics."""
        return dict(self._emit_counts)

    def clear(self) -> None:
        """Remove every subscription (used during shutdown)."""
        self._subscribers.clear()
        self._handler_snapshot.clear()
        self._emit_counts.clear()
