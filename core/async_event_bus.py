"""
Async Event Bus -- lightweight pub/sub backbone for ATOM v14.

All inter-module communication flows through this bus.
Subscribers are invoked as asyncio.Tasks so emitters never block.

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


class AsyncEventBus:
    """Fire-and-forget async event bus with error isolation and task lifecycle management."""

    __slots__ = ("_subscribers", "_loop", "_active_tasks")

    def __init__(self) -> None:
        self._subscribers: dict[str, list[EventHandler]] = defaultdict(list)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._active_tasks: weakref.WeakSet[asyncio.Task] = weakref.WeakSet()

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.get_running_loop()
        return self._loop

    def on(self, event: str, handler: EventHandler) -> None:
        """Register an async handler for *event*."""
        if handler not in self._subscribers[event]:
            self._subscribers[event].append(handler)

    def off(self, event: str, handler: EventHandler) -> None:
        """Unregister a previously registered handler."""
        try:
            self._subscribers[event].remove(handler)
        except ValueError:
            pass

    def emit(self, event: str, **data: Any) -> None:
        """
        Emit *event* to all subscribers.

        Each handler runs as an independent asyncio.Task with a timeout so:
        - The emitter is never blocked.
        - A failing handler cannot crash others.
        - A hung handler is cancelled after HANDLER_TIMEOUT_S.
        """
        handlers = list(self._subscribers.get(event, []))
        if not handlers:
            return

        loop = self._get_loop()
        for handler in handlers:
            task = loop.create_task(self._safe_call(handler, event, data))
            self._active_tasks.add(task)
            task.add_done_callback(self._task_done)

    def emit_fast(self, event: str, **data: Any) -> None:
        """Lightweight emit for trivial handlers (<1ms, guaranteed safe).

        Skips the timeout wrapper and slow-handler tracking. Use only for
        handlers that do simple in-memory work (metrics, logging, UI state).
        """
        handlers = list(self._subscribers.get(event, []))
        if not handlers:
            return

        loop = self._get_loop()
        for handler in handlers:
            task = loop.create_task(self._fast_call(handler, event, data))
            self._active_tasks.add(task)
            task.add_done_callback(self._task_done)

    def emit_long(self, event: str, **data: Any) -> None:
        """Emit for long-running handlers like TTS playback (up to 60s).

        Uses a generous timeout to prevent cancellation mid-audio.
        No slow-handler warnings since long runtime is expected.
        """
        handlers = list(self._subscribers.get(event, []))
        if not handlers:
            return

        loop = self._get_loop()
        for handler in handlers:
            task = loop.create_task(
                self._long_call(handler, event, data))
            self._active_tasks.add(task)
            task.add_done_callback(self._task_done)

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

    def clear(self) -> None:
        """Remove every subscription (used during shutdown)."""
        self._subscribers.clear()
