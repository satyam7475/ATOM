"""
ATOM -- Background Task Manager.

Centralized tracking for named async background tasks. Replaces scattered
``asyncio.create_task()`` calls with a registry that supports naming,
status queries, cancellation, and listing.

Usage:
    mgr = TaskManager()
    await mgr.run("download_update", some_coroutine())
    mgr.status("download_update")    # -> "running" | "done" | "cancelled" | "failed"
    mgr.list_active()                # -> ["download_update"]
    mgr.cancel("download_update")    # -> True if was running
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Coroutine

logger = logging.getLogger("atom.task_manager")


@dataclass
class _TaskRecord:
    name: str
    task: asyncio.Task[Any]
    status: str = "running"


class TaskManager:
    """Named background task registry with cancel/status/list."""

    __slots__ = ("_tasks",)

    def __init__(self) -> None:
        self._tasks: dict[str, _TaskRecord] = {}

    async def run(self, name: str, coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        """Create a named background task. Cancels any existing task with the same name."""
        if name in self._tasks:
            existing = self._tasks[name]
            if not existing.task.done():
                logger.info("Task '%s' already running — cancelling old instance", name)
                existing.task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(existing.task), timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                    pass

        task = asyncio.create_task(coro, name=f"atom:{name}")

        record = _TaskRecord(name=name, task=task)
        self._tasks[name] = record

        task.add_done_callback(lambda t: self._on_done(name, t))
        logger.info("Background task started: '%s'", name)
        return task

    def cancel(self, name: str) -> bool:
        """Cancel a running task by name. Returns True if it was running."""
        record = self._tasks.get(name)
        if record is None or record.task.done():
            return False
        record.task.cancel()
        record.status = "cancelled"
        logger.info("Background task cancelled: '%s'", name)
        return True

    def status(self, name: str) -> str:
        """Return status of a task: 'running', 'done', 'cancelled', 'failed', or 'unknown'."""
        record = self._tasks.get(name)
        if record is None:
            return "unknown"
        if not record.task.done():
            return "running"
        return record.status

    def list_active(self) -> list[str]:
        """Return names of all currently running tasks."""
        return [
            name for name, rec in self._tasks.items()
            if not rec.task.done()
        ]

    def list_all(self) -> list[dict[str, str]]:
        """Return all tracked tasks with their statuses."""
        return [
            {"name": name, "status": self.status(name)}
            for name in self._tasks
        ]

    def _on_done(self, name: str, task: asyncio.Task[Any]) -> None:
        """Callback when a task finishes."""
        record = self._tasks.get(name)
        if record is None:
            return

        if task.cancelled():
            record.status = "cancelled"
        elif task.exception() is not None:
            record.status = "failed"
            logger.warning(
                "Background task '%s' failed: %s",
                name, task.exception(),
            )
        else:
            record.status = "done"
            logger.info("Background task completed: '%s'", name)

    def cleanup_finished(self) -> int:
        """Remove records of finished tasks. Returns count removed."""
        finished = [n for n, r in self._tasks.items() if r.task.done()]
        for n in finished:
            del self._tasks[n]
        return len(finished)

    async def cancel_all(self) -> int:
        """Cancel all running tasks. Returns count cancelled."""
        count = 0
        for name in list(self._tasks):
            if self.cancel(name):
                count += 1
        return count

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            "total_tracked": len(self._tasks),
            "active": self.list_active(),
            "all": self.list_all(),
        }


__all__ = ["TaskManager"]
