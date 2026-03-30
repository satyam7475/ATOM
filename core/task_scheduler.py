"""
ATOM -- Task Scheduler (AI OS Kernel Service).

Persistent reminder and scheduled task engine:
  - One-time reminders ("remind me in 30 minutes to check email")
  - Named tasks with due times
  - Background async loop checks every 30 seconds
  - Persists to logs/tasks.json
  - Emits 'reminder_due' events when tasks fire

Zero external dependencies. Uses asyncio for background polling.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus

logger = logging.getLogger("atom.scheduler")

_TASKS_FILE = Path("logs/tasks.json")
_MAX_TASKS = 100


class ScheduledTask:
    """A single scheduled task / reminder."""

    __slots__ = ("id", "label", "due_ts", "created_ts", "recurring_seconds", "fired")

    def __init__(
        self,
        label: str,
        due_ts: float,
        recurring_seconds: int = 0,
        task_id: str | None = None,
    ) -> None:
        self.id = task_id or uuid.uuid4().hex[:8]
        self.label = label
        self.due_ts = due_ts
        self.created_ts = time.time()
        self.recurring_seconds = recurring_seconds
        self.fired = False

    def is_due(self) -> bool:
        return not self.fired and time.time() >= self.due_ts

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "due_ts": self.due_ts,
            "created_ts": self.created_ts,
            "recurring_seconds": self.recurring_seconds,
            "fired": self.fired,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ScheduledTask:
        task = cls(
            label=d["label"],
            due_ts=d["due_ts"],
            recurring_seconds=d.get("recurring_seconds", 0),
            task_id=d.get("id"),
        )
        task.created_ts = d.get("created_ts", time.time())
        task.fired = d.get("fired", False)
        return task

    def human_due(self) -> str:
        remaining = self.due_ts - time.time()
        if remaining <= 0:
            return "now"
        if remaining < 60:
            return f"{int(remaining)} seconds"
        if remaining < 3600:
            return f"{int(remaining // 60)} minutes"
        hours = int(remaining // 3600)
        mins = int((remaining % 3600) // 60)
        return f"{hours}h {mins}m"


class TaskScheduler:
    """Background task scheduler for ATOM AI OS.

    Polls every 30 seconds, fires events for due reminders,
    handles recurring tasks by rescheduling automatically.
    """

    def __init__(self, bus: AsyncEventBus) -> None:
        self._bus = bus
        self._tasks: list[ScheduledTask] = []
        self._task: asyncio.Task | None = None
        self._stop = False
        self._load()

    def _load(self) -> None:
        try:
            if _TASKS_FILE.exists():
                data = json.loads(_TASKS_FILE.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self._tasks = [
                        ScheduledTask.from_dict(d)
                        for d in data
                        if not d.get("fired", False)
                    ]
                    logger.info("Loaded %d pending tasks", len(self._tasks))
        except Exception:
            logger.debug("No tasks file found, starting fresh")
            self._tasks = []

    def persist(self) -> None:
        try:
            _TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = [t.to_dict() for t in self._tasks]
            _TASKS_FILE.write_text(
                json.dumps(data, indent=2), encoding="utf-8",
            )
            try:
                import os
                os.chmod(_TASKS_FILE, 0o600)
            except OSError:
                pass
        except Exception:
            logger.debug("Failed to save tasks", exc_info=True)

    def add_reminder(self, label: str, delay_seconds: int) -> ScheduledTask:
        """Add a one-time reminder that fires after delay_seconds."""
        task = ScheduledTask(
            label=label, due_ts=time.time() + delay_seconds,
        )
        self._tasks.append(task)
        if len(self._tasks) > _MAX_TASKS:
            self._tasks = [t for t in self._tasks if not t.fired][-_MAX_TASKS:]
        self.persist()
        logger.info("Reminder added: '%s' in %ds (id=%s)",
                     label, delay_seconds, task.id)
        return task

    def add_recurring(self, label: str, interval_seconds: int) -> ScheduledTask:
        """Add a recurring task that fires every interval_seconds."""
        task = ScheduledTask(
            label=label,
            due_ts=time.time() + interval_seconds,
            recurring_seconds=interval_seconds,
        )
        self._tasks.append(task)
        self.persist()
        logger.info("Recurring task: '%s' every %ds (id=%s)",
                     label, interval_seconds, task.id)
        return task

    def list_pending(self) -> list[ScheduledTask]:
        return [t for t in self._tasks if not t.fired]

    def cancel_all(self) -> int:
        count = len([t for t in self._tasks if not t.fired])
        self._tasks = [t for t in self._tasks if t.fired]
        self.persist()
        return count

    def format_pending(self) -> str:
        pending = self.list_pending()
        if not pending:
            return "No pending reminders, Boss."
        parts = [f"You have {len(pending)} reminder{'s' if len(pending) > 1 else ''}:"]
        for t in pending:
            parts.append(f"- {t.label} (due in {t.human_due()})")
        return " ".join(parts)

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop = False
            self._task = asyncio.create_task(self._run())
            logger.info("TaskScheduler started")

    def stop(self) -> None:
        self._stop = True
        if self._task and not self._task.done():
            self._task.cancel()
        self.persist()
        logger.info("TaskScheduler stopped")

    async def _run(self) -> None:
        try:
            while not self._stop:
                self._check_due()
                await asyncio.sleep(30)
        except asyncio.CancelledError:
            pass

    def _check_due(self) -> None:
        fired_any = False
        for task in self._tasks:
            if task.is_due():
                task.fired = True
                fired_any = True
                self._bus.emit_long(
                    "reminder_due", label=task.label, task_id=task.id,
                )
                logger.info("Reminder fired: '%s' (id=%s)", task.label, task.id)

                if task.recurring_seconds > 0:
                    new_task = ScheduledTask(
                        label=task.label,
                        due_ts=time.time() + task.recurring_seconds,
                        recurring_seconds=task.recurring_seconds,
                    )
                    self._tasks.append(new_task)
                    logger.info("Recurring task rescheduled: '%s'", task.label)

        if fired_any:
            self._tasks = [t for t in self._tasks if not t.fired]
            self.persist()
