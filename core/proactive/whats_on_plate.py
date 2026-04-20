"""
ATOM -- "What's on my plate" skill (Sprint D2).

On-demand summary of the user's immediate obligations: today's calendar
events + any pending ATOM reminders (from ``TaskScheduler``). Designed
to be invoked from the intent router's sync dispatch path, so it uses
the **blocking** calendar helper (``fetch_today_events_sync``) with a
short timeout.

Email support is intentionally *not* wired yet — ATOM does not have an
email reader in-tree. The composer leaves a slot so we can drop in an
integration (AppleScript to Mail, Gmail API, IMAP) without touching
the intent layer.

Owner: Satyam
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Iterable

logger = logging.getLogger("atom.proactive.whats_on_plate")


def _weekday_and_date() -> str:
    now = datetime.now()
    return now.strftime("%A, %B %-d")


def _format_reminders(reminders: Iterable[Any]) -> str:
    items = list(reminders)
    if not items:
        return ""
    if len(items) == 1:
        t = items[0]
        label = getattr(t, "label", "a task")
        due = ""
        human_due = getattr(t, "human_due", None)
        if callable(human_due):
            try:
                due = human_due()
            except Exception:
                due = ""
        if due:
            return f"One reminder: {label}, due in {due}."
        return f"One reminder: {label}."
    tops = []
    for t in items[:3]:
        label = getattr(t, "label", None) or "unnamed"
        tops.append(label)
    head = ", ".join(tops)
    return f"{len(items)} reminders on deck — {head}."


def _format_events(events: list[str]) -> str:
    if not events:
        return ""
    if len(events) == 1:
        return f"One calendar event today: {events[0]}."
    head = ", ".join(events[:3])
    if len(events) <= 3:
        return f"{len(events)} calendar events today: {head}."
    return f"{len(events)} calendar events today, starting with {events[0]}."


def _empty_plate_line() -> str:
    return (
        "Your plate looks clear, Boss. "
        "No calendar events or reminders queued right now."
    )


def generate_plate_summary_sync(
    *,
    task_scheduler: Any = None,
    include_calendar: bool = True,
    include_reminders: bool = True,
    calendar_timeout_s: float = 3.0,
) -> str:
    """Compose a spoken answer for "what's on my plate today?".

    Blocking. Safe to call from a sync router action.
    """
    pieces: list[str] = []
    pieces.append(f"Here's your plate for {_weekday_and_date()}.")

    events: list[str] = []
    if include_calendar:
        try:
            from core.macos.calendar_today import fetch_today_events_sync
            events = fetch_today_events_sync(calendar_timeout_s) or []
        except Exception:
            logger.debug("calendar fetch failed in plate summary", exc_info=True)
            events = []
    ev_line = _format_events(events)
    if ev_line:
        pieces.append(ev_line)

    reminders_line = ""
    if include_reminders and task_scheduler is not None:
        try:
            pending = task_scheduler.list_pending() or []
        except Exception:
            logger.debug("reminder fetch failed in plate summary", exc_info=True)
            pending = []
        reminders_line = _format_reminders(pending)
    if reminders_line:
        pieces.append(reminders_line)

    if not ev_line and not reminders_line:
        return _empty_plate_line()

    return " ".join(pieces)


__all__ = ["generate_plate_summary_sync"]
