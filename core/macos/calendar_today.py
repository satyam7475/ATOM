"""
ATOM -- macOS Calendar.app "today's events" helper.

Thin wrapper around an AppleScript query against the user's default
calendars. Returns a list of simple event strings formatted as::

    "<title> at <HH:MM AM/PM>"

Both a blocking (``fetch_today_events_sync``) and an awaitable
(``fetch_today_events``) form are exposed so callers from sync router
actions *and* async proactive services can share one AppleScript
payload.

Graceful degradation is the contract: if ``osascript`` is missing, the
timeout expires, the user has denied Calendar permission, or
Calendar.app returns nothing, we return ``[]``. We *never* raise.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from typing import List

logger = logging.getLogger("atom.macos.calendar_today")


_APPLESCRIPT = (
    'set todayStart to current date\n'
    'set hours of todayStart to 0\n'
    'set minutes of todayStart to 0\n'
    'set seconds of todayStart to 0\n'
    'set todayEnd to todayStart + (1 * days)\n'
    'set output to ""\n'
    'tell application "Calendar"\n'
    '  repeat with cal in calendars\n'
    '    try\n'
    '      set todaysEvents to (every event of cal whose start date '
    '≥ todayStart and start date < todayEnd)\n'
    '      repeat with ev in todaysEvents\n'
    '        set evStart to start date of ev\n'
    '        set evTitle to summary of ev\n'
    '        set timeStr to (time string of evStart)\n'
    '        set output to output & evTitle & " at " & timeStr & "\\n"\n'
    '      end repeat\n'
    '    end try\n'
    '  end repeat\n'
    'end tell\n'
    'return output\n'
)


def _parse(raw: str) -> List[str]:
    if not raw:
        return []
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    events = [ev for ev in lines if not ev.lower().startswith("error")]
    return events[:10]


def fetch_today_events_sync(timeout_s: float = 3.0) -> List[str]:
    """Blocking form. Safe to call from a sync router action."""
    try:
        result = subprocess.run(
            ["osascript", "-e", _APPLESCRIPT],
            capture_output=True,
            text=True,
            timeout=max(0.5, float(timeout_s)),
            check=False,
        )
    except FileNotFoundError:
        return []
    except subprocess.TimeoutExpired:
        logger.debug("calendar_today sync fetch timed out")
        return []
    except Exception:
        logger.debug("calendar_today sync fetch errored", exc_info=True)
        return []
    return _parse(result.stdout or "")


async def fetch_today_events(timeout_s: float = 3.0) -> List[str]:
    """Async form. Shells out to ``osascript`` without blocking the loop."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "osascript", "-e", _APPLESCRIPT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return []
    except Exception:
        logger.debug("calendar_today async launch failed", exc_info=True)
        return []

    try:
        stdout, _ = await asyncio.wait_for(
            proc.communicate(),
            timeout=max(0.5, float(timeout_s)),
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        logger.debug("calendar_today async fetch timed out")
        return []
    except Exception:
        logger.debug("calendar_today async fetch errored", exc_info=True)
        return []

    raw = (stdout or b"").decode("utf-8", errors="replace")
    return _parse(raw)


__all__ = ["fetch_today_events", "fetch_today_events_sync"]
