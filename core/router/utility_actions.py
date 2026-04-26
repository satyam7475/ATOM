"""ATOM -- Utility action handlers (macOS-only).

Handles: minimize_window, maximize_window, switch_window,
         next_window_in_app, switch_space, read_clipboard, timer.

Sprint P4.7 (Apr 26 2026): Windows ``win32`` ctypes branches removed.
ATOM ships only on Apple Silicon, so the Windows code paths were
unreachable. See ``docs/ATOM_NEXT_STEPS_PLAN.md`` § P4.7.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys

logger = logging.getLogger("atom.router.utility")


def minimize_active_window() -> None:
    if sys.platform == "darwin":
        subprocess.run(
            [
                "osascript",
                "-e",
                'tell application "System Events" to set miniaturized of first window of (first application process whose frontmost is true) to true',
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def maximize_active_window() -> None:
    if sys.platform == "darwin":
        subprocess.run(
            [
                "osascript",
                "-e",
                (
                    'tell application "Finder" to set screenBounds to bounds of window of desktop\n'
                    'tell application "System Events"\n'
                    'tell (first application process whose frontmost is true)\n'
                    'set position of first window to {item 1 of screenBounds, item 2 of screenBounds}\n'
                    "set size of first window to {(item 3 of screenBounds) - (item 1 of screenBounds), "
                    "(item 4 of screenBounds) - (item 2 of screenBounds)}\n"
                    "end tell\n"
                    "end tell"
                ),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def switch_active_window() -> None:
    """Switch to the next *application* (macOS: Cmd+Tab)."""
    if sys.platform == "darwin":
        subprocess.run(
            [
                "osascript",
                "-e",
                'tell application "System Events" to keystroke tab using command down',
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def next_window_in_app() -> None:
    """Cycle to the next window of the *current* app.

    macOS: Cmd+\\` (the standard "Move focus to next window" shortcut).

    This complements ``switch_active_window`` which switches between
    applications. Voice commands like "next window" should reach
    *this* function so users can flip between, say, two Chrome
    windows without leaving Chrome.
    """
    if sys.platform == "darwin":
        subprocess.run(
            [
                "osascript",
                "-e",
                'tell application "System Events" to keystroke "`" using command down',
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def switch_space(direction: str = "right") -> None:
    """Move to the adjacent macOS Mission Control desktop space.

    direction: "right" (Ctrl+→) for the next space, "left" (Ctrl+←)
    for the previous. The user must enable "Use keyboard shortcuts to
    switch Spaces" under System Settings -> Keyboard -> Keyboard
    Shortcuts -> Mission Control. We log a hint when the keystroke
    appears to no-op, but we cannot verify space changes from
    AppleScript.
    """
    arrow = "right arrow" if direction.lower() == "right" else "left arrow"
    if sys.platform == "darwin":
        subprocess.run(
            [
                "osascript",
                "-e",
                f'tell application "System Events" to key code '
                f'{124 if direction.lower() == "right" else 123} '
                f'using control down',
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        logger.info("Switch space %s (%s)", direction, arrow)


def read_clipboard_text() -> str:
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["pbpaste"], capture_output=True, text=True, check=False,
            )
            if result.returncode != 0:
                return ""
            text = result.stdout or ""
            return text[:300] if text else ""
        except Exception:
            return ""
    return ""


async def run_timer(seconds: int, label: str, bus) -> None:
    """Background timer that speaks when complete."""
    await asyncio.sleep(seconds)
    bus.emit_long("response_ready",
                  text=f"Time's up, boss! Your {label} timer is done.")
