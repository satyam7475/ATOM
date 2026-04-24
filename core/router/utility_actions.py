"""
ATOM -- Utility action handlers.

Handles: minimize_window, maximize_window, switch_window,
         next_window_in_app, switch_space, read_clipboard, timer
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
    elif sys.platform == "win32":
        import ctypes

        SW_MINIMIZE = 6
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        ctypes.windll.user32.ShowWindow(hwnd, SW_MINIMIZE)


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
    elif sys.platform == "win32":
        import ctypes

        SW_MAXIMIZE = 3
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        ctypes.windll.user32.ShowWindow(hwnd, SW_MAXIMIZE)


def switch_active_window() -> None:
    """Switch to the next *application* (macOS: Cmd+Tab, Windows: Alt+Tab)."""
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
    elif sys.platform == "win32":
        import ctypes

        VK_MENU = 0x12
        VK_TAB = 0x09
        ctypes.windll.user32.keybd_event(VK_MENU, 0, 0, 0)
        ctypes.windll.user32.keybd_event(VK_TAB, 0, 0, 0)
        ctypes.windll.user32.keybd_event(VK_TAB, 0, 2, 0)
        ctypes.windll.user32.keybd_event(VK_MENU, 0, 2, 0)


def next_window_in_app() -> None:
    """Cycle to the next window of the *current* app.

    macOS: Cmd+\` (the standard "Move focus to next window" shortcut).
    Windows: Alt+Esc (cycles open windows in z-order).

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
    elif sys.platform == "win32":
        import ctypes

        VK_MENU = 0x12
        VK_ESCAPE = 0x1B
        ctypes.windll.user32.keybd_event(VK_MENU, 0, 0, 0)
        ctypes.windll.user32.keybd_event(VK_ESCAPE, 0, 0, 0)
        ctypes.windll.user32.keybd_event(VK_ESCAPE, 0, 2, 0)
        ctypes.windll.user32.keybd_event(VK_MENU, 0, 2, 0)


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
    elif sys.platform == "win32":
        import ctypes

        # Win+Ctrl+→/← cycles virtual desktops on Windows 10+.
        VK_LWIN = 0x5B
        VK_CONTROL = 0x11
        VK_RIGHT = 0x27
        VK_LEFT = 0x25
        key = VK_RIGHT if direction.lower() == "right" else VK_LEFT
        u32 = ctypes.windll.user32
        u32.keybd_event(VK_LWIN, 0, 0, 0)
        u32.keybd_event(VK_CONTROL, 0, 0, 0)
        u32.keybd_event(key, 0, 0, 0)
        u32.keybd_event(key, 0, 2, 0)
        u32.keybd_event(VK_CONTROL, 0, 2, 0)
        u32.keybd_event(VK_LWIN, 0, 2, 0)


def read_clipboard_text() -> str:
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["pbpaste"], capture_output=True, text=True, check=False
            )
            if result.returncode != 0:
                return ""
            text = result.stdout or ""
            return text[:300] if text else ""
        except Exception:
            return ""
    if sys.platform == "win32":
        import ctypes

        try:
            CF_UNICODETEXT = 13
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            if not user32.OpenClipboard(0):
                return ""
            try:
                handle = user32.GetClipboardData(CF_UNICODETEXT)
                if not handle:
                    return ""
                kernel32.GlobalLock.restype = ctypes.c_void_p
                ptr = kernel32.GlobalLock(handle)
                if not ptr:
                    return ""
                try:
                    text = ctypes.wstring_at(ptr)
                    return text[:300] if text else ""
                finally:
                    kernel32.GlobalUnlock(handle)
            finally:
                user32.CloseClipboard()
        except Exception:
            return ""
    return ""


async def run_timer(seconds: int, label: str, bus) -> None:
    """Background timer that speaks when complete."""
    await asyncio.sleep(seconds)
    bus.emit_long("response_ready",
                  text=f"Time's up, boss! Your {label} timer is done.")
