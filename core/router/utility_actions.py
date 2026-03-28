"""
ATOM -- Utility action handlers.

Handles: minimize_window, maximize_window, switch_window,
         read_clipboard, timer
"""

from __future__ import annotations

import asyncio
import ctypes
import logging

logger = logging.getLogger("atom.router.utility")


def minimize_active_window() -> None:
    SW_MINIMIZE = 6
    hwnd = ctypes.windll.user32.GetForegroundWindow()
    ctypes.windll.user32.ShowWindow(hwnd, SW_MINIMIZE)


def maximize_active_window() -> None:
    SW_MAXIMIZE = 3
    hwnd = ctypes.windll.user32.GetForegroundWindow()
    ctypes.windll.user32.ShowWindow(hwnd, SW_MAXIMIZE)


def switch_active_window() -> None:
    VK_MENU = 0x12
    VK_TAB = 0x09
    ctypes.windll.user32.keybd_event(VK_MENU, 0, 0, 0)
    ctypes.windll.user32.keybd_event(VK_TAB, 0, 0, 0)
    ctypes.windll.user32.keybd_event(VK_TAB, 0, 2, 0)
    ctypes.windll.user32.keybd_event(VK_MENU, 0, 2, 0)


def read_clipboard_text() -> str:
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


async def run_timer(seconds: int, label: str, bus) -> None:
    """Background timer that speaks when complete."""
    await asyncio.sleep(seconds)
    bus.emit_long("response_ready",
                  text=f"Time's up, boss! Your {label} timer is done.")
