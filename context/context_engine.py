"""
ATOM -- Context Engine with privacy filtering.

Gathers environment context (active window, clipboard, CWD, timestamp)
using Windows API via ctypes. Zero external dependencies.

All Win32 calls are wrapped in try/except so failures return empty
strings rather than crashing. This module is safe to call from any
thread or async context.

Privacy: clipboard content is scrubbed through privacy_filter.redact()
before inclusion in the context bundle, preventing accidental leakage
of API keys, passwords, tokens, and other secrets to the external LLM.
"""

from __future__ import annotations

import os
import time
from typing import Any

from context.privacy_filter import redact as _redact_sensitive
import ctypes


def _get_foreground_window_title() -> str:
    """Active window title via user32.GetWindowTextW. Returns '' on failure."""
    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        hwnd = user32.GetForegroundWindow()
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value
    except Exception:
        return ""


def _get_clipboard_text(max_chars: int = 500) -> str:
    """Clipboard text via user32 OpenClipboard/GetClipboardData.

    Truncated to max_chars. Returns '' on failure or non-text clipboard.
    """
    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

        CF_UNICODETEXT = 13

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
                return text[:max_chars] if text else ""
            finally:
                kernel32.GlobalUnlock(handle)
        finally:
            user32.CloseClipboard()
    except Exception:
        return ""


def _extract_app_name(window_title: str) -> str:
    """Extract the application name from a window title.

    Heuristic: many Windows apps show 'Document - AppName', so we take
    the last segment after ' - '. Falls back to the full title.
    """
    if not window_title:
        return ""
    if " - " in window_title:
        return window_title.rsplit(" - ", 1)[-1].strip()
    return window_title.strip()


class ContextEngine:
    """
    Collects environment context for prompt injection.

    All data is gathered lazily on each get_bundle() call.
    Sub-millisecond execution (ctypes calls are fast).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = (config or {}).get("context", {})
        self._enable_clipboard: bool = cfg.get("enable_clipboard", True)
        self._enable_window: bool = cfg.get("enable_active_window", True)
        self._clipboard_max: int = cfg.get("clipboard_max_chars", 500)

    def get_active_window(self) -> str:
        """Return the active window title, or '' if disabled/failed."""
        if not self._enable_window:
            return ""
        return _get_foreground_window_title()

    def get_clipboard(self) -> str:
        """Return clipboard text (truncated), or '' if disabled/failed."""
        if not self._enable_clipboard:
            return ""
        return _get_clipboard_text(self._clipboard_max)

    def get_bundle(self) -> dict[str, str]:
        """
        Gather all context into a single dict for prompt injection.

        Keys: active_app, window_title, clipboard, cwd, timestamp.
        Empty strings for unavailable or disabled fields.
        Clipboard is privacy-filtered to redact secrets before LLM exposure.
        """
        window_title = self.get_active_window()
        app_name = _extract_app_name(window_title)

        return {
            "active_app": app_name,
            "window_title": window_title,
            "clipboard": _redact_sensitive(self.get_clipboard()),
            "cwd": os.getcwd(),
            "timestamp": time.strftime("%H:%M:%S"),
        }
