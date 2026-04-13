"""
ATOM -- Context Engine with privacy filtering.

Gathers environment context (active window, clipboard, CWD, timestamp).
On macOS, uses Quartz + AppKit for the foreground window and NSPasteboard
(or ``pbpaste``) for clipboard. On Windows, uses Win32 via ctypes when
``sys.platform == \"win32\"``. Other platforms return empty window/clipboard.

All platform calls are wrapped in try/except so failures return empty
strings rather than crashing. This module is safe to call from any
thread or async context.

Privacy: clipboard content is scrubbed through privacy_filter.redact()
before inclusion in the context bundle, preventing accidental leakage
of API keys, passwords, tokens, and other secrets to the external LLM.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

from context.privacy_filter import redact as _redact_sensitive


def _get_foreground_window_title() -> str:
    """Active window title. Returns '' on failure or unsupported OS."""
    if sys.platform == "darwin":
        try:
            from context.context_darwin import get_foreground_window_title as _darwin_title

            return _darwin_title()
        except Exception:
            return ""
    if sys.platform != "win32":
        return ""
    try:
        import ctypes
    except Exception:
        return ""
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
    """Clipboard text, truncated to *max_chars*. Returns '' on failure."""
    if sys.platform == "darwin":
        try:
            from context.context_darwin import get_clipboard_text as _darwin_clip

            return _darwin_clip(max_chars)
        except Exception:
            return ""
    if sys.platform != "win32":
        return ""
    try:
        import ctypes
    except Exception:
        return ""
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

    Heuristic: many apps show 'Document - AppName', so we take
    the last segment after ' - '. Falls back to the full title.
    """
    if not window_title:
        return ""
    if " - " in window_title:
        return window_title.rsplit(" - ", 1)[-1].strip()
    return window_title.strip()


def _get_foreground_context() -> dict[str, Any]:
    """Return the active app, window title, and pid when available."""
    info = {
        "active_app": "",
        "window_title": "",
        "frontmost_pid": 0,
    }
    if sys.platform == "darwin":
        try:
            from context.context_darwin import get_foreground_window_info as _darwin_info

            data = _darwin_info()
            info["active_app"] = str(data.get("app_name") or "").strip()
            info["window_title"] = str(data.get("window_title") or "").strip()
            info["frontmost_pid"] = int(data.get("pid") or 0)
            if not info["window_title"]:
                info["window_title"] = info["active_app"]
            return info
        except Exception:
            return info

    title = _get_foreground_window_title()
    info["window_title"] = title
    info["active_app"] = _extract_app_name(title)
    return info


def _classify_non_darwin_activity(
    app_name: str,
    window_title: str,
    *,
    media_playing: bool = False,
    idle_minutes: float = 0.0,
) -> tuple[str, float]:
    """Fallback activity classifier for non-macOS platforms."""
    app = str(app_name or "").strip().lower()
    title = str(window_title or "").strip().lower()
    if idle_minutes >= 5.0:
        return "idle", 0.92
    if media_playing or any(
        hint in title for hint in ("youtube", "spotify", "music", "netflix", "soundcloud")
    ):
        return "media", 0.88
    if app in {
        "cursor",
        "visual studio code",
        "code",
        "terminal",
        "iterm2",
        "warp",
        "xcode",
        "powershell",
        "cmd",
    }:
        return "coding", 0.9
    if app in {"zoom", "microsoft teams", "webex"} or any(
        hint in title for hint in ("zoom meeting", "google meet", "teams", "webex")
    ):
        return "meeting", 0.86
    if app:
        return "browsing", 0.6
    return "idle", 0.3


class ContextEngine:
    """
    Collects environment context for prompt injection.

    All data is gathered lazily on each get_bundle() call.
    Sub-millisecond execution (native calls are fast).
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

    def get_runtime_snapshot(
        self,
        *,
        idle_minutes: float = 0.0,
        media: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Canonical foreground context snapshot for runtime state."""
        if self._enable_window:
            info = _get_foreground_context()
            active_app = str(info.get("active_app") or "")
            window_title = str(info.get("window_title") or "")
            frontmost_pid = int(info.get("frontmost_pid") or 0)
        else:
            active_app = ""
            window_title = ""
            frontmost_pid = 0

        media_payload = dict(media or {})
        media_playing = bool(media_payload.get("playing"))

        activity_type = "idle"
        confidence = 0.0
        if sys.platform == "darwin":
            try:
                from context.context_darwin import classify_activity as _classify_activity

                activity_type, confidence = _classify_activity(
                    active_app,
                    window_title,
                    media_playing=media_playing,
                    idle_minutes=float(idle_minutes or 0.0),
                )
            except Exception:
                activity_type, confidence = _classify_non_darwin_activity(
                    active_app,
                    window_title,
                    media_playing=media_playing,
                    idle_minutes=float(idle_minutes or 0.0),
                )
        else:
            activity_type, confidence = _classify_non_darwin_activity(
                active_app,
                window_title,
                media_playing=media_playing,
                idle_minutes=float(idle_minutes or 0.0),
            )

        return {
            "active_app": active_app,
            "window_title": window_title,
            "frontmost_pid": frontmost_pid,
            "activity_type": activity_type,
            "confidence": float(confidence),
        }

    def get_bundle(self) -> dict[str, str]:
        """
        Gather all context into a single dict for prompt injection.

        Keys: active_app, window_title, clipboard, cwd, timestamp.
        Empty strings for unavailable or disabled fields.
        Clipboard is privacy-filtered to redact secrets before LLM exposure.
        """
        context = self.get_runtime_snapshot()

        return {
            "active_app": str(context.get("active_app") or ""),
            "window_title": str(context.get("window_title") or ""),
            "clipboard": _redact_sensitive(self.get_clipboard()),
            "cwd": os.getcwd(),
            "timestamp": time.strftime("%H:%M:%S"),
        }
