"""
macOS perception helpers for ContextEngine.

Foreground window title via Quartz (CGWindowListCopyWindowInfo) + AppKit
(NSWorkspace frontmost PID). Clipboard via AppKit NSPasteboard, with
``pbpaste`` fallback if string types are unavailable.

Safe to call from any thread: failures yield empty strings (logged at debug).
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any

logger = logging.getLogger("atom.context.darwin")


def get_foreground_window_title() -> str:
    """Return the frontmost window title for the active app, or ``\"\"``."""
    try:
        from AppKit import NSWorkspace
        import Quartz
    except Exception:
        logger.debug("AppKit/Quartz unavailable for foreground window", exc_info=True)
        return ""

    try:
        ws = NSWorkspace.sharedWorkspace()
        fa = ws.frontmostApplication()
        if fa is None:
            return ""
        pid = int(fa.processIdentifier())
        fallback = str(fa.localizedName() or "")

        opt = (
            Quartz.kCGWindowListOptionOnScreenOnly
            | Quartz.kCGWindowListExcludeDesktopElements
        )
        windows: Any = Quartz.CGWindowListCopyWindowInfo(
            opt,
            Quartz.kCGNullWindowID,
        )
        for w in windows:
            d = dict(w)
            if int(d.get("kCGWindowOwnerPID", -1)) != pid:
                continue
            if int(d.get("kCGWindowLayer", 0)) != 0:
                continue
            name = d.get("kCGWindowName") or ""
            if name:
                return str(name)
        return fallback
    except Exception:
        logger.debug("Foreground window title probe failed", exc_info=True)
        return ""


def get_clipboard_text(max_chars: int = 500) -> str:
    """Return UTF-8 clipboard text, truncated to *max_chars*."""
    try:
        from AppKit import NSPasteboard
    except Exception:
        logger.debug("AppKit unavailable for clipboard", exc_info=True)
        return _clipboard_pbpaste(max_chars)

    try:
        pb = NSPasteboard.generalPasteboard()
        for type_id in (
            "public.utf8-plain-text",
            "NSStringPboardType",
            "public.plain-text",
        ):
            text = pb.stringForType_(type_id)
            if text:
                s = str(text)
                return s[:max_chars] if len(s) > max_chars else s
    except Exception:
        logger.debug("NSPasteboard read failed", exc_info=True)

    return _clipboard_pbpaste(max_chars)


def _clipboard_pbpaste(max_chars: int) -> str:
    try:
        r = subprocess.run(
            ["pbpaste"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if r.returncode != 0 or not r.stdout:
            return ""
        s = r.stdout
        return s[:max_chars] if len(s) > max_chars else s
    except Exception:
        logger.debug("pbpaste fallback failed", exc_info=True)
        return ""
