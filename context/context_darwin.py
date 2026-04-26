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


_CODING_APPS = {
    "cursor", "visual studio code", "code", "terminal", "iterm2", "xcode", "warp",
}
_MEETING_APPS = {
    "zoom", "microsoft teams", "slack", "google chrome", "arc", "safari", "firefox",
}
_MEETING_TITLE_HINTS = {
    "zoom meeting", "google meet", "meet -", "microsoft teams", "huddle", "webex",
}
_BROWSER_APPS = {"google chrome", "safari", "firefox", "arc", "brave browser"}
_MEDIA_TITLE_HINTS = {"youtube", "spotify", "music", "soundcloud", "netflix", "prime video"}


def get_foreground_window_info() -> dict[str, Any]:
    """Return frontmost macOS app + title using AppKit and Quartz."""
    info = {"app_name": "", "window_title": "", "pid": 0}
    try:
        from AppKit import NSWorkspace
        import Quartz
    except Exception:
        logger.debug("AppKit/Quartz unavailable for foreground window info", exc_info=True)
        return info

    try:
        ws = NSWorkspace.sharedWorkspace()
        fa = ws.frontmostApplication()
        if fa is None:
            return info
        pid = int(fa.processIdentifier())
        app_name = str(fa.localizedName() or "")
        info["app_name"] = app_name
        info["pid"] = pid

        opt = (
            Quartz.kCGWindowListOptionOnScreenOnly
            | Quartz.kCGWindowListExcludeDesktopElements
        )
        windows: Any = Quartz.CGWindowListCopyWindowInfo(opt, Quartz.kCGNullWindowID)
        for w in windows:
            d = dict(w)
            if int(d.get("kCGWindowOwnerPID", -1)) != pid:
                continue
            if int(d.get("kCGWindowLayer", 0)) != 0:
                continue
            name = str(d.get("kCGWindowName") or "").strip()
            if name:
                info["window_title"] = name
                break
        if not info["window_title"]:
            info["window_title"] = app_name
        return info
    except Exception:
        logger.debug("Foreground window info probe failed", exc_info=True)
        return info


def get_foreground_window_title() -> str:
    """Return the frontmost window title for the active app, or ``\"\"``."""
    return str(get_foreground_window_info().get("window_title") or "")


def quartz_window_titles(*, max_count: int = 64) -> list[str]:
    """Return a list of all on-screen window titles via Quartz.

    Sprint P4.7 (Apr 26 2026): macOS-native replacement for the
    Win32 ``EnumWindows`` path that used to live in
    :py:mod:`core.process_manager`. Empty / system / dock titles
    are filtered out so the output is voice-friendly.

    Returns ``[]`` if Quartz is unavailable (e.g. headless CI box,
    Linux dev container).
    """
    titles: list[str] = []
    try:
        import Quartz
    except Exception:
        logger.debug("Quartz unavailable for window enumeration", exc_info=True)
        return titles

    try:
        opt = (
            Quartz.kCGWindowListOptionOnScreenOnly
            | Quartz.kCGWindowListExcludeDesktopElements
        )
        windows: Any = Quartz.CGWindowListCopyWindowInfo(opt, Quartz.kCGNullWindowID)
        seen: set[str] = set()
        for w in windows:
            d = dict(w)
            if int(d.get("kCGWindowLayer", 0)) != 0:
                continue
            name = str(d.get("kCGWindowName") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            titles.append(name[:120])
            if len(titles) >= max_count:
                break
    except Exception:
        logger.debug("Quartz window enumeration failed", exc_info=True)
    return titles


def classify_activity(
    app_name: str,
    window_title: str,
    *,
    media_playing: bool = False,
    idle_minutes: float = 0.0,
) -> tuple[str, float]:
    """Map frontmost context to coding, browsing, meeting, media, or idle."""
    app = str(app_name or "").strip().lower()
    title = str(window_title or "").strip().lower()
    if idle_minutes >= 5.0:
        return "idle", 0.92
    if media_playing or any(hint in title for hint in _MEDIA_TITLE_HINTS):
        return "media", 0.9
    if app in _CODING_APPS:
        return "coding", 0.95
    if app in _BROWSER_APPS and any(hint in title for hint in _MEETING_TITLE_HINTS):
        return "meeting", 0.88
    if app in {"zoom", "microsoft teams", "webex"}:
        return "meeting", 0.95
    if app in _BROWSER_APPS:
        return "browsing", 0.8
    if app:
        return "browsing", 0.55
    return "idle", 0.35


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
