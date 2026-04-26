"""ATOM -- Media and volume action handlers (macOS-only).

Handles: set_volume, mute/unmute, play/pause, play_youtube.

Sprint P4.7 (Apr 26 2026): Windows ``win32`` ctypes branches removed.
A Linux ``xdg-open`` fallback is retained for ``_open_url`` because
it's a single line and survives without imports -- useful in CI/dev
boxes that aren't Apple Silicon. See
``docs/ATOM_NEXT_STEPS_PLAN.md`` § P4.7.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import urllib.parse

logger = logging.getLogger("atom.router.media")

_IS_MAC = sys.platform == "darwin"


def _open_url(url: str) -> None:
    """Open a URL in the default browser. macOS uses ``open``; the
    Linux ``xdg-open`` fallback keeps headless CI / dev boxes happy."""
    if _IS_MAC:
        subprocess.Popen(["open", url],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.Popen(["xdg-open", url],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _osascript(script: str) -> str:
    """Run an AppleScript snippet and return stdout."""
    try:
        r = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip()
    except Exception as exc:
        logger.debug("osascript failed: %s", exc)
        return ""


def send_media_play_pause() -> None:
    if _IS_MAC:
        _osascript(
            'tell application "System Events" to key code 16 using {command down}',
        )


def set_system_volume_percent(percent: int) -> None:
    vol = max(0, min(100, percent))
    if _IS_MAC:
        _osascript(f"set volume output volume {vol}")
        logger.info("macOS volume set to %d%%", vol)


def send_mute_toggle() -> None:
    if _IS_MAC:
        current = _osascript("output muted of (get volume settings)")
        new_state = "false" if current == "true" else "true"
        _osascript(f"set volume output muted {new_state}")
        logger.info("macOS mute toggled to %s", new_state)


def play_youtube(query: str, auto_play: bool = True) -> str:
    """Open YouTube search in the default browser."""
    url = ("https://www.youtube.com/results?search_query="
           + urllib.parse.quote_plus(query))
    _open_url(url)
    logger.info("Play YouTube query: %s (auto_play=%s)", query, auto_play)

    if auto_play:
        import threading

        def _auto_select() -> None:
            import time
            time.sleep(4)
            try:
                import pyautogui
                pyautogui.FAILSAFE = True
                for _ in range(6):
                    pyautogui.press("tab")
                    time.sleep(0.15)
                pyautogui.press("enter")
                logger.info("YouTube auto-play: Tab+Enter sent")
            except Exception:
                logger.debug("YouTube auto-play failed", exc_info=True)

        threading.Thread(target=_auto_select, daemon=True).start()

    return url
