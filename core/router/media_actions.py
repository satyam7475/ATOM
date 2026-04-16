"""
ATOM -- Media and volume action handlers (cross-platform).

Handles: set_volume, mute/unmute, play/pause, play_youtube
"""

from __future__ import annotations

import logging
import subprocess
import sys
import urllib.parse

logger = logging.getLogger("atom.router.media")

_IS_MAC = sys.platform == "darwin"
_IS_WIN = sys.platform == "win32"


def _open_url(url: str) -> None:
    """Open a URL in the default browser, platform-aware."""
    if _IS_MAC:
        subprocess.Popen(["open", url],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif _IS_WIN:
        subprocess.Popen(["cmd", "/c", "start", url],
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
            'tell application "System Events" to key code 16 using {command down}'
        )
    elif _IS_WIN:
        import ctypes
        VK_MEDIA_PLAY_PAUSE = 0xB3
        ctypes.windll.user32.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, 0, 0)
        ctypes.windll.user32.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, 2, 0)


def set_system_volume_percent(percent: int) -> None:
    vol = max(0, min(100, percent))
    if _IS_MAC:
        _osascript(f"set volume output volume {vol}")
        logger.info("macOS volume set to %d%%", vol)
    elif _IS_WIN:
        import ctypes
        vk_down, vk_up = 0xAE, 0xAF
        target_steps = round(vol / 2)
        for _ in range(60):
            ctypes.windll.user32.keybd_event(vk_down, 0, 0, 0)
            ctypes.windll.user32.keybd_event(vk_down, 0, 2, 0)
        for _ in range(target_steps):
            ctypes.windll.user32.keybd_event(vk_up, 0, 0, 0)
            ctypes.windll.user32.keybd_event(vk_up, 0, 2, 0)


def send_mute_toggle() -> None:
    if _IS_MAC:
        current = _osascript("output muted of (get volume settings)")
        new_state = "false" if current == "true" else "true"
        _osascript(f"set volume output muted {new_state}")
        logger.info("macOS mute toggled to %s", new_state)
    elif _IS_WIN:
        import ctypes
        VK_VOLUME_MUTE = 0xAD
        ctypes.windll.user32.keybd_event(VK_VOLUME_MUTE, 0, 0, 0)
        ctypes.windll.user32.keybd_event(VK_VOLUME_MUTE, 0, 2, 0)


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
