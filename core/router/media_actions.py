"""
ATOM v14 -- Media and volume action handlers.

Handles: set_volume, mute/unmute, play/pause, play_youtube
"""

from __future__ import annotations

import ctypes
import logging
import subprocess
import urllib.parse

logger = logging.getLogger("atom.router.media")


def send_media_play_pause() -> None:
    VK_MEDIA_PLAY_PAUSE = 0xB3
    ctypes.windll.user32.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, 0, 0)
    ctypes.windll.user32.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, 2, 0)


def send_volume_key(vk: int, times: int) -> None:
    for _ in range(max(0, times)):
        ctypes.windll.user32.keybd_event(vk, 0, 0, 0)
        ctypes.windll.user32.keybd_event(vk, 0, 2, 0)


def set_system_volume_percent(percent: int) -> None:
    vk_down = 0xAE
    vk_up = 0xAF
    target_steps = round(max(0, min(100, percent)) / 2)
    send_volume_key(vk_down, 60)
    send_volume_key(vk_up, target_steps)


def send_mute_toggle() -> None:
    VK_VOLUME_MUTE = 0xAD
    ctypes.windll.user32.keybd_event(VK_VOLUME_MUTE, 0, 0, 0)
    ctypes.windll.user32.keybd_event(VK_VOLUME_MUTE, 0, 2, 0)


def play_youtube(query: str, auto_play: bool = True) -> str:
    """Open YouTube search and optionally auto-play the first result.

    1. Opens the YouTube search URL in the default browser.
    2. If auto_play, waits for page load then uses Tab+Enter
       to select and play the first video result.
    """
    url = ("https://www.youtube.com/results?search_query="
           + urllib.parse.quote_plus(query))
    subprocess.Popen(["cmd", "/c", "start", url],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
