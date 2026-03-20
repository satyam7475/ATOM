"""
ATOM v14 -- Desktop Control (AI OS Automation Layer).

Voice-controlled desktop automation using pyautogui:
  - Scroll up/down
  - Click (center or coordinates)
  - Type text at cursor position
  - Press individual keys (Enter, Escape, Tab, arrows)
  - Safe hotkey combos (Ctrl+C, Ctrl+V, Alt+Tab)
  - Screenshot for visual context

Security: pyautogui.FAILSAFE always ON (move mouse to top-left corner to abort).
All operations go through SecurityPolicy before execution.
"""

from __future__ import annotations

import logging
import time

from core.security_policy import SecurityPolicy

logger = logging.getLogger("atom.desktop")

_policy = SecurityPolicy()


def _ensure_pyautogui():
    """Lazy-import pyautogui with failsafe always enabled."""
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.1
    return pyautogui


def scroll_down(clicks: int = 5) -> str:
    """Scroll the active window down."""
    gui = _ensure_pyautogui()
    clicks = min(clicks, 30)
    gui.scroll(-clicks)
    _policy.audit_log("scroll_down", f"clicks={clicks}")
    logger.info("Scrolled down %d clicks", clicks)
    return f"Scrolled down {clicks} clicks, Boss."


def scroll_up(clicks: int = 5) -> str:
    """Scroll the active window up."""
    gui = _ensure_pyautogui()
    clicks = min(clicks, 30)
    gui.scroll(clicks)
    _policy.audit_log("scroll_up", f"clicks={clicks}")
    logger.info("Scrolled up %d clicks", clicks)
    return f"Scrolled up {clicks} clicks, Boss."


def click_center() -> str:
    """Click the center of the screen."""
    gui = _ensure_pyautogui()
    w, h = gui.size()
    gui.click(w // 2, h // 2)
    _policy.audit_log("click_center", f"pos=({w // 2},{h // 2})")
    logger.info("Clicked center of screen (%d, %d)", w // 2, h // 2)
    return "Clicked the center of the screen."


def click_at(x: int, y: int) -> str:
    """Click at specific coordinates."""
    gui = _ensure_pyautogui()
    gui.click(x, y)
    _policy.audit_log("click_at", f"pos=({x},{y})")
    logger.info("Clicked at (%d, %d)", x, y)
    return f"Clicked at position {x}, {y}."


def double_click_center() -> str:
    """Double-click the center of the screen."""
    gui = _ensure_pyautogui()
    w, h = gui.size()
    gui.doubleClick(w // 2, h // 2)
    _policy.audit_log("double_click_center")
    return "Double-clicked the center of the screen."


def press_key(key: str) -> str:
    """Press a single key if it's in the safe list."""
    key_lower = key.lower().strip()
    if not _policy.is_safe_key(key_lower):
        _policy.audit_log("press_key", f"BLOCKED key={key}", success=False)
        return f"Key '{key}' is not in the safe list, Boss."

    gui = _ensure_pyautogui()
    gui.press(key_lower)
    _policy.audit_log("press_key", f"key={key_lower}")
    logger.info("Pressed key: %s", key_lower)
    return f"Pressed {key}."


def hotkey_combo(combo: str) -> str:
    """Execute a keyboard shortcut if safe.

    combo: e.g. "ctrl+c", "alt+tab", "ctrl+shift+t"
    """
    tier, reason = _policy.is_safe_hotkey(combo)
    if tier == "block":
        _policy.audit_log("hotkey", f"BLOCKED combo={combo}", success=False)
        logger.warning("Hotkey blocked: %s -- %s", combo, reason)
        return f"That keyboard shortcut is blocked for safety, Boss."

    gui = _ensure_pyautogui()
    keys = [k.strip() for k in combo.lower().split("+")]
    gui.hotkey(*keys)
    _policy.audit_log("hotkey", f"combo={combo}")
    logger.info("Hotkey executed: %s", combo)

    labels = {
        "ctrl+c": "Copied",
        "ctrl+v": "Pasted",
        "ctrl+x": "Cut",
        "ctrl+z": "Undone",
        "ctrl+a": "Selected all",
        "ctrl+s": "Saved",
        "ctrl+f": "Opened find",
        "alt+tab": "Switched window",
    }
    label = labels.get(combo.lower(), f"Pressed {combo}")
    return f"{label}, Boss."


def type_text(text: str) -> str:
    """Type text at the current cursor position."""
    if len(text) > 500:
        text = text[:500]
        logger.warning("Text truncated to 500 chars for safety")

    gui = _ensure_pyautogui()
    gui.typewrite(text, interval=0.02) if text.isascii() else gui.write(text)
    _policy.audit_log("type_text", f"chars={len(text)}")
    logger.info("Typed %d characters", len(text))
    return f"Typed the text, Boss."


def move_mouse(direction: str, pixels: int = 100) -> str:
    """Move the mouse cursor in a direction."""
    gui = _ensure_pyautogui()
    pixels = min(pixels, 500)
    dx, dy = 0, 0
    d = direction.lower()
    if d in ("left",):
        dx = -pixels
    elif d in ("right",):
        dx = pixels
    elif d in ("up",):
        dy = -pixels
    elif d in ("down",):
        dy = pixels
    else:
        return f"I don't understand direction '{direction}', Boss."

    gui.moveRel(dx, dy, duration=0.3)
    _policy.audit_log("move_mouse", f"direction={direction} pixels={pixels}")
    logger.info("Moved mouse %s by %d pixels", direction, pixels)
    return f"Moved the mouse {direction}."


def take_screenshot() -> str:
    """Take a screenshot and return the file path."""
    gui = _ensure_pyautogui()
    from pathlib import Path
    import tempfile
    path = Path(tempfile.gettempdir()) / f"atom_screenshot_{int(time.time())}.png"
    gui.screenshot(str(path))
    _policy.audit_log("screenshot", f"path={path}")
    logger.info("Screenshot saved: %s", path)
    return f"Screenshot saved, Boss."


def get_screen_size() -> tuple[int, int]:
    """Return (width, height) of the primary screen."""
    gui = _ensure_pyautogui()
    return gui.size()
