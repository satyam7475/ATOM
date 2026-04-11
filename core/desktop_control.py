"""
ATOM -- Desktop Control (AI OS Automation Layer).

Voice-controlled desktop automation:
  - Scroll up/down
  - Click (center or coordinates)
  - Type text at cursor position
  - Read/fill focused macOS text fields via Accessibility API
  - Click macOS UI elements by accessible label
  - Press individual keys (Enter, Escape, Tab, arrows)
  - Safe hotkey combos (Ctrl+C / Cmd+C, etc.)
  - Screenshot for visual context

Primary backend: pyautogui (cross-platform).
macOS fallback: AppleScript via osascript for keyboard operations,
                screencapture for screenshots.

On macOS, ctrl hotkeys are automatically mapped to command
(e.g. "ctrl+c" becomes "command+c").

Security: pyautogui.FAILSAFE always ON (move mouse to top-left corner to abort).
All operations go through SecurityPolicy before execution.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path

from core.macos import AccessibilityAPI, AppleScriptEngine
from core.security_policy import SecurityPolicy

logger = logging.getLogger("atom.desktop")

_policy = SecurityPolicy()
_IS_MACOS = sys.platform == "darwin"
_APPLE_SCRIPT = AppleScriptEngine() if _IS_MACOS else None
_ACCESSIBILITY = AccessibilityAPI() if _IS_MACOS else None


# ── pyautogui availability ───────────────────────────────────────────

_pyautogui = None
_pyautogui_checked = False


def _ensure_pyautogui():
    """Lazy-import pyautogui with failsafe always enabled. Returns None if unavailable."""
    global _pyautogui, _pyautogui_checked
    if _pyautogui_checked:
        return _pyautogui
    _pyautogui_checked = True
    try:
        import pyautogui
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1
        _pyautogui = pyautogui
        return pyautogui
    except ImportError:
        logger.info(
            "pyautogui not installed — using %s fallback. "
            "Install with: pip install pyautogui",
            "AppleScript" if _IS_MACOS else "limited",
        )
        return None


# ── macOS modifier key mapping ───────────────────────────────────────

_CTRL_TO_CMD = {
    "ctrl": "command",
    "alt": "option",
    "win": "command",
}

_PYAUTOGUI_KEY_MAP = {
    "command": "command" if _IS_MACOS else "win",
    "option": "option" if _IS_MACOS else "alt",
    "cmd": "command",
}


def _macos_combo(combo: str) -> str:
    """On macOS, map ctrl→command, alt→option for hotkey combos."""
    if not _IS_MACOS:
        return combo
    parts = [k.strip().lower() for k in combo.split("+")]
    mapped = [_CTRL_TO_CMD.get(p, p) for p in parts]
    return "+".join(mapped)


# ── AppleScript fallback helpers ─────────────────────────────────────

def _osascript(script: str) -> str:
    """Run an AppleScript snippet via osascript. Returns stdout."""
    if _APPLE_SCRIPT is None:
        return ""
    return _APPLE_SCRIPT.run(script, timeout=5.0)


def _applescript_keystroke(key: str, modifiers: list | None = None) -> bool:
    """Send a keystroke via AppleScript System Events."""
    if _APPLE_SCRIPT is None:
        return False
    return _APPLE_SCRIPT.press_key(key, list(modifiers or []))


def _applescript_type(text: str) -> bool:
    """Type text via AppleScript keystroke (handles Unicode)."""
    if _APPLE_SCRIPT is None:
        return False
    return _APPLE_SCRIPT.type_text(text)


def _accessibility_ready(prompt: bool = False) -> bool:
    if not _IS_MACOS or _ACCESSIBILITY is None or not _ACCESSIBILITY.is_available:
        return False
    return _ACCESSIBILITY.is_trusted(prompt=prompt)


def _applescript_scroll(direction: str, clicks: int) -> bool:
    """Scroll via AppleScript. Limited but functional."""
    if direction == "down":
        for _ in range(clicks):
            _osascript(
                'tell application "System Events" to key code 125 '
                'using {option down}'
            )
    elif direction == "up":
        for _ in range(clicks):
            _osascript(
                'tell application "System Events" to key code 126 '
                'using {option down}'
            )
    else:
        return False
    return True


# ── Public API ───────────────────────────────────────────────────────

def scroll_down(clicks: int = 5) -> str:
    """Scroll the active window down."""
    clicks = min(clicks, 30)
    gui = _ensure_pyautogui()
    if gui:
        gui.scroll(-clicks)
    elif _IS_MACOS:
        _applescript_scroll("down", clicks)
    else:
        return "Scroll not available — pyautogui not installed, Boss."
    _policy.audit_log("scroll_down", f"clicks={clicks}")
    logger.info("Scrolled down %d clicks", clicks)
    return f"Scrolled down {clicks} clicks, Boss."


def scroll_up(clicks: int = 5) -> str:
    """Scroll the active window up."""
    clicks = min(clicks, 30)
    gui = _ensure_pyautogui()
    if gui:
        gui.scroll(clicks)
    elif _IS_MACOS:
        _applescript_scroll("up", clicks)
    else:
        return "Scroll not available — pyautogui not installed, Boss."
    _policy.audit_log("scroll_up", f"clicks={clicks}")
    logger.info("Scrolled up %d clicks", clicks)
    return f"Scrolled up {clicks} clicks, Boss."


def click_center() -> str:
    """Click the center of the screen."""
    gui = _ensure_pyautogui()
    if gui:
        w, h = gui.size()
        gui.click(w // 2, h // 2)
        _policy.audit_log("click_center", f"pos=({w // 2},{h // 2})")
        logger.info("Clicked center of screen (%d, %d)", w // 2, h // 2)
        return "Clicked the center of the screen."
    if _IS_MACOS:
        _osascript(
            'tell application "System Events" to click at {960, 540}'
        )
        _policy.audit_log("click_center", "applescript fallback")
        return "Clicked center of screen (approximate), Boss."
    return "Click not available — pyautogui not installed, Boss."


def click_at(x: int, y: int) -> str:
    """Click at specific coordinates."""
    gui = _ensure_pyautogui()
    if gui:
        gui.click(x, y)
    elif _IS_MACOS:
        _osascript(
            f'tell application "System Events" to click at {{{x}, {y}}}'
        )
    else:
        return "Click not available — pyautogui not installed, Boss."
    _policy.audit_log("click_at", f"pos=({x},{y})")
    logger.info("Clicked at (%d, %d)", x, y)
    return f"Clicked at position {x}, {y}."


def double_click_center() -> str:
    """Double-click the center of the screen."""
    gui = _ensure_pyautogui()
    if gui:
        w, h = gui.size()
        gui.doubleClick(w // 2, h // 2)
        _policy.audit_log("double_click_center")
        return "Double-clicked the center of the screen."
    return "Double-click not available — pyautogui not installed, Boss."


def press_key(key: str) -> str:
    """Press a single key if it's in the safe list."""
    key_lower = key.lower().strip()
    if not _policy.is_safe_key(key_lower):
        _policy.audit_log("press_key", f"BLOCKED key={key}", success=False)
        return f"Key '{key}' is not in the safe list, Boss."

    gui = _ensure_pyautogui()
    if gui:
        gui.press(key_lower)
    elif _IS_MACOS:
        _applescript_keystroke(key_lower)
    else:
        return "Key press not available — pyautogui not installed, Boss."

    _policy.audit_log("press_key", f"key={key_lower}")
    logger.info("Pressed key: %s", key_lower)
    return f"Pressed {key}."


def hotkey_combo(combo: str) -> str:
    """Execute a keyboard shortcut if safe.

    combo: e.g. "ctrl+c", "alt+tab", "ctrl+shift+t"
    On macOS, ctrl is automatically mapped to command.
    """
    original_combo = combo
    macos_combo = _macos_combo(combo) if _IS_MACOS else combo

    tier, reason = _policy.is_safe_hotkey(combo)
    if tier == "block":
        _policy.audit_log("hotkey", f"BLOCKED combo={combo}", success=False)
        logger.warning("Hotkey blocked: %s -- %s", combo, reason)
        return "That keyboard shortcut is blocked for safety, Boss."

    gui = _ensure_pyautogui()
    keys = [k.strip() for k in macos_combo.lower().split("+")]

    if gui:
        gui_keys = [_PYAUTOGUI_KEY_MAP.get(k, k) for k in keys]
        gui.hotkey(*gui_keys)
    elif _IS_MACOS:
        if len(keys) >= 2:
            modifiers = keys[:-1]
            final_key = keys[-1]
            _applescript_keystroke(final_key, modifiers)
        else:
            _applescript_keystroke(keys[0])
    else:
        return "Hotkey not available — pyautogui not installed, Boss."

    _policy.audit_log("hotkey", f"combo={macos_combo}")
    logger.info("Hotkey executed: %s (mapped from %s)", macos_combo, original_combo)

    labels = {
        "ctrl+c": "Copied", "command+c": "Copied",
        "ctrl+v": "Pasted", "command+v": "Pasted",
        "ctrl+x": "Cut", "command+x": "Cut",
        "ctrl+z": "Undone", "command+z": "Undone",
        "ctrl+a": "Selected all", "command+a": "Selected all",
        "ctrl+s": "Saved", "command+s": "Saved",
        "ctrl+f": "Opened find", "command+f": "Opened find",
        "alt+tab": "Switched window", "command+tab": "Switched window",
    }
    label = labels.get(macos_combo.lower(),
                       labels.get(original_combo.lower(),
                                  f"Pressed {macos_combo}"))
    return f"{label}, Boss."


def type_text(text: str) -> str:
    """Type text at the current cursor position."""
    if len(text) > 500:
        text = text[:500]
        logger.warning("Text truncated to 500 chars for safety")

    gui = _ensure_pyautogui()
    if gui:
        if text.isascii() and not _IS_MACOS:
            gui.typewrite(text, interval=0.02)
        else:
            gui.write(text)
    elif _IS_MACOS:
        _applescript_type(text)
    else:
        return "Typing not available — pyautogui not installed, Boss."

    _policy.audit_log("type_text", f"chars={len(text)}")
    logger.info("Typed %d characters", len(text))
    return "Typed the text, Boss."


def move_mouse(direction: str, pixels: int = 100) -> str:
    """Move the mouse cursor in a direction."""
    gui = _ensure_pyautogui()
    if not gui:
        return "Mouse movement not available — pyautogui not installed, Boss."

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
    import tempfile
    path = Path(tempfile.gettempdir()) / f"atom_screenshot_{int(time.time())}.png"

    if _IS_MACOS:
        try:
            result = subprocess.run(
                ["screencapture", "-x", "-t", "png", str(path)],
                capture_output=True, timeout=5,
            )
            if result.returncode == 0 and path.exists() and path.stat().st_size > 0:
                _policy.audit_log("screenshot", f"path={path}")
                logger.info("Screenshot saved (screencapture): %s", path)
                return "Screenshot saved, Boss."
        except Exception as exc:
            logger.debug("screencapture failed: %s", exc)

    gui = _ensure_pyautogui()
    if gui:
        gui.screenshot(str(path))
        _policy.audit_log("screenshot", f"path={path}")
        logger.info("Screenshot saved (pyautogui): %s", path)
        return "Screenshot saved, Boss."

    return "Screenshot not available — screencapture failed and pyautogui not installed, Boss."


def get_screen_size() -> tuple:
    """Return (width, height) of the primary screen."""
    gui = _ensure_pyautogui()
    if gui:
        return gui.size()
    if _IS_MACOS:
        try:
            out = _osascript(
                'tell application "Finder" to get bounds of window of desktop'
            )
            if out:
                parts = out.split(", ")
                if len(parts) == 4:
                    return (int(parts[2]), int(parts[3]))
        except Exception:
            pass
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True, text=True, timeout=5,
            )
            import json
            data = json.loads(result.stdout)
            for gpu in data.get("SPDisplaysDataType", []):
                for disp in gpu.get("spdisplays_ndrvs", []):
                    res = disp.get("_spdisplays_resolution", "")
                    if " x " in res:
                        parts = res.split(" x ")
                        w = int(parts[0].strip())
                        h = int(parts[1].split()[0].strip())
                        return (w, h)
        except Exception:
            pass
    return (1920, 1080)


def describe_focused_element() -> str:
    """Describe the currently focused macOS accessibility element."""
    if not _IS_MACOS:
        return "Focused UI inspection is only available on macOS, Boss."
    if not _accessibility_ready(prompt=False):
        return (
            "macOS Accessibility permission is required to inspect UI elements, Boss."
        )

    data = _ACCESSIBILITY.get_focused_element() if _ACCESSIBILITY is not None else {}
    role = str(data.get("role", "") or "UI element")
    title = str(data.get("title", "") or data.get("description", "") or "").strip()
    value = str(data.get("value", "") or "").strip()
    app = str((data.get("frontmost_app") or {}).get("name", "") or "").strip()

    parts = [role]
    if title:
        parts.append(f"'{title}'")
    if value and value != title:
        parts.append(f"value '{value[:120]}'")
    if app:
        parts.append(f"in {app}")
    return "Focused element: " + " ".join(parts).strip() + "."


def read_focused_text(max_chars: int = 500) -> str:
    """Read the currently focused macOS text field if accessible."""
    if not _IS_MACOS:
        return "Focused text reading is only available on macOS, Boss."
    if not _accessibility_ready(prompt=False):
        return (
            "macOS Accessibility permission is required to read focused text, Boss."
        )

    text = ""
    if _ACCESSIBILITY is not None:
        text = _ACCESSIBILITY.read_focused_text(max_chars=max_chars)
    if not text:
        return "I couldn't read text from the focused UI element, Boss."
    _policy.audit_log("read_focused_text", f"chars={len(text)}")
    logger.info("Read focused text (%d chars)", len(text))
    return text


def set_focused_text(text: str, append: bool = False) -> str:
    """Set the focused macOS text field using the Accessibility API."""
    if not _IS_MACOS:
        return "Focused text filling is only available on macOS, Boss."
    if not _accessibility_ready(prompt=True):
        return (
            "macOS Accessibility permission is required to fill focused text, Boss."
        )

    payload = (text or "")[:1000]
    if not payload:
        return "No text provided for the focused field, Boss."
    ok = _ACCESSIBILITY.set_focused_text(payload, append=append) if _ACCESSIBILITY is not None else False
    if not ok:
        return "I couldn't write into the focused UI element, Boss."
    _policy.audit_log("set_focused_text", f"chars={len(payload)} append={append}")
    logger.info("Focused text updated (%d chars, append=%s)", len(payload), append)
    return "Focused text field updated, Boss."


def click_ui_element(label: str, role: str | None = None) -> str:
    """Click a macOS UI element by accessible label."""
    if not _IS_MACOS:
        return "UI element clicks by label are only available on macOS, Boss."
    if not _accessibility_ready(prompt=False):
        return (
            "macOS Accessibility permission is required to click UI elements, Boss."
        )

    target = (label or "").strip()[:120]
    if not target:
        return "No UI label provided to click, Boss."
    ok = (
        _ACCESSIBILITY.click_element_by_title(target, role=role)
        if _ACCESSIBILITY is not None
        else False
    )
    if not ok:
        return f"I couldn't find a clickable UI element matching '{target}', Boss."
    _policy.audit_log("click_ui_element", f"label={target} role={role or ''}")
    logger.info("Clicked UI element label=%s role=%s", target, role or "")
    return f"Clicked '{target}', Boss."
