"""
ATOM -- macOS Focus / Do Not Disturb control (Phase F3 of the
Jarvis-OS plan).

Modern macOS (Ventura+) walled off the old `defaults` plist that the
NotificationCenter used for "doNotDisturb", so the only reliable
programmatic switch is **Apple Shortcuts**. This module drives focus
state by invoking three user-installed shortcuts:

    "ATOM Focus On"          -> turn on Do Not Disturb
    "ATOM Focus Off"         -> turn off Do Not Disturb
    "ATOM Focus Status"      -> echo "on" or "off" (optional)

Setup steps for the human are documented in ``docs/FOCUS_SETUP.md``.
We surface a polite, actionable message when the shortcut is missing
so the user knows exactly how to fix it instead of silent failure.

Public API::

    enable_focus(*, duration_minutes: int | None = None) -> tuple[bool, str]
    disable_focus() -> tuple[bool, str]
    toggle_focus() -> tuple[bool, str]
    focus_state() -> "on" | "off" | "unknown"
    diagnostics() -> dict

Each call returns a ``(ok, message)`` tuple so the router can speak
the message verbatim (no "Done." swallowing of error context).

Owner: Satyam
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from typing import Any

logger = logging.getLogger("atom.router.focus")

_IS_MAC: bool = sys.platform == "darwin"


SHORTCUT_ON = "ATOM Focus On"
SHORTCUT_OFF = "ATOM Focus Off"
SHORTCUT_STATUS = "ATOM Focus Status"

_SETUP_HINT = (
    "I need an Apple Shortcut named '{name}' to control Focus, Boss. "
    "Open Shortcuts, create a 2-step shortcut that toggles Do Not "
    "Disturb, and name it exactly '{name}'. Full setup guide is in "
    "docs/FOCUS_SETUP.md."
)


# ── shell runner (tests monkeypatch this) ────────────────────────────


def _run_shortcuts(args: list[str], *, timeout_s: float = 4.0) -> tuple[bool, str, str]:
    """Run the macOS ``shortcuts`` CLI and return ``(ok, stdout, stderr)``.

    ``ok`` is False on any failure path so the caller can fall back
    cleanly. ``stdout``/``stderr`` are trimmed strings.
    """
    if not _IS_MAC:
        return False, "", "non-darwin"
    binary = shutil.which("shortcuts")
    if not binary:
        return False, "", "shortcuts CLI not installed"
    try:
        result = subprocess.run(
            [binary, *args],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        logger.warning("shortcuts CLI timed out after %.1fs", timeout_s)
        return False, "", "timeout"
    except Exception:
        logger.debug("shortcuts CLI raised", exc_info=True)
        return False, "", "exception"
    out = (result.stdout or "").strip()
    err = (result.stderr or "").strip()
    if result.returncode != 0:
        return False, out, err or f"rc={result.returncode}"
    return True, out, err


# ── shortcut presence ────────────────────────────────────────────────


def _list_shortcuts() -> set[str]:
    """Return the set of installed shortcut names (lower-case)."""
    ok, out, _err = _run_shortcuts(["list"], timeout_s=2.5)
    if not ok or not out:
        return set()
    return {line.strip().lower() for line in out.splitlines() if line.strip()}


def _has_shortcut(name: str) -> bool:
    return name.lower() in _list_shortcuts()


# ── focus on / off / toggle ─────────────────────────────────────────


def _run_named_shortcut(name: str, *, input_text: str = "") -> tuple[bool, str]:
    """Run a named shortcut, returning ``(ok, message)``."""
    if not _IS_MAC:
        return False, "Focus control is macOS-only, Boss."
    if not shutil.which("shortcuts"):
        return False, ("The macOS shortcuts CLI is missing -- run "
                       "`xcode-select --install` and re-try, Boss.")
    args = ["run", name]
    if input_text:
        args.extend(["--input-path", "-"])
    try:
        proc = subprocess.run(
            ["shortcuts", *args],
            input=input_text if input_text else None,
            capture_output=True,
            text=True,
            timeout=6.0,
        )
    except subprocess.TimeoutExpired:
        return False, f"Shortcut '{name}' timed out after 6 seconds, Boss."
    except Exception:
        logger.debug("shortcuts run raised", exc_info=True)
        return False, f"Couldn't run shortcut '{name}', Boss."
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip().splitlines()
        first = stderr[0] if stderr else ""
        if "couldn't be found" in first.lower() or "not found" in first.lower():
            return False, _SETUP_HINT.format(name=name)
        return False, (
            f"Shortcut '{name}' failed: {first[:120]}" if first
            else f"Shortcut '{name}' failed with exit code {proc.returncode}."
        )
    out = (proc.stdout or "").strip()
    return True, out


def enable_focus(*, duration_minutes: int | None = None) -> tuple[bool, str]:
    """Activate Do Not Disturb via the ATOM Focus On shortcut.

    ``duration_minutes`` is forwarded as the shortcut's input (the
    user's shortcut can read it with the "Get Shortcut Input" action
    to set a timed focus); when None, a permanent on-until-canceled
    toggle is requested.
    """
    payload = ""
    if duration_minutes is not None and duration_minutes > 0:
        payload = str(int(duration_minutes))
    ok, message = _run_named_shortcut(SHORTCUT_ON, input_text=payload)
    if ok:
        if duration_minutes:
            return True, f"Focus on for {int(duration_minutes)} minutes."
        return True, message or "Focus on. I'll guard your attention, Boss."
    return False, message


def disable_focus() -> tuple[bool, str]:
    """Deactivate Do Not Disturb via the ATOM Focus Off shortcut."""
    ok, message = _run_named_shortcut(SHORTCUT_OFF)
    if ok:
        return True, message or "Focus off. Notifications are back, Boss."
    return False, message


def focus_state() -> str:
    """Return ``"on"``, ``"off"``, or ``"unknown"`` based on the
    optional Status shortcut. Missing the status shortcut is fine --
    we just degrade to "unknown" rather than guessing.
    """
    if not _IS_MAC or not shutil.which("shortcuts"):
        return "unknown"
    if not _has_shortcut(SHORTCUT_STATUS):
        return "unknown"
    ok, message = _run_named_shortcut(SHORTCUT_STATUS)
    if not ok:
        return "unknown"
    text = (message or "").strip().lower()
    if text in {"on", "true", "1", "yes", "enabled"}:
        return "on"
    if text in {"off", "false", "0", "no", "disabled"}:
        return "off"
    return "unknown"


def toggle_focus() -> tuple[bool, str]:
    """Flip whichever state is currently active.

    Falls back to ``enable_focus`` when the status shortcut is missing
    (better to over-enable than to silently no-op).
    """
    state = focus_state()
    if state == "on":
        return disable_focus()
    return enable_focus()


# ── diagnostics ──────────────────────────────────────────────────────


def diagnostics() -> dict[str, Any]:
    """Snapshot for /diagnostics: shortcuts available, current state."""
    if not _IS_MAC:
        return {"available": False, "reason": "non-darwin"}
    if not shutil.which("shortcuts"):
        return {"available": False, "reason": "shortcuts-cli-missing"}
    installed = _list_shortcuts()
    return {
        "available": True,
        "shortcut_on_present": SHORTCUT_ON.lower() in installed,
        "shortcut_off_present": SHORTCUT_OFF.lower() in installed,
        "shortcut_status_present": SHORTCUT_STATUS.lower() in installed,
        "current_state": focus_state(),
    }


__all__ = [
    "SHORTCUT_ON",
    "SHORTCUT_OFF",
    "SHORTCUT_STATUS",
    "enable_focus",
    "disable_focus",
    "focus_state",
    "toggle_focus",
    "diagnostics",
]
