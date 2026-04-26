"""ATOM system-level action handlers (macOS-only).

Sprint P4.7 (Apr 26 2026): dropped Windows ``win32`` branches. ATOM
targets macOS on Apple Silicon -- the Windows code paths were dead
weight that survived Phase 0. See ``docs/ATOM_NEXT_STEPS_PLAN.md``
section P4.7 for the rationale.
"""

from __future__ import annotations

import datetime
import logging
import re
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("atom.router.system")


def lock_screen() -> None:
    if sys.platform == "darwin":
        subprocess.Popen(
            [
                "osascript",
                "-e",
                'tell application "System Events" to keystroke "q" '
                "using {control down, command down}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("Screen locked")


def take_screenshot() -> None:
    """Capture screen to a timestamped image on Desktop."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    desktop = Path.home() / "Desktop"
    if sys.platform == "darwin":
        filepath = desktop / f"screenshot_{ts}.png"
        subprocess.run(
            ["screencapture", "-x", str(filepath)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        logger.info("Screenshot saved: %s", filepath)


def _darwin_brightness_current_percent() -> int:
    try:
        proc = subprocess.run(
            ["brightness", "-l"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        m = re.search(r"brightness\s+([\d.]+)", proc.stdout, re.I)
        if not m:
            return 50
        v = float(m.group(1))
        if v <= 1.0:
            return int(round(v * 100))
        return int(round(max(0, min(100, v))))
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return 50


def set_brightness(
    percent: int | None = None, delta: int | None = None,
) -> int:
    """Set or adjust screen brightness; returns target percent."""
    if percent is not None:
        target = max(0, min(100, percent))
    elif delta is not None:
        if sys.platform == "darwin":
            current = _darwin_brightness_current_percent()
        else:
            current = 50
        target = max(0, min(100, current + delta))
    else:
        target = 50

    if sys.platform == "darwin":
        try:
            subprocess.run(
                ["brightness", str(target / 100.0)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
                check=False,
            )
        except FileNotFoundError:
            pass

    return target


def shutdown_pc() -> None:
    if sys.platform == "darwin":
        subprocess.Popen(
            ["osascript", "-e", 'tell app "System Events" to shut down'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def restart_pc() -> None:
    if sys.platform == "darwin":
        subprocess.Popen(
            ["osascript", "-e", 'tell app "System Events" to restart'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def logoff() -> None:
    if sys.platform == "darwin":
        subprocess.Popen(
            ["osascript", "-e", 'tell app "System Events" to log out'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def sleep_pc() -> None:
    if sys.platform == "darwin":
        subprocess.Popen(
            ["pmset", "sleepnow"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def empty_recycle_bin() -> None:
    if sys.platform == "darwin":
        subprocess.Popen(
            [
                "osascript",
                "-e",
                'tell application "Finder" to empty trash',
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def flush_dns() -> None:
    if sys.platform == "darwin":
        subprocess.run(
            ["dscacheutil", "-flushcache"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
