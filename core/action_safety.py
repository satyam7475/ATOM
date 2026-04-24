"""
ATOM V6.5 -- Action risk levels, confirmation policy, and audit logging.

Integrates with existing ConfirmationManager patterns: HIGH/CRITICAL actions
must obtain explicit confirmation before execution.
"""

from __future__ import annotations

import json
import logging
import time
from enum import IntEnum
from pathlib import Path
from threading import Lock
from typing import Any, Optional

logger = logging.getLogger("atom.action_safety")

_AUDIT_LOCK = Lock()
_DEFAULT_AUDIT_PATH = Path("logs/audit_v65.jsonl")


class ActionRisk(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


def risk_requires_confirmation(level: ActionRisk) -> bool:
    return level >= ActionRisk.HIGH


def risk_label(level: ActionRisk) -> str:
    return level.name


def default_risk_for_action(action: str) -> ActionRisk:
    """Conservative defaults aligned with router confirmation lists.

    Phase F5 ("Jarvis-grade frictionless control"): media playback,
    focus toggling, and screen lock are demoted to LOW so the
    assistant *acts* instead of asking. Anything that mutates files,
    powers down the machine, or drives arbitrary keystrokes stays at
    HIGH+ and still requires confirmation.
    """
    critical = {
        "shutdown_pc", "restart_pc", "logoff", "sleep_pc",
        "empty_recycle_bin", "kill_process", "format_disk",
    }
    high = {
        "move_path", "copy_path", "delete_path", "close_app",
        "create_folder", "type_text",
        # v22 Advanced Control -- still need confirmation because
        # they synthesise arbitrary input on the user's behalf.
        "set_focused_text", "click_ui_element",
        "set_process_priority", "optimize_for_atom",
    }
    medium = {
        "open_app", "open_url", "set_reminder", "spotlight_search",
    }
    # Frictionless tools (Phase F5): explicit allow-list of
    # high-frequency conversational actions that must NEVER prompt.
    low_frictionless = {
        "lock_screen", "screenshot", "set_brightness", "set_volume",
        "mute", "unmute", "stop_music",
        "play_youtube",
        "music_play", "music_pause", "music_next", "music_prev",
        "music_current", "music_play_specific",
        "focus_on", "focus_off", "focus_state",
        "minimize_window", "maximize_window", "switch_window",
        "next_window_in_app", "switch_space",
    }
    if action in critical:
        return ActionRisk.CRITICAL
    if action in high:
        return ActionRisk.HIGH
    if action in medium:
        return ActionRisk.MEDIUM
    if action in low_frictionless:
        return ActionRisk.LOW
    return ActionRisk.LOW


def append_audit_record(
    *,
    actor: str,
    action: str,
    risk: str,
    reason: str,
    result: str,
    extra: Optional[dict[str, Any]] = None,
    path: Optional[Path] = None,
) -> None:
    """Append one JSON line: who, what, when, why, result."""

    rec = {
        "ts": time.time(),
        "actor": actor,
        "action": action,
        "risk": risk,
        "why": reason,
        "result": result,
    }
    if extra:
        rec["extra"] = extra

    log_path = path or _DEFAULT_AUDIT_PATH
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(rec, default=str) + "\n"
        with _AUDIT_LOCK:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        logger.exception("audit append failed")


__all__ = [
    "ActionRisk",
    "risk_requires_confirmation",
    "risk_label",
    "default_risk_for_action",
    "append_audit_record",
]
