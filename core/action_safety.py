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
    """Conservative defaults aligned with router confirmation lists."""
    critical = {
        "shutdown_pc", "restart_pc", "logoff", "sleep_pc",
        "empty_recycle_bin", "kill_process", "format_disk",
    }
    high = {
        "move_path", "copy_path", "delete_path", "close_app",
        "create_folder", "play_youtube", "type_text",
    }
    medium = {"open_app", "open_url", "set_reminder"}
    if action in critical:
        return ActionRisk.CRITICAL
    if action in high:
        return ActionRisk.HIGH
    if action in medium:
        return ActionRisk.MEDIUM
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
