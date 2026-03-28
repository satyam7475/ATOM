"""
ATOM -- Assistant mode (ATOM vs brain routing).

Controls whether open-ended queries may invoke the LLM (hybrid / conversational)
or are limited to commands + quick replies only (command_only).

Separate from assistant_brain profiles (atom / balanced / brain), which tune
the local model when it *does* run.

Security: allowlisted modes only; changes audited; optional persistence with chmod 600.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom.assistant_mode")

ALLOWED_MODES: frozenset[str] = frozenset({"command_only", "hybrid", "conversational"})

_STATE_FILE = Path("logs/atom_assistant_mode.json")


class AssistantModeManager:
    """Thread-safe assistant routing mode."""

    def __init__(self, config: dict | None = None) -> None:
        self._config = config or {}
        self._lock = threading.RLock()
        ctrl = self._config.get("control", {})
        default = ctrl.get("assistant_mode", "hybrid")
        if default not in ALLOWED_MODES:
            default = "hybrid"
        persisted = self._load_persisted()
        if persisted in ALLOWED_MODES and ctrl.get("restore_persisted_assistant_mode", True):
            self._active = persisted
        else:
            self._active = default
        self._audit = bool(ctrl.get("audit_assistant_mode_changes", True))
        self._security: Any = None

    def attach_security(self, security: Any) -> None:
        self._security = security

    @property
    def active(self) -> str:
        with self._lock:
            return self._active

    def allows_llm_fallback(self) -> bool:
        """False in command_only — router / cursor_query should not run inference."""
        return self.active != "command_only"

    def command_only_message(self) -> str:
        msg = self._config.get("control", {}).get(
            "command_only_message",
            "I'm in command-only mode, Boss — no chat brain. "
            "Try a direct command like time, open notepad, or cpu. "
            "Say hybrid mode or conversation mode to re-enable chat.",
        )
        return str(msg)[:800]

    def set_mode(self, name: str) -> tuple[bool, str]:
        if not name or not isinstance(name, str):
            return False, "Invalid mode name."
        key = name.strip().lower().replace(" ", "_")
        if key not in ALLOWED_MODES:
            return False, (
                f"Unknown assistant mode '{name}'. "
                "Use command only, hybrid, or conversational."
            )
        with self._lock:
            old = self._active
            self._active = key
        logger.info("Assistant mode: %s -> %s", old, key)
        if self._audit and self._security is not None:
            try:
                self._security.audit_log(
                    "assistant_mode_switch",
                    f"{old} -> {key}",
                    success=True,
                )
            except Exception:
                logger.debug("audit_log failed", exc_info=True)
        self._persist(key)
        labels = {
            "command_only": "Commands and quick replies only — LLM disabled.",
            "hybrid": "Commands first; open chat uses the brain when needed.",
            "conversational": "Same routing as hybrid; use for future tuning.",
        }
        return True, f"Assistant mode is now {key}, Boss. {labels.get(key, '')}"

    def _persist(self, name: str) -> None:
        if not self._config.get("control", {}).get("persist_assistant_mode", True):
            return
        try:
            _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = _STATE_FILE.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"assistant_mode": name}, f, indent=2)
            tmp.replace(_STATE_FILE)
            try:
                os.chmod(_STATE_FILE, 0o600)
            except OSError:
                pass
        except Exception:
            logger.warning("Could not persist assistant mode", exc_info=True)

    def _load_persisted(self) -> str | None:
        try:
            if not _STATE_FILE.is_file():
                return None
            with open(_STATE_FILE, encoding="utf-8") as f:
                data = json.load(f)
            v = data.get("assistant_mode", "")
            if isinstance(v, str) and v in ALLOWED_MODES:
                return v
        except Exception:
            pass
        return None
