"""
ATOM v15 -- Brain / ATOM mode profiles (production-safe).

Controls how the local LLM runs: ATOM mode (fast, short), Balanced, Brain mode (deeper).
All profile names are allowlisted. Changes can be audited and optionally persisted.

This is separate from personality_modes (work/focus/chill/sleep) — that adjusts tone;
assistant_brain profiles adjust inference parameters and optional model path.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom.brain_mode")

ALLOWED_PROFILES: frozenset[str] = frozenset({"atom", "balanced", "brain"})
_PROFILE_RE = re.compile(r"^[a-z]+$")

_STATE_FILE = Path("logs/atom_brain_profile.json")


class BrainModeManager:
    """Thread-safe active profile + merged effective brain parameters."""

    def __init__(self, config: dict | None = None) -> None:
        self._config = config or {}
        self._lock = threading.RLock()
        ab = self._config.get("assistant_brain", {})
        default = ab.get("active_profile", "balanced")
        if default not in ALLOWED_PROFILES:
            default = "balanced"
        persisted = self._load_persisted_profile()
        if persisted in ALLOWED_PROFILES and ab.get("restore_persisted_profile", True):
            self._active = persisted
        else:
            self._active = default
        self._audit = bool(self._config.get("assistant_brain", {}).get("audit_profile_changes", True))
        self._security = None

    def attach_security(self, security: Any) -> None:
        """Optional SecurityPolicy for audit_log on profile changes."""
        self._security = security

    @property
    def active_profile(self) -> str:
        with self._lock:
            return self._active

    def set_profile(self, name: str) -> tuple[bool, str]:
        """Validate and switch profile. Returns (ok, message for user)."""
        if not name or not isinstance(name, str):
            return False, "Invalid profile name."
        key = name.strip().lower()
        if key not in ALLOWED_PROFILES:
            return False, (
                f"Unknown brain profile '{name}'. "
                f"Say atom mode, balanced mode, or brain mode."
            )
        with self._lock:
            old = self._active
            self._active = key
        logger.info("Brain profile: %s -> %s", old, key)
        if self._audit and self._security is not None:
            try:
                self._security.audit_log(
                    "brain_profile_switch",
                    f"{old} -> {key}",
                    success=True,
                )
            except Exception:
                logger.debug("audit_log failed", exc_info=True)
        self._persist_profile(key)
        labels = {
            "atom": "ATOM mode — faster, shorter replies.",
            "balanced": "Balanced mode — default speed and depth.",
            "brain": "Brain mode — deeper answers, longer generation allowed.",
        }
        return True, f"Switched to {key} profile, Boss. {labels.get(key, '')}"

    def effective_params(self) -> dict[str, Any]:
        """Merged view for MiniLLM: base brain + active profile overrides."""
        with self._lock:
            prof_name = self._active
        base = dict(self._config.get("brain", {}))
        profiles = self._config.get("assistant_brain", {}).get("profiles", {})
        ov = dict(profiles.get(prof_name, {}) if isinstance(profiles, dict) else {})

        def _pick(key: str, default: Any) -> Any:
            if key in ov and ov[key] is not None and ov[key] != "":
                return ov[key]
            return base.get(key, default)

        model_path = _pick("model_path", base.get("model_path", ""))
        if not model_path:
            model_path = base.get("model_path", "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")

        extra_stops = ov.get("extra_stop_sequences")
        if not isinstance(extra_stops, list):
            extra_stops = []

        return {
            "profile": prof_name,
            "model_path": str(model_path),
            "n_ctx": int(_pick("n_ctx", 2048)),
            "n_threads": int(_pick("n_threads", max(2, (os.cpu_count() or 4) // 2))),
            "n_gpu_layers": int(_pick("n_gpu_layers", 0)),
            "max_tokens": int(_pick("max_tokens", 150)),
            "temperature": float(_pick("temperature", 0.4)),
            "timeout_seconds": float(_pick("timeout_seconds", 60)),
            "extra_stop_sequences": [str(s) for s in extra_stops if s][:12],
        }

    def fingerprint(self) -> tuple[str, int, int]:
        p = self.effective_params()
        return (p["model_path"], p["n_ctx"], p["n_threads"])

    def _persist_profile(self, name: str) -> None:
        if not self._config.get("assistant_brain", {}).get("persist_active_profile", True):
            return
        try:
            _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = _STATE_FILE.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"active_profile": name}, f, indent=2)
            tmp.replace(_STATE_FILE)
            try:
                os.chmod(_STATE_FILE, 0o600)
            except OSError:
                pass
        except Exception:
            logger.warning("Could not persist brain profile", exc_info=True)

    def _load_persisted_profile(self) -> str | None:
        try:
            if not _STATE_FILE.is_file():
                return None
            with open(_STATE_FILE, encoding="utf-8") as f:
                data = json.load(f)
            v = data.get("active_profile", "")
            if isinstance(v, str) and v in ALLOWED_PROFILES:
                return v
        except Exception:
            pass
        return None
