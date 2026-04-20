"""
ATOM -- Brain / performance profiles (production-safe).

Controls how the local LLM and background cognition run on Apple Silicon.
Canonical user-facing profiles are:
  - optimal          -> stable daily buddy mode
  - full_performance -> deeper mode when the Mac has headroom

Legacy aliases (`atom`, `balanced`, `brain`) still resolve cleanly.

This is separate from personality_modes (work/focus/chill/sleep) — that adjusts tone;
assistant_brain profiles adjust inference parameters and optional model path.
"""

from __future__ import annotations

import collections
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom.brain_mode")

PROFILE_ALIASES: dict[str, str] = {
    "atom": "optimal",
    "balanced": "optimal",
    "optimal": "optimal",
    "brain": "full_performance",
    "full_performance": "full_performance",
    "full-performance": "full_performance",
    "fullperformance": "full_performance",
}
ALLOWED_PROFILES: frozenset[str] = frozenset(PROFILE_ALIASES)
CANONICAL_PROFILES: frozenset[str] = frozenset({"optimal", "full_performance"})
_PROFILE_RE = re.compile(r"^[a-z]+$")

_PROFILE_LABELS: dict[str, str] = {
    "optimal": "Optimal mode — stable buddy mode.",
    "full_performance": "Full Performance mode — deeper answers when your Mac has headroom.",
}
_PROFILE_FEATURES: dict[str, dict[str, bool]] = {
    "optimal": {
        "autonomy": False,
        "prediction_background": False,
        "prediction_prefetch": False,
        "dream": False,
        "curiosity": False,
        "self_optimizer": False,
        "proactive_background": False,
    },
    "full_performance": {
        "autonomy": True,
        "prediction_background": True,
        "prediction_prefetch": True,
        "dream": True,
        "curiosity": True,
        "self_optimizer": True,
        "proactive_background": True,
    },
}

_STATE_FILE = Path("logs/atom_brain_profile.json")


class BrainModeManager:
    """Thread-safe active profile + merged effective brain parameters."""

    @classmethod
    def canonical_profile_name(cls, name: str | None) -> str | None:
        if not name or not isinstance(name, str):
            return None
        key = name.strip().lower().replace("-", "_").replace(" ", "_")
        return PROFILE_ALIASES.get(key)

    @staticmethod
    def display_name(name: str | None) -> str:
        canonical = BrainModeManager.canonical_profile_name(name) or "optimal"
        return canonical.replace("_", " ").title()

    _COOLDOWN_S = 5.0
    _MAX_SWITCHES_PER_WINDOW = 6
    _RATE_WINDOW_S = 60.0

    def __init__(self, config: dict | None = None) -> None:
        self._config = config or {}
        self._lock = threading.RLock()
        ab = self._config.get("assistant_brain", {})
        default = self.canonical_profile_name(str(ab.get("active_profile", "optimal"))) or "optimal"
        persisted = self._load_persisted_profile()
        if persisted in ALLOWED_PROFILES and ab.get("restore_persisted_profile", True):
            self._active = self.canonical_profile_name(persisted) or default
        else:
            self._active = default
        self._audit = bool(self._config.get("assistant_brain", {}).get("audit_profile_changes", True))
        self._security = None
        self._last_switch_time: float = 0.0
        self._switch_timestamps: collections.deque[float] = collections.deque(maxlen=self._MAX_SWITCHES_PER_WINDOW)

    def attach_security(self, security: Any) -> None:
        """Optional SecurityPolicy for audit_log on profile changes."""
        self._security = security

    @property
    def active_profile(self) -> str:
        with self._lock:
            return self._active

    @property
    def active_profile_label(self) -> str:
        return self.display_name(self.active_profile)

    def is_full_performance(self, profile: str | None = None) -> bool:
        canonical = self.canonical_profile_name(profile or self.active_profile) or "optimal"
        return canonical == "full_performance"

    def is_optimal(self, profile: str | None = None) -> bool:
        return not self.is_full_performance(profile)

    def feature_enabled(self, feature: str, profile: str | None = None) -> bool:
        canonical = self.canonical_profile_name(profile or self.active_profile) or "optimal"
        features = _PROFILE_FEATURES.get(canonical, _PROFILE_FEATURES["optimal"])
        return bool(features.get(feature, True))

    def _rate_limited(self) -> bool:
        """True if too many switches happened in the rolling window."""
        now = time.monotonic()
        cutoff = now - self._RATE_WINDOW_S
        while self._switch_timestamps and self._switch_timestamps[0] < cutoff:
            self._switch_timestamps.popleft()
        return len(self._switch_timestamps) >= self._MAX_SWITCHES_PER_WINDOW

    def set_profile(self, name: str, *, force: bool = False) -> tuple[bool, str]:
        """Validate and switch profile. Returns (ok, message for user).

        Guards:
          - Same-state: no-op if already in the requested profile.
          - Cooldown: rejects switches within _COOLDOWN_S of the last one.
          - Rate limit: max _MAX_SWITCHES_PER_WINDOW switches per _RATE_WINDOW_S.

        ``force=True`` bypasses the cooldown + rate-limit guards. Reserved
        for boot-time restoration, internal recovery flows, and tests; do
        NOT pass it from user-facing voice commands.
        """
        if not name or not isinstance(name, str):
            return False, "Invalid profile name."
        key = self.canonical_profile_name(name)
        if key not in CANONICAL_PROFILES:
            return False, (
                f"Unknown brain profile '{name}'. "
                f"Say optimal mode or full performance mode."
            )
        with self._lock:
            if key == self._active:
                logger.debug("Brain profile already %s — no-op", key)
                return True, f"Already in {self.display_name(key)}, Boss."

            now = time.monotonic()
            if (
                not force
                and self._last_switch_time
                and (now - self._last_switch_time) < self._COOLDOWN_S
            ):
                remaining = self._COOLDOWN_S - (now - self._last_switch_time)
                logger.debug("Brain profile cooldown (%.1fs remaining)", remaining)
                return False, "Profile switch on cooldown, Boss. Try again in a moment."

            if not force and self._rate_limited():
                logger.warning("Brain profile rate limit hit (%d in %.0fs)",
                               self._MAX_SWITCHES_PER_WINDOW, self._RATE_WINDOW_S)
                return False, "Too many profile switches. Try again in a minute, Boss."

            old = self._active
            self._active = key
            self._last_switch_time = now
            self._switch_timestamps.append(now)

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
        return True, f"Switched to {self.display_name(key)}, Boss. {_PROFILE_LABELS.get(key, '')}"

    def effective_params(self) -> dict[str, Any]:
        """Merged view for MiniLLM: base brain + active profile overrides."""
        with self._lock:
            prof_name = self._active
        base = dict(self._config.get("brain", {}))
        profiles = self._config.get("assistant_brain", {}).get("profiles", {})
        ov = dict(profiles.get(prof_name, {}) if isinstance(profiles, dict) else {})
        if not ov and isinstance(profiles, dict):
            for alias, canonical in PROFILE_ALIASES.items():
                if canonical == prof_name and alias in profiles:
                    ov = dict(profiles.get(alias, {}) or {})
                    break

        def _pick(key: str, default: Any) -> Any:
            if key in ov and ov[key] is not None and ov[key] != "":
                return ov[key]
            return base.get(key, default)

        model_path = _pick("model_path", base.get("model_path", ""))

        extra_stops = ov.get("extra_stop_sequences")
        if not isinstance(extra_stops, list):
            extra_stops = []

        return {
            "profile": prof_name,
            "model_path": str(model_path),
            "n_ctx": int(_pick("n_ctx", 8192)),
            "n_threads": int(_pick("n_threads", max(2, (os.cpu_count() or 4) // 2))),
            "n_gpu_layers": int(_pick("n_gpu_layers", -1)),
            "n_batch": int(_pick("n_batch", 512)),
            "max_tokens": int(_pick("max_tokens", 512)),
            "temperature": float(_pick("temperature", 0.7)),
            "top_p": float(_pick("top_p", 0.9)),
            "repeat_penalty": float(_pick("repeat_penalty", 1.1)),
            "timeout_seconds": float(_pick("timeout_seconds", 30)),
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
            if isinstance(v, str):
                return self.canonical_profile_name(v)
        except Exception:
            logger.debug('Directory ensure failed', exc_info=True)
        return None
