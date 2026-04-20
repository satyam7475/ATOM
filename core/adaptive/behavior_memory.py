"""
ATOM -- Lightweight behavioral memory.

Maintains a rolling window of TTS delivery metrics and derives a
user profile (preferred rate, verbosity, interrupt tolerance) using
simple running statistics.

Sprint D5 adds opt-in persistence: the learned profile (not the raw
history — that's per-session and noisy) is saved to disk after each
update cycle and restored on boot so ATOM doesn't "forget" how the
owner likes to be talked to every time the laptop reboots.

Owner: Satyam
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom.adaptive.memory")


_DEFAULT_PROFILE: dict[str, float] = {
    "preferred_rate": 1.0,
    "preferred_pause": 1.0,
    "interrupt_tolerance": 0.5,
    "verbosity": 0.5,
}

_PERSIST_INTERVAL_S = 30.0  # debounce disk writes


class BehaviorMemory:
    """Rolling-window behavioral learning from delivery metrics."""

    __slots__ = (
        "_history", "_user_profile", "_max_history",
        "_persist_path", "_last_persist_t", "_persist_enabled",
    )

    def __init__(
        self,
        max_history: int = 50,
        persist_path: str | Path | None = "data/behavior_profile.json",
    ) -> None:
        self._max_history = max_history
        self._history: deque[dict[str, Any]] = deque(maxlen=max_history)
        self._user_profile: dict[str, float] = dict(_DEFAULT_PROFILE)
        self._persist_path: Path | None = (
            Path(persist_path) if persist_path else None
        )
        self._persist_enabled = self._persist_path is not None
        self._last_persist_t: float = 0.0
        if self._persist_enabled:
            self._restore_profile()

    def record(self, metrics: dict[str, Any]) -> None:
        self._history.append(metrics)

    def get_profile(self) -> dict[str, float]:
        return dict(self._user_profile)

    def update_from_metrics(self) -> None:
        """Recompute user profile from the full rolling window."""
        if not self._history:
            return

        interrupts = [m.get("interrupt_count", 0) for m in self._history]
        durations = [m.get("duration_ms", 0) for m in self._history]
        words = [m.get("words_spoken", 0) for m in self._history]

        avg_interrupts = sum(interrupts) / len(interrupts)
        total_ms = max(1.0, sum(durations))
        avg_wpm = sum(words) / (total_ms / 60_000.0)

        p = self._user_profile

        if avg_interrupts > 1.5:
            p["verbosity"] = max(0.2, p["verbosity"] - 0.1)
        elif avg_interrupts < 0.3:
            p["verbosity"] = min(0.8, p["verbosity"] + 0.05)

        if avg_interrupts > 1.0:
            p["preferred_rate"] = min(1.3, p["preferred_rate"] + 0.08)
        elif avg_interrupts < 0.3:
            p["preferred_rate"] = max(0.85, p["preferred_rate"] - 0.02)

        p["preferred_pause"] = round(2.0 - p["preferred_rate"], 3)

        p["interrupt_tolerance"] = round(
            max(0.0, min(1.0, 1.0 - avg_interrupts / 3.0)), 3
        )

        self._decay_toward_defaults()

        logger.debug(
            "Profile updated: verb=%.2f rate=%.2f pause=%.2f tol=%.2f (wpm=%.0f)",
            p["verbosity"], p["preferred_rate"],
            p["preferred_pause"], p["interrupt_tolerance"],
            avg_wpm,
        )

        self._maybe_persist()

    def _decay_toward_defaults(self) -> None:
        """Slowly pull profile back toward neutral when behavior normalizes.

        Called on every update cycle so the profile never gets permanently
        stuck at an extreme.  The pull is gentle enough that active signals
        (e.g. frequent interrupts) easily overpower it.
        """
        p = self._user_profile
        p["preferred_rate"] += (1.0 - p["preferred_rate"]) * 0.02
        p["preferred_pause"] += (1.0 - p["preferred_pause"]) * 0.02
        p["verbosity"] += (0.5 - p["verbosity"]) * 0.02

    # ── Persistence (Sprint D5) ──────────────────────────────────

    def _maybe_persist(self, *, force: bool = False) -> None:
        if not self._persist_enabled or self._persist_path is None:
            return
        now = time.monotonic()
        if not force and (now - self._last_persist_t) < _PERSIST_INTERVAL_S:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._persist_path.with_name(self._persist_path.name + ".tmp")
            payload = {
                "profile": {k: float(v) for k, v in self._user_profile.items()},
                "saved_at": time.time(),
                "version": 1,
            }
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self._persist_path)
            self._last_persist_t = now
            logger.debug(
                "Behavior profile persisted to %s", self._persist_path,
            )
        except Exception:
            logger.debug("Behavior profile persist failed", exc_info=True)

    def _restore_profile(self) -> None:
        if self._persist_path is None or not self._persist_path.exists():
            return
        try:
            raw = json.loads(self._persist_path.read_text(encoding="utf-8"))
            prof = raw.get("profile") if isinstance(raw, dict) else None
            if not isinstance(prof, dict):
                return
            for key in _DEFAULT_PROFILE:
                if key in prof:
                    try:
                        self._user_profile[key] = float(prof[key])
                    except (TypeError, ValueError):
                        pass
            logger.info(
                "Behavior profile restored from %s (verb=%.2f, rate=%.2f)",
                self._persist_path,
                self._user_profile.get("verbosity", 0.5),
                self._user_profile.get("preferred_rate", 1.0),
            )
        except Exception:
            logger.debug("Behavior profile restore failed", exc_info=True)

    def flush(self) -> None:
        """Force-save the current profile (call on shutdown)."""
        self._maybe_persist(force=True)
