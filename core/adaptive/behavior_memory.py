"""
ATOM -- Lightweight behavioral memory.

Maintains a rolling window of TTS delivery metrics and derives a
user profile (preferred rate, verbosity, interrupt tolerance) using
simple running statistics.  No ML, no DB — structured for future
upgrade to persistent storage.

Owner: Satyam
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

logger = logging.getLogger("atom.adaptive.memory")


class BehaviorMemory:
    """Rolling-window behavioral learning from delivery metrics."""

    __slots__ = ("_history", "_user_profile", "_max_history")

    def __init__(self, max_history: int = 50) -> None:
        self._max_history = max_history
        self._history: deque[dict[str, Any]] = deque(maxlen=max_history)
        self._user_profile: dict[str, float] = {
            "preferred_rate": 1.0,
            "preferred_pause": 1.0,
            "interrupt_tolerance": 0.5,
            "verbosity": 0.5,
        }

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
