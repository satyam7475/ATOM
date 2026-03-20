"""
ATOM v14 -- User Behavior Tracker with Habit Detection Engine.

Logs timestamped user actions to logs/behavior.json and detects
recurring patterns as "habits" with confidence scores.

Habits are stored separately in logs/habits.json with:
  - Confidence scores (0.0 to 1.0) adjusted by user feedback
  - Time-of-day patterns (morning/afternoon/evening/night)
  - Day patterns (weekday/weekend/daily)
  - Decay over time for unused habits

Max 2000 raw entries (~100KB), max 50 habits. Zero external deps.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom.behavior")

_BEHAVIOR_FILE = Path("logs/behavior.json")
_HABITS_FILE = Path("logs/habits.json")
_MAX_ENTRIES = 2000
_MIN_OCCURRENCES = 3
_MAX_HABITS = 50

_DEFAULT_DECAY_DAYS = 7
_DEFAULT_DECAY_RATE = 0.02
_INITIAL_CONFIDENCE = 0.3

_TIME_SLOTS: dict[str, range] = {
    "morning": range(5, 12),
    "afternoon": range(12, 17),
    "evening": range(17, 21),
    "night_early": range(21, 24),
    "night_late": range(0, 5),
}


def _time_pattern(hour: int) -> str:
    for label, rng in _TIME_SLOTS.items():
        if hour in rng:
            if label in ("night_early", "night_late"):
                return "night"
            return label
    return "night"


def _day_pattern(weekday: int) -> str:
    return "weekday" if weekday < 5 else "weekend"


def _habit_id(action: str, target: str, time_pat: str, day_pat: str) -> str:
    safe_target = (target or "").replace(" ", "_")[:30]
    return f"{action}_{safe_target}_{time_pat}_{day_pat}".strip("_")


class BehaviorTracker:
    """Track user actions, detect habits with confidence scoring."""

    def __init__(self, config: dict | None = None) -> None:
        auto_cfg = (config or {}).get("autonomy", {})
        self._decay_days: int = auto_cfg.get("habit_decay_days", _DEFAULT_DECAY_DAYS)
        self._decay_rate: float = auto_cfg.get("habit_decay_rate", _DEFAULT_DECAY_RATE)
        self._max_habits: int = auto_cfg.get("max_habits", _MAX_HABITS)

        self._entries: list[dict] = []
        self._habits: dict[str, dict] = {}
        self._dirty = False
        self._habits_dirty = False
        self._last_decay_time: float = time.time()
        self._log_since_detect: int = 0
        self._detect_every_n: int = 10
        self._load()
        self._load_habits()

    # ── Persistence ───────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if _BEHAVIOR_FILE.exists():
                data = json.loads(_BEHAVIOR_FILE.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self._entries = data[-_MAX_ENTRIES:]
                    logger.info("Loaded %d behavior entries", len(self._entries))
        except Exception:
            logger.debug("No behavior log found, starting fresh")
            self._entries = []

    def _load_habits(self) -> None:
        try:
            if _HABITS_FILE.exists():
                data = json.loads(_HABITS_FILE.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._habits = data
                    logger.info("Loaded %d habits", len(self._habits))
        except Exception:
            logger.debug("No habits file found, starting fresh")
            self._habits = {}

    def persist(self) -> None:
        """Flush raw entries and habits to disk."""
        if self._dirty:
            try:
                _BEHAVIOR_FILE.parent.mkdir(parents=True, exist_ok=True)
                _BEHAVIOR_FILE.write_text(
                    json.dumps(self._entries, separators=(",", ":")),
                    encoding="utf-8",
                )
                try:
                    import os
                    os.chmod(_BEHAVIOR_FILE, 0o600)
                except OSError:
                    pass
                self._dirty = False
                logger.debug("Behavior log saved (%d entries)", len(self._entries))
            except Exception:
                logger.debug("Failed to save behavior log", exc_info=True)

        self._persist_habits()

    def _persist_habits(self) -> None:
        if not self._habits_dirty:
            return
        try:
            _HABITS_FILE.parent.mkdir(parents=True, exist_ok=True)
            _HABITS_FILE.write_text(
                json.dumps(self._habits, indent=2),
                encoding="utf-8",
            )
            try:
                import os
                os.chmod(_HABITS_FILE, 0o600)
            except OSError:
                pass
            self._habits_dirty = False
            logger.debug("Habits saved (%d habits)", len(self._habits))
        except Exception:
            logger.debug("Failed to save habits", exc_info=True)

    # ── Action Logging ────────────────────────────────────────────────

    def log(self, action: str, target: str = "") -> None:
        """Record a user action with timestamp context."""
        now = datetime.now()
        entry = {
            "action": action,
            "target": target.lower()[:50] if target else "",
            "hour": now.hour,
            "weekday": now.weekday(),
            "ts": time.time(),
        }
        self._entries.append(entry)
        if len(self._entries) > _MAX_ENTRIES:
            self._entries = self._entries[-_MAX_ENTRIES:]
        self._dirty = True

        self._log_since_detect += 1
        if self._log_since_detect >= self._detect_every_n:
            self._log_since_detect = 0
            self._detect_habits()

    # ── Habit Detection ───────────────────────────────────────────────

    def _detect_habits(self) -> None:
        """Scan raw entries and create/update habit records."""
        if len(self._entries) < _MIN_OCCURRENCES:
            return

        pattern_groups: dict[str, list[dict]] = defaultdict(list)
        for entry in self._entries:
            action = entry.get("action", "")
            target = entry.get("target", "")
            hour = entry.get("hour", 0)
            weekday = entry.get("weekday", 0)
            tp = _time_pattern(hour)
            dp = _day_pattern(weekday)
            hid = _habit_id(action, target, tp, dp)
            pattern_groups[hid].append(entry)

        for hid, group in pattern_groups.items():
            if len(group) < _MIN_OCCURRENCES:
                continue

            sample = group[-1]
            action = sample.get("action", "")
            target = sample.get("target", "")
            hour = sample.get("hour", 0)
            weekday = sample.get("weekday", 0)

            if hid in self._habits:
                existing = self._habits[hid]
                old_count = existing.get("occurrences", 0)
                new_count = len(group)
                if new_count > old_count:
                    bump = min(0.05 * (new_count - old_count), 0.15)
                    existing["confidence"] = min(
                        1.0, existing["confidence"] + bump,
                    )
                    existing["occurrences"] = new_count
                    existing["last_seen"] = sample.get("ts", time.time())
                    self._habits_dirty = True
            else:
                if len(self._habits) >= self._max_habits:
                    self._evict_weakest_habit()

                self._habits[hid] = {
                    "id": hid,
                    "action": action,
                    "target": target,
                    "time_pattern": _time_pattern(hour),
                    "day_pattern": _day_pattern(weekday),
                    "confidence": _INITIAL_CONFIDENCE,
                    "occurrences": len(group),
                    "last_seen": sample.get("ts", time.time()),
                    "last_executed": None,
                    "auto_execute": False,
                }
                self._habits_dirty = True
                logger.info("New habit detected: %s (count=%d)", hid, len(group))

    def _evict_weakest_habit(self) -> None:
        """Remove the habit with lowest confidence to make room."""
        if not self._habits:
            return
        weakest = min(self._habits, key=lambda k: self._habits[k]["confidence"])
        del self._habits[weakest]
        self._habits_dirty = True

    # ── Confidence Management ─────────────────────────────────────────

    def adjust_confidence(self, habit_id: str, delta: float) -> None:
        """Adjust a habit's confidence score, bounded to [0.0, 1.0]."""
        if habit_id not in self._habits:
            return
        habit = self._habits[habit_id]
        habit["confidence"] = max(0.0, min(1.0, habit["confidence"] + delta))
        habit["auto_execute"] = habit["confidence"] >= 0.8
        self._habits_dirty = True
        logger.debug("Habit %s confidence -> %.2f", habit_id, habit["confidence"])

    def apply_decay(self) -> None:
        """Reduce confidence for habits not seen recently."""
        now = time.time()
        if now - self._last_decay_time < 3600:
            return
        self._last_decay_time = now

        cutoff = now - (self._decay_days * 86400)
        changed = False
        to_remove: list[str] = []

        for hid, habit in self._habits.items():
            if habit.get("last_seen", now) < cutoff:
                habit["confidence"] = max(
                    0.0, habit["confidence"] - self._decay_rate,
                )
                habit["auto_execute"] = habit["confidence"] >= 0.8
                changed = True
                if habit["confidence"] <= 0.0:
                    to_remove.append(hid)

        for hid in to_remove:
            del self._habits[hid]
            logger.info("Habit decayed to zero, removed: %s", hid)
            changed = True

        if changed:
            self._habits_dirty = True

    # ── Habit Queries ─────────────────────────────────────────────────

    def get_active_habits(self, context: dict | None = None) -> list[dict]:
        """Return habits matching current context with confidence > 0.3."""
        if not self._habits:
            return []

        ctx = context or {}
        now = datetime.now()
        current_tp = ctx.get("time_of_day", _time_pattern(now.hour))
        current_dp = _day_pattern(ctx.get("weekday", now.weekday()))

        matched: list[dict] = []
        for habit in self._habits.values():
            if habit["confidence"] <= 0.3:
                continue
            if habit.get("time_pattern") == current_tp or current_tp in habit.get("time_pattern", ""):
                if habit.get("day_pattern") in (current_dp, "daily"):
                    matched.append(habit)

        matched.sort(key=lambda h: h["confidence"], reverse=True)
        return matched

    def get_auto_habits(self, context: dict | None = None) -> list[dict]:
        """Return habits with confidence >= 0.8 matching current context."""
        return [
            h for h in self.get_active_habits(context)
            if h["confidence"] >= 0.8
        ]

    # ── Legacy predict() (backward compatible) ────────────────────────

    def predict(self) -> list[str]:
        """Scan for patterns matching the current time and return suggestions.

        Groups entries by (action, target, hour, is_weekday).
        If 3+ entries match a pattern and the current time aligns,
        returns a human-readable suggestion.
        """
        if len(self._entries) < _MIN_OCCURRENCES:
            return []

        now = datetime.now()
        current_hour = now.hour
        is_weekday = now.weekday() < 5

        pattern_counts: dict[tuple, int] = defaultdict(int)
        for entry in self._entries:
            key = (
                entry.get("action", ""),
                entry.get("target", ""),
                entry.get("hour", -1),
                entry.get("weekday", -1) < 5,
            )
            pattern_counts[key] += 1

        suggestions = []
        for (action, target, hour, entry_weekday), count in pattern_counts.items():
            if count < _MIN_OCCURRENCES:
                continue
            if hour != current_hour:
                continue
            if entry_weekday != is_weekday:
                continue

            suggestion = self._format_suggestion(action, target, count)
            if suggestion:
                suggestions.append(suggestion)

        return suggestions[:2]

    @staticmethod
    def _format_suggestion(action: str, target: str, count: int) -> str:
        """Build a natural-language suggestion from a pattern."""
        if action == "open_app" and target:
            app_name = target.replace(".exe", "").title()
            return f"You usually open {app_name} around this time. Want me to open it?"
        if action == "search" and target:
            return "You often search around this time. Need anything looked up?"
        if action == "play_youtube":
            return "Time for some music? You usually play something around now."
        if action == "weather":
            return "Want me to check the weather? You usually ask around this time."
        return ""

    @staticmethod
    def format_habit_suggestion(habit: dict) -> str:
        """Build a suggestion string from a habit dict."""
        action = habit.get("action", "")
        target = habit.get("target", "")

        if action == "open_app" and target:
            app_name = target.replace(".exe", "").replace("_", " ").title()
            return f"Boss, you usually open {app_name} around this time. Want me to open it?"
        if action == "search":
            return "Boss, you often search around this time. Need anything looked up?"
        if action == "play_youtube":
            return "Time for some music, Boss? You usually play something around now."
        if action == "weather":
            return "Want me to check the weather, Boss? You usually ask around this time."
        if action and target:
            return f"Boss, you often use {action.replace('_', ' ')} for {target} around this time. Should I?"
        if action:
            return f"Boss, you often {action.replace('_', ' ')} around this time. Want me to?"
        return ""

    # ── Properties ────────────────────────────────────────────────────

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def habit_count(self) -> int:
        return len(self._habits)

    @property
    def habits(self) -> dict[str, dict]:
        return dict(self._habits)
