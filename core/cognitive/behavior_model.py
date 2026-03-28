"""
ATOM -- Personal Behavior Model + Energy Awareness.

Builds a dynamic user profile from observed behavior:
  - Peak productivity hours
  - Focus patterns per app
  - App categorization (deep_work / communication / browsing)
  - Energy state inference (high / medium / low / resting)
  - Session tracking (actions, breaks, dominant category)

Energy state inference uses NO ML -- pure heuristics based on:
  action rate, app switching frequency, idle time, and time-of-day.

Profile updates are debounced: energy state every snapshot,
full profile recalculation every 15 minutes.

Persistence: logs/user_profile.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus

logger = logging.getLogger("atom.behavior_model")

_PROFILE_FILE = Path("logs/user_profile.json")

_APP_CATEGORIES: dict[str, list[str]] = {
    "deep_work": [
        "code", "cursor", "visual studio", "pycharm", "intellij",
        "jupyter", "notepad++", "sublime", "vim", "emacs", "android studio",
        "unity", "blender", "autocad", "matlab",
    ],
    "communication": [
        "slack", "teams", "outlook", "gmail", "thunderbird", "zoom",
        "skype", "discord", "whatsapp", "telegram", "webex",
    ],
    "browsing": [
        "chrome", "edge", "firefox", "brave", "safari", "opera", "vivaldi",
    ],
    "productivity": [
        "word", "excel", "powerpoint", "notion", "obsidian", "onenote",
        "google docs", "google sheets", "trello", "asana", "jira",
    ],
    "media": [
        "spotify", "youtube", "vlc", "netflix", "prime video",
        "music", "photos", "paint",
    ],
}

_ENERGY_THRESHOLDS = {
    "high": {"min_actions_10m": 5, "max_switch_rate": 0.3},
    "medium": {"min_actions_10m": 2, "max_switch_rate": 0.6},
    "low": {"min_actions_10m": 0, "max_switch_rate": 1.0},
}


def _categorize_app(app_title: str) -> str:
    """Determine app category from window title."""
    title_lower = (app_title or "").lower()
    for category, keywords in _APP_CATEGORIES.items():
        for kw in keywords:
            if kw in title_lower:
                return category
    return "other"


class BehaviorModel:
    """Personal behavior model with energy state inference."""

    __slots__ = (
        "_bus", "_config", "_profile", "_dirty",
        "_task", "_shutdown",
        "_action_timestamps", "_app_switches",
        "_last_full_update", "_last_app", "_focus_start",
        "_session_start", "_session_actions",
        "_update_interval", "_energy_interval",
    )

    def __init__(self, bus: AsyncEventBus, config: dict | None = None) -> None:
        self._bus = bus
        cfg = (config or {}).get("cognitive", {})
        self._config = cfg
        self._update_interval: float = max(300.0, cfg.get("behavior_update_interval_s", 900.0))
        self._energy_interval: float = cfg.get("energy_inference_interval_s", 120.0)

        self._profile: dict[str, Any] = {}
        self._dirty = False
        self._task: asyncio.Task | None = None
        self._shutdown = asyncio.Event()

        self._action_timestamps: list[float] = []
        self._app_switches: list[float] = []
        self._last_full_update: float = 0
        self._last_app: str = ""
        self._focus_start: float = time.time()
        self._session_start: float = time.time()
        self._session_actions: int = 0

        self._load()

    # ── Persistence ────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if _PROFILE_FILE.exists():
                self._profile = json.loads(
                    _PROFILE_FILE.read_text(encoding="utf-8")
                )
                logger.info("Behavior profile loaded")
        except Exception:
            self._profile = {}

    def persist(self) -> None:
        if not self._dirty:
            return
        try:
            _PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
            _PROFILE_FILE.write_text(
                json.dumps(self._profile, indent=2, default=str),
                encoding="utf-8",
            )
            self._dirty = False
            logger.debug("Behavior profile persisted")
        except Exception:
            logger.debug("Failed to persist behavior profile", exc_info=True)

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        if not self._config.get("behavior_model_enabled", True):
            logger.info("Behavior model disabled via config")
            return
        self._bus.on("context_snapshot", self._on_context_snapshot)
        self._bus.on("intent_classified", self._on_intent)
        self._task = asyncio.create_task(self._run())
        logger.info("Behavior model started")

    def stop(self) -> None:
        self._shutdown.set()
        if self._task and not self._task.done():
            self._task.cancel()
            self._task = None
        self.persist()

    async def _run(self) -> None:
        await asyncio.sleep(30.0)
        while not self._shutdown.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(), timeout=self._update_interval,
                )
                break
            except asyncio.TimeoutError:
                pass
            try:
                self._full_profile_update()
            except Exception:
                logger.exception("Behavior model update error")

    # ── Event Handlers ─────────────────────────────────────────────────

    async def _on_context_snapshot(self, **kw: Any) -> None:
        """Called every health check cycle. Update energy state."""
        active_app = kw.get("active_app", "")
        idle_min = kw.get("idle_minutes", 0)
        cpu = kw.get("cpu", 0)
        hour = kw.get("hour", datetime.now().hour)

        if active_app and active_app != self._last_app:
            self._app_switches.append(time.time())
            now = time.time()
            if self._last_app:
                duration = now - self._focus_start
                self._record_focus_session(self._last_app, duration)
            self._last_app = active_app
            self._focus_start = now

        cutoff = time.time() - 600
        self._app_switches = [t for t in self._app_switches if t > cutoff]

        energy = self._infer_energy(idle_min, hour)
        old_energy = self._profile.get("energy_state", {}).get("current", "")
        self._profile.setdefault("energy_state", {})
        self._profile["energy_state"]["current"] = energy
        self._profile["energy_state"]["last_updated"] = time.time()
        self._dirty = True

        if energy != old_energy:
            self._bus.emit_fast(
                "user_energy_state",
                energy=energy,
                idle_minutes=idle_min,
                active_app=active_app,
            )

    async def _on_intent(self, intent: str = "", **_kw: Any) -> None:
        """Track action timestamps for energy inference."""
        if intent and intent not in ("empty", "fallback"):
            self._action_timestamps.append(time.time())
            self._session_actions += 1
            cutoff = time.time() - 600
            self._action_timestamps = [t for t in self._action_timestamps if t > cutoff]

    # ── Energy Inference ───────────────────────────────────────────────

    def _infer_energy(self, idle_min: float, hour: int) -> str:
        """Infer user energy state from activity signals. No ML."""
        if idle_min > 5 or (hour >= 23 or hour < 5):
            return "resting"

        now = time.time()
        cutoff_10m = now - 600
        actions_10m = sum(1 for t in self._action_timestamps if t > cutoff_10m)
        switches_10m = sum(1 for t in self._app_switches if t > cutoff_10m)

        switch_rate = switches_10m / max(1, actions_10m)

        if actions_10m >= 5 and switch_rate <= 0.3:
            return "high"
        elif actions_10m >= 2 and switch_rate <= 0.6:
            return "medium"
        else:
            return "low"

    # ── Focus Session Recording ────────────────────────────────────────

    def _record_focus_session(self, app_title: str, duration_s: float) -> None:
        """Record how long the user stayed in one app."""
        if duration_s < 10:
            return

        category = _categorize_app(app_title)
        focus = self._profile.setdefault("focus_patterns", {})
        cat_key = category
        entry = focus.setdefault(cat_key, {"total_s": 0, "sessions": 0})
        entry["total_s"] = entry.get("total_s", 0) + duration_s
        entry["sessions"] = entry.get("sessions", 0) + 1
        entry["avg_session_min"] = round(
            entry["total_s"] / max(1, entry["sessions"]) / 60, 1
        )
        self._dirty = True

    # ── Full Profile Update (debounced to every ~15 min) ───────────────

    def _full_profile_update(self) -> None:
        """Recalculate peak hours and productivity scores."""
        now = time.time()
        if now - self._last_full_update < self._update_interval:
            return
        self._last_full_update = now

        self._profile["app_categories"] = dict(_APP_CATEGORIES)

        session = self._profile.setdefault("session", {})
        session["started"] = self._session_start
        session["actions_count"] = self._session_actions
        session["uptime_min"] = round((now - self._session_start) / 60, 1)

        focus = self._profile.get("focus_patterns", {})
        if focus:
            dominant = max(focus.items(), key=lambda x: x[1].get("total_s", 0))
            session["dominant_category"] = dominant[0]

        self._dirty = True
        self.persist()
        logger.debug("Full behavior profile updated")

    # ── Queries ────────────────────────────────────────────────────────

    @property
    def energy_state(self) -> str:
        return self._profile.get("energy_state", {}).get("current", "unknown")

    def should_interrupt(self) -> bool:
        """Returns False if user is in deep focus."""
        energy = self.energy_state
        if energy == "high":
            focus = self._profile.get("focus_patterns", {})
            deep = focus.get("deep_work", {})
            if deep.get("sessions", 0) > 0:
                current_focus_min = (time.time() - self._focus_start) / 60
                if current_focus_min > 15:
                    return False
        return True

    def get_scheduling_advice(self, task_minutes: int = 30) -> str:
        """Suggest optimal time for a task based on behavior patterns."""
        energy = self.energy_state
        hour = datetime.now().hour
        if energy == "high":
            return f"You're in peak focus right now. Good time for a {task_minutes}-minute task."
        elif energy == "resting":
            return "You seem to be resting. Maybe tackle this when you're more active."
        elif 9 <= hour <= 11:
            return "Morning hours are usually productive. This could be a good time."
        elif 14 <= hour <= 16:
            return "Afternoon focus window. Good for moderate tasks."
        return "Consider scheduling demanding tasks during your peak hours."

    def get_profile_summary(self) -> str:
        energy = self.energy_state
        session = self._profile.get("session", {})
        focus = self._profile.get("focus_patterns", {})

        parts = [f"Current energy: {energy}."]
        if session.get("uptime_min"):
            parts.append(f"Session: {session['uptime_min']:.0f} min, {session.get('actions_count', 0)} actions.")
        if session.get("dominant_category"):
            parts.append(f"Dominant activity: {session['dominant_category']}.")
        if focus:
            top = sorted(focus.items(), key=lambda x: x[1].get("total_s", 0), reverse=True)[:3]
            focus_strs = [f"{k}: {v.get('avg_session_min', 0):.0f}min avg" for k, v in top]
            parts.append(f"Focus: {', '.join(focus_strs)}.")
        return " ".join(parts)

    def get_profile_for_dashboard(self) -> dict:
        return {
            "energy_state": self.energy_state,
            "session_actions": self._profile.get("session", {}).get("actions_count", 0),
            "session_uptime_min": self._profile.get("session", {}).get("uptime_min", 0),
            "dominant_category": self._profile.get("session", {}).get("dominant_category", ""),
            "focus_patterns": self._profile.get("focus_patterns", {}),
        }
