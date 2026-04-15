"""
ATOM -- User Memory (unified long-term owner model).

Aggregates the scattered persistence layers into a single queryable
interface for the Router and LLM prompt builder:

  - PreferenceStore: explicit learned preferences (language, style, etc.)
  - BehaviorTracker: detected habits with confidence scores
  - Routines: time-of-day patterns (morning coding, evening browsing, etc.)

Persists a compact user profile to ``data/user_profile.json`` that
survives restarts and can be loaded in <5ms at boot.

This is what makes ATOM say "I know you prefer Safari" instead of
asking every time.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom.user_memory")

_PROFILE_PATH = Path("data/user_profile.json")


_BROWSER_APPS = frozenset({
    "Safari", "Google Chrome", "Firefox", "Arc", "Brave Browser",
    "Microsoft Edge",
})
_EDITOR_APPS = frozenset({
    "Code", "Visual Studio Code", "Xcode", "PyCharm", "IntelliJ",
    "Sublime Text", "TextEdit", "BBEdit", "Nova", "Cursor",
})


class UserMemory:
    """Unified long-term user model.

    Wraps PreferenceStore + BehaviorTracker + local profile JSON
    into a single interface for context injection.
    """

    def __init__(
        self,
        *,
        preference_store: Any = None,
        behavior_tracker: Any = None,
        profile_path: str | Path | None = None,
    ) -> None:
        self._pref_store = preference_store
        self._behavior = behavior_tracker
        self._profile_path = Path(profile_path or _PROFILE_PATH)
        self._profile: dict[str, Any] = self._load_profile()
        self._dirty = False
        self._bus: Any = None
        self._last_auto_learn_app: str = ""

    def _load_profile(self) -> dict[str, Any]:
        try:
            if self._profile_path.exists():
                data = json.loads(self._profile_path.read_text(encoding="utf-8"))
                logger.info("User profile loaded (%d keys)", len(data))
                return data
        except Exception:
            logger.debug("User profile load failed", exc_info=True)
        return self._default_profile()

    def _default_profile(self) -> dict[str, Any]:
        return {
            "preferred_browser": "",
            "preferred_editor": "",
            "coding_hours": [],
            "frequent_apps": [],
            "communication_style": "professional",
            "response_length": "concise",
            "name": "Boss",
            "routines": {},
            "updated_at": 0.0,
        }

    def persist(self) -> None:
        """Save the profile to disk."""
        if not self._dirty and self._profile_path.exists():
            return
        try:
            self._profile["updated_at"] = time.time()
            self._profile_path.parent.mkdir(parents=True, exist_ok=True)
            self._profile_path.write_text(
                json.dumps(self._profile, indent=2, default=str),
                encoding="utf-8",
            )
            self._dirty = False
        except Exception:
            logger.debug("User profile persist failed", exc_info=True)

    def learn_app_preference(self, category: str, app_name: str) -> None:
        """Learn which app the user prefers for a category.

        e.g. learn_app_preference("browser", "Safari")
        """
        key = f"preferred_{category}"
        self._profile[key] = app_name
        self._dirty = True

        if self._pref_store is not None:
            self._pref_store.learn("apps", category, app_name, confidence=0.7)

    def learn_routine(self, time_slot: str, activity: str) -> None:
        """Record a routine for a time slot (morning, afternoon, evening, night)."""
        routines = self._profile.setdefault("routines", {})
        slot_data = routines.setdefault(time_slot, {})
        count = slot_data.get(activity, 0) + 1
        slot_data[activity] = count
        self._dirty = True

    def track_app_usage(self, app_name: str) -> None:
        """Increment usage counter for an app."""
        if not app_name:
            return
        freq = self._profile.setdefault("frequent_apps_count", {})
        freq[app_name] = freq.get(app_name, 0) + 1
        self._dirty = True

        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        self._profile["frequent_apps"] = [a for a, _ in top[:10]]

    def get_preference(self, key: str, default: str = "") -> str:
        """Get a profile preference."""
        val = self._profile.get(key, "")
        if val:
            return str(val)
        if self._pref_store is not None:
            parts = key.split("_", 1)
            if len(parts) == 2:
                stored = self._pref_store.get(parts[0], parts[1])
                if stored:
                    return stored
        return default

    def get_current_routine_hints(self) -> list[str]:
        """What the user typically does at this time."""
        now = datetime.now()
        hour = now.hour
        if 5 <= hour < 12:
            slot = "morning"
        elif 12 <= hour < 17:
            slot = "afternoon"
        elif 17 <= hour < 22:
            slot = "evening"
        else:
            slot = "night"

        routines = self._profile.get("routines", {})
        slot_data = routines.get(slot, {})
        if not slot_data:
            return []

        sorted_activities = sorted(slot_data.items(), key=lambda x: x[1], reverse=True)
        return [activity for activity, count in sorted_activities[:5] if count >= 2]

    def get_active_habits(self) -> list[dict[str, Any]]:
        """Get currently relevant habits from the behavior tracker."""
        if self._behavior is None:
            return []
        try:
            now = datetime.now()
            context = {
                "hour": now.hour,
                "weekday": now.weekday(),
            }
            return self._behavior.get_active_habits(context=context)
        except Exception:
            return []

    def context_for_prompt(self) -> str:
        """Generate a compact context block for LLM prompt injection."""
        lines: list[str] = []

        name = self._profile.get("name", "Boss")
        if name and name != "Boss":
            lines.append(f"[USER] Name: {name}")

        browser = self._profile.get("preferred_browser", "")
        editor = self._profile.get("preferred_editor", "")
        if browser:
            lines.append(f"[PREFS] Browser: {browser}")
        if editor:
            lines.append(f"[PREFS] Editor: {editor}")

        freq_apps = self._profile.get("frequent_apps", [])
        if freq_apps:
            lines.append(f"[APPS] Frequent: {', '.join(freq_apps[:5])}")

        routine_hints = self.get_current_routine_hints()
        if routine_hints:
            lines.append(f"[ROUTINE] Typical now: {', '.join(routine_hints[:3])}")

        if self._pref_store is not None:
            pref_block = self._pref_store.get_context_block()
            if pref_block:
                lines.append(pref_block)

        return "\n".join(lines)

    def wire_bus(self, bus: Any) -> None:
        """Subscribe to system_state_update to auto-learn preferences."""
        self._bus = bus
        bus.on("system_state_update", self._on_system_state_update)
        logger.info("UserMemory wired to bus for auto-learning")

    async def _on_system_state_update(
        self,
        snapshot: dict[str, Any] | None = None,
        changed_app: bool = False,
        **_kw: Any,
    ) -> None:
        """Auto-learn app preferences and routines from system state."""
        if not snapshot or not changed_app:
            return
        app = snapshot.get("active_app", "")
        if not app or app == self._last_auto_learn_app:
            return
        self._last_auto_learn_app = app

        self.track_app_usage(app)

        if app in _BROWSER_APPS:
            self.learn_app_preference("browser", app)
        elif app in _EDITOR_APPS:
            self.learn_app_preference("editor", app)

        now = datetime.now()
        hour = now.hour
        if 5 <= hour < 12:
            slot = "morning"
        elif 12 <= hour < 17:
            slot = "afternoon"
        elif 17 <= hour < 22:
            slot = "evening"
        else:
            slot = "night"
        self.learn_routine(slot, app)

        self.persist()

    def get_diagnostics(self) -> dict[str, Any]:
        habits = self.get_active_habits()
        return {
            "profile_keys": list(self._profile.keys()),
            "frequent_apps": self._profile.get("frequent_apps", [])[:5],
            "active_habits": len(habits),
            "routines": len(self._profile.get("routines", {})),
            "dirty": self._dirty,
        }


__all__ = ["UserMemory"]
