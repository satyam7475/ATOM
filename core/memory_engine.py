"""
ATOM v14 -- Lightweight keyword-match memory engine + interaction logger.

Zero external dependencies -- no ML frameworks.
Stores recent Q&A pairs in a JSON file and retrieves by keyword overlap.
Supports configurable max_entries and top_k from settings.json.

Additionally logs every classified interaction (command, action,
system state, result, time-of-day) to logs/interactions.json for
behavior analysis and preference derivation.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("atom.memory")

_PERSIST_FILE = Path("logs/memory.json")
_INTERACTIONS_FILE = Path("logs/interactions.json")
_MAX_INTERACTIONS = 5000

TECH_KEYWORD_PATTERN = re.compile(
    r"\b(process|system|cpu|memory|disk|network|battery|resource|"
    r"automate|schedule|reminder|scroll|click|desktop|browser|"
    r"api|sql|docker|deploy|pipeline|config|configure|configuration|python|java|node|git|"
    r"install|update|backup|monitor|performance|diagnostic|"
    r"spring|kafka|kubernetes|gradle|maven)\b",
    re.IGNORECASE,
)

_TIME_SLOTS = {
    range(5, 12): "morning",
    range(12, 17): "afternoon",
    range(17, 21): "evening",
}


def _time_of_day(hour: int | None = None) -> str:
    if hour is None:
        hour = datetime.now().hour
    for rng, label in _TIME_SLOTS.items():
        if hour in rng:
            return label
    return "night"


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"\w{3,}", text)}


class MemoryEngine:
    """Simple keyword-overlap memory + full interaction log."""

    def __init__(self, config: dict | None = None) -> None:
        mem_cfg = (config or {}).get("memory", {})
        self._max_entries: int = mem_cfg.get("max_entries", 500)
        self._default_top_k: int = mem_cfg.get("top_k", 3)
        self._entries: list[dict] = []
        self._interactions: list[dict] = []
        self._interactions_dirty: bool = False
        self._load()
        self._load_interactions()

    # ── Q&A Memory (unchanged) ────────────────────────────────────────

    def _load(self) -> None:
        if _PERSIST_FILE.exists():
            try:
                with open(_PERSIST_FILE, "r", encoding="utf-8") as f:
                    self._entries = json.load(f)
                logger.info("Memory loaded: %d entries", len(self._entries))
            except Exception:
                self._entries = []

    @staticmethod
    def should_store(query: str) -> bool:
        if len(query.split()) > 10:
            return True
        if TECH_KEYWORD_PATTERN.search(query):
            return True
        return False

    async def add(self, query: str, summary: str) -> None:
        if not self.should_store(query):
            return

        from context.privacy_filter import redact as _redact
        self._entries.append({
            "query": _redact(query),
            "summary": _redact(summary),
            "keywords": list(_tokenize(query + " " + summary)),
            "timestamp": time.time(),
        })

        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

    async def retrieve(self, query: str, k: int = 2) -> list[str]:
        if not self._entries:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scored: list[tuple[float, str]] = []
        for entry in self._entries:
            entry_tokens = set(entry.get("keywords", []))
            overlap = len(query_tokens & entry_tokens)
            if overlap > 0:
                scored.append((overlap, entry["summary"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:k]]

    def persist(self) -> None:
        _PERSIST_FILE.parent.mkdir(exist_ok=True)
        try:
            with open(_PERSIST_FILE, "w", encoding="utf-8") as f:
                json.dump(self._entries, f)
            try:
                import os
                os.chmod(_PERSIST_FILE, 0o600)
            except OSError:
                pass
            logger.info("Memory persisted: %d entries", len(self._entries))
        except Exception:
            logger.exception("Failed to persist memory")

        self._persist_interactions()

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    # ── Interaction Log (new) ─────────────────────────────────────────

    def _load_interactions(self) -> None:
        if _INTERACTIONS_FILE.exists():
            try:
                with open(_INTERACTIONS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self._interactions = data[-_MAX_INTERACTIONS:]
                logger.info("Interactions loaded: %d entries",
                            len(self._interactions))
            except Exception:
                self._interactions = []

    def log_interaction(
        self,
        command: str,
        action: str,
        system_state: dict | None = None,
        result: str = "success",
    ) -> None:
        """Record a classified interaction with full context."""
        now = datetime.now()
        entry = {
            "command": (command or "")[:200],
            "action": action,
            "timestamp": time.time(),
            "system_state": system_state or {},
            "result": result,
            "time_of_day": _time_of_day(now.hour),
            "hour": now.hour,
            "weekday": now.weekday(),
        }
        self._interactions.append(entry)
        if len(self._interactions) > _MAX_INTERACTIONS:
            self._interactions = self._interactions[-_MAX_INTERACTIONS:]
        self._interactions_dirty = True

    def _persist_interactions(self) -> None:
        if not self._interactions_dirty:
            return
        try:
            _INTERACTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(_INTERACTIONS_FILE, "w", encoding="utf-8") as f:
                json.dump(self._interactions, f, separators=(",", ":"))
            try:
                import os
                os.chmod(_INTERACTIONS_FILE, 0o600)
            except OSError:
                pass
            self._interactions_dirty = False
            logger.debug("Interactions persisted: %d entries",
                         len(self._interactions))
        except Exception:
            logger.debug("Failed to persist interactions", exc_info=True)

    @property
    def interaction_count(self) -> int:
        return len(self._interactions)

    # ── Preferences (derived from interactions) ───────────────────────

    @property
    def preferences(self) -> dict:
        """Derive user preferences from interaction history."""
        if not self._interactions:
            return {}

        prefs: dict = {}

        app_counts: Counter = Counter()
        action_counts: Counter = Counter()
        tod_counts: Counter = Counter()

        for ix in self._interactions:
            action_counts[ix.get("action", "")] += 1
            tod_counts[ix.get("time_of_day", "")] += 1
            if ix.get("action") == "open_app":
                target = ix.get("command", "")
                if target:
                    app_counts[target] += 1

        if app_counts:
            prefs["top_apps"] = [
                app for app, _ in app_counts.most_common(5)
            ]

        if action_counts:
            prefs["top_actions"] = [
                act for act, _ in action_counts.most_common(10)
            ]

        if tod_counts:
            prefs["most_active_time"] = tod_counts.most_common(1)[0][0]

        prefs["total_interactions"] = len(self._interactions)
        return prefs
