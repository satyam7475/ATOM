"""
ATOM v15 -- Named skills with optional multi-step chaining.

Loads `config/skills.json`. Case-insensitive trigger matching; first match
wins. When a skill has a `chain` list, those extra utterances are returned
for the Router to execute in sequence after the primary expansion.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom.skills")

_DEFAULT_PATH = Path("config/skills.json")


class SkillMatch:
    __slots__ = ("primary", "chain", "skill_id")

    def __init__(self, primary: str, chain: list[str], skill_id: str) -> None:
        self.primary = primary
        self.chain = chain
        self.skill_id = skill_id


class SkillsRegistry:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = (config or {}).get("skills", {}) or {}
        self._enabled: bool = bool(cfg.get("enabled", True))
        raw_path = cfg.get("path") or str(_DEFAULT_PATH)
        self._path = Path(raw_path)
        self._entries: list[tuple[str, str, list[str], str]] = []
        if self._enabled:
            self._load()

    def _load(self) -> None:
        self._entries = []
        if not self._path.is_file():
            logger.debug("Skills file not found: %s", self._path)
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not load skills from %s: %s", self._path, e)
            return
        skills = data.get("skills") if isinstance(data, dict) else None
        if not isinstance(skills, list):
            return
        for item in skills:
            if not isinstance(item, dict):
                continue
            expand = (item.get("expand_to") or "").strip()
            if not expand:
                continue
            sid = str(item.get("id", "")).strip() or "skill"
            chain_raw = item.get("chain") or []
            if isinstance(chain_raw, str):
                chain_raw = [chain_raw]
            chain = [str(c).strip() for c in chain_raw if str(c).strip()]
            triggers = item.get("triggers") or []
            if isinstance(triggers, str):
                triggers = [triggers]
            if not isinstance(triggers, list):
                continue
            for tr in triggers:
                t = str(tr).strip().lower()
                if t:
                    self._entries.append((t, expand, chain, sid))
        logger.info(
            "Skills registry: %d trigger(s), %d with chains from %s",
            len(self._entries),
            sum(1 for _, _, c, _ in self._entries if c),
            self._path,
        )

    def try_expand(self, clean_text: str) -> tuple[str, str] | None:
        """Backward-compatible: return (expanded, skill_id) or None."""
        if not self._enabled or not self._entries or not clean_text:
            return None
        low = clean_text.strip().lower()
        if not low:
            return None
        for trigger, expand, _chain, sid in self._entries:
            if low == trigger or trigger in low:
                if low == expand.lower():
                    return None
                return expand, sid
        return None

    def try_expand_full(self, clean_text: str) -> SkillMatch | None:
        """Return full SkillMatch (primary + chain) or None."""
        if not self._enabled or not self._entries or not clean_text:
            return None
        low = clean_text.strip().lower()
        if not low:
            return None
        for trigger, expand, chain, sid in self._entries:
            if low == trigger or trigger in low:
                if low == expand.lower():
                    return None
                return SkillMatch(expand, chain, sid)
        return None
