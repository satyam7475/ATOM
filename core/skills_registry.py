"""
ATOM -- Named skills with optional multi-step chaining.

Loads `config/skills.json`. Case-insensitive trigger matching; first match
wins. When a skill has a `chain` list, those extra utterances are returned
for the Router to execute in sequence after the primary expansion.
"""

from __future__ import annotations

import json
import logging
import re
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
        # Index: trigger phrase -> entry index. Built once at load so
        # ``try_expand_full`` can resolve a regex match in O(1) instead
        # of re-walking ``_entries``.
        self._trigger_index: dict[str, int] = {}
        # Single compiled regex covering every trigger phrase with named
        # groups. Replaces the per-call O(n) substring loop that was
        # spending ~150 ms on a 30-trigger registry under cold cache
        # (atom_log.txt L597-599: skill expansion drove the second
        # intent_classify pass over its 250 ms budget).
        self._compiled_re: re.Pattern[str] | None = None
        if self._enabled:
            self._load()

    def _load(self) -> None:
        self._entries = []
        self._trigger_index = {}
        self._compiled_re = None
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
        self._build_trigger_index()
        logger.info(
            "Skills registry: %d trigger(s), %d with chains from %s "
            "(regex cached=%s)",
            len(self._entries),
            sum(1 for _, _, c, _ in self._entries if c),
            self._path,
            self._compiled_re is not None,
        )

    def _build_trigger_index(self) -> None:
        """Compile the trigger union regex and the trigger->entry index.

        Triggers are sorted longest-first so the regex engine prefers
        the most specific trigger when multiple substrings could match
        ("system status report" before "status report"). Word-boundary
        anchored so a trigger like "snake" in a skill doesn't match
        "snakeskin" by accident.
        """
        if not self._entries:
            self._trigger_index = {}
            self._compiled_re = None
            return

        seen: set[str] = set()
        for idx, (trigger, _exp, _chain, _sid) in enumerate(self._entries):
            if trigger in seen:
                continue
            seen.add(trigger)
            self._trigger_index.setdefault(trigger, idx)

        ordered = sorted(seen, key=lambda s: (-len(s), s))
        try:
            pattern = (
                r"(?:^|\W)(?P<m>"
                + "|".join(re.escape(t) for t in ordered)
                + r")(?=\W|$)"
            )
            self._compiled_re = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:  # pragma: no cover - guard against malformed input
            logger.warning(
                "SkillsRegistry: failed to compile combined trigger regex (%s); "
                "falling back to per-call substring scan",
                exc,
            )
            self._compiled_re = None

    def try_expand(self, clean_text: str) -> tuple[str, str] | None:
        """Backward-compatible: return (expanded, skill_id) or None."""
        match = self.try_expand_full(clean_text)
        if match is None:
            return None
        return match.primary, match.skill_id

    def try_expand_full(self, clean_text: str) -> SkillMatch | None:
        """Return full SkillMatch (primary + chain) or None.

        Uses the compiled-at-boot trigger regex so the per-call cost is
        a single regex search regardless of how many triggers the user
        has configured. Falls back to the legacy substring loop only if
        regex compilation failed at load time (malformed trigger).
        """
        if not self._enabled or not self._entries or not clean_text:
            return None
        low = clean_text.strip().lower()
        if not low:
            return None

        if self._compiled_re is not None:
            m = self._compiled_re.search(low)
            if m is not None:
                trigger = m.group("m")
                idx = self._trigger_index.get(trigger)
                if idx is not None:
                    _t, expand, chain, sid = self._entries[idx]
                    if low == expand.lower():
                        return None
                    return SkillMatch(expand, chain, sid)
            return None

        # Legacy substring fallback — only reached when regex compile
        # failed (extremely unusual, kept so a bad config doesn't break
        # skills entirely).
        for trigger, expand, chain, sid in self._entries:
            if low == trigger or trigger in low:
                if low == expand.lower():
                    return None
                return SkillMatch(expand, chain, sid)
        return None

    def expansion_targets(self) -> list[str]:
        """Return every distinct ``expand_to`` plus chain step.

        Used by the cold-start optimizer to pre-cache intent
        classification for every skill destination, eliminating the
        ~150 ms second-pass classify cost the live log exposed
        (atom_log.txt L597-599).
        """
        seen: set[str] = set()
        out: list[str] = []
        for _t, expand, chain, _sid in self._entries:
            for candidate in (expand, *chain):
                norm = (candidate or "").strip()
                if not norm:
                    continue
                key = norm.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(norm)
        return out
