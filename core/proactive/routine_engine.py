"""
ATOM -- User-defined routine engine (Sprint D4).

Routines are named presets the user can trigger conversationally:

    "enter deep work mode"   → mute, lock brain to optimal, quieter TTS
    "bedtime mode"           → volume 5, silent assistant, dim the world
    "focus mode"             → alias for deep_work
    "exit deep work"         → roll back to defaults (or user-defined exit state)

Each routine is a flat struct of *enter* and *exit* phases. Each phase
is a list of **small, well-bounded steps** (``volume``, ``mute``,
``brain_profile``, ``assistant_mode``). Steps are dispatched through an
injected callable so the engine is loosely coupled to the router (great
for tests).

Routines are loaded from (in priority order):

    1. ``config["routines"]`` if present in the caller's config dict
    2. ``config/routines.json`` if it exists
    3. Built-in defaults (deep_work, bedtime)

Owner: Satyam
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Tuple

logger = logging.getLogger("atom.proactive.routine_engine")


StepDispatcher = Callable[[str, dict], str]


_DEFAULT_ROUTINES: list[dict[str, Any]] = [
    {
        "name": "deep_work",
        "aliases": ["deep work", "deep work mode", "focus mode", "focus"],
        "enter_say": "Entering deep work mode, Boss. I'll keep things quiet.",
        "enter_steps": [
            {"kind": "volume", "percent": 15},
            {"kind": "assistant_mode", "mode": "silent_mode"},
            {"kind": "brain_profile", "profile": "optimal"},
        ],
        "exit_say": "Exiting deep work. Welcome back.",
        "exit_steps": [
            {"kind": "volume", "percent": 50},
            {"kind": "assistant_mode", "mode": "standard"},
        ],
    },
    {
        "name": "bedtime",
        "aliases": ["bedtime", "bedtime mode", "good night mode", "sleep mode", "night mode"],
        "enter_say": "Goodnight, Boss. Dimming things down.",
        "enter_steps": [
            {"kind": "volume", "percent": 5},
            {"kind": "assistant_mode", "mode": "silent_mode"},
        ],
        "exit_say": "Good morning again, Boss.",
        "exit_steps": [
            {"kind": "volume", "percent": 40},
            {"kind": "assistant_mode", "mode": "standard"},
        ],
    },
    {
        "name": "meeting",
        "aliases": ["meeting mode", "meeting", "call mode", "in a meeting"],
        "enter_say": "Meeting mode on. I'll stay out of your way.",
        "enter_steps": [
            {"kind": "volume", "percent": 40},
            {"kind": "assistant_mode", "mode": "silent_mode"},
        ],
        "exit_say": "Meeting done. Back to normal.",
        "exit_steps": [
            {"kind": "assistant_mode", "mode": "standard"},
        ],
    },
]


_SUPPORTED_KINDS: frozenset[str] = frozenset(
    {"volume", "mute", "unmute", "brain_profile", "assistant_mode"}
)


@dataclass
class Routine:
    name: str
    aliases: list[str] = field(default_factory=list)
    enter_say: str = ""
    enter_steps: list[dict[str, Any]] = field(default_factory=list)
    exit_say: str = ""
    exit_steps: list[dict[str, Any]] = field(default_factory=list)

    @property
    def display(self) -> str:
        pretty = self.name.replace("_", " ")
        return pretty


def _normalize_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _routine_from_dict(raw: dict[str, Any]) -> Routine | None:
    name = str(raw.get("name") or "").strip().lower().replace(" ", "_")
    if not name:
        return None
    aliases = []
    for a in raw.get("aliases", []) or []:
        if isinstance(a, str) and a.strip():
            aliases.append(_normalize_phrase(a))
    enter_steps = [s for s in raw.get("enter_steps", []) or [] if isinstance(s, dict)]
    exit_steps = [s for s in raw.get("exit_steps", []) or [] if isinstance(s, dict)]
    return Routine(
        name=name,
        aliases=aliases,
        enter_say=str(raw.get("enter_say") or "").strip(),
        enter_steps=enter_steps,
        exit_say=str(raw.get("exit_say") or "").strip(),
        exit_steps=exit_steps,
    )


_ENTER_RE = re.compile(
    r"\b(?:enter|start|activate|begin|turn\s+on|engage|go\s+into|kick\s+off)\b",
    re.IGNORECASE,
)
_EXIT_RE = re.compile(
    r"\b(?:exit|leave|stop|end|disable|turn\s+off|deactivate|cancel|get\s+out\s+of)\b",
    re.IGNORECASE,
)


class RoutineEngine:
    """Loads routines and dispatches enter/exit phases."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        dispatcher: StepDispatcher | None = None,
    ) -> None:
        self._config = config or {}
        self._dispatcher: StepDispatcher | None = dispatcher
        self._routines: dict[str, Routine] = {}
        self._alias_index: dict[str, str] = {}
        self._active: str | None = None
        self._load_routines()

    # ── Loading ───────────────────────────────────────────────────

    def _load_routines(self) -> None:
        raw_list = self._raw_routine_source()
        for raw in raw_list:
            routine = _routine_from_dict(raw)
            if routine is None:
                continue
            self._routines[routine.name] = routine
            for alias in {*routine.aliases, routine.display, routine.name}:
                if alias:
                    self._alias_index[_normalize_phrase(alias)] = routine.name
        if self._routines:
            logger.info("Loaded %d routines: %s", len(self._routines), ", ".join(sorted(self._routines)))

    def _raw_routine_source(self) -> list[dict[str, Any]]:
        inline = self._config.get("routines")
        if isinstance(inline, list) and inline:
            return list(inline)
        path = Path(self._config.get("routine_path") or "config/routines.json")
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    raw_list = raw.get("routines", [])
                else:
                    raw_list = raw
                if isinstance(raw_list, list):
                    return list(raw_list)
            except Exception:
                logger.info("Failed to load %s; falling back to defaults", path, exc_info=True)
        return list(_DEFAULT_ROUTINES)

    # ── Public API ────────────────────────────────────────────────

    @property
    def active(self) -> str | None:
        return self._active

    def list_routines(self) -> list[Routine]:
        return list(self._routines.values())

    def set_dispatcher(self, dispatcher: StepDispatcher | None) -> None:
        self._dispatcher = dispatcher

    def match(self, text: str) -> tuple[str, str] | None:
        """Return ``(routine_name, phase)`` for an utterance, or None."""
        if not text:
            return None
        lowered = _normalize_phrase(text)
        phase: str | None = None
        if _ENTER_RE.search(lowered):
            phase = "enter"
        elif _EXIT_RE.search(lowered):
            phase = "exit"

        hit_name = self._find_routine_in(lowered)
        if hit_name is not None:
            return (hit_name, phase or "enter")

        if phase == "exit" and self._active is not None:
            if re.search(r"\b(?:routine|mode)\b", lowered):
                return (self._active, "exit")

        return None

    def execute(self, name: str, phase: str = "enter") -> str:
        routine = self._routines.get(name)
        if routine is None:
            return f"I don't know a routine called {name.replace('_', ' ')}, Boss."
        if phase not in ("enter", "exit"):
            phase = "enter"

        steps = routine.enter_steps if phase == "enter" else routine.exit_steps
        say = routine.enter_say if phase == "enter" else routine.exit_say

        executed = 0
        failures: list[str] = []
        for step in steps:
            kind = str(step.get("kind") or "").strip().lower()
            if kind not in _SUPPORTED_KINDS:
                failures.append(f"unknown step '{kind}'")
                continue
            args = {k: v for k, v in step.items() if k != "kind"}
            if self._dispatcher is None:
                executed += 1
                continue
            try:
                msg = self._dispatcher(kind, args)
                executed += 1
                if msg and any(tok in str(msg).lower() for tok in ("sorry", "not active", "blocked", "denied")):
                    failures.append(str(msg))
            except Exception as exc:
                logger.info("Routine step '%s' failed: %s", kind, exc, exc_info=True)
                failures.append(f"{kind}: {exc.__class__.__name__}")

        if phase == "enter":
            self._active = routine.name
        else:
            if self._active == routine.name:
                self._active = None

        default_line = (
            f"{routine.display} mode on." if phase == "enter"
            else f"{routine.display} mode off."
        )
        spoken = say or default_line
        if failures and self._dispatcher is not None:
            spoken += f" (Some steps had issues: {failures[0]})"
        return spoken

    # ── Matching helpers ──────────────────────────────────────────

    def _find_routine_in(self, text: str) -> str | None:
        for alias, name in sorted(
            self._alias_index.items(), key=lambda kv: len(kv[0]), reverse=True,
        ):
            if alias and alias in text:
                return name
        return None


__all__ = ["Routine", "RoutineEngine"]
