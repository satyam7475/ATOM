"""
ATOM -- Routine intents (Sprint D4).

Routes utterances like "enter deep work mode", "start bedtime mode",
"exit focus mode", "disable deep work" to the ``run_routine`` action.

The heavy lifting lives in :mod:`core.proactive.routine_engine`. This
module only does regex matching; the engine does the dispatch. A
:class:`RoutineEngine` instance is injected at boot via
:func:`set_routine_engine` so we can support user-defined routines
without rebuilding the regex table.
"""

from __future__ import annotations

import re
from typing import Any

from .base import IntentResult


_engine: Any | None = None


def set_routine_engine(engine: Any | None) -> None:
    """Inject the ``RoutineEngine`` instance (called once at boot)."""
    global _engine
    _engine = engine


_CUE = re.compile(
    r"\b(?:enter|start|activate|begin|turn\s+on|engage|go\s+into|kick\s+off|"
    r"exit|leave|stop|end|disable|turn\s+off|deactivate|cancel|get\s+out\s+of)\b"
    r"|\bmode\b",
    re.IGNORECASE,
)


def check(text: str) -> IntentResult | None:
    if not text or _engine is None:
        return None
    if not _CUE.search(text):
        return None
    hit = None
    try:
        hit = _engine.match(text)
    except Exception:
        return None
    if hit is None:
        return None
    name, phase = hit
    return IntentResult(
        "run_routine",
        action="run_routine",
        action_args={"name": name, "phase": phase},
    )


__all__ = ["check", "set_routine_engine"]
