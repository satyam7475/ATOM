"""
ATOM -- Cognitive Intent Patterns.

Voice commands for the cognitive layer:
  - Goal management (create, show, progress, decompose)
  - Prediction queries
  - Mode switching (work, focus, chill, sleep)
  - Behavior/profile queries
  - Second brain queries (remember, recall, preferences)
"""

from __future__ import annotations

import re
from core.intent_engine.base import IntentResult

# ── Goal Intents ──────────────────────────────────────────────────────

_GOAL_CREATE = re.compile(
    r"\b(?:set|create|make|add|new)\s+(?:a\s+)?goal\s+(?:to\s+)?(.+)",
    re.IGNORECASE,
)

_GOAL_SHOW = re.compile(
    r"\b(?:show|list|display|what\s+are)\s+(?:my\s+)?goals?\b",
    re.IGNORECASE,
)

_GOAL_PROGRESS = re.compile(
    r"\b(?:goal\s+progress|how(?:'s| is| am)\s+(?:my\s+)?(?:progress|doing)(?:\s+on\s+(.+))?)\b",
    re.IGNORECASE,
)

_GOAL_DECOMPOSE = re.compile(
    r"\b(?:break\s+down|decompose|plan\s+out|split)\s+(?:this\s+)?(?:goal|that)\b",
    re.IGNORECASE,
)

_GOAL_LOG = re.compile(
    r"\b(?:i\s+)?(?:studied|worked|practiced|spent|logged)\s+(?:on\s+)?(.+?)(?:\s+for\s+)?(\d+)\s*(?:hour|hr|minute|min)",
    re.IGNORECASE,
)

_GOAL_COMPLETE_STEP = re.compile(
    r"\b(?:complete|finish|done\s+with)\s+(?:step|task)\s+(.+)",
    re.IGNORECASE,
)

_GOAL_PAUSE = re.compile(
    r"\b(?:pause|hold|freeze)\s+(?:goal\s+)?(.+)",
    re.IGNORECASE,
)

_GOAL_RESUME = re.compile(
    r"\b(?:resume|continue|unpause)\s+(?:goal\s+)?(.+)",
    re.IGNORECASE,
)

_GOAL_ABANDON = re.compile(
    r"\b(?:abandon|cancel|drop|give\s+up\s+on)\s+(?:goal\s+)?(.+)",
    re.IGNORECASE,
)

# ── Prediction Intents ────────────────────────────────────────────────

_PREDICT = re.compile(
    r"\b(?:what\s+(?:do\s+you\s+think|will)\s+(?:i|I)\s*(?:'ll|will)?\s*(?:do|need)|"
    r"predict\s+(?:my\s+)?(?:next|action)|"
    r"what\s+should\s+(?:i|I)\s+do\s+(?:now|next))\b",
    re.IGNORECASE,
)

# ── Mode Intents ──────────────────────────────────────────────────────

_MODE_SWITCH = re.compile(
    r"\b(?:switch\s+to\s+|activate\s+|enable\s+|go\s+(?:to\s+)?|enter\s+)?"
    r"(work|focus|chill|sleep)\s*mode\b",
    re.IGNORECASE,
)

_MODE_FOCUS_ALT = re.compile(
    r"\b(?:i\s+need\s+to\s+concentrate|"
    r"don'?t\s+disturb\s+me|"
    r"leave\s+me\s+alone|"
    r"no\s+interruptions)\b",
    re.IGNORECASE,
)

_MODE_CHILL_ALT = re.compile(
    r"\b(?:i'?m\s+done\s+working|"
    r"i'?m\s+taking\s+a\s+break|"
    r"relax\s+mode|"
    r"chill\s+out)\b",
    re.IGNORECASE,
)

# ── Behavior / Profile Intents ────────────────────────────────────────

_PRODUCTIVITY = re.compile(
    r"\b(?:when\s+am\s+i\s+most\s+productive|"
    r"my\s+peak\s+hours|"
    r"show\s+(?:my\s+)?(?:behavior|activity|productivity)\s*(?:report)?|"
    r"how'?s?\s+my\s+energy)\b",
    re.IGNORECASE,
)

_SCHEDULING = re.compile(
    r"\b(?:when\s+should\s+i\s+(?:do|schedule|work\s+on)|"
    r"best\s+time\s+(?:to|for))\b",
    re.IGNORECASE,
)

# ── Second Brain Intents ──────────────────────────────────────────────

_REMEMBER = re.compile(
    r"\b(?:remember|note|save|store)\s+(?:that\s+)?(.+)",
    re.IGNORECASE,
)

_RECALL = re.compile(
    r"\b(?:what\s+do\s+you\s+know\s+about|"
    r"recall|remind\s+me\s+about|"
    r"tell\s+me\s+about)\s+(.+)",
    re.IGNORECASE,
)

_PREFERENCES = re.compile(
    r"\b(?:what\s+are\s+my\s+preferences|"
    r"show\s+(?:my\s+)?preferences|"
    r"my\s+settings)\b",
    re.IGNORECASE,
)

_OPTIMIZATION = re.compile(
    r"\b(?:optimize\s+(?:yourself|atom|system)|"
    r"self\s+optimize|"
    r"how\s+can\s+you\s+improve|"
    r"optimization\s+report)\b",
    re.IGNORECASE,
)


def check(text: str) -> IntentResult | None:
    """Check text against all cognitive intent patterns."""

    m = _GOAL_CREATE.search(text)
    if m:
        title = m.group(1).strip().rstrip(".")
        return IntentResult(
            "goal_create", action="goal_create",
            action_args={"title": title},
        )

    if _GOAL_SHOW.search(text):
        return IntentResult("goal_show", action="goal_show")

    m = _GOAL_PROGRESS.search(text)
    if m:
        target = (m.group(1) or "").strip()
        return IntentResult(
            "goal_progress", action="goal_progress",
            action_args={"target": target},
        )

    if _GOAL_DECOMPOSE.search(text):
        return IntentResult("goal_decompose", action="goal_decompose")

    m = _GOAL_LOG.search(text)
    if m:
        topic = m.group(1).strip()
        amount = int(m.group(2))
        unit = "hour" if "hour" in text.lower() or "hr" in text.lower() else "minute"
        minutes = amount * 60 if unit == "hour" else amount
        return IntentResult(
            "goal_log_progress", action="goal_log_progress",
            action_args={"topic": topic, "minutes": minutes},
        )

    m = _GOAL_COMPLETE_STEP.search(text)
    if m:
        step_name = m.group(1).strip()
        return IntentResult(
            "goal_complete_step", action="goal_complete_step",
            action_args={"step_name": step_name},
        )

    m = _GOAL_PAUSE.search(text)
    if m:
        return IntentResult(
            "goal_pause", action="goal_pause",
            action_args={"target": m.group(1).strip()},
        )

    m = _GOAL_RESUME.search(text)
    if m:
        return IntentResult(
            "goal_resume", action="goal_resume",
            action_args={"target": m.group(1).strip()},
        )

    m = _GOAL_ABANDON.search(text)
    if m:
        return IntentResult(
            "goal_abandon", action="goal_abandon",
            action_args={"target": m.group(1).strip()},
        )

    if _PREDICT.search(text):
        return IntentResult("prediction", action="prediction")

    m = _MODE_SWITCH.search(text)
    if m:
        mode = m.group(1).lower()
        return IntentResult(
            "mode_switch", action="mode_switch",
            action_args={"mode": mode},
        )

    if _MODE_FOCUS_ALT.search(text):
        return IntentResult(
            "mode_switch", action="mode_switch",
            action_args={"mode": "focus"},
        )

    if _MODE_CHILL_ALT.search(text):
        return IntentResult(
            "mode_switch", action="mode_switch",
            action_args={"mode": "chill"},
        )

    if _PRODUCTIVITY.search(text):
        return IntentResult("behavior_report", action="cognitive_behavior_report")

    if _SCHEDULING.search(text):
        return IntentResult("scheduling_advice", action="scheduling_advice")

    m = _REMEMBER.search(text)
    if m:
        fact = m.group(1).strip()
        return IntentResult(
            "brain_remember", action="brain_remember",
            action_args={"fact": fact},
        )

    m = _RECALL.search(text)
    if m:
        query = m.group(1).strip()
        return IntentResult(
            "brain_recall", action="brain_recall",
            action_args={"query": query},
        )

    if _PREFERENCES.search(text):
        return IntentResult("brain_preferences", action="brain_preferences")

    if _OPTIMIZATION.search(text):
        return IntentResult("self_optimize", action="self_optimize")

    return None
