"""
Voice / text commands to switch assistant mode and local brain profile.
"""

from __future__ import annotations

import re

from .base import IntentResult

# Brain profile (speed vs depth) — maps to BrainModeManager
_ATOM_PROFILE = re.compile(
    r"\b(atom\s+mode|fast\s+brain|speed\s+mode|quick\s+brain|optimal\s+mode|stable\s+buddy\s+mode)\b",
    re.I,
)
_BRAIN_PROFILE = re.compile(
    r"\b(brain\s+mode|smart\s+brain|deep\s+mode|full\s+brain|full\s+performance(?:\s+mode)?|full\s+feature(?:\s+mode)?)\b",
    re.I,
)
_BALANCED_PROFILE = re.compile(
    r"\b(balanced\s+mode|normal\s+brain\s+mode)\b", re.I,
)
_MODE_SWITCH_VERB = re.compile(
    r"\b(switch|set|change|use|enable|go(?:\s+to|\s+into)?|turn\s+on)\b",
    re.I,
)
_PROFILE_EXACT_MAP = {
    "atom mode": "optimal",
    "fast brain": "optimal",
    "speed mode": "optimal",
    "quick brain": "optimal",
    "optimal mode": "optimal",
    "stable buddy mode": "optimal",
    "balanced mode": "optimal",
    "normal brain mode": "optimal",
    "brain mode": "full_performance",
    "smart brain": "full_performance",
    "deep mode": "full_performance",
    "full brain": "full_performance",
    "full performance": "full_performance",
    "full performance mode": "full_performance",
    "full feature mode": "full_performance",
}

# Assistant mode (LLM on fallback or not)
_COMMAND_ONLY = re.compile(
    r"\b(commands?\s+only\s+mode|command\s+only|no\s+chat\s+mode|"
    r"disable\s+(?:the\s+)?brain|turn\s+off\s+(?:the\s+)?chat\s+brain)\b", re.I,
)
_HYBRID = re.compile(
    r"\b(hybrid\s+mode|default\s+assistant\s+mode|enable\s+hybrid|"
    r"normal\s+assistant\s+mode)\b", re.I,
)
_CONVERSATIONAL = re.compile(
    r"\b(conversational\s+mode|conversation\s+mode|chat\s+mode|"
    r"enable\s+(?:the\s+)?chat\s+brain|full\s+assistant)\b", re.I,
)


def check(text: str) -> IntentResult | None:
    t = text.strip()
    if not t:
        return None
    norm = re.sub(r"\s+", " ", t.lower()).strip()
    is_switch_command = bool(_MODE_SWITCH_VERB.search(t))

    if norm in _PROFILE_EXACT_MAP:
        return IntentResult(
            intent="set_brain_profile",
            action="set_brain_profile",
            action_args={"profile": _PROFILE_EXACT_MAP[norm]},
        )
    if is_switch_command and _ATOM_PROFILE.search(t):
        return IntentResult(
            intent="set_brain_profile",
            action="set_brain_profile",
            action_args={"profile": "optimal"},
        )
    if is_switch_command and _BRAIN_PROFILE.search(t):
        return IntentResult(
            intent="set_brain_profile",
            action="set_brain_profile",
            action_args={"profile": "full_performance"},
        )
    if is_switch_command and _BALANCED_PROFILE.search(t):
        return IntentResult(
            intent="set_brain_profile",
            action="set_brain_profile",
            action_args={"profile": "optimal"},
        )
    if _COMMAND_ONLY.search(t):
        return IntentResult(
            intent="set_assistant_mode",
            action="set_assistant_mode",
            action_args={"mode": "command_only"},
        )
    if _HYBRID.search(t):
        return IntentResult(
            intent="set_assistant_mode",
            action="set_assistant_mode",
            action_args={"mode": "hybrid"},
        )
    if _CONVERSATIONAL.search(t):
        return IntentResult(
            intent="set_assistant_mode",
            action="set_assistant_mode",
            action_args={"mode": "conversational"},
        )
    return None
