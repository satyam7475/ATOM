"""
Voice / text commands to switch assistant mode and local brain profile.
"""

from __future__ import annotations

import re

from .base import IntentResult

# Brain profile (speed vs depth) — maps to BrainModeManager
_ATOM_PROFILE = re.compile(
    r"\b(atom\s+mode|fast\s+brain|speed\s+mode|quick\s+brain)\b", re.I,
)
_BRAIN_PROFILE = re.compile(
    r"\b(brain\s+mode|smart\s+brain|deep\s+mode|full\s+brain)\b", re.I,
)
_BALANCED_PROFILE = re.compile(
    r"\b(balanced\s+mode|normal\s+brain\s+mode)\b", re.I,
)

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

    if _ATOM_PROFILE.search(t):
        return IntentResult(
            intent="set_brain_profile",
            action="set_brain_profile",
            action_args={"profile": "atom"},
        )
    if _BRAIN_PROFILE.search(t):
        return IntentResult(
            intent="set_brain_profile",
            action="set_brain_profile",
            action_args={"profile": "brain"},
        )
    if _BALANCED_PROFILE.search(t):
        return IntentResult(
            intent="set_brain_profile",
            action="set_brain_profile",
            action_args={"profile": "balanced"},
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
