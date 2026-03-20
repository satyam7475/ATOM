"""
ATOM Intent Engine -- Meta intents (greeting, thanks, status, exit, confirm, deny, usage, silent_mode).
"""

from __future__ import annotations

import re

from core import personality
from .base import IntentResult

_EXIT = re.compile(
    r"^(shutdown|quit|exit|stop atom|shut\s*down|"
    r"close atom|turn off|power off|"
    r"good\s*bye\s+shut\s*down|bye\s+shut\s*down|"
    r"good\s*bye\s+exit|bye\s+exit|"
    r"good\s*bye\s+quit|bye\s+quit|"
    r"good\s*bye\s+atom\s+shut\s*down|"
    r"alvida|band\s+karo\s+atom)[\s!.]*$", re.I)

_SILENT_MODE = re.compile(
    r"^(bye|good\s*bye|goodbye|good\s*night|"
    r"go\s+(to\s+)?sleep|sleep\s+mode|go\s+silent|silent\s+mode|"
    r"be\s+quiet|shut\s+up|quiet|shh+|hush|rest\s+now|take\s+a\s+break|"
    r"chup\s+karo|chup\s+ho\s*ja|band\s+karo|band\s+ho\s*ja|"
    r"band\s+kar\s+do|ruk\s+ja|bas\s+kar|bas\s+karo|"
    r"chalo\s+bhago(\s+ab)?|chalo\s+jao|ja\s+ab|hat\s+ja|"
    r"so\s+ja|so\s+jao|mute\s+atom|stop\s+listening|"
    r"enough|that's\s+enough|that\s+is\s+enough)[\s!.]*$", re.I)

_GREETING = re.compile(
    r"^(hi|hello|hey|namaste|howdy|"
    r"good\s+morning|good\s+evening|good\s+afternoon|good\s+night|"
    r"what's\s+up|sup|yo|hola)"
    r"(\s+(atom|buddy|bro|boss|sir|madam|ma'am|dear|dude|friend|mate|jee))*[\s!.]*$", re.I)

_THANKS = re.compile(
    r"^(thanks?|thank\s*you|thx|ty|shukriya|dhanyavaad|nice|good\s+job|well\s+done|great)"
    r"(\s+atom|\s+buddy|\s+boss)?[\s!.]*$", re.I)

_STATUS = re.compile(
    r"(are\s+you\s+(there|alive|ready|listening|awake)|"
    r"you\s+there|atom\s+status|status\s+check|"
    r"can\s+you\s+hear\s+me|hello.*(there|atom))", re.I)

_USAGE = re.compile(
    r"(how\s+much\s+(llm|brain)|local\s+brain\s+usage|llm\s+usage|"
    r"percentage\s+tasks.*(llm|brain))", re.I)

_CONFIRM = re.compile(
    r"^(yes|yeah|yep|yup|sure|okay|ok|go|go\s+ahead|yes\s+play|play|play\s+it|"
    r"haan|ha|han|theek\s+hai|chalu\s+karo|kar\s+do|confirm|do\s+it|proceed|"
    r"go\s+for\s+it|absolutely|definitely)[\s!.]*$", re.I)

_DENY = re.compile(
    r"^(no|nah|nahi|mat\s+karo|cancel|stop|don't|dont)[\s!.]*$", re.I)


def check(text: str) -> IntentResult | None:
    if _SILENT_MODE.search(text):
        return IntentResult("go_silent", response=personality.silent_response())
    if _EXIT.search(text):
        return IntentResult("exit", response=personality.exit_response())
    if _CONFIRM.search(text):
        return IntentResult("confirm")
    if _DENY.search(text):
        return IntentResult("deny", response="Okay boss, cancelled.")
    if _GREETING.search(text):
        return IntentResult("greeting", response=personality.greeting_response())
    if _THANKS.search(text):
        return IntentResult("thanks", response=personality.thanks_response())
    if _STATUS.search(text):
        return IntentResult("status", response=personality.status_response())
    if _USAGE.search(text):
        return IntentResult("status", response="Here's your usage status, boss.")
    return None


def quick_match(text: str) -> str | None:
    """Fast check for meta intents used by STT early-exit."""
    if _SILENT_MODE.search(text):
        return "go_silent"
    if _EXIT.search(text):
        return "exit"
    if _CONFIRM.search(text):
        return "confirm"
    if _DENY.search(text):
        return "deny"
    if _GREETING.search(text):
        return "greeting"
    if _THANKS.search(text):
        return "thanks"
    if _STATUS.search(text):
        return "status"
    return None
