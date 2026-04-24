"""
ATOM Intent Engine -- Meta intents (greeting, thanks, status, exit, confirm, deny, usage, silent_mode).
"""

from __future__ import annotations

import re

from core import adaptive_personality as personality
from .base import IntentResult

# Casual "bye" / "goodbye" are go_silent (see _CASUAL_BYE), not hard exit — otherwise
# STT noise matching "bye" shuts down the app seconds after launch.
_EXIT = re.compile(
    r"^(shutdown|quit|exit|stop atom|shut\s*down|"
    r"close atom|turn off|power off|"
    r"shutdown\s+atom|shut\s*down\s+atom|"
    r"alvida|band\s+karo\s+atom)[\s!.]*$", re.I)

_CASUAL_BYE = re.compile(
    r"^(bye|good\s*bye|goodbye)(\s+(atom|boss|buddy|bro|sir|madam))?[\s!.]*$", re.I)

_SILENT_MODE = re.compile(
    r"^(go\s+(to\s+)?sleep|sleep\s+mode|go\s+silent|silent\s+mode|"
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
    r"(\s+(atom|adam|buddy|bro|boss|sir|madam|ma'am|dear|dude|friend|mate|jee))*"
    r"(\s+(how\s+are\s+you|kaise\s+ho|kya\s+haal|what's\s+up))?[\s!?.]*$", re.I)

_THANKS = re.compile(
    r"^(thanks?|thank\s*you|thx|ty|shukriya|dhanyavaad|nice|good\s+job|well\s+done|great)"
    r"(\s+atom|\s+buddy|\s+boss)?[\s!.]*$", re.I)

_STATUS = re.compile(
    r"(are\s+you\s+(there|alive|ready|listening|awake|online|up)|"
    r"you\s+(there|alive|ready|listening|awake|online|up)|"
    r"atom\s+status|status\s+check|"
    # Health / system-status intent — casual check of "is everything ok"
    # AND verbose "what is the health status of atom" style queries.
    # The previous regex missed all of these and routed them to the LLM
    # which then leaked chain-of-thought answers.
    r"(?:what(?:'|\u2019)?s?|tell\s+me|give\s+me|show\s+me|check)\s+"
    r"(?:the\s+|your\s+|atom(?:'s)?\s+)?"
    r"(?:health|system|status|health\s+status|system\s+status|overall\s+status)\b|"
    r"health\s+(?:status|check|report)\b|"
    r"system\s+(?:status|health|check|report)\b|"
    r"(?:everything|all|it|we)\s+(?:ok|okay|good|fine|alright|running)|"
    r"how(?:'|\u2019)?s?\s+(?:it|everything|atom|the\s+system)\s+"
    r"(?:going|running|doing|looking)|"
    r"can\s+you\s+hear\s+me|^hello\s+there[\s!?.]*$)", re.I)

_USAGE = re.compile(
    r"(how\s+much\s+(llm|brain)|local\s+brain\s+usage|llm\s+usage|"
    r"percentage\s+tasks.*(llm|brain))", re.I)

_CONFIRM = re.compile(
    r"^(yes|yeah|yep|yup|sure|okay|ok|go|go\s+ahead|yes\s+play|play|play\s+it|"
    r"haan|ha|han|theek\s+hai|chalu\s+karo|kar\s+do|confirm|do\s+it|proceed|"
    r"go\s+for\s+it|absolutely|definitely|"
    r"yes\s+confirm|yes\s+go|yes\s+go\s+ahead|yes\s+do\s+it|confirm\s+it|"
    r"yes\s+proceed|yes\s+please|sure\s+go\s+ahead|ok\s+go|okay\s+go|"
    r"haan\s+chalu\s+karo|haan\s+kar\s+do|"
    r"confirm\s+yes|confirm\s+confirm\s+yes|confirm\s+please|"
    r"confirm\s+go|confirm\s+go\s+ahead|"
    r"yes\s+yes|yeah\s+yeah|ok\s+ok|sure\s+sure|"
    r"haan\s+haan)[\s!.]*$", re.I)

_DENY = re.compile(
    r"^(no|nah|nahi|nope|mat\s+karo|cancel|stop|don't|dont|"
    r"no\s+cancel|nahi\s+mat\s+karo|no\s+don't|no\s+stop|"
    r"deny|reject|abort|abort\s+it|cancel\s+it|"
    r"no\s+no|nahi\s+nahi|nope\s+nope)[\s!.]*$", re.I)

# Confirm-dominant matcher — handles STT garbage like "Confirm confirm yes"
# / "yes please yes ok" by accepting any short utterance whose tokens are
# ALL in the confirmation vocabulary. Anchored to <= 5 tokens so it can't
# false-positive on real queries that happen to start with "yes".
_CONFIRM_VOCAB = frozenset({
    "yes", "yeah", "yep", "yup", "sure", "okay", "ok", "go", "ahead",
    "play", "haan", "ha", "han", "confirm", "do", "it", "proceed", "for",
    "absolutely", "definitely", "please",
})
_DENY_VOCAB = frozenset({
    "no", "nah", "nahi", "nope", "cancel", "stop", "don't", "dont",
    "deny", "reject", "abort", "mat", "karo",
})


def _is_confirm_dominant(text: str) -> bool:
    """True when the utterance is short AND every word is a confirmation
    token. Catches STT-garbled inputs like 'Confirm confirm yes' or
    'yes ok sure' that the strict regex misses."""
    tokens = re.findall(r"[a-zA-Z']+", (text or "").lower())
    if not tokens or len(tokens) > 5:
        return False
    return all(tok in _CONFIRM_VOCAB for tok in tokens)


def _is_deny_dominant(text: str) -> bool:
    tokens = re.findall(r"[a-zA-Z']+", (text or "").lower())
    if not tokens or len(tokens) > 5:
        return False
    return all(tok in _DENY_VOCAB for tok in tokens)


def check(text: str) -> IntentResult | None:
    if _SILENT_MODE.search(text):
        return IntentResult("go_silent", response=personality.silent_response())
    if _CASUAL_BYE.search(text):
        return IntentResult("go_silent", response=personality.silent_response())
    if _EXIT.search(text):
        return IntentResult("exit", response=personality.exit_response())
    if _CONFIRM.search(text) or _is_confirm_dominant(text):
        return IntentResult("confirm")
    if _DENY.search(text) or _is_deny_dominant(text):
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
    if _CASUAL_BYE.search(text):
        return "go_silent"
    if _EXIT.search(text):
        return "exit"
    if _CONFIRM.search(text) or _is_confirm_dominant(text):
        return "confirm"
    if _DENY.search(text) or _is_deny_dominant(text):
        return "deny"
    if _GREETING.search(text):
        return "greeting"
    if _THANKS.search(text):
        return "thanks"
    if _STATUS.search(text):
        return "status"
    return None
