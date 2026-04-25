"""ATOM -- Offer Synthesizer (Sprint J: Jarvis Offer Protocol).

Reads a user query (and optionally the assistant's reply) and decides
whether ATOM should follow up with a Jarvis-style "Want me to do that
for you, Boss?" offer that maps to a real, dispatchable action.

The synthesizer is intentionally a *deterministic, regex-driven*
mapping rather than an LLM call -- the whole point is to avoid the
3-second LLM round-trip that ate the previous Jarvis attempts. We
match the most common how-to / what-is / explain / tell-me-about
patterns, extract any obvious entity (app name, city, topic), and
build a ready-to-execute ``(action, args, offer_text)`` triple.

Coverage rationale (matches the live log evidence in atomLogs.txt
and the user's #1 complaints):

  * App open / close / launch       -> open_app / close_app
  * Volume / brightness control      -> set_volume / mute / set_brightness
  * Music playback                   -> music_play / music_pause
  * Weather (with city extraction)   -> weather
  * Battery / CPU / RAM enquiry      -> resource_report
  * Wifi / network                   -> wifi_status
  * Reminders / agenda               -> set_reminder / whats_on_my_plate
  * Screenshot / lock / sleep        -> screenshot / lock_screen
  * Vision / screen description      -> screen_describe / vision_describe
  * Search / explainer ("tell me
    about X", "what is X")           -> research_topic
  * News                             -> news_headlines
  * Daily briefing / day planning    -> daily_briefing

Anything outside this list returns ``None`` -- the assistant's reply
ships without an offer, exactly as today. This keeps false-positive
offers (like the watchdog's "Want me to recover?" on a casual greeting)
from bleeding into normal conversation.

Public API:

    synthesize_offer(query: str, response: str = "") -> OfferProposal | None

The router stashes the proposal in ``OfferRegistry`` and appends the
``offer_text`` to the assistant's reply just before TTS.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("atom.cognitive.offer")


# ── Public types ───────────────────────────────────────────────────


@dataclass
class OfferProposal:
    """A staged Jarvis offer ready for the registry + reply append."""

    action: str
    args: dict[str, Any] = field(default_factory=dict)
    offer_text: str = ""
    category: str = ""

    def __post_init__(self) -> None:
        if not self.action:
            raise ValueError("OfferProposal.action is required")
        if not self.offer_text:
            raise ValueError("OfferProposal.offer_text is required")


# ── Trigger detection ──────────────────────────────────────────────

# A query is "explainer-shaped" if it starts with how / how to / what /
# what's / what is / tell me / explain / can you tell. Action verbs
# ("play", "open", "set", "close") are deliberately excluded -- those
# already route to a real intent and do NOT need a follow-up offer.
_EXPLAINER_PREFIX = re.compile(
    r"^\s*"
    r"(?:hey\s+atom[,\s]+|atom[,\s]+|boss[,\s]+)?"
    r"(?:please\s+)?"
    r"(?:"
    r"  how\s+(?:do\s+i|to|can\s+i|would\s+i|can\s+you|"
    r"          would\s+you|much|many|long|often|"
    r"          (?:'s|\s+is)\s+(?:my|the))|"
    r"  what(?:\s+is|'s|\s+are|s|\s+do)?\s+|"
    r"  tell\s+me\s+(?:about|how|what)|"
    r"  explain\s+(?:me\s+)?(?:how|about|what)?|"
    r"  do\s+you\s+know\s+(?:how|what)|"
    r"  can\s+you\s+(?:tell|explain|see)|"
    r"  describe\s+(?:how|what)|"
    r"  am\s+i\s+(?:connected|on|using)"
    r")",
    re.X | re.I,
)


def is_explainer_query(text: str) -> bool:
    """Cheap predicate: does this look like a 'how / what / explain'
    question that an offer might fit?

    The router calls this before invoking the full synthesizer so we
    skip the regex matrix entirely on bare commands like "play music"
    or "open chrome".
    """
    if not text:
        return False
    return bool(_EXPLAINER_PREFIX.search(text))


# ── Pattern library ────────────────────────────────────────────────

# Known app aliases -> canonical app name the open_app dispatcher
# already understands. Kept tiny here -- the full mapping lives in
# core/intent_engine/app_intents.APP_MAP. We reuse the most common
# six because they cover ~80% of the live log's "how do I open X"
# turns and we don't want to import the full map (cold-import cost).
_APP_ALIASES: dict[str, str] = {
    "chrome": "Google Chrome",
    "google chrome": "Google Chrome",
    "safari": "Safari",
    "spotify": "Spotify",
    "vscode": "Visual Studio Code",
    "vs code": "Visual Studio Code",
    "code": "Visual Studio Code",
    "cursor": "Cursor",
    "terminal": "Terminal",
    "iterm": "iTerm",
    "iterm2": "iTerm",
    "slack": "Slack",
    "notes": "Notes",
    "calendar": "Calendar",
    "messages": "Messages",
    "whatsapp": "WhatsApp",
    "discord": "Discord",
    "zoom": "zoom.us",
    "finder": "Finder",
    "music": "Music",
    "mail": "Mail",
}


def _normalise_app(raw: str) -> str:
    """Map a captured app name to its canonical form, stripping
    decorators like 'app', 'application', leading articles."""
    if not raw:
        return ""
    cleaned = re.sub(r"\b(app|application|browser)\b", "", raw, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,?!")
    return _APP_ALIASES.get(cleaned.lower(), cleaned.title()) if cleaned else ""


# Regex -> handler. Each handler receives ``(match, query)`` and
# returns ``OfferProposal | None``. We try them in order; first hit
# wins. Keep the most specific patterns first so generic catch-alls
# don't steal them.

_AppMatch = re.Match[str]
_HandlerFn = Callable[[_AppMatch, str], "OfferProposal | None"]


def _h_open_app(m: _AppMatch, _q: str) -> OfferProposal | None:
    raw = (m.group("app") or "").strip()
    canonical = _normalise_app(raw)
    if not canonical:
        return None
    return OfferProposal(
        action="open_app",
        args={"app_name": canonical},
        offer_text=f"Want me to open {canonical} for you, Boss?",
        category="app_open",
    )


def _h_close_app(m: _AppMatch, _q: str) -> OfferProposal | None:
    raw = (m.group("app") or "").strip()
    canonical = _normalise_app(raw)
    if not canonical:
        return None
    return OfferProposal(
        action="close_app",
        args={"app_name": canonical},
        offer_text=f"Want me to close {canonical}, Boss?",
        category="app_close",
    )


def _h_weather(m: _AppMatch, _q: str) -> OfferProposal | None:
    city = (m.groupdict().get("city") or "").strip(" .,?!")
    args: dict[str, Any] = {}
    if city:
        args["city"] = city
        offer = f"Want me to pull the weather for {city.title()}, Boss?"
    else:
        offer = "Want me to pull the latest weather, Boss?"
    return OfferProposal(
        action="weather",
        args=args,
        offer_text=offer,
        category="weather",
    )


def _h_resource(_m: _AppMatch, _q: str) -> OfferProposal | None:
    return OfferProposal(
        action="resource_report",
        args={},
        offer_text="Want me to run a quick system check, Boss?",
        category="system_resource",
    )


def _h_wifi(_m: _AppMatch, _q: str) -> OfferProposal | None:
    return OfferProposal(
        action="wifi_status",
        args={},
        offer_text="Want me to check the network status, Boss?",
        category="network",
    )


def _h_news(_m: _AppMatch, _q: str) -> OfferProposal | None:
    return OfferProposal(
        action="news_headlines",
        args={},
        offer_text="Want me to read out today's headlines, Boss?",
        category="news",
    )


def _h_volume_up(_m: _AppMatch, _q: str) -> OfferProposal | None:
    return OfferProposal(
        action="set_volume",
        args={"level": 80},
        offer_text="Want me to bump the volume up for you, Boss?",
        category="volume",
    )


def _h_volume_down(_m: _AppMatch, _q: str) -> OfferProposal | None:
    return OfferProposal(
        action="set_volume",
        args={"level": 30},
        offer_text="Want me to lower the volume, Boss?",
        category="volume",
    )


def _h_mute(_m: _AppMatch, _q: str) -> OfferProposal | None:
    return OfferProposal(
        action="mute",
        args={},
        offer_text="Want me to mute it, Boss?",
        category="volume",
    )


def _h_brightness(_m: _AppMatch, _q: str) -> OfferProposal | None:
    return OfferProposal(
        action="set_brightness",
        args={"level": 80},
        offer_text="Want me to adjust the brightness, Boss?",
        category="display",
    )


def _h_screenshot(_m: _AppMatch, _q: str) -> OfferProposal | None:
    return OfferProposal(
        action="screenshot",
        args={},
        offer_text="Want me to grab a screenshot, Boss?",
        category="screen",
    )


def _h_lock(_m: _AppMatch, _q: str) -> OfferProposal | None:
    return OfferProposal(
        action="lock_screen",
        args={},
        offer_text="Want me to lock the screen for you, Boss?",
        category="security",
    )


def _h_screen_describe(_m: _AppMatch, _q: str) -> OfferProposal | None:
    return OfferProposal(
        action="screen_describe",
        args={},
        offer_text="Want me to take a look at your screen and describe it, Boss?",
        category="vision",
    )


def _h_vision_describe(_m: _AppMatch, _q: str) -> OfferProposal | None:
    return OfferProposal(
        action="vision_describe",
        args={"prompt": "user-facing self-check"},
        offer_text="Want me to take a quick look through the camera, Boss?",
        category="vision",
    )


def _h_music_play(_m: _AppMatch, _q: str) -> OfferProposal | None:
    return OfferProposal(
        action="music_play",
        args={},
        offer_text="Want me to start the music for you, Boss?",
        category="music",
    )


def _h_reminder(m: _AppMatch, _q: str) -> OfferProposal | None:
    topic = (m.groupdict().get("topic") or "").strip(" .,?!")
    args: dict[str, Any] = {}
    if topic:
        args["text"] = topic
        offer = f"Want me to set a reminder for '{topic}', Boss?"
    else:
        offer = "Want me to set a reminder, Boss?"
    return OfferProposal(
        action="set_reminder",
        args=args,
        offer_text=offer,
        category="reminder",
    )


def _h_agenda(_m: _AppMatch, _q: str) -> OfferProposal | None:
    return OfferProposal(
        action="whats_on_my_plate",
        args={},
        offer_text="Want me to walk you through what's on your plate, Boss?",
        category="agenda",
    )


def _h_briefing(_m: _AppMatch, _q: str) -> OfferProposal | None:
    return OfferProposal(
        action="daily_briefing",
        args={},
        offer_text="Want me to run your daily briefing, Boss?",
        category="briefing",
    )


def _h_research(m: _AppMatch, _q: str) -> OfferProposal | None:
    topic = (m.groupdict().get("topic") or "").strip(" .,?!")
    if not topic or len(topic) > 80:
        return None
    return OfferProposal(
        action="research_topic",
        args={"topic": topic},
        offer_text=f"Want me to do a deeper dive on {topic}, Boss?",
        category="research",
    )


# Pattern table -- order matters (specific before generic). Each entry
# is ``(compiled_regex, handler)``. The regex is matched against the
# *normalised lowercased* query.
_PATTERNS: list[tuple[re.Pattern[str], _HandlerFn]] = [
    # ── App open ───────────────────────────────────────────────────
    (
        re.compile(
            r"\b(?:how\s+(?:do\s+i|to|can\s+i)|how\s+would\s+i)\s+"
            r"(?:open|launch|start|fire\s+up|run)\s+"
            r"(?P<app>[a-z][a-z0-9 .+-]{1,30}?)"
            r"(?:\s+(?:app|application|on\s+my\s+(?:mac|laptop)))?\s*[?.!]?\s*$",
            re.X | re.I,
        ),
        _h_open_app,
    ),
    # ── App close ──────────────────────────────────────────────────
    (
        re.compile(
            r"\b(?:how\s+(?:do\s+i|to|can\s+i))\s+"
            r"(?:close|quit|kill|shut\s+down|stop)\s+"
            r"(?P<app>[a-z][a-z0-9 .+-]{1,30}?)"
            r"(?:\s+(?:app|application))?\s*[?.!]?\s*$",
            re.X | re.I,
        ),
        _h_close_app,
    ),
    # ── Weather (with optional city) ───────────────────────────────
    (
        re.compile(
            r"\b(?:what(?:'s|\s+is|s)?|how(?:'s|\s+is)?|tell\s+me)\b"
            r"[^?]*?\bweather\b"
            r"(?:\s+(?:like\s+)?(?:in|at|for|of)\s+"
            r"(?P<city>[a-z][a-z .'-]{1,40}?))?"
            r"\s*[?.!]?\s*$",
            re.X | re.I,
        ),
        _h_weather,
    ),
    # ── Battery / CPU / RAM / system resource ──────────────────────
    (
        re.compile(
            r"\b(?:what(?:'s|\s+is|s)?|how(?:'s|\s+is|\s+much))\b[^?]*?"
            r"\b(?:battery|cpu|ram|memory|storage|disk|temperature)\b",
            re.X | re.I,
        ),
        _h_resource,
    ),
    # ── Wifi / network ─────────────────────────────────────────────
    (
        re.compile(
            r"\b(?:what(?:'s|\s+is|s)?|am\s+i|check)\b[^?]*?"
            r"\b(?:wifi|wi-?fi|internet|network)\b",
            re.X | re.I,
        ),
        _h_wifi,
    ),
    # ── News ───────────────────────────────────────────────────────
    (
        re.compile(
            r"\b(?:what(?:'s|\s+is|s)?|tell\s+me|read\s+me)\b[^?]*?"
            r"\b(?:news|headlines|happening\s+today)\b",
            re.X | re.I,
        ),
        _h_news,
    ),
    # ── Volume up ──────────────────────────────────────────────────
    (
        re.compile(
            r"\bhow\s+(?:do\s+i|to|can\s+i)\s+"
            r"(?:turn\s+up|increase|raise|bump\s+up|boost)\s+"
            r"(?:the\s+)?(?:volume|sound|audio)\b",
            re.X | re.I,
        ),
        _h_volume_up,
    ),
    # ── Volume down ────────────────────────────────────────────────
    (
        re.compile(
            r"\bhow\s+(?:do\s+i|to|can\s+i)\s+"
            r"(?:turn\s+down|decrease|lower|reduce)\s+"
            r"(?:the\s+)?(?:volume|sound|audio)\b",
            re.X | re.I,
        ),
        _h_volume_down,
    ),
    # ── Mute ───────────────────────────────────────────────────────
    (
        re.compile(
            r"\bhow\s+(?:do\s+i|to|can\s+i)\s+"
            r"(?:mute|silence)\s+(?:the\s+|my\s+)?"
            r"(?:volume|sound|audio|mac|laptop|computer|mic|microphone)",
            re.X | re.I,
        ),
        _h_mute,
    ),
    # ── Brightness ─────────────────────────────────────────────────
    (
        re.compile(
            r"\bhow\s+(?:do\s+i|to|can\s+i)\s+"
            r"(?:adjust|change|increase|decrease|turn\s+up|turn\s+down|set)\s+"
            r"(?:the\s+)?brightness\b",
            re.X | re.I,
        ),
        _h_brightness,
    ),
    # ── Screenshot ─────────────────────────────────────────────────
    (
        re.compile(
            r"\bhow\s+(?:do\s+i|to|can\s+i)\s+"
            r"(?:take|grab|capture|snap)\s+(?:a\s+)?screenshot\b",
            re.X | re.I,
        ),
        _h_screenshot,
    ),
    # ── Lock screen ────────────────────────────────────────────────
    (
        re.compile(
            r"\bhow\s+(?:do\s+i|to|can\s+i)\s+lock\s+"
            r"(?:my\s+|the\s+)?(?:screen|mac|laptop|computer)\b",
            re.X | re.I,
        ),
        _h_lock,
    ),
    # ── Music play ─────────────────────────────────────────────────
    (
        re.compile(
            r"\bhow\s+(?:do\s+i|to|can\s+i)\s+"
            r"(?:play|start|put\s+on)\s+(?:some\s+)?(?:music|songs?|spotify)\b",
            re.X | re.I,
        ),
        _h_music_play,
    ),
    # ── Screen describe ────────────────────────────────────────────
    (
        re.compile(
            r"\b(?:what(?:'s|\s+is)?|how)\b[^?]*?"
            r"\bon\s+(?:my\s+|the\s+)?(?:screen|display|monitor)\b",
            re.X | re.I,
        ),
        _h_screen_describe,
    ),
    # ── Vision (camera) describe ───────────────────────────────────
    (
        re.compile(
            r"\b(?:what\s+do\s+you\s+see|how\s+do\s+i\s+look|"
            r"can\s+you\s+see\s+me)\b",
            re.X | re.I,
        ),
        _h_vision_describe,
    ),
    # ── Reminder (optional topic capture) ──────────────────────────
    (
        re.compile(
            r"\bhow\s+(?:do\s+i|to|can\s+i)\s+"
            r"(?:set|create|add|make)\s+(?:a\s+)?reminder"
            r"(?:\s+(?:for|to|about)\s+(?P<topic>[a-z0-9 ._-]{1,80}))?"
            r"\s*[?.!]?\s*$",
            re.X | re.I,
        ),
        _h_reminder,
    ),
    # ── Agenda / what's on plate ───────────────────────────────────
    (
        re.compile(
            r"\b(?:what(?:'s|\s+is|s)?\s+on\s+my\s+(?:plate|agenda|schedule|"
            r"list|calendar)|what\s+(?:do|should)\s+i\s+(?:have|do)\s+today)\b",
            re.X | re.I,
        ),
        _h_agenda,
    ),
    # ── Daily briefing ─────────────────────────────────────────────
    (
        re.compile(
            r"\b(?:what(?:'s|\s+is|s)?\s+my\s+day\s+(?:like|looking)|"
            r"what(?:'s|\s+is|s)?\s+the\s+(?:plan|day)\s+(?:today|like)|"
            r"daily\s+briefing|morning\s+briefing)\b",
            re.X | re.I,
        ),
        _h_briefing,
    ),
    # ── Research / explainer ("tell me about X" / "what is X") ─────
    # Generic catch-all -- runs LAST so the more specific patterns
    # above always win. ``topic`` capture is bounded to 80 chars to
    # avoid swallowing entire monologues.
    (
        re.compile(
            r"\b(?:tell\s+me\s+about|what\s+is|what's|explain)\s+"
            r"(?P<topic>[a-z0-9][a-z0-9 ._'-]{2,80}?)"
            r"\s*[?.!]?\s*$",
            re.X | re.I,
        ),
        _h_research,
    ),
]


# ── Public synthesizer ─────────────────────────────────────────────


_NEGATIVE_INTENT_HINTS = (
    "don't", "dont", "no need", "not now", "later", "skip", "never mind",
    "nevermind", "forget it", "cancel",
)


def synthesize_offer(
    query: str,
    response: str = "",
    *,
    skip_if_short_response: bool = True,
) -> OfferProposal | None:
    """Map a user query to an actionable Jarvis follow-up offer.

    Returns ``None`` when:

      * The query is empty or not explainer-shaped.
      * No pattern matches (we don't fabricate generic offers --
        offering for the sake of offering is the chatbox failure mode).
      * The reply is so short it's already an action ack
        (``skip_if_short_response`` skips offers when reply < 12 chars).
      * The query contains a negative-intent phrase ("don't", "no need")
        the user explicitly used to wave off action.

    The router is expected to:
      1. Append ``proposal.offer_text`` to the assistant reply.
      2. Stash the proposal in ``OfferRegistry``.
      3. On the next turn, if the user confirms, build an
         ``IntentResult(intent='confirm_offer', action=proposal.action,
         action_args=proposal.args)`` and run it through
         ``_execute_action``.
    """
    if not query or not query.strip():
        return None
    text = query.strip()

    if any(hint in text.lower() for hint in _NEGATIVE_INTENT_HINTS):
        return None

    if skip_if_short_response and len((response or "").strip()) < 12:
        return None

    if not is_explainer_query(text):
        return None

    for pattern, handler in _PATTERNS:
        m = pattern.search(text)
        if m is None:
            continue
        try:
            proposal = handler(m, text)
        except Exception as exc:  # pragma: no cover -- defensive
            logger.warning(
                "Offer handler %s failed for '%s': %s",
                handler.__name__, text[:60], exc,
            )
            return None
        if proposal is not None:
            logger.info(
                "OfferSynth: '%s' -> %s (%s)",
                text[:60], proposal.action, proposal.category,
            )
            return proposal

    return None
