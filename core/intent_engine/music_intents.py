"""
ATOM Intent Engine -- Music intents (Spotify-first transport + search).

These intents take precedence over the YouTube fallback in
``media_intents.py`` so that the local Spotify desktop app handles all
"play / pause / next / what's playing" verbs. The YouTube path stays
reserved for explicit "play X on YouTube" requests.

Resolved actions::

    music_play          -> resume current Spotify selection
    music_pause         -> pause Spotify
    music_next          -> next track
    music_prev          -> previous track
    music_current       -> announce now-playing
    music_play_specific -> play a specific song / album / artist via search

Each intent returns an ``IntentResult`` that the router dispatches to
the matching ``_do_music_*`` handler.
"""

from __future__ import annotations

import re

from .base import IntentResult, clean_slot


# ── current track / now playing ──────────────────────────────────────


_NOW_PLAYING = re.compile(
    r"\b("
    r"what(?:'s|\s+is)?\s+playing|"
    r"what\s+song(?:\s+is)?\s+(?:this|that|playing|on)|"
    r"what\s+(?:am\s+i|are\s+we)\s+listening\s+to|"
    r"current\s+song|current\s+track|now\s+playing|"
    r"(?:tell|show)\s+me\s+(?:the\s+)?(?:current\s+)?(?:song|track)|"
    r"who\s+is\s+singing"
    r")\b",
    re.I,
)


# ── transport: pause ─────────────────────────────────────────────────


_MUSIC_PAUSE = re.compile(
    r"\b("
    r"pause(?:\s+(?:the\s+)?(?:music|song|track|spotify|playback|audio))?|"
    r"hold\s+(?:the\s+)?(?:music|song|track|playback)|"
    r"music\s+(?:band|ruk|chup)|"
    r"gana\s+(?:band|ruk|pause)\s*(?:karo|kar\s+do|do)?"
    r")\b",
    re.I,
)


# ── transport: next ──────────────────────────────────────────────────


_MUSIC_NEXT = re.compile(
    r"\b("
    r"(?:play\s+)?next(?:\s+(?:song|track|gana))?|"
    r"skip(?:\s+(?:this\s+)?(?:song|track|gana))?|"
    r"skip\s+ahead|"
    r"next\s+one|"
    r"forward\s+(?:song|track|gana)|"
    r"agla\s+gana|gana\s+badlo|gana\s+change\s+karo"
    r")\b",
    re.I,
)


# ── transport: previous ──────────────────────────────────────────────


_MUSIC_PREV = re.compile(
    r"\b("
    r"previous(?:\s+(?:song|track|gana))?|"
    r"(?:play|go)\s+(?:back\s+)?(?:to\s+)?(?:the\s+)?(?:previous|last|prev)(?:\s+(?:song|track|gana))?|"
    r"(?:replay|repeat)\s+(?:that|the\s+last\s+song|previous)|"
    r"go\s+back\s+(?:a\s+)?(?:song|track)|"
    r"pichla\s+gana"
    r")\b",
    re.I,
)


# ── transport: resume / play (no specific song) ──────────────────────


_MUSIC_RESUME = re.compile(
    r"\b("
    r"resume(?:\s+(?:the\s+)?(?:music|song|track|playback|spotify))?|"
    r"continue\s+(?:the\s+)?(?:music|song|track|playback)|"
    r"unpause(?:\s+(?:the\s+)?(?:music|song|track|playback|spotify))?|"
    r"start\s+(?:the\s+)?(?:music|song|track|playback)\s+again|"
    r"keep\s+playing|"
    r"gana\s+(?:chalu|fir|wapas)\s*(?:karo|kar\s+do|do)?"
    r")\b",
    re.I,
)


# Genre / mood / language adjectives we accept as descriptors in
# "play some pop songs" / "play me some lofi music" / "play hindi
# songs". When a genre is captured we route to ``music_play_specific``
# with ``kind="genre"`` so the Spotify search picks up the descriptor.
_MUSIC_GENRE_TOKEN = (
    r"(?:pop|rock|jazz|blues|edm|"
    r"lofi|lo-?fi|chill|focus|study|workout|gym|party|sleep|relax|"
    r"hindi|bollywood|punjabi|tamil|telugu|english|"
    r"classical|instrumental|acoustic|romantic|sad|happy|"
    r"hip[\s-]?hop|rap|indie|metal|punk|country|reggae|"
    r"k-?pop|j-?pop|desi|sufi|bhajan|qawwali|ghazal)"
)
_MUSIC_GENRE_RE = re.compile(rf"\b{_MUSIC_GENRE_TOKEN}\b", re.I)


# Generic "play music" / "play some music" / "can you play some music
# for me" / "play some pop-up songs" etc. -- no specific song name
# and no "on youtube". Lands on Spotify (resume if no genre, search
# if a genre adjective is captured).
#
# The leading polite/wake-prefix block accepts:
#   "hey atom", "atom", "can you", "could you", "would you", "please"
#   in any order -- log line 318 ("Can you play some music for me")
#   was the original miss.
_MUSIC_GENERIC_PLAY = re.compile(
    r"^\s*"
    r"(?:hey\s+atom[,\s]+)?"
    r"(?:atom[,\s]+)?"
    r"(?:(?:can|could|would|will)\s+(?:you|u)\s+)?"
    r"(?:please\s+)?"
    r"(?:play|put\s+on|start|fire\s+up|spin\s+up|throw\s+on)\s+"
    r"(?:me\s+)?"
    r"(?:some\s+|the\s+|my\s+|a\s+(?:bit\s+of\s+)?)?"
    r"(?:"
    rf"  (?P<genre_with_noun>{_MUSIC_GENRE_TOKEN})(?:[\s-]+(?:up|out))?"
    r"   \s+(?:music|songs?|tunes?|spotify|gana|gaana|sangeet|playlist|jams?|tracks?)"
    r"|"
    rf"  (?P<genre_only>{_MUSIC_GENRE_TOKEN})"
    r"|"
    r"  (?:music|songs?|tunes?|spotify|gana|gaana|sangeet|playlist|jams?|tracks?)"
    r")"
    r"(?:\s+(?:for\s+me|please|atom|boss))*\s*[?.!]?\s*$",
    re.X | re.I,
)


# ── play a specific song ─────────────────────────────────────────────


# "play X on spotify", "play X by Y", "play song X", "spotify play X"
_MUSIC_PLAY_SPECIFIC = re.compile(
    r"\b(?:"
    r"(?:please\s+)?play\s+(?:(?:the\s+)?(?:song|track|album|artist)\s+)?"
    r"(?P<query>.+?)\s+on\s+spotify|"
    r"spotify(?:\s*[:,])?\s*play\s+(?P<query2>.+?)|"
    r"(?:please\s+)?play\s+(?P<query3>.+?)\s+by\s+(?P<artist>.+?)"
    r")\s*$",
    re.I,
)


# Strict "play <song name>" -- must contain at least two words after
# 'play', and the first word must NOT be a generic music token (so we
# don't double-fire on "play music"). This is intentionally narrow so
# the YouTube path keeps owning ambiguous "play X" without a platform.
_MUSIC_PLAY_BARE = re.compile(
    r"^\s*(?:hey\s+atom\s+)?(?:please\s+)?play\s+"
    r"(?P<query>(?!(?:music|song|songs|tunes|spotify|gana|gaana|sangeet|"
    r"youtube|video|some|the|my|a|an)\b)[\w'’-]+(?:\s+[\w'’-]+){1,8})"
    r"(?:\s+on\s+spotify)?\s*$",
    re.I,
)


_GENERIC_FILLER_TOKENS = {"the", "a", "an", "song", "track", "music", "please"}

# Phrases that explicitly steer playback to YouTube / web video. When
# present we bow out so ``media_intents.check()`` (next in the cascade)
# can hand the request to the YouTube launcher.
_YOUTUBE_HINT_RE = re.compile(
    r"\b(?:on\s+(?:youtube|screen|tv)|in\s+(?:youtube|chrome|browser)|"
    r"the\s+(?:youtube\s+)?video|youtube\s+video)\b",
    re.I,
)


def _normalise_query(raw: str) -> str:
    """Strip wake words, polite filler, and trailing platform hints."""
    if not raw:
        return ""
    cleaned = clean_slot(raw)
    cleaned = re.sub(r"\bon\s+spotify\s*$", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^(?:hey\s+atom|atom)[,\s]+", "", cleaned, flags=re.I)
    cleaned = cleaned.strip(" .,'\"")
    if not cleaned:
        return ""
    tokens = cleaned.split()
    while tokens and tokens[0].lower() in _GENERIC_FILLER_TOKENS:
        tokens.pop(0)
    return " ".join(tokens).strip()


# ── public API ───────────────────────────────────────────────────────


def check(text: str) -> IntentResult | None:
    """Return a music ``IntentResult`` or None if nothing matches.

    Order matters: now-playing first (so "what's playing right now" is
    not eaten by play_specific), then transport verbs, then bare
    "play <song>" matchers last because they're the loosest.
    """
    # Hand off explicit YouTube/screen requests to ``media_intents``.
    if _YOUTUBE_HINT_RE.search(text):
        return None

    if _NOW_PLAYING.search(text):
        return IntentResult("music_current", action="music_current",
                            action_args={})

    if _MUSIC_PAUSE.search(text):
        return IntentResult("music_pause", action="music_pause",
                            action_args={})

    if _MUSIC_NEXT.search(text):
        return IntentResult("music_next", action="music_next",
                            action_args={})

    if _MUSIC_PREV.search(text):
        return IntentResult("music_prev", action="music_prev",
                            action_args={})

    if _MUSIC_RESUME.search(text):
        return IntentResult("music_play", action="music_play",
                            action_args={})

    m_generic = _MUSIC_GENERIC_PLAY.search(text)
    if m_generic:
        gd = m_generic.groupdict()
        genre = clean_slot(gd.get("genre_with_noun") or gd.get("genre_only"))
        if genre:
            return IntentResult(
                "music_play_specific",
                action="music_play_specific",
                action_args={"query": genre, "kind": "genre"},
            )
        return IntentResult("music_play", action="music_play",
                            action_args={})

    m = _MUSIC_PLAY_SPECIFIC.search(text)
    if m:
        gd = m.groupdict()
        artist = clean_slot(gd.get("artist"))
        track_query = (
            clean_slot(gd.get("query"))
            or clean_slot(gd.get("query2"))
            or clean_slot(gd.get("query3"))
        )
        track_query = _normalise_query(track_query)
        if artist:
            full = f"{track_query} {artist}".strip()
        else:
            full = track_query
        if full:
            return IntentResult(
                "music_play_specific",
                action="music_play_specific",
                action_args={"query": full, "kind": "track"},
            )

    m = _MUSIC_PLAY_BARE.search(text)
    if m:
        query = _normalise_query(m.group("query"))
        if query:
            return IntentResult(
                "music_play_specific",
                action="music_play_specific",
                action_args={"query": query, "kind": "track"},
            )
    return None


def quick_match(text: str) -> str | None:
    """Lightweight match used by STT for early-finalize."""
    if _YOUTUBE_HINT_RE.search(text):
        return None
    if _NOW_PLAYING.search(text):
        return "music_current"
    if _MUSIC_PAUSE.search(text):
        return "music_pause"
    if _MUSIC_NEXT.search(text):
        return "music_next"
    if _MUSIC_PREV.search(text):
        return "music_prev"
    if _MUSIC_RESUME.search(text) or _MUSIC_GENERIC_PLAY.search(text):
        return "music_play"
    return None
