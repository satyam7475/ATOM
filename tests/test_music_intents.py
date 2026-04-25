"""
ATOM -- regression suite for ``core.intent_engine.music_intents`` (F2).

Pins three behaviours:

1. The Spotify-first verbs ("pause music", "next song", "what's playing")
   resolve to ``music_*`` intents *before* the YouTube fallback in
   ``media_intents.py`` runs.
2. ``play <song>`` shapes correctly extract the song name slot and
   route to ``music_play_specific`` with a clean ``query`` arg.
3. Negative cases that look musical but mean something else (e.g.
   "play this video on YouTube") still go to YouTube, not Spotify.
"""

from __future__ import annotations

import pytest

from core.intent_engine import IntentEngine
from core.intent_engine import music_intents


# ── transport verbs ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "phrase",
    [
        "pause music",
        "pause the music",
        "pause the song",
        "pause spotify",
        "hold the music",
        "music band karo",
        "gana pause karo",
    ],
)
def test_pause_phrases_resolve_to_music_pause(phrase: str) -> None:
    result = music_intents.check(phrase)
    assert result is not None
    assert result.intent == "music_pause"
    assert result.action == "music_pause"
    assert result.action_args == {}


@pytest.mark.parametrize(
    "phrase",
    [
        "next song",
        "next track",
        "next",
        "skip this song",
        "skip ahead",
        "play next",
        "agla gana",
    ],
)
def test_next_phrases_resolve_to_music_next(phrase: str) -> None:
    result = music_intents.check(phrase)
    assert result is not None
    assert result.intent == "music_next"
    assert result.action == "music_next"


@pytest.mark.parametrize(
    "phrase",
    [
        "previous track",
        "previous song",
        "play the previous song",
        "go back to the last song",
        "replay that",
        "repeat the last song",
        "pichla gana",
    ],
)
def test_prev_phrases_resolve_to_music_prev(phrase: str) -> None:
    result = music_intents.check(phrase)
    assert result is not None
    assert result.intent == "music_prev"


@pytest.mark.parametrize(
    "phrase",
    [
        "resume music",
        "resume the song",
        "continue the playback",
        "unpause spotify",
        "keep playing",
        "play music",
        "play some music",
        "play music for me",
        "put on some music",
        "start the music",
        "play my songs",
        "gana chalu karo",
    ],
)
def test_resume_phrases_resolve_to_music_play(phrase: str) -> None:
    result = music_intents.check(phrase)
    assert result is not None
    assert result.intent == "music_play"
    assert result.action == "music_play"


@pytest.mark.parametrize(
    "phrase",
    [
        "what's playing",
        "what is playing",
        "what song is this",
        "what song is playing",
        "what am I listening to",
        "current song",
        "now playing",
        "tell me the current track",
        "who is singing",
    ],
)
def test_now_playing_phrases_resolve_to_music_current(phrase: str) -> None:
    result = music_intents.check(phrase)
    assert result is not None
    assert result.intent == "music_current"


# ── play <song> ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "phrase, expected_query",
    [
        ("play despacito on spotify", "despacito"),
        ("play smells like teen spirit on spotify", "smells like teen spirit"),
        ("spotify play stairway to heaven", "stairway to heaven"),
        ("play the song bohemian rhapsody on spotify", "bohemian rhapsody"),
        ("play the album OK Computer on spotify", "OK Computer"),
    ],
)
def test_play_specific_extracts_clean_query(
    phrase: str, expected_query: str,
) -> None:
    result = music_intents.check(phrase)
    assert result is not None, f"no intent for {phrase!r}"
    assert result.intent == "music_play_specific"
    assert result.action == "music_play_specific"
    assert result.action_args is not None
    assert result.action_args["query"].lower() == expected_query.lower()
    assert result.action_args["kind"] == "track"


def test_play_x_by_y_attaches_artist_to_query() -> None:
    result = music_intents.check("play yesterday by the beatles")
    assert result is not None
    assert result.intent == "music_play_specific"
    assert result.action_args is not None
    query = result.action_args["query"].lower()
    assert "yesterday" in query
    assert "beatles" in query


def test_bare_play_two_word_song_routes_to_spotify() -> None:
    result = music_intents.check("play teen spirit")
    assert result is not None
    assert result.intent == "music_play_specific"
    assert result.action_args is not None
    assert result.action_args["query"].lower() == "teen spirit"


def test_bare_play_single_word_does_not_match_specific() -> None:
    """Single-word "play X" is too ambiguous to send to Spotify search."""
    result = music_intents.check("play despacito")
    # single-word body should NOT match _MUSIC_PLAY_BARE
    if result is not None:
        assert result.intent != "music_play_specific"


# ── precedence vs. media_intents (YouTube) ─────────────────────────


def test_engine_prefers_spotify_over_youtube_for_pause_music() -> None:
    """Ensure music_intents wins the cascade ordering inside IntentEngine."""
    engine = IntentEngine()
    out = engine.classify("pause music")
    assert out.intent == "music_pause"


def test_engine_prefers_spotify_for_play_some_music() -> None:
    engine = IntentEngine()
    out = engine.classify("play some music")
    assert out.intent == "music_play"


def test_engine_routes_play_x_on_youtube_to_youtube() -> None:
    """Explicit YouTube target must keep going to YouTube, not Spotify."""
    engine = IntentEngine()
    out = engine.classify("play despacito on youtube")
    assert out.intent == "play_youtube"


def test_engine_routes_play_x_on_screen_to_youtube() -> None:
    engine = IntentEngine()
    out = engine.classify("play music on screen")
    assert out.intent == "play_youtube"


def test_engine_routes_open_spotify_to_open_app() -> None:
    """'open spotify' must still launch the app, not press play."""
    engine = IntentEngine()
    out = engine.classify("open spotify")
    assert out.intent == "open_app"


def test_engine_keeps_volume_intents_intact() -> None:
    """We didn't break unrelated media intents (volume / mute)."""
    engine = IntentEngine()
    assert engine.classify("set volume to 40 percent").intent == "set_volume"
    assert engine.classify("mute system").intent == "mute"


def test_engine_quick_match_returns_music_pause() -> None:
    engine = IntentEngine()
    assert engine.quick_match("pause music") == "music_pause"
    assert engine.quick_match("next song") == "music_next"
    assert engine.quick_match("previous song") == "music_prev"
    assert engine.quick_match("what's playing") == "music_current"
    assert engine.quick_match("play music") == "music_play"


# ── safety: no false positives ──────────────────────────────────────


@pytest.mark.parametrize(
    "phrase",
    [
        "what time is it",
        "tell me the weather",
        "open chrome",
        "lock the screen",
        "set volume to 40 percent",
    ],
)
def test_non_music_phrases_do_not_match(phrase: str) -> None:
    assert music_intents.check(phrase) is None


# ── A2 sprint: phrases the live log proves we used to miss ──────────


@pytest.mark.parametrize(
    "phrase",
    [
        "can you play some music for me",
        "could you play some music",
        "would you play some music please",
        "can you play music",
        "atom can you play some music",
        "hey atom can you play some music for me",
        "please play some music for me",
        "fire up some music",
        "throw on some tunes",
        "play me some songs",
        "spin up the playlist",
        "put on a bit of music please",
    ],
)
def test_polite_prefix_play_music_routes_to_music_play(phrase: str) -> None:
    """All 12 phrases the atomLogs.txt sprint analysis flagged as
    misclassified to LLM must now resolve to ``music_play`` natively."""
    result = music_intents.check(phrase)
    assert result is not None, f"no intent for {phrase!r}"
    assert result.intent == "music_play", (
        f"{phrase!r} -> {result.intent} (expected music_play)"
    )


@pytest.mark.parametrize(
    "phrase, expected_genre",
    [
        ("play some pop songs", "pop"),
        ("play some pop-up songs", "pop"),
        ("play me some lofi music", "lofi"),
        ("play some lo-fi tunes", "lo-fi"),
        ("play some hindi songs", "hindi"),
        ("can you play some bollywood music", "bollywood"),
        ("put on some chill music for me", "chill"),
        ("play some focus music", "focus"),
        ("fire up some rock songs", "rock"),
        ("play some classical music please", "classical"),
        ("play some jazz", "jazz"),
        ("play some workout music", "workout"),
    ],
)
def test_genre_descriptor_routes_to_play_specific(
    phrase: str, expected_genre: str,
) -> None:
    result = music_intents.check(phrase)
    assert result is not None, f"no intent for {phrase!r}"
    assert result.intent == "music_play_specific"
    assert result.action_args is not None
    assert result.action_args["kind"] == "genre"
    assert result.action_args["query"].lower() == expected_genre.lower()


def test_engine_classifies_polite_play_as_music_play() -> None:
    engine = IntentEngine()
    out = engine.classify("can you play some music for me")
    assert out.intent == "music_play"


def test_engine_classifies_genre_as_play_specific() -> None:
    engine = IntentEngine()
    out = engine.classify("play some lofi music")
    assert out.intent == "music_play_specific"
    assert out.action_args is not None
    assert out.action_args.get("kind") == "genre"
