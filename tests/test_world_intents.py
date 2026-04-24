"""B8: news_headlines must require a verb anchor — bare "news" used
as an adjective for another noun (e.g. "good news new movies",
"news report style article") used to false-positive and route to
news_headlines (atom_log.txt L418-420).
"""
from __future__ import annotations

import pytest

from core.intent_engine.world_intents import check


# Real news queries that MUST classify as news_headlines. Note that
# "what's happening in the world" and "news briefing" are intentionally
# omitted — they correctly land on world_status / daily_briefing
# respectively (which serve the user just as well).
_TRUE_NEWS = [
    "tell me the news",
    "give me the latest news",
    "show me today's headlines",
    "what's the latest news",
    "any breaking news",
    "any latest headlines",
    "read me the top stories",
    "world news please",
    "breaking news today",
    "fetch latest headlines",
    "top headlines",
    "kya chal raha",
    "news update",
    "news report",
]

# Strings where "news" is an adjective for another noun, NOT a request
# for the news. These were the false-positive cases in the live log.
_NOT_NEWS = [
    "good news new movies",
    "news new movies",
    "any news new movies coming this weekend",
    "news report style article",
    "news of the world album review",
    "news article about climate",
    "old news kind of feeling",
    "news story about ATOM",
]


@pytest.mark.parametrize("text", _TRUE_NEWS)
def test_news_queries_classify_as_news_headlines(text: str) -> None:
    result = check(text)
    assert result is not None, f"check() returned None for {text!r}"
    assert result.intent == "news_headlines", (
        f"check() returned intent={result.intent!r} for {text!r}"
    )
    assert result.action == "news_headlines"


@pytest.mark.parametrize("text", _NOT_NEWS)
def test_news_as_adjective_does_not_classify_as_news(text: str) -> None:
    """Queries where "news" modifies another noun must fall through
    to whatever else might match — or to None (LLM fallback). They
    must NOT be intercepted by news_headlines."""
    result = check(text)
    if result is not None:
        assert result.intent != "news_headlines", (
            f"false-positive news_headlines on {text!r}: {result}"
        )


def test_briefing_does_not_steal_from_news() -> None:
    """'give me a brief' is a daily_briefing, not news_headlines —
    confirms the priority order in check() is preserved."""
    result = check("give me a brief")
    assert result is not None
    assert result.intent == "daily_briefing"


def test_world_status_does_not_steal_from_news() -> None:
    """'global situation' is world_status, not news_headlines."""
    result = check("global situation")
    assert result is not None
    assert result.intent == "world_status"
