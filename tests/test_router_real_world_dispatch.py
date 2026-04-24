"""Regression for B6: weather/news/clock intents must dispatch through
Router._do_* handlers (synchronous tool call) instead of falling
through to the LLM fallback (which hallucinated weather in
atom_log.txt L356-392).
"""
from __future__ import annotations

from unittest.mock import MagicMock

from core.router.router import Router


def test_real_world_intents_are_in_action_dispatch_table():
    """The dispatch table must own every intent that the bus-side
    `wire_real_world` handler used to reactively answer. Without
    these entries the router falls through to the LLM fallback and
    the model hallucinates weather/news/time."""
    expected = {
        "weather_report",
        "news_headlines",
        "world_clock",
        "daily_briefing",
        "temporal_info",
        "world_status",
    }
    missing = expected - set(Router._ACTION_DISPATCH.keys())
    assert not missing, (
        f"intents missing from Router._ACTION_DISPATCH: {missing}. "
        f"Router will fall through to the LLM and the model will "
        f"hallucinate the answer instead of calling the tool."
    )


def test_weather_report_dispatch_calls_real_world_get_weather():
    """When real_world_intel is wired, _do_weather_report must call
    its get_weather_summary() — not the legacy _do_weather (which
    needs the online_weather feature flag)."""
    rw = MagicMock()
    rw.get_weather_summary.return_value = "Current weather: 22C, sunny."

    router = Router.__new__(Router)
    router._real_world = rw
    router._config = {"features": {}}

    out = Router._do_weather_report(router, "weather_report", {})
    assert "22C" in out
    rw.get_weather_summary.assert_called_once()


def test_news_headlines_dispatch_calls_real_world_get_news():
    rw = MagicMock()
    rw.get_news_summary.return_value = "Top headline: ATOM ships v3.4."

    router = Router.__new__(Router)
    router._real_world = rw

    out = Router._do_news_headlines(router, "news_headlines", {"count": 3})
    assert "ATOM ships v3.4" in out
    rw.get_news_summary.assert_called_once()
    args, kwargs = rw.get_news_summary.call_args
    assert kwargs.get("count") == 3 or (args and args[0] == 3)


def test_real_world_handlers_degrade_gracefully_without_attachment():
    """Calling the dispatch handlers when no real_world_intel is wired
    must NOT crash — return a polite "not wired" message instead so
    the router still emits a response and doesn't fall back to LLM."""
    router = Router.__new__(Router)
    router._real_world = None
    router._config = {"features": {}}

    for action, fn in [
        ("news_headlines", Router._do_news_headlines),
        ("world_clock", Router._do_world_clock),
        ("daily_briefing", Router._do_daily_briefing),
        ("world_status", Router._do_world_status),
    ]:
        out = fn(router, action, {})
        assert isinstance(out, str) and out, (
            f"{action} returned empty string when real_world_intel=None"
        )
