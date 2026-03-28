"""
ATOM Intent Engine -- Real-world intelligence intents.

Patterns for weather reports, news headlines, daily briefing,
world clock, temporal info, and world status queries.
"""

from __future__ import annotations

import re

from .base import IntentResult

_WEATHER_REPORT = re.compile(
    r"\b(weather\s+report|full\s+weather|weather\s+update|"
    r"how('?s|\s+is)\s+(it\s+)?outside|weather\s+outside|"
    r"what('?s|\s+is)\s+the\s+(weather|temperature)|"
    r"forecast|mausam|temperature\s+outside)\b", re.I)

_NEWS = re.compile(
    r"\b(news|headlines|what('?s|\s+is)\s+(happening|going\s+on)|"
    r"top\s+(stories|headlines|news)|latest\s+news|"
    r"world\s+news|news\s+update|kya\s+chal\s+raha)\b", re.I)

_BRIEFING = re.compile(
    r"\b(brief(ing)?|morning\s+brief|daily\s+brief|"
    r"give\s+me\s+(a\s+)?brief|catch\s+me\s+up|"
    r"what\s+did\s+i\s+miss|update\s+me|"
    r"what('?s|\s+is)\s+new|status\s+report)\b", re.I)

_WORLD_CLOCK = re.compile(
    r"\b(world\s+(clock|time)|time\s+(in|at|around)\s+\w+|"
    r"international\s+time|timezone|time\s+zones?|"
    r"what\s+time\s+is\s+it\s+in)\b", re.I)

_TEMPORAL = re.compile(
    r"\b(what\s+season|sunrise|sunset|moon\s+phase|"
    r"when('?s|\s+is)\s+(the\s+)?(sunrise|sunset|weekend)|"
    r"is\s+it\s+(a\s+)?holiday|any\s+holiday|"
    r"how\s+(many\s+)?days\s+(until|till)\s+(the\s+)?weekend)\b", re.I)

_WORLD_STATUS = re.compile(
    r"\b(world\s+status|what('?s|\s+is)\s+happening\s+(in\s+the\s+)?world|"
    r"global\s+(status|update|situation)|"
    r"tell\s+me\s+(about\s+)?(the\s+)?world|"
    r"world\s+report|situation\s+report|sitrep)\b", re.I)


def check(text: str) -> IntentResult | None:
    if _WORLD_STATUS.search(text):
        return IntentResult(
            "world_status", action="world_status", action_args={})

    if _BRIEFING.search(text):
        return IntentResult(
            "daily_briefing", action="daily_briefing", action_args={})

    if _NEWS.search(text):
        return IntentResult(
            "news_headlines", action="news_headlines", action_args={})

    if _WEATHER_REPORT.search(text):
        return IntentResult(
            "weather_report", action="weather_report", action_args={})

    if _WORLD_CLOCK.search(text):
        return IntentResult(
            "world_clock", action="world_clock", action_args={})

    if _TEMPORAL.search(text):
        return IntentResult(
            "temporal_info", action="temporal_info", action_args={})

    return None
