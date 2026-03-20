"""
ATOM Intent Engine -- Network intents (open_url, weather).
"""

from __future__ import annotations

import re

from .base import IntentResult

_OPEN_URL = re.compile(
    r"\b(open|go\s+to|navigate\s+to|visit)\s+(https?://\S+|www\.\S+)", re.I)

_WEATHER = re.compile(
    r"\b(weather|what('?s|\s+is)\s+the\s+weather|temperature\s+outside|"
    r"forecast|mausam|how('?s|\s+is)\s+the\s+weather)\b", re.I)


def check(text: str) -> IntentResult | None:
    m = _OPEN_URL.search(text)
    if m:
        url = m.group(2).strip()
        if not url.startswith("http"):
            url = "https://" + url
        return IntentResult("open_url", action="open_url", action_args={"url": url})

    if _WEATHER.search(text):
        return IntentResult("weather", action="weather", action_args={})
    return None
