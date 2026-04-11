"""Focused tests for command cache freshness rules."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.command_cache import CommandCache


class FakeIntentResult:
    def __init__(self, intent: str, response: str = "") -> None:
        self.intent = intent
        self.response = response


def test_dynamic_info_results_are_not_cached() -> None:
    cache = CommandCache()
    time_result = FakeIntentResult("time", response="It's 9:41 AM.")

    cache.put("what time is it", time_result)
    cache.put_intent_key("info:time", time_result)

    assert cache.get("what time is it") is None
    assert cache.get("info:time") is None
    print("  PASS: Dynamic info results stay uncached")


def test_repeat_actions_still_use_cache() -> None:
    cache = CommandCache()
    open_result = FakeIntentResult("open_app", response="Opening Chrome, Boss.")

    cache.put("open chrome", open_result)

    assert cache.get("open chrome") is open_result
    print("  PASS: Repeat action commands still use the cache")


if __name__ == "__main__":
    test_dynamic_info_results_are_not_cached()
    test_repeat_actions_still_use_cache()
    print("\ntest_command_cache: ALL PASSED")
