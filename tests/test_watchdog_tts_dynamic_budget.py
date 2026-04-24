"""Regression tests for the dynamic TTS budget in RuntimeWatchdog.

Live log evidence (atom_log.txt L390-392):
    A 22-word weather reply at rate 172 wpm took ~16s to render but the
    static 15s ``watchdog_tts_timeout_s`` killed it 1s before the last
    sentence finished. The fix scales the budget to
    ``max(static_floor, words * watchdog_tts_per_word_s)`` capped at
    ``watchdog_tts_max_dynamic_s``, so:

    * one-liner ("On it, Boss.")        -> 15s floor
    * 22-word weather reply             -> max(15, 11)  = 15s
    * 40-word explanation               -> max(15, 20)  = 20s
    * 100-word essay                    -> max(15, 50)  = 45s (cap)

These tests pin the helper without booting the full async loop.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.runtime_watchdog import RuntimeWatchdog


class _FakeBus:
    def emit(self, *args: Any, **kwargs: Any) -> None: ...
    def emit_fast(self, *args: Any, **kwargs: Any) -> None: ...
    def emit_long(self, *args: Any, **kwargs: Any) -> None: ...
    def on(self, *args: Any, **kwargs: Any) -> None: ...


class _FakeState:
    def __init__(self) -> None:
        self.current = None


def _make_watchdog(perf: dict | None = None) -> RuntimeWatchdog:
    return RuntimeWatchdog(
        bus=_FakeBus(),
        state=_FakeState(),
        config={"performance": perf or {}},
    )


# ---------------------------------------------------------------------------
# Word counting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", 0),
        ("   ", 0),
        ("Hello", 1),
        ("On it, Boss.", 3),
        ("a b c d e", 5),
        ("multi  spaced  words", 3),
        ("\nleading\tand trailing\n", 3),
    ],
)
def test_count_tts_words(text: str, expected: int) -> None:
    assert RuntimeWatchdog._count_tts_words(text) == expected


# ---------------------------------------------------------------------------
# effective_tts_budget_s
# ---------------------------------------------------------------------------


def test_effective_budget_uses_static_floor_for_short_replies() -> None:
    """A one-liner gets at most the static 15s floor.

    Quick recovery from a stuck synth on short utterances must not be
    punished by the dynamic scaler — short replies still hit the floor.
    """
    wd = _make_watchdog(
        {"watchdog_tts_timeout_s": 15, "watchdog_tts_per_word_s": 0.5},
    )
    assert wd.effective_tts_budget_s(0) == 15.0
    assert wd.effective_tts_budget_s(1) == 15.0
    assert wd.effective_tts_budget_s(10) == 15.0   # 5s scaled < 15s floor


def test_effective_budget_scales_with_word_count() -> None:
    """Long replies (> floor / per_word_s) trigger the dynamic scaler.

    With per_word_s=0.5 and floor=15:
      * 30 words -> 15s scaled = 15s (matches floor exactly)
      * 40 words -> 20s scaled (wins)
      * 60 words -> 30s scaled (wins, still under 45s cap)
    """
    wd = _make_watchdog(
        {"watchdog_tts_timeout_s": 15, "watchdog_tts_per_word_s": 0.5},
    )
    assert wd.effective_tts_budget_s(30) == 15.0
    assert wd.effective_tts_budget_s(40) == 20.0
    assert wd.effective_tts_budget_s(60) == 30.0


def test_effective_budget_caps_at_max_dynamic() -> None:
    """A runaway 1000-word stream must still be cut off — the cap
    enforces a hard upper bound regardless of word count."""
    wd = _make_watchdog(
        {
            "watchdog_tts_timeout_s": 15,
            "watchdog_tts_per_word_s": 0.5,
            "watchdog_tts_max_dynamic_s": 45,
        },
    )
    assert wd.effective_tts_budget_s(100) == 45.0   # 50s scaled > 45 cap
    assert wd.effective_tts_budget_s(1000) == 45.0


def test_effective_budget_disabled_when_per_word_zero() -> None:
    """Setting per_word=0 disables the scaler — useful for users who
    want the legacy strict floor behaviour."""
    wd = _make_watchdog(
        {"watchdog_tts_timeout_s": 15, "watchdog_tts_per_word_s": 0.0},
    )
    assert wd.effective_tts_budget_s(0) == 15.0
    assert wd.effective_tts_budget_s(50) == 15.0


def test_effective_budget_uses_active_word_count_by_default() -> None:
    """When no word count is passed in, the helper reads the count
    captured at TTS-start (so the watchdog loop doesn't have to plumb
    text through every iteration)."""
    wd = _make_watchdog(
        {"watchdog_tts_timeout_s": 15, "watchdog_tts_per_word_s": 0.5},
    )
    wd._tts_active_word_count = 40
    assert wd.effective_tts_budget_s() == 20.0
    wd._tts_active_word_count = 0
    assert wd.effective_tts_budget_s() == 15.0


# ---------------------------------------------------------------------------
# _on_tts_started captures the word count
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_tts_started_records_word_count() -> None:
    wd = _make_watchdog(
        {"watchdog_tts_timeout_s": 15, "watchdog_tts_per_word_s": 0.5},
    )
    text = "It is twenty-two degrees and partly cloudy in your area."
    await wd._on_tts_started(text=text, is_first=True)
    assert wd._tts_active_word_count == RuntimeWatchdog._count_tts_words(text)
    # Effective budget reflects the captured words.
    assert wd.effective_tts_budget_s() >= wd._tts_s


@pytest.mark.asyncio
async def test_on_tts_complete_resets_word_count() -> None:
    wd = _make_watchdog(
        {"watchdog_tts_timeout_s": 15, "watchdog_tts_per_word_s": 0.5},
    )
    await wd._on_tts_started(text="forty word reply", is_first=True)
    assert wd._tts_active_word_count > 0
    await wd._on_tts_complete()
    assert wd._tts_active_word_count == 0


# ---------------------------------------------------------------------------
# Config schema acceptance
# ---------------------------------------------------------------------------


def test_settings_json_has_dynamic_budget_keys() -> None:
    """Config must ship the new keys with sensible defaults so the
    dynamic scaler is engaged out of the box."""
    import json
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "config" / "settings.json").read_text())
    perf = cfg.get("performance", {})
    assert "watchdog_tts_per_word_s" in perf
    assert "watchdog_tts_max_dynamic_s" in perf
    assert perf["watchdog_tts_per_word_s"] >= 0
    assert perf["watchdog_tts_max_dynamic_s"] >= perf["watchdog_tts_timeout_s"]


def test_schema_accepts_dynamic_budget_keys() -> None:
    from core.config_schema import validate_config

    base = {
        "performance": {
            "watchdog_tts_timeout_s": 15,
            "watchdog_tts_per_word_s": 0.5,
            "watchdog_tts_max_dynamic_s": 45,
        },
    }
    validate_config(base)  # must not raise
