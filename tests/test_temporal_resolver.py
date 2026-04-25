"""Regression tests for Sprint N3 -- :func:`temporal_resolver.resolve`."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from core.memory.temporal_resolver import resolve


# Use a fixed "now" so weekday math is deterministic. 2026-04-25 is a
# Saturday — picked on purpose so "last Tuesday" lands on a different
# week than "tuesday" alone.
NOW = datetime(2026, 4, 25, 15, 30)


def _label(out) -> str:
    return getattr(out, "label", "")


def test_today_window() -> None:
    r = resolve("today", now=NOW)
    assert r is not None
    assert r.label == "today"
    assert r.start == NOW.replace(hour=0, minute=0, second=0, microsecond=0)
    assert r.end == NOW


def test_yesterday_window_is_full_day() -> None:
    r = resolve("yesterday", now=NOW)
    assert r is not None
    assert r.label == "yesterday"
    assert r.start.date() == (NOW - timedelta(days=1)).date()
    assert r.end.date() == (NOW - timedelta(days=1)).date()
    assert r.start.hour == 0
    assert r.end.hour == 23


def test_last_tuesday_resolves_to_previous_week() -> None:
    r = resolve("last Tuesday", now=NOW)
    assert r is not None
    # Last Tuesday before Sat 2026-04-25 is 2026-04-21 (4 days back).
    assert r.start.date() == datetime(2026, 4, 21).date()
    assert r.label.lower().startswith("last")


def test_bare_weekday_resolves_to_most_recent_past_instance() -> None:
    r = resolve("monday", now=NOW)  # Sat -> previous Monday is 2026-04-20
    assert r is not None
    assert r.start.date() == datetime(2026, 4, 20).date()


@pytest.mark.parametrize(
    "phrase, expected_start_hour",
    [
        ("this morning", 5),
        ("this afternoon", 12),
        ("tonight", 19),         # tonight = 19-23 in our mapping
        ("this evening", 17),    # evening = 17-21
    ],
)
def test_dayparts_today(phrase: str, expected_start_hour: int) -> None:
    r = resolve(phrase, now=NOW)
    assert r is not None
    assert r.start.date() == NOW.date()
    assert r.start.hour == expected_start_hour


def test_yesterday_morning_full_window() -> None:
    r = resolve("yesterday morning", now=NOW)
    assert r is not None
    yday = (NOW - timedelta(days=1)).date()
    assert r.start.date() == yday
    assert r.start.hour == 5
    assert r.end.date() == yday
    assert r.end.hour == 11 or r.end.hour == 12


def test_x_minutes_ago_centers_on_anchor() -> None:
    r = resolve("30 minutes ago", now=NOW)
    assert r is not None
    anchor = NOW - timedelta(minutes=30)
    assert r.start <= anchor <= r.end


def test_past_n_unit_phrasing() -> None:
    r = resolve("the past 2 hours", now=NOW)
    assert r is not None
    assert r.end == NOW
    assert (NOW - r.start).seconds // 3600 == 2


def test_clock_time_today_when_in_past() -> None:
    r = resolve("10am today", now=NOW)
    assert r is not None
    assert r.start.date() == NOW.date()
    assert r.start.hour == 9 or r.start.hour == 10


def test_clock_time_yesterday_when_dayref_present() -> None:
    r = resolve("5pm yesterday", now=NOW)
    assert r is not None
    yday = (NOW - timedelta(days=1)).date()
    assert r.start.date() == yday
    assert r.start.hour in (16, 17)  # ±30 minute window


def test_unknown_phrase_returns_none() -> None:
    assert resolve("the second Tuesday of every leap year", now=NOW) is None
    assert resolve("", now=NOW) is None


def test_since_lunch_uses_lunch_window() -> None:
    r = resolve("since lunch", now=NOW)
    assert r is not None
    assert r.start.hour == 12
    assert r.end == NOW


def test_last_week_full_week_window() -> None:
    r = resolve("last week", now=NOW)
    assert r is not None
    assert r.label == "last week"
    # NOW is Saturday; last week is the previous Monday-Sunday block.
    assert r.start.date() <= datetime(2026, 4, 19).date()
    assert r.end.date() >= datetime(2026, 4, 19).date()
