"""ATOM Sprint N3 -- temporal phrase → (start_ts, end_ts) resolver.

The existing memory recall path supports phrases like "yesterday" /
"today" / "last week" via simple regex heuristics. Friday-class
behaviour needs much richer time references that Boss naturally uses:

    * "last Tuesday"
    * "this morning" / "this afternoon" / "tonight"
    * "5 pm yesterday" / "10 in the morning"
    * "two hours ago" / "a couple of days ago"
    * "since lunch" / "before lunch"
    * "the past 30 minutes"

This module is a small, dependency-free resolver. It returns a
``TemporalRange`` with start/end Unix timestamps and a label so the
caller can both query memory *and* tell Boss what window it used.

We intentionally avoid `dateparser` to keep the install footprint
small and the latency sub-millisecond. The patterns cover Boss's
day-to-day phrasing -- anything truly unusual ("the last Friday in
Q3") falls through to ``None`` and the caller can ask the LLM.

Owner: Boss (Satyam).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dt_time
from typing import Optional

logger = logging.getLogger("atom.memory.temporal")


@dataclass(slots=True)
class TemporalRange:
    """A resolved time window with a human-readable label."""

    start: datetime
    end: datetime
    label: str

    @property
    def start_ts(self) -> float:
        return self.start.timestamp()

    @property
    def end_ts(self) -> float:
        return self.end.timestamp()

    @property
    def span_seconds(self) -> float:
        return max(0.0, self.end_ts - self.start_ts)


_WEEKDAYS = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

_NUM_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "couple": 2, "few": 3,
    "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "twelve": 12, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty": 40, "forty-five": 45,
    "fifty": 50, "sixty": 60, "ninety": 90,
}

_DAYPART_WINDOWS = {
    "morning": (5, 12),
    "noon": (12, 13),
    "afternoon": (12, 17),
    "evening": (17, 21),
    "tonight": (19, 23),
    "night": (20, 24),
    "midnight": (0, 1),
    "lunch": (12, 14),
    "breakfast": (6, 10),
    "dinner": (18, 21),
}


def _parse_int_word(token: str) -> int | None:
    token = token.strip().lower()
    if not token:
        return None
    if token.isdigit():
        return int(token)
    return _NUM_WORDS.get(token)


def _start_of(d: datetime) -> datetime:
    return d.replace(hour=0, minute=0, second=0, microsecond=0)


def _end_of(d: datetime) -> datetime:
    return d.replace(hour=23, minute=59, second=59, microsecond=999_999)


# ── public resolver ────────────────────────────────────────────────────


def resolve(
    phrase: str,
    *,
    now: datetime | None = None,
) -> Optional[TemporalRange]:
    """Resolve a free-form temporal phrase.

    Returns ``None`` if no rule matched (caller may default to e.g.
    last 24h).
    """
    if not phrase or not phrase.strip():
        return None
    now = now or datetime.now()
    text = phrase.strip().lower()

    # ── handle "since X" up-front -- "since lunch" means [lunch_start, now]
    # not the lunch *window* itself, so we must intercept it before the
    # generic prefix stripping below.
    m_since = re.match(r"since\s+(?P<rest>.+)", text)
    if m_since:
        rest = m_since.group("rest").strip()
        if rest in _DAYPART_WINDOWS:
            h0, _ = _DAYPART_WINDOWS[rest]
            start = now.replace(hour=h0, minute=0, second=0, microsecond=0)
            return TemporalRange(start, now, f"since {rest}")
        clock = _parse_clock(rest, now)
        if clock is not None:
            return TemporalRange(clock, now, f"since {rest}")

    # Strip surrounding noise.
    text = re.sub(r"^(at|on|in)\s+", "", text)

    # ── instantaneous keywords ────────────────────────────────────
    if text in {"now", "right now", "this second"}:
        return TemporalRange(now - timedelta(minutes=1), now, "the past minute")

    if text in {"today", "today so far"}:
        return TemporalRange(_start_of(now), now, "today")

    if text in {"yesterday"}:
        d = now - timedelta(days=1)
        return TemporalRange(_start_of(d), _end_of(d), "yesterday")

    if text in {"tomorrow"}:
        d = now + timedelta(days=1)
        return TemporalRange(_start_of(d), _end_of(d), "tomorrow")

    if text in {"this week"}:
        start = _start_of(now - timedelta(days=now.weekday()))
        return TemporalRange(start, now, "this week")

    if text in {"last week"}:
        end_day = now - timedelta(days=now.weekday() + 1)
        start_day = end_day - timedelta(days=6)
        return TemporalRange(
            _start_of(start_day), _end_of(end_day), "last week",
        )

    if text in {"this month"}:
        start = now.replace(day=1, hour=0, minute=0,
                            second=0, microsecond=0)
        return TemporalRange(start, now, "this month")

    # ── "(this|past) X hours/minutes/days" ────────────────────────
    m = re.match(
        r"(?:the\s+)?(?:past|last)\s+(?P<n>\w+(?:-\w+)?)\s+"
        r"(?P<unit>seconds?|minutes?|mins?|hours?|hrs?|hr|days?|"
        r"weeks?|months?)",
        text,
    )
    if m:
        n = _parse_int_word(m.group("n"))
        if n is not None:
            unit = m.group("unit")
            label = f"the past {n} {unit}"
            if unit.startswith("sec"):
                start = now - timedelta(seconds=n)
            elif unit.startswith("min"):
                start = now - timedelta(minutes=n)
            elif unit.startswith("hour") or unit.startswith("hr"):
                start = now - timedelta(hours=n)
            elif unit.startswith("day"):
                start = now - timedelta(days=n)
            elif unit.startswith("week"):
                start = now - timedelta(weeks=n)
            elif unit.startswith("month"):
                start = now - timedelta(days=n * 30)
            else:
                return None
            return TemporalRange(start, now, label)

    # ── "X minutes/hours ago" ─────────────────────────────────────
    m = re.match(
        r"(?P<n>\w+(?:-\w+)?)\s+"
        r"(?P<unit>seconds?|minutes?|mins?|hours?|hrs?|hr|days?|"
        r"weeks?|months?)\s+ago",
        text,
    )
    if m:
        n = _parse_int_word(m.group("n"))
        if n is not None:
            unit = m.group("unit")
            if unit.startswith("sec"):
                delta = timedelta(seconds=n)
            elif unit.startswith("min"):
                delta = timedelta(minutes=n)
            elif unit.startswith("hour") or unit.startswith("hr"):
                delta = timedelta(hours=n)
            elif unit.startswith("day"):
                delta = timedelta(days=n)
            elif unit.startswith("week"):
                delta = timedelta(weeks=n)
            elif unit.startswith("month"):
                delta = timedelta(days=n * 30)
            else:
                return None
            anchor = now - delta
            window = max(timedelta(seconds=60), delta * 0.05)
            return TemporalRange(
                anchor - window, anchor + window, f"{n} {unit} ago",
            )

    # ── "this morning / this afternoon / tonight / lunch" ─────────
    for keyword, (h0, h1) in _DAYPART_WINDOWS.items():
        if re.fullmatch(rf"(this\s+)?{keyword}", text):
            day = now
            start = day.replace(hour=h0, minute=0, second=0, microsecond=0)
            end = day.replace(
                hour=min(h1, 23), minute=59, second=59, microsecond=0,
            )
            if end > now:
                end = now
            return TemporalRange(start, end, f"this {keyword}")

        if re.fullmatch(rf"yesterday\s+{keyword}", text):
            day = now - timedelta(days=1)
            start = day.replace(hour=h0, minute=0, second=0, microsecond=0)
            end = day.replace(
                hour=min(h1, 23), minute=59, second=59, microsecond=0,
            )
            return TemporalRange(start, end, f"yesterday {keyword}")

    # ── "5pm yesterday" / "10am today" / "5 pm" ───────────────────
    clock = _parse_clock(text, now)
    if clock is not None:
        window = timedelta(minutes=30)
        return TemporalRange(
            clock - window, clock + window, f"around {text}",
        )

    # ── "(last|this) <weekday>" ───────────────────────────────────
    m = re.match(
        r"(?P<rel>last|this|previous|past|next)\s+(?P<wd>\w+)", text,
    )
    if m:
        wd_name = m.group("wd").lower()
        if wd_name in _WEEKDAYS:
            target = _WEEKDAYS[wd_name]
            current = now.weekday()
            if m.group("rel") in {"last", "previous", "past"}:
                delta = (current - target) % 7
                if delta == 0:
                    delta = 7
                d = now - timedelta(days=delta)
            elif m.group("rel") == "next":
                delta = (target - current) % 7
                if delta == 0:
                    delta = 7
                d = now + timedelta(days=delta)
            else:  # "this"
                delta = (target - current) % 7
                d = now + timedelta(days=delta)
            return TemporalRange(
                _start_of(d), _end_of(d),
                f"{m.group('rel')} {wd_name.capitalize()}",
            )

    # Bare weekday "tuesday" -> most recent past instance.
    if text in _WEEKDAYS:
        target = _WEEKDAYS[text]
        current = now.weekday()
        delta = (current - target) % 7
        if delta == 0:
            delta = 7
        d = now - timedelta(days=delta)
        return TemporalRange(
            _start_of(d), _end_of(d), text.capitalize(),
        )

    return None


# ── clock-time parsing helper ─────────────────────────────────────────


_CLOCK_RE = re.compile(
    r"""(?ix)
    (?P<hour>\d{1,2})
    (?::(?P<minute>\d{2}))?
    \s*(?P<period>am|pm|a\.m\.|p\.m\.)?
    (?:\s*(?P<dayref>today|yesterday|tomorrow))?
    """,
)


def _parse_clock(text: str, now: datetime) -> datetime | None:
    m = _CLOCK_RE.search(text)
    if not m:
        return None
    hour = int(m.group("hour"))
    if hour > 23:
        return None
    minute = int(m.group("minute") or 0)
    if minute > 59:
        return None
    period = (m.group("period") or "").replace(".", "").lower()
    if period == "pm" and hour < 12:
        hour += 12
    elif period == "am" and hour == 12:
        hour = 0

    dayref = (m.group("dayref") or "").lower()
    base = now
    if dayref == "yesterday":
        base = now - timedelta(days=1)
    elif dayref == "tomorrow":
        base = now + timedelta(days=1)

    candidate = base.replace(
        hour=hour, minute=minute, second=0, microsecond=0,
    )

    # If user said just "5 pm" with no day, prefer "today"; if that's
    # in the future and feels unlikely (>10 min), pick yesterday.
    if not dayref:
        if candidate > now + timedelta(minutes=15):
            candidate = candidate - timedelta(days=1)

    return candidate


__all__ = ["TemporalRange", "resolve"]
