"""Regression tests for Sprint N3 -- :class:`OmniRecall`."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from core.memory.omni_recall import (
    OmniRecall,
    OmniRecallConfig,
    RecallReport,
)


# ── stubs ──────────────────────────────────────────────────────────────


class _StubTimeline:
    def __init__(self, events: list[Any]) -> None:
        self.events = events

    def search_user_queries(
        self, keyword: str, *, since_ts: float | None = None,
        until_ts: float | None = None, limit: int = 20,
    ) -> list[Any]:
        kw = (keyword or "").lower()
        out: list[Any] = []
        for ev in self.events:
            ts = float(getattr(ev, "timestamp", 0.0))
            if since_ts is not None and ts < since_ts:
                continue
            if until_ts is not None and ts > until_ts:
                continue
            text = (ev.data.get("text") or "").lower()
            if kw and kw not in text:
                continue
            out.append(ev)
        return out[-limit:]


class _StubConversation:
    def __init__(self, turns: list[Any]) -> None:
        self._turns = turns

    def all_turns(self) -> list[Any]:
        return list(self._turns)


class _StubScreen:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def query(
        self, *, since_ts: float | None = None, until_ts: float | None = None,
        text_contains: str | None = None, limit: int = 50,
        app: str | None = None,
    ) -> list[dict[str, Any]]:
        out = []
        for r in self._rows:
            ts = float(r.get("ts") or 0.0)
            if since_ts is not None and ts < since_ts:
                continue
            if until_ts is not None and ts > until_ts:
                continue
            if (
                text_contains
                and text_contains.lower() not in (r.get("text") or "").lower()
            ):
                continue
            out.append(r)
        return out[:limit]


def _make_event(text: str, ts: float) -> Any:
    ev = SimpleNamespace()
    ev.type = "user_query"
    ev.timestamp = ts
    ev.data = {"text": text}
    return ev


def _make_turn(text: str, ts: float) -> Any:
    return SimpleNamespace(user_text=text, timestamp=ts)


# ── tests ──────────────────────────────────────────────────────────────


def test_recall_uses_default_lookback_window_when_when_blank() -> None:
    now = time.time()
    timeline = _StubTimeline([
        _make_event("ATOM is awesome", now - 3600),
    ])
    rec = OmniRecall(
        timeline_memory=timeline,
        conversation_memory=None,
        screen_perception_loop=None,
        config=OmniRecallConfig(default_lookback_hours=2.0),
    )
    report = rec.recall(query="atom")
    assert report.used_default_window
    assert report.window is not None
    assert "the past 2h" in report.window.label
    assert len(report.hits) == 1


def test_recall_returns_hits_from_all_sources() -> None:
    now = time.time()
    timeline = _StubTimeline([
        _make_event("Boss asked about gemini", now - 1800),
    ])
    convo = _StubConversation([
        _make_turn("can you check the weather", now - 600),
    ])
    screen = _StubScreen([
        {"ts": now - 300, "app": "Slack", "text": "team standup notes",
         "tokens": 4},
    ])
    rec = OmniRecall(
        timeline_memory=timeline,
        conversation_memory=convo,
        screen_perception_loop=screen,
    )
    report = rec.recall(query="", when="the past 2 hours")
    by_src = report.by_source()
    assert "timeline" in by_src
    assert "conversation" in by_src
    assert "screen" in by_src


def test_recall_window_filters_out_older_events() -> None:
    now = time.time()
    timeline = _StubTimeline([
        _make_event("very old", now - 86400 * 10),
        _make_event("recent", now - 60),
    ])
    rec = OmniRecall(
        timeline_memory=timeline,
        config=OmniRecallConfig(default_lookback_hours=1.0),
    )
    report = rec.recall(query="")
    assert all(h.text == "recent" for h in report.hits)


def test_recall_speak_summary_when_empty() -> None:
    rec = OmniRecall()
    report = rec.recall(query="non-existent")
    text = report.speak()
    assert "Nothing" in text


def test_recall_speak_summary_when_hits_present() -> None:
    now = time.time()
    timeline = _StubTimeline([_make_event("Boss asked X", now - 30)])
    rec = OmniRecall(timeline_memory=timeline)
    report = rec.recall(query="")
    text = report.speak()
    assert "Found" in text
    assert "thing" in text


def test_recall_respects_explicit_sources() -> None:
    now = time.time()
    timeline = _StubTimeline([_make_event("alpha", now - 30)])
    convo = _StubConversation([_make_turn("beta", now - 30)])
    rec = OmniRecall(timeline_memory=timeline, conversation_memory=convo)
    report = rec.recall(query="", sources=["conversation"])
    assert all(h.source == "conversation" for h in report.hits)


def test_recall_orders_hits_most_recent_first() -> None:
    now = time.time()
    timeline = _StubTimeline([
        _make_event("first", now - 3000),
        _make_event("second", now - 1500),
        _make_event("third", now - 100),
    ])
    rec = OmniRecall(timeline_memory=timeline)
    report = rec.recall(query="")
    assert [h.text for h in report.hits] == ["third", "second", "first"]
