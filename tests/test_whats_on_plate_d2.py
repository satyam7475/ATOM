"""
ATOM -- Sprint D2 focused tests.

Verifies:
    1. Intent regex matches the canonical "on my plate" phrasings.
    2. The summary generator composes calendar + reminder lines.
    3. Empty-plate fallback runs when both sources are empty.
    4. Missing task_scheduler degrades gracefully.
"""

from __future__ import annotations

from unittest.mock import patch

from core.intent_engine import productivity_intents
from core.proactive.whats_on_plate import generate_plate_summary_sync


class _FakeTask:
    def __init__(self, label: str, human: str = "10 minutes") -> None:
        self.label = label
        self._human = human

    def human_due(self) -> str:
        return self._human


class _FakeScheduler:
    def __init__(self, pending: list[_FakeTask] | None = None) -> None:
        self._pending = pending or []

    def list_pending(self) -> list[_FakeTask]:
        return list(self._pending)


def test_intent_matches_common_phrasings() -> None:
    phrasings = [
        "what's on my plate today",
        "What do I have today?",
        "what's my schedule",
        "tell me about my day",
        "what's coming up today",
        "what does my day look like",
        "my schedule for today",
    ]
    for p in phrasings:
        res = productivity_intents.check(p)
        assert res is not None, f"Expected match for: {p!r}"
        assert res.intent == "whats_on_my_plate"
        assert res.action == "whats_on_my_plate"


def test_intent_ignores_unrelated_text() -> None:
    misses = [
        "what did I ask yesterday about deployment",
        "play some music",
        "what's the weather",
        "shutdown the computer",
    ]
    for p in misses:
        res = productivity_intents.check(p)
        assert res is None, f"Expected NO match for: {p!r} (got {res.intent if res else None})"


def test_summary_with_events_and_reminders() -> None:
    sched = _FakeScheduler([
        _FakeTask("review Anurag's PR", human="5 minutes"),
        _FakeTask("call Mom", human="2 hours"),
    ])
    with patch(
        "core.macos.calendar_today.fetch_today_events_sync",
        return_value=["Standup at 10:00 AM", "Lunch with Priya at 1:00 PM"],
    ):
        summary = generate_plate_summary_sync(task_scheduler=sched)
    assert "Standup" in summary
    assert "Lunch" in summary
    assert "reminders" in summary.lower() or "reminder" in summary.lower()
    assert "Anurag" in summary or "Mom" in summary


def test_summary_empty_plate() -> None:
    sched = _FakeScheduler([])
    with patch(
        "core.macos.calendar_today.fetch_today_events_sync",
        return_value=[],
    ):
        summary = generate_plate_summary_sync(task_scheduler=sched)
    assert "clear" in summary.lower() or "no " in summary.lower()


def test_summary_no_scheduler() -> None:
    with patch(
        "core.macos.calendar_today.fetch_today_events_sync",
        return_value=["Standup at 10:00 AM"],
    ):
        summary = generate_plate_summary_sync(task_scheduler=None)
    assert "Standup" in summary


def test_summary_calendar_failure_degrades() -> None:
    sched = _FakeScheduler([_FakeTask("finish D2 sprint")])

    def _raise(*_a, **_kw):
        raise RuntimeError("simulated osascript failure")

    with patch(
        "core.macos.calendar_today.fetch_today_events_sync",
        side_effect=_raise,
    ):
        summary = generate_plate_summary_sync(task_scheduler=sched)
    assert "D2" in summary or "reminder" in summary.lower()


if __name__ == "__main__":
    test_intent_matches_common_phrasings()
    test_intent_ignores_unrelated_text()
    test_summary_with_events_and_reminders()
    test_summary_empty_plate()
    test_summary_no_scheduler()
    test_summary_calendar_failure_degrades()
    print("[D2] All what's-on-my-plate tests passed.")
