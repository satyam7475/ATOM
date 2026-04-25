"""Regression tests for Sprint A5 -- STT watchdog breaker UX.

Pins:

* Breaker cooldown is 60s (was 300s), per the Sprint A5 plan.
* When the breaker first opens, an audible ``tts_say`` is emitted so
  the user knows ATOM has stopped listening.
* When the breaker closes again, a recovery announcement fires.
* No spurious announcements while the breaker is closed/healthy.
"""

from __future__ import annotations

import time

import pytest

import voice.stt_watchdog as watchdog_mod
from voice.stt_watchdog import STTWatchdog


class _RecordingBus:
    """Minimal bus stub: collects every (event, kwargs) emit call."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []
        # subscriptions accepted for ``bus.on(...)`` -- not invoked here
        self.subscriptions: list[tuple[str, object]] = []

    def emit_fast(self, event: str, **kw) -> None:
        self.events.append((event, kw))

    def emit_long(self, event: str, **kw) -> None:
        self.events.append((event, kw))

    def on(self, event: str, handler) -> None:
        self.subscriptions.append((event, handler))


def _ts_say_payloads(bus: _RecordingBus) -> list[str]:
    return [kw.get("text", "") for evt, kw in bus.events if evt == "tts_say"]


# ── cooldown tightened (Sprint K: 60s → 30s) ────────────────────────


def test_restart_window_is_30s_after_sprint_k():
    """Sprint K: shrunk window further from 60s → 30s and lowered the
    per-window restart budget to 3, so the breaker reopens faster
    after a transient SFSpeechRecognizer hiccup."""
    assert watchdog_mod._RESTART_WINDOW_S == 30.0
    assert watchdog_mod._MAX_RESTARTS_PER_WINDOW == 3


# ── breaker-open announcement ───────────────────────────────────────


def test_breaker_open_emits_tts_say_once():
    bus = _RecordingBus()
    wd = STTWatchdog(bus)
    # Pre-fill the restart history so the next _can_restart() trips
    # the breaker immediately.
    now = time.monotonic()
    wd._restart_times = [now - 1.0] * watchdog_mod._MAX_RESTARTS_PER_WINDOW

    assert wd._can_restart() is False
    payloads = _ts_say_payloads(bus)
    assert len(payloads) == 1
    assert "STT recovering" in payloads[0]
    assert "Boss" in payloads[0]


def test_breaker_open_announcement_is_idempotent():
    """While the breaker stays open we must not spam the speaker
    every check tick."""
    bus = _RecordingBus()
    wd = STTWatchdog(bus)
    now = time.monotonic()
    wd._restart_times = [now - 1.0] * watchdog_mod._MAX_RESTARTS_PER_WINDOW

    for _ in range(5):
        wd._can_restart()
    payloads = _ts_say_payloads(bus)
    assert payloads == ["STT recovering, give me a moment, Boss."]


def test_breaker_recovery_announces_listening_again():
    bus = _RecordingBus()
    wd = STTWatchdog(bus)
    now = time.monotonic()
    wd._restart_times = [now - 1.0] * watchdog_mod._MAX_RESTARTS_PER_WINDOW
    assert wd._can_restart() is False

    # Age the window past the cooldown so the next call closes the
    # breaker and emits the recovery announcement.
    wd._restart_times = [
        t - (watchdog_mod._RESTART_WINDOW_S + 1.0)
        for t in wd._restart_times
    ]
    assert wd._can_restart() is True

    payloads = _ts_say_payloads(bus)
    assert any("Listening again" in p for p in payloads)


def test_no_announcement_when_breaker_never_opened():
    bus = _RecordingBus()
    wd = STTWatchdog(bus)
    for _ in range(5):
        wd._can_restart()
    assert _ts_say_payloads(bus) == []


def test_speak_breaker_open_swallows_bus_exceptions():
    class _BrokenBus:
        def emit_fast(self, *a, **kw):
            raise RuntimeError("bus down")

        def emit_long(self, *a, **kw):
            raise RuntimeError("bus down")

        def on(self, *a, **kw):
            pass

    wd = STTWatchdog(_BrokenBus())
    # Must not raise
    wd._speak_breaker_open()
    wd._speak_breaker_recovered()
