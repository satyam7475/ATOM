"""ATOM — Sprint Ω.13: WhisperKit self-heal regression tests.

Locks in the watchdog wiring added in Sprint Ω.13:

  1. ``_record_transcribe_failure`` increments the counter, caches the
     last failed *final* audio buffer, emits the
     ``whisperkit_serve_unresponsive`` bus event, and schedules
     :meth:`WhisperKitSTT.restart_serve_async` on the asyncio loop once
     the configured threshold has been hit.
  2. ``restart_serve_async`` kills the wedged ``whisperkit-cli serve``
     subprocess, re-launches it via ``_maybe_start_serve`` +
     ``_wait_for_serve_ready``, retries the cached audio once, resets
     the failure counter, and emits ``whisperkit_serve_restarted``.
  3. The per-hour restart cap refuses to relaunch and announces a
     spoken recovery prompt once the budget is exhausted.

The tests bypass ``WhisperKitSTT.__init__`` (which spins up the
subprocess + audio stream) by constructing the instance via
``__new__`` and seeding only the attributes the methods under test
read. Real subprocess management is mocked.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class _BusStub:
    """Minimal bus that records emits without dispatching."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def emit(self, evt: str, **kw: Any) -> None:
        self.events.append((evt, kw))

    def emit_fast(self, evt: str, **kw: Any) -> None:
        self.events.append((evt, kw))

    def emit_long(self, evt: str, **kw: Any) -> None:
        self.events.append((evt, kw))

    def on(self, *_a: Any, **_k: Any) -> None:
        pass

    def find(self, evt: str) -> list[dict[str, Any]]:
        return [d for e, d in self.events if e == evt]


def _make_stt(
    bus: _BusStub,
    *,
    failure_threshold: int = 2,
    restart_max: int = 5,
    serve_pid: int = 12345,
):
    """Build a ``WhisperKitSTT`` instance bypassing ``__init__``."""
    from voice.stt_whisperkit import WhisperKitSTT
    stt = WhisperKitSTT.__new__(WhisperKitSTT)
    stt._bus = bus
    stt._loop = None
    stt._listening = False
    stt._serve_proc = SimpleNamespace(pid=serve_pid)
    stt._restart_in_flight = False
    stt._restart_count_hour = []
    stt._restart_max_per_hour = restart_max
    stt._last_failed_audio = None
    stt._transcribe_failure_count = 0
    stt._transcribe_failure_threshold = failure_threshold
    stt._consecutive_health_probe_failures = 0
    stt._voice_module_restart_announced_at = 0.0
    stt._stt_io = None
    return stt


# ── _record_transcribe_failure ───────────────────────────────────────


def test_record_transcribe_failure_below_threshold_caches_audio() -> None:
    bus = _BusStub()
    stt = _make_stt(bus, failure_threshold=2)
    audio = b"final-audio-buffer-v1"

    stt._record_transcribe_failure(
        audio, partial=False, reason="HTTPError 500",
    )

    assert stt._transcribe_failure_count == 1
    assert stt._last_failed_audio is audio, (
        "final transcribe failure must cache audio for one-shot retry"
    )
    unresponsive = bus.find("whisperkit_serve_unresponsive")
    assert unresponsive, (
        "expected whisperkit_serve_unresponsive bus event after failure"
    )
    assert unresponsive[0]["failures"] == 1


def test_record_transcribe_failure_partial_does_not_cache_audio() -> None:
    """Partial transcribes are too noisy to retry — they must increment
    the failure counter but never become the cached retry payload."""
    bus = _BusStub()
    stt = _make_stt(bus, failure_threshold=5)

    stt._record_transcribe_failure(
        "partial-audio", partial=True, reason="x",
    )

    assert stt._transcribe_failure_count == 1
    assert stt._last_failed_audio is None


def test_record_transcribe_failure_at_threshold_schedules_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = _BusStub()
    stt = _make_stt(bus, failure_threshold=2)
    fake_loop = SimpleNamespace(is_closed=lambda: False)
    stt._loop = fake_loop

    scheduled: list[Any] = []

    def _fake_run(coro: Any, lp: Any) -> Any:
        scheduled.append((coro, lp))
        try:
            coro.close()  # suppress "coroutine never awaited"
        except Exception:
            pass
        return SimpleNamespace(cancel=lambda: None)

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", _fake_run)

    stt._record_transcribe_failure(b"a1", partial=False, reason="t1")
    assert stt._transcribe_failure_count == 1
    assert not scheduled, "should not schedule restart below threshold"

    stt._record_transcribe_failure(b"a2", partial=False, reason="t2")
    assert stt._transcribe_failure_count == 2
    assert len(scheduled) == 1, (
        "expected exactly one restart_serve_async scheduled on threshold"
    )


# ── restart_serve_async ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_restart_serve_async_kills_relaunches_retries_resets(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Full recovery flow: SIGTERM → relaunch → retry cached audio →
    reset counter. Verifies the prescribed STT_WATCHDOG log lines and
    the ``whisperkit_serve_restarted`` bus event."""
    bus = _BusStub()
    stt = _make_stt(bus, failure_threshold=2)
    stt._loop = asyncio.get_running_loop()
    stt._listening = True
    stt._last_failed_audio = b"cached-audio-bytes"
    stt._transcribe_failure_count = 5
    stt._stt_io = ThreadPoolExecutor(max_workers=1)
    new_proc = SimpleNamespace(pid=98765)

    kill_calls = MagicMock()

    def _replace_proc() -> None:
        stt._serve_proc = new_proc

    start_calls = MagicMock(side_effect=_replace_proc)
    ready_calls = MagicMock()
    transcribe_calls = MagicMock(return_value="hello atom")
    emit_final_calls = MagicMock()

    monkeypatch.setattr(stt, "_kill_serve_process", kill_calls)
    monkeypatch.setattr(stt, "_maybe_start_serve", start_calls)
    monkeypatch.setattr(stt, "_wait_for_serve_ready", ready_calls)
    monkeypatch.setattr(stt, "_http_transcribe", transcribe_calls)
    monkeypatch.setattr(stt, "_emit_final", emit_final_calls)

    caplog.set_level(logging.INFO, logger="atom.stt_whisperkit")

    ok = await stt.restart_serve_async("test_reason")

    assert ok is True, "restart should report success when relaunch lands"
    assert kill_calls.call_count == 1, "expected one SIGTERM/SIGKILL pass"
    assert start_calls.call_count == 1
    assert ready_calls.call_count == 1
    assert transcribe_calls.called, (
        "cached audio must be retried once after relaunch"
    )
    assert emit_final_calls.called, (
        "retry transcript must be funnelled through _emit_final"
    )
    assert stt._transcribe_failure_count == 0
    assert stt._last_failed_audio is None

    restarted = bus.find("whisperkit_serve_restarted")
    assert restarted, (
        "whisperkit_serve_restarted bus event must fire after success"
    )
    assert restarted[0]["pid"] == 98765

    msgs = [r.getMessage() for r in caplog.records]
    assert any("STT_WATCHDOG: restarting whisperkit" in m for m in msgs), (
        f"missing 'restarting whisperkit' log; got {msgs!r}"
    )
    assert any("STT_WATCHDOG: whisperkit healthy" in m for m in msgs), (
        f"missing 'whisperkit healthy' log; got {msgs!r}"
    )
    assert any(
        "STT_WATCHDOG: retry transcription success" in m for m in msgs
    ), f"missing retry-success log; got {msgs!r}"

    stt._stt_io.shutdown(wait=True)


@pytest.mark.asyncio
async def test_restart_serve_async_respects_per_hour_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the rolling 1-hour restart window is full, the watchdog
    must refuse the relaunch, leave the subprocess alone, and trigger
    the spoken recovery prompt instead."""
    bus = _BusStub()
    stt = _make_stt(bus, restart_max=1)
    stt._loop = asyncio.get_running_loop()
    stt._restart_count_hour = [time.monotonic()]

    kill = MagicMock()
    start = MagicMock()
    ready = MagicMock()
    monkeypatch.setattr(stt, "_kill_serve_process", kill)
    monkeypatch.setattr(stt, "_maybe_start_serve", start)
    monkeypatch.setattr(stt, "_wait_for_serve_ready", ready)

    ok = await stt.restart_serve_async("excess")

    assert ok is False
    assert kill.call_count == 0, "must not kill when cap is hit"
    assert start.call_count == 0, "must not relaunch when cap is hit"
    speak = bus.find("system_speak")
    assert speak, (
        "expected system_speak (recovery prompt) when restart cap hit"
    )
    assert "Voice module restarted" in speak[0]["text"]


@pytest.mark.asyncio
async def test_restart_serve_async_idempotent_under_concurrent_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two concurrent restart triggers must serialise — only one runs
    the kill/relaunch pair."""
    bus = _BusStub()
    stt = _make_stt(bus, restart_max=10)
    stt._loop = asyncio.get_running_loop()

    new_proc = SimpleNamespace(pid=4242)
    kill = MagicMock()

    def _replace_proc() -> None:
        stt._serve_proc = new_proc

    start = MagicMock(side_effect=_replace_proc)
    ready = MagicMock()
    monkeypatch.setattr(stt, "_kill_serve_process", kill)
    monkeypatch.setattr(stt, "_maybe_start_serve", start)
    monkeypatch.setattr(stt, "_wait_for_serve_ready", ready)

    results = await asyncio.gather(
        stt.restart_serve_async("first"),
        stt.restart_serve_async("second"),
    )

    assert any(results), "at least one restart should report success"
    assert kill.call_count <= 1, (
        "kill must run at most once across concurrent restarts"
    )
    assert start.call_count <= 1
