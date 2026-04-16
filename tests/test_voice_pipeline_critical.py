"""
Critical voice pipeline tests -- the most important path in ATOM.

These tests verify the core voice pipeline doesn't regress:
  1. Event bus handles both sync and async handlers
  2. STT watchdog resets timers on listening transition
  3. STT watchdog preserves partial text before restart
  4. Wake word filter detects "atom" in text
  5. Voice pipeline builds without crashing
"""

from __future__ import annotations

import asyncio
import time
import pytest


class FakeBus:
    """Minimal event bus for testing."""

    def __init__(self):
        self._handlers: dict[str, list] = {}
        self._emitted: list[tuple[str, dict]] = []

    def on(self, event: str, handler):
        self._handlers.setdefault(event, []).append(handler)

    def emit(self, event: str, **data):
        self._emitted.append((event, data))

    def emit_fast(self, event: str, **data):
        self._emitted.append((event, data))

    @property
    def emitted_events(self):
        return [(e, d) for e, d in self._emitted]


# ── Test 1: AsyncEventBus handles sync handlers ──

@pytest.mark.asyncio
async def test_event_bus_handles_sync_handler():
    """Sync handlers should not crash the event bus."""
    from core.async_event_bus import AsyncEventBus

    bus = AsyncEventBus()
    results = []

    def sync_handler(**kw):
        results.append(kw.get("text", ""))

    bus.on("test_event", sync_handler)
    bus.start()
    bus.emit("test_event", text="hello")
    await asyncio.sleep(0.3)
    await bus.stop()
    assert "hello" in results, f"Sync handler not called, results={results}"


@pytest.mark.asyncio
async def test_event_bus_handles_async_handler():
    """Async handlers should work as expected."""
    from core.async_event_bus import AsyncEventBus

    bus = AsyncEventBus()
    results = []

    async def async_handler(**kw):
        results.append(kw.get("text", ""))

    bus.on("test_event", async_handler)
    bus.start()
    bus.emit("test_event", text="world")
    await asyncio.sleep(0.3)
    await bus.stop()
    assert "world" in results, f"Async handler not called, results={results}"


# ── Test 2: STT Watchdog timer reset ──

@pytest.mark.asyncio
async def test_watchdog_resets_on_listening_transition():
    """Watchdog timers should reset when STT transitions to listening."""
    from voice.stt_watchdog import STTWatchdog

    bus = FakeBus()
    wd = STTWatchdog(bus)

    old_time = wd._last_partial_time
    wd._last_partial_time = old_time - 30.0

    class FakeSTT:
        _listening = False
        _running_async = True
        _tap_buffer_count = 0
        _last_audio_rms_db = -96.0
        _last_speech_candidate_time = 0.0
        _last_partial = ""

    stt = FakeSTT()
    wd.attach_stt(stt)

    stt._listening = True
    await wd._check_health()

    assert time.monotonic() - wd._last_partial_time < 2.0, \
        "Watchdog did not reset timers on listening transition"


# ── Test 3: Watchdog preserves partial text ──

@pytest.mark.asyncio
async def test_watchdog_salvages_partial_before_restart():
    """When restarting, watchdog should emit last partial text."""
    bus = FakeBus()

    from voice.stt_watchdog import STTWatchdog
    wd = STTWatchdog(bus)

    class FakeSTT:
        _listening = True
        _running_async = True
        _tap_buffer_count = 100
        _last_audio_rms_db = -30.0
        _last_speech_candidate_time = time.monotonic()
        _last_partial = "hey atom what time is it"
        _callback_starvation_count = 0
        _native_requires_on_device = True

        def _on_recognition_starvation(self):
            self._callback_starvation_count += 1

        def _restart_recognition_chain(self):
            pass

    stt = FakeSTT()
    wd.attach_stt(stt)
    wd._was_listening = True
    wd._last_partial_time = time.monotonic() - 20.0
    wd._last_tap_count = 50

    await wd._check_health()

    salvaged = [e for e, d in bus._emitted if e == "speech_partial"]
    assert len(salvaged) > 0, "Watchdog did not salvage partial text"
    assert bus._emitted[0][1]["text"] == "hey atom what time is it"


# ── Test 4: Wake word filter ──

def test_wake_word_filter_detects_atom():
    """WakeWordFilter should detect 'atom' in text."""
    from voice.listening_modes import WakeWordFilter

    wf = WakeWordFilter(cooldown_s=0.0)
    result = wf.check("hey atom")
    assert result is not None, "WakeWordFilter did not detect 'hey atom'"

    wf2 = WakeWordFilter(cooldown_s=0.0)
    result2 = wf2.check("tell me atom")
    assert result2 is not None, "WakeWordFilter did not detect trailing 'atom'"


def test_wake_word_filter_ignores_non_wake():
    """WakeWordFilter should not trigger on random text."""
    from voice.listening_modes import WakeWordFilter

    wf = WakeWordFilter(cooldown_s=0.0)
    result = wf.check("what time is it")
    assert result is None, "WakeWordFilter falsely triggered"


# ── Test 5: Voice pipeline builds ──

def test_voice_pipeline_imports():
    """Voice pipeline module should import without errors."""
    from voice.voice_pipeline import VoicePipeline
    assert VoicePipeline is not None
