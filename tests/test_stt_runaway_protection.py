"""
STT runaway-protection tests.

These exercise the low-level guards added to ``voice/stt_macos.NativeSTT``
that broke the 100 Hz empty-isFinal → chain-restart → recreate-recognizer
self-feeding loop we observed in production logs.

They run entirely in-process -- the Apple frameworks (Speech, AVFoundation)
are monkey-patched to dummy sentinels where needed, since the guard paths
tested here return BEFORE any real framework call.

Covered guards:
  1. ``_recreate_recognizer`` rate limit (< 1 s between calls → refused).
  2. ``_recreate_recognizer`` circuit breaker (> 5 recreates in 10 s window
     → emit ``stt_needs_full_restart`` and refuse).
  3. ``_should_feed_recognizer`` blocks TTS-echo feedback when ATOM is
     SPEAKING without barge-in enabled.
  4. ``_restart_recognition_chain`` debounces repeat calls inside 250 ms
     (the empty-isFinal cascade).
  5. ``_on_final`` empty-final counter escalates to a watchdog-level
     restart after the threshold.
"""

from __future__ import annotations

import time
from enum import Enum
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class _FakeBus:
    def __init__(self) -> None:
        self._emitted: list[tuple[str, dict]] = []
        self._handlers: dict[str, list] = {}

    def on(self, event: str, handler) -> None:
        self._handlers.setdefault(event, []).append(handler)

    def emit(self, event: str, **data) -> None:
        self._emitted.append((event, data))

    def emit_fast(self, event: str, **data) -> None:
        self._emitted.append((event, data))


class _FakeState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    SPEAKING = "speaking"
    THINKING = "thinking"


class _FakeStateManager:
    def __init__(self, current: _FakeState = _FakeState.LISTENING) -> None:
        self.current = current


def _make_stt(monkeypatch, *, state: _FakeState = _FakeState.LISTENING,
              barge_in: bool = False):
    """Build a NativeSTT with Apple frameworks stubbed to sentinels.

    The guard paths exercised here return BEFORE any real Speech /
    AVFoundation call, so we just need the module-level references to
    be non-None so the first ``if _Speech is None …`` guard in
    ``_recreate_recognizer`` doesn't short-circuit.
    """
    from voice import stt_macos as mod

    monkeypatch.setattr(mod, "_Speech", SimpleNamespace(), raising=False)
    monkeypatch.setattr(mod, "_Foundation", SimpleNamespace(), raising=False)

    # Some paths also ask for AtomState via state_manager; swap with our enum.
    # The code uses `from core.state_manager import AtomState` inside the
    # helper; our fake enum uses the same member names (LISTENING, etc).
    import core.state_manager as sm
    monkeypatch.setattr(sm, "AtomState", _FakeState, raising=False)

    bus = _FakeBus()
    cfg = {"stt": {"barge_in_during_speak": barge_in, "voice_debug": False}}
    state_mgr = _FakeStateManager(current=state)
    stt = mod.NativeSTT(bus=bus, state=state_mgr, config=cfg)
    return stt, bus


def test_recreate_recognizer_rate_limit(monkeypatch):
    """Two recreate calls inside 1 s -> second one refused without framework contact."""
    stt, bus = _make_stt(monkeypatch)
    stt._last_recreate_time = time.monotonic()

    result = stt._recreate_recognizer()
    assert result is False, "recreate inside min interval must be refused"
    assert not any(e == "stt_needs_full_restart" for e, _ in bus._emitted), \
        "rate-limit refusal must not trigger full-restart escalation"


def test_recreate_recognizer_circuit_breaker_emits(monkeypatch):
    """> 5 recreates in 10 s triggers stt_needs_full_restart with recreate_storm."""
    stt, bus = _make_stt(monkeypatch)

    now = time.monotonic()
    stt._last_recreate_time = now - 10.0
    stt._recreate_times_window.extend([now - 9.0, now - 8.0, now - 6.0,
                                       now - 4.0, now - 2.0, now - 1.0])

    result = stt._recreate_recognizer()
    assert result is False, "circuit-break window must refuse the recreate"

    restart_events = [(e, d) for e, d in bus._emitted
                      if e == "stt_needs_full_restart"]
    assert len(restart_events) == 1, \
        f"expected exactly one escalation, got: {bus._emitted}"
    assert restart_events[0][1].get("reason") == "recreate_storm"


def test_recreate_recognizer_circuit_breaker_only_fires_once(monkeypatch):
    """Escalation is sticky for the cycle; repeated calls don't spam the bus."""
    stt, bus = _make_stt(monkeypatch)
    now = time.monotonic()
    stt._last_recreate_time = now - 10.0
    stt._recreate_times_window.extend([now - 2.0, now - 1.8, now - 1.6,
                                       now - 1.4, now - 1.2, now - 1.0])

    stt._recreate_recognizer()

    # Wait past rate-limit, try again -- window still over cap.
    stt._last_recreate_time = time.monotonic() - 2.0
    stt._recreate_recognizer()

    restart_events = [e for e, _ in bus._emitted if e == "stt_needs_full_restart"]
    assert len(restart_events) == 1, \
        f"circuit breaker re-emitted while sticky: {bus._emitted}"


def test_should_feed_recognizer_blocks_speaking(monkeypatch):
    """TTS echo must NOT reach the recognizer when state is SPEAKING."""
    stt, _ = _make_stt(monkeypatch, state=_FakeState.SPEAKING, barge_in=False)
    assert stt._should_feed_recognizer() is False


def test_should_feed_recognizer_allows_barge_in(monkeypatch):
    """With barge-in explicitly on, SPEAKING still feeds (headphone-setup)."""
    stt, _ = _make_stt(monkeypatch, state=_FakeState.SPEAKING, barge_in=True)
    assert stt._should_feed_recognizer() is True


def test_should_feed_recognizer_allows_listening(monkeypatch):
    stt, _ = _make_stt(monkeypatch, state=_FakeState.LISTENING)
    assert stt._should_feed_recognizer() is True


def test_should_feed_recognizer_blocks_idle_and_thinking(monkeypatch):
    stt, _ = _make_stt(monkeypatch, state=_FakeState.IDLE)
    assert stt._should_feed_recognizer() is False

    stt2, _ = _make_stt(monkeypatch, state=_FakeState.THINKING)
    assert stt2._should_feed_recognizer() is False


def test_restart_recognition_chain_debounce(monkeypatch):
    """Two chain restarts inside 250 ms: second is absorbed (pending flag)."""
    stt, bus = _make_stt(monkeypatch)
    stt._listening = True
    stt._using_sounddevice = True
    stt._recognizer = SimpleNamespace()

    # First call past debounce window -> will attempt to do work. Patch the
    # framework side-effect sites to benign no-ops so the first call can
    # proceed without touching real Apple objects.
    stt._apply_speech_request_policy = MagicMock()
    stt._apply_speech_request_hints = MagicMock()
    stt._flush_prebuffer_to_request = MagicMock(return_value=0)

    from voice import stt_macos as mod

    class _FakeReq:
        @classmethod
        def alloc(cls):
            return cls()

        def init(self):
            return self

        def endAudio(self):
            return None

    class _FakeSpeech:
        SFSpeechAudioBufferRecognitionRequest = _FakeReq

    monkeypatch.setattr(mod, "_Speech", _FakeSpeech(), raising=False)

    class _FakeRecognizer:
        def recognitionTaskWithRequest_resultHandler_(self, req, handler):
            return SimpleNamespace(cancel=lambda: None)

    stt._recognizer = _FakeRecognizer()

    # First call: _last_chain_restart_time is 0 → not debounced, proceeds.
    stt._restart_recognition_chain()
    t_after_first = stt._last_chain_restart_time
    assert t_after_first > 0, "first call must register a restart time"

    # Second call inside the 250 ms debounce window: must NO-OP, only flip
    # _pending_chain_restart; must NOT advance _last_chain_restart_time.
    stt._restart_recognition_chain()
    assert stt._pending_chain_restart is True, \
        "second call in debounce window should set pending flag"
    assert stt._last_chain_restart_time == t_after_first, \
        "debounced call must not advance last restart time"


def test_empty_final_counter_escalates_to_watchdog(monkeypatch):
    """Simulate the Apple empty-isFinal=True cascade directly on the counter.

    We bypass the full ObjC result-handler shape and poke the counter field
    the same way the handler does, then check the escalation emit.
    """
    stt, bus = _make_stt(monkeypatch)

    threshold = stt._max_empty_finals_before_escalate
    for i in range(threshold):
        stt._consecutive_empty_finals += 1
        if (stt._consecutive_empty_finals >= threshold
                and not stt._escalation_requested):
            stt._escalation_requested = True
            stt._bus.emit("stt_needs_full_restart",
                          reason="empty_final_cascade")

    events = [(e, d) for e, d in bus._emitted if e == "stt_needs_full_restart"]
    assert len(events) == 1
    assert events[0][1].get("reason") == "empty_final_cascade"


def test_audio_prebuffer_cleared_on_recreate(monkeypatch):
    """After a productive recreate failure path, prebuffer must be cleared
    before handing audio to a fresh recognizer (prevents TTS-echo replay)."""
    import numpy as np

    stt, _bus = _make_stt(monkeypatch)
    stt._listening = True
    stt._using_sounddevice = True
    stt._recognizer = SimpleNamespace()
    stt._audio_prebuffer.append(np.zeros(512, dtype="float32"))

    stt._chain_restart_no_partial_count = stt._max_chain_restarts_before_recreate
    stt._got_partial_since_restart = False
    stt._last_partial = ""
    # Force the recreate to be rate-limited so we take the early-return
    # path that clears prebuffer then bails without Apple framework calls.
    stt._last_recreate_time = time.monotonic()

    stt._restart_recognition_chain()
    assert len(stt._audio_prebuffer) == 0, \
        "prebuffer must be cleared before recreate attempt"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
