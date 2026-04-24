"""
Regression tests for ``voice.tts_macos._NativeSynth`` first-word warmup
and tail-drain behaviour (Phase E1 + E2).

These tests pin two pieces of audio-clipping protection:

  * Pre-roll silence (``_first_word_warmup_s``) before the very first
    ``startSpeakingString_`` so Bluetooth / USB-C dongles don't latch
    onto the first phoneme. Skipped when the previous utterance
    finished within ``_warmup_skip_window_s`` so a continuous stream
    doesn't pay the warmup tax twice.

  * Tail drain (``_tail_drain_s`` / ``_tail_drain_bluetooth_s``) after
    ``isSpeaking()`` flips to False so the last sample fully flushes
    through CoreAudio's render buffer (Bluetooth headsets have an extra
    ~80ms hardware latency that otherwise eats the final word).

The tests run on every platform: ``_NativeSynth`` is constructed
directly with a fake synth handle and a fake ``Foundation`` namespace,
so we never touch AppKit / pyobjc.
"""

from __future__ import annotations

import sys

import pytest

from voice import tts_macos


# ── Fakes ────────────────────────────────────────────────────────────


class _FakeSynth:
    """Stand-in for ``NSSpeechSynthesizer``.

    ``isSpeaking()`` returns True for ``speak_iters`` polls after each
    ``startSpeakingString_`` call, then flips to False so the speak
    loop transitions through the tail-drain branch exactly once.
    """

    def __init__(self, speak_iters: int = 2) -> None:
        self.start_calls: list[str] = []
        self.set_rate_calls: list[float] = []
        self.set_voice_calls: list[str] = []
        self.stop_calls: int = 0
        self._iters_left = 0
        self._speak_iters = speak_iters

    def setRate_(self, r: float) -> None:
        self.set_rate_calls.append(float(r))

    def setVoice_(self, v: str) -> None:
        self.set_voice_calls.append(v)

    def startSpeakingString_(self, text: str) -> None:
        self.start_calls.append(text)
        self._iters_left = self._speak_iters

    def isSpeaking(self) -> bool:
        if self._iters_left > 0:
            self._iters_left -= 1
            return True
        return False

    def stopSpeaking(self) -> None:
        self.stop_calls += 1

    def objectForProperty_error_(self, _prop, _err):
        return (None, None)

    def setObject_forProperty_error_(self, *_a, **_k):
        return None


class _FakeRunLoop:
    def runMode_beforeDate_(self, _mode, _date) -> None:
        return None


class _FakeNSRunLoop:
    @staticmethod
    def currentRunLoop() -> _FakeRunLoop:
        return _FakeRunLoop()


class _FakeNSDate:
    @staticmethod
    def dateWithTimeIntervalSinceNow_(_s) -> object:
        return object()


class _FakeFoundation:
    NSRunLoop = _FakeNSRunLoop
    NSDate = _FakeNSDate
    NSDefaultRunLoopMode = "default"


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def patched_clock(monkeypatch):
    """Deterministic ``time.sleep`` / ``time.monotonic`` for the synth."""
    state = {"now": 1000.0, "sleeps": []}

    def fake_sleep(seconds: float) -> None:
        state["sleeps"].append(float(seconds))
        state["now"] += float(seconds)

    def fake_monotonic() -> float:
        # Tick forward a hair on every poll so the run-loop guard
        # eventually trips its progress / startup deadlines if the
        # fake synth never flips ``isSpeaking()``.
        state["now"] += 0.001
        return state["now"]

    monkeypatch.setattr(tts_macos.time, "sleep", fake_sleep)
    monkeypatch.setattr(tts_macos.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(tts_macos, "_Foundation", _FakeFoundation)
    return state


def _make_synth(speak_iters: int = 2) -> tts_macos._NativeSynth:
    native = tts_macos._NativeSynth(voice_id="", rate=180.0, pitch_shift=0.0)
    fake = _FakeSynth(speak_iters=speak_iters)
    # Inject the fake handle so ``_ensure_synth`` never touches AppKit.
    native._synth = fake
    return native


# ── Tests: pre-roll ─────────────────────────────────────────────────


def test_first_speak_inserts_preroll_silence(patched_clock):
    """Cold-start speak gets the full ``_first_word_warmup_s`` pre-roll."""
    synth = _make_synth()
    synth.set_warmup_drain(first_word_warmup_s=0.140, tail_drain_s=0.120)

    synth.speak_blocking("Hello, Boss.")

    sleeps = patched_clock["sleeps"]
    assert sleeps, "expected at least one sleep call (pre-roll + tail drain)"
    # Pre-roll is the first sleep ever observed -- before
    # startSpeakingString_ is invoked.
    assert sleeps[0] == pytest.approx(0.140), (
        f"expected first sleep to be pre-roll (140ms), got {sleeps}"
    )
    fake = synth._synth  # type: ignore[attr-defined]
    assert fake.start_calls == ["Hello, Boss."]


def test_continuous_speech_skips_preroll_within_skip_window(patched_clock):
    """Two speaks back-to-back: only the first pays the warmup tax."""
    synth = _make_synth()
    synth.set_warmup_drain(
        first_word_warmup_s=0.140,
        tail_drain_s=0.120,
        warmup_skip_window_s=0.800,
    )

    synth.speak_blocking("first sentence")
    first_sleeps = list(patched_clock["sleeps"])
    assert first_sleeps[0] == pytest.approx(0.140), (
        "first speak must pay pre-roll cost"
    )

    # Within the skip window (no clock advance beyond the tiny
    # monotonic ticks accrued above) -- second speak must skip
    # the pre-roll entirely and only sleep for the tail drain.
    synth.speak_blocking("second sentence")
    second_only = patched_clock["sleeps"][len(first_sleeps):]

    assert second_only, "second speak should still record a tail-drain sleep"
    assert all(s == pytest.approx(0.120) for s in second_only), (
        f"second speak sleeps must all be tail drain (no pre-roll), got {second_only}"
    )


def test_preroll_returns_after_warmup_skip_window_elapses(patched_clock):
    """If the audio device went silent for >= skip window, warmup re-engages."""
    synth = _make_synth()
    synth.set_warmup_drain(
        first_word_warmup_s=0.140,
        tail_drain_s=0.120,
        warmup_skip_window_s=0.800,
    )

    synth.speak_blocking("first")
    first_count = len(patched_clock["sleeps"])

    # Simulate a silent gap longer than the skip window.
    patched_clock["now"] += 1.500

    synth.speak_blocking("second")
    second_only = patched_clock["sleeps"][first_count:]

    assert second_only, "second speak should record sleeps"
    assert second_only[0] == pytest.approx(0.140), (
        f"speak after silent gap must pay pre-roll again, got {second_only}"
    )


def test_zero_warmup_disables_preroll_entirely(patched_clock):
    """Setting warmup to 0 means we never insert pre-roll silence."""
    synth = _make_synth()
    synth.set_warmup_drain(first_word_warmup_s=0.0, tail_drain_s=0.120)

    synth.speak_blocking("hi")

    # Only the tail drain should be recorded.
    assert all(s == pytest.approx(0.120) for s in patched_clock["sleeps"]), (
        f"warmup=0 must skip the pre-roll, got {patched_clock['sleeps']}"
    )


# ── Tests: tail drain ───────────────────────────────────────────────


def test_tail_drain_runs_after_isspeaking_clears(patched_clock):
    """After speech ends, sleep for the configured tail drain."""
    synth = _make_synth()
    synth.set_warmup_drain(first_word_warmup_s=0.140, tail_drain_s=0.120)

    synth.speak_blocking("done")

    assert patched_clock["sleeps"][-1] == pytest.approx(0.120), (
        f"last sleep must be tail drain (120ms), got {patched_clock['sleeps']}"
    )


def test_tail_drain_lengthens_on_bluetooth_output(patched_clock):
    """Bluetooth has an extra ~80ms hardware latency -- use longer drain."""
    synth = _make_synth()
    synth.set_warmup_drain(
        first_word_warmup_s=0.140,
        tail_drain_s=0.120,
        tail_drain_bluetooth_s=0.200,
    )
    synth.set_output_is_bluetooth(True)

    synth.speak_blocking("hi")

    assert patched_clock["sleeps"][-1] == pytest.approx(0.200), (
        f"bluetooth tail drain must be 200ms, got {patched_clock['sleeps']}"
    )

    # Toggle off -- next speak must drop back to the wired drain.
    synth.set_output_is_bluetooth(False)
    patched_clock["now"] += 1.500  # force pre-roll again so we see clean tail
    prior_count = len(patched_clock["sleeps"])
    synth.speak_blocking("again")
    new_sleeps = patched_clock["sleeps"][prior_count:]
    assert new_sleeps[-1] == pytest.approx(0.120), (
        f"non-bluetooth tail drain must be 120ms after toggle, got {new_sleeps}"
    )


def test_zero_tail_drain_disables_post_speech_sleep(patched_clock):
    """Setting tail drain to 0 must omit the post-speech sleep entirely."""
    synth = _make_synth()
    synth.set_warmup_drain(first_word_warmup_s=0.140, tail_drain_s=0.0)

    synth.speak_blocking("hi")

    # Only the pre-roll should be present; no tail drain sleep.
    assert patched_clock["sleeps"] == [pytest.approx(0.140)], (
        f"tail_drain=0 must skip post-speech sleep, got {patched_clock['sleeps']}"
    )


# ── Tests: configuration plumbing ───────────────────────────────────


def test_set_warmup_drain_clamps_negative_values(patched_clock):
    """Negative tunings are clamped to zero so we never sleep backwards."""
    synth = _make_synth()
    synth.set_warmup_drain(
        first_word_warmup_s=-0.5,
        tail_drain_s=-1.0,
        tail_drain_bluetooth_s=-2.0,
        warmup_skip_window_s=-3.0,
    )

    assert synth._first_word_warmup_s == 0.0
    assert synth._tail_drain_s == 0.0
    assert synth._tail_drain_bluetooth_s == 0.0
    assert synth._warmup_skip_window_s == 0.0


def test_macos_tts_async_propagates_warmup_drain_config(monkeypatch):
    """``MacOSTTSAsync`` forwards ms-config into ``_NativeSynth`` units."""
    if sys.platform != "darwin":
        pytest.skip("MacOSTTSAsync.init_voice path requires darwin")

    # Build the wrapper without an event loop / state machine.
    bus = object()
    state = object()
    tts = tts_macos.MacOSTTSAsync(
        bus,  # type: ignore[arg-type]
        state,  # type: ignore[arg-type]
        first_word_warmup_ms=200,
        tail_drain_ms=150,
        tail_drain_bluetooth_ms=250,
        warmup_skip_window_ms=900,
    )

    assert tts._first_word_warmup_s == pytest.approx(0.200)
    assert tts._tail_drain_s == pytest.approx(0.150)
    assert tts._tail_drain_bluetooth_s == pytest.approx(0.250)
    assert tts._warmup_skip_window_s == pytest.approx(0.900)
