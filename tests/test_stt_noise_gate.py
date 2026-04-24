"""
Regression tests for ``voice.stt_macos.NativeSTT`` noise gate (Phase E3).

The gate sits between the audio callback and ``appendAudioPCMBuffer_`` on
``SFSpeechRecognitionRequest``. It exists to keep ambient room noise out
of the recognizer so it cannot accumulate into ghost "you" / "uh huh"
partials during long quiet stretches of LISTENING.

Behaviour pinned by these tests:

  * Frames at or above ``_noise_floor_dbfs`` always pass through and
    immediately reopen the gate (so the leading edge of a real
    utterance is never clipped).
  * Frames below the floor close the gate only after
    ``_noise_gate_consecutive`` of them in a row.
  * Setting ``_noise_floor_dbfs <= -96`` disables the gate entirely
    (escape hatch for noisy outdoor sessions / debugging).
  * The end-to-end ``_sd_audio_callback`` path skips
    ``appendAudioPCMBuffer_`` when the gate is closed and resumes the
    moment a loud frame arrives.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest

if sys.platform != "darwin":
    pytest.skip("stt_macos noise gate tests require darwin", allow_module_level=True)

from voice import stt_macos  # noqa: E402
from voice.stt_macos import NativeSTT  # noqa: E402


# ── Minimal fakes ────────────────────────────────────────────────────


class _FakeBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, name: str, **data: Any) -> None:
        self.events.append((name, data))

    emit_fast = emit

    def on(self, *_a: Any, **_k: Any) -> None:
        return None


class _FakeStateManager:
    """Pretends ATOM is in LISTENING so ``_should_feed_recognizer`` passes."""

    def __init__(self) -> None:
        from core.state_manager import AtomState
        self.current = AtomState.LISTENING


class _FakeRequest:
    def __init__(self) -> None:
        self.append_calls: list[Any] = []

    def appendAudioPCMBuffer_(self, buf: Any) -> None:
        self.append_calls.append(buf)


def _make_stt(noise_floor_dbfs: float = -55.0, consecutive: int = 5) -> NativeSTT:
    bus = _FakeBus()
    state = _FakeStateManager()
    config = {
        "stt": {
            "noise_floor_dbfs": noise_floor_dbfs,
            "noise_gate_consecutive": consecutive,
        },
    }
    return NativeSTT(bus, state, config=config)  # type: ignore[arg-type]


# ── Direct gate logic tests ─────────────────────────────────────────


def test_supra_floor_frame_does_not_block():
    stt = _make_stt()
    assert stt._noise_gate_blocks(-30.0) is False
    assert stt._noise_gate_below_count == 0


def test_single_sub_floor_frame_does_not_close_gate():
    stt = _make_stt()
    assert stt._noise_gate_blocks(-70.0) is False
    assert stt._noise_gate_below_count == 1


def test_five_consecutive_sub_floor_frames_close_gate():
    stt = _make_stt(consecutive=5)
    # Frames 1-4 increment the counter but stay open.
    for i in range(1, 5):
        assert stt._noise_gate_blocks(-80.0) is False
        assert stt._noise_gate_below_count == i
    # Frame 5 closes the gate.
    assert stt._noise_gate_blocks(-80.0) is True
    assert stt._noise_gate_below_count == 5
    # And keeps it closed for sustained silence.
    assert stt._noise_gate_blocks(-80.0) is True
    assert stt._noise_gate_dropped_total == 2


def test_supra_floor_frame_immediately_reopens_gate():
    stt = _make_stt(consecutive=5)
    for _ in range(8):
        stt._noise_gate_blocks(-80.0)
    # Gate is closed; one loud frame must reopen it.
    assert stt._noise_gate_blocks(-25.0) is False
    assert stt._noise_gate_below_count == 0
    # Subsequent sub-floor frames start the count from scratch.
    assert stt._noise_gate_blocks(-80.0) is False
    assert stt._noise_gate_below_count == 1


def test_disabled_floor_lets_everything_through():
    stt = _make_stt(noise_floor_dbfs=-96.0)
    for _ in range(50):
        assert stt._noise_gate_blocks(-99.0) is False
    assert stt._noise_gate_dropped_total == 0


def test_none_rms_treated_as_supra_floor():
    """Missing measurements must never wedge the gate closed."""
    stt = _make_stt()
    for _ in range(10):
        stt._noise_gate_blocks(-80.0)
    # Gate is closed; an unknown-RMS frame opens it.
    assert stt._noise_gate_blocks(None) is False
    assert stt._noise_gate_below_count == 0


def test_consecutive_threshold_is_configurable():
    stt = _make_stt(consecutive=3)
    assert stt._noise_gate_blocks(-80.0) is False
    assert stt._noise_gate_blocks(-80.0) is False
    assert stt._noise_gate_blocks(-80.0) is True


def test_zero_or_negative_consecutive_clamped_to_one():
    stt = _make_stt(consecutive=0)
    # The sanitised threshold is at least 1, so the very first sub-floor
    # frame closes the gate.
    assert stt._noise_gate_consecutive == 1
    assert stt._noise_gate_blocks(-80.0) is True


# ── End-to-end callback test ────────────────────────────────────────


def _silent_block(samples: int = 2048) -> np.ndarray:
    """A floor-level block (pure silence ~ -96 dBFS)."""
    return np.zeros(samples, dtype=np.float32)


def _loud_block(samples: int = 2048, amp: float = 0.3) -> np.ndarray:
    """A clearly above-floor block (~ -10 dBFS sine)."""
    t = np.arange(samples, dtype=np.float32) / 48000.0
    return (amp * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)


def test_sd_callback_drops_sustained_silence_from_recognizer(monkeypatch):
    stt = _make_stt(noise_floor_dbfs=-55.0, consecutive=5)
    req = _FakeRequest()
    stt._recognition_request = req

    sentinel = object()
    monkeypatch.setattr(stt, "_numpy_to_pcm_buffer", lambda *_a, **_k: sentinel)

    silent = _silent_block()
    for _ in range(20):
        stt._sd_audio_callback(silent, silent.shape[0], None, None)

    # First 4 silent frames slip through (gate count 1..4); frames 5..20
    # are dropped. So we expect exactly 4 appends, all the sentinel.
    assert len(req.append_calls) == 4
    assert all(c is sentinel for c in req.append_calls)
    assert stt._noise_gate_dropped_total == 16


def test_sd_callback_recovers_immediately_on_loud_frame(monkeypatch):
    stt = _make_stt(noise_floor_dbfs=-55.0, consecutive=5)
    req = _FakeRequest()
    stt._recognition_request = req

    sentinel = object()
    monkeypatch.setattr(stt, "_numpy_to_pcm_buffer", lambda *_a, **_k: sentinel)

    silent = _silent_block()
    loud = _loud_block()

    # Drive the gate fully closed.
    for _ in range(10):
        stt._sd_audio_callback(silent, silent.shape[0], None, None)
    appends_after_silence = len(req.append_calls)
    assert stt._noise_gate_dropped_total >= 5

    # One loud frame must reach the recognizer immediately.
    stt._sd_audio_callback(loud, loud.shape[0], None, None)
    assert len(req.append_calls) == appends_after_silence + 1
    assert stt._noise_gate_below_count == 0
