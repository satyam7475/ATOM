"""v3 Phase 4 — WhisperConfirmer regression tests.

Covers:
  * Disabled-by-default semantics (no model load, no surprise cost).
  * Suspect-detection rules (blank, single noise token, low conf, short).
  * Healthy finals pass through unmodified.
  * Ring buffer is bounded and resamples correctly.
  * Confirmation budget watchdog tags slow runs.
  * Whisper-unavailable fallback returns the original text safely.
  * stt_macos.attach_whisper_confirmer wires the instance.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice.whisper_confirmer import (
    ConfirmResult,
    WhisperConfirmer,
    _NOISE_TOKENS,
    _RingBuffer,
)


# ── Defaults / disabled mode ──────────────────────────────────────────


def test_confirmer_is_disabled_by_default() -> None:
    """Construction without config yields a no-op confirmer.
    Important: must NOT touch faster-whisper at import or construction
    time. Cold boot regression."""
    wc = WhisperConfirmer({})
    assert wc.is_enabled() is False
    out = wc.confirm("anything", 0.0)
    assert out.text == "anything"
    assert out.used_whisper is False
    assert out.reason == "ok"


def test_confirmer_disabled_does_not_fill_ring() -> None:
    """When disabled, feed_audio is a no-op so we waste no memory."""
    wc = WhisperConfirmer({})
    wc.feed_audio(b"\x00" * 4096)
    assert wc._ring.duration_s() == 0.0


def test_confirmer_enabled_via_config() -> None:
    cfg = {"whisper_confirm": {"enabled": True, "model_size": "tiny"}}
    wc = WhisperConfirmer(cfg)
    assert wc.is_enabled() is True
    assert wc._model_size == "tiny"


# ── Suspect detection ────────────────────────────────────────────────


@pytest.mark.parametrize("noise_token", sorted(_NOISE_TOKENS))
def test_suspect_detects_known_noise_tokens(noise_token: str) -> None:
    wc = WhisperConfirmer({"whisper_confirm": {"enabled": True}})
    suspect, reason = wc._is_suspect(noise_token, 0.99)
    assert suspect is True
    assert reason == "noise"


def test_suspect_detects_blank_or_whitespace() -> None:
    wc = WhisperConfirmer({"whisper_confirm": {"enabled": True}})
    for blank in ("", " ", "\n", "\t  "):
        suspect, reason = wc._is_suspect(blank, 0.99)
        assert suspect is True
        assert reason == "noise"


def test_suspect_detects_low_confidence() -> None:
    wc = WhisperConfirmer({
        "whisper_confirm": {"enabled": True, "min_confidence": 0.7},
    })
    suspect, reason = wc._is_suspect("can you do that", 0.4)
    assert suspect is True
    assert reason == "low_conf"


def test_suspect_detects_short_text() -> None:
    wc = WhisperConfirmer({
        "whisper_confirm": {"enabled": True, "min_text_chars": 4},
    })
    suspect, reason = wc._is_suspect("hi", 0.99)
    assert suspect is True
    assert reason == "short"


def test_healthy_text_is_not_suspect() -> None:
    wc = WhisperConfirmer({"whisper_confirm": {"enabled": True}})
    suspect, reason = wc._is_suspect("what is the weather today", 0.92)
    assert suspect is False
    assert reason == "ok"


# ── Confirmation flow ────────────────────────────────────────────────


def test_confirm_passes_healthy_text_through_without_loading_whisper() -> None:
    """A healthy final should never trigger a model load. This is what
    keeps mean-case latency identical to the no-confirmer path."""
    wc = WhisperConfirmer({"whisper_confirm": {"enabled": True}})
    out = wc.confirm("what is the weather today", 0.95)
    assert out.text == "what is the weather today"
    assert out.used_whisper is False
    assert out.reason == "ok"
    assert wc._model is None
    assert wc._model_load_attempted is False


def test_confirm_returns_original_when_whisper_unavailable() -> None:
    """If faster-whisper cannot load, the streaming text must still
    flow through. Safety: never silence the user because of a missing
    optional dep."""
    wc = WhisperConfirmer({"whisper_confirm": {"enabled": True}})
    wc._model_unavailable = True
    out = wc.confirm("uh", 0.2)
    assert out.text == "uh"
    assert out.used_whisper is False
    assert out.reason == "whisper_unavailable"


def test_confirm_uses_stub_model_to_correct_text() -> None:
    """Inject a stub WhisperModel so we can exercise the suspect →
    re-decode → corrected-text path without downloading anything."""

    class _StubSeg:
        def __init__(self, text: str) -> None:
            self.text = text

    class _StubWhisper:
        def transcribe(self, samples, **_kw):
            return iter([_StubSeg(" what time is it ")]), {}

    wc = WhisperConfirmer({
        "whisper_confirm": {"enabled": True, "min_confidence": 0.9},
    })
    wc._model = _StubWhisper()
    samples = (np.ones(16000, dtype=np.float32) * 0.01).tobytes()
    wc.feed_audio(samples)
    # Use a non-noise short string so we exercise the low_conf branch
    # specifically (noise has higher precedence and shadows it).
    out = wc.confirm("watt im is it", 0.5)
    assert out.text == "what time is it"
    assert out.used_whisper is True
    assert out.reason == "low_conf"
    assert wc.confirmed == 1


def test_confirm_collapses_noise_to_empty_when_whisper_agrees() -> None:
    """If the streaming engine emits a noise token but Whisper says
    silence, we trust Whisper and drop the false positive."""

    class _SilentWhisper:
        def transcribe(self, samples, **_kw):
            return iter(()), {}

    wc = WhisperConfirmer({"whisper_confirm": {"enabled": True}})
    wc._model = _SilentWhisper()
    wc.feed_audio((np.zeros(16000, dtype=np.float32)).tobytes())
    out = wc.confirm("uh", 0.99)
    assert out.text == ""
    assert out.used_whisper is True
    assert out.reason == "noise"


def test_confirm_returns_original_when_whisper_decode_raises() -> None:
    """A buggy/unavailable runtime must NOT eat the user's transcript."""

    class _BrokenWhisper:
        def transcribe(self, samples, **_kw):
            raise RuntimeError("oops")

    wc = WhisperConfirmer({
        "whisper_confirm": {"enabled": True, "min_confidence": 0.9},
    })
    wc._model = _BrokenWhisper()
    wc.feed_audio((np.ones(16000, dtype=np.float32) * 0.01).tobytes())
    out = wc.confirm("uh", 0.5)
    assert out.text == "uh"
    assert out.used_whisper is False
    assert out.reason == "whisper_failed"


def test_confirm_records_elapsed_within_budget() -> None:
    """The confirm() loop should be tens of ms with a stub model;
    well inside the 250ms budget."""

    class _FastWhisper:
        def transcribe(self, samples, **_kw):
            class _S:
                text = "hello"
            return iter([_S()]), {}

    wc = WhisperConfirmer({"whisper_confirm": {"enabled": True}})
    wc._model = _FastWhisper()
    wc.feed_audio((np.ones(8000, dtype=np.float32) * 0.05).tobytes())
    out = wc.confirm("uh", 0.3)
    assert out.elapsed_ms < 100.0, (
        f"Stub-model confirm should be near-instant; got {out.elapsed_ms:.1f}ms"
    )


# ── Ring buffer ──────────────────────────────────────────────────────


def test_ring_buffer_caps_at_capacity() -> None:
    rb = _RingBuffer(max_seconds=1.0, sample_rate=16000)
    big = (np.ones(48000, dtype=np.float32)).tobytes()  # 3 seconds
    rb.feed(big)
    snap = rb.snapshot(seconds=2.0)
    # Cap is 1s = 16000 samples = 64000 bytes
    assert len(snap) == 64000
    assert rb.duration_s() == pytest.approx(1.0, rel=0.01)


def test_ring_buffer_snapshot_returns_tail_only() -> None:
    rb = _RingBuffer(max_seconds=2.0, sample_rate=16000)
    head = (np.ones(16000, dtype=np.float32) * 0.1).tobytes()
    tail = (np.ones(16000, dtype=np.float32) * 0.9).tobytes()
    rb.feed(head)
    rb.feed(tail)
    snap = rb.snapshot(seconds=1.0)
    arr = np.frombuffer(snap, dtype=np.float32)
    assert arr.size == 16000
    assert float(arr.mean()) == pytest.approx(0.9, rel=1e-4)


def test_ring_buffer_clear() -> None:
    rb = _RingBuffer(max_seconds=1.0, sample_rate=16000)
    rb.feed((np.ones(16000, dtype=np.float32)).tobytes())
    assert rb.duration_s() > 0
    rb.clear()
    assert rb.duration_s() == 0.0
    assert rb.snapshot(2.0) == b""


def test_ring_buffer_handles_empty_feed() -> None:
    rb = _RingBuffer(max_seconds=1.0, sample_rate=16000)
    rb.feed(b"")
    assert rb.duration_s() == 0.0


def test_set_sample_rate_resets_ring() -> None:
    """Switching sample rate must reset the buffer (stale samples at the
    old rate would corrupt the next decode)."""
    wc = WhisperConfirmer({"whisper_confirm": {"enabled": True}})
    wc.feed_audio((np.ones(16000, dtype=np.float32)).tobytes())
    assert wc._ring.duration_s() > 0
    wc.set_sample_rate(48000)
    assert wc._ring.duration_s() == 0.0
    assert wc._ring.sample_rate == 48000


# ── Resampling sanity (the path Whisper actually receives) ────────────


def test_internal_resample_44k_to_16k_preserves_duration() -> None:
    """When the audio comes from a 44.1kHz device, _whisper_decode
    resamples it to 16kHz before handing to Whisper. Verify duration
    is preserved within ~1 sample (linear interp slack)."""

    captured: list[np.ndarray] = []

    class _CaptureWhisper:
        def transcribe(self, samples, **_kw):
            captured.append(samples)
            class _S:
                text = ""
            return iter([_S()]), {}

    wc = WhisperConfirmer({"whisper_confirm": {"enabled": True, "decode_seconds": 1.0}})
    wc.set_sample_rate(44100)
    wc._model = _CaptureWhisper()
    wc.feed_audio((np.ones(44100, dtype=np.float32) * 0.05).tobytes())
    wc.confirm("uh", 0.0)
    assert captured, "stub Whisper was not called"
    arr = captured[0]
    # 1 second of audio resampled to 16kHz
    assert abs(arr.size - 16000) <= 2, f"expected ~16000 samples, got {arr.size}"
    assert arr.dtype == np.float32


# ── stt_macos wiring ─────────────────────────────────────────────────


def test_stt_macos_exposes_attach_whisper_confirmer() -> None:
    """Without instantiating the real macOS STT (requires pyobjc), just
    check the method is present on the class and attaches the object."""
    from voice import stt_macos

    cls = getattr(stt_macos, "MacOSNativeSTT", None) or getattr(
        stt_macos, "MacOSSTT", None,
    )
    if cls is None:
        # Class name has shifted before; find any class with the attach.
        cls = next(
            (
                obj for name, obj in vars(stt_macos).items()
                if isinstance(obj, type)
                and hasattr(obj, "attach_whisper_confirmer")
            ),
            None,
        )
    assert cls is not None, "no STT class with attach_whisper_confirmer found"
    assert hasattr(cls, "attach_whisper_confirmer")

    inst = cls.__new__(cls)
    inst._whisper_confirmer = None
    sentinel = object()
    inst.attach_whisper_confirmer(sentinel)
    assert inst._whisper_confirmer is sentinel
    inst.attach_whisper_confirmer(None)
    assert inst._whisper_confirmer is None


# ── Settings schema sanity ───────────────────────────────────────────


def test_settings_json_includes_whisper_confirm_block() -> None:
    """settings.json must ship with a fully-formed whisper_confirm block.

    Sprint P1.6 (Apr 26 2026): the default flipped from ``false`` to
    ``true`` because the second-pass confirm path is now wired into
    ``WhisperSTT`` (P2.2). Older versions of this test asserted the
    default was opt-out; we now assert the block exists and has the
    expected shape, and that ``enabled`` is a bool either way so a
    user can flip it back without hitting a schema error.
    """
    import json
    from pathlib import Path

    cfg_path = Path(__file__).resolve().parent.parent / "config" / "settings.json"
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    stt = cfg.get("stt") or {}
    wc = stt.get("whisper_confirm") or {}
    assert "enabled" in wc
    assert isinstance(wc.get("enabled"), bool), "enabled must be a bool"
    assert wc.get("model_size") in {"tiny", "tiny.en", "base", "base.en"}
    assert isinstance(wc.get("decode_seconds"), (int, float))
    assert isinstance(wc.get("max_confirm_ms"), (int, float))


def test_confirmer_stats_shape() -> None:
    wc = WhisperConfirmer({"whisper_confirm": {"enabled": True}})
    s = wc.stats()
    for key in (
        "enabled", "calls", "confirmed", "last_elapsed_ms",
        "ring_duration_s", "model_loaded", "model_unavailable", "model_size",
    ):
        assert key in s, f"stats() missing key: {key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
