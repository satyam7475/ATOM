"""
Regression tests for ``voice.stt_macos.NativeSTT`` confidence-based
promotion gate (Phase E4).

The recognizer assigns each segment a confidence in ``[0.0, 1.0]``.
``_speech_pyobjc_block`` averages those into ``_last_confidence``. We
refuse to promote any final whose confidence sits below the configured
floors so noise hypotheses ("you", "uh", random homophones during a
silent room) cannot leak through:

  * ``promotion_min_confidence`` -- absolute floor (default 0.50).
    Below this, the text is junk regardless of context.
  * ``promotion_min_confidence_no_wake`` -- the bar a cold opener has
    to clear when there is no recent wake context (default 0.65).
    A high-confidence sentence still lands; a medium-confidence one
    requires the user to have just said the wake word.

Confidence ``== 0.0`` is treated as "unknown" (some Apple builds set
0.0 when on-device-only is active and the recognizer hasn't yet
populated per-segment scores) and must never block promotion.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

if sys.platform != "darwin":
    pytest.skip("stt_macos confidence tests require darwin", allow_module_level=True)

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
    def __init__(self) -> None:
        from core.state_manager import AtomState
        self.current = AtomState.LISTENING


def _make_stt(
    *,
    min_conf: float = 0.50,
    min_conf_no_wake: float = 0.65,
) -> NativeSTT:
    bus = _FakeBus()
    state = _FakeStateManager()
    config = {
        "stt": {
            "promotion_min_confidence": min_conf,
            "promotion_min_confidence_no_wake": min_conf_no_wake,
            "promotion_min_alpha_chars": 0,
            "promotion_wake_grace_s": 3.0,
        },
    }
    return NativeSTT(bus, state, config=config)  # type: ignore[arg-type]


# ── Absolute floor (always-on) ───────────────────────────────────────


def test_below_absolute_floor_blocks_even_with_wake_context():
    import time as _t
    stt = _make_stt(min_conf=0.50, min_conf_no_wake=0.65)
    stt._last_wake_seen_t = _t.monotonic()  # fresh wake context
    stt._last_confidence = 0.30
    assert stt._should_block_low_confidence_promotion("hello") is True
    assert stt._low_confidence_dropped_count == 1


def test_above_absolute_floor_with_wake_context_passes_when_below_no_wake_floor():
    """Wake context lets a moderate-confidence hypothesis through."""
    import time as _t
    stt = _make_stt(min_conf=0.50, min_conf_no_wake=0.65)
    stt._last_wake_seen_t = _t.monotonic()
    stt._last_confidence = 0.55  # above absolute, below no-wake floor
    assert stt._should_block_low_confidence_promotion("yes go") is False


def test_at_or_above_absolute_floor_without_wake_blocks_below_no_wake_floor():
    stt = _make_stt(min_conf=0.50, min_conf_no_wake=0.65)
    # No wake -> _last_wake_seen_t remains 0.0
    stt._last_confidence = 0.55
    assert stt._should_block_low_confidence_promotion("hello there") is True
    assert stt._low_confidence_dropped_count == 1


def test_above_no_wake_floor_passes_without_wake_context():
    stt = _make_stt(min_conf=0.50, min_conf_no_wake=0.65)
    stt._last_confidence = 0.80
    assert stt._should_block_low_confidence_promotion("clear sentence") is False


# ── Edge cases ───────────────────────────────────────────────────────


def test_zero_confidence_is_treated_as_unknown_and_passes():
    """Some Apple builds report 0.0 for on-device-only -- must not block."""
    stt = _make_stt(min_conf=0.50, min_conf_no_wake=0.65)
    stt._last_confidence = 0.0
    assert stt._should_block_low_confidence_promotion("hello world") is False
    assert stt._low_confidence_dropped_count == 0


def test_empty_text_does_not_block_or_count():
    stt = _make_stt()
    stt._last_confidence = 0.10
    assert stt._should_block_low_confidence_promotion("   ") is False
    assert stt._low_confidence_dropped_count == 0


def test_zero_floors_disable_the_gate_entirely():
    stt = _make_stt(min_conf=0.0, min_conf_no_wake=0.0)
    stt._last_confidence = 0.05
    assert stt._should_block_low_confidence_promotion("anything") is False


def test_wake_context_expires_after_grace_window():
    import time as _t
    stt = _make_stt(min_conf=0.50, min_conf_no_wake=0.65)
    # Stale wake -> longer than _promotion_wake_grace_s ago.
    stt._last_wake_seen_t = _t.monotonic() - 10.0
    stt._last_confidence = 0.55
    assert stt._should_block_low_confidence_promotion("hello") is True


# ── Configuration plumbing ──────────────────────────────────────────


def test_constructor_loads_floors_from_stt_config():
    stt = _make_stt(min_conf=0.40, min_conf_no_wake=0.80)
    assert stt._promotion_min_confidence == pytest.approx(0.40)
    assert stt._promotion_min_confidence_no_wake == pytest.approx(0.80)


def test_default_floors_match_phase_e4_spec():
    """Default config (no explicit overrides) ships with E4 thresholds."""
    bus = _FakeBus()
    state = _FakeStateManager()
    stt = NativeSTT(bus, state, config={"stt": {}})  # type: ignore[arg-type]
    assert stt._promotion_min_confidence == pytest.approx(0.50)
    assert stt._promotion_min_confidence_no_wake == pytest.approx(0.65)
