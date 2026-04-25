"""Regression tests for Sprint N7 -- mood-driven voice prosody."""

from __future__ import annotations

import pytest

from core.speech_controller import SpeechController
from voice.mood_voice_profile import for_mood, known_moods


def test_for_mood_neutral_when_unknown() -> None:
    p = for_mood("does-not-exist")
    assert p.mood == "neutral"
    assert p.rate_multiplier == 1.0
    assert p.pause_multiplier == 1.0


def test_for_mood_focused_speeds_up_and_tightens_pauses() -> None:
    p = for_mood("focused")
    assert p.rate_multiplier > 1.0
    assert p.pause_multiplier < 1.0


def test_for_mood_tired_slows_down_and_widens_pauses() -> None:
    p = for_mood("tired")
    assert p.rate_multiplier < 1.0
    assert p.pause_multiplier > 1.0


def test_for_mood_returns_speech_params_dict() -> None:
    p = for_mood("alert")
    params = p.to_speech_params()
    assert set(params.keys()) == {"rate_multiplier", "pause_multiplier"}


def test_known_moods_includes_core_set() -> None:
    moods = set(known_moods())
    expected = {
        "neutral", "focused", "engaged", "calm", "tired",
        "stressed", "alert", "urgent", "happy", "distracted", "idle",
    }
    assert expected.issubset(moods)


def test_speech_controller_mood_channel_composes_with_others() -> None:
    s = SpeechController()
    s.set_perception(rate_multiplier=1.10, pause_multiplier=0.90)
    s.set_adaptive(rate_multiplier=1.00, pause_multiplier=1.00)
    s.set_mood(rate_multiplier=0.90, pause_multiplier=1.10)

    merged = s.merged()
    # 1.10 * 1.00 * 0.90 = 0.99
    assert merged["rate_multiplier"] == pytest.approx(0.99)
    # 0.90 * 1.00 * 1.10 = 0.99
    assert merged["pause_multiplier"] == pytest.approx(0.99)


def test_speech_controller_reset_keeps_mood_persistent() -> None:
    s = SpeechController()
    s.set_perception(rate_multiplier=1.5, pause_multiplier=0.5)
    s.set_mood(rate_multiplier=1.05, pause_multiplier=0.95)
    s.reset()  # only resets perception + adaptive

    merged = s.merged()
    # mood still present after reset
    assert merged["rate_multiplier"] == pytest.approx(1.05)
    assert merged["pause_multiplier"] == pytest.approx(0.95)
