"""
Regression tests for Phase E5: auto-restore Voice Processing I/O when
the active input + output return to the same hardware.

Two paths are pinned:

  1. ``NativeSTT.notify_audio_output_change`` -- the receiver-side
     contract. Restores VPIO only when the user originally requested
     it AND the engine-restart fallback had since lowered it.

  2. ``AudioIntelligenceEngine._notify_output_change`` -- the
     publisher-side contract. After every output reassignment it must
     forward the new state to STT (for VPIO) and TTS (for the
     Bluetooth tail-drain budget) and emit ``audio_output_changed``
     on the bus.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

if sys.platform != "darwin":
    pytest.skip("VPIO auto-restore tests require darwin", allow_module_level=True)

from voice import stt_macos  # noqa: E402
from voice.audio_intelligence import AudioDeviceProfile, AudioIntelligenceEngine  # noqa: E402
from voice.stt_macos import NativeSTT  # noqa: E402


# ── Fakes ────────────────────────────────────────────────────────────


class _RecordingBus:
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


class _FakeSTT:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def notify_audio_output_change(
        self, *, output_type: str | None, hw_match: bool,
    ) -> None:
        self.calls.append({"output_type": output_type, "hw_match": hw_match})


class _FakeTTS:
    def __init__(self) -> None:
        self.bt_calls: list[bool] = []

    def set_output_is_bluetooth(self, is_bt: bool) -> None:
        self.bt_calls.append(bool(is_bt))


def _profile(name: str, *, kind: str, is_input: bool, is_output: bool) -> AudioDeviceProfile:
    return AudioDeviceProfile(
        index=0,
        name=name,
        host_api="CoreAudio",
        sample_rate=48000.0,
        channels=2,
        is_input=is_input,
        is_output=is_output,
        device_type=kind,
    )


# ── STT side ────────────────────────────────────────────────────────


def _make_stt(
    *,
    cfg_vpio: bool = True,
    runtime_vpio: bool = False,
) -> NativeSTT:
    bus = _RecordingBus()
    state = _FakeStateManager()
    config = {"stt": {"native_voice_processing": cfg_vpio}}
    stt = NativeSTT(bus, state, config=config)  # type: ignore[arg-type]
    stt._native_voice_processing = runtime_vpio
    stt._engine_restart_count = 5  # pretend we walked the disable ladder
    stt._consecutive_silent_buffers = 12
    return stt


def test_notify_restores_vpio_when_hw_matches_and_cfg_wants_it(monkeypatch):
    stt = _make_stt(cfg_vpio=True, runtime_vpio=False)
    scheduled: list[bool] = []
    monkeypatch.setattr(stt, "_schedule_engine_restart", lambda: scheduled.append(True))

    stt.notify_audio_output_change(output_type="builtin", hw_match=True)

    assert stt._native_voice_processing is True
    assert stt._engine_restart_count == 0
    assert stt._consecutive_silent_buffers == 0
    assert scheduled == [True]


def test_notify_no_op_when_hw_does_not_match(monkeypatch):
    stt = _make_stt(cfg_vpio=True, runtime_vpio=False)
    scheduled: list[bool] = []
    monkeypatch.setattr(stt, "_schedule_engine_restart", lambda: scheduled.append(True))

    stt.notify_audio_output_change(output_type="bluetooth", hw_match=False)

    assert stt._native_voice_processing is False
    assert stt._engine_restart_count == 5  # unchanged
    assert scheduled == []


def test_notify_no_op_when_user_disabled_vpio_in_config(monkeypatch):
    """If the user explicitly turned VPIO off, never resurrect it."""
    stt = _make_stt(cfg_vpio=False, runtime_vpio=False)
    scheduled: list[bool] = []
    monkeypatch.setattr(stt, "_schedule_engine_restart", lambda: scheduled.append(True))

    stt.notify_audio_output_change(output_type="builtin", hw_match=True)

    assert stt._native_voice_processing is False
    assert scheduled == []


def test_notify_no_op_when_vpio_is_already_active(monkeypatch):
    """No spurious restart when VPIO is already on."""
    stt = _make_stt(cfg_vpio=True, runtime_vpio=True)
    stt._engine_restart_count = 0
    scheduled: list[bool] = []
    monkeypatch.setattr(stt, "_schedule_engine_restart", lambda: scheduled.append(True))

    stt.notify_audio_output_change(output_type="builtin", hw_match=True)

    assert scheduled == []


# ── AudioIntelligenceEngine side ─────────────────────────────────────


def _make_engine() -> AudioIntelligenceEngine:
    bus = _RecordingBus()
    state = _FakeStateManager()
    return AudioIntelligenceEngine(
        bus,  # type: ignore[arg-type]
        state,  # type: ignore[arg-type]
        config={"audio_intelligence": {}},
    )


def test_notify_output_change_forwards_to_stt_and_tts_and_bus():
    engine = _make_engine()
    fake_stt = _FakeSTT()
    fake_tts = _FakeTTS()
    engine.configure(stt=fake_stt, tts=fake_tts)

    engine._selected_input = _profile(
        "MacBook Pro Microphone", kind="builtin", is_input=True, is_output=False,
    )
    engine._selected_output = _profile(
        "MacBook Pro Speakers", kind="builtin", is_input=False, is_output=True,
    )

    engine._notify_output_change()

    assert fake_stt.calls == [{"output_type": "builtin", "hw_match": True}]
    assert fake_tts.bt_calls == [False]
    assert engine._bus.events  # type: ignore[attr-defined]
    name, data = engine._bus.events[-1]  # type: ignore[attr-defined]
    assert name == "audio_output_changed"
    assert data["output_name"] == "MacBook Pro Speakers"
    assert data["output_type"] == "builtin"
    assert data["input_type"] == "builtin"
    assert data["hw_match"] is True
    assert data["is_bluetooth"] is False


def test_notify_output_change_marks_bluetooth_for_tts():
    engine = _make_engine()
    fake_stt = _FakeSTT()
    fake_tts = _FakeTTS()
    engine.configure(stt=fake_stt, tts=fake_tts)

    engine._selected_input = _profile(
        "MacBook Pro Microphone", kind="builtin", is_input=True, is_output=False,
    )
    engine._selected_output = _profile(
        "AirPods Pro", kind="bluetooth", is_input=False, is_output=True,
    )

    engine._notify_output_change()

    assert fake_tts.bt_calls == [True]
    # Mismatched hardware: STT is told hw_match=False so it does NOT
    # try to flip VPIO back on.
    assert fake_stt.calls == [{"output_type": "bluetooth", "hw_match": False}]
    name, data = engine._bus.events[-1]  # type: ignore[attr-defined]
    assert name == "audio_output_changed"
    assert data["hw_match"] is False
    assert data["is_bluetooth"] is True


def test_notify_output_change_falls_back_to_native_synth_when_wrapper_lacks_method():
    engine = _make_engine()
    fake_stt = _FakeSTT()

    class _OnlyNativeTTS:
        def __init__(self) -> None:
            self._native_synth = _FakeTTS()

    only_native = _OnlyNativeTTS()
    engine.configure(stt=fake_stt, tts=only_native)

    engine._selected_input = _profile(
        "MacBook Pro Microphone", kind="builtin", is_input=True, is_output=False,
    )
    engine._selected_output = _profile(
        "AirPods Pro", kind="bluetooth", is_input=False, is_output=True,
    )

    engine._notify_output_change()

    assert only_native._native_synth.bt_calls == [True]


def test_notify_output_change_silent_when_no_selected_output():
    engine = _make_engine()
    fake_stt = _FakeSTT()
    fake_tts = _FakeTTS()
    engine.configure(stt=fake_stt, tts=fake_tts)
    engine._selected_output = None

    engine._notify_output_change()

    assert fake_stt.calls == []
    assert fake_tts.bt_calls == []
    assert engine._bus.events == []  # type: ignore[attr-defined]


def test_match_output_device_emits_notification_for_bluetooth_input():
    engine = _make_engine()
    fake_stt = _FakeSTT()
    fake_tts = _FakeTTS()
    engine.configure(stt=fake_stt, tts=fake_tts)

    bt_in = _profile("AirPods Pro", kind="bluetooth", is_input=True, is_output=False)
    bt_out = _profile("AirPods Pro", kind="bluetooth", is_input=False, is_output=True)
    engine._output_devices = [bt_out]
    engine._selected_input = bt_in

    matched = engine.match_output_device(bt_in)

    assert matched is bt_out
    assert engine._selected_output is bt_out
    # Bluetooth on both ends -> hw_match True, BT TTS flag set.
    assert fake_stt.calls == [{"output_type": "bluetooth", "hw_match": True}]
    assert fake_tts.bt_calls == [True]
    assert engine._bus.events  # type: ignore[attr-defined]
    assert engine._bus.events[-1][0] == "audio_output_changed"  # type: ignore[attr-defined]


def test_match_output_device_emits_notification_for_default_fallback():
    engine = _make_engine()
    fake_stt = _FakeSTT()
    fake_tts = _FakeTTS()
    engine.configure(stt=fake_stt, tts=fake_tts)

    builtin_in = _profile(
        "MacBook Pro Microphone", kind="builtin", is_input=True, is_output=False,
    )
    builtin_out = _profile(
        "MacBook Pro Speakers", kind="builtin", is_input=False, is_output=True,
    )
    builtin_out.is_default_output = True
    engine._output_devices = [builtin_out]
    engine._selected_input = builtin_in

    matched = engine.match_output_device(builtin_in)

    assert matched is builtin_out
    assert fake_stt.calls == [{"output_type": "builtin", "hw_match": True}]
    assert fake_tts.bt_calls == [False]
