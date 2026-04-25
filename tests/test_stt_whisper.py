"""Sprint B5 -- regression tests for WhisperSTT + voice_pipeline factory.

The whisper.cpp / sounddevice / webrtcvad native deps are unlikely to
be present in CI, so every test here injects mocks via
``monkeypatch.setattr`` against the module-level singletons
(`_pwc_model`, `_sd`, `_webrtcvad`, `_np`) defined in
``voice/stt_whisper.py``. We never load the real GGML model.
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ── Helpers ──────────────────────────────────────────────────────


class FakeBus:
    """Sync stand-in for AsyncEventBus -- captures everything emitted."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def emit(self, event: str, **kw) -> None:
        self.calls.append((event, kw))

    def emit_fast(self, event: str, **kw) -> None:
        self.calls.append((event, kw))

    def on(self, event: str, handler) -> None:  # pragma: no cover - unused
        return None


class _FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeWhisperModel:
    def __init__(self, *_args, **_kw) -> None:
        self.calls: list[bytes] = []

    def transcribe(self, audio, **_kw):  # noqa: D401
        self.calls.append(bytes(audio.tobytes()) if hasattr(audio, "tobytes")
                          else bytes(audio))
        return [_FakeSegment("hello there")]


class _FakeVad:
    def __init__(self, _aggr: int) -> None:
        self.aggr = _aggr

    def is_speech(self, _frame: bytes, _rate: int) -> bool:
        return True


class _FakeStream:
    def __init__(self, **kw) -> None:
        self.kw = kw
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


def _install_stt_whisper_mocks(monkeypatch):
    """Patch the soft-imported native deps inside voice.stt_whisper."""
    import numpy as _np_real  # type: ignore[import-untyped]
    from voice import stt_whisper as mod

    fake_pwc = types.SimpleNamespace(Model=_FakeWhisperModel)
    fake_sd = types.SimpleNamespace(RawInputStream=_FakeStream)
    fake_webrtcvad = types.SimpleNamespace(Vad=_FakeVad)

    monkeypatch.setattr(mod, "_pwc_model", fake_pwc, raising=False)
    monkeypatch.setattr(mod, "_sd", fake_sd, raising=False)
    monkeypatch.setattr(mod, "_webrtcvad", fake_webrtcvad, raising=False)
    monkeypatch.setattr(mod, "_np", _np_real, raising=False)
    return mod


def _make_config(tmp_path: Path, **stt_overrides) -> dict:
    """Return a minimal settings.json-shaped config; the model file
    is created so ``preload`` doesn't bail on the existence check."""
    model_path = tmp_path / "ggml-test.bin"
    model_path.write_bytes(b"\x00" * 16)
    base = {
        "stt": {
            "engine": "whisper_cpp",
            "whisper_model_path": str(model_path),
            "whisper_n_threads": 1,
            "whisper_partial_interval_s": 0.05,
            "whisper_trailing_silence_s": 0.05,
            "whisper_max_utterance_s": 5.0,
            "whisper_vad_aggressiveness": 1,
        },
    }
    base["stt"].update(stt_overrides)
    return base


# ── 1. Lifecycle ────────────────────────────────────────────────


def test_whisper_stt_preload_succeeds_when_deps_present(tmp_path, monkeypatch):
    mod = _install_stt_whisper_mocks(monkeypatch)
    config = _make_config(tmp_path)

    bus = FakeBus()
    state = MagicMock()
    stt = mod.WhisperSTT(bus, state, config)

    assert stt.is_available is False
    ok = stt.preload()
    assert ok is True
    assert stt.is_available is True
    assert "whisper.cpp" in stt.backend_name


def test_whisper_stt_preload_fails_when_pywhispercpp_missing(
    tmp_path, monkeypatch,
):
    mod = _install_stt_whisper_mocks(monkeypatch)
    monkeypatch.setattr(mod, "_pwc_model", None, raising=False)
    config = _make_config(tmp_path)

    stt = mod.WhisperSTT(FakeBus(), MagicMock(), config)

    assert stt.preload() is False
    assert stt.is_available is False
    assert "pywhispercpp" in (stt._last_error or "")


def test_whisper_stt_preload_fails_when_model_missing(tmp_path, monkeypatch):
    mod = _install_stt_whisper_mocks(monkeypatch)
    config = _make_config(tmp_path)
    Path(config["stt"]["whisper_model_path"]).unlink()

    stt = mod.WhisperSTT(FakeBus(), MagicMock(), config)

    assert stt.preload() is False
    assert stt.is_available is False
    assert "Whisper model not found" in (stt._last_error or "")


# ── 2. Bus emission shape (matches NativeSTT contract) ──────────


@pytest.mark.asyncio
async def test_async_start_listening_emits_speech_events(tmp_path, monkeypatch):
    mod = _install_stt_whisper_mocks(monkeypatch)
    config = _make_config(tmp_path)

    bus = FakeBus()
    state = MagicMock()
    stt = mod.WhisperSTT(bus, state, config)
    assert stt.preload() is True

    loop = asyncio.get_running_loop()
    listen_task = asyncio.create_task(stt.async_start_listening())
    await asyncio.sleep(0.05)

    stt._emit_partial("hi")
    await asyncio.sleep(0.05)
    stt._emit_final("hello there")
    await asyncio.sleep(0.05)

    stt.stop_listening()
    await asyncio.sleep(0.05)
    listen_task.cancel()
    try:
        await listen_task
    except asyncio.CancelledError:
        pass

    events = {evt for evt, _ in bus.calls}
    assert "speech_partial" in events
    assert "speech_final" in events

    final_payloads = [d for evt, d in bus.calls if evt == "speech_final"]
    assert any(p.get("language") == "en" for p in final_payloads), \
        f"speech_final missing language=en, got {final_payloads}"

    voice_finals = [d for evt, d in bus.calls if evt == "voice.final"]
    assert voice_finals, "voice.final should fire alongside speech_final"
    assert "engine" in voice_finals[0] and "mic" in voice_finals[0]


def test_echo_guard_suppresses_emission(tmp_path, monkeypatch):
    """If TTS.is_echo says yes, we must NOT emit -- otherwise ATOM
    will hear its own voice and Jarvis-loop again."""
    mod = _install_stt_whisper_mocks(monkeypatch)
    config = _make_config(tmp_path)

    bus = FakeBus()
    stt = mod.WhisperSTT(bus, MagicMock(), config)
    stt.preload()

    seen = []
    stt._on_partial = lambda t: seen.append(("p", t))
    stt._on_final = lambda t: seen.append(("f", t))

    stt._echo_guard = lambda _t: True

    stt._emit_partial("ok boss")
    stt._emit_final("ok boss")

    assert seen == [], (
        "Echo-guarded text reached the on_partial/on_final callback"
    )
    assert all(evt != "speech_partial" and evt != "speech_final"
               for evt, _ in bus.calls)


# ── 3. is_whisper_available ────────────────────────────────────


def test_is_whisper_available_false_when_deps_missing(tmp_path, monkeypatch):
    mod = _install_stt_whisper_mocks(monkeypatch)
    monkeypatch.setattr(mod, "_pwc_model", None, raising=False)

    config = _make_config(tmp_path)
    assert mod.is_whisper_available(config) is False


def test_is_whisper_available_false_when_model_missing(tmp_path, monkeypatch):
    mod = _install_stt_whisper_mocks(monkeypatch)
    config = _make_config(tmp_path)
    Path(config["stt"]["whisper_model_path"]).unlink()

    assert mod.is_whisper_available(config) is False


def test_is_whisper_available_true_with_full_setup(tmp_path, monkeypatch):
    mod = _install_stt_whisper_mocks(monkeypatch)
    config = _make_config(tmp_path)

    assert mod.is_whisper_available(config) is True


# ── 4. Diagnostics -- mirrors NativeSTT.get_diagnostics shape ──


def test_get_diagnostics_shape(tmp_path, monkeypatch):
    mod = _install_stt_whisper_mocks(monkeypatch)
    config = _make_config(tmp_path)

    stt = mod.WhisperSTT(FakeBus(), MagicMock(), config)
    stt.preload()

    diag = stt.get_diagnostics()
    expected = {
        "engine", "available", "listening", "running_async",
        "model_path", "model_present", "tap_buffer_count",
        "last_audio_rms_db", "since_last_speech_s", "last_error",
    }
    assert expected.issubset(diag.keys()), \
        f"get_diagnostics missing keys, got {sorted(diag)}"
    assert diag["available"] is True
    assert diag["model_present"] is True


# ── 5. voice_pipeline engine selection ─────────────────────────


def _stub_disabled_native(monkeypatch):
    """Force NativeSTT 'unsupported' so the macOS branch can flow
    through the whisper.cpp path without touching CoreFoundation."""
    from voice import voice_pipeline as vp

    def _fake_native(self):
        return None, "native disabled in test"

    monkeypatch.setattr(vp.VoicePipeline, "_build_native_stt", _fake_native)


def _stub_whisper_factory(monkeypatch, *, ok: bool, reason: str = ""):
    from voice import voice_pipeline as vp

    sentinel = MagicMock(name="WhisperSTT")
    sentinel.async_start_listening = MagicMock()
    sentinel.shutdown = MagicMock()

    def _fake_whisper(self):
        if ok:
            return sentinel, ""
        return None, reason

    monkeypatch.setattr(vp.VoicePipeline, "_build_whisper_cpp_stt", _fake_whisper)
    return sentinel


def _make_pipeline_config(engine: str) -> dict:
    return {"stt": {"engine": engine}, "tts": {"engine": "macos_native"}}


@pytest.mark.skipif(sys.platform != "darwin",
                    reason="macOS branch under test")
def test_pipeline_selects_whisper_when_engine_pref_is_whisper_cpp(
    monkeypatch,
):
    from core.async_event_bus import AsyncEventBus
    from voice.voice_pipeline import VoicePipeline

    sentinel = _stub_whisper_factory(monkeypatch, ok=True)

    bus = AsyncEventBus()
    state = MagicMock()
    config = _make_pipeline_config("whisper_cpp")
    pipeline = VoicePipeline(bus, state, config)
    pipeline._build_stt()

    assert pipeline.stt is sentinel, \
        "engine=whisper_cpp should route to WhisperSTT"
    assert "whisper.cpp" in pipeline.stt_runtime_label.lower()


@pytest.mark.skipif(sys.platform != "darwin",
                    reason="macOS branch under test")
def test_pipeline_aliases_resolve_to_whisper_cpp(monkeypatch):
    from core.async_event_bus import AsyncEventBus
    from voice.voice_pipeline import VoicePipeline

    sentinel = _stub_whisper_factory(monkeypatch, ok=True)

    for alias in ("whisper", "whispercpp", "whisper.cpp"):
        bus = AsyncEventBus()
        state = MagicMock()
        config = _make_pipeline_config(alias)
        pipeline = VoicePipeline(bus, state, config)
        pipeline._build_stt()
        assert pipeline.stt is sentinel, f"alias {alias} should map to whisper_cpp"


@pytest.mark.skipif(sys.platform != "darwin",
                    reason="macOS branch under test")
def test_pipeline_auto_prefers_whisper_when_available(monkeypatch):
    from core.async_event_bus import AsyncEventBus
    from voice.voice_pipeline import VoicePipeline

    sentinel = _stub_whisper_factory(monkeypatch, ok=True)
    _stub_disabled_native(monkeypatch)

    bus = AsyncEventBus()
    state = MagicMock()
    config = _make_pipeline_config("auto")
    pipeline = VoicePipeline(bus, state, config)
    pipeline._build_stt()

    assert pipeline.stt is sentinel, \
        "engine=auto should pick WhisperSTT before NativeSTT"


@pytest.mark.skipif(sys.platform != "darwin",
                    reason="macOS branch under test")
def test_pipeline_auto_disables_stt_when_whisper_missing(monkeypatch):
    """Sprint K hardening: engine=auto must NOT silently fall back to
    SFSpeechRecognizer when whisper.cpp is missing -- it must surface a
    disabled STT so Boss is told to install whisper.cpp."""
    from core.async_event_bus import AsyncEventBus
    from voice import voice_pipeline as vp
    from voice.voice_pipeline import VoicePipeline, _DisabledSTT

    _stub_whisper_factory(monkeypatch, ok=False, reason="model missing")

    native_stub = MagicMock(name="NativeSTT")
    native_stub.async_start_listening = MagicMock()
    native_stub.shutdown = MagicMock()
    monkeypatch.setattr(
        vp.VoicePipeline, "_build_native_stt",
        lambda self: (native_stub, ""),
    )

    bus = AsyncEventBus()
    state = MagicMock()
    config = _make_pipeline_config("auto")
    pipeline = VoicePipeline(bus, state, config)
    pipeline._build_stt()

    assert isinstance(pipeline.stt, _DisabledSTT), (
        "engine=auto must surface a disabled STT (Sprint K) instead of a "
        f"silent native fallback -- got {type(pipeline.stt).__name__}"
    )
    assert "whisper.cpp" in pipeline.stt_runtime_label.lower()


@pytest.mark.skipif(sys.platform != "darwin",
                    reason="macOS branch under test")
def test_pipeline_whisper_cpp_disables_stt_when_unavailable(monkeypatch):
    """Sprint K hardening: engine=whisper_cpp must NOT fall back to
    SFSpeechRecognizer when whisper.cpp can't load."""
    from core.async_event_bus import AsyncEventBus
    from voice import voice_pipeline as vp
    from voice.voice_pipeline import VoicePipeline, _DisabledSTT

    _stub_whisper_factory(monkeypatch, ok=False, reason="model missing")
    native_stub = MagicMock(name="NativeSTT")
    native_stub.async_start_listening = MagicMock()
    native_stub.shutdown = MagicMock()
    monkeypatch.setattr(
        vp.VoicePipeline, "_build_native_stt",
        lambda self: (native_stub, ""),
    )

    bus = AsyncEventBus()
    state = MagicMock()
    config = _make_pipeline_config("whisper_cpp")
    pipeline = VoicePipeline(bus, state, config)
    pipeline._build_stt()

    assert isinstance(pipeline.stt, _DisabledSTT)
    assert "whisper.cpp" in pipeline.stt_runtime_label.lower()


# ── 6. Schema validation ────────────────────────────────────────


def test_settings_schema_accepts_whisper_cpp_engine():
    from core.config_schema import validate_config

    cfg = {"stt": {"engine": "whisper_cpp"}}
    errs = validate_config(cfg)
    assert all("stt.engine" not in e for e in errs), \
        f"validate_config rejected stt.engine=whisper_cpp: {errs}"
