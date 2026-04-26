"""Sprint Ω.6.C — regression tests for the WhisperKit launch command.

Locks down the fix for the 2026-04-26 outage where ``whisperkit-cli`` 0.18.0
rejected the bare ``--download`` flag and ATOM waited the full
``startup_timeout_s`` (60 s) for a port that never bound.
"""

from __future__ import annotations

import asyncio
import subprocess
from typing import Any

import pytest


@pytest.fixture()
def stt(monkeypatch: pytest.MonkeyPatch) -> Any:
    from voice import stt_whisperkit as m

    # Bypass the real binary lookup so the test runs in any CI.
    monkeypatch.setattr(m, "_whisperkit_cli_path", lambda: "/fake/whisperkit-cli")

    class _Bus:
        def emit(self, *a: Any, **k: Any) -> None: ...
        def emit_fast(self, *a: Any, **k: Any) -> None: ...
        def emit_long(self, *a: Any, **k: Any) -> None: ...
        def on(self, *a: Any, **k: Any) -> None: ...

    class _State:
        class _S:
            value = "idle"

        current = _S()

    return m, _Bus, _State


def _make_stt(stt_module: Any, bus_cls: Any, state_cls: Any, **wk_overrides: Any) -> Any:
    config = {
        "stt": {
            "whisperkit": {
                "model": "whisper-large-v3-v20240930_turbo_632MB",
                "host": "127.0.0.1",
                "port": 50060,
                "auto_download": True,
                "startup_timeout_s": 60.0,
                "model_dir": None,
                **wk_overrides,
            }
        }
    }
    return stt_module.WhisperKitSTT(bus_cls(), state_cls(), config=config)


def test_launch_cmd_does_not_pass_download_flag(stt: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """whisperkit-cli 0.18.0 rejects --download. Must not appear in the cmd
    even when ``stt.whisperkit.auto_download=true`` (legacy config)."""
    m, bus_cls, state_cls = stt
    backend = _make_stt(m, bus_cls, state_cls, auto_download=True)

    captured: dict[str, Any] = {}

    class _StubProc:
        def __init__(self, cmd: list[str], **_: Any) -> None:
            captured["cmd"] = cmd

        def poll(self) -> int | None:  # never died
            return None

    monkeypatch.setattr(m, "_port_is_open", lambda *a, **k: False)
    monkeypatch.setattr(subprocess, "Popen", _StubProc)

    backend._maybe_start_serve()

    cmd = captured["cmd"]
    assert "--download" not in cmd, (
        f"--download must not be in launch cmd (whisperkit-cli 0.18+ "
        f"rejects it). Got: {cmd}"
    )
    assert "--model" in cmd
    assert cmd[cmd.index("--model") + 1] == "whisper-large-v3-v20240930_turbo_632MB"
    assert cmd[:2] == ["/fake/whisperkit-cli", "serve"]


def test_launch_cmd_routes_model_dir_to_download_model_path(
    stt: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``model_dir`` belongs in ``--download-model-path``, NOT
    ``--model-prefix`` (which only accepts the variant tag openai|distil)."""
    m, bus_cls, state_cls = stt
    backend = _make_stt(m, bus_cls, state_cls, model_dir="/tmp/whisperkit_models")

    captured: dict[str, Any] = {}

    class _StubProc:
        def __init__(self, cmd: list[str], **_: Any) -> None:
            captured["cmd"] = cmd

        def poll(self) -> int | None:
            return None

    monkeypatch.setattr(m, "_port_is_open", lambda *a, **k: False)
    monkeypatch.setattr(subprocess, "Popen", _StubProc)

    backend._maybe_start_serve()

    cmd = captured["cmd"]
    assert "--download-model-path" in cmd
    assert (
        cmd[cmd.index("--download-model-path") + 1] == "/tmp/whisperkit_models"
    )
    assert "--model-prefix" not in cmd, (
        "--model-prefix is for variant tags (openai|distil), not directories"
    )


def test_wait_for_serve_ready_short_circuits_on_early_death(
    stt: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If whisperkit-cli exits before binding, surface the error in <1 s
    instead of waiting the full 60 s timeout."""
    m, bus_cls, state_cls = stt
    backend = _make_stt(m, bus_cls, state_cls, startup_timeout_s=60.0)

    class _DeadProc:
        returncode = 1

        def __init__(self) -> None:
            class _Stdout:
                def read(self_inner) -> bytes:
                    return b"Error: Unknown option '--download'\nUsage: whisperkit-cli serve <options>\n"

            self.stdout = _Stdout()

        def poll(self) -> int:
            return 1

    backend._serve_proc = _DeadProc()  # type: ignore[assignment]
    monkeypatch.setattr(m, "_port_is_open", lambda *a, **k: False)

    with pytest.raises(RuntimeError) as exc_info:
        backend._wait_for_serve_ready()

    msg = str(exc_info.value)
    assert "exited early" in msg
    assert "returncode=1" in msg
    assert "Unknown option" in msg, (
        f"early-death error must include subprocess output for diagnosis; got: {msg}"
    )


def test_wait_for_serve_ready_succeeds_when_port_and_health_open(
    stt: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity: positive path still works."""
    import urllib.request

    m, bus_cls, state_cls = stt
    backend = _make_stt(m, bus_cls, state_cls, startup_timeout_s=5.0)

    class _AliveProc:
        def poll(self) -> int | None:
            return None

    backend._serve_proc = _AliveProc()  # type: ignore[assignment]
    monkeypatch.setattr(m, "_port_is_open", lambda *a, **k: True)

    class _OkResp:
        status = 200

        def __enter__(self) -> "_OkResp":
            return self

        def __exit__(self, *a: Any) -> None:
            pass

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _OkResp())

    backend._wait_for_serve_ready()


def test_maybe_start_serve_reaps_unhealthy_stale_listener(
    stt: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bound but unhealthy stale serve must not cost the full boot timeout."""
    m, bus_cls, state_cls = stt
    backend = _make_stt(m, bus_cls, state_cls)
    captured: dict[str, Any] = {}
    port_checks = iter([True, False, False])

    class _StubProc:
        def __init__(self, cmd: list[str], **_: Any) -> None:
            captured["cmd"] = cmd

        def poll(self) -> int | None:
            return None

    monkeypatch.setattr(m, "_port_is_open", lambda *a, **k: next(port_checks))
    monkeypatch.setattr(backend, "_serve_health_ok", lambda **k: False)
    monkeypatch.setattr(backend, "_reap_stale_serve_on_port", lambda: True)
    monkeypatch.setattr(subprocess, "Popen", _StubProc)

    backend._maybe_start_serve()

    assert captured["cmd"][:2] == ["/fake/whisperkit-cli", "serve"]


def test_maybe_start_serve_attaches_to_healthy_listener(
    stt: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    m, bus_cls, state_cls = stt
    backend = _make_stt(m, bus_cls, state_cls)

    monkeypatch.setattr(m, "_port_is_open", lambda *a, **k: True)
    monkeypatch.setattr(backend, "_serve_health_ok", lambda **k: True)
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *a, **k: pytest.fail("healthy existing serve should be reused"),
    )

    backend._maybe_start_serve()


def test_diagnostic_self_input_is_not_routed(
    stt: Any,
) -> None:
    m, bus_cls, state_cls = stt

    class _Bus(bus_cls):
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, Any]]] = []

        def emit(self, event: str, **kw: Any) -> None:
            self.events.append((event, kw))

    bus = _Bus()
    backend = m.WhisperKitSTT(bus, state_cls(), config={"stt": {"whisperkit": {}}})

    backend._emit_final("rack pressure mode on at 82% snippet budget reduced to 1%")

    assert bus.events == []


def test_atom_self_speech_final_is_not_routed(
    stt: Any,
) -> None:
    m, bus_cls, state_cls = stt

    class _Bus(bus_cls):
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, Any]]] = []

        def emit(self, event: str, **kw: Any) -> None:
            self.events.append((event, kw))

    bus = _Bus()
    backend = m.WhisperKitSTT(bus, state_cls(), config={"stt": {"whisperkit": {}}})

    backend._emit_final("What do you need?")
    backend._emit_final("One moment.")
    backend._emit_final("Working on it.")

    assert bus.events == []


def test_atom_self_prefix_is_stripped_from_owner_suffix(
    stt: Any,
) -> None:
    m, bus_cls, state_cls = stt

    class _Bus(bus_cls):
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, Any]]] = []

        def emit(self, event: str, **kw: Any) -> None:
            self.events.append((event, kw))

    bus = _Bus()
    backend = m.WhisperKitSTT(bus, state_cls(), config={"stt": {"whisperkit": {}}})

    backend._emit_final(
        "I'm it, boss. System is degraded, boss. Can you see me, Adtan? Okay."
    )

    assert bus.events == [
        ("speech_final", {"text": "Can you see me, atom?", "language": "auto"}),
    ]


@pytest.mark.asyncio
async def test_tts_tail_mute_drops_delayed_final(
    stt: Any,
) -> None:
    m, bus_cls, state_cls = stt

    class _Bus(bus_cls):
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, Any]]] = []

        def emit(self, event: str, **kw: Any) -> None:
            self.events.append((event, kw))

    bus = _Bus()
    backend = m.WhisperKitSTT(bus, state_cls(), config={"stt": {"whisperkit": {}}})

    await backend.on_tts_complete()
    backend._emit_final("I'm good, Boss. Ready for you.")

    assert bus.events == []


def test_noise_gate_blocks_quiet_whisperkit_frames(
    stt: Any,
) -> None:
    m, bus_cls, state_cls = stt
    backend = m.WhisperKitSTT(
        bus_cls(),
        state_cls(),
        config={
            "stt": {
                "noise_floor_dbfs": -45.0,
                "noise_gate_consecutive": 3,
                "whisperkit": {},
            },
        },
    )

    assert backend._noise_gate_blocks(-60.0) is False
    assert backend._noise_gate_blocks(-60.0) is False
    assert backend._noise_gate_blocks(-60.0) is True
    assert backend._noise_gate_dropped_total == 1
    assert backend._noise_gate_blocks(-35.0) is False


def test_whisperkit_min_utterance_uses_config(
    stt: Any,
) -> None:
    m, bus_cls, state_cls = stt
    backend = m.WhisperKitSTT(
        bus_cls(),
        state_cls(),
        config={
            "stt": {
                "min_audio_duration_s": 0.55,
                "whisperkit": {},
            },
        },
    )

    backend._utterance_frames = [b"\x00" * 960] * 10  # 300 ms

    assert backend._flush_utterance(force=True) == ""


@pytest.mark.asyncio
async def test_async_start_listening_does_not_duplicate_speech_events(
    stt: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WhisperKit callbacks add voice.* metadata; _emit_* owns speech_*."""
    m, _, state_cls = stt

    class _Bus:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []

        def emit(self, event: str, **kw: Any) -> None:
            self.calls.append((event, kw))

        def emit_fast(self, event: str, **kw: Any) -> None:
            self.calls.append((event, kw))

        def emit_long(self, event: str, **kw: Any) -> None:
            self.calls.append((event, kw))

        def on(self, *a: Any, **k: Any) -> None:
            return None

    bus = _Bus()
    backend = m.WhisperKitSTT(
        bus,
        state_cls(),
        config={
            "stt": {
                "whisperkit": {
                    "model": "whisper-large-v3-v20240930_turbo_632MB",
                    "host": "127.0.0.1",
                    "port": 50060,
                    "startup_timeout_s": 5.0,
                },
            },
        },
    )

    def _fake_start_listening(*, loop, on_final, on_partial) -> bool:
        backend._on_final = on_final
        backend._on_partial = on_partial
        backend._listening = True
        return True

    monkeypatch.setattr(backend, "start_listening", _fake_start_listening)

    task = asyncio.create_task(backend.async_start_listening())
    await asyncio.sleep(0)

    backend._emit_partial("hello")
    backend._emit_final("hello there")
    await asyncio.sleep(0.05)

    backend._running_async = False
    backend._listening = False
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    event_names = [event for event, _ in bus.calls]
    assert event_names.count("speech_partial") == 1
    assert event_names.count("speech_final") == 1
    assert event_names.count("voice.partial") == 1
    assert event_names.count("voice.final") == 1
