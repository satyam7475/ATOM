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
