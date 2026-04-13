"""Focused tests for the AtomState event/state contract."""

from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context.context_darwin import classify_activity
from core.state_manager import AtomState
from core.router.diagnostics_handler import DiagnosticsHandler
from core.state.event_bus import AtomRuntimeStateBridge
from voice.stt_google import STTGoogle
from ui.web_dashboard import WebDashboard
import core.system.system_monitor as system_monitor


class FakeBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, event: str, **data) -> None:
        self.events.append((event, data))

    def emit_fast(self, event: str, **data) -> None:
        self.events.append((event, data))

    def emit_long(self, event: str, **data) -> None:
        self.events.append((event, data))

    def on(self, event: str, handler) -> None:
        return None


class FakeSTT:
    def __init__(self) -> None:
        self.mic_name = "MacBook Microphone"
        self.backend_name = "macOS Native (SFSpeechRecognizer)"
        self._last_error = None
        self.speech_permission_status = "authorized"
        self.microphone_permission_status = "authorized"


class FakeTTS:
    _backend = "macOS Native (Daniel)"

    async def on_response(self, *_args, **_kwargs) -> None:
        return None

    async def on_partial_response(self, *_args, **_kwargs) -> None:
        return None

    async def speak_ack(self, *_args, **_kwargs) -> None:
        return None

    async def stop(self) -> None:
        return None


class FakeBrain:
    available = True


class FakeState:
    current = None


class FakeDisabledSTT:
    def __init__(self) -> None:
        self.mic_name = "Voice input unavailable"
        self.backend_name = "Disabled"
        self._last_error = "current process bundle lacks NSSpeechRecognitionUsageDescription"
        self.speech_permission_status = "bundle_missing_usage_description"
        self.microphone_permission_status = "unknown"

    async def async_start_listening(self, **_kw) -> None:
        return None

    async def on_state_changed(self, *_args, **_kwargs) -> None:
        return None


class FakeListeningState:
    current = AtomState.LISTENING
    always_listen = True

    async def transition(self, *_args, **_kwargs) -> None:
        return None

    async def on_tts_complete(self, *_args, **_kwargs) -> None:
        return None

    async def on_silence_timeout(self, *_args, **_kwargs) -> None:
        return None

    async def on_error(self, *_args, **_kwargs) -> None:
        return None


class FakeIndicator:
    def on_state_changed(self, **_kw) -> None:
        return None

    def add_log(self, *_args, **_kwargs) -> None:
        return None

    def show_hearing(self, *_args, **_kwargs) -> None:
        return None

    def clear_hearing(self) -> None:
        return None

    def set_last_query(self, *_args, **_kwargs) -> None:
        return None

    def set_language(self, *_args, **_kwargs) -> None:
        return None

    def set_last_intent(self, *_args, **_kwargs) -> None:
        return None

    def set_mic_name(self, *_args, **_kwargs) -> None:
        return None

    def broadcast_thinking_progress(self, *_args, **_kwargs) -> None:
        return None

    def set_last_latency_ms(self, *_args, **_kwargs) -> None:
        return None


class FakeRouter:
    async def on_speech(self, *_args, **_kwargs) -> None:
        return None

    def record_turn(self, *_args, **_kwargs) -> None:
        return None

    def _suggest_follow_up(self, *_args, **_kwargs):
        return None


class FakeMetrics:
    def inc(self, *_args, **_kwargs) -> None:
        return None

    def record_latency(self, *_args, **_kwargs) -> None:
        return None

    def snapshot(self) -> dict:
        return {}


def test_state_bridge_emits_diff_and_snapshot() -> None:
    bus = FakeBus()
    bridge = AtomRuntimeStateBridge(bus)

    bridge.patch_section("voice", {"status": "listening", "mic": "Studio Mic"}, source="test.voice")
    snap = bridge.emit_snapshot(source="test.snapshot")

    assert snap["voice"]["status"] == "listening"
    assert snap["voice"]["mic"] == "Studio Mic"
    assert any(event == "state.diff" for event, _ in bus.events)
    assert any(event == "state.snapshot" for event, _ in bus.events)
    print("  PASS: AtomRuntimeStateBridge emits diff + snapshot")


def test_diagnostics_self_check_uses_state_and_publishes() -> None:
    published: list[dict] = []
    handler = DiagnosticsHandler({"performance": {"mode": "auto"}})
    original_get_system_state = system_monitor.get_system_state
    system_monitor.get_system_state = lambda: {
        "foreground_app": "Cursor",
        "foreground_window_title": "main.py",
        "active_applications": ["Cursor", "Terminal"],
    }
    handler.configure(
        stt=FakeSTT(),
        tts=FakeTTS(),
        local_brain=FakeBrain(),
        state_snapshot_provider=lambda: {
            "system": {
                "cpu": 18.0,
                "memory_pct": 44.0,
                "battery_pct": 72.0,
                "charging": False,
                "disk_free_gb": 212.0,
                "top_processes": ["Cursor", "Terminal"],
            },
            "context": {
                "active_app": "Cursor",
                "window_title": "main.py",
                "activity_type": "coding",
                "confidence": 0.96,
                "idle_minutes": 0.4,
                "media": {"playing": False},
            },
            "voice": {
                "stt_engine": "macOS Native (SFSpeechRecognizer)",
                "tts_engine": "macOS Native (Daniel)",
                "mic": "MacBook Microphone",
                "error": None,
            },
            "mode": {
                "requested": "auto",
                "effective": "optimal",
                "reason": "Booted in Optimal due to battery power and an active development session.",
            },
        },
        report_publisher=lambda report: published.append(report),
    )
    try:
        report = handler.self_check_report()
    finally:
        system_monitor.get_system_state = original_get_system_state

    assert report["context"]["activity_type"] == "coding"
    assert report["mode"]["effective"] == "optimal"
    assert "Cursor" in report["running"]["summary"]
    assert published and published[-1]["summary_text"] == report["summary_text"]
    print("  PASS: Diagnostics self-check uses shared state and publishes")


def test_diagnostics_self_check_degrades_when_stt_unavailable() -> None:
    handler = DiagnosticsHandler({"performance": {"mode": "auto"}})
    original_get_system_state = system_monitor.get_system_state
    system_monitor.get_system_state = lambda: {
        "foreground_app": "Cursor",
        "foreground_window_title": "main.py",
        "active_applications": [{"name": "Cursor", "cpu_percent": 5.0, "memory_percent": 3.0}],
    }
    handler.configure(
        stt=FakeDisabledSTT(),
        tts=FakeTTS(),
        local_brain=FakeBrain(),
        state_snapshot_provider=lambda: {
            "system": {
                "cpu": 12.0,
                "memory_pct": 51.0,
                "battery_pct": 58.0,
                "charging": False,
                "disk_free_gb": 210.0,
                "top_processes": [{"name": "Cursor", "cpu_percent": 5.0, "memory_percent": 3.0}],
            },
            "context": {
                "active_app": "Cursor",
                "window_title": "main.py",
                "activity_type": "coding",
                "confidence": 0.95,
                "idle_minutes": 0.2,
                "media": {"playing": False, "summary": "No media playing"},
            },
            "voice": {
                "stt_engine": "Disabled",
                "tts_engine": "macOS Native (Daniel)",
                "mic": "Voice input unavailable",
                "error": "current process bundle lacks NSSpeechRecognitionUsageDescription",
                "permissions": {
                    "speech": "bundle_missing_usage_description",
                    "microphone": "unknown",
                },
            },
            "mode": {
                "requested": "auto",
                "effective": "optimal",
                "reason": "Switched to optimal due to battery power and the active development session.",
            },
            "health": {
                "warnings": ["stt_engine: Native unavailable"],
                "readiness": {"summary": {"failures": 1, "warnings": 1}},
            },
            "reasoning": {
                "why_this_mode": "Switched to optimal due to battery power and the active development session.",
            },
        },
    )
    try:
        report = handler.self_check_report()
    finally:
        system_monitor.get_system_state = original_get_system_state

    assert report["voice"]["stt_ok"] is False
    assert report["health_score"] < 10.0
    assert any("Speech input unavailable" in warning for warning in report["warnings"])
    assert report["summary_text"].startswith("System is degraded")
    print("  PASS: Diagnostics self-check reports degraded state when STT is unavailable")


def test_diagnostics_mode_status_uses_shared_state() -> None:
    handler = DiagnosticsHandler({"performance": {"mode": "auto"}})
    handler.configure(
        state_snapshot_provider=lambda: {
            "mode": {
                "requested": "auto",
                "effective": "optimal",
                "reason": "Switched to optimal due to battery power and the active development session.",
                "assistant_mode": "hybrid",
            },
            "reasoning": {
                "why_this_mode": "Switched to optimal due to battery power and the active development session.",
            },
        },
    )
    text = handler.mode_status()
    assert "optimal mode" in text.lower()
    assert "battery power" in text.lower()
    assert "assistant mode is hybrid" in text.lower()
    print("  PASS: Mode status is grounded in shared state")


def test_diagnostics_detailed_status_mentions_warnings_and_processes() -> None:
    handler = DiagnosticsHandler({"performance": {"mode": "auto"}})
    original_get_system_state = system_monitor.get_system_state
    system_monitor.get_system_state = lambda: {
        "foreground_app": "Cursor",
        "foreground_window_title": "main.py",
        "active_applications": [
            {"name": "Cursor Helper", "cpu_percent": 18.0, "memory_percent": 6.0},
            {"name": "Python", "cpu_percent": 8.0, "memory_percent": 4.0},
        ],
    }
    handler.configure(
        stt=FakeDisabledSTT(),
        tts=FakeTTS(),
        local_brain=FakeBrain(),
        state_snapshot_provider=lambda: {
            "system": {
                "cpu": 16.0,
                "memory_pct": 60.0,
                "battery_pct": 55.0,
                "charging": False,
                "disk_free_gb": 300.0,
            },
            "context": {
                "active_app": "Cursor",
                "window_title": "main.py",
                "activity_type": "coding",
                "confidence": 0.95,
                "idle_minutes": 0.5,
                "media": {"playing": False, "summary": "No media playing"},
            },
            "voice": {
                "stt_engine": "Disabled",
                "tts_engine": "macOS Native (Daniel)",
                "mic": "Voice input unavailable",
                "error": "current process bundle lacks NSSpeechRecognitionUsageDescription",
            },
            "mode": {
                "requested": "auto",
                "effective": "optimal",
                "reason": "Switched to optimal due to battery power and the active development session.",
            },
            "health": {
                "warnings": ["stt_engine: Native unavailable"],
                "readiness_summary": "1 system needs attention.",
                "readiness": {"summary": {"failures": 1, "warnings": 0}},
            },
        },
    )
    try:
        text = handler.detailed_status()
    finally:
        system_monitor.get_system_state = original_get_system_state

    assert "Warnings first:" in text
    assert "Top processes:" in text
    assert "Cursor Helper" in text
    assert "Speech input is unavailable" in text
    print("  PASS: Detailed status includes warnings and top processes")


async def test_dashboard_state_contract() -> None:
    dashboard = WebDashboard(auto_open=False)
    sent: list[dict] = []

    async def _capture(data: dict) -> None:
        sent.append(data)

    dashboard._broadcast = _capture  # type: ignore[method-assign]

    await dashboard.on_state_diff(diff={"voice": {"status": "processing"}}, source="test.diff")
    await dashboard.on_state_snapshot(snapshot={"mode": {"effective": "optimal"}}, source="test.snapshot")

    assert sent[0]["type"] == "state_diff"
    assert sent[0]["diff"]["voice"]["status"] == "processing"
    assert sent[1]["type"] == "state_snapshot"
    assert sent[1]["snapshot"]["mode"]["effective"] == "optimal"
    print("  PASS: Dashboard forwards state diff/snapshot contract")


def test_macos_activity_classifier() -> None:
    assert classify_activity("Cursor", "main.py", idle_minutes=0.1) == ("coding", 0.95)
    assert classify_activity("Google Chrome", "YouTube - demo", idle_minutes=0.1) == ("media", 0.9)
    assert classify_activity("Safari", "Google Meet", idle_minutes=0.1) == ("meeting", 0.88)
    assert classify_activity("Arc", "Docs", idle_minutes=0.1) == ("browsing", 0.8)
    print("  PASS: macOS activity classifier covers coding/media/meeting/browsing")


async def test_google_stt_emits_typed_voice_events() -> None:
    bus = FakeBus()
    stt = STTGoogle(bus, FakeState(), config={"stt": {"bilingual": False}})
    stt._loop = asyncio.get_running_loop()
    stt.mic_name = "USB Mic"
    stt._last_confidence = 0.91
    stt._detected_language = "en"

    with patch.object(stt, "_listen_loop", return_value="open notes"):
        await stt.start_listening()

    event_names = [event for event, _ in bus.events]
    assert "voice.final" in event_names
    assert "speech_final" in event_names
    final_payload = next(data for event, data in bus.events if event == "voice.final")
    assert final_payload["engine"] == "Google Online"
    assert final_payload["mic"] == "USB Mic"
    assert final_payload["text"] == "open notes"
    print("  PASS: Google STT emits typed voice.final metadata")


async def test_restart_listening_uses_async_hook_when_available() -> None:
    import core.boot.wiring as wiring

    calls: list[str] = []

    state = FakeListeningState()
    stt = FakeDisabledSTT()

    async def _track_start(**_kw) -> None:
        calls.append("async_start_listening")

    stt.async_start_listening = _track_start  # type: ignore[method-assign]

    def _immediate_create_task(coro):
        return asyncio.get_running_loop().create_task(coro)

    # FakeBus.on is a no-op, so invoke the wiring function directly by re-reading the closure
    registered: list[tuple[str, object]] = []

    class CaptureBus(FakeBus):
        def on(self, event: str, handler) -> None:
            registered.append((event, handler))

    capture_bus = CaptureBus()
    wiring.wire_events(
        bus=capture_bus,
        state=state,
        stt=stt,
        tts=FakeTTS(),
        router=FakeRouter(),
        indicator=FakeIndicator(),
        cache=type("Cache", (), {"put": staticmethod(lambda *_a, **_k: None)})(),
        memory=type(
            "Memory",
            (),
            {
                "add": staticmethod(lambda *_a, **_k: asyncio.sleep(0)),
                "persist": staticmethod(lambda: None),
            },
        )(),
        metrics=FakeMetrics(),
        config={},
        behavior=object(),
        v3=False,
        v4=False,
    )
    restart_handler = next(handler for event, handler in registered if event == "restart_listening")

    with patch.object(wiring.asyncio, "create_task", _immediate_create_task):
        await restart_handler()
        await asyncio.sleep(0.15)

    assert calls == ["async_start_listening"]
    print("  PASS: restart_listening uses async_start_listening when available")


async def run_all() -> None:
    print("\n=== Atom State Contract Tests ===\n")
    test_state_bridge_emits_diff_and_snapshot()
    test_diagnostics_self_check_uses_state_and_publishes()
    test_diagnostics_self_check_degrades_when_stt_unavailable()
    test_diagnostics_mode_status_uses_shared_state()
    test_diagnostics_detailed_status_mentions_warnings_and_processes()
    test_macos_activity_classifier()
    await test_dashboard_state_contract()
    await test_google_stt_emits_typed_voice_events()
    await test_restart_listening_uses_async_hook_when_available()
    print("\n=== ALL TESTS PASSED ===\n")


if __name__ == "__main__":
    asyncio.run(run_all())
