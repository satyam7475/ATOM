"""Focused tests for native macOS STT bundle safety guards."""

from __future__ import annotations

import os
import plistlib
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import voice.stt_macos as stt_macos


class _FakeBundle:
    def __init__(self, info: dict | None) -> None:
        self._info = info

    def infoDictionary(self):
        return self._info


class _FakeNSBundle:
    @staticmethod
    def mainBundle():
        return _FakeBundle({})


class _FakeLocaleBuilder:
    def initWithLocaleIdentifier_(self, _locale):
        return object()


class _FakeNSLocale:
    @staticmethod
    def alloc():
        return _FakeLocaleBuilder()


class _FakeFoundation:
    NSBundle = _FakeNSBundle
    NSLocale = _FakeNSLocale


class _ExplodingSpeech:
    class SFSpeechRecognizer:
        @staticmethod
        def authorizationStatus():
            raise AssertionError(
                "authorizationStatus should not be called without usage description"
            )

        @staticmethod
        def alloc():
            raise AssertionError(
                "recognizer allocation should not run without usage description"
            )


def test_native_stt_launch_supported_requires_bundle_usage_description() -> None:
    with patch.object(stt_macos, "_HAS_SPEECH", True), \
            patch.object(stt_macos, "_Speech", _ExplodingSpeech), \
            patch.object(stt_macos, "_Foundation", _FakeFoundation), \
            patch.object(stt_macos.sys, "platform", "darwin"):
        ok, reason = stt_macos.native_stt_launch_supported()

    assert ok is False
    assert reason == "current process bundle lacks NSSpeechRecognitionUsageDescription"
    print("  PASS: native STT refuses invalid bundle before hitting Speech APIs")


def test_native_stt_preload_fails_cleanly_when_bundle_is_invalid() -> None:
    stt = stt_macos.NativeSTT(bus=None, state=None, config={"stt": {}})

    with patch.object(stt_macos, "_HAS_SPEECH", True), \
            patch.object(stt_macos, "_Speech", _ExplodingSpeech), \
            patch.object(stt_macos, "_Foundation", _FakeFoundation), \
            patch.object(stt_macos.sys, "platform", "darwin"):
        ok = stt.preload()

    assert ok is False
    assert stt._last_error == "current process bundle lacks NSSpeechRecognitionUsageDescription"
    assert stt.speech_permission_status == "bundle_missing_usage_description"
    print("  PASS: preload degrades safely instead of crashing on invalid bundle")


def test_bundle_has_usage_reads_atom_app_plist_when_main_bundle_is_python() -> None:
    """Venv ``exec`` makes NSBundle.mainBundle Python.framework; ATOM_APP_BUNDLE plist supplies usage strings."""
    info_plist = {
        "CFBundleIdentifier": "com.test.atom",
        "NSSpeechRecognitionUsageDescription": "Speech",
        "NSMicrophoneUsageDescription": "Mic",
    }
    with tempfile.TemporaryDirectory() as td:
        app = Path(td) / "ATOM.app"
        plist_path = app / "Contents" / "Info.plist"
        plist_path.parent.mkdir(parents=True)
        with open(plist_path, "wb") as f:
            plistlib.dump(info_plist, f)
        old_mode = os.environ.get("ATOM_LAUNCH_MODE")
        old_bundle = os.environ.get("ATOM_APP_BUNDLE")
        try:
            os.environ["ATOM_LAUNCH_MODE"] = "bundle"
            os.environ["ATOM_APP_BUNDLE"] = str(app)
            with patch.object(stt_macos, "_Foundation", _FakeFoundation):
                assert stt_macos._bundle_has_usage_description(
                    "NSSpeechRecognitionUsageDescription"
                )
                assert stt_macos._bundle_has_usage_description(
                    "NSMicrophoneUsageDescription"
                )
        finally:
            if old_mode is None:
                os.environ.pop("ATOM_LAUNCH_MODE", None)
            else:
                os.environ["ATOM_LAUNCH_MODE"] = old_mode
            if old_bundle is None:
                os.environ.pop("ATOM_APP_BUNDLE", None)
            else:
                os.environ["ATOM_APP_BUNDLE"] = old_bundle


if __name__ == "__main__":
    test_native_stt_launch_supported_requires_bundle_usage_description()
    test_native_stt_preload_fails_cleanly_when_bundle_is_invalid()
    test_bundle_has_usage_reads_atom_app_plist_when_main_bundle_is_python()
    print("Native STT bundle guard tests passed.")
