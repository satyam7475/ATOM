"""
ATOM v9 -- Phase 4 Context Engine Tests.

Tests the ContextEngine's bundle generation, config-driven enable/disable,
app name extraction, and failure safety -- all without mocking Win32 APIs.

Run: python -m tests.test_context_engine
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context.context_engine import ContextEngine, _extract_app_name


def test_get_bundle_returns_dict() -> None:
    """Bundle has all expected keys and correct types."""
    engine = ContextEngine()
    bundle = engine.get_bundle()

    expected_keys = {"active_app", "window_title", "clipboard", "cwd", "timestamp"}
    assert set(bundle.keys()) == expected_keys, f"Missing keys: {expected_keys - set(bundle.keys())}"

    for key in expected_keys:
        assert isinstance(bundle[key], str), f"Key '{key}' should be str, got {type(bundle[key])}"

    assert bundle["cwd"] == os.getcwd()
    assert len(bundle["timestamp"]) == 8  # HH:MM:SS

    print("  PASS: get_bundle() returns dict with all keys")


def test_clipboard_disabled() -> None:
    """Clipboard returns empty when config disables it."""
    engine = ContextEngine({"context": {"enable_clipboard": False}})

    assert engine.get_clipboard() == ""

    bundle = engine.get_bundle()
    assert bundle["clipboard"] == ""

    print("  PASS: Clipboard disabled returns empty")


def test_window_disabled() -> None:
    """Active window returns empty when config disables it."""
    engine = ContextEngine({"context": {"enable_active_window": False}})

    assert engine.get_active_window() == ""

    bundle = engine.get_bundle()
    assert bundle["active_app"] == ""
    assert bundle["window_title"] == ""

    print("  PASS: Active window disabled returns empty")


def test_extract_app_name() -> None:
    """App name extraction heuristic works for common title patterns."""
    assert _extract_app_name("main.py - Visual Studio Code") == "Visual Studio Code"
    assert _extract_app_name("Google Chrome") == "Google Chrome"
    assert _extract_app_name("Document1 - Microsoft Word") == "Microsoft Word"
    assert _extract_app_name("src/app.ts - ATOM - Visual Studio Code") == "Visual Studio Code"
    assert _extract_app_name("") == ""
    assert _extract_app_name("Untitled - Notepad") == "Notepad"

    print("  PASS: App name extraction heuristic")


def test_bundle_never_crashes() -> None:
    """get_bundle() always returns a dict even with unusual configs."""
    for cfg in [None, {}, {"context": {}}, {"context": {"enable_clipboard": False, "enable_active_window": False}}]:
        engine = ContextEngine(cfg)
        bundle = engine.get_bundle()
        assert isinstance(bundle, dict)
        assert "cwd" in bundle
        assert "timestamp" in bundle

    print("  PASS: Bundle never crashes (various configs)")


def test_default_config() -> None:
    """Default config enables both clipboard and window."""
    engine = ContextEngine()
    assert engine._enable_clipboard is True
    assert engine._enable_window is True
    assert engine._clipboard_max == 500

    print("  PASS: Default config enables all features")


def test_custom_clipboard_max() -> None:
    """Clipboard max chars is configurable."""
    engine = ContextEngine({"context": {"clipboard_max_chars": 100}})
    assert engine._clipboard_max == 100

    print("  PASS: Configurable clipboard max chars")


def test_live_window_title() -> None:
    """On a real Windows machine, get_active_window should return something."""
    engine = ContextEngine()
    title = engine.get_active_window()
    # On CI or headless, this might be empty -- that's fine
    assert isinstance(title, str)
    if title:
        print(f"  PASS: Live window title: '{title[:60]}'")
    else:
        print("  PASS: Live window title (empty -- headless or non-Windows)")


def run_all() -> None:
    print("\n=== ATOM v9 Phase 4 -- Context Engine Tests ===\n")

    test_get_bundle_returns_dict()
    test_clipboard_disabled()
    test_window_disabled()
    test_extract_app_name()
    test_bundle_never_crashes()
    test_default_config()
    test_custom_clipboard_max()
    test_live_window_title()

    print("\n=== ALL TESTS PASSED ===\n")


if __name__ == "__main__":
    run_all()
