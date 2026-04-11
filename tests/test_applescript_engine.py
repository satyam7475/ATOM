"""
Focused tests for the macOS AppleScript engine.

Run: python3 -m tests.test_applescript_engine
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _FakeRunner:
    def __init__(self, responses: dict[str, tuple[int, str, str]] | None = None) -> None:
        self.responses = responses or {}
        self.calls: list[tuple[list[str], float]] = []

    def __call__(self, command: list[str], timeout: float) -> tuple[int, str, str]:
        self.calls.append((command, timeout))
        script = command[-1]
        return self.responses.get(script, (0, "", ""))


def test_frontmost_app_parsing() -> None:
    from core.macos.applescript_engine import AppleScriptEngine

    script = (
        'tell application "System Events"\n'
        '  set frontApp to first application process whose frontmost is true\n'
        '  set appName to name of frontApp\n'
        '  set appPID to unix id of frontApp\n'
        '  try\n'
        '    set winTitle to name of front window of frontApp\n'
        '  on error\n'
        '    set winTitle to ""\n'
        '  end try\n'
        'end tell\n'
        'return appName & "|" & appPID & "|" & winTitle'
    )
    runner = _FakeRunner({script: (0, "Cursor|1234|ATOM - main.py", "")})
    engine = AppleScriptEngine(runner=runner)
    front = engine.get_frontmost_app()
    assert front == {"app": "Cursor", "pid": "1234", "title": "ATOM - main.py"}
    print("  PASS: frontmost app parsing")


def test_volume_settings_parse_types() -> None:
    from core.macos.applescript_engine import AppleScriptEngine

    runner = _FakeRunner(
        {"get volume settings": (0, "output volume:45, input volume:77, output muted:false", "")},
    )
    engine = AppleScriptEngine(runner=runner)
    settings = engine.get_volume_settings()
    assert settings["output volume"] == 45
    assert settings["input volume"] == 77
    assert settings["output muted"] is False
    print("  PASS: volume settings parsing")


def test_press_key_builds_modifier_clause() -> None:
    from core.macos.applescript_engine import AppleScriptEngine

    runner = _FakeRunner()
    engine = AppleScriptEngine(runner=runner)
    ok = engine.press_key("a", ["command", "shift"])
    assert ok is True
    assert runner.calls
    assert (
        runner.calls[-1][0][-1]
        == 'tell application "System Events" to keystroke "a" using {command down, shift down}'
    )
    print("  PASS: key press builds modifier clause")


def test_notification_escapes_quotes() -> None:
    from core.macos.applescript_engine import AppleScriptEngine

    runner = _FakeRunner()
    engine = AppleScriptEngine(runner=runner)
    ok = engine.send_notification('Boss "Alert"', 'Line 1\nLine "2"')
    assert ok is True
    script = runner.calls[-1][0][-1]
    assert 'display notification "Line 1 Line \\"2\\""' in script
    assert 'with title "Boss \\"Alert\\""' in script
    assert 'sound name "Glass"' in script
    print("  PASS: notifications escape quotes and newlines")


def test_run_returns_empty_on_failure() -> None:
    from core.macos.applescript_engine import AppleScriptEngine

    runner = _FakeRunner({"bad script": (1, "", "execution error")})
    engine = AppleScriptEngine(runner=runner)
    assert engine.run("bad script") == ""
    print("  PASS: failed AppleScript returns empty output")


def run_all() -> None:
    test_frontmost_app_parsing()
    test_volume_settings_parse_types()
    test_press_key_builds_modifier_clause()
    test_notification_escapes_quotes()
    test_run_returns_empty_on_failure()
    print("\n=== APPLESCRIPT ENGINE TESTS PASSED ===\n")


if __name__ == "__main__":
    run_all()
