"""AppleScript execution engine for deep macOS control."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger("atom.macos.applescript")

_KEY_CODE_MAP: dict[str, int] = {
    "enter": 36,
    "return": 36,
    "escape": 53,
    "esc": 53,
    "tab": 48,
    "space": 49,
    "delete": 51,
    "backspace": 51,
    "up": 126,
    "down": 125,
    "left": 123,
    "right": 124,
    "home": 115,
    "end": 119,
    "pageup": 116,
    "pagedown": 121,
    "f1": 122,
    "f2": 120,
    "f3": 99,
    "f4": 118,
    "f5": 96,
    "f6": 97,
    "f7": 98,
    "f8": 100,
    "f9": 101,
    "f10": 109,
    "f11": 103,
    "f12": 111,
}

_MODIFIER_MAP = {
    "command": "command down",
    "cmd": "command down",
    "option": "option down",
    "alt": "option down",
    "control": "control down",
    "ctrl": "control down",
    "shift": "shift down",
}

AppleScriptRunner = Callable[[list[str], float], tuple[int, str, str]]


@dataclass
class AppleScriptResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0


def _default_runner(command: list[str], timeout: float) -> tuple[int, str, str]:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return completed.returncode, completed.stdout or "", completed.stderr or ""


class AppleScriptEngine:
    """Thin AppleScript wrapper with small typed helpers."""

    __slots__ = ("_runner", "_timeout")

    def __init__(
        self,
        runner: AppleScriptRunner | None = None,
        default_timeout: float = 10.0,
    ) -> None:
        self._runner = runner or _default_runner
        self._timeout = float(default_timeout)

    @staticmethod
    def _escape(text: str) -> str:
        return (
            str(text or "")
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\r", " ")
            .replace("\n", " ")
        )

    @staticmethod
    def _modifier_clause(modifiers: list[str] | None) -> str:
        if not modifiers:
            return ""
        mapped: list[str] = []
        for modifier in modifiers:
            key = str(modifier or "").strip().lower()
            value = _MODIFIER_MAP.get(key)
            if value and value not in mapped:
                mapped.append(value)
        if not mapped:
            return ""
        return " using {" + ", ".join(mapped) + "}"

    def _execute(
        self,
        script: str,
        *,
        timeout: float | None = None,
    ) -> AppleScriptResult:
        try:
            returncode, stdout, stderr = self._runner(
                ["osascript", "-e", script],
                timeout or self._timeout,
            )
        except Exception as exc:
            logger.debug("AppleScript execution failed", exc_info=True)
            return AppleScriptResult(
                success=False,
                stdout="",
                stderr=str(exc),
                returncode=1,
            )

        stdout = stdout.strip()
        stderr = stderr.strip()
        success = returncode == 0
        if not success:
            logger.debug("AppleScript error rc=%s stderr=%s", returncode, stderr[:200])
        return AppleScriptResult(
            success=success,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
        )

    def run(self, script: str, *, timeout: float | None = None) -> str:
        result = self._execute(script, timeout=timeout)
        if not result.success:
            return ""
        return result.stdout

    def open_app(self, name: str) -> bool:
        app = self._escape(name)
        if not app:
            return False
        return self._execute(f'tell application "{app}" to activate').success

    def get_frontmost_app(self) -> dict[str, str]:
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
        raw = self.run(script, timeout=5.0)
        if not raw:
            return {"app": "", "title": "", "pid": ""}
        parts = raw.split("|", 2)
        return {
            "app": parts[0] if len(parts) > 0 else "",
            "pid": parts[1] if len(parts) > 1 else "",
            "title": parts[2] if len(parts) > 2 else "",
        }

    def get_volume_settings(self) -> dict[str, Any]:
        raw = self.run("get volume settings", timeout=5.0)
        if not raw:
            return {}
        parsed: dict[str, Any] = {}
        for pair in raw.split(", "):
            if ":" not in pair:
                continue
            key, value = pair.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value.lower() in {"true", "false"}:
                parsed[key] = value.lower() == "true"
                continue
            try:
                parsed[key] = int(value)
            except ValueError:
                parsed[key] = value
        return parsed

    def set_volume(self, level: int) -> bool:
        bounded = max(0, min(100, int(level)))
        return self._execute(f"set volume output volume {bounded}", timeout=5.0).success

    def set_muted(self, muted: bool) -> bool:
        state = "true" if muted else "false"
        return self._execute(f"set volume output muted {state}", timeout=5.0).success

    def get_safari_url(self) -> str:
        return self.run(
            'tell application "Safari" to get URL of current tab of front window',
            timeout=5.0,
        )

    def get_chrome_url(self) -> str:
        return self.run(
            'tell application "Google Chrome" to get URL of active tab of front window',
            timeout=5.0,
        )

    def send_notification(
        self,
        title: str,
        message: str,
        *,
        sound_name: str | None = "Glass",
    ) -> bool:
        safe_title = self._escape(title)
        safe_message = self._escape(message)
        script = f'display notification "{safe_message}" with title "{safe_title}"'
        if sound_name:
            script += f' sound name "{self._escape(sound_name)}"'
        return self._execute(script, timeout=5.0).success

    def type_text(self, text: str) -> bool:
        safe = self._escape(text)
        if not safe:
            return False
        script = f'tell application "System Events" to keystroke "{safe}"'
        return self._execute(script, timeout=5.0).success

    def press_key(self, key: str | int, modifiers: list[str] | None = None) -> bool:
        mod_clause = self._modifier_clause(modifiers)
        if isinstance(key, int) or str(key).strip().isdigit():
            code = int(str(key).strip())
            return self._execute(
                f'tell application "System Events" to key code {code}{mod_clause}',
                timeout=5.0,
            ).success

        key_str = str(key or "").strip()
        if not key_str:
            return False
        if len(key_str) == 1:
            safe = self._escape(key_str)
            return self._execute(
                f'tell application "System Events" to keystroke "{safe}"{mod_clause}',
                timeout=5.0,
            ).success

        mapped = _KEY_CODE_MAP.get(key_str.lower())
        if mapped is None:
            logger.warning("No AppleScript key code for '%s'", key_str)
            return False
        return self._execute(
            f'tell application "System Events" to key code {mapped}{mod_clause}',
            timeout=5.0,
        ).success
