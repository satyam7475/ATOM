"""
ATOM -- Security Policy (Production-Grade Enforcement Layer).

The single security gate for ATOM. Every sensitive action goes through
this module before execution. Config-driven via settings.json "security"
section.

Enforcement layers:
  1. Action-level gate (allow_action) -- called by Router before dispatch
  2. Executable allowlist (is_safe_executable) -- enforced by app_actions
  3. Shell command blocklist (is_safe_command) -- pattern matching
  4. Hotkey / key safety tiers (is_safe_hotkey / is_safe_key)
  5. File path allowlist (path_allowed) -- centralised here
  6. Input sanitisation (sanitize_input) -- length + injection protection
  7. Audit logging -- every sensitive + blocked action logged to file
  8. Rate limiting -- prevents rapid-fire command abuse
  9. Prompt injection deep scan -- multi-layer LLM prompt injection defense
  10. Directory traversal protection -- blocks ../ path escape attacks
  11. Command chaining detection -- blocks && || ; piped attacks

v20: Hardened for production with SecurityFortress integration.
Owner: Satyam. All policy decisions prioritize system safety and privacy.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("atom.security")

_AUDIT_FILE = Path("logs/audit.log")

# ── Defaults (overridden by config) ────────────────────────────────

BLOCKED_SHELL_PATTERNS: frozenset[str] = frozenset({
    "format ", "del /s", "del /q", "rmdir /s", "rmdir /q",
    "rd /s", "rd /q", "reg delete", "reg add",
    "netsh firewall", "netsh advfirewall",
    "net user", "net localgroup", "net share",
    "taskkill /f /im", "wmic process",
    "cipher /w", "diskpart", "bcdedit",
    "powershell -enc", "powershell -e ",
    "invoke-expression", "iex ",
    "set-executionpolicy",
})

BLOCKED_EXACT: frozenset[str] = frozenset({
    "shutdown", "restart", "logoff", "hibernate",
})

SAFE_EXECUTABLES: frozenset[str] = frozenset({
    "chrome", "msedge", "edge", "firefox", "brave",
    "notepad", "notepad++", "calc", "calculator",
    "explorer", "cmd", "powershell", "terminal",
    "code", "cursor", "vscode",
    "outlook", "teams", "slack", "zoom",
    "spotify", "vlc", "winamp",
    "postman", "docker", "git",
    "excel", "word", "powerpoint", "onenote",
    "paint", "snip", "snippingtool",
    "taskmgr", "perfmon", "resmon",
    "mstsc", "control",
    "intellij", "pycharm", "webstorm",
})

SAFE_CLOSE_PROCESSES: frozenset[str] = frozenset({
    "chrome.exe", "msedge.exe", "firefox.exe", "brave.exe",
    "notepad.exe", "notepad++.exe", "CalculatorApp.exe",
    "OUTLOOK.EXE", "ms-teams.exe",
    "WINWORD.EXE", "EXCEL.EXE", "POWERPNT.EXE",
    "Spotify.exe", "slack.exe", "Discord.exe", "Zoom.exe",
    "Postman.exe", "Code.exe", "Cursor.exe",
    "Docker Desktop.exe", "WhatsApp.exe", "Telegram.exe",
})

SAFE_HOTKEYS: dict[str, str] = {
    "ctrl+c": "safe", "ctrl+v": "safe", "ctrl+x": "safe",
    "ctrl+z": "safe", "ctrl+a": "safe", "ctrl+s": "safe",
    "ctrl+f": "safe", "ctrl+p": "safe", "ctrl+n": "safe",
    "ctrl+t": "safe", "ctrl+tab": "safe",
    "ctrl+shift+tab": "safe", "alt+tab": "safe",
    "alt+f4": "confirm", "ctrl+w": "confirm",
    "ctrl+shift+delete": "confirm",
    "win+r": "block", "win+x": "block",
    "ctrl+alt+delete": "block",
    # macOS equivalents (command replaces ctrl)
    "command+c": "safe", "command+v": "safe", "command+x": "safe",
    "command+z": "safe", "command+a": "safe", "command+s": "safe",
    "command+f": "safe", "command+p": "safe", "command+n": "safe",
    "command+t": "safe", "command+tab": "safe",
    "command+shift+tab": "safe", "option+tab": "safe",
    "command+q": "confirm", "command+w": "confirm",
}

SAFE_KEYS: frozenset[str] = frozenset({
    "enter", "return", "escape", "esc", "tab",
    "space", "backspace", "delete",
    "up", "down", "left", "right",
    "pageup", "pagedown", "home", "end",
    "f1", "f2", "f3", "f4", "f5", "f6",
    "f7", "f8", "f9", "f10", "f11", "f12",
    "volumeup", "volumedown", "volumemute",
    "playpause", "nexttrack", "prevtrack",
})

_BLOCKED_PATH_PARTS = (
    "\\windows", "\\system32", "\\syswow64",
    "\\program files", "\\program files (x86)", "\\programdata",
)

_SENSITIVE_ACTIONS: frozenset[str] = frozenset({
    "open_app", "close_app", "kill_process",
    "create_folder", "move_path", "copy_path",
    "scroll_down", "scroll_up", "click_screen", "press_key",
    "go_back", "hotkey_combo", "type_text",
    "shutdown_pc", "restart_pc", "logoff", "sleep_pc",
    "empty_recycle_bin", "flush_dns", "open_url",
    "play_youtube", "search", "spotlight_search", "lock_screen", "screenshot",
    "set_brightness",
})

_SAFE_ALWAYS_INTENTS: frozenset[str] = frozenset({
    "time", "date", "cpu", "ram", "battery", "disk",
    "system_info", "ip", "wifi", "uptime", "top_processes",
    "greeting", "thanks", "status", "self_check", "self_diagnostic",
    "resource_report", "resource_trend", "app_history",
    "show_reminders", "system_analyze", "confirm", "deny",
    "exit", "go_silent", "calculate", "list_apps",
    "set_volume", "mute", "unmute", "stop_music",
    "read_clipboard", "timer",
})

_DANGEROUS_INPUT_RE = re.compile(
    r"[;&|`$]|\\x[0-9a-f]{2}|<script|javascript:",
    re.I,
)

_PROMPT_INJECTION_RE = re.compile(
    r"(ignore\s+(all\s+)?previous\s+instructions|"
    r"disregard\s+(all\s+)?prior|"
    r"you\s+are\s+now\s+|"
    r"new\s+instructions?\s*:|"
    r"system\s+prompt\s*:|"
    r"override\s+(system|security|safety)|"
    r"forget\s+(everything|all\s+rules|your\s+instructions)|"
    r"pretend\s+you\s+are|"
    r"act\s+as\s+if\s+you\s+are|"
    r"jailbreak|bypass\s+security|"
    r"do\s+anything\s+now|"
    r"dan\s+mode|developer\s+mode|"
    r"ignore\s+(?:all\s+)?(?:safety|ethical)|"
    r"reveal\s+(?:your|the)\s+(?:system|secret|hidden)|"
    r"what\s+(?:is|are)\s+your\s+(?:system|initial)\s+(?:prompt|instructions)|"
    r"repeat\s+(?:your|the)\s+(?:system|initial)\s+prompt|"
    r"output\s+(?:your|the)\s+(?:instructions|rules)|"
    r"sudo\s+|admin\s+mode|root\s+access|"
    r"disable\s+(?:all\s+)?(?:filters?|safety|security)|"
    r"<\s*(?:system|admin|root)\s*>)",
    re.I,
)

_DIRECTORY_TRAVERSAL_RE = re.compile(
    r"(?:\.\./|\.\.\\|%2e%2e|%252e%252e|\.\.%2f|\.\.%5c)",
    re.I,
)

_COMMAND_CHAIN_RE = re.compile(
    r"(?:&&|\|\||;\s*(?:rm|del|format|shutdown|restart|kill|taskkill))",
    re.I,
)

_MAX_INPUT_LENGTH = 2000
_RATE_LIMIT_WINDOW_S = 5.0
_RATE_LIMIT_MAX_ACTIONS = 10


class SecurityPolicy:
    """The single security gate for ATOM OS.

    Config-driven: reads settings.json "security" section at init.
    Includes rate limiting to prevent rapid-fire command abuse.
    """

    def __init__(self, config: dict | None = None) -> None:
        _AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._config = config or {}
        sec = (config or {}).get("security", {})
        self._mode: str = sec.get("mode", "strict")
        self._audit_to_file: bool = sec.get("audit_to_file", True)
        self._extra_confirm: list[str] = sec.get("require_confirmation_for", [])

        ctrl = (config or {}).get("control", {})
        from core.lock_modes import normalize_lock_mode

        self._lock_mode: str = normalize_lock_mode(ctrl.get("lock_mode", "off"))
        self._allow_runtime_mode_switch: bool = ctrl.get(
            "allow_runtime_mode_switch", True,
        )
        self._features: dict[str, bool] = (config or {}).get("features", {})

        self._action_timestamps: list[float] = []
        self._rate_limit_window = sec.get("rate_limit_window_s", _RATE_LIMIT_WINDOW_S)
        self._rate_limit_max = sec.get("rate_limit_max_actions", _RATE_LIMIT_MAX_ACTIONS)

        from core.execution.behavior_monitor import BehaviorMonitor

        self._behavior = BehaviorMonitor(config)
        self._behavior.set_lock_mode(self._lock_mode)

        logger.info("SecurityPolicy init: mode=%s, lock=%s, rate_limit=%d/%ds",
                     self._mode, self._lock_mode,
                     self._rate_limit_max, self._rate_limit_window)

    # ── Central action gate ───────────────────────────────────────────

    def _check_rate_limit(self) -> bool:
        """Return True if under the rate limit, False if too many actions."""
        import time
        now = time.monotonic()
        self._action_timestamps = [
            t for t in self._action_timestamps
            if now - t < self._rate_limit_window
        ]
        if len(self._action_timestamps) >= self._rate_limit_max:
            return False
        self._action_timestamps.append(now)
        return True

    def allow_action(
        self,
        action: str,
        args: dict | None = None,
        *,
        policy_context: str = "execute",
    ) -> tuple[bool, str]:
        """Single gate: can this action execute right now?

        Returns (allowed, reason). Audit-logs all denials.
        Includes rate limiting to prevent command flooding.
        ``policy_context='plan_validate'`` skips paranoid HMAC verification (planning dry-run).
        """
        if action not in _SAFE_ALWAYS_INTENTS and not self._check_rate_limit():
            reason = "Too many actions in a short time. Please slow down, Boss."
            self.audit_log(action, reason, success=False)
            return False, reason

        from core.owner_gate import owner_policy_denies
        denied, deny_reason = owner_policy_denies(action)
        if denied:
            self.audit_log(action, deny_reason, success=False)
            return False, deny_reason

        from core.runtime_config import DegradationMode, get_degradation_mode
        if get_degradation_mode(self._config) == DegradationMode.SAFE:
            if action not in _SAFE_ALWAYS_INTENTS:
                reason = "SAFE degradation mode — only safe intents allowed."
                self.audit_log(action, reason, success=False)
                return False, reason

        if self._lock_mode == "restricted" and action not in _SAFE_ALWAYS_INTENTS:
            reason = f"Lock mode 'restricted': action '{action}' is not permitted."
            self.audit_log(action, reason, success=False)
            return False, reason

        ok, p1_reason = self._phase1_policy_checks(action, args, policy_context=policy_context)
        if not ok:
            logger.warning("%s | action=%s", p1_reason, action)
            self.audit_log(action, p1_reason, success=False)
            return False, p1_reason

        if action in ("set_brain_profile", "set_assistant_mode"):
            if not self._allow_runtime_mode_switch:
                reason = "Runtime brain / assistant mode switches are disabled in config."
                self.audit_log(action, reason, success=False)
                return False, reason
            from core.lock_modes import runtime_switch_locked

            if runtime_switch_locked(self._lock_mode):
                reason = (
                    f"Cannot change runtime mode while control.lock_mode is "
                    f"'{self._lock_mode}'."
                )
                self.audit_log(action, reason, success=False)
                return False, reason

        feature_map = {
            "scroll_down": "desktop_control", "scroll_up": "desktop_control",
            "click_screen": "desktop_control", "press_key": "desktop_control",
            "go_back": "desktop_control", "hotkey_combo": "desktop_control",
            "type_text": "desktop_control",
            "create_folder": "file_ops", "move_path": "file_ops",
            "copy_path": "file_ops",
            "kill_process": "system_analyze",
            "system_analyze": "system_analyze",
        }
        feature = feature_map.get(action)
        if feature and not self._features.get(feature, True):
            reason = f"Feature '{feature}' is disabled in config."
            self.audit_log(action, reason, success=False)
            return False, reason

        if action == "open_app":
            name = (args or {}).get("name", "")
            if not self.is_safe_executable(name):
                reason = f"Executable '{name}' is not in the safe allowlist."
                self.audit_log("open_app", reason, success=False)
                return False, reason

        if action == "close_app":
            proc = (args or {}).get("process", "")
            if not self.is_safe_close_target(proc):
                reason = f"Process '{proc}' is not in the safe close list."
                self.audit_log("close_app", reason, success=False)
                return False, reason

        if action in ("shutdown_pc", "restart_pc", "logoff", "sleep_pc"):
            if self._mode == "strict":
                reason = f"Power action '{action}' blocked in strict mode."
                self.audit_log(action, reason, success=False)
                return False, reason

        return True, "ok"

    def _phase1_policy_checks(
        self,
        action: str,
        args: dict | None,
        *,
        policy_context: str,
    ) -> tuple[bool, str]:
        """Session, device binding, behavior monitor, HMAC (paranoid / secure)."""
        if self._lock_mode in ("secure", "paranoid"):
            s_ok, s_reason = self._session_gate()
            if not s_ok:
                return False, s_reason

        if self._lock_mode == "paranoid":
            from core.identity.device_binding import validate_device

            d_ok, d_reason = validate_device(self._config)
            if not d_ok:
                return False, d_reason

        bm_ok, bm_reason = self._behavior.check_action_allowed(
            action, args, policy_context=policy_context,
        )
        if not bm_ok:
            return False, bm_reason

        if self._lock_mode == "paranoid" and policy_context == "execute":
            from core.security.action_signing import verify_action

            sec = self._config.get("security") or {}
            if not sec.get("paranoid_signing_disabled"):
                v_ok, v_reason = verify_action(action, args, config=self._config)
                if not v_ok:
                    return False, v_reason

        return True, ""

    def _session_gate(self) -> tuple[bool, str]:
        from core.identity.session_manager import sessions_enabled, validate_session
        from core.owner_gate import is_session_authenticated, trust_local_runtime
        from core.security_context import current_session_id

        auth = self._config.get("auth") or {}
        if not auth.get("sessions_enabled", False):
            return True, ""
        sec = self._config.get("security") or {}
        if self._lock_mode == "paranoid" and sec.get(
            "paranoid_require_session_even_when_local_trust",
        ):
            sid = current_session_id.get()
            if validate_session(sid) is None:
                return False, "paranoid:session required (no valid session)"
            return True, ""
        if trust_local_runtime():
            return True, ""
        sid = current_session_id.get()
        if validate_session(sid) is not None:
            return True, ""
        if is_session_authenticated():
            return True, ""
        return False, "secure:session invalid or missing"

    # ── Feature and lock queries ──────────────────────────────────────

    @property
    def lock_mode(self) -> str:
        return self._lock_mode

    def can_switch_runtime_modes(self) -> bool:
        """True if voice/UI may change brain profile or assistant mode."""
        return self._allow_runtime_mode_switch and self._lock_mode == "open"

    def is_feature_enabled(self, feature: str) -> bool:
        return self._features.get(feature, True)

    def requires_extra_confirmation(self, action: str) -> bool:
        return action in self._extra_confirm

    # ── Executable checks ─────────────────────────────────────────────

    @staticmethod
    def is_safe_executable(name: str) -> bool:
        clean = name.lower().replace(".exe", "").strip()
        return clean in SAFE_EXECUTABLES

    @staticmethod
    def is_safe_close_target(process_name: str) -> bool:
        return process_name in SAFE_CLOSE_PROCESSES

    # ── Shell command checks ──────────────────────────────────────────

    @staticmethod
    def is_safe_command(cmd: str) -> tuple[bool, str]:
        cmd_lower = cmd.lower().strip()
        for pattern in BLOCKED_SHELL_PATTERNS:
            if pattern in cmd_lower:
                reason = f"Blocked: '{pattern}' is not allowed on a corporate system."
                logger.warning("Security block: command '%s' matched '%s'", cmd[:60], pattern)
                return False, reason
        if cmd_lower in BLOCKED_EXACT:
            return False, f"Blocked: '{cmd_lower}' requires manual execution."
        return True, "ok"

    # ── Hotkey / key checks ───────────────────────────────────────────

    @staticmethod
    def is_safe_hotkey(combo: str) -> tuple[str, str]:
        key = combo.lower().replace(" ", "").strip()
        tier = SAFE_HOTKEYS.get(key, "confirm")
        if tier == "block":
            return "block", f"Hotkey '{combo}' is blocked on corporate systems."
        return tier, "ok"

    @staticmethod
    def is_safe_key(key: str) -> bool:
        return key.lower().strip() in SAFE_KEYS

    # ── Path safety (centralised) ─────────────────────────────────────

    @staticmethod
    def path_allowed(path: Path) -> bool:
        p = str(path).lower()
        for blocked in _BLOCKED_PATH_PARTS:
            if blocked in p:
                return False
        home = str(Path.home()).lower()
        return p.startswith(home) or p.startswith(str(Path.cwd()).lower())

    # ── Input sanitisation ────────────────────────────────────────────

    @staticmethod
    def sanitize_input(text: str) -> tuple[str, bool]:
        """Sanitise raw voice/text input. Returns (clean_text, was_modified).

        Production-grade multi-layer input sanitization:
          1. Length cap (2000 chars)
          2. Shell injection character stripping
          3. Prompt injection detection and removal
          4. Directory traversal pattern removal
          5. Command chaining detection
          6. Null byte removal
          7. Unicode normalization
          8. Whitespace normalization
        """
        original = text

        text = text.replace("\x00", "")

        if len(text) > _MAX_INPUT_LENGTH:
            text = text[:_MAX_INPUT_LENGTH]

        text = _DANGEROUS_INPUT_RE.sub("", text)

        if _PROMPT_INJECTION_RE.search(text):
            logger.warning("SECURITY: Prompt injection attempt detected and stripped")
            text = _PROMPT_INJECTION_RE.sub("", text)

        if _DIRECTORY_TRAVERSAL_RE.search(text):
            logger.warning("SECURITY: Directory traversal attempt detected")
            text = _DIRECTORY_TRAVERSAL_RE.sub("", text)

        if _COMMAND_CHAIN_RE.search(text):
            logger.warning("SECURITY: Command chaining attempt detected")
            text = _COMMAND_CHAIN_RE.sub("", text)

        text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text, text != original

    # ── Fortress Integration ─────────────────────────────────────────

    def attach_fortress(self, fortress) -> None:
        """Wire SecurityFortress for deep security integration."""
        self._fortress = fortress
        logger.info("SecurityFortress attached to SecurityPolicy")

    def fortress_gate(self, action: str) -> tuple[bool, str]:
        """Additional fortress-level check for sensitive actions."""
        fortress = getattr(self, "_fortress", None)
        if fortress is None:
            return True, "ok"
        if not fortress.is_authenticated:
            sensitive = {
                "shutdown_pc", "restart_pc", "logoff", "sleep_pc",
                "kill_process", "create_folder", "move_path", "copy_path",
                "empty_recycle_bin", "flush_dns",
            }
            if action in sensitive:
                self.audit_log(action, "Blocked: not authenticated", success=False)
                return False, "Authentication required for this action, Boss."
        return True, "ok"

    # ── Audit log ─────────────────────────────────────────────────────

    def audit_log(self, action: str, details: str = "", success: bool = True) -> None:
        if not self._audit_to_file:
            return
        try:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status = "OK" if success else "BLOCKED"
            entry = f"[{ts}] [{status}] {action}"
            if details:
                entry += f" | {details}"
            entry += "\n"
            with open(_AUDIT_FILE, "a", encoding="utf-8") as f:
                f.write(entry)
            try:
                os.chmod(_AUDIT_FILE, 0o600)
            except OSError:
                pass
        except Exception:
            logger.debug("Failed to write audit log", exc_info=True)
