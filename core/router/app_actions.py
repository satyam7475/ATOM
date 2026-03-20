"""
ATOM OS -- Application management action handlers.

Handles: open_app, close_app, list_apps

All launch/kill operations go through SecurityPolicy before execution.
"""

from __future__ import annotations

import logging
import subprocess
import time

from core.security_policy import SecurityPolicy

logger = logging.getLogger("atom.router.app")

_apps_cache_text: str | None = None
_apps_cache_ts: float = 0.0

_policy = SecurityPolicy()


def open_app(exe: str, args: list[str] | None = None) -> None:
    if not _policy.is_safe_executable(exe):
        _policy.audit_log("open_app", f"BLOCKED executable '{exe}'", success=False)
        raise PermissionError(f"Executable '{exe}' is not in the safe allowlist.")
    _policy.audit_log("open_app", f"exe={exe}")
    subprocess.Popen([exe] + (args or []),
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logger.info("Opened app: %s", exe)


def close_app(process_name: str) -> None:
    if not _policy.is_safe_close_target(process_name):
        _policy.audit_log("close_app", f"BLOCKED process '{process_name}'", success=False)
        raise PermissionError(f"Process '{process_name}' is not in the safe close list.")
    _policy.audit_log("close_app", f"process={process_name}")
    subprocess.Popen(["taskkill", "/IM", process_name, "/F"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logger.info("Closed app: %s", process_name)


def list_installed_apps() -> str:
    cmd = [
        "powershell", "-NoProfile", "-Command",
        "Get-StartApps | Sort-Object Name "
        "| Select-Object -ExpandProperty Name",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=4)
    if proc.returncode != 0:
        return "I could not list apps right now."
    names = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if not names:
        return "No apps found in Start Apps list."
    unique: list[str] = []
    seen: set[str] = set()
    for n in names:
        low = n.lower()
        if low in seen:
            continue
        seen.add(low)
        unique.append(n)
    preview = ", ".join(unique[:25])
    remaining = max(0, len(unique) - 25)
    if remaining:
        return (f"I found {len(unique)} apps. Top apps: {preview}. "
                f"And {remaining} more.")
    return f"I found {len(unique)} apps: {preview}."


def list_installed_apps_cached() -> str:
    global _apps_cache_text, _apps_cache_ts
    now = time.monotonic()
    if _apps_cache_text and (now - _apps_cache_ts) < 300:
        return _apps_cache_text
    text = list_installed_apps()
    _apps_cache_text = text
    _apps_cache_ts = now
    return text
