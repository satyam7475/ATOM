"""
ATOM OS -- Application management action handlers.

Handles: open_app, close_app, list_apps

All launch/kill operations go through SecurityPolicy before execution.
"""

from __future__ import annotations


from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path

from core.security_policy import SecurityPolicy, get_global_policy

logger = logging.getLogger("atom.router.app")

_apps_cache_text: str | None = None
_apps_cache_ts: float = 0.0

# Sprint Ω.1: lazy proxy so we don't construct a second SecurityPolicy
# at module-load time. The first call routes through ``get_global_policy``
# which returns the canonical instance set up in ``main.py``.


class _PolicyProxy:
    """Forward attribute access to the canonical SecurityPolicy.

    Created at import time, but the underlying ``SecurityPolicy`` is
    fetched lazily on first ``__getattr__`` call so we don't trigger
    the duplicate "SecurityPolicy init" log line during boot.
    """

    __slots__ = ()

    def __getattr__(self, item: str):  # type: ignore[override]
        return getattr(get_global_policy(), item)


_policy: SecurityPolicy = _PolicyProxy()  # type: ignore[assignment]


_IS_MACOS = sys.platform == "darwin"


def open_app(exe: str, args: list[str] | None = None, *, name: str = "") -> None:
    if _IS_MACOS and exe == "open":
        app_name = ""
        a = args or []
        if len(a) >= 2 and a[0] == "-a":
            app_name = a[1]
        _policy.audit_log("open_app", f"macOS open -a '{app_name or name}'")
        subprocess.Popen(["open"] + a,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info("Opened app (macOS): %s", app_name or name)
        return

    if not _policy.is_safe_executable(exe):
        _policy.audit_log("open_app", f"BLOCKED executable '{exe}'", success=False)
        raise PermissionError(f"Executable '{exe}' is not in the safe allowlist.")
    _policy.audit_log("open_app", f"exe={exe}")
    subprocess.Popen([exe] + (args or []),
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logger.info("Opened app: %s", exe)


def close_app(process_name: str, *, name: str = "") -> None:
    if not _policy.is_safe_close_target(process_name):
        _policy.audit_log("close_app", f"BLOCKED process '{process_name}'", success=False)
        raise PermissionError(f"Process '{process_name}' is not in the safe close list.")
    _policy.audit_log("close_app", f"process={process_name}")

    if _IS_MACOS:
        subprocess.Popen(
            ["osascript", "-e", f'quit app "{process_name}"'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    else:
        subprocess.Popen(["taskkill", "/IM", process_name, "/F"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logger.info("Closed app: %s", process_name)


def list_installed_apps() -> str:
    if sys.platform == "darwin":
        roots = [
            Path("/Applications"),
            Path("/System/Applications"),
            Path.home() / "Applications",
        ]
        names: list[str] = []
        for root in roots:
            if not root.is_dir():
                continue
            try:
                for app in sorted(root.rglob("*.app")):
                    names.append(app.stem)
            except Exception:
                logger.debug("App scan failed for %s", root, exc_info=True)
        if not names:
            return "I couldn't find any macOS apps right now."
        unique: list[str] = []
        seen: set[str] = set()
        for name in names:
            low = name.lower()
            if low in seen:
                continue
            seen.add(low)
            unique.append(name)
        preview = ", ".join(unique[:25])
        remaining = max(0, len(unique) - 25)
        if remaining:
            return (
                f"I found {len(unique)} apps on this Mac. Sample: {preview}. "
                f"And {remaining} more."
            )
        return f"I found {len(unique)} apps on this Mac: {preview}."

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
