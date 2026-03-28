"""
ATOM — Owner identity and access gate (Satyam-only deployment model).

Physical machine + local runtime are treated as trusted by default
(``owner.trust_local_runtime``). Optional dashboard token gates WebSocket
control when ``ui.dashboard_access_token`` or env ``ATOM_DASHBOARD_TOKEN`` is set.

``control.lock_mode`` in ``secure`` / ``paranoid`` (legacy: ``owner_only``)
restricts non-safe intents until the owner session is considered unlocked
(local trust and/or dashboard auth).
"""

from __future__ import annotations

import logging
import os
import secrets
import threading
from typing import Any

logger = logging.getLogger("atom.owner_gate")

_lock = threading.RLock()
_config: dict[str, Any] = {}
_session_authenticated: bool = False


def configure(config: dict | None = None) -> None:
    """Load owner + UI security flags from config (call once at startup)."""
    global _config
    with _lock:
        _config = dict(config or {})
    try:
        from core.identity.session_manager import configure as _sess_cfg
        _sess_cfg(config)
    except Exception:
        pass
    oc = _config.get("owner", {})
    ui = _config.get("ui", {})
    tok = (oc.get("dashboard_access_token") or ui.get("dashboard_access_token")
           or os.environ.get("ATOM_DASHBOARD_TOKEN", "").strip())
    if tok:
        logger.info(
            "Owner gate: dashboard WebSocket token is configured (%d chars).",
            len(tok),
        )
    else:
        logger.debug("Owner gate: no dashboard token (localhost UI open).")

    if oc.get("exclusive_use"):
        logger.info(
            "Owner gate: exclusive_use enabled — ATOM is bound to owner policy.",
        )


def owner_display_name() -> str:
    return str((_config.get("owner") or {}).get("name", "Satyam"))


def trust_local_runtime() -> bool:
    """Voice + Router on this machine are trusted (default True)."""
    return bool((_config.get("owner") or {}).get("trust_local_runtime", True))


def exclusive_use() -> bool:
    return bool((_config.get("owner") or {}).get("exclusive_use", False))


def dashboard_token_expected() -> str | None:
    ui = _config.get("ui", {})
    if ui.get("password_auth_enabled") is False:
        return None
    oc = _config.get("owner", {})
    tok = (oc.get("dashboard_access_token") or ui.get("dashboard_access_token")
           or os.environ.get("ATOM_DASHBOARD_TOKEN", "").strip())
    return tok or None


def validate_dashboard_token(token: str | None) -> bool:
    """Constant-time compare against configured or env token."""
    expected = dashboard_token_expected()
    if not expected:
        return True
    if not token or not isinstance(token, str):
        return False
    try:
        return secrets.compare_digest(token.strip(), expected)
    except Exception:
        return False


def mark_session_authenticated(valid: bool = True) -> None:
    """Call when dashboard WS auth succeeds."""
    global _session_authenticated
    with _lock:
        _session_authenticated = bool(valid)


def is_session_authenticated() -> bool:
    with _lock:
        return _session_authenticated


def is_owner_unlocked() -> bool:
    """Enough trust to allow non-safe intents under secure/paranoid lock."""
    if trust_local_runtime():
        return True
    return is_session_authenticated()


def owner_policy_denies(action: str) -> tuple[bool, str]:
    """Return (True, reason) if SecurityPolicy should block this action."""
    from core.lock_modes import requires_owner_session
    from core.security_policy import _SAFE_ALWAYS_INTENTS  # noqa: SLF001

    if action in _SAFE_ALWAYS_INTENTS:
        return False, ""
    if is_owner_unlocked():
        return False, ""
    ctrl = _config.get("control", {})
    if requires_owner_session(ctrl.get("lock_mode", "off")) or exclusive_use():
        return True, (
            "ATOM owner policy: non-safe actions require an unlocked owner session. "
            "Options: set owner.trust_local_runtime true (local machine), "
            "authenticate dashboard with ui.dashboard_access_token, "
            "or relax control.lock_mode / owner.exclusive_use."
        )
    return False, ""


__all__ = [
    "configure",
    "owner_display_name",
    "trust_local_runtime",
    "exclusive_use",
    "dashboard_token_expected",
    "validate_dashboard_token",
    "mark_session_authenticated",
    "is_session_authenticated",
    "is_owner_unlocked",
    "owner_policy_denies",
]
