"""Stable device fingerprint for paranoid lock mode."""

from __future__ import annotations

import hashlib
import logging
import platform
import secrets
import uuid

logger = logging.getLogger("atom.device")


def get_device_id() -> str:
    """Deterministic id for this machine (soft binding)."""
    parts = [
        platform.node() or "",
        platform.system() or "",
        platform.machine() or "",
        str(uuid.getnode()),
    ]
    raw = "|".join(parts).encode("utf-8", errors="replace")
    return hashlib.sha256(raw).hexdigest()[:64]


def validate_device(config: dict | None) -> tuple[bool, str]:
    """Match against owner.registered_device_fingerprint if set."""
    cfg = config or {}
    expected = (cfg.get("owner") or {}).get("registered_device_fingerprint")
    if not expected or not str(expected).strip():
        return True, ""
    current = get_device_id()
    ok = secrets_compare(str(expected).strip(), current)
    if ok:
        return True, ""
    logger.warning("paranoid:device_mismatch expected=%s… got=%s…", expected[:12], current[:12])
    return False, "Device fingerprint does not match registered device (paranoid mode)."


def secrets_compare(a: str, b: str) -> bool:
    if len(a) != len(b):
        return False
    try:
        return secrets.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
    except Exception:
        return a == b
