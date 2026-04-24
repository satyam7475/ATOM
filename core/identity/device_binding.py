"""Stable device fingerprints for paranoid lock mode + trusted iPhone.

Two separate concerns live here, intentionally colocated:

1. **The Mac itself** -- :py:func:`get_device_id` + :py:func:`validate_device`
   produce / check a SHA-256 fingerprint so paranoid mode refuses to
   run ATOM on an unfamiliar Mac.
2. **The trusted iPhone (Phase 1)** -- :py:func:`is_trusted_iphone` /
   :py:func:`registered_iphone_hash` wrap the single-slot registry
   that the Shortcuts bridge writes into. Both helpers default to
   ``data/trusted_iphone.json`` so callers don't need to know the
   path.

The iPhone-side state lives in :py:mod:`core.cross_device.trusted_device`;
these wrappers exist so code that already imports ``device_binding`` for
the Mac check has one obvious place to ask "is that iPhone the one?"
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import secrets
import uuid
from pathlib import Path

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


# ── Trusted iPhone (Phase 1) ───────────────────────────────────────

def _resolve_trusted_iphone_path(
    config: dict | None = None,
    state_path: str | os.PathLike[str] | None = None,
) -> Path:
    """Choose the on-disk location of the trusted-iPhone state file.

    Precedence: explicit ``state_path`` arg -> config
    ``cross_device.trusted_device_path`` -> ``data/trusted_iphone.json``
    under the ATOM root.
    """
    if state_path:
        return Path(state_path).expanduser()
    cfg = config or {}
    cross = cfg.get("cross_device") or {}
    raw = cross.get("trusted_device_path")
    if raw:
        return Path(str(raw)).expanduser()
    # Walk up to ATOM root (folder containing config/settings.json)
    here = Path(__file__).resolve()
    for cand in (here, *here.parents):
        if (cand / "config" / "settings.json").exists():
            return cand / "data" / "trusted_iphone.json"
    return Path.cwd() / "data" / "trusted_iphone.json"


def registered_iphone_hash(
    config: dict | None = None,
    *,
    state_path: str | os.PathLike[str] | None = None,
) -> str | None:
    """Return the SHA-256 hex of the currently trusted iPhone, or None.

    Reads the same file the bridge wrote. Does **not** import aiohttp
    or any other heavy module so paranoid-mode boot can call this
    cheaply.
    """
    from core.cross_device.trusted_device import TrustedIPhoneRegistry
    p = _resolve_trusted_iphone_path(config, state_path)
    return TrustedIPhoneRegistry(p).registered_hash()


def is_trusted_iphone(
    device_id: str,
    config: dict | None = None,
    *,
    state_path: str | os.PathLike[str] | None = None,
) -> bool:
    """True iff *device_id* hashes to the registered iPhone's hash."""
    if not device_id:
        return False
    from core.cross_device.trusted_device import hash_device_id
    incoming = hash_device_id(device_id)
    current = registered_iphone_hash(config, state_path=state_path)
    if not current:
        return False
    return secrets_compare(incoming, current)
