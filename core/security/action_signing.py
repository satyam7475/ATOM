"""HMAC signing for action + payload (paranoid mode)."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger("atom.action_signing")

_SIGN_KEYS = frozenset({"_atom_ts", "_atom_sig", "_atom_nonce"})


def _canonical_payload(action: str, args: dict[str, Any] | None) -> str:
    clean = {k: v for k, v in (args or {}).items() if k not in _SIGN_KEYS}
    keys = sorted(clean.keys())
    obj = {"action": action, "args": {k: clean[k] for k in keys}}
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _get_secret(config: dict | None) -> bytes:
    sec = (config or {}).get("security") or {}
    raw = (sec.get("action_signing_secret") or os.environ.get("ATOM_ACTION_SECRET") or "").strip()
    if not raw:
        return b"atom-dev-unsigned-change-me"
    return raw.encode("utf-8")


def sign_action(
    action: str,
    args: dict[str, Any] | None,
    *,
    config: dict | None = None,
    ts: float | None = None,
) -> dict[str, Any]:
    """Return args with _atom_ts, _atom_nonce, _atom_sig (hex)."""
    t = ts if ts is not None else time.time()
    nonce = hashlib.sha256(f"{t}:{action}".encode()).hexdigest()[:16]
    payload = _canonical_payload(action, args) + f"|{t:.6f}|{nonce}"
    secret = _get_secret(config)
    sig = hmac.new(secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()
    out = dict(args or {})
    out["_atom_ts"] = t
    out["_atom_nonce"] = nonce
    out["_atom_sig"] = sig
    return out


def verify_action(
    action: str,
    args: dict[str, Any] | None,
    *,
    config: dict | None = None,
    max_skew_s: float = 120.0,
) -> tuple[bool, str]:
    """Verify HMAC and timestamp skew."""
    a = args or {}
    ts = a.get("_atom_ts")
    sig = a.get("_atom_sig")
    nonce = a.get("_atom_nonce")
    if ts is None or sig is None:
        return False, "paranoid:signature missing _atom_ts or _atom_sig"
    try:
        tf = float(ts)
    except (TypeError, ValueError):
        return False, "paranoid:signature invalid timestamp"
    if abs(time.time() - tf) > max_skew_s:
        return False, "paranoid:signature timestamp skew too large"
    payload = _canonical_payload(action, a) + f"|{tf:.6f}|{nonce or ''}"
    secret = _get_secret(config)
    expected = hmac.new(secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, str(sig)):
        logger.warning("paranoid:signature mismatch action=%s", action)
        return False, "paranoid:signature HMAC verification failed"
    return True, ""


def should_attach_signature(config: dict | None, lock_mode: str) -> bool:
    if lock_mode != "paranoid":
        return False
    sec = (config or {}).get("security") or {}
    if sec.get("paranoid_signing_disabled"):
        return False
    return True


def merge_signed_args(
    security_policy: Any,
    action: str,
    args: dict | None,
) -> dict:
    """Attach signature when paranoid mode requires it."""
    cfg = getattr(security_policy, "_config", None) or {}
    lm = getattr(security_policy, "lock_mode", "open")
    if not should_attach_signature(cfg, lm):
        return dict(args or {})
    return sign_action(action, args, config=cfg)
