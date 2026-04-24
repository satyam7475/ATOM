"""Tier-3+ identity-freshness gate.

Goal
----
A destructive voice command (tier 3 or 4) should only execute if the
owner has recently pressed Face ID on their iPhone via the Shortcuts
bridge. This stops a nearby voice ("hey ATOM, empty the trash") from
nuking the Mac while Boss is across the room / in a Zoom / asleep.

Scope
-----
* Applies **only** when ``cross_device.enabled == true`` AND an
  iPhone has been paired (``data/trusted_iphone.json`` present).
  This avoids locking out a fresh install where the bridge has never
  run -- Boss can still execute tier-3 actions before any iPhone is
  registered.
* Tier 1 (read-only) and tier 2 (light UX) always pass.
* Tier 3 and tier 4 require ``identity.is_owner_verified()`` True
  within the configured freshness window.

Returns
-------
``(ok: bool, reason: str)``. On ``ok=False`` the caller (Router) is
expected to speak the reason -- the wording is user-friendly and
includes the path to recovery ("re-run 'Verify with ATOM' on your
iPhone").
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from core.security_tiers import action_tier

logger = logging.getLogger("atom.router.identity_gate")


class _IdentityLike(Protocol):
    def is_owner_verified(self, window_s: float | None = ...) -> bool: ...
    def faceid_freshness_info(self) -> dict[str, Any]: ...


_UNVERIFIED_REASON = (
    "Face ID is stale, Boss. Tap 'Verify with ATOM' on your iPhone "
    "first and then try that again."
)
_NO_IDENTITY_REASON = (
    "Identity engine isn't wired, Boss. I can't confirm it's really you."
)


def gate_requires_fresh_identity(
    *,
    config: dict[str, Any] | None,
    action: str,
) -> bool:
    """Return True iff this action needs a fresh Face ID to run.

    Cheap to call -- does not touch the identity engine or disk.
    """
    cfg = config or {}
    cross = cfg.get("cross_device") or {}
    if not cross.get("enabled"):
        return False

    # Phase 1: we only gate tier 3 and tier 4 actions. Lower tiers
    # stay behind the existing SecurityPolicy rate-limit + per-feature
    # toggles.
    if action_tier(action) < 3:
        return False

    # Don't lock out a first-time user whose iPhone has never paired.
    # As soon as the Shortcut runs once and writes the trust file,
    # the gate activates on the next tier-3 attempt.
    try:
        from core.identity.device_binding import registered_iphone_hash
    except Exception:
        return False
    try:
        current_hash = registered_iphone_hash(cfg)
    except Exception:
        logger.debug("registered_iphone_hash lookup failed", exc_info=True)
        return False
    return current_hash is not None


def check_identity_freshness(
    *,
    config: dict[str, Any] | None,
    identity_engine: _IdentityLike | None,
    action: str,
) -> tuple[bool, str]:
    """Return (allowed, reason). ``allowed=True, reason=""`` for the
    common case (gate off or owner verified). ``allowed=False, reason``
    is a human-readable string safe to speak.
    """
    if not gate_requires_fresh_identity(config=config, action=action):
        return True, ""

    if identity_engine is None:
        # Fail closed when the gate should apply but we lack the
        # engine -- this only happens in degraded boot.
        return False, _NO_IDENTITY_REASON

    try:
        verified = bool(identity_engine.is_owner_verified())
    except Exception:
        logger.exception("identity_engine.is_owner_verified failed")
        return False, _NO_IDENTITY_REASON

    if verified:
        return True, ""

    return False, _UNVERIFIED_REASON


__all__ = [
    "gate_requires_fresh_identity",
    "check_identity_freshness",
]
