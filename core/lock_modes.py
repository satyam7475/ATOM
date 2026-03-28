"""Normalized control.lock_mode values for ATOM security policy."""

from __future__ import annotations

# Legacy config values → canonical names
_LEGACY: dict[str, str] = {
    "off": "open",
    "safe_only": "restricted",
    "owner_only": "secure",
}

_CANONICAL = frozenset({"open", "restricted", "secure", "paranoid"})


def normalize_lock_mode(raw: str | None) -> str:
    """Map legacy lock_mode strings to open/restricted/secure/paranoid."""
    m = (raw or "off").strip().lower()
    if m in _LEGACY:
        return _LEGACY[m]
    if m in _CANONICAL:
        return m
    return "open"


def requires_owner_session(lock_mode: str) -> bool:
    """True when non-safe actions need an unlocked owner session."""
    return normalize_lock_mode(lock_mode) in ("secure", "paranoid")


def runtime_switch_locked(lock_mode: str) -> bool:
    """True when brain/assistant mode switches are disallowed."""
    return normalize_lock_mode(lock_mode) != "open"
