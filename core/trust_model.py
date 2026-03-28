"""
ATOM V7 — Trust tiers for actions (SYSTEM / USER / UNKNOWN).

UNKNOWN is restricted from high-impact desktop/OS actions until voice or
other owner verification is implemented.

See also ``core/owner_gate.py`` for owner session + dashboard token policy
and ``SecurityPolicy.allow_action`` for the central enforcement gate.
"""

from __future__ import annotations

from enum import Enum


class TrustLevel(str, Enum):
    SYSTEM = "system"
    USER = "user"
    UNKNOWN = "unknown"


_UNKNOWN_BLOCKED: frozenset[str] = frozenset({
    "shutdown_pc", "restart_pc", "logoff", "sleep_pc",
    "kill_process", "empty_recycle_bin", "format_disk",
    "delete_path", "move_path", "run_script",
})


def trust_allows_action(trust: TrustLevel, action: str) -> bool:
    if trust is not TrustLevel.UNKNOWN:
        return True
    return action not in _UNKNOWN_BLOCKED


__all__ = ["TrustLevel", "trust_allows_action", "_UNKNOWN_BLOCKED"]
