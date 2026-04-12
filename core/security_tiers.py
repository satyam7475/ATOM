"""
Permission tiers for SecurityPolicy — maps intents to coarse risk bands.

Used with ``settings.json`` → ``security.mode`` to cap which tiers may execute
without changing per-action feature flags. Complements ``control.lock_mode``.
"""

from __future__ import annotations

# Tier 1 — read-only / meta: must match ``_SAFE_ALWAYS_INTENTS`` in ``security_policy.py``.
_TIER_1: frozenset[str] = frozenset({
    "time", "date", "cpu", "ram", "battery", "disk",
    "system_info", "ip", "wifi", "uptime", "top_processes",
    "greeting", "thanks", "status", "self_check", "self_diagnostic",
    "resource_report", "resource_trend", "app_history",
    "show_reminders", "system_analyze", "confirm", "deny",
    "exit", "go_silent", "calculate", "list_apps",
    "set_volume", "mute", "unmute", "stop_music",
    "read_clipboard", "timer",
})

# Tier 2 — session / light UX (still low risk)
_TIER_2: frozenset[str] = frozenset({
    "set_brain_profile", "set_assistant_mode",
})

# Tier 4 — power, irreversible system, or high-impact
_TIER_4: frozenset[str] = frozenset({
    "shutdown_pc", "restart_pc", "logoff", "sleep_pc",
    "kill_process", "empty_recycle_bin",
    "flush_dns",
})

# Everything else gated by allow_action (open_app, file ops, desktop, URLs, …) → tier 3


def action_tier(action: str) -> int:
    """Return 1–4; higher means more privileged / sensitive."""
    a = (action or "").strip()
    if a in _TIER_1:
        return 1
    if a in _TIER_2:
        return 2
    if a in _TIER_4:
        return 4
    return 3


def max_tier_for_security_mode(mode: str | None) -> int:
    """``strict`` caps at tier 3 (blocks tier-4 power actions). Others allow tier 4."""
    m = (mode or "strict").strip().lower()
    if m == "strict":
        return 3
    if m in ("standard", "balanced", "permissive", "development"):
        return 4
    # unknown → conservative
    return 3


def tier_allowed(action: str, max_tier: int) -> tuple[bool, str]:
    """If denied, returns (False, human-readable reason)."""
    t = action_tier(action)
    if t <= max_tier:
        return True, ""
    return (
        False,
        f"Permission tier: '{action}' requires tier {t}, but security.mode allows "
        f"up to tier {max_tier}.",
    )
