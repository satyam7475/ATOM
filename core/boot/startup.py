"""Early boot configuration and owner binding."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.boot.config_loader import load_config

logger = logging.getLogger("atom.main")


@dataclass(slots=True)
class StartupBootstrap:
    config: dict[str, Any]
    secret_snapshot: dict[str, str]
    deployment_dashboard_badge: Callable[[dict[str, Any]], tuple[str, bool]]


def prepare_startup() -> StartupBootstrap:
    """Load/validate config and wire owner-level process state.

    This preserves the original main.py order: config validation first,
    owner/session binding next, secret scrub after dashboard token setup,
    deployment profile logging, then adaptive-personality owner context.
    """
    config = load_config()

    from core.config_schema import validate_and_log
    if not validate_and_log(config):
        logger.error("Invalid configuration — fix config/settings.json and restart.")
        sys.exit(1)

    from core.owner_gate import configure as _configure_owner_gate, owner_display_name
    _configure_owner_gate(config)
    try:
        from core.identity.session_manager import configure as _configure_sessions
        _configure_sessions(config)
    except Exception:
        logger.debug("Session manager configure skipped or failed", exc_info=True)
    logger.info(
        "ATOM owner binding: %s — access control via core/owner_gate.py",
        owner_display_name(),
    )

    try:
        from core.security_secret_scrub import scrub_sensitive_env

        secret_snapshot = scrub_sensitive_env(preserve=("ATOM_DASHBOARD_TOKEN",))
    except Exception:
        secret_snapshot = {}
        logger.debug("Secret scrub skipped or failed", exc_info=True)

    from core.deployment_profile import (
        deployment_dashboard_badge,
        log_deployment_bootstrap,
    )
    log_deployment_bootstrap(config)

    from core.adaptive_personality import set_owner as _set_owner
    owner_cfg = config.get("owner", {})
    _set_owner(
        name=owner_cfg.get("name", "Satyam"),
        title=owner_cfg.get("title", "Boss"),
    )

    return StartupBootstrap(
        config=config,
        secret_snapshot=secret_snapshot,
        deployment_dashboard_badge=deployment_dashboard_badge,
    )
