"""
ATOM -- Configuration Loader

Handles loading settings.json and applying runtime overrides for the OS.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Global overrides for runtime modifications (e.g., via CLI)
_CONFIG_OVERRIDES: dict[str, Any] = {}
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "config" / "settings.json"


def set_config_overrides(overrides: dict[str, Any]) -> None:
    """Set global configuration overrides (used by run_v4.py, etc.)."""
    global _CONFIG_OVERRIDES
    _CONFIG_OVERRIDES.update(overrides)


def load_config() -> dict[str, Any]:
    """Parse config/settings.json and apply any runtime overrides."""
    base: dict[str, Any] = {}

    if _CONFIG_PATH.exists():
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                base = json.load(f)
        except Exception as e:
            import logging
            logging.getLogger("atom.boot").error(
                "Failed to parse %s: %s", _CONFIG_PATH, e
            )

    if _CONFIG_OVERRIDES:
        for key, val in _CONFIG_OVERRIDES.items():
            if isinstance(val, dict) and isinstance(base.get(key), dict):
                base[key].update(val)
            else:
                base[key] = val

    return base
