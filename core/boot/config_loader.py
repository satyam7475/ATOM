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


def set_config_overrides(overrides: dict[str, Any]) -> None:
    """Set global configuration overrides (used by run_v4.py, etc.)."""
    global _CONFIG_OVERRIDES
    _CONFIG_OVERRIDES.update(overrides)


def load_config() -> dict[str, Any]:
    """Parse config/settings.json and apply any runtime overrides."""
    cfg_path = Path("config/settings.json")
    base: dict[str, Any] = {}
    
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                base = json.load(f)
        except Exception as e:
            import logging
            logging.getLogger("atom.boot").error(
                "Failed to parse config/settings.json: %s", e
            )
            
    if _CONFIG_OVERRIDES:
        for key, val in _CONFIG_OVERRIDES.items():
            if isinstance(val, dict) and isinstance(base.get(key), dict):
                base[key].update(val)
            else:
                base[key] = val
                
    return base
