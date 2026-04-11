"""Configuration and hint helpers for :class:`core.macos.fs_watcher.FSWatcher`."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_PATHS: tuple[str, ...] = ("~/Desktop", "~/Downloads", "~/Documents")
DEFAULT_HINT_EXTENSIONS: tuple[str, ...] = (
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".zip",
    ".dmg",
    ".txt",
)


def fs_watcher_settings(config: dict[str, Any] | None) -> dict[str, Any]:
    """Return resolved FSWatcher options (no pyobjc dependency)."""
    mac = (config or {}).get("macos") or {}
    fw = mac.get("fs_watcher") or {}
    paths = fw.get("paths")
    if not isinstance(paths, list) or not paths:
        paths = list(DEFAULT_PATHS)
    exts = fw.get("hint_extensions")
    if isinstance(exts, list) and exts:
        hint_extensions = tuple(str(x).lower() for x in exts if x)
    else:
        hint_extensions = DEFAULT_HINT_EXTENSIONS
    return {
        "enabled": bool(fw.get("enabled", True)),
        "paths": [str(p) for p in paths],
        "hints_enabled": bool(fw.get("hints_enabled", True)),
        "hint_extensions": hint_extensions,
        "hint_cooldown_s": float(fw.get("hint_cooldown_seconds", 120)),
        "emit_voice": bool(fw.get("emit_voice", False)),
    }


def notable_file_hint(
    *,
    path: str,
    event: str,
    is_dir: bool,
    config: dict[str, Any] | None,
) -> str | None:
    """Return a short proactive line for interesting new files, or None.

    Triggers on ``created`` / ``modified`` under ``Downloads`` (case-insensitive)
    when the suffix matches ``hint_extensions``.
    """
    if is_dir or not path:
        return None
    if event not in ("created", "modified", "changed"):
        return None

    settings = fs_watcher_settings(config)
    if not settings["hints_enabled"]:
        return None

    suffix = Path(path).suffix.lower()
    if suffix not in settings["hint_extensions"]:
        return None

    parts = {p.lower() for p in Path(path).parts}
    if "downloads" not in parts:
        return None

    base = os.path.basename(path)
    kind = "file"
    if suffix == ".pdf":
        kind = "PDF"
    elif suffix in (".doc", ".docx"):
        kind = "document"
    elif suffix in (".zip", ".dmg"):
        kind = "archive"
    return f'New {kind} in Downloads: "{base}". Want me to open or summarize it?'


__all__ = ["DEFAULT_HINT_EXTENSIONS", "DEFAULT_PATHS", "fs_watcher_settings", "notable_file_hint"]
