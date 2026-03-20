"""
ATOM OS -- File and folder action handlers.

Handles: create_folder, move_path, copy_path
Path safety is delegated to SecurityPolicy.path_allowed (single source of truth).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from core.security_policy import SecurityPolicy


def resolve_path(raw: str) -> Path:
    cleaned = raw.strip().strip("\"'")
    expanded = os.path.expandvars(os.path.expanduser(cleaned))
    return Path(expanded).resolve()


def path_allowed(path: Path) -> bool:
    return SecurityPolicy.path_allowed(path)


def create_folder(folder_name: str, base_path: str) -> str:
    if not folder_name:
        raise ValueError("Folder name missing")
    root = resolve_path(base_path) if base_path else Path.cwd()
    if not path_allowed(root):
        raise PermissionError(f"Path blocked: {root}")
    target = root / folder_name
    if not path_allowed(target):
        raise PermissionError(f"Path blocked: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return str(target)


def move_path(src: str, dst: str) -> str:
    src_path = resolve_path(src)
    dst_path = resolve_path(dst)
    if not src_path.exists():
        raise FileNotFoundError(f"Source not found: {src_path}")
    if not path_allowed(src_path) or not path_allowed(dst_path):
        raise PermissionError("Move blocked outside allowed paths")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    moved_to = shutil.move(str(src_path), str(dst_path))
    return str(Path(moved_to))


def copy_path(src: str, dst: str) -> str:
    src_path = resolve_path(src)
    dst_path = resolve_path(dst)
    if not src_path.exists():
        raise FileNotFoundError(f"Source not found: {src_path}")
    if not path_allowed(src_path) or not path_allowed(dst_path):
        raise PermissionError("Copy blocked outside allowed paths")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if src_path.is_dir():
        if dst_path.exists():
            raise FileExistsError(f"Destination exists: {dst_path}")
        shutil.copytree(src_path, dst_path)
    else:
        shutil.copy2(src_path, dst_path)
    return str(dst_path)
