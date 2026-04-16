"""
ATOM -- Siri Shortcuts Bridge for macOS.

Runs Siri Shortcuts from ATOM via the macOS ``shortcuts`` CLI, giving ATOM
access to HomeKit, Calendar, Reminders, Focus Modes, Translate, and any
user-created automation without needing direct framework access.

Entirely offline for local shortcuts; network-dependent ones behave as
the shortcut itself dictates.

Requires: macOS 12+ (Monterey) where the ``shortcuts`` CLI is available.

Owner: Satyam
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("atom.macos.siri_shortcuts")

_SHORTCUTS_BIN: str | None = shutil.which("shortcuts")


@dataclass
class ShortcutResult:
    success: bool
    output: str = ""
    error: str = ""
    shortcut_name: str = ""


@dataclass
class ShortcutInfo:
    name: str
    folder: str = ""


class SiriShortcutsBridge:
    """Execute and enumerate macOS Siri Shortcuts from ATOM.

    All calls are async-safe — heavy subprocess work is offloaded to
    the default executor so the event loop stays free.
    """

    _TIMEOUT_LIST: float = 10.0
    _TIMEOUT_RUN: float = 30.0

    def __init__(self) -> None:
        self._available: bool = (
            sys.platform == "darwin" and _SHORTCUTS_BIN is not None
        )
        self._cache: list[ShortcutInfo] | None = None
        if not self._available:
            logger.info(
                "Siri Shortcuts bridge unavailable (platform=%s, cli=%s)",
                sys.platform,
                _SHORTCUTS_BIN,
            )

    @property
    def is_available(self) -> bool:
        return self._available

    # ── List shortcuts ────────────────────────────────────────────

    async def list_shortcuts(self, *, refresh: bool = False) -> list[ShortcutInfo]:
        """Return all installed Siri Shortcuts (cached after first call)."""
        if not self._available:
            return []
        if self._cache is not None and not refresh:
            return list(self._cache)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self._list_sync)
        self._cache = result
        return list(result)

    def _list_sync(self) -> list[ShortcutInfo]:
        try:
            proc = subprocess.run(
                ["shortcuts", "list"],
                capture_output=True,
                text=True,
                timeout=self._TIMEOUT_LIST,
            )
            if proc.returncode != 0:
                logger.warning("shortcuts list failed: %s", proc.stderr.strip())
                return []
            names = [
                ln.strip() for ln in proc.stdout.strip().splitlines() if ln.strip()
            ]
            return [ShortcutInfo(name=n) for n in names]
        except subprocess.TimeoutExpired:
            logger.warning("shortcuts list timed out after %.0fs", self._TIMEOUT_LIST)
            return []
        except Exception:
            logger.exception("shortcuts list error")
            return []

    # ── Run a shortcut ────────────────────────────────────────────

    async def run(
        self,
        name: str,
        *,
        input_text: str = "",
        timeout: float | None = None,
    ) -> ShortcutResult:
        """Run a named Siri Shortcut, optionally piping text input."""
        if not self._available:
            return ShortcutResult(
                success=False,
                error="Siri Shortcuts unavailable on this platform",
                shortcut_name=name,
            )
        if not name:
            return ShortcutResult(
                success=False,
                error="No shortcut name provided",
                shortcut_name=name,
            )

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._run_sync,
            name,
            input_text,
            timeout or self._TIMEOUT_RUN,
        )

    def _run_sync(
        self, name: str, input_text: str, timeout: float,
    ) -> ShortcutResult:
        cmd = ["shortcuts", "run", name]
        stdin_data: str | None = None
        if input_text:
            cmd.extend(["--input-type", "text"])
            stdin_data = input_text

        try:
            proc = subprocess.run(
                cmd,
                input=stdin_data,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            success = proc.returncode == 0
            if not success:
                logger.warning(
                    "Shortcut '%s' failed (rc=%d): %s",
                    name, proc.returncode, proc.stderr.strip()[:200],
                )
            else:
                logger.info("Shortcut '%s' executed successfully", name)
            return ShortcutResult(
                success=success,
                output=proc.stdout.strip(),
                error=proc.stderr.strip(),
                shortcut_name=name,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Shortcut '%s' timed out after %.0fs", name, timeout)
            return ShortcutResult(
                success=False,
                error=f"Timed out after {timeout:.0f}s",
                shortcut_name=name,
            )
        except Exception as exc:
            logger.exception("Shortcut '%s' error", name)
            return ShortcutResult(
                success=False,
                error=str(exc),
                shortcut_name=name,
            )

    # ── Convenience wrappers ──────────────────────────────────────

    async def find_shortcut(self, query: str) -> ShortcutInfo | None:
        """Fuzzy-find a shortcut by name substring (case-insensitive)."""
        all_shortcuts = await self.list_shortcuts()
        q = query.strip().lower()
        for sc in all_shortcuts:
            if q == sc.name.lower():
                return sc
        for sc in all_shortcuts:
            if q in sc.name.lower():
                return sc
        return None

    async def run_by_query(
        self, query: str, *, input_text: str = "",
    ) -> ShortcutResult:
        """Find and run a shortcut by fuzzy name match."""
        match = await self.find_shortcut(query)
        if match is None:
            return ShortcutResult(
                success=False,
                error=f"No shortcut matching '{query}' found",
                shortcut_name=query,
            )
        return await self.run(match.name, input_text=input_text)

    def invalidate_cache(self) -> None:
        self._cache = None
