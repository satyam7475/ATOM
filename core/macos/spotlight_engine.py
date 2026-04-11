"""macOS Spotlight search via ``mdfind`` (metadata query)."""

from __future__ import annotations

import logging
import subprocess
import sys
from typing import Callable

logger = logging.getLogger("atom.macos.spotlight")

MdFindRunner = Callable[[list[str], float], tuple[int, str, str]]

_MAX_QUERY_LEN = 2000
_MAX_LIMIT = 500


def _default_runner(command: list[str], timeout: float) -> tuple[int, str, str]:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return completed.returncode, completed.stdout or "", completed.stderr or ""


def spotlight_search(query: str, limit: int = 10, *, timeout: float = 10.0) -> list[dict[str, str]]:
    """Search using macOS Spotlight (plan-compatible helper).

    Returns ``[{"path": "/abs/path"}, ...]`` for each hit. Non-macOS or
    empty query yields an empty list.
    """
    return SpotlightEngine().search(query, limit=limit, timeout=timeout)


class SpotlightEngine:
    """Thin ``mdfind`` wrapper with injectable runner (tests / sandbox)."""

    __slots__ = ("_runner",)

    def __init__(self, runner: MdFindRunner | None = None) -> None:
        self._runner = runner or _default_runner

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        timeout: float = 10.0,
    ) -> list[dict[str, str]]:
        """Run Spotlight and return up to ``limit`` filesystem paths."""
        raw = (query or "").strip()
        if not raw or sys.platform != "darwin":
            return []

        q = raw[:_MAX_QUERY_LEN]
        lim = max(1, min(int(limit), _MAX_LIMIT))

        cmd = ["mdfind", "-limit", str(lim), q]
        try:
            code, out, err = self._runner(cmd, float(timeout))
        except Exception:
            logger.debug("mdfind failed", exc_info=True)
            return []

        if code != 0 and err:
            logger.debug("mdfind stderr (rc=%s): %s", code, err[:500])

        hits: list[dict[str, str]] = []
        for line in (out or "").splitlines():
            path = line.strip()
            if path:
                hits.append({"path": path})
        return hits

    def find_first_path(self, query: str, *, timeout: float = 2.0) -> str:
        """Return the first path from a metadata query, or ``\"\"``."""
        rows = self.search(query, limit=1, timeout=timeout)
        if not rows:
            return ""
        return str(rows[0].get("path") or "")


__all__ = ["MdFindRunner", "SpotlightEngine", "spotlight_search"]
