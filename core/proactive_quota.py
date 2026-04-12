"""
Shared quota + audit for `jarvis_insight` emissions.

Coordinates `JarvisCore` and `ProactiveIntelligenceEngine` so proactive
nudges stay bounded; critical priorities can bypass the hourly cap.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom.proactive_quota")


class ProactiveInsightQuota:
    """Rolling-window cap on proactive insight bus emissions + optional audit log."""

    __slots__ = (
        "_max_per_hour",
        "_critical_priority_max",
        "_audit_enabled",
        "_audit_path",
        "_window_s",
        "_events",
        "_lock",
    )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = (config or {}).get("proactive_coordination") or {}
        self._max_per_hour: int = int(cfg.get("max_insights_per_hour", 24))
        self._critical_priority_max: int = int(cfg.get("critical_priority_max", 3))
        self._audit_enabled: bool = bool(cfg.get("audit_log", True))
        self._audit_path = Path(str(cfg.get("audit_log_path", "logs/proactive_insights.log")))
        self._window_s: float = float(cfg.get("window_seconds", 3600.0))
        self._events: deque[tuple[float, str, str]] = deque()
        self._lock = threading.Lock()

    def allow_emit(
        self,
        source: str,
        category: str,
        priority: int | None = None,
    ) -> bool:
        """Return True if emit is allowed. Critical priorities bypass the cap (audited)."""
        if priority is not None and priority <= self._critical_priority_max:
            self._audit(True, source, category, "critical_priority_bypass", priority)
            return True

        now = time.time()
        with self._lock:
            cutoff = now - self._window_s
            while self._events and self._events[0][0] < cutoff:
                self._events.popleft()
            if len(self._events) >= self._max_per_hour:
                self._audit(False, source, category, "hourly_quota", priority)
                return False
            self._events.append((now, source, category))

        self._audit(True, source, category, "allowed", priority)
        return True

    def _audit(
        self,
        allowed: bool,
        source: str,
        category: str,
        reason: str,
        priority: int | None,
    ) -> None:
        if not self._audit_enabled:
            return
        try:
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pr = f" p={priority}" if priority is not None else ""
            flag = "ALLOW" if allowed else "DENY"
            line = f"[{ts}] {flag} {source} cat={category}{pr} | {reason}\n"
            with open(self._audit_path, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError:
            logger.debug("proactive audit write failed", exc_info=True)
