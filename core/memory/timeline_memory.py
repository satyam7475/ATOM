"""
Short rolling timeline of user queries, actions, files, and outcomes.

Thread-safe for Router + brain + executor callbacks.
"""

from __future__ import annotations

import re
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

logger = __import__("logging").getLogger("atom.timeline")


@dataclass
class TimelineEvent:
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class TimelineMemory:
    """Append-only recent history with time-window reads."""

    def __init__(self, max_events: int = 500, summarize_on_prune: bool = False) -> None:
        self._max = max(50, int(max_events))
        self._summarize_on_prune = bool(summarize_on_prune)
        self._events: deque[TimelineEvent] = deque(maxlen=self._max)
        self._lock = threading.RLock()

    def event_count(self) -> int:
        with self._lock:
            return len(self._events)

    def recent_preview(self, n: int = 8) -> list[dict[str, Any]]:
        """Shallow preview for observability (no pattern scans)."""
        out: list[dict[str, Any]] = []
        with self._lock:
            for ev in list(self._events)[-n:]:
                out.append({"type": ev.type, "ts": ev.timestamp})
        return out

    def append_event(
        self,
        type: str,
        data: dict[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> None:
        ts = time.time() if timestamp is None else float(timestamp)
        ev = TimelineEvent(type=type, data=dict(data or {}), timestamp=ts)
        with self._lock:
            self._events.append(ev)
        try:
            logger.info("v7_timeline type=%s keys=%s", type, list(ev.data.keys()))
        except Exception:
            pass

    def get_recent_events(self, window_sec: float) -> list[TimelineEvent]:
        cutoff = time.time() - max(0.0, float(window_sec))
        with self._lock:
            return [e for e in self._events if e.timestamp >= cutoff]

    def get_last_active_task(self) -> dict[str, Any] | None:
        """Most recent task-scoped event (best-effort)."""
        with self._lock:
            for ev in reversed(self._events):
                if ev.type != "task":
                    continue
                d = ev.data
                if isinstance(d, dict) and d:
                    return dict(d)
        return None

    def context_snippets_for_prediction(self, limit: int = 8) -> list[str]:
        """Short strings for predictor (no large payloads)."""
        recent = self.get_recent_events(3600.0)[-limit:]
        out: list[str] = []
        for ev in recent:
            if ev.type == "user_query":
                q = (ev.data.get("text") or "")[:120]
                if q:
                    out.append(q)
            elif ev.type == "action":
                name = ev.data.get("tool") or ev.data.get("name") or ""
                if name:
                    out.append(f"action:{name}")
            elif ev.type == "file":
                path = ev.data.get("path") or ev.data.get("file") or ""
                if path:
                    out.append(f"file:{path[:80]}")
        return out

    def summary_for_prompt(self, window_sec: float = 600.0, max_lines: int = 6) -> str:
        """Compact bullet list for planner / optional prompt injection."""
        evs = self.get_recent_events(window_sec)[-max_lines * 2 :]
        if not evs:
            return ""
        lines: list[str] = []
        for ev in evs[-max_lines:]:
            if ev.type == "user_query":
                t = (ev.data.get("text") or "")[:100]
                if t:
                    lines.append(f"- said: {t}")
            elif ev.type == "action":
                n = ev.data.get("tool") or ev.data.get("name") or "action"
                ok = ev.data.get("success")
                lines.append(f"- {n} ({'ok' if ok else 'done'})")
            elif ev.type == "error":
                lines.append(f"- error: {(ev.data.get('message') or '')[:80]}")
        return "\n".join(lines)

    def detect_patterns(
        self,
        window_sec: float = 86400.0,
        min_count: int = 3,
    ) -> list[dict[str, Any]]:
        """Repeated normalized user_query texts within the window."""
        norm: list[str] = []
        for ev in self.get_recent_events(window_sec):
            if ev.type != "user_query":
                continue
            raw = (ev.data.get("text") or "").strip().lower()
            raw = re.sub(r"\s+", " ", raw)
            raw = raw[:120]
            if len(raw) > 6:
                norm.append(raw)
        if not norm:
            return []
        counts = Counter(norm)
        out: list[dict[str, Any]] = []
        for pat, n in counts.most_common(16):
            if n >= min_count:
                out.append({"pattern": pat, "count": n})
        try:
            logger.info("v7_timeline_patterns found=%d", len(out))
        except Exception:
            pass
        return out

    def get_repeated_tasks(
        self,
        window_sec: float = 86400.0,
        min_count: int = 2,
    ) -> list[str]:
        """User utterances that look like ongoing work items, repeated often enough."""
        task_kw = re.compile(
            r"\b(fix|implement|todo|task|bug|feature|refactor|continue|finish)\b",
            re.I,
        )
        candidates: list[str] = []
        for ev in self.get_recent_events(window_sec):
            if ev.type != "user_query":
                continue
            t = (ev.data.get("text") or "").strip()
            if len(t) > 8 and task_kw.search(t):
                candidates.append(t[:200])
        if not candidates:
            return []
        counts = Counter(candidates)
        return [p for p, n in counts.most_common(8) if n >= min_count]

    def suggest_next_from_pattern(self) -> str | None:
        """If a strong repeat exists, return a short suggested follow-up phrase."""
        pats = self.detect_patterns(window_sec=48 * 3600.0, min_count=3)
        if not pats:
            return None
        top = pats[0]
        return f"Continue with: {top['pattern'][:80]}"

    def user_recently_active(self, idle_sec: float = 120.0) -> bool:
        """True if meaningful timeline activity within idle_sec."""
        evs = self.get_recent_events(idle_sec)
        return len(evs) >= 2
