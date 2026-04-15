"""
ATOM -- Session Memory (short-term command history).

Lightweight in-memory ring buffer tracking the recent command session.
Unlike TimelineMemory (generic events) or ConversationMemory (turn threading),
SessionMemory stores a compact per-command record that the CommandLoop and
Router can query instantly for context:

  - Last N commands with their result status, active app, and latency
  - Last intent name for repeat/continuation detection
  - Active app at time of each command
  - Conversation buffer for quick follow-up resolution

This is the "working memory" -- what a human assistant would remember
from the last 5 minutes of interaction without taking notes.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("atom.session_memory")

_DEFAULT_CAPACITY = 20


@dataclass
class CommandRecord:
    """Single command entry in session memory."""
    text: str
    app: str = ""
    intent: str = ""
    trace_id: str = ""
    success: bool = True
    response_snippet: str = ""
    elapsed_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


class SessionMemory:
    """Short-term memory for the current session.

    Thread-safe ring buffer of recent commands with fast accessors
    for the most common queries the pipeline needs.
    """

    def __init__(self, capacity: int = _DEFAULT_CAPACITY) -> None:
        self._capacity = max(5, int(capacity))
        self._commands: deque[CommandRecord] = deque(maxlen=self._capacity)
        self._lock = threading.Lock()
        self._last_intent: str = ""
        self._last_app: str = ""
        self._command_count: int = 0

    def record_command(
        self,
        text: str,
        *,
        app: str = "",
        intent: str = "",
        trace_id: str = "",
        success: bool = True,
        response_snippet: str = "",
        elapsed_ms: float = 0.0,
    ) -> None:
        """Record a completed command."""
        record = CommandRecord(
            text=text[:200],
            app=app,
            intent=intent,
            trace_id=trace_id,
            success=success,
            response_snippet=response_snippet[:200],
            elapsed_ms=elapsed_ms,
        )
        with self._lock:
            self._commands.append(record)
            self._command_count += 1
            if intent:
                self._last_intent = intent
            if app:
                self._last_app = app

    def update_last(
        self,
        *,
        intent: str = "",
        success: bool | None = None,
        response_snippet: str = "",
        elapsed_ms: float = 0.0,
    ) -> None:
        """Update the most recent command record with result info."""
        with self._lock:
            if not self._commands:
                return
            rec = self._commands[-1]
            if intent:
                rec.intent = intent
                self._last_intent = intent
            if success is not None:
                rec.success = success
            if response_snippet:
                rec.response_snippet = response_snippet[:200]
            if elapsed_ms > 0:
                rec.elapsed_ms = elapsed_ms

    @property
    def last_command(self) -> str:
        with self._lock:
            return self._commands[-1].text if self._commands else ""

    @property
    def last_intent(self) -> str:
        with self._lock:
            return self._last_intent

    @property
    def last_app(self) -> str:
        with self._lock:
            return self._last_app

    @property
    def command_count(self) -> int:
        return self._command_count

    def get_recent(self, n: int = 5) -> list[dict[str, Any]]:
        """Return the last N command records as dicts."""
        with self._lock:
            records = list(self._commands)[-n:]
        return [
            {
                "text": r.text,
                "app": r.app,
                "intent": r.intent,
                "success": r.success,
                "elapsed_ms": r.elapsed_ms,
                "timestamp": r.timestamp,
            }
            for r in records
        ]

    def get_recent_texts(self, n: int = 5) -> list[str]:
        """Just the command texts for quick context injection."""
        with self._lock:
            return [r.text for r in list(self._commands)[-n:]]

    def context_for_prompt(self, n: int = 5) -> str:
        """Compact context block for LLM prompt injection."""
        records = self.get_recent(n)
        if not records:
            return ""
        lines = ["[RECENT COMMANDS]"]
        for r in records:
            status = "ok" if r["success"] else "failed"
            app_hint = f" ({r['app']})" if r["app"] else ""
            lines.append(f"  - {r['text'][:80]}{app_hint} [{status}]")
        return "\n".join(lines)

    def is_follow_up(self, text: str) -> bool:
        """Heuristic: is this a likely follow-up to the last command?"""
        lower = text.lower().strip()
        follow_up_signals = {
            "again", "one more", "also", "and", "what about",
            "do that", "same thing", "repeat", "continue",
            "next", "go on", "keep going", "more",
        }
        for signal in follow_up_signals:
            if lower.startswith(signal) or lower == signal:
                return True
        if lower in {"yes", "yeah", "yep", "sure", "ok", "okay"}:
            return True
        return False

    def detect_repetition(self) -> str | None:
        """If the user repeats the same command 3+ times, return it."""
        with self._lock:
            if len(self._commands) < 3:
                return None
            recent = [r.text.lower().strip() for r in list(self._commands)[-3:]]
        if len(set(recent)) == 1:
            return recent[0]
        return None

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            "total_commands": self._command_count,
            "buffer_size": len(self._commands),
            "capacity": self._capacity,
            "last_intent": self._last_intent,
            "last_app": self._last_app,
        }


__all__ = ["SessionMemory", "CommandRecord"]
