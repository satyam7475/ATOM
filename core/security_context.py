"""Per-task security context (session id, request source) via contextvars."""

from __future__ import annotations

from contextvars import ContextVar

current_session_id: ContextVar[str | None] = ContextVar("current_session_id", default=None)
current_request_source: ContextVar[str | None] = ContextVar("current_request_source", default=None)


def reset_session_context() -> None:
    current_session_id.set(None)
    current_request_source.set(None)
