"""Fixed-size ring buffer of recent control events for V7 replay (idempotent-only)."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class RingEvent:
    name: str
    payload: dict[str, Any]
    t: float


class EventRingBuffer:
    def __init__(self, max_events: int = 32) -> None:
        self._max = max(8, max_events)
        self._q: deque[RingEvent] = deque(maxlen=self._max)

    def push(self, name: str, payload: dict[str, Any] | None = None) -> None:
        self._q.append(RingEvent(name=name, payload=dict(payload or {}), t=time.time()))

    def recent(self) -> list[RingEvent]:
        return list(self._q)

    def clear(self) -> None:
        self._q.clear()
