"""
ATOM -- State event helpers layered on AsyncEventBus.

Provides:
  - typed state snapshot / diff emission
  - a runtime bridge for AtomStateStore
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from .atom_state import AtomStateStore

logger = logging.getLogger("atom.state.events")

STATE_DIFF_EVENT = "state.diff"
STATE_SNAPSHOT_EVENT = "state.snapshot"


class StateEventEmitter:
    """Thin helper around AsyncEventBus for state-oriented events."""

    __slots__ = ("_bus",)

    def __init__(self, bus: Any) -> None:
        self._bus = bus

    def emit_diff(self, diff: Mapping[str, Any], *, source: str = "") -> None:
        self._bus.emit_fast(
            STATE_DIFF_EVENT,
            diff=dict(diff),
            source=source,
        )

    def emit_snapshot(self, snapshot: Mapping[str, Any], *, source: str = "") -> None:
        self._bus.emit_fast(
            STATE_SNAPSHOT_EVENT,
            snapshot=dict(snapshot),
            source=source,
        )

    def emit_execution_update(self, **payload: Any) -> None:
        self._bus.emit_fast("execution.update", **payload)

    def emit_voice_partial(self, **payload: Any) -> None:
        self._bus.emit_fast("voice.partial", **payload)

    def emit_voice_final(self, **payload: Any) -> None:
        self._bus.emit_fast("voice.final", **payload)

    def emit_system_warning(self, **payload: Any) -> None:
        self._bus.emit_fast("system.warning", **payload)

    def emit_mode_change(self, **payload: Any) -> None:
        self._bus.emit_fast("mode.change", **payload)


class AtomRuntimeStateBridge:
    """Owns the shared AtomState store and emits diffs on change."""

    __slots__ = ("_store", "_events")

    def __init__(self, bus: Any, store: AtomStateStore | None = None) -> None:
        self._store = store or AtomStateStore()
        self._events = StateEventEmitter(bus)

    @property
    def store(self) -> AtomStateStore:
        return self._store

    @property
    def events(self) -> StateEventEmitter:
        return self._events

    def patch_section(self, section: str, patch: Mapping[str, Any], *, source: str = "") -> dict[str, Any]:
        diff, snapshot = self._store.patch_section(section, patch)
        if diff:
            self._events.emit_diff(diff, source=source or section)
        return snapshot

    def set_values(self, updates: Mapping[str, Any], *, source: str = "") -> dict[str, Any]:
        diff, snapshot = self._store.set_values(updates)
        if diff:
            self._events.emit_diff(diff, source=source or "set_values")
        return snapshot

    def replace_health_report(
        self,
        *,
        readiness: Mapping[str, Any] | None = None,
        self_check: Mapping[str, Any] | None = None,
        score: float | None = None,
        warnings: list[str] | None = None,
        status: str | None = None,
        scan_summary: str | None = None,
        readiness_summary: str | None = None,
        source: str = "health",
    ) -> dict[str, Any]:
        diff, snapshot = self._store.replace_health_report(
            readiness=readiness,
            self_check=self_check,
            score=score,
            warnings=warnings,
            status=status,
            scan_summary=scan_summary,
            readiness_summary=readiness_summary,
        )
        if diff:
            self._events.emit_diff(diff, source=source)
        return snapshot

    def emit_snapshot(self, *, source: str = "snapshot") -> dict[str, Any]:
        snapshot = self._store.snapshot()
        self._events.emit_snapshot(snapshot, source=source)
        return snapshot

    def snapshot(self) -> dict[str, Any]:
        return self._store.snapshot()

    def log_if_unchanged(self, source: str) -> None:
        logger.debug("State bridge source=%s produced no changes", source)
