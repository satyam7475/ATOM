"""
ATOM -- Unified runtime world state.

Layered over the existing lifecycle state machine and async event bus.
This store is the single authoritative read model for dashboard, reports,
mode reasoning, and runtime observability.
"""

from __future__ import annotations

import copy
import threading
import time
from collections.abc import Mapping
from typing import Any


ATOM_STATE_DEFAULTS: dict[str, Any] = {
    "system": {
        "cpu": 0.0,
        "memory_pct": 0.0,
        "battery_pct": 100.0,
        "thermal_pressure": "unknown",
        "disk_free_gb": 0.0,
        "network": "unknown",
        "charging": False,
        "chip": "",
        "ram_used_gb": 0.0,
        "ram_total_gb": 0.0,
        "ram_gb": 0.0,
        "disk_total_gb": 0.0,
        "power_source": "",
        "top_processes": [],
        "hardware": {
            "chip": "",
            "gpu_name": "",
            "gpu_cores": 0,
            "memory_total_mb": 0.0,
            "memory_used_mb": 0.0,
            "memory_available_mb": 0.0,
            "on_battery": False,
            "power_watts": 0.0,
            "cpu_temp_c": 0.0,
            "is_throttled": False,
        },
        "updated_at": 0.0,
    },
    "context": {
        "active_app": "",
        "window_title": "",
        "activity_type": "idle",
        "confidence": 0.0,
        "time_of_day": "",
        "idle_minutes": 0.0,
        "is_weekday": True,
        "weekday": 0,
        "frontmost_pid": 0,
        "media": {
            "playing": False,
            "type": "",
            "source": "",
            "title": "",
            "artist": "",
            "album": "",
            "position": 0.0,
            "duration": 0.0,
            "summary": "",
        },
        "updated_at": 0.0,
    },
    "execution": {
        "active_task": "",
        "queue_depth": 0,
        "scheduler_queue_depth": 0,
        "last_action": "",
        "last_intent": "",
        "last_query": "",
        "latency_ms": 0.0,
        "status": "idle",
        "label": "idle",
        "llm_queue_pending": False,
        "mlx_generating": False,
        "cache_entries": 0,
        "cache_max": 0,
        "updated_at": 0.0,
    },
    "voice": {
        "stt_engine": "",
        "tts_engine": "",
        "status": "idle",
        "mic": "",
        "confidence": 0.0,
        "error": None,
        "language": "",
        "voice_name": "",
        "fallback_chain": [],
        "permissions": {
            "speech": "unknown",
            "microphone": "unknown",
        },
        "listening": False,
        "speaking": False,
        "last_partial": "",
        "last_final": "",
        "last_spoken": "",
        "launch_mode": "",
        "app_bundle": "",
        "perceived_latency_ms": None,
        "updated_at": 0.0,
    },
    "mode": {
        "requested": "",
        "effective": "",
        "reason": "",
        "profile": "",
        "assistant_mode": "",
        "product_tier": "",
        "cloud_enabled": True,
        "updated_at": 0.0,
    },
    "health": {
        "score": 0.0,
        "warnings": [],
        "last_check": 0.0,
        "status": "unknown",
        "readiness": {},
        "readiness_summary": "",
        "self_check": {},
        "scan_summary": "",
    },
    "reasoning": {
        "why_this_mode": "",
        "last_decision": "",
        "last_report": "",
        "severity": "info",
        "updated_at": 0.0,
    },
    "lifecycle": {
        "state": "sleep",
        "label": "SLEEP",
        "status": "",
        "always_listen": False,
        "time_in_state_s": 0.0,
        "updated_at": 0.0,
    },
    "meta": {
        "version": 0,
        "updated_at": 0.0,
    },
}


def _clone_default() -> dict[str, Any]:
    return copy.deepcopy(ATOM_STATE_DEFAULTS)


def _now_ts() -> float:
    return time.time()


def _merge_dict(target: dict[str, Any], patch: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    changed: dict[str, Any] = {}
    any_changed = False
    for key, value in patch.items():
        if isinstance(value, Mapping):
            existing = target.get(key)
            if not isinstance(existing, dict):
                existing = {}
                target[key] = existing
            nested_changed, nested_any = _merge_dict(existing, value)
            if nested_any:
                changed[key] = nested_changed
                any_changed = True
        else:
            old = target.get(key)
            if old != value:
                target[key] = value
                changed[key] = value
                any_changed = True
    return changed, any_changed


def _set_path_value(target: dict[str, Any], path: str, value: Any) -> bool:
    parts = [part for part in path.split(".") if part]
    if not parts:
        return False
    cursor = target
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    leaf = parts[-1]
    if cursor.get(leaf) == value:
        return False
    cursor[leaf] = value
    return True


class AtomStateStore:
    """Thread-safe world-state store with diff generation."""

    __slots__ = ("_lock", "_state")

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state: dict[str, Any] = _clone_default()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._state)

    def get_section(self, section: str) -> dict[str, Any]:
        with self._lock:
            value = self._state.get(section, {})
            return copy.deepcopy(value) if isinstance(value, dict) else {}

    def patch_section(
        self,
        section: str,
        patch: Mapping[str, Any],
        *,
        updated_at: float | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        with self._lock:
            section_state = self._state.get(section)
            if not isinstance(section_state, dict):
                section_state = {}
                self._state[section] = section_state

            changed, any_changed = _merge_dict(section_state, patch)
            if not any_changed:
                return {}, copy.deepcopy(self._state)

            ts = float(updated_at or _now_ts())
            section_state["updated_at"] = ts
            changed["updated_at"] = ts
            self._state["meta"]["version"] = int(self._state["meta"].get("version", 0)) + 1
            self._state["meta"]["updated_at"] = ts
            return {section: changed}, copy.deepcopy(self._state)

    def set_values(
        self,
        updates: Mapping[str, Any],
        *,
        updated_at: float | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        with self._lock:
            changed: dict[str, Any] = {}
            any_changed = False
            for path, value in updates.items():
                if _set_path_value(self._state, path, value):
                    cursor = changed
                    parts = [part for part in path.split(".") if part]
                    for part in parts[:-1]:
                        next_cursor = cursor.get(part)
                        if not isinstance(next_cursor, dict):
                            next_cursor = {}
                            cursor[part] = next_cursor
                        cursor = next_cursor
                    cursor[parts[-1]] = value
                    any_changed = True

            if not any_changed:
                return {}, copy.deepcopy(self._state)

            ts = float(updated_at or _now_ts())
            root_sections = {path.split(".", 1)[0] for path in updates}
            for section in root_sections:
                section_state = self._state.get(section)
                if isinstance(section_state, dict):
                    section_state["updated_at"] = ts
                    diff_section = changed.setdefault(section, {})
                    if isinstance(diff_section, dict):
                        diff_section.setdefault("updated_at", ts)
            self._state["meta"]["version"] = int(self._state["meta"].get("version", 0)) + 1
            self._state["meta"]["updated_at"] = ts
            return changed, copy.deepcopy(self._state)

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
        updated_at: float | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        patch: dict[str, Any] = {}
        if readiness is not None:
            patch["readiness"] = dict(readiness)
        if self_check is not None:
            patch["self_check"] = dict(self_check)
        if score is not None:
            patch["score"] = float(score)
        if warnings is not None:
            patch["warnings"] = list(warnings)
        if status is not None:
            patch["status"] = str(status)
        if scan_summary is not None:
            patch["scan_summary"] = str(scan_summary)
        if readiness_summary is not None:
            patch["readiness_summary"] = str(readiness_summary)
        patch["last_check"] = float(updated_at or _now_ts())
        return self.patch_section("health", patch, updated_at=updated_at)
