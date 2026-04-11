"""
ATOM -- Cold-start optimizer for faster first response.

Preloads the fast MLX role, warms embeddings, restores a small slice of the
previous conversation, seeds the hot command cache, and replays the most
recent system context so the next boot feels alive immediately.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from core.command_cache import get_command_cache
from core.persistence_manager import persistence_manager

logger = logging.getLogger("atom.boot.cold_start")

_SNAPSHOT_PATH = Path("logs/cold_start_snapshot.json")
_SNAPSHOT_KEY = "cold_start_snapshot"
_DEFAULT_TOP_COMMANDS = 24
_DEFAULT_SESSION_TURNS = 8
_MAX_RESTORED_CONTEXT_AGE_S = 6 * 3600
_INFO_INTENTS = frozenset({
    "time", "date", "cpu", "ram", "battery", "disk",
    "system_info", "ip", "wifi", "uptime", "top_processes",
    "resource_report", "resource_trend", "app_history",
    "show_reminders", "self_diagnostic", "system_analyze",
    "self_check", "behavior_report",
})


@dataclass
class ColdStartReport:
    elapsed_ms: float
    fast_model_ready: bool
    embeddings_ready: bool
    restored_turns: int
    cached_commands: int
    restored_context_available: bool


class ColdStartOptimizer:
    """Warm the startup path and persist a lightweight next-boot snapshot."""

    def __init__(
        self,
        *,
        config: dict[str, Any] | None,
        state_manager: Any,
        memory_store: Any,
        intent_engine: Any,
        bus: Any = None,
        local_brain: Any = None,
        conversation_memory: Any = None,
        system_monitor: Any = None,
        snapshot_path: str | Path | None = None,
    ) -> None:
        self._config = config or {}
        self._state = state_manager
        self._memory = memory_store
        self._intent = intent_engine
        self._bus = bus
        self._local_brain = local_brain
        self._conversation_memory = conversation_memory
        self._system_monitor = system_monitor
        self._snapshot_path = Path(snapshot_path or _SNAPSHOT_PATH)
        self._boot_time = 0.0
        self._restored_snapshot: dict[str, Any] = {}
        self._restored_context_emitted = False

        persistence_manager.register(_SNAPSHOT_KEY, self._snapshot_path)

    async def warm_up(self) -> ColdStartReport:
        """Preload the hot path pieces needed for the first real query."""
        self._boot_time = time.monotonic()
        self._restored_snapshot = self._load_snapshot()

        results = await asyncio.gather(
            self._preload_fast_model(),
            self._preload_embeddings(),
            self._restore_session(),
            self._cache_top_commands(),
            return_exceptions=True,
        )

        fast_model_ready = self._coerce_bool(results[0], "fast_model")
        embeddings_ready = self._coerce_bool(results[1], "embeddings")
        restored_turns = self._coerce_int(results[2], "session_restore")
        cached_commands = self._coerce_int(results[3], "command_cache")
        elapsed_ms = (time.monotonic() - self._boot_time) * 1000
        restored_context_available = bool(
            (self._restored_snapshot or {}).get("system_state"),
        )

        logger.info(
            "Cold start ready in %.0fms (fast=%s embeddings=%s session=%d cache=%d context=%s)",
            elapsed_ms,
            fast_model_ready,
            embeddings_ready,
            restored_turns,
            cached_commands,
            restored_context_available,
        )

        return ColdStartReport(
            elapsed_ms=elapsed_ms,
            fast_model_ready=fast_model_ready,
            embeddings_ready=embeddings_ready,
            restored_turns=restored_turns,
            cached_commands=cached_commands,
            restored_context_available=restored_context_available,
        )

    async def emit_restored_context(self) -> bool:
        """Replay the last lightweight system snapshot after handlers are wired."""
        if self._restored_context_emitted or self._bus is None:
            return False
        snapshot = self._restored_snapshot or self._load_snapshot()
        payload = self._build_context_payload(snapshot)
        if not payload:
            return False

        self._bus.emit_fast("context_snapshot", **payload)
        self._restored_context_emitted = True
        logger.info(
            "Cold start restored context: cpu=%.1f ram=%.1f active_app=%s",
            float(payload.get("cpu", 0.0)),
            float(payload.get("ram", 0.0)),
            payload.get("active_app", ""),
        )
        return True

    def persist_snapshot(self) -> bool:
        """Store a compact boot snapshot for the next launch."""
        snapshot = {
            "saved_at": time.time(),
            "atom_state": getattr(getattr(self._state, "current", None), "value", "idle"),
            "conversation_pairs": self._capture_conversation_pairs(),
            "system_state": self._capture_system_state(),
        }
        try:
            persistence_manager.save_now(_SNAPSHOT_KEY, snapshot)
            self._restored_snapshot = snapshot
            return True
        except Exception:
            logger.debug("Cold start snapshot persist failed", exc_info=True)
            return False

    def _load_snapshot(self) -> dict[str, Any]:
        try:
            loaded = persistence_manager.load(_SNAPSHOT_KEY)
        except Exception:
            logger.debug("Cold start snapshot load failed", exc_info=True)
            return {}

        if not isinstance(loaded, dict):
            return {}

        saved_at = float(loaded.get("saved_at", 0.0) or 0.0)
        if saved_at > 0:
            age_s = max(0.0, time.time() - saved_at)
            logger.info("Cold start snapshot found (age %.0fs)", age_s)
        return loaded

    async def _preload_fast_model(self) -> bool:
        if self._local_brain is None:
            return False
        warm_up = getattr(self._local_brain, "warm_up", None)
        if not callable(warm_up):
            return False

        try:
            try:
                result = await warm_up(model_role="fast")
            except TypeError:
                result = await warm_up()
            return bool(result)
        except Exception:
            logger.debug("Cold start fast-model preload failed", exc_info=True)
            return False

    async def _preload_embeddings(self) -> bool:
        warm_up = getattr(self._memory, "warm_up_embeddings", None)
        if not callable(warm_up):
            return False
        try:
            return bool(await warm_up())
        except Exception:
            logger.debug("Cold start embedding preload failed", exc_info=True)
            return False

    async def _restore_session(self) -> int:
        if self._conversation_memory is None:
            return 0
        if int(getattr(self._conversation_memory, "turn_count", 0) or 0) > 0:
            return 0

        snapshot = self._restored_snapshot or {}
        raw_pairs = snapshot.get("conversation_pairs") or []
        if not isinstance(raw_pairs, list):
            return 0

        restored = 0
        for pair in raw_pairs[-self._session_turn_limit() :]:
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            query = str(pair[0] or "").strip()
            response = str(pair[1] or "").strip()
            if not query or not response:
                continue
            try:
                self._conversation_memory.record(query, "restored_session", response)
                restored += 1
            except Exception:
                logger.debug("Cold start session turn restore failed", exc_info=True)
        return restored

    async def _cache_top_commands(self) -> int:
        get_top_commands = getattr(self._memory, "get_top_commands", None)
        if not callable(get_top_commands):
            return 0

        try:
            commands = list(get_top_commands(limit=self._top_command_limit()))
        except Exception:
            logger.debug("Cold start top-command lookup failed", exc_info=True)
            return 0

        if not commands:
            return 0

        cmd_cache = get_command_cache()
        cached = 0
        seen: set[str] = set()
        for command in commands:
            text = str(command or "").strip()
            if not text:
                continue
            norm = text.lower()
            if norm in seen:
                continue
            seen.add(norm)
            try:
                result = self._intent.classify(text)
            except Exception:
                logger.debug("Cold start classify failed for '%s'", text[:80], exc_info=True)
                continue

            intent = str(getattr(result, "intent", "") or "")
            if intent in {"", "fallback", "confirm", "deny"}:
                continue
            if intent in _INFO_INTENTS:
                continue
            cmd_cache.put(text, result)
            cached += 1
        return cached

    def _capture_conversation_pairs(self) -> list[list[str]]:
        if self._conversation_memory is None:
            return []

        getter = getattr(self._conversation_memory, "get_pairs", None)
        if not callable(getter):
            return []

        try:
            pairs = list(getter())
        except Exception:
            logger.debug("Cold start conversation capture failed", exc_info=True)
            return []

        out: list[list[str]] = []
        for query, response in pairs[-self._session_turn_limit() :]:
            q = str(query or "").strip()
            r = str(response or "").strip()
            if q and r:
                out.append([q, r])
        return out

    def _capture_system_state(self) -> dict[str, Any]:
        if self._system_monitor is None:
            return {}
        getter = getattr(self._system_monitor, "get_system_state", None)
        if not callable(getter):
            return {}
        try:
            state = getter()
            return state if isinstance(state, dict) else {}
        except Exception:
            logger.debug("Cold start system snapshot failed", exc_info=True)
            return {}

    def _build_context_payload(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        system_state = snapshot.get("system_state") or {}
        if not isinstance(system_state, dict) or not system_state:
            return {}

        ts = float(
            system_state.get("ts")
            or snapshot.get("saved_at")
            or 0.0,
        )
        if ts > 0:
            age_s = max(0.0, time.time() - ts)
            if age_s > _MAX_RESTORED_CONTEXT_AGE_S:
                logger.info(
                    "Cold start context snapshot too old (%.0fs); skipping restore",
                    age_s,
                )
                return {}
            dt = datetime.fromtimestamp(ts)
        else:
            dt = datetime.now()

        cpu = float(system_state.get("cpu_percent", system_state.get("cpu", 0.0)) or 0.0)
        ram = float(system_state.get("ram_percent", system_state.get("ram", 0.0)) or 0.0)
        active_app = str(
            system_state.get("foreground_window_title")
            or system_state.get("active_app")
            or "",
        )[:120]
        weekday = dt.weekday()

        return {
            "time_of_day": self._time_of_day(dt.hour),
            "hour": dt.hour,
            "cpu": cpu,
            "ram": ram,
            "idle_minutes": 0.0,
            "active_app": active_app,
            "is_weekday": weekday < 5,
            "weekday": weekday,
        }

    def _top_command_limit(self) -> int:
        return _DEFAULT_TOP_COMMANDS

    def _session_turn_limit(self) -> int:
        return _DEFAULT_SESSION_TURNS

    @staticmethod
    def _time_of_day(hour: int) -> str:
        if 5 <= hour < 12:
            return "morning"
        if 12 <= hour < 17:
            return "afternoon"
        if 17 <= hour < 22:
            return "evening"
        return "night"

    @staticmethod
    def _coerce_bool(value: Any, stage: str) -> bool:
        if isinstance(value, Exception):
            logger.debug("Cold start stage failed: %s (%s)", stage, value)
            return False
        return bool(value)

    @staticmethod
    def _coerce_int(value: Any, stage: str) -> int:
        if isinstance(value, Exception):
            logger.debug("Cold start stage failed: %s (%s)", stage, value)
            return 0
        try:
            return int(value)
        except Exception:
            return 0


__all__ = ["ColdStartOptimizer", "ColdStartReport"]
