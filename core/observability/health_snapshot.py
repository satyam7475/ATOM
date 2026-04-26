"""
ATOM — Per-subsystem health snapshot (Sprint C5).

Collects a compact, JSON-serializable view of every live subsystem so
the web dashboard (``GET /health``) and smoke scripts can tell at a
glance whether ATOM is happy.

Design goals:

* ``build()`` must never raise. Each subsystem probe is isolated so
  one bad component doesn't take down the whole health page.
* No network or disk I/O — we only read in-memory state that is
  already maintained by the running subsystems.
* Stable shape. Callers depend on the ``ok``/``status``/``detail``
  triple for each subsystem.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("atom.obs.health")

_STATUS_OK = "ok"
_STATUS_DEGRADED = "degraded"
_STATUS_DOWN = "down"
_STATUS_UNKNOWN = "unknown"


@dataclass
class HealthSnapshotBuilder:
    """Collects live subsystem handles and renders a health payload."""

    bus: Any = None
    state: Any = None
    stt: Any = None
    tts: Any = None
    local_brain: Any = None
    embedding_engine: Any = None
    semantic_cache: Any = None
    memory: Any = None
    silicon_governor: Any = None
    health_monitor: Any = None
    error_monitor: Any = None
    mic_manager: Any = None
    started_at: float = field(default_factory=time.time)

    # ── Main entrypoint ───────────────────────────────────────────

    def build(self) -> dict[str, Any]:
        subsystems = {
            "bus": self._probe_bus(),
            "stt": self._probe_stt(),
            "tts": self._probe_tts(),
            "brain": self._probe_brain(),
            "embeddings": self._probe_embeddings(),
            "semantic_cache": self._probe_semantic_cache(),
            "memory": self._probe_memory(),
            "silicon": self._probe_silicon(),
            "state_machine": self._probe_state(),
            "error_rate": self._probe_errors(),
        }

        # Aggregate: down > degraded > ok.
        overall = _STATUS_OK
        for sub in subsystems.values():
            if not isinstance(sub, dict):
                continue
            status = str(sub.get("status", _STATUS_UNKNOWN))
            if status == _STATUS_DOWN:
                overall = _STATUS_DOWN
                break
            if status == _STATUS_DEGRADED and overall == _STATUS_OK:
                overall = _STATUS_DEGRADED

        return {
            "ok": overall == _STATUS_OK,
            "status": overall,
            "uptime_s": max(0.0, time.time() - float(self.started_at)),
            "generated_at": time.time(),
            "subsystems": subsystems,
            # Sprint P4.6 (Apr 26 2026): unified status badge. Distilled
            # from the same per-subsystem probes into a single
            # "ATOM is OK" / "ATOM has N warnings" / "ATOM is critical"
            # line + colour so the dashboard menubar / iPhone widget /
            # smoke scripts can render the same state without
            # reimplementing the rollup logic.
            "badge": summarize_health(subsystems, overall=overall),
        }

    # ── Probes ────────────────────────────────────────────────────

    def _probe_bus(self) -> dict[str, Any]:
        try:
            if self.bus is None:
                return {"status": _STATUS_UNKNOWN, "detail": "bus not wired"}
            active = 0
            try:
                active = len(list(self.bus._active_tasks))  # type: ignore[attr-defined]
            except Exception:
                active = -1
            return {
                "status": _STATUS_OK,
                "active_tasks": active,
                "worker_running": bool(
                    getattr(self.bus, "_worker_task", None) is not None
                    and not getattr(self.bus._worker_task, "done", lambda: True)()
                ),
            }
        except Exception as exc:
            return {"status": _STATUS_DEGRADED, "detail": repr(exc)}

    def _probe_stt(self) -> dict[str, Any]:
        stt = self.stt
        if stt is None:
            return {"status": _STATUS_UNKNOWN, "detail": "stt not wired"}
        try:
            listening = bool(
                getattr(stt, "listening", False) or getattr(stt, "_running", False)
            )
            last_err = getattr(stt, "_last_error_code", None)
            restarts = int(getattr(stt, "_restart_count", 0) or 0)
            state = _STATUS_OK if listening else _STATUS_DEGRADED
            detail = {
                "listening": listening,
                "restart_count": restarts,
            }
            if last_err:
                detail["last_error"] = str(last_err)
            return {"status": state, **detail}
        except Exception as exc:
            return {"status": _STATUS_DEGRADED, "detail": repr(exc)}

    def _probe_tts(self) -> dict[str, Any]:
        tts = self.tts
        if tts is None:
            return {"status": _STATUS_UNKNOWN, "detail": "tts not wired"}
        try:
            speaking = bool(getattr(tts, "_speaking", False))
            deadman_active = getattr(tts, "_deadman_task", None) is not None
            budget_s = float(getattr(tts, "_speak_budget_s", 0.0) or 0.0)
            return {
                "status": _STATUS_OK,
                "speaking": speaking,
                "deadman_armed": deadman_active,
                "current_budget_s": budget_s,
            }
        except Exception as exc:
            return {"status": _STATUS_DEGRADED, "detail": repr(exc)}

    def _probe_brain(self) -> dict[str, Any]:
        lb = self.local_brain
        if lb is None:
            return {"status": _STATUS_UNKNOWN, "detail": "local brain disabled"}
        try:
            available = bool(getattr(lb, "available", False))
            stats_fn = getattr(lb, "prompt_cache_stats", None)
            cache_stats: dict[str, Any] = {}
            if callable(stats_fn):
                try:
                    cache_stats = dict(stats_fn() or {})
                except Exception:
                    cache_stats = {}
            llm = getattr(lb, "_llm", None)
            clamp_ratio = float(getattr(llm, "_thermal_clamp_ratio", 1.0) or 1.0) if llm is not None else 1.0
            return {
                "status": _STATUS_OK if available else _STATUS_DEGRADED,
                "available": available,
                "runtime_mode": getattr(lb, "_current_runtime_mode", "unknown"),
                "prompt_cache_stats": cache_stats,
                "thermal_clamp_ratio": clamp_ratio,
            }
        except Exception as exc:
            return {"status": _STATUS_DEGRADED, "detail": repr(exc)}

    def _probe_embeddings(self) -> dict[str, Any]:
        emb = self.embedding_engine
        if emb is None:
            return {"status": _STATUS_UNKNOWN, "detail": "embedding engine not wired"}
        try:
            loaded = bool(getattr(emb, "_model_loaded", False))
            cached = int(getattr(emb, "_cache_size", 0) or 0)
            warm_enabled = bool(getattr(emb, "_warm_enabled", False))
            return {
                "status": _STATUS_OK,
                "model_loaded": loaded,
                "cached_vectors": cached,
                "warm_file_enabled": warm_enabled,
            }
        except Exception as exc:
            return {"status": _STATUS_DEGRADED, "detail": repr(exc)}

    def _probe_semantic_cache(self) -> dict[str, Any]:
        sc = self.semantic_cache
        if sc is None:
            return {"status": _STATUS_UNKNOWN, "detail": "semantic cache not wired"}
        try:
            hit_fn = getattr(sc, "get_stats", None)
            stats: dict[str, Any] = {}
            if callable(hit_fn):
                try:
                    stats = dict(hit_fn() or {})
                except Exception:
                    stats = {}
            return {
                "status": _STATUS_OK,
                **stats,
            }
        except Exception as exc:
            return {"status": _STATUS_DEGRADED, "detail": repr(exc)}

    def _probe_memory(self) -> dict[str, Any]:
        mem = self.memory
        if mem is None:
            return {"status": _STATUS_UNKNOWN, "detail": "memory not wired"}
        try:
            diag = getattr(mem, "diagnostics", None)
            if callable(diag):
                return {"status": _STATUS_OK, **(dict(diag() or {}))}
            return {"status": _STATUS_OK}
        except Exception as exc:
            return {"status": _STATUS_DEGRADED, "detail": repr(exc)}

    def _probe_silicon(self) -> dict[str, Any]:
        sg = self.silicon_governor
        if sg is None or not getattr(sg, "is_available", False):
            return {"status": _STATUS_UNKNOWN, "detail": "silicon governor unavailable"}
        try:
            s = sg.get_stats()
            memory_pct = float(getattr(s, "memory_pct", 0.0) or 0.0)
            cpu_pct = float(getattr(s, "cpu_pct", 0.0) or 0.0)
            thermal = str(getattr(s, "thermal_pressure", "nominal") or "nominal")
            throttled = bool(getattr(s, "is_throttled", False))
            status = _STATUS_OK
            if throttled or thermal in {"critical", "hot", "heavy"}:
                status = _STATUS_DEGRADED
            if memory_pct > 92.0:
                status = _STATUS_DEGRADED
            return {
                "status": status,
                "cpu_pct": cpu_pct,
                "memory_pct": memory_pct,
                "thermal_pressure": thermal,
                "throttled": throttled,
            }
        except Exception as exc:
            return {"status": _STATUS_DEGRADED, "detail": repr(exc)}

    def _probe_state(self) -> dict[str, Any]:
        st = self.state
        if st is None:
            return {"status": _STATUS_UNKNOWN}
        try:
            current = getattr(st, "current", None)
            current_name = getattr(current, "name", str(current))
            return {"status": _STATUS_OK, "current": current_name}
        except Exception as exc:
            return {"status": _STATUS_DEGRADED, "detail": repr(exc)}

    def _probe_errors(self) -> dict[str, Any]:
        em = self.error_monitor
        if em is None:
            try:
                from core.observability.error_rate_monitor import get_error_rate_monitor
                em = get_error_rate_monitor()
            except Exception:
                em = None
        if em is None:
            return {"status": _STATUS_UNKNOWN}
        try:
            diag = em.diagnostics()
            rate = int(diag.get("recent_rate", 0))
            threshold = int(diag.get("threshold", 0))
            status = _STATUS_OK
            if threshold and rate >= threshold:
                status = _STATUS_DEGRADED
            return {"status": status, **diag}
        except Exception as exc:
            return {"status": _STATUS_DEGRADED, "detail": repr(exc)}


def summarize_health(
    subsystems: dict[str, Any],
    *,
    overall: str | None = None,
) -> dict[str, Any]:
    """Distil per-subsystem probes into one menubar-ready badge.

    Sprint P4.6 (Apr 26 2026). Returns a stable, JSON-serializable
    dict shaped::

        {
            "level": "ok" | "warn" | "critical" | "unknown",
            "color": "green" | "amber" | "red" | "grey",
            "text":  "ATOM is OK"             # or
                     "ATOM has 2 warnings"     # or
                     "ATOM is critical",
            "headline":   "stt: degraded · brain: down",
            "warnings":   [{"name": "stt", "status": "degraded", ...}, ...],
            "criticals":  [{"name": "brain", "status": "down", ...}, ...],
            "subsystems_total": 10,
        }

    The function is pure (no I/O, no logging) so the dashboard, the
    iPhone bridge, and the menubar poller can all call it on the same
    snapshot and get identical output.

    ``subsystems`` is the dict produced by
    :py:meth:`HealthSnapshotBuilder.build`. ``overall`` may be supplied
    (cheap path) or recomputed from ``subsystems`` if ``None`` (used by
    callers that hold the raw subsystems dict but not the rollup).
    """
    warnings: list[dict[str, Any]] = []
    criticals: list[dict[str, Any]] = []
    unknown: list[dict[str, Any]] = []

    if overall is None:
        rollup = _STATUS_OK
        for sub in subsystems.values():
            if not isinstance(sub, dict):
                continue
            status = str(sub.get("status", _STATUS_UNKNOWN))
            if status == _STATUS_DOWN:
                rollup = _STATUS_DOWN
                break
            if status == _STATUS_DEGRADED and rollup == _STATUS_OK:
                rollup = _STATUS_DEGRADED
        overall = rollup

    for name, sub in (subsystems or {}).items():
        if not isinstance(sub, dict):
            continue
        status = str(sub.get("status", _STATUS_UNKNOWN))
        entry = {"name": name, "status": status}
        detail = sub.get("detail")
        if detail:
            entry["detail"] = str(detail)[:160]
        if status == _STATUS_DOWN:
            criticals.append(entry)
        elif status == _STATUS_DEGRADED:
            warnings.append(entry)
        elif status == _STATUS_UNKNOWN:
            unknown.append(entry)

    if overall == _STATUS_DOWN:
        level, color = "critical", "red"
        n = len(criticals) or 1
        text = (
            "ATOM is critical"
            if n == 1
            else f"ATOM is critical ({n} subsystems down)"
        )
    elif overall == _STATUS_DEGRADED:
        level, color = "warn", "amber"
        n = len(warnings) or 1
        text = (
            "ATOM has 1 warning"
            if n == 1
            else f"ATOM has {n} warnings"
        )
    elif overall == _STATUS_OK:
        level, color = "ok", "green"
        text = "ATOM is OK"
    else:
        level, color = "unknown", "grey"
        text = "ATOM status unknown"

    headline_parts: list[str] = []
    for e in criticals[:3]:
        headline_parts.append(f"{e['name']}: down")
    for e in warnings[:3]:
        headline_parts.append(f"{e['name']}: degraded")
    headline = " · ".join(headline_parts)

    return {
        "level": level,
        "color": color,
        "text": text,
        "headline": headline,
        "warnings": warnings,
        "criticals": criticals,
        "unknown": unknown,
        "subsystems_total": len(subsystems or {}),
    }


__all__ = ["HealthSnapshotBuilder", "summarize_health"]
