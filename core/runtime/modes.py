"""
V7 runtime modes: balance latency vs retrieval depth vs hardening.

Does not modify SecurityPolicy; SECURE affects RAG prefetch / late-restart only.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from core.rag.query_classifier import QueryComplexity, classify_query

if TYPE_CHECKING:
    from core.runtime.v7_context import V7RuntimeContext

logger = logging.getLogger("atom.runtime.modes")

MODES = frozenset({"FAST", "SMART", "DEEP", "SECURE"})


def resolve_runtime_mode(
    config: dict[str, Any] | None,
    *,
    query: str,
    gpu_util_pct: float = 0.0,
    user_override: str | None = None,
    system_state: dict[str, Any] | None = None,
    user_activity: str | None = None,
    prediction_accuracy: float | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return (mode_name, debug_info)."""
    cfg = (config or {}).get("v7_intelligence") or {}
    default_mode = str(cfg.get("default_mode", "SMART")).upper()
    if default_mode not in MODES:
        default_mode = "SMART"

    cpu_pct = float((system_state or {}).get("cpu_percent") or 0)
    info: dict[str, Any] = {
        "default": default_mode,
        "gpu_util_pct": gpu_util_pct,
        "cpu_percent": cpu_pct,
    }

    if user_override:
        u = str(user_override).upper().strip()
        if u in MODES:
            info["reason"] = "user_override"
            logger.info("v7_mode_selected mode=%s reason=user_override", u)
            return u, info

    auto = bool(cfg.get("auto_mode", True))
    high_gpu = float(cfg.get("gpu_util_fast_threshold", 92))
    cpu_fast = float(cfg.get("cpu_force_fast_above", 88))
    cpu_idle = float(cfg.get("cpu_idle_deep_below", 28))
    deep_len = int(cfg.get("deep_query_min_chars", 140))
    simple_max = int(cfg.get("simple_query_max_chars", 48))
    pred_acc = float(prediction_accuracy) if prediction_accuracy is not None else None

    cx = classify_query(query or "")
    q = (query or "").strip()

    if not auto:
        logger.info("v7_mode_selected mode=%s reason=config_auto_off", default_mode)
        return default_mode, info

    # System CPU pressure → prefer FAST (latency)
    if cpu_pct >= cpu_fast:
        info["reason"] = "system_cpu"
        logger.info("v7_mode_selected mode=FAST reason=system_cpu cpu=%.1f", cpu_pct)
        return "FAST", info

    if gpu_util_pct >= high_gpu:
        info["reason"] = "gpu_load"
        logger.info("v7_mode_selected mode=FAST reason=gpu_load util=%.1f", gpu_util_pct)
        return "FAST", info

    # Idle system + calm user → allow DEEP for substantive queries
    if (
        cpu_pct < cpu_idle
        and (user_activity or "").lower() == "idle"
        and cx == QueryComplexity.COMPLEX
        and len(q) >= deep_len
    ):
        info["reason"] = "idle_system_deep"
        logger.info("v7_mode_selected mode=DEEP reason=idle_system cpu=%.1f", cpu_pct)
        return "DEEP", info

    if cx == QueryComplexity.COMPLEX and len(q) >= deep_len:
        if pred_acc is not None and pred_acc < float(
            cfg.get("low_prediction_accuracy_deep_threshold", 0.35),
        ):
            info["reason"] = "query_complexity_cautious"
            logger.info("v7_mode_selected mode=SMART reason=low_pred_acc")
            return "SMART", info
        info["reason"] = "query_complexity"
        logger.info("v7_mode_selected mode=DEEP reason=complexity")
        return "DEEP", info

    if cx == QueryComplexity.SIMPLE and len(q) < simple_max:
        info["reason"] = "simple_short"
        logger.info("v7_mode_selected mode=FAST reason=simple_short")
        return "FAST", info

    secure_pref = bool(cfg.get("prefer_secure_when_paranoid_ui", False))
    if secure_pref:
        try:
            from core.lock_modes import normalize_lock_mode
            ctrl = (config or {}).get("control") or {}
            if normalize_lock_mode(ctrl.get("lock_mode", "off")) == "paranoid":
                info["reason"] = "paranoid_lock_hint"
                logger.info("v7_mode_selected mode=SECURE reason=paranoid_hint")
                return "SECURE", info
        except Exception:
            pass

    logger.info("v7_mode_selected mode=%s reason=default_heuristic", default_mode)
    return default_mode, info


class RuntimeModeResolver:
    """Thin wrapper with optional stability guard (cooldown + significance)."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        v7 = self._config.get("v7_intelligence") or {}
        self._stability = v7.get("mode_stability") or {}
        self._cooldown_turns = int(self._stability.get("cooldown_turns", 3))
        self._cpu_delta_sig = float(self._stability.get("significant_cpu_delta", 12.0))
        self._gpu_delta_sig = float(self._stability.get("significant_gpu_delta", 15.0))

        self._last_applied: str | None = None
        self._queries_since_last_switch = 0
        self._last_cx: str | None = None
        self._last_cpu: float | None = None
        self._last_gpu: float | None = None

    def _significant_change(
        self,
        cx: QueryComplexity,
        cpu_pct: float,
        gpu_util_pct: float,
    ) -> bool:
        cx_name = cx.name if hasattr(cx, "name") else str(cx)
        if self._last_cx is not None and cx_name != self._last_cx:
            return True
        if self._last_cpu is not None and abs(cpu_pct - self._last_cpu) >= self._cpu_delta_sig:
            return True
        if self._last_gpu is not None and abs(gpu_util_pct - self._last_gpu) >= self._gpu_delta_sig:
            return True
        if self._last_cpu is None:
            return True
        return False

    def resolve(
        self,
        query: str,
        *,
        gpu_util_pct: float = 0.0,
        user_override: str | None = None,
        system_state: dict[str, Any] | None = None,
        user_activity: str | None = None,
        prediction_accuracy: float | None = None,
        context: "V7RuntimeContext | None" = None,
    ) -> tuple[str, dict[str, Any]]:
        if context is not None:
            system_state = context.system_state or system_state
            gpu_util_pct = context.gpu_util_pct
            prediction_accuracy = context.prediction_accuracy

        candidate, info = resolve_runtime_mode(
            self._config,
            query=query,
            gpu_util_pct=gpu_util_pct,
            user_override=user_override,
            system_state=system_state,
            user_activity=user_activity,
            prediction_accuracy=prediction_accuracy,
        )

        if info.get("reason") == "user_override":
            self._last_applied = candidate
            self._queries_since_last_switch = 0
            cx = classify_query(query or "")
            self._last_cx = cx.name if hasattr(cx, "name") else str(cx)
            self._last_cpu = float((system_state or {}).get("cpu_percent") or 0)
            self._last_gpu = float(gpu_util_pct)
            logger.info(
                "v7_mode_switch from=None to=%s reason=user_override blocked=False",
                candidate,
            )
            return candidate, info

        if not bool((self._config.get("v7_intelligence") or {}).get("mode_stability_enabled", True)):
            self._last_applied = candidate
            return candidate, info

        cpu_pct = float((system_state or {}).get("cpu_percent") or 0)
        cx = classify_query(query or "")

        if self._last_applied is None:
            self._last_applied = candidate
            self._queries_since_last_switch = 0
            self._last_cx = cx.name if hasattr(cx, "name") else str(cx)
            self._last_cpu = cpu_pct
            self._last_gpu = float(gpu_util_pct)
            return candidate, info

        if candidate == self._last_applied:
            self._queries_since_last_switch += 1
            return candidate, info

        if self._queries_since_last_switch < self._cooldown_turns:
            info2 = dict(info)
            info2["reason"] = "stability_cooldown"
            info2["blocked_candidate"] = candidate
            logger.info(
                "v7_mode_switch from=%s to=%s reason=stability_cooldown blocked=True",
                self._last_applied,
                candidate,
            )
            return self._last_applied, info2

        if not self._significant_change(cx, cpu_pct, gpu_util_pct):
            info2 = dict(info)
            info2["reason"] = "stability_no_significant_change"
            info2["blocked_candidate"] = candidate
            logger.info(
                "v7_mode_switch from=%s to=%s reason=no_significant_change blocked=True",
                self._last_applied,
                candidate,
            )
            return self._last_applied, info2

        prev = self._last_applied
        self._last_applied = candidate
        self._queries_since_last_switch = 0
        self._last_cx = cx.name if hasattr(cx, "name") else str(cx)
        self._last_cpu = cpu_pct
        self._last_gpu = float(gpu_util_pct)
        logger.info(
            "v7_mode_switch from=%s to=%s reason=%s blocked=False",
            prev,
            candidate,
            info.get("reason", ""),
        )
        return candidate, info
