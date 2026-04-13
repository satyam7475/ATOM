"""
ATOM -- Indicator adapter that mirrors runtime UI calls into AtomState.
"""

from __future__ import annotations

from typing import Any

from .event_bus import AtomRuntimeStateBridge

_STATE_META: dict[str, dict[str, str]] = {
    "sleep": {"label": "SLEEP", "status": "Standby"},
    "idle": {"label": "IDLE", "status": "Ready"},
    "listening": {"label": "LISTENING", "status": "Listening..."},
    "thinking": {"label": "THINKING", "status": "Processing..."},
    "speaking": {"label": "SPEAKING", "status": "Speaking..."},
    "error_recovery": {"label": "RECOVERY", "status": "Recovering..."},
}


class StateAwareIndicator:
    """Delegate UI calls while keeping AtomState as the read model."""

    __slots__ = ("_base", "_bridge")

    def __init__(self, base: Any, bridge: AtomRuntimeStateBridge) -> None:
        self._base = base
        self._bridge = bridge

    @property
    def raw(self) -> Any:
        return self._base

    def update_state(self, state_value: str) -> None:
        meta = _STATE_META.get(state_value, _STATE_META["sleep"])
        self._bridge.patch_section(
            "lifecycle",
            {
                "state": state_value,
                "label": meta["label"],
                "status": meta["status"],
            },
            source="indicator.update_state",
        )
        if hasattr(self._base, "update_state"):
            self._base.update_state(state_value)

    async def on_state_changed(self, old: Any, new: Any, **_kw: Any) -> None:
        state_value = getattr(new, "value", str(new))
        self.update_state(state_value)
        if hasattr(self._base, "on_state_changed"):
            await self._base.on_state_changed(old, new, **_kw)

    def add_log(self, tag: str, message: str) -> None:
        msg = str(message or "").strip()
        patch_execution: dict[str, Any] = {}
        patch_voice: dict[str, Any] = {}
        patch_reasoning: dict[str, Any] = {}
        patch_health: dict[str, Any] = {}

        if tag == "heard":
            patch_execution["last_query"] = msg
        elif tag == "speaking":
            patch_voice["last_spoken"] = msg
        elif tag in {"action", "jarvis", "reminder"}:
            patch_execution["last_action"] = msg
            patch_reasoning["last_report"] = msg
        elif tag in {"info", "warning"}:
            patch_reasoning["last_report"] = msg
            if tag == "warning":
                patch_reasoning["severity"] = "warning"
                warnings = list(self._bridge.store.get_section("health").get("warnings", []))
                if msg and msg not in warnings:
                    warnings.append(msg)
                patch_health["warnings"] = warnings[-10:]
        if patch_execution:
            self._bridge.patch_section("execution", patch_execution, source=f"log.{tag}")
        if patch_voice:
            self._bridge.patch_section("voice", patch_voice, source=f"log.{tag}")
        if patch_reasoning:
            self._bridge.patch_section("reasoning", patch_reasoning, source=f"log.{tag}")
        if patch_health:
            self._bridge.patch_section("health", patch_health, source=f"log.{tag}")
        if hasattr(self._base, "add_log"):
            self._base.add_log(tag, message)

    def show_hearing(self, text: str) -> None:
        self._bridge.patch_section(
            "voice",
            {
                "last_partial": str(text or "")[:240],
                "status": "listening",
            },
            source="indicator.show_hearing",
        )
        if hasattr(self._base, "show_hearing"):
            self._base.show_hearing(text)

    def clear_hearing(self) -> None:
        self._bridge.patch_section(
            "voice",
            {"last_partial": ""},
            source="indicator.clear_hearing",
        )
        if hasattr(self._base, "clear_hearing"):
            self._base.clear_hearing()

    def set_mic_name(self, name: str) -> None:
        self._bridge.patch_section(
            "voice",
            {"mic": str(name or "")},
            source="indicator.set_mic_name",
        )
        if hasattr(self._base, "set_mic_name"):
            self._base.set_mic_name(name)

    def set_owner_status(self, detected: bool = False, status: str = "disabled") -> None:
        self._bridge.patch_section(
            "reasoning",
            {
                "last_decision": f"Owner status: {status}",
            },
            source="indicator.owner_status",
        )
        if hasattr(self._base, "set_owner_status"):
            self._base.set_owner_status(detected=detected, status=status)

    def set_last_query(self, text: str) -> None:
        self._bridge.patch_section(
            "execution",
            {"last_query": str(text or "")[:240]},
            source="indicator.set_last_query",
        )
        if hasattr(self._base, "set_last_query"):
            self._base.set_last_query(text)

    def set_last_intent(self, intent: str) -> None:
        self._bridge.patch_section(
            "execution",
            {"last_intent": str(intent or "")},
            source="indicator.set_last_intent",
        )
        if hasattr(self._base, "set_last_intent"):
            self._base.set_last_intent(intent)

    def set_last_latency_ms(self, ms: float) -> None:
        self._bridge.patch_section(
            "execution",
            {"latency_ms": float(ms or 0.0)},
            source="indicator.set_last_latency_ms",
        )
        if hasattr(self._base, "set_last_latency_ms"):
            self._base.set_last_latency_ms(ms)

    def set_status(self, text: str) -> None:
        self._bridge.patch_section(
            "lifecycle",
            {"status": str(text or "")},
            source="indicator.set_status",
        )
        if hasattr(self._base, "set_status"):
            self._base.set_status(text)

    def set_init_info(self, **kwargs: Any) -> None:
        voice_patch: dict[str, Any] = {}
        mode_patch: dict[str, Any] = {}
        reasoning_patch: dict[str, Any] = {}
        if "stt" in kwargs:
            voice_patch["stt_engine"] = str(kwargs.get("stt") or "")
        if "tts" in kwargs:
            voice_patch["tts_engine"] = str(kwargs.get("tts") or "")
        if "voice_note" in kwargs:
            reasoning_patch["last_report"] = str(kwargs.get("voice_note") or "")
        if "perf_mode_requested" in kwargs:
            mode_patch["requested"] = str(kwargs.get("perf_mode_requested") or "")
        if "perf_mode" in kwargs:
            mode_patch["effective"] = str(kwargs.get("perf_mode") or "")
        if "brain_profile" in kwargs:
            mode_patch["profile"] = str(kwargs.get("brain_profile") or "")
        if "assistant_mode" in kwargs:
            mode_patch["assistant_mode"] = str(kwargs.get("assistant_mode") or "")
        if voice_patch:
            self._bridge.patch_section("voice", voice_patch, source="indicator.init")
        if mode_patch:
            self._bridge.patch_section("mode", mode_patch, source="indicator.init")
        if reasoning_patch:
            self._bridge.patch_section("reasoning", reasoning_patch, source="indicator.init")
        if hasattr(self._base, "set_init_info"):
            self._base.set_init_info(**kwargs)

    def broadcast_perf_mode(
        self,
        mode: str,
        *,
        requested_mode: str | None = None,
        reason: str = "",
    ) -> None:
        patch_mode = {
            "effective": str(mode or ""),
            "reason": str(reason or ""),
        }
        if requested_mode is not None:
            patch_mode["requested"] = str(requested_mode)
        self._bridge.patch_section("mode", patch_mode, source="indicator.broadcast_perf_mode")
        if reason:
            self._bridge.patch_section(
                "reasoning",
                {
                    "why_this_mode": str(reason),
                    "last_decision": f"Mode -> {mode}",
                },
                source="indicator.broadcast_perf_mode",
            )
        self._bridge.events.emit_mode_change(
            mode=str(mode or ""),
            requested_mode=str(requested_mode or ""),
            reason=str(reason or ""),
        )
        if hasattr(self._base, "broadcast_perf_mode"):
            self._base.broadcast_perf_mode(mode, requested_mode=requested_mode, reason=reason)

    async def broadcast_runtime_settings(self, brain_profile: str, assistant_mode: str) -> None:
        self._bridge.patch_section(
            "mode",
            {
                "profile": str(brain_profile or ""),
                "assistant_mode": str(assistant_mode or ""),
            },
            source="indicator.runtime_settings",
        )
        if hasattr(self._base, "broadcast_runtime_settings"):
            result = self._base.broadcast_runtime_settings(brain_profile, assistant_mode)
            if hasattr(result, "__await__"):
                await result

    def broadcast_governor(self, throttled: bool) -> None:
        self._bridge.patch_section(
            "system",
            {"hardware": {"is_throttled": bool(throttled)}},
            source="indicator.broadcast_governor",
        )
        if hasattr(self._base, "broadcast_governor"):
            self._base.broadcast_governor(throttled)

    def broadcast_thinking_progress(self, elapsed_s: float, estimate_s: float) -> None:
        remaining = max(0.0, float(estimate_s or 0.0) - float(elapsed_s or 0.0))
        self._bridge.patch_section(
            "execution",
            {
                "status": "running",
                "label": f"thinking ({remaining:.0f}s left)",
            },
            source="indicator.broadcast_thinking_progress",
        )
        if hasattr(self._base, "broadcast_thinking_progress"):
            self._base.broadcast_thinking_progress(elapsed_s, estimate_s)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)
