"""ATOM -- Intelligence & security event handler wiring.

Handles self-healing, code introspection, security fortress, voice auth,
behavioral auth, and real-world intelligence event handlers.

Extracted from main.py for testability.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.security_fortress import SecurityFortress
    from core.self_healing import SelfHealingEngine
    from core.code_introspector import CodeIntrospector
    from core.real_world_intel import RealWorldIntelligence
    from core.context_fusion import ContextFusionEngine
    from context.context_engine import ContextEngine

logger = logging.getLogger("atom.wiring.intelligence")


def wire_self_healing(
    *,
    bus: AsyncEventBus,
    self_healing: SelfHealingEngine,
    code_introspector: CodeIntrospector,
    security_fortress: SecurityFortress,
    context_engine: Any,
) -> None:
    """Wire self-healing, code introspection, and security event handlers."""

    async def _on_diagnose_failure(intent: str = "", **kw) -> None:
        if intent == "diagnose_failure":
            msg = self_healing.diagnose_failure()
            bus.emit_long("response_ready", text=msg)

    async def _on_fix_it(intent: str = "", **kw) -> None:
        if intent == "fix_it":
            msg = self_healing.fix_last_failure()
            bus.emit_long("response_ready", text=msg)

    async def _on_fix_all(intent: str = "", **kw) -> None:
        if intent == "fix_all":
            msg = self_healing.fix_all()
            bus.emit_long("response_ready", text=msg)

    async def _on_module_health(intent: str = "", **kw) -> None:
        if intent == "module_health":
            msg = self_healing.get_health_summary()
            bus.emit_long("response_ready", text=msg)

    async def _on_read_own_code(intent: str = "", **kw) -> None:
        if intent == "read_own_code":
            if not code_introspector.is_scanned:
                code_introspector.scan()
            msg = code_introspector.explain_architecture()
            health = code_introspector.format_code_health()
            bus.emit_long("response_ready", text=f"{msg} {health}")

    async def _on_explain_module(intent: str = "", **kw) -> None:
        if intent == "explain_module":
            args = kw.get("action_args", {}) or {}
            module = args.get("module", "")
            if module:
                if not code_introspector.is_scanned:
                    code_introspector.scan()
                msg = code_introspector.explain_module(module)
                bus.emit_long("response_ready", text=msg)

    async def _on_search_code(intent: str = "", **kw) -> None:
        if intent == "search_code":
            args = kw.get("action_args", {}) or {}
            query = args.get("query", "")
            if query:
                if not code_introspector.is_scanned:
                    code_introspector.scan()
                results = code_introspector.search_code(query)
                if results:
                    lines = [f"{r['file']}:{r['line']}: {r['content']}" for r in results[:5]]
                    msg = f"Found {len(results)} matches. " + " | ".join(lines)
                else:
                    msg = f"No matches found for '{query}' in the codebase."
                bus.emit_long("response_ready", text=msg)

    async def _on_security_status(intent: str = "", **kw) -> None:
        if intent == "security_status":
            msg = security_fortress.get_security_status()
            integrity_ok, integrity_msg = security_fortress.check_integrity()
            bus.emit_long("response_ready", text=f"{msg} Integrity: {integrity_msg}")

    async def _on_security_lockdown(intent: str = "", **kw) -> None:
        if intent == "security_lockdown":
            security_fortress.log_security_event("lockdown_activated", severity="HIGH")
            bus.emit_long("response_ready",
                          text="Security lockdown activated, Boss. All sensitive operations require authentication.")

    async def _on_failure_report(intent: str = "", **kw) -> None:
        import asyncio
        if intent in ("self_diagnostic", "self_check"):
            failure_report = self_healing.get_failure_report()
            if "No failures" not in failure_report:
                await asyncio.sleep(0.5)
                bus.emit_long("response_ready", text=failure_report)

    bus.on("intent_classified", _on_diagnose_failure)
    bus.on("intent_classified", _on_fix_it)
    bus.on("intent_classified", _on_fix_all)
    bus.on("intent_classified", _on_module_health)
    bus.on("intent_classified", _on_read_own_code)
    bus.on("intent_classified", _on_explain_module)
    bus.on("intent_classified", _on_search_code)
    bus.on("intent_classified", _on_security_status)
    bus.on("intent_classified", _on_security_lockdown)
    bus.on("intent_classified", _on_failure_report)

    logger.info("Self-healing + code introspection + security event handlers wired")


def wire_voice_auth(
    *,
    bus: AsyncEventBus,
    security_fortress: SecurityFortress,
    context_engine: Any,
) -> None:
    """Wire voice auth and behavioral auth event handlers."""

    async def _on_voice_enroll(intent: str = "", **kw) -> None:
        if intent == "voice_enroll":
            if not security_fortress.voice_auth.is_available:
                bus.emit_long("response_ready",
                              text="Voice authentication isn't available, Boss. "
                                   "Install resemblyzer or numpy to enable it.")
                return
            bus.emit_long("response_ready",
                          text="Starting voice enrollment. Say a few natural sentences "
                               "and I'll learn your voice, Boss. I need at least 3 samples.")
            security_fortress.log_security_event(
                "voice_enroll_started", severity="INFO",
            )

    async def _on_voice_verify(intent: str = "", **kw) -> None:
        if intent == "voice_verify":
            if not security_fortress.voice_auth.is_enrolled:
                bus.emit_long("response_ready",
                              text="Voice not enrolled yet, Boss. "
                                   "Say 'enroll my voice' to set up voice authentication.")
                return
            bus.emit_long("response_ready",
                          text="Voice verification available. Your last speech will be "
                               "compared against your voice print. "
                               + security_fortress.voice_auth.get_status_message())

    async def _on_voice_auth_status(intent: str = "", **kw) -> None:
        if intent == "voice_auth_status":
            msg = security_fortress.voice_auth.get_status_message()
            bus.emit_long("response_ready", text=msg)

    async def _on_voice_reset(intent: str = "", **kw) -> None:
        if intent == "voice_reset":
            msg = security_fortress.voice_reset()
            bus.emit_long("response_ready", text=msg)

    async def _on_behavior_auth_status(intent: str = "", **kw) -> None:
        if intent == "behavior_auth_status":
            msg = security_fortress.behavior_auth.get_anomaly_report()
            bus.emit_long("response_ready", text=msg)

    async def _on_intent_for_behavior_auth(intent: str = "", **kw) -> None:
        if intent and intent not in ("confirm", "deny", "empty"):
            text = kw.get("text", "")
            target = kw.get("target", "") or kw.get("name", "")
            try:
                active_app = ""
                if hasattr(context_engine, "get_active_window"):
                    active_app = context_engine.get_active_window() or ""
            except Exception:
                active_app = ""
            security_fortress.observe_behavior(
                action=intent,
                detail=target,
                query_text=text,
                active_app=active_app,
            )

    bus.on("intent_classified", _on_voice_enroll)
    bus.on("intent_classified", _on_voice_verify)
    bus.on("intent_classified", _on_voice_auth_status)
    bus.on("intent_classified", _on_voice_reset)
    bus.on("intent_classified", _on_behavior_auth_status)
    bus.on("intent_classified", _on_intent_for_behavior_auth)

    async def _on_behavior_reauth_needed(**_kw) -> None:
        bus.emit_long("response_ready",
                      text="Boss, your usage patterns seem different from usual. "
                           "For security, please verify your identity. "
                           "Say 'verify my voice' or authenticate with your passphrase.")
        security_fortress.log_security_event(
            "behavioral_reauth_triggered",
            f"trust={security_fortress.trust_score:.2f}",
            severity="HIGH",
        )
    bus.on("reauth_required", _on_behavior_reauth_needed)

    def _reauth_callback() -> None:
        bus.emit_long("reauth_required")
    security_fortress.behavior_auth.set_reauth_callback(_reauth_callback)

    logger.info("Auth handlers wired: voice enrollment, verification, behavioral auth")


def wire_real_world(
    *,
    bus: AsyncEventBus,
    real_world_intel: Any,
    context_fusion: Any,
) -> None:
    """Wire real-world intelligence event handlers."""

    async def _on_weather_request(intent: str = "", **_kw) -> None:
        if intent == "weather_report":
            msg = real_world_intel.get_weather_summary()
            bus.emit_long("response_ready", text=msg)

    async def _on_news_request(intent: str = "", **_kw) -> None:
        if intent == "news_headlines":
            msg = real_world_intel.get_news_summary(count=5)
            bus.emit_long("response_ready", text=msg)

    async def _on_world_clock_request(intent: str = "", **_kw) -> None:
        if intent == "world_clock":
            msg = real_world_intel.get_world_clock_summary()
            bus.emit_long("response_ready", text=msg)

    async def _on_briefing_request(intent: str = "", **_kw) -> None:
        if intent == "daily_briefing":
            msg = real_world_intel.get_briefing()
            bus.emit_long("response_ready", text=msg)

    async def _on_temporal_request(intent: str = "", **_kw) -> None:
        if intent == "temporal_info":
            msg = real_world_intel.get_temporal_summary()
            bus.emit_long("response_ready", text=msg)

    async def _on_world_status_request(intent: str = "", **_kw) -> None:
        if intent == "world_status":
            ctx = real_world_intel.get_world_context()
            parts = [real_world_intel.get_temporal_summary()]
            if not ctx.weather.is_stale:
                parts.append(real_world_intel.get_weather_summary())
            if ctx.headlines:
                parts.append(real_world_intel.get_news_summary(3))
            parts.append(real_world_intel.get_world_clock_summary())
            parts.append(f"World intelligence quality: {ctx.quality_score():.0%}")
            bus.emit_long("response_ready", text=" ".join(parts))

    async def _on_intent_for_context_fusion(intent: str = "", **kw) -> None:
        if intent and intent not in ("confirm", "deny", "empty"):
            context_fusion.log_action(intent, kw.get("text", "")[:60])

    bus.on("intent_classified", _on_weather_request)
    bus.on("intent_classified", _on_news_request)
    bus.on("intent_classified", _on_world_clock_request)
    bus.on("intent_classified", _on_briefing_request)
    bus.on("intent_classified", _on_temporal_request)
    bus.on("intent_classified", _on_world_status_request)
    bus.on("intent_classified", _on_intent_for_context_fusion)

    logger.info("Real-world intelligence handlers wired: weather, news, briefing, world clock")
