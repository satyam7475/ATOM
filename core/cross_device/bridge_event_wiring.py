"""Wire iPhone bridge events into identity + proactive engines.

Separate module so :py:mod:`main` just calls one function and ships
no cross-device-specific knowledge. Easier to test, easier to
delete if Phase 1 ever gets rolled back.

Events subscribed
-----------------

* ``iphone.faceid.verified`` -- updates
  :py:class:`core.identity_engine.IdentityEngine` so the tier-3 gate
  and the proactive engine can ask ``is_owner_verified()``.
* ``iphone.presence.changed`` -- routes through
  :py:class:`core.proactive_awareness.ProactiveAwareness.handle_iphone_presence`
  and forwards the resulting hint to ``speak`` (injected callable).
* ``iphone.trigger.fired`` -- routes through
  ``ProactiveAwareness.handle_iphone_trigger`` and dispatches the
  trigger name to a caller-supplied handler (``on_trigger`` arg).
  This keeps MorningBriefing / focus-mode invocation out of this
  module.

Every subscriber is async + error-isolated so a buggy downstream
handler cannot poison the event bus.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger("atom.bridge.wiring")


SpeakFn = Callable[[str], Any]
"""Callable that speaks a hint (sync or async). Typically ATOM's TTS
``say`` method; tests pass a ``list.append`` sink."""

TriggerFn = Callable[[str, dict[str, Any]], Any]
"""Callable invoked when a named iPhone trigger fires. Receives
``(name, args)``; should return quickly (the bridge handler is
already fire-and-forget at the HTTP layer, but we don't want to
pile up event-bus tasks)."""


def wire_bridge_events(
    *,
    bus: Any,
    identity_engine: Any,
    proactive: Any,
    speak: Optional[SpeakFn] = None,
    on_trigger: Optional[TriggerFn] = None,
) -> None:
    """Register async subscribers on *bus*. Idempotent -- calling twice
    registers twice, so callers should ensure one-shot during boot.
    """
    if bus is None:
        logger.warning("bridge wiring skipped: bus is None")
        return

    async def _on_faceid(**data: Any) -> None:
        try:
            if identity_engine is None:
                return
            verified = bool(data.get("verified"))
            ts = data.get("timestamp")
            device_id = str(data.get("device_id") or "")
            label = str(data.get("label") or "")
            identity_engine.record_faceid_verification(
                verified,
                timestamp=ts,
                device_id=device_id,
                label=label,
            )
        except Exception:  # noqa: BLE001 -- event-bus subscribers must never raise
            logger.exception("iphone.faceid.verified handler failed")

    async def _on_presence(**data: Any) -> None:
        try:
            if proactive is None:
                return
            state = str(data.get("state") or "").strip().lower()
            ts = data.get("timestamp")
            hint = proactive.handle_iphone_presence(state, timestamp=ts)
            if hint and speak is not None:
                await _maybe_await(speak(hint))
        except Exception:
            logger.exception("iphone.presence.changed handler failed")

    async def _on_trigger(**data: Any) -> None:
        try:
            if proactive is None:
                return
            name = str(data.get("name") or "").strip().lower()
            args_raw = data.get("args")
            args = args_raw if isinstance(args_raw, dict) else {}
            envelope = proactive.handle_iphone_trigger(name, args=args)
            if not envelope:
                return
            if speak is not None:
                await _maybe_await(speak(envelope["ack"]))
            if on_trigger is not None:
                await _maybe_await(on_trigger(envelope["trigger"], args))
        except Exception:
            logger.exception("iphone.trigger.fired handler failed")

    bus.on("iphone.faceid.verified", _on_faceid)
    bus.on("iphone.presence.changed", _on_presence)
    bus.on("iphone.trigger.fired", _on_trigger)
    logger.info("iphone bridge subscribers wired (faceid, presence, trigger)")


async def _maybe_await(result: Any) -> None:
    """Allow the caller to pass either sync or async callbacks."""
    if result is None:
        return
    if inspect.isawaitable(result):
        await result
