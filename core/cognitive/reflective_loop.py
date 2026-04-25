"""
ATOM -- Reflective Cognitive Loop (Phase G1).

The "Jarvis-grade" rewrite stops treating each user turn as the end
of the conversation. After ATOM speaks, this loop pauses for a beat,
asks the local LLM "given what just happened, is there anything I
should DO next, ADVISE about, or CLARIFY?", and acts on the answer.

Design contract (kept deliberately small to stay safe and snappy):

* **Event-driven, not polled.** We subscribe to ``tts_complete`` and
  reflect once the user is back to IDLE. No background timer.
* **Hard cooldown** between reflections (default 60 s) so we never
  pile chains of unsolicited follow-ups.
* **Short-circuit on new user turn.** If the user starts speaking
  again before we finish reflecting, we silently abort -- the turn
  cycle takes precedence over reflection.
* **JSON-only LLM call.** A tight prompt asks for
  ``{"decision": "...", "intent": "...", "text": "..."}``. Anything
  not parseable becomes ``{"decision": "none"}`` -- silence is the
  default.
* **Decisions** = ``none`` (do nothing), ``advise`` (speak a short
  insight), ``clarify`` (ask a one-line question), ``execute`` (queue
  a synthetic command back through CommandLoop).

The module is wired in `core/boot/wiring.py` and gets its LLM via
dependency injection so unit tests can stub it out.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus

logger = logging.getLogger("atom.cognitive.reflect")


# ── public types ────────────────────────────────────────────────────


VALID_DECISIONS = ("none", "advise", "clarify", "execute")


@dataclass(slots=True)
class ReflectionDecision:
    """Structured outcome from one reflective pass."""

    decision: str = "none"
    intent: str = ""
    text: str = ""
    raw: str = ""

    def is_actionable(self) -> bool:
        return self.decision in {"advise", "clarify", "execute"} \
            and bool(self.text.strip())


# Type alias for the LLM provider. Returns (text, ok).
LLMProvider = Callable[[str], Awaitable[tuple[str, bool]]]


# ── internal turn snapshot ──────────────────────────────────────────


@dataclass(slots=True)
class _TurnSnapshot:
    user_text: str = ""
    response_text: str = ""
    intent: str = ""
    action: str = ""
    timestamp: float = field(default_factory=time.monotonic)


# ── prompt template ────────────────────────────────────────────────


_REFLECTION_PROMPT = """\
You are ATOM's *reflection layer*. The user (Boss) just had this short
exchange with you. Decide whether you, as a vigilant Jarvis-style
assistant, should follow up. Return ONLY one JSON object on a single
line. No markdown, no prose, no ``` fences.

Schema:
  {{
    "decision": "none" | "advise" | "clarify" | "execute",
    "intent":   "<short label, e.g. 'reminder', 'next_step', 'idle'>",
    "text":     "<one-sentence response if decision != none, else ''>"
  }}

Rules:
* Default to "none". Only follow up when there is real, specific
  value -- not chit-chat, not obvious filler.
* "advise"  = surface a useful insight or nudge in <= 18 words.
* "clarify" = ask ONE concise question to unblock the user.
* "execute" = name a tool the user clearly implied next; the text
  field should be a natural sentence the assistant will say WHILE
  the action runs.
* Never repeat what you just said.
* Never moralise, never apologise.

Last turn:
  user:   {user_text}
  intent: {intent}
  action: {action}
  atom:   {response_text}

Return JSON now."""


def build_prompt(snapshot: _TurnSnapshot) -> str:
    return _REFLECTION_PROMPT.format(
        user_text=(snapshot.user_text or "").strip()[:400] or "(empty)",
        intent=snapshot.intent or "unknown",
        action=snapshot.action or "none",
        response_text=(snapshot.response_text or "").strip()[:400] or "(empty)",
    )


# ── parser (forgiving) ─────────────────────────────────────────────


def parse_decision(raw: str) -> ReflectionDecision:
    """Parse the model output into a :class:`ReflectionDecision`.

    The parser is intentionally forgiving:

    * Strips Markdown code fences and any prose around the JSON.
    * Falls back to ``decision="none"`` on any error.
    """
    if not raw:
        return ReflectionDecision(raw="")

    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        logger.debug("reflection parse: no JSON object in %r", raw[:120])
        return ReflectionDecision(raw=raw)

    blob = text[start : end + 1]
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        logger.debug("reflection parse: JSON decode failed on %r", blob[:120])
        return ReflectionDecision(raw=raw)
    if not isinstance(data, dict):
        return ReflectionDecision(raw=raw)

    decision = str(data.get("decision", "none")).lower().strip()
    if decision not in VALID_DECISIONS:
        decision = "none"
    intent = str(data.get("intent", "") or "").strip()
    text_out = str(data.get("text", "") or "").strip()

    if decision == "none":
        text_out = ""
    if decision != "none" and not text_out:
        decision = "none"

    return ReflectionDecision(
        decision=decision,
        intent=intent[:64],
        text=text_out[:240],
        raw=raw,
    )


# ── the loop ───────────────────────────────────────────────────────


class ReflectiveLoop:
    """Subscribes to turn-end events and runs an LLM reflection pass.

    The loop is **always opt-in**: callers must call :meth:`attach`
    on the live :class:`~core.async_event_bus.AsyncEventBus`. We
    also accept a ``state_provider`` callable so that
    :class:`~core.state_manager.StateManager` can short-circuit us
    when ATOM is busy listening or speaking.
    """

    __slots__ = (
        "_bus", "_llm", "_state_provider",
        "_cooldown_s", "_min_user_chars",
        "_last_reflection_at", "_turn",
        "_in_flight", "_attached",
        "_total_attempts", "_total_advise", "_total_clarify",
        "_total_execute", "_total_silent",
        "_response_emitter", "_execute_emitter",
        "_consecutive_failures", "_failure_threshold",
        "_disabled_until", "_disable_cooldown_s",
        "_total_provider_failures",
    )

    def __init__(
        self,
        bus: "AsyncEventBus",
        llm: LLMProvider,
        *,
        cooldown_s: float = 60.0,
        min_user_chars: int = 5,
        state_provider: Callable[[], str] | None = None,
        response_emitter: Callable[[str], None] | None = None,
        execute_emitter: Callable[[str], None] | None = None,
        failure_threshold: int = 3,
        disable_cooldown_s: float = 600.0,
    ) -> None:
        self._bus = bus
        self._llm = llm
        self._cooldown_s = float(cooldown_s)
        self._min_user_chars = int(min_user_chars)
        self._state_provider = state_provider
        self._last_reflection_at = 0.0
        self._turn = _TurnSnapshot()
        self._in_flight = False
        self._attached = False
        self._total_attempts = 0
        self._total_advise = 0
        self._total_clarify = 0
        self._total_execute = 0
        self._total_silent = 0
        self._response_emitter = response_emitter
        self._execute_emitter = execute_emitter
        # Circuit-breaker: after N consecutive provider failures, stop
        # trying for `disable_cooldown_s` seconds. Stops the log spam
        # we saw in atomLogs.txt when the wrong object was wired in.
        self._consecutive_failures = 0
        self._failure_threshold = max(1, int(failure_threshold))
        self._disabled_until = 0.0
        self._disable_cooldown_s = max(1.0, float(disable_cooldown_s))
        self._total_provider_failures = 0

    # ── public API ───────────────────────────────────────────────

    def attach(self) -> None:
        """Register event subscribers. Idempotent."""
        if self._attached:
            return
        self._bus.on("command_loop_trace", self._on_command_trace)
        self._bus.on("response_ready", self._on_response_ready)
        self._bus.on("tts_complete", self._on_tts_complete)
        self._bus.on("speech_final", self._on_speech_final)
        self._attached = True
        logger.info(
            "ReflectiveLoop attached (cooldown=%.1fs, min_chars=%d)",
            self._cooldown_s, self._min_user_chars,
        )

    def detach(self) -> None:
        if not self._attached:
            return
        self._bus.off("command_loop_trace", self._on_command_trace)
        self._bus.off("response_ready", self._on_response_ready)
        self._bus.off("tts_complete", self._on_tts_complete)
        self._bus.off("speech_final", self._on_speech_final)
        self._attached = False

    @property
    def metrics(self) -> dict[str, Any]:
        return {
            "attempts": self._total_attempts,
            "advise":   self._total_advise,
            "clarify":  self._total_clarify,
            "execute":  self._total_execute,
            "silent":   self._total_silent,
            "cooldown_s": self._cooldown_s,
            "last_reflection_age_s": (
                round(time.monotonic() - self._last_reflection_at, 2)
                if self._last_reflection_at else None
            ),
        }

    # ── event handlers ───────────────────────────────────────────

    async def _on_command_trace(
        self,
        *,
        stage: str = "",
        text: str = "",
        intent: str = "",
        action: str = "",
        **_kw: Any,
    ) -> None:
        if stage == "start":
            self._turn = _TurnSnapshot(
                user_text=text,
                intent=intent,
                action=action,
            )
        elif stage in ("done", "error", "cancelled"):
            if intent and not self._turn.intent:
                self._turn.intent = intent
            if action and not self._turn.action:
                self._turn.action = action

    async def _on_response_ready(self, *, text: str = "", **_kw: Any) -> None:
        if text:
            self._turn.response_text = text

    async def _on_speech_final(self, **_kw: Any) -> None:
        """Short-circuit: a new user turn arrived; reset reflection guard."""
        if self._in_flight:
            logger.debug("reflection: cancelling -- new user turn started")
        self._in_flight = False

    async def _on_tts_complete(self, **_kw: Any) -> None:
        if self._in_flight:
            return
        if self._breaker_open():
            return
        if not self._cooldown_passed():
            return
        if not self._has_meaningful_turn():
            return
        if not self._idle_or_unknown():
            return

        self._in_flight = True
        try:
            await self._reflect()
        except asyncio.CancelledError:  # pragma: no cover
            raise
        except Exception:
            logger.exception("ReflectiveLoop: unexpected error")
        finally:
            self._in_flight = False

    # ── reflection core ─────────────────────────────────────────

    async def _reflect(self) -> None:
        snapshot = _TurnSnapshot(
            user_text=self._turn.user_text,
            response_text=self._turn.response_text,
            intent=self._turn.intent,
            action=self._turn.action,
            timestamp=self._turn.timestamp,
        )

        prompt = build_prompt(snapshot)
        self._total_attempts += 1

        try:
            raw, ok = await self._llm(prompt)
        except Exception:
            logger.exception("ReflectiveLoop: LLM call raised")
            self._note_provider_failure()
            return

        if not ok:
            self._note_provider_failure()
        else:
            self._consecutive_failures = 0

        decision = parse_decision(raw if ok else "")
        self._last_reflection_at = time.monotonic()

        # Re-check after the LLM round trip -- the user may have
        # started talking again while we were waiting.
        if not self._idle_or_unknown():
            logger.debug("reflection: aborting (state no longer idle)")
            self._total_silent += 1
            return

        if not decision.is_actionable():
            self._total_silent += 1
            logger.debug("reflection: silent (decision=%s)", decision.decision)
            return

        await self._dispatch(decision)

    async def _dispatch(self, decision: ReflectionDecision) -> None:
        text = decision.text.strip()
        if decision.decision == "advise":
            self._total_advise += 1
            self._emit_response(text)
        elif decision.decision == "clarify":
            self._total_clarify += 1
            self._emit_response(text)
        elif decision.decision == "execute":
            self._total_execute += 1
            self._emit_execute(text)

        logger.info(
            "reflect dispatched: decision=%s intent=%s text=%r",
            decision.decision, decision.intent, text[:80],
        )

    def _emit_response(self, text: str) -> None:
        if self._response_emitter is not None:
            try:
                self._response_emitter(text)
                return
            except Exception:
                logger.exception("response_emitter raised; falling back to bus")
        self._bus.emit_long(
            "response_ready",
            text=text,
            source="reflective_loop",
            proactive=True,
        )

    def _emit_execute(self, text: str) -> None:
        # Speak the natural sentence first; the actual command is
        # routed through the same path a user voice command takes,
        # via ``execute_emitter`` which wiring.py supplies (it pushes
        # the text into CommandLoop.submit). When no emitter is
        # given we degrade to "advise" -- safer than blind exec.
        if self._execute_emitter is not None:
            try:
                self._execute_emitter(text)
                return
            except Exception:
                logger.exception("execute_emitter raised; falling back to advise")
        self._emit_response(text)

    # ── guards ──────────────────────────────────────────────────

    def _cooldown_passed(self) -> bool:
        if self._last_reflection_at == 0.0:
            return True
        return (time.monotonic() - self._last_reflection_at) >= self._cooldown_s

    def _has_meaningful_turn(self) -> bool:
        if not self._turn.user_text:
            return False
        return len(self._turn.user_text.strip()) >= self._min_user_chars

    def _idle_or_unknown(self) -> bool:
        if self._state_provider is None:
            return True
        try:
            state = (self._state_provider() or "").lower()
        except Exception:
            return True
        return state in ("", "idle", "listening")

    # ── circuit-breaker ─────────────────────────────────────────

    def _breaker_open(self) -> bool:
        if self._disabled_until <= 0.0:
            return False
        if time.monotonic() >= self._disabled_until:
            # Cooldown expired -- give the provider a fresh chance.
            logger.info(
                "ReflectiveLoop: breaker cooldown elapsed, re-enabling",
            )
            self._disabled_until = 0.0
            self._consecutive_failures = 0
            return False
        return True

    def _note_provider_failure(self) -> None:
        self._total_provider_failures += 1
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._failure_threshold:
            self._disabled_until = (
                time.monotonic() + self._disable_cooldown_s
            )
            logger.warning(
                "ReflectiveLoop: %d consecutive provider failures -- "
                "tripping breaker for %.0fs",
                self._consecutive_failures, self._disable_cooldown_s,
            )


# ── factory helper for wiring.py ───────────────────────────────────


def make_default_llm_provider(
    mlx_brain: Any,
    *,
    model_role: str = "fast",
    max_tokens: int = 220,
) -> LLMProvider:
    """Wrap an :class:`brain.mlx_llm.MLXBrain` into the loop's contract."""

    async def _provider(prompt: str) -> tuple[str, bool]:
        try:
            return await mlx_brain.generate(
                prompt,
                model_role=model_role,
                max_tokens_override=max_tokens,
            )
        except Exception:
            logger.exception("ReflectiveLoop LLM provider failed")
            return "", False

    return _provider


__all__ = [
    "ReflectiveLoop",
    "ReflectionDecision",
    "VALID_DECISIONS",
    "build_prompt",
    "parse_decision",
    "make_default_llm_provider",
]
