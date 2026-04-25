"""ATOM Sprint Ω6 -- agent supervisor (Friday-class plan decomposer).

This is the small layer between "Boss said something" and "the plan
executor runs a DAG of tools". It does three things:

1. **Decide whether a query even needs a plan.** Light intents
   (greetings, single-tool jumps like "open Spotify") are passed through
   unchanged; the regular Router fast paths handle them in <100 ms.
2. **For real plans, prompt the LLM with four specialist personas.**
   Same model, four system-prompt fragments, one structured response.
   The personas are pure prompt engineering -- there are no separate
   processes, no separate model loads, and no extra LLM calls. We get
   ~80% of the multi-agent quality boost at 0% of the orchestration
   complexity.
3. **Hand the plan off to** :class:`ParallelPlanExecutor`. Single
   security-gated execution path; all the existing audit / metrics /
   confirmation policy still apply.

Personas (kept on purpose -- this is the entire taxonomy)::

    researcher  - "What do I need to know? Pull from memory, web,
                   screen, calendar before deciding anything."
    planner     - "Break the goal into ordered or independent steps.
                   Mark dependencies. Keep it minimal -- 1-5 steps."
    executor    - "Pick the right tool from the registry for each
                   step. Use real tool names, real argument schemas."
    reviewer    - "Before answering Boss, sanity-check: did the plan
                   actually solve the goal? If not, propose a follow-up."

The reviewer step is the cheapest correctness guard we have: it runs
*after* the executor wave finishes and only fires if any step failed
or produced empty output. That's where this layer earns its keep on
hard queries.

Owner: Satyam
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("atom.reasoning.agent_supervisor")


# ── Persona prompts ─────────────────────────────────────────────────


PERSONA_BLOCK = """\
You are ATOM, Satyam's personal AI OS. For complex requests you
operate as a small team wearing four hats. ALWAYS think through all
four hats before emitting a plan, but only emit the final JSON plan.

[RESEARCHER] What context do I already have (memory, screen, last
turns) and what do I genuinely need to fetch? Don't fetch what you
already know.
[PLANNER] What is the smallest correct sequence of tool calls? Two
unrelated lookups should be parallel. A write that depends on a read
should be marked `depends_on`. Cap at 5 steps; if you can't, ask Boss
for clarification instead of guessing.
[EXECUTOR] For each step, pick the EXACT tool name from the registry
and the EXACT argument schema. Never invent tools. If no tool fits, say
so in `rationale` and emit no plan.
[REVIEWER] After the plan runs, the same model is given the step
outcomes and asked to decide whether the user's goal is satisfied. The
reviewer can request a single follow-up step or escalate to Boss.
"""


# ── Heuristics for "does this even need a plan?" ────────────────────


_LIGHT_INTENT_PATTERNS = (
    r"^\s*(?:hi|hey|hello|yo|thanks?|thank you|bye|goodnight)",
    r"^\s*(?:play|pause|stop|next|previous|skip|mute|unmute)\b",
    r"^\s*(?:open|close|launch|quit|kill)\s+\w",
    r"^\s*(?:set|increase|decrease|raise|lower)\s+(?:the\s+)?volume",
    r"^\s*(?:what\s+time|what's\s+the\s+time|what\s+is\s+the\s+time)",
    r"^\s*(?:status|are\s+you\s+there|you\s+up)\s*\??\s*$",
    r"^\s*(?:yes|yeah|yep|no|nope|sure|ok|okay)\s*\.?\s*$",
)
_LIGHT_INTENT_RE = re.compile("|".join(_LIGHT_INTENT_PATTERNS), re.I)


_MULTI_STEP_HINTS = (
    r"\band\s+then\b",
    r"\bafter\s+that\b",
    r"\bonce\s+that\b",
    r"\bthen\s+(?:also|please|can\s+you)\b",
    r"\bin\s+parallel\b",
    r"\bsimultaneously\b",
    r"\bat\s+the\s+same\s+time\b",
    r"\bwhile\s+(?:you|that)\b",
    r"\b(?:research|compare|analyze|investigate|review|audit)\b.{0,40}\b(?:and|then|plus)\b",
    r"\bplan\s+(?:my|the|a)\b",
    r"\bbrief\s+me\b",
)
_MULTI_STEP_RE = re.compile("|".join(_MULTI_STEP_HINTS), re.I)


_ACTION_VERBS = re.compile(
    r"\b(open|play|pause|set|search|find|create|write|send|post|"
    r"delete|move|copy|run|launch|close|increase|decrease|mute|"
    r"compose|draft|schedule|remind|book|cancel|summari[sz]e|"
    r"email|message|text|call|fetch|download|install|update)\b",
    re.I,
)


# Sprint Ω.2 — match an action verb followed by a coordinating
# conjunction within 40 chars. Catches "open spotify and play X",
# "summarise this then send it", "fetch logs plus diff them" without
# tripping on conversational fillers like "hi and how are you".
_COMPOUND_AFTER_VERB_RE = re.compile(
    r"\b(open|play|pause|set|search|find|create|write|send|post|"
    r"delete|move|copy|run|launch|close|increase|decrease|mute|"
    r"compose|draft|schedule|remind|book|cancel|summari[sz]e|"
    r"email|message|text|call|fetch|download|install|update)\b"
    r"[^.?!]{0,40}\b(?:and|then|plus|also|after that|followed by)\b",
    re.I,
)


# ── Data shapes ─────────────────────────────────────────────────────


@dataclass(slots=True)
class SupervisorConfig:
    enabled: bool = True
    min_query_chars_for_plan: int = 24
    max_plan_steps: int = 5
    max_concurrency: int = 3
    per_step_timeout_s: float = 12.0
    review_on_partial_failure: bool = True


@dataclass(slots=True)
class Triage:
    """Pure-CPU triage decision: do we plan, or pass through?"""

    needs_plan: bool
    reason: str
    confidence: float = 0.0


@dataclass(slots=True)
class SupervisorResult:
    """Outcome of running a planned query through the supervisor."""

    used_plan: bool = False
    plan_blob: str = ""
    summary: str = ""
    plan_result: Any = None  # DAGPlanResult | None
    error: str = ""
    follow_up: str = ""
    elapsed_ms: float = 0.0
    decisions: dict[str, int] = field(default_factory=dict)


# ── Supervisor ──────────────────────────────────────────────────────


class AgentSupervisor:
    """Thin coordinator that turns a Boss query into a parallel plan.

    The supervisor is intentionally *pluggable*: it does not own the
    LLM client. The caller passes ``llm_call`` -- any async callable
    ``(prompt: str, system: str) -> str`` that returns the model's raw
    text. This keeps the supervisor testable with a mock and lets it
    sit on top of either the local MLX brain or the cloud router.
    """

    def __init__(
        self,
        *,
        tool_registry: Any,
        action_executor: Any,
        config: SupervisorConfig | None = None,
        llm_call: Any = None,
    ) -> None:
        self.registry = tool_registry
        self.action_executor = action_executor
        self.config = config or SupervisorConfig()
        self._decisions: dict[str, int] = {}
        # Optional default LLM call; if set, ``run(query)`` works
        # without an explicit ``llm_call=`` argument. The router
        # binds this to ``LocalBrainController.generate_async``
        # at boot so the supervisor can drive the same brain that
        # serves Layer 3 of the router.
        self._default_llm_call = llm_call

        # Lazy import keeps the module graph minimal at boot.
        from core.reasoning.parallel_plan_executor import (
            ParallelPlanExecutor,
            ParallelPlannerConfig,
        )

        self.planner = ParallelPlanExecutor(
            tool_registry=tool_registry,
            action_executor=action_executor,
            config=ParallelPlannerConfig(
                max_steps=self.config.max_plan_steps,
                max_concurrency=self.config.max_concurrency,
                per_step_timeout_s=self.config.per_step_timeout_s,
            ),
        )
        # Backwards-compat alias for any existing callers.
        self._planner = self.planner

    def set_llm_call(self, llm_call: Any) -> None:
        """Bind (or rebind) the default async LLM callable used by
        :py:meth:`run`. Useful when the supervisor is constructed
        before ``LocalBrainController`` exists at boot.
        """
        self._default_llm_call = llm_call

    # ── Public surface ────────────────────────────────────────

    def triage(self, query: str) -> Triage:
        """Decide whether the query needs a multi-step plan at all.

        Cheap regex-only heuristic. Wrong calls here only cost a wasted
        LLM round-trip; they never break correctness because the
        supervisor's ``run`` method falls back to ``used_plan=False``
        on any planning error.
        """
        if not self.config.enabled:
            return self._record(
                Triage(False, "supervisor_disabled", 0.0),
            )
        clean = (query or "").strip()
        if not clean:
            return self._record(Triage(False, "empty_query", 0.0))

        # Sprint Ω.2 — order matters. The previous flow checked
        # ``_LIGHT_INTENT_RE`` first, which matched "open spotify" at the
        # *start* of "open spotify and play focus playlist" and silently
        # short-circuited compound queries straight to a single-shot
        # dispatch. We now compute the multi-step signals first and only
        # honour the light shortcut when the query is genuinely a
        # single-verb micro-intent.
        verb_hits = len(_ACTION_VERBS.findall(clean))

        if _MULTI_STEP_RE.search(clean):
            return self._record(Triage(True, "multi_step_hint", 0.9))

        if verb_hits >= 2:
            return self._record(
                Triage(True, "multi_action_verbs", 0.7),
            )

        # Compound conjunction without an explicit "then" still implies
        # >1 step ("open spotify and play X", "summarise this and email it").
        # We only trip on conjunctions that come *after* a recognised
        # action verb so chit-chat ("hi and how are you") stays light.
        if verb_hits >= 1 and _COMPOUND_AFTER_VERB_RE.search(clean):
            return self._record(
                Triage(True, "compound_after_verb", 0.75),
            )

        if _LIGHT_INTENT_RE.search(clean):
            return self._record(Triage(False, "light_intent", 0.95))

        # Long, prose-y queries are the planner's bread and butter.
        if len(clean) >= self.config.min_query_chars_for_plan and "?" not in clean:
            if verb_hits >= 1:
                return self._record(
                    Triage(True, "long_action_query", 0.6),
                )

        return self._record(Triage(False, "single_intent", 0.55))

    async def run(
        self,
        query: str,
        *,
        llm_call: Any = None,
        context_block: str = "",
        context: dict[str, Any] | None = None,
    ) -> SupervisorResult:
        """Plan + execute a Boss query.

        ``llm_call`` is an async callable ``(prompt, system) -> str``.
        When omitted, the supervisor uses the LLM bound at construction
        (or via :py:meth:`set_llm_call`). On any failure the result
        has ``used_plan=False`` and the caller should fall back to its
        single-shot path.

        ``context`` is an optional dict of routing context; if provided
        and ``context_block`` is empty, a short text rendition is
        derived from it. This keeps the API ergonomic for callers
        (router, MCP handlers) that already track context as a dict.
        """
        import time as _time

        t0 = _time.perf_counter()
        result = SupervisorResult()

        if llm_call is None:
            llm_call = self._default_llm_call
        if llm_call is None:
            result.error = "no_llm_call_bound"
            result.elapsed_ms = (_time.perf_counter() - t0) * 1000
            return result

        if not context_block and context:
            context_block = self._render_context_block(context)

        triage = self.triage(query)
        if not triage.needs_plan:
            result.summary = ""
            result.elapsed_ms = (_time.perf_counter() - t0) * 1000
            return result

        try:
            plan_text = await llm_call(
                self._build_user_prompt(query, context_block),
                self._build_system_prompt(),
            )
        except Exception as exc:
            logger.warning("Supervisor LLM call failed: %s", exc)
            result.error = f"llm_call_failed: {exc}"
            result.elapsed_ms = (_time.perf_counter() - t0) * 1000
            return result

        plan_blob = (plan_text or "").strip()
        if not plan_blob:
            result.error = "empty_plan"
            result.elapsed_ms = (_time.perf_counter() - t0) * 1000
            return result

        # Quick parse-only pass so we can downgrade gracefully when the
        # model returned prose, an invalid step name, etc. The executor
        # would also catch this, but doing it here lets us label the
        # outcome cleanly so the caller knows to fall back.
        steps, _rationale, _stop, parse_error = self._planner.parse_plan(plan_blob)
        if parse_error:
            logger.info("Supervisor plan parse failed: %s", parse_error)
            result.error = f"parse_failed: {parse_error}"
            result.plan_blob = plan_blob
            result.elapsed_ms = (_time.perf_counter() - t0) * 1000
            return result

        if not steps:
            result.error = "no_steps"
            result.plan_blob = plan_blob
            result.elapsed_ms = (_time.perf_counter() - t0) * 1000
            return result

        # Execute.
        plan_result = await self._planner.execute(plan_blob)
        result.used_plan = True
        result.plan_blob = plan_blob
        result.plan_result = plan_result
        result.summary = plan_result.speak_summary()

        # Reviewer pass for partial-failure cases.
        if (
            self.config.review_on_partial_failure
            and plan_result.any_failed
            and not plan_result.all_succeeded
        ):
            try:
                follow_up = await llm_call(
                    self._build_review_prompt(query, plan_result),
                    self._build_system_prompt(),
                )
                if follow_up and follow_up.strip():
                    result.follow_up = follow_up.strip()[:400]
            except Exception:
                logger.debug(
                    "Supervisor reviewer call failed", exc_info=True,
                )

        result.elapsed_ms = (_time.perf_counter() - t0) * 1000
        return result

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "decisions": dict(self._decisions),
        }

    # ── Internals ─────────────────────────────────────────────

    def _record(self, triage: Triage) -> Triage:
        key = ("plan:" if triage.needs_plan else "skip:") + triage.reason
        self._decisions[key] = self._decisions.get(key, 0) + 1
        return triage

    def _build_system_prompt(self) -> str:
        catalogue = self._planner.planner_prompt_block()
        return f"{PERSONA_BLOCK}\n\n{catalogue}"

    def _build_user_prompt(self, query: str, context_block: str) -> str:
        ctx = context_block.strip()
        head = f"Boss said: {query.strip()}"
        if ctx:
            return f"{head}\n\nLive context:\n{ctx}\n\nReturn ONLY the JSON plan."
        return f"{head}\n\nReturn ONLY the JSON plan."

    @staticmethod
    def _render_context_block(context: dict[str, Any]) -> str:
        """Compress a routing-context dict into a short text block.

        We only include keys the model can actually act on so the
        prompt stays small even when callers throw the kitchen sink
        at us. Unknown keys are silently dropped.
        """
        if not context:
            return ""
        keys = (
            "screen_summary", "active_app", "now",
            "last_user_turn", "last_atom_turn", "mood",
            "recent_actions", "memory_recall",
        )
        lines: list[str] = []
        for k in keys:
            v = context.get(k)
            if not v:
                continue
            lines.append(f"- {k}: {str(v)[:240]}")
        return "\n".join(lines)

    @staticmethod
    def _build_review_prompt(query: str, plan_result: Any) -> str:
        outcomes = "\n".join(s.short() for s in getattr(plan_result, "steps", []))
        return (
            f"Boss originally asked: {query.strip()}\n\n"
            f"Plan outcomes:\n{outcomes}\n\n"
            "[REVIEWER] In one sentence to Boss: did this fully solve "
            "the request? If not, suggest the single next step."
        )


__all__ = [
    "AgentSupervisor",
    "SupervisorConfig",
    "SupervisorResult",
    "Triage",
    "PERSONA_BLOCK",
]
