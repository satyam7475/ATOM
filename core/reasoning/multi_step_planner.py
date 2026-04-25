"""ATOM Sprint N2 -- multi-step planner.

The existing :mod:`core.reasoning.tool_parser` + :class:`ActionExecutor`
handle *single* tool calls beautifully. Real Friday-class behaviour
needs **chained** plans:

    Boss: "Open Spotify, play my Focus playlist, then dim the screen
          to 30 percent."

This module produces a structured plan in JSON, validates each step
against the live :class:`ToolRegistry`, and runs them in order while
giving the caller back a per-step execution trace.

Design constraints (kept on purpose):

* No new LLM dependency -- the planner is **purely deterministic**:
  it accepts an already-generated plan blob (string, list, or
  :class:`ToolCall` sequence) and validates / executes it. This keeps
  it cheap to test and lets the caller plug in any LLM.
* The planner is **idempotent** with respect to validation: it never
  mutates the registry or executor state.
* "Stop on first failure" by default; configurable to "continue on
  failure" for best-effort cleanup chains.
* Always returns a :class:`PlanResult` with rich metadata so the
  reflective loop can decide whether to retry or escalate.

The expected planner JSON shape (free-form fields are ignored):

    {
      "plan": [
        {"name": "music_play_specific",
         "arguments": {"query": "Focus", "kind": "playlist"}},
        {"name": "set_volume", "arguments": {"percent": 30}}
      ],
      "rationale": "Boss wanted music + lower volume",
      "stop_on_error": true
    }

A bare list of tool-call dicts is also accepted.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("atom.reasoning.planner")


# ── data shapes ────────────────────────────────────────────────────────


@dataclass(slots=True)
class PlannedStep:
    """One validated step in a multi-step plan."""

    index: int
    tool_name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class StepOutcome:
    """The outcome of executing one :class:`PlannedStep`."""

    step: PlannedStep
    success: bool
    output: str = ""
    error: str = ""
    elapsed_ms: float = 0.0
    blocked: bool = False
    block_reason: str = ""
    skipped: bool = False
    skip_reason: str = ""

    def short(self) -> str:
        head = f"[{self.step.index + 1}] {self.step.tool_name}"
        if self.skipped:
            return f"{head} -- SKIPPED ({self.skip_reason})"
        if self.blocked:
            return f"{head} -- BLOCKED ({self.block_reason})"
        if self.success:
            tail = self.output.strip().splitlines()
            tail_str = tail[0] if tail else "ok"
            return f"{head} -- ok ({self.elapsed_ms:.0f}ms): {tail_str[:160]}"
        return f"{head} -- FAILED ({self.elapsed_ms:.0f}ms): {self.error[:160]}"


@dataclass(slots=True)
class PlanResult:
    """Full result of running a multi-step plan."""

    steps: list[StepOutcome] = field(default_factory=list)
    rationale: str = ""
    stop_on_error: bool = True
    total_elapsed_ms: float = 0.0
    error: str = ""

    @property
    def all_succeeded(self) -> bool:
        return bool(self.steps) and all(
            s.success for s in self.steps if not s.skipped
        )

    @property
    def any_failed(self) -> bool:
        return any(
            (not s.success and not s.skipped) for s in self.steps
        )

    @property
    def n_completed(self) -> int:
        return sum(1 for s in self.steps if s.success and not s.skipped)

    def speak_summary(self) -> str:
        """One-sentence summary that's safe to hand to TTS."""
        if not self.steps:
            return "Plan was empty, Boss."
        if self.all_succeeded:
            verbs = ", ".join(_verb_for(s.step.tool_name) for s in self.steps)
            return f"Done, Boss -- {verbs}."
        ok = sum(1 for s in self.steps if s.success and not s.skipped)
        bad = next(
            (s for s in self.steps if not s.success and not s.skipped),
            None,
        )
        if bad is None:
            return f"Plan finished partially, Boss ({ok} of {len(self.steps)})."
        return (
            f"Got {ok} of {len(self.steps)} done, Boss -- "
            f"{bad.step.tool_name} couldn't run "
            f"({bad.error[:80] if bad.error else 'unknown'})."
        )


def _verb_for(tool_name: str) -> str:
    return tool_name.replace("_", " ")


# ── planner ────────────────────────────────────────────────────────────


@dataclass(slots=True)
class PlannerConfig:
    max_steps: int = 6
    stop_on_error_default: bool = True
    per_step_timeout_s: float = 12.0
    allow_unknown_tools: bool = False


class MultiStepPlanner:
    """Validate + execute a structured tool-call plan.

    The class is intentionally *pure* (no LLM client) so it is easy to
    unit test and can be wrapped by either the local brain or the cloud
    brain without changes.

    Pass a ``ToolRegistry`` (for validation) and any object with an
    ``execute_async(tool_call) -> ActionResult`` method (typically the
    :class:`ActionExecutor`).
    """

    def __init__(
        self,
        tool_registry: Any,
        action_executor: Any,
        *,
        config: PlannerConfig | None = None,
    ) -> None:
        self.registry = tool_registry
        self.executor = action_executor
        self.config = config or PlannerConfig()

    # ── parse / validate ──────────────────────────────────────────

    def parse_plan(
        self, plan_blob: str | dict | list,
    ) -> tuple[list[PlannedStep], str, bool, str]:
        """Parse and validate the planner output.

        Returns a tuple of (steps, rationale, stop_on_error, parse_error).
        ``parse_error`` is non-empty when the blob couldn't be parsed at
        all; the caller should surface it as a PlanResult.error.
        """
        rationale = ""
        stop_on_error = self.config.stop_on_error_default
        steps_raw: list[Any] = []

        if isinstance(plan_blob, str):
            try:
                obj = self._extract_json(plan_blob)
            except ValueError as exc:
                return [], "", stop_on_error, str(exc)
        else:
            obj = plan_blob

        if isinstance(obj, list):
            steps_raw = obj
        elif isinstance(obj, dict):
            steps_raw = obj.get("plan") or obj.get("steps") or []
            rationale = str(obj.get("rationale") or obj.get("reason") or "")
            if "stop_on_error" in obj:
                stop_on_error = bool(obj.get("stop_on_error"))
        else:
            return [], "", stop_on_error, "Plan must be a list or object."

        if not isinstance(steps_raw, list) or not steps_raw:
            return [], rationale, stop_on_error, "Plan list is empty."

        steps: list[PlannedStep] = []
        for i, raw in enumerate(steps_raw[: self.config.max_steps]):
            if not isinstance(raw, dict):
                return [], rationale, stop_on_error, (
                    f"Step #{i + 1} is not an object."
                )
            name = str(raw.get("name") or raw.get("tool") or "").strip()
            if not name:
                return [], rationale, stop_on_error, (
                    f"Step #{i + 1} is missing 'name'."
                )
            args_blob = (
                raw.get("arguments")
                or raw.get("args")
                or raw.get("parameters")
                or {}
            )
            if not isinstance(args_blob, dict):
                return [], rationale, stop_on_error, (
                    f"Step #{i + 1} arguments must be an object."
                )

            if not self.config.allow_unknown_tools:
                tool = self.registry.get(name) if self.registry else None
                if tool is None:
                    return [], rationale, stop_on_error, (
                        f"Step #{i + 1}: tool '{name}' is not registered."
                    )

            steps.append(
                PlannedStep(index=i, tool_name=name, arguments=dict(args_blob)),
            )

        return steps, rationale, stop_on_error, ""

    @staticmethod
    def _extract_json(blob: str) -> Any:
        text = blob.strip()
        if not text:
            raise ValueError("Plan blob is empty.")

        # Allow a fenced markdown block.
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fenced is not None:
            text = fenced.group(1).strip()

        # Trim leading prose like "Sure, here is the plan:" by finding
        # the first { or [.
        first_obj = text.find("{")
        first_arr = text.find("[")
        candidates = [c for c in (first_obj, first_arr) if c >= 0]
        if not candidates:
            raise ValueError("No JSON object/array found in plan blob.")
        start = min(candidates)
        text = text[start:]

        # Try to parse incrementally to tolerate trailing prose.
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            # Best-effort truncation to last balanced } or ]
            for end in range(len(text), 0, -1):
                snippet = text[:end]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    continue
            raise ValueError(f"Invalid JSON plan: {exc}")

    # ── execute ───────────────────────────────────────────────────

    async def execute(
        self,
        plan_blob: str | dict | list,
        *,
        on_step_start: Any = None,
        on_step_end: Any = None,
    ) -> PlanResult:
        """Validate then run every step of the plan."""
        t_total = time.perf_counter()
        steps, rationale, stop_on_error, parse_error = self.parse_plan(plan_blob)
        result = PlanResult(rationale=rationale, stop_on_error=stop_on_error)
        if parse_error:
            result.error = parse_error
            return result

        # Dynamic import to keep module dependency graph minimal.
        from core.reasoning.tool_parser import ToolCall

        for step in steps:
            outcome = StepOutcome(step=step, success=False)

            if on_step_start is not None:
                try:
                    if asyncio.iscoroutinefunction(on_step_start):
                        await on_step_start(step)
                    else:
                        on_step_start(step)
                except Exception:
                    logger.debug("on_step_start callback raised", exc_info=True)

            tc = ToolCall(name=step.tool_name, arguments=dict(step.arguments))
            t0 = time.perf_counter()
            try:
                action_result = await asyncio.wait_for(
                    self.executor.execute_async(tc),
                    timeout=self.config.per_step_timeout_s,
                )
            except asyncio.TimeoutError:
                outcome.error = (
                    f"Step timed out after "
                    f"{self.config.per_step_timeout_s:.0f}s"
                )
                outcome.elapsed_ms = (time.perf_counter() - t0) * 1000
            except Exception as exc:
                outcome.error = f"Executor raised: {exc}"
                outcome.elapsed_ms = (time.perf_counter() - t0) * 1000
            else:
                outcome.elapsed_ms = float(
                    getattr(action_result, "elapsed_ms", 0.0)
                    or (time.perf_counter() - t0) * 1000,
                )
                outcome.success = bool(getattr(action_result, "success", False))
                outcome.output = str(getattr(action_result, "output", "") or "")
                outcome.error = str(getattr(action_result, "error", "") or "")
                outcome.blocked = bool(
                    getattr(action_result, "blocked", False),
                )
                outcome.block_reason = str(
                    getattr(action_result, "block_reason", "") or "",
                )

            result.steps.append(outcome)

            if on_step_end is not None:
                try:
                    if asyncio.iscoroutinefunction(on_step_end):
                        await on_step_end(outcome)
                    else:
                        on_step_end(outcome)
                except Exception:
                    logger.debug("on_step_end callback raised", exc_info=True)

            if not outcome.success and stop_on_error:
                # Mark remaining steps as skipped for trace clarity.
                for remaining in steps[step.index + 1:]:
                    result.steps.append(
                        StepOutcome(
                            step=remaining, success=False,
                            skipped=True,
                            skip_reason="stop_on_error",
                        ),
                    )
                break

        result.total_elapsed_ms = (time.perf_counter() - t_total) * 1000
        return result

    # ── helper: render a planner-friendly tool catalogue ──────────

    def planner_prompt_block(self) -> str:
        """A focused tool catalogue for the planner LLM.

        The heavy `to_prompt_description` text is fine for single-call
        prompts but burns tokens fast in a planner. Here we ship a
        compact name/desc/arg signature per tool.
        """
        if self.registry is None:
            return ""
        lines: list[str] = ["Available tools (call any subset):"]
        for tool in self.registry.get_all():
            args = ", ".join(
                f"{p.name}{'?' if not p.required else ''}:{p.type}"
                for p in tool.parameters
            )
            sig = f"  - {tool.name}({args}) :: {tool.description}"
            lines.append(sig[:240])
        lines.append("")
        lines.append(
            "Respond ONLY with JSON: "
            '{"plan":[{"name":"...", "arguments":{}} ...], '
            '"rationale":"...", "stop_on_error":true}'
        )
        return "\n".join(lines)
