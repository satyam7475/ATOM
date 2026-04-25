"""ATOM Sprint Ω5 -- DAG-based parallel plan executor.

Why this exists
---------------
:class:`core.reasoning.multi_step_planner.MultiStepPlanner` validates and
runs a sequence of tool calls strictly in order. That keeps things safe
for chains that have implicit ordering (``open_app`` -> ``play_music``)
but it leaves a *lot* of latency on the table for plans whose steps are
genuinely independent ("check the weather, look up my calendar, and
draft a status email") -- those should fan out concurrently and join.

This module adds DAG semantics on top of the same plan dialect:

    {
      "plan": [
        {"id": "weather", "name": "weather_get",
         "arguments": {"city": "Delhi"}},
        {"id": "cal",     "name": "calendar_today"},
        {"id": "summary", "name": "compose_text",
         "arguments": {"prompt": "Summarise the morning"},
         "depends_on": ["weather", "cal"]}
      ]
    }

* Every step gets an ``id`` (auto-derived from index when missing).
* ``depends_on`` lists the ids that must succeed first.
* Steps with no remaining dependencies run via ``asyncio.gather`` --
  bounded by a ``max_concurrency`` knob so we never thrash MLX/Spotify
  with too many parallel calls.
* ``stop_on_error`` (true by default) cancels not-yet-started steps as
  soon as any in-flight step fails. In-flight steps are allowed to
  finish so we don't half-execute a side effect.
* Cycle detection runs at parse time so a malformed plan fails *before*
  any tool runs.

The executor reuses :class:`core.reasoning.action_executor.ActionExecutor`
exactly as ``MultiStepPlanner`` does, so the same security gating,
schema validation, and confirmation policy apply to every parallel
step. There is no second code path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("atom.reasoning.parallel_planner")


# ── Data shapes ─────────────────────────────────────────────────────


@dataclass(slots=True)
class DAGStep:
    """One node in the parallel plan DAG."""

    id: str
    tool_name: str
    arguments: dict[str, Any]
    depends_on: tuple[str, ...] = ()


@dataclass(slots=True)
class DAGStepOutcome:
    """The outcome of a single DAG step."""

    step: DAGStep
    success: bool = False
    output: str = ""
    error: str = ""
    elapsed_ms: float = 0.0
    blocked: bool = False
    block_reason: str = ""
    skipped: bool = False
    skip_reason: str = ""

    def short(self) -> str:
        head = f"[{self.step.id}] {self.step.tool_name}"
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
class DAGPlanResult:
    """Full result of running a DAG plan."""

    steps: list[DAGStepOutcome] = field(default_factory=list)
    rationale: str = ""
    stop_on_error: bool = True
    total_elapsed_ms: float = 0.0
    parallel_waves: int = 0
    max_concurrent_in_wave: int = 0
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
        if not self.steps:
            return "Plan was empty, Boss."
        if self.all_succeeded:
            verbs = ", ".join(
                s.step.tool_name.replace("_", " ") for s in self.steps
            )
            return f"Done in parallel, Boss -- {verbs}."
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


# ── Config ──────────────────────────────────────────────────────────


@dataclass(slots=True)
class ParallelPlannerConfig:
    """Tuning knobs for :class:`ParallelPlanExecutor`."""

    max_steps: int = 10
    max_concurrency: int = 3
    per_step_timeout_s: float = 12.0
    stop_on_error_default: bool = True
    allow_unknown_tools: bool = False


# ── Executor ────────────────────────────────────────────────────────


class ParallelPlanExecutor:
    """DAG-aware version of :class:`MultiStepPlanner`.

    Public surface mirrors :class:`MultiStepPlanner` so callers can swap
    one for the other based on whether the LLM returned a flat list (use
    sequential) or a DAG with ``depends_on`` edges (use parallel).
    """

    def __init__(
        self,
        tool_registry: Any,
        action_executor: Any,
        *,
        config: ParallelPlannerConfig | None = None,
    ) -> None:
        self.registry = tool_registry
        self.executor = action_executor
        self.config = config or ParallelPlannerConfig()

    # ── Parse / validate ──────────────────────────────────────────

    def parse_plan(
        self, plan_blob: str | dict | list,
    ) -> tuple[list[DAGStep], str, bool, str]:
        """Parse and validate a DAG plan.

        Returns (steps, rationale, stop_on_error, parse_error). On any
        validation failure ``steps`` is empty and ``parse_error`` carries
        the human-readable cause for the caller to surface.
        """
        rationale = ""
        stop_on_error = self.config.stop_on_error_default

        if isinstance(plan_blob, str):
            try:
                obj = self._extract_json(plan_blob)
            except ValueError as exc:
                return [], "", stop_on_error, str(exc)
        else:
            obj = plan_blob

        if isinstance(obj, list):
            steps_raw: list[Any] = obj
        elif isinstance(obj, dict):
            steps_raw = obj.get("plan") or obj.get("steps") or []
            rationale = str(obj.get("rationale") or obj.get("reason") or "")
            if "stop_on_error" in obj:
                stop_on_error = bool(obj.get("stop_on_error"))
        else:
            return [], "", stop_on_error, "Plan must be a list or object."

        if not isinstance(steps_raw, list) or not steps_raw:
            return [], rationale, stop_on_error, "Plan list is empty."

        if len(steps_raw) > self.config.max_steps:
            return [], rationale, stop_on_error, (
                f"Plan has {len(steps_raw)} steps; "
                f"max is {self.config.max_steps}."
            )

        steps: list[DAGStep] = []
        seen_ids: set[str] = set()
        for i, raw in enumerate(steps_raw):
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

            step_id = str(raw.get("id") or f"s{i + 1}").strip() or f"s{i + 1}"
            if step_id in seen_ids:
                return [], rationale, stop_on_error, (
                    f"Step #{i + 1} duplicates id '{step_id}'."
                )
            seen_ids.add(step_id)

            dep_raw = raw.get("depends_on") or raw.get("after") or ()
            if isinstance(dep_raw, (list, tuple)):
                deps = tuple(str(d).strip() for d in dep_raw if str(d).strip())
            elif isinstance(dep_raw, str):
                deps = (dep_raw.strip(),) if dep_raw.strip() else ()
            else:
                return [], rationale, stop_on_error, (
                    f"Step '{step_id}' depends_on must be a list of ids."
                )

            if not self.config.allow_unknown_tools:
                tool = self.registry.get(name) if self.registry else None
                if tool is None:
                    return [], rationale, stop_on_error, (
                        f"Step '{step_id}': tool '{name}' is not registered."
                    )

            steps.append(
                DAGStep(
                    id=step_id,
                    tool_name=name,
                    arguments=dict(args_blob),
                    depends_on=deps,
                ),
            )

        # Validate dependency references and detect cycles.
        ids = {s.id for s in steps}
        for s in steps:
            for d in s.depends_on:
                if d not in ids:
                    return [], rationale, stop_on_error, (
                        f"Step '{s.id}' depends on unknown id '{d}'."
                    )
                if d == s.id:
                    return [], rationale, stop_on_error, (
                        f"Step '{s.id}' depends on itself."
                    )

        cycle = self._find_cycle(steps)
        if cycle:
            return [], rationale, stop_on_error, (
                f"Cycle detected in plan: {' -> '.join(cycle)}"
            )

        return steps, rationale, stop_on_error, ""

    @staticmethod
    def _extract_json(blob: str) -> Any:
        text = blob.strip()
        if not text:
            raise ValueError("Plan blob is empty.")
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fenced is not None:
            text = fenced.group(1).strip()
        first_obj = text.find("{")
        first_arr = text.find("[")
        candidates = [c for c in (first_obj, first_arr) if c >= 0]
        if not candidates:
            raise ValueError("No JSON object/array found in plan blob.")
        text = text[min(candidates):]
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            for end in range(len(text), 0, -1):
                snippet = text[:end]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    continue
            raise ValueError(f"Invalid JSON plan: {exc}") from exc

    @staticmethod
    def _find_cycle(steps: list[DAGStep]) -> list[str]:
        """Return one cycle path if present, else ``[]``.

        Standard 3-color DFS. We return the path for diagnostics so the
        operator can see exactly which edge to break.
        """
        graph = {s.id: list(s.depends_on) for s in steps}
        WHITE, GRAY, BLACK = 0, 1, 2
        color = dict.fromkeys(graph, WHITE)
        parent: dict[str, str | None] = dict.fromkeys(graph, None)

        def visit(node: str) -> list[str]:
            color[node] = GRAY
            for nbr in graph.get(node, ()):
                if color.get(nbr) == GRAY:
                    cycle = [nbr]
                    cur: str | None = node
                    while cur is not None and cur != nbr:
                        cycle.append(cur)
                        cur = parent.get(cur)
                    cycle.append(nbr)
                    return list(reversed(cycle))
                if color.get(nbr) == WHITE:
                    parent[nbr] = node
                    found = visit(nbr)
                    if found:
                        return found
            color[node] = BLACK
            return []

        for n in graph:
            if color[n] == WHITE:
                found = visit(n)
                if found:
                    return found
        return []

    # ── Execute ───────────────────────────────────────────────────

    async def execute(
        self,
        plan_blob: str | dict | list,
        *,
        on_step_start: Any = None,
        on_step_end: Any = None,
    ) -> DAGPlanResult:
        """Validate then run every step of the DAG plan."""
        t_total = time.perf_counter()
        steps, rationale, stop_on_error, parse_error = self.parse_plan(plan_blob)
        result = DAGPlanResult(
            rationale=rationale, stop_on_error=stop_on_error,
        )
        if parse_error:
            result.error = parse_error
            return result

        from core.reasoning.tool_parser import ToolCall

        outcomes: dict[str, DAGStepOutcome] = {
            s.id: DAGStepOutcome(step=s, success=False) for s in steps
        }
        completed: set[str] = set()
        failed: set[str] = set()
        skipped: set[str] = set()
        sem = asyncio.Semaphore(max(1, self.config.max_concurrency))
        abort = asyncio.Event()

        async def _run_step(step: DAGStep) -> DAGStepOutcome:
            outcome = outcomes[step.id]
            if abort.is_set():
                outcome.skipped = True
                outcome.skip_reason = "stop_on_error"
                return outcome
            async with sem:
                if on_step_start is not None:
                    try:
                        cb = on_step_start
                        if asyncio.iscoroutinefunction(cb):
                            await cb(step)
                        else:
                            cb(step)
                    except Exception:
                        logger.debug(
                            "on_step_start callback raised", exc_info=True,
                        )
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
                    outcome.success = bool(
                        getattr(action_result, "success", False),
                    )
                    outcome.output = str(
                        getattr(action_result, "output", "") or "",
                    )
                    outcome.error = str(
                        getattr(action_result, "error", "") or "",
                    )
                    outcome.blocked = bool(
                        getattr(action_result, "blocked", False),
                    )
                    outcome.block_reason = str(
                        getattr(action_result, "block_reason", "") or "",
                    )
                if on_step_end is not None:
                    try:
                        cb = on_step_end
                        if asyncio.iscoroutinefunction(cb):
                            await cb(outcome)
                        else:
                            cb(outcome)
                    except Exception:
                        logger.debug(
                            "on_step_end callback raised", exc_info=True,
                        )
                return outcome

        # Wave-by-wave scheduler.
        wave_index = 0
        while True:
            ready: list[DAGStep] = []
            for s in steps:
                if (
                    s.id in completed or s.id in failed or s.id in skipped
                ):
                    continue
                if all(d in completed for d in s.depends_on):
                    if abort.is_set():
                        outcomes[s.id].skipped = True
                        outcomes[s.id].skip_reason = "stop_on_error"
                        skipped.add(s.id)
                        continue
                    ready.append(s)

            if not ready:
                # Either we're done, or every remaining step is gated by
                # a failed/skipped dependency. Mark blocked-out steps as
                # skipped with a precise reason and break.
                pending = [
                    s for s in steps
                    if s.id not in completed
                    and s.id not in failed
                    and s.id not in skipped
                ]
                if not pending:
                    break
                for s in pending:
                    bad = [
                        d for d in s.depends_on
                        if d in failed or d in skipped
                    ]
                    o = outcomes[s.id]
                    o.skipped = True
                    o.skip_reason = (
                        f"dependency_failed: {','.join(bad)}"
                        if bad else "unreachable"
                    )
                    skipped.add(s.id)
                break

            wave_index += 1
            result.parallel_waves = wave_index
            result.max_concurrent_in_wave = max(
                result.max_concurrent_in_wave, len(ready),
            )

            wave_results = await asyncio.gather(
                *[_run_step(s) for s in ready],
                return_exceptions=False,
            )
            for o in wave_results:
                if o.success:
                    completed.add(o.step.id)
                elif o.skipped:
                    skipped.add(o.step.id)
                else:
                    failed.add(o.step.id)
                    if stop_on_error:
                        abort.set()

        # Preserve original declaration order in the report.
        result.steps = [outcomes[s.id] for s in steps]
        result.total_elapsed_ms = (time.perf_counter() - t_total) * 1000
        return result

    # ── Helpers ──────────────────────────────────────────────────

    def planner_prompt_block(self) -> str:
        """Compact tool catalogue + DAG schema for the planner LLM."""
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
            "Respond ONLY with JSON. To run steps in parallel, give each "
            "an id and list its prerequisites in 'depends_on':"
        )
        lines.append(
            '{"plan":['
            '{"id":"a","name":"weather_get","arguments":{"city":"Delhi"}},'
            '{"id":"b","name":"calendar_today"},'
            '{"id":"c","name":"compose_text",'
            '"arguments":{"prompt":"Brief"},'
            '"depends_on":["a","b"]}'
            '], "rationale":"...", "stop_on_error":true}'
        )
        return "\n".join(lines)


__all__ = [
    "DAGStep",
    "DAGStepOutcome",
    "DAGPlanResult",
    "ParallelPlannerConfig",
    "ParallelPlanExecutor",
]
