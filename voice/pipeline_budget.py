"""
ATOM -- Voice Pipeline Latency Budget & Metrics.

Enforces hard latency budgets per pipeline stage and logs
per-command timing breakdowns for debugging. This is the
difference between "it feels slow sometimes" and knowing
exactly which stage is slow and why.

Budget stages:
  wake_detect:     50ms max
  stt_partial:    100ms max
  intent:          50ms max
  ack:            100ms max
  full_response:  800ms max

Metrics are emitted per-command via the event bus and logged
in a structured format for easy parsing.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("atom.pipeline_budget")


class StageBudget:
    """Hard budget for a single pipeline stage."""

    __slots__ = ("name", "budget_ms", "start_time", "end_time", "overbudget")

    def __init__(self, name: str, budget_ms: float) -> None:
        self.name = name
        self.budget_ms = budget_ms
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.overbudget: bool = False

    def start(self) -> None:
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        self.end_time = time.perf_counter()
        elapsed = (self.end_time - self.start_time) * 1000
        self.overbudget = elapsed > self.budget_ms
        return elapsed

    @property
    def elapsed_ms(self) -> float:
        if self.end_time > 0:
            return (self.end_time - self.start_time) * 1000
        if self.start_time > 0:
            return (time.perf_counter() - self.start_time) * 1000
        return 0.0


_STAGE_BUDGETS: dict[str, float] = {
    "wake_detect": 50.0,
    "stt_partial": 100.0,
    "stt_final": 200.0,
    "intent": 50.0,
    "ack": 100.0,
    "action": 500.0,
    "full_response": 800.0,
    "tts_start": 150.0,
}


class PipelineBudgetTracker:
    """Track and enforce latency budgets across one command's lifecycle."""

    def __init__(self, trace_id: str = "", command: str = "") -> None:
        self._trace_id = trace_id
        self._command = command[:80]
        self._stages: dict[str, StageBudget] = {}
        self._t0 = time.perf_counter()
        self._violations: list[str] = []

    def start_stage(self, name: str) -> StageBudget:
        budget_ms = _STAGE_BUDGETS.get(name, 500.0)
        stage = StageBudget(name, budget_ms)
        stage.start()
        self._stages[name] = stage
        return stage

    def end_stage(self, name: str) -> float:
        stage = self._stages.get(name)
        if stage is None:
            return 0.0
        elapsed = stage.stop()
        if stage.overbudget:
            self._violations.append(
                f"{name}={elapsed:.0f}ms (budget={stage.budget_ms:.0f}ms)"
            )
            logger.warning(
                "BUDGET_VIOLATION [%s] %s: %.0fms > %.0fms budget",
                self._trace_id, name, elapsed, stage.budget_ms,
            )
        return elapsed

    @property
    def total_elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000

    @property
    def has_violations(self) -> bool:
        return len(self._violations) > 0

    def should_fallback_fast(self) -> bool:
        """Return True if we should abandon normal path and use fast fallback."""
        intent_stage = self._stages.get("intent")
        if intent_stage and intent_stage.elapsed_ms > 100.0:
            return True
        if self.total_elapsed_ms > 1000.0:
            return True
        return False

    def summary(self) -> dict[str, Any]:
        """Generate a structured timing summary for this command."""
        stage_timings: dict[str, float] = {}
        for name, stage in self._stages.items():
            stage_timings[name] = round(stage.elapsed_ms, 1)

        return {
            "trace_id": self._trace_id,
            "command": self._command,
            "total_ms": round(self.total_elapsed_ms, 1),
            "stages": stage_timings,
            "violations": self._violations,
        }

    def log_summary(self) -> None:
        """Log a one-line timing summary."""
        parts = []
        for name, stage in self._stages.items():
            ms = stage.elapsed_ms
            flag = "!" if stage.overbudget else ""
            parts.append(f"{name}={ms:.0f}ms{flag}")

        total = self.total_elapsed_ms
        violation_str = f" VIOLATIONS: {', '.join(self._violations)}" if self._violations else ""
        logger.info(
            "VOICE_PIPELINE [%s] '%s' | %s | total=%.0fms%s",
            self._trace_id,
            self._command,
            " | ".join(parts),
            total,
            violation_str,
        )


class VoicePipelineMetrics:
    """Aggregate metrics across all commands for dashboard/diagnostics."""

    def __init__(self) -> None:
        self._total_commands: int = 0
        self._total_violations: int = 0
        self._total_latency_ms: float = 0.0
        self._worst_latency_ms: float = 0.0
        self._best_latency_ms: float = float("inf")
        self._stage_totals: dict[str, float] = {}
        self._stage_counts: dict[str, int] = {}

    def record(self, tracker: PipelineBudgetTracker) -> None:
        summary = tracker.summary()
        total = summary["total_ms"]

        self._total_commands += 1
        self._total_latency_ms += total
        if summary["violations"]:
            self._total_violations += 1
        if total > self._worst_latency_ms:
            self._worst_latency_ms = total
        if total < self._best_latency_ms:
            self._best_latency_ms = total

        for stage_name, stage_ms in summary["stages"].items():
            self._stage_totals[stage_name] = self._stage_totals.get(stage_name, 0.0) + stage_ms
            self._stage_counts[stage_name] = self._stage_counts.get(stage_name, 0) + 1

    def get_diagnostics(self) -> dict[str, Any]:
        avg = self._total_latency_ms / max(1, self._total_commands)
        stage_avgs = {
            name: round(self._stage_totals[name] / max(1, self._stage_counts[name]), 1)
            for name in self._stage_totals
        }
        return {
            "total_commands": self._total_commands,
            "total_violations": self._total_violations,
            "avg_latency_ms": round(avg, 1),
            "best_latency_ms": round(self._best_latency_ms, 1) if self._best_latency_ms < float("inf") else 0,
            "worst_latency_ms": round(self._worst_latency_ms, 1),
            "stage_averages": stage_avgs,
        }


__all__ = ["PipelineBudgetTracker", "VoicePipelineMetrics", "StageBudget"]
