"""
ATOM V6.5 -- Validation engine (determinism, stress, chaos).

Produces structured stability and performance metrics for certification.
Uses injectable callables so tests can target MemoryGraph, planners, or mocks
without starting the full stack.

Output shape (example):
{
  "determinism_score": 0.98,
  "failure_rate": 0.02,
  "recovery_time_ms": 120.0,
  "stress": {...},
  "chaos": {...}
}
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger("atom.validation")

JsonDict = Dict[str, Any]


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def _hash_obj(obj: Any) -> str:
    return hashlib.sha256(_stable_json(obj).encode("utf-8")).hexdigest()


@dataclass
class DeterminismResult:
    runs: int
    plan_hashes: List[str]
    result_hashes: List[str]
    score_hashes: List[str]
    determinism_score: float
    identical_plans: bool
    identical_results: bool
    identical_scores: bool


def run_determinism_test(
    run_once: Callable[[], JsonDict],
    runs: int = 10,
) -> DeterminismResult:
    """
    Execute the same goal path *runs* times; compare hashed plan/result/score keys.

    Expects each return dict to optionally include keys:
    ``plan``, ``result``, ``score`` (any JSON-serializable values).
    """
    plan_hashes: List[str] = []
    result_hashes: List[str] = []
    score_hashes: List[str] = []

    for _ in range(max(1, runs)):
        out = run_once()
        plan_hashes.append(_hash_obj(out.get("plan")))
        result_hashes.append(_hash_obj(out.get("result")))
        score_hashes.append(_hash_obj(out.get("score")))

    ip = len(set(plan_hashes)) <= 1
    ir = len(set(result_hashes)) <= 1
    is_ = len(set(score_hashes)) <= 1
    # Partial credit if some layers vary (e.g. timestamps in result)
    bits = sum([ip, ir, is_])
    determinism_score = bits / 3.0

    return DeterminismResult(
        runs=runs,
        plan_hashes=plan_hashes,
        result_hashes=result_hashes,
        score_hashes=score_hashes,
        determinism_score=determinism_score,
        identical_plans=ip,
        identical_results=ir,
        identical_scores=is_,
    )


@dataclass
class StressResult:
    goals: int
    failures: int
    failure_rate: float
    wall_time_ms: float
    latency_mean_ms: float
    latency_max_ms: float
    latency_samples: int


async def run_stress_test(
    goal_coro: Callable[[int], Awaitable[JsonDict]],
    concurrent_goals: int = 100,
) -> StressResult:
    """Run *concurrent_goals* goals concurrently; collect latency and failures."""

    latencies: List[float] = []
    failures = 0
    t0 = time.perf_counter()

    async def one(i: int) -> None:
        nonlocal failures
        t1 = time.perf_counter()
        try:
            await goal_coro(i)
        except Exception:
            failures += 1
            logger.debug("stress goal failed", exc_info=True)
        finally:
            latencies.append((time.perf_counter() - t1) * 1000.0)

    await asyncio.gather(*(one(i) for i in range(max(1, concurrent_goals))))

    wall_ms = (time.perf_counter() - t0) * 1000.0
    n = len(latencies)
    mean = sum(latencies) / n if n else 0.0
    mx = max(latencies) if latencies else 0.0
    fr = failures / max(1, concurrent_goals)

    return StressResult(
        goals=concurrent_goals,
        failures=failures,
        failure_rate=fr,
        wall_time_ms=wall_ms,
        latency_mean_ms=mean,
        latency_max_ms=mx,
        latency_samples=n,
    )


@dataclass
class ChaosHooks:
    """Inject faults into cooperative doubles."""

    memory_kill: bool = False
    simulation_kill: bool = False
    response_delay_s: float = 0.0


@dataclass
class ChaosResult:
    completed: bool
    used_fallback: bool
    recovery_time_ms: float
    errors: List[str] = field(default_factory=list)


async def run_chaos_test(
    run_pipeline: Callable[[ChaosHooks], Awaitable[JsonDict]],
    hooks: Optional[ChaosHooks] = None,
) -> ChaosResult:
    """
    Simulate degraded dependencies; expect graceful fallback (no crash).

    *run_pipeline* receives ChaosHooks and should avoid raising when possible,
    returning e.g. {"status": "ok", "fallback": true}.
    """
    h = hooks or ChaosHooks(
        memory_kill=True,
        simulation_kill=True,
        response_delay_s=0.05,
    )
    errors: List[str] = []
    t0 = time.perf_counter()
    used_fallback = False
    completed = False
    try:
        out = await run_pipeline(h)
        completed = True
        used_fallback = bool(out.get("fallback")) if isinstance(out, dict) else False
    except Exception as e:
        errors.append(str(e))
    recovery_ms = (time.perf_counter() - t0) * 1000.0

    return ChaosResult(
        completed=completed,
        used_fallback=used_fallback,
        recovery_time_ms=recovery_ms,
        errors=errors,
    )


def build_certification_report(
    det: DeterminismResult,
    stress: StressResult,
    chaos: ChaosResult,
) -> JsonDict:
    """Merge sub-reports into a single JSON-serializable certification blob."""

    failure_rate = stress.failure_rate
    recovery_ms = chaos.recovery_time_ms

    return {
        "determinism_score": round(det.determinism_score, 4),
        "failure_rate": round(failure_rate, 4),
        "recovery_time_ms": round(recovery_ms, 3),
        "determinism": {
            "runs": det.runs,
            "identical_plans": det.identical_plans,
            "identical_results": det.identical_results,
            "identical_scores": det.identical_scores,
        },
        "stress": {
            "goals": stress.goals,
            "failures": stress.failures,
            "failure_rate": round(stress.failure_rate, 4),
            "wall_time_ms": round(stress.wall_time_ms, 3),
            "latency_mean_ms": round(stress.latency_mean_ms, 3),
            "latency_max_ms": round(stress.latency_max_ms, 3),
        },
        "chaos": {
            "completed": chaos.completed,
            "used_fallback": chaos.used_fallback,
            "recovery_time_ms": round(chaos.recovery_time_ms, 3),
            "errors": chaos.errors,
        },
    }


__all__ = [
    "DeterminismResult",
    "StressResult",
    "ChaosHooks",
    "ChaosResult",
    "run_determinism_test",
    "run_stress_test",
    "run_chaos_test",
    "build_certification_report",
]
