"""
Stress harness helpers — return metrics for external runs on target hardware.

Does not require full ATOM main(); can use PriorityScheduler-only paths.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import Counter
from typing import Any, Callable

from core.priority_scheduler import PRIORITY_LLM, PRIORITY_VOICE, PriorityScheduler


async def simulate_heavy_queries(
    n: int,
    *,
    mode_fn: Callable[[int], str] | None = None,
) -> dict[str, Any]:
    """Enqueue many jobs; report avg latency, error rate, mode distribution."""
    return await _run_simulation(
        n,
        mode_fn=mode_fn or (lambda _i: random.choice(["FAST", "SMART", "DEEP", "SECURE"])),
    )


async def simulate_rapid_queries(
    n: int,
    *,
    mode_fn: Callable[[int], str] | None = None,
) -> dict[str, Any]:
    """Minimal delay between submissions."""
    return await _run_simulation(
        n,
        mode_fn=mode_fn or (lambda _i: "SMART"),
    )


async def simulate_idle_then_burst(
    n: int,
    *,
    idle_ms: float = 500.0,
    mode_fn: Callable[[int], str] | None = None,
) -> dict[str, Any]:
    """Sleep then burst enqueue."""
    await asyncio.sleep(idle_ms / 1000.0)
    return await _run_simulation(
        n,
        mode_fn=mode_fn or (lambda _i: "SMART"),
        idle_before_ms=idle_ms,
    )


async def _run_simulation(
    n: int,
    *,
    mode_fn: Callable[[int], str],
    idle_before_ms: float = 0.0,
) -> dict[str, Any]:
    sched = PriorityScheduler(metrics=None)
    sched.start()
    latencies: list[float] = []
    errors = 0
    modes: list[str] = []
    t0 = time.perf_counter()

    def make_job(i: int) -> Callable[[], Any]:
        mode = mode_fn(i)
        modes.append(mode)

        async def _run() -> None:
            nonlocal errors
            t1 = time.perf_counter()
            try:
                await asyncio.sleep(0)
            except Exception:
                errors += 1
            latencies.append((time.perf_counter() - t1) * 1000)

        return _run

    for i in range(n):
        prio = PRIORITY_VOICE if i % 50 == 0 else PRIORITY_LLM
        sched.submit(prio, f"stress_{i}", make_job(i))

    for _ in range(4000):
        await asyncio.sleep(0.005)
        if sched.queue_depth == 0:
            break
    await sched.shutdown()

    wall_s = time.perf_counter() - t0
    avg_lat = sum(latencies) / max(1, len(latencies))
    dist = dict(Counter(modes))
    return {
        "n": n,
        "avg_latency_ms": avg_lat,
        "error_rate": errors / max(1, n),
        "mode_distribution": dist,
        "wall_s": wall_s,
        "idle_before_ms": idle_before_ms,
        "diagnostics": sched.get_diagnostics(),
    }
