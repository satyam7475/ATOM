#!/usr/bin/env python3
"""
ATOM V7 stress harness: enqueue N synthetic priority-scheduler jobs (no ATOM main).

Run on target hardware: ``python ATOM/scripts/v7_stress_test.py --n 1000``
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.metrics import MetricsCollector
from core.priority_scheduler import PRIORITY_LLM, PRIORITY_VOICE, PriorityScheduler


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000, help="number of synthetic jobs")
    args = p.parse_args()

    m = MetricsCollector()
    sched = PriorityScheduler(metrics=m)
    sched.start()
    t0 = time.perf_counter()

    done = 0

    def make_coro_factory() -> Any:
        async def _run() -> None:
            nonlocal done
            await asyncio.sleep(0)
            done += 1
        return _run

    for i in range(args.n):
        prio = PRIORITY_VOICE if i % 50 == 0 else PRIORITY_LLM
        sched.submit(prio, f"stress_{i}", make_coro_factory())

    for _ in range(2000):
        await asyncio.sleep(0.01)
        if sched.queue_depth == 0:
            break
    await sched.shutdown()

    dt = time.perf_counter() - t0
    print(f"stress: n={args.n} completed={done} wall_s={dt:.2f} diag={sched.get_diagnostics()}")


if __name__ == "__main__":
    asyncio.run(main())
