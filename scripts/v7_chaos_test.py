#!/usr/bin/env python3
"""
ATOM V7 chaos harness: RecoveryManager + EventRing without live workers.

Simulates worker crash notifications and ring replay. For integration tests
with real subprocesses, combine with ServiceWatchdog on the target machine.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    from core.recovery_manager import RecoveryManager

    cfg = {"v7_gpu": {"event_replay_max": 16}}
    r = RecoveryManager(bus=None, config=cfg)
    r.record_event("test_event", {"x": 1})
    r.record_event("test_event", {"x": 2})

    seen: list[tuple[str, dict]] = []

    def h(name: str, payload: dict) -> None:
        seen.append((name, payload))

    n = r.replay_recent(h)
    print(f"chaos: replayed={n} events={seen}")
    r.on_worker_crash("fake_worker", exit_code=-1)
    print("chaos: worker_crash handler ok")


if __name__ == "__main__":
    main()
