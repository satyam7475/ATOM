#!/usr/bin/env python3
"""
Phase 6.4 — Autonomy integration smoke (proactive + goals + dream).

Verifies in one process (no 30-minute soak):
  - Proactive M5 context triggers produce structured insights
  - Goal steps + tool completion advances progress
  - Dream cycle runs and returns a result dict with Phase 6.3 fields

Run from repo root:
  python3 scripts/phase6_autonomy_smoke.py

Exit 0 on success, 1 on failure.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_config() -> dict:
    p = ROOT / "config" / "settings.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _check_proactive() -> None:
    from core.cognitive.proactive_engine import ProactiveIntelligenceEngine
    from unittest.mock import MagicMock

    bus = MagicMock()
    bus.on = MagicMock()
    bus.emit_long = MagicMock()
    cfg = _load_config()
    eng = ProactiveIntelligenceEngine(bus=bus, config=cfg)
    eng._last_scan = {
        "ram_percent": 88,
        "battery": {"percent": 12, "plugged": False},
    }
    out = eng._scan_m5_context_triggers(time.time())
    cats = {x.get("category") for x in out}
    assert "system_battery" in cats, cats
    assert "system_memory" in cats, cats


def _check_goals() -> None:
    from unittest.mock import MagicMock

    from core.behavior_tracker import BehaviorTracker
    from core.cognitive.goal_engine import GoalEngine, _finalize_step_record
    from core.cognitive.second_brain import SecondBrain
    from core.memory_engine import MemoryEngine

    bus = MagicMock()
    bus.on = MagicMock()
    bus.emit_fast = MagicMock()
    cfg = _load_config()
    mem = MemoryEngine(cfg)
    beh = BehaviorTracker(cfg)
    brain = SecondBrain(mem, beh, cfg)
    goal = GoalEngine(bus, brain, cfg)
    g = goal.create_goal("Smoke integration goal")
    assert "error" not in g, g
    gid = g["id"]
    goal_obj = goal._find_by_id(gid)
    assert goal_obj is not None
    goal_obj["steps"] = [
        {
            "id": "smoke1",
            "title": "Remember the smoke test marker",
            "status": "pending",
            "minutes_logged": 0,
            "created_at": "",
            "updated_at": "",
        },
    ]
    _finalize_step_record(goal_obj["steps"][0])
    assert goal_obj["steps"][0].get("suggested_tool") == "remember"
    ok = goal.apply_tool_completion(
        "remember",
        {"fact": "Remember the smoke test marker"},
    )
    assert ok is True
    assert goal_obj["steps"][0]["status"] == "completed"


async def _check_dream() -> None:
    from unittest.mock import MagicMock

    from core.async_event_bus import AsyncEventBus
    from core.behavior_tracker import BehaviorTracker
    from core.cognitive.dream_engine import DreamEngine
    from core.cognitive.second_brain import SecondBrain
    from core.memory_engine import MemoryEngine

    bus = AsyncEventBus()
    bus.start()
    cfg = _load_config()
    mem = MemoryEngine(cfg)
    beh = BehaviorTracker(cfg)
    brain = SecondBrain(mem, beh, cfg)
    dream = DreamEngine(bus, cfg)
    dream.wire(second_brain=brain)
    for i in range(8):
        dream.record_interaction(
            f"integration query {i} about python asyncio patterns",
            f"response {i} explaining tasks and gather",
            intent="chat",
            emotion="neutral",
        )
    result = await dream.dream()
    assert result.get("status") != "nothing_to_dream"
    assert "interactions_processed" in result
    assert "pattern_summary" in result
    assert "brain_pruned" in result
    assert "embedding_warmups" in result
    await bus.stop()


def main() -> int:
    print("Phase 6.4 smoke: proactive …")
    _check_proactive()
    print("Phase 6.4 smoke: goals …")
    _check_goals()
    print("Phase 6.4 smoke: dream …")
    asyncio.run(_check_dream())
    print("Phase 6.4 autonomy smoke: OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as e:
        print("FAIL:", e, file=sys.stderr)
        raise SystemExit(1)
