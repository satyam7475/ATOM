"""Phase 6.4 cognitive autonomy integration (importable tests).

Run: python3 -m tests.test_phase6_integration
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _cfg() -> dict:
    return json.loads((_ROOT / "config" / "settings.json").read_text(encoding="utf-8"))


def test_proactive_m5_scan() -> None:
    from unittest.mock import MagicMock

    from core.cognitive.proactive_engine import ProactiveIntelligenceEngine

    bus = MagicMock()
    bus.on = MagicMock()
    eng = ProactiveIntelligenceEngine(bus=bus, config=_cfg())
    eng._last_scan = {"ram_percent": 90, "battery": {"percent": 10, "plugged": False}}
    import time as _t

    out = eng._scan_m5_context_triggers(_t.time())
    assert any(x.get("category") == "system_battery" for x in out)


def test_goal_tool_then_complete() -> None:
    from unittest.mock import MagicMock

    from core.behavior_tracker import BehaviorTracker
    from core.cognitive.goal_engine import GoalEngine, _finalize_step_record
    from core.cognitive.second_brain import SecondBrain
    from core.memory_engine import MemoryEngine

    bus = MagicMock()
    bus.on = MagicMock()
    bus.emit_fast = MagicMock()
    geng = GoalEngine(bus, SecondBrain(MemoryEngine(_cfg()), BehaviorTracker(_cfg()), _cfg()), _cfg())
    g = geng.create_goal("Integration test goal")
    gid = g["id"]
    goal = geng._find_by_id(gid)
    goal["steps"] = [
        {
            "id": "t1",
            "title": "Open Notes app for review",
            "status": "pending",
            "minutes_logged": 0,
            "created_at": "",
            "updated_at": "",
        },
    ]
    _finalize_step_record(goal["steps"][0])
    assert goal["steps"][0].get("suggested_tool") == "open_app"
    assert geng.apply_tool_completion("open_app", {"name": "Notes"}) is True


async def _dream_once() -> dict:
    from core.async_event_bus import AsyncEventBus
    from core.behavior_tracker import BehaviorTracker
    from core.cognitive.dream_engine import DreamEngine
    from core.cognitive.second_brain import SecondBrain
    from core.memory_engine import MemoryEngine

    bus = AsyncEventBus()
    bus.start()
    cfg = _cfg()
    brain = SecondBrain(MemoryEngine(cfg), BehaviorTracker(cfg), cfg)
    d = DreamEngine(bus, cfg)
    d.wire(second_brain=brain)
    for i in range(6):
        d.record_interaction("hello world " * 3 + str(i), "ack " * 5, intent="chat")
    r = await d.dream()
    await bus.stop()
    return r


def test_dream_cycle_shape() -> None:
    r = asyncio.run(_dream_once())
    assert r.get("interactions_processed", 0) >= 6
    assert "connections" in r


def main() -> None:
    test_proactive_m5_scan()
    test_goal_tool_then_complete()
    test_dream_cycle_shape()
    print("phase6 integration tests passed")


if __name__ == "__main__":
    main()
