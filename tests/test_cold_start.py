"""Focused cold-start bootstrap tests."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakeBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit_fast(self, event: str, **data) -> None:
        self.events.append((event, data))


class FakeState:
    class _State:
        value = "idle"

    def __init__(self) -> None:
        self.current = self._State()


class FakeLocalBrain:
    def __init__(self) -> None:
        self.calls: list[tuple[str | None, bool]] = []

    @property
    def available(self) -> bool:
        return True

    async def warm_up(
        self,
        *,
        model_role: str | None = None,
        load_all: bool = False,
    ) -> bool:
        self.calls.append((model_role, load_all))
        return True


class FakeMemory:
    def __init__(self) -> None:
        self.embedding_warmed = False

    async def warm_up_embeddings(self) -> bool:
        self.embedding_warmed = True
        return True

    def get_top_commands(self, limit: int = 10) -> list[str]:
        cmds = ["open chrome", "what time is it", "mute volume", "open chrome"]
        return cmds[:limit]


class FakeSystemMonitor:
    def get_system_state(self) -> dict:
        return {
            "cpu_percent": 8.5,
            "ram_percent": 41.0,
            "foreground_window_title": "Cursor",
            "ts": time.time(),
        }


def test_cold_start_warm_up_and_restore() -> None:
    from core.boot.cold_start import ColdStartOptimizer
    from core.command_cache import get_command_cache
    from core.conversation_memory import ConversationMemory

    class FakeIntentResult:
        def __init__(self, intent: str, response: str = "") -> None:
            self.intent = intent
            self.response = response

    class FakeIntentEngine:
        def classify(self, text: str) -> FakeIntentResult:
            if "open chrome" in text:
                return FakeIntentResult("open_app", response="Opening Chrome, Boss.")
            if "mute" in text:
                return FakeIntentResult("mute", response="Muted, Boss.")
            if "time" in text:
                return FakeIntentResult("time", response="It is testing time, Boss.")
            return FakeIntentResult("cpu", response="CPU is calm, Boss.")

    async def _run() -> None:
        cmd_cache = get_command_cache()
        cmd_cache.clear()

        bus = FakeBus()
        state = FakeState()
        memory = FakeMemory()
        local_brain = FakeLocalBrain()
        conv = ConversationMemory()

        with tempfile.TemporaryDirectory() as td:
            snapshot_path = Path(td) / "cold_start.json"
            snapshot = {
                "saved_at": time.time(),
                "conversation_pairs": [
                    ["open chrome", "Opening Chrome, Boss."],
                    ["check cpu usage", "CPU is at 10 percent, Boss."],
                ],
                "system_state": {
                    "cpu_percent": 12.0,
                    "ram_percent": 33.0,
                    "foreground_window_title": "Cursor",
                    "ts": time.time(),
                },
            }
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

            cold_start = ColdStartOptimizer(
                config={},
                bus=bus,
                state_manager=state,
                local_brain=local_brain,
                memory_store=memory,
                conversation_memory=conv,
                intent_engine=FakeIntentEngine(),
                system_monitor=FakeSystemMonitor(),
                snapshot_path=snapshot_path,
            )

            report = await cold_start.warm_up()
            assert report.fast_model_ready is True
            assert report.embeddings_ready is True
            assert report.restored_turns == 2
            assert report.cached_commands == 2
            assert conv.turn_count == 2
            assert local_brain.calls == [("fast", False)]
            assert memory.embedding_warmed is True
            assert cmd_cache.get("open chrome") is not None
            assert cmd_cache.get("mute volume") is not None
            assert cmd_cache.get("what time is it") is None
            assert cmd_cache.get("info:time") is None

            emitted = await cold_start.emit_restored_context()
            assert emitted is True
            assert len(bus.events) == 1
            event, payload = bus.events[0]
            assert event == "context_snapshot"
            assert payload["cpu"] == 12.0
            assert payload["ram"] == 33.0
            assert payload["active_app"] == "Cursor"

        cmd_cache.clear()

    asyncio.run(_run())
    print("  PASS: ColdStart warm-up restores session, cache, and context")


def test_cold_start_persist_snapshot() -> None:
    from core.boot.cold_start import ColdStartOptimizer
    from core.conversation_memory import ConversationMemory

    conv = ConversationMemory()
    conv.record("check cpu usage", "cpu", "CPU is at 9 percent, Boss.")
    conv.record("open cursor", "open_app", "Opening Cursor, Boss.")

    with tempfile.TemporaryDirectory() as td:
        snapshot_path = Path(td) / "cold_start_saved.json"
        cold_start = ColdStartOptimizer(
            config={},
            bus=FakeBus(),
            state_manager=FakeState(),
            local_brain=FakeLocalBrain(),
            memory_store=FakeMemory(),
            conversation_memory=conv,
            intent_engine=object(),
            system_monitor=FakeSystemMonitor(),
            snapshot_path=snapshot_path,
        )

        assert cold_start.persist_snapshot() is True
        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
        assert len(data["conversation_pairs"]) == 2
        assert data["conversation_pairs"][0][0] == "check cpu usage"
        assert data["system_state"]["foreground_window_title"] == "Cursor"

    print("  PASS: ColdStart persists snapshot for next boot")


if __name__ == "__main__":
    test_cold_start_warm_up_and_restore()
    test_cold_start_persist_snapshot()
    print("\ntest_cold_start: ALL PASSED")
