"""Focused tests for Sprint C4/C5/C6 observability additions."""

from __future__ import annotations

import asyncio
import importlib
import sys
import time
import unittest


class _FakeBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit_fast(self, name: str, **data) -> None:
        self.events.append((name, data))


class ErrorRateMonitorTests(unittest.TestCase):
    def setUp(self) -> None:
        # Force fresh singleton for every test so counters don't leak.
        sys.modules.pop("core.observability.error_rate_monitor", None)
        self.mod = importlib.import_module("core.observability.error_rate_monitor")
        self.mod._instance = None  # type: ignore[attr-defined]

    def test_records_and_reports_rate(self) -> None:
        monitor = self.mod.get_error_rate_monitor(
            window_s=5.0, threshold=3, poll_interval_s=0.2,
        )
        for i in range(4):
            self.mod.record_error("bus.stt_error", f"boom {i}")
        self.assertEqual(monitor.rate(), 4)
        self.assertEqual(monitor.top_sources()[0][0], "bus.stt_error")

    def test_emits_burst_event(self) -> None:
        bus = _FakeBus()

        async def _run() -> None:
            monitor = self.mod.get_error_rate_monitor(
                window_s=5.0, threshold=2, poll_interval_s=0.1,
            )
            monitor.start(bus)
            for _ in range(5):
                self.mod.record_error("bus.voice", "err")
            await asyncio.sleep(0.4)
            monitor.stop()
            await asyncio.sleep(0.05)

        asyncio.run(_run())
        burst_events = [e for e in bus.events if e[0] == "atom_error_burst_detected"]
        self.assertTrue(burst_events, "expected at least one burst event")
        self.assertGreaterEqual(burst_events[0][1]["rate"], 2)

    def test_rolling_window_expires(self) -> None:
        monitor = self.mod.get_error_rate_monitor(
            window_s=0.2, threshold=100, poll_interval_s=1.0,
        )
        for _ in range(3):
            self.mod.record_error("bus.fast")
        self.assertEqual(monitor.rate(), 3)
        time.sleep(0.35)
        self.assertEqual(monitor.rate(), 0)


class ThermalClampTests(unittest.TestCase):
    def test_effective_inference_respects_clamp(self) -> None:
        from brain.mlx_llm import MLXBrain

        class _Mgr:
            def effective_params(self):
                return {
                    "profile": "steady",
                    "max_tokens": 800,
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "timeout_seconds": 60,
                    "extra_stop_sequences": [],
                }

        brain = MLXBrain.__new__(MLXBrain)
        brain._brain_mode_mgr = _Mgr()
        brain._normalize_role = lambda r=None: "primary"
        brain._path_for_role = lambda r: "/fake"
        brain._max_tokens = 800
        brain._temperature = 0.3
        brain._top_p = 0.9
        brain._timeout = 60
        brain._thermal_clamp_ratio = 1.0
        brain._thermal_clamp_reason = ""

        MLXBrain.set_thermal_clamp(brain, 0.6, reason="hot")
        eff = MLXBrain._effective_inference(brain)
        self.assertEqual(eff["max_tokens_base"], 800)
        self.assertEqual(eff["max_tokens"], 480)
        self.assertAlmostEqual(eff["thermal_clamp_ratio"], 0.6)

        MLXBrain.set_thermal_clamp(brain, 0.05)
        self.assertGreaterEqual(brain._thermal_clamp_ratio, 0.25)

        MLXBrain.set_thermal_clamp(brain, 1.0, reason="cool")
        eff = MLXBrain._effective_inference(brain)
        self.assertEqual(eff["max_tokens"], 800)


class HealthSnapshotTests(unittest.TestCase):
    def test_health_snapshot_aggregates_statuses(self) -> None:
        from core.observability.health_snapshot import HealthSnapshotBuilder

        class _STT:
            listening = True
            _running = True
            _restart_count = 0

        class _TTS:
            _speaking = False
            _deadman_task = None
            _speak_budget_s = 0.0

        class _Brain:
            available = True
            _current_runtime_mode = "SMART"

            class _LLM:
                _thermal_clamp_ratio = 1.0

            _llm = _LLM()

            def prompt_cache_stats(self):
                return {"hits": 2, "misses": 1}

        class _Emb:
            _model_loaded = True
            _cache_size = 5
            _warm_enabled = True

        class _Sem:
            def get_stats(self):
                return {"hits": 1}

        class _State:
            class _C:
                name = "IDLE"

            current = _C()

        builder = HealthSnapshotBuilder(
            bus=_FakeBus(),
            state=_State(),
            stt=_STT(),
            tts=_TTS(),
            local_brain=_Brain(),
            embedding_engine=_Emb(),
            semantic_cache=_Sem(),
        )
        payload = builder.build()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["status"], "ok")
        subs = payload["subsystems"]
        self.assertEqual(subs["stt"]["status"], "ok")
        self.assertEqual(subs["brain"]["status"], "ok")
        self.assertEqual(subs["state_machine"]["current"], "IDLE")


if __name__ == "__main__":
    unittest.main()
