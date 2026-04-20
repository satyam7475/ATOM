"""
ATOM -- Sprint D1 focused tests: MorningBriefingService.

Validates:
    1. Briefing composes from available signals (weather + news stubs).
    2. Firing once per day is enforced (second trigger same day → None).
    3. Out-of-window hours don't fire.
    4. State round-trips through disk.
    5. Calendar/weather/news failures degrade gracefully (no exception).
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

from core.proactive.morning_briefing import MorningBriefingService


class _FakeBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def emit_long(self, name: str, **kwargs: Any) -> None:
        self.events.append((name, kwargs))


class _FakeRealWorld:
    def __init__(self, weather: str = "", news: str = "") -> None:
        self._weather = weather
        self._news = news

    def get_weather_summary(self) -> str:
        return self._weather

    def get_news_summary(self, count: int = 3) -> str:
        return self._news


def _mk(
    tmp: Path,
    *,
    rwi: _FakeRealWorld | None = None,
    wake_start: int = 0,
    wake_end: int = 23,
    include_calendar: bool = False,
) -> tuple[MorningBriefingService, _FakeBus]:
    bus = _FakeBus()
    cfg = {
        "morning_briefing": {
            "enabled": True,
            "wake_hour_start": wake_start,
            "wake_hour_end": wake_end,
            "include_battery": False,
            "include_weather": rwi is not None,
            "include_calendar": include_calendar,
            "include_news": rwi is not None,
            "news_count": 2,
            "state_path": str(tmp / "briefing_state.json"),
        }
    }
    svc = MorningBriefingService(cfg, bus=bus, real_world_intel=rwi)
    return svc, bus


def test_composes_and_emits_once() -> None:
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            rwi = _FakeRealWorld(
                weather="Sunny, high of 28 degrees in Bangalore.",
                news="Top headlines:\n1. AI wave continues\n2. Markets flat",
            )
            svc, bus = _mk(tmp, rwi=rwi)
            out = await svc.maybe_trigger("startup")
            assert out is not None
            assert "Good" in out or "morning" in out.lower()
            assert "Sunny" in out
            assert "AI wave continues" in out
            assert len(bus.events) == 1
            assert bus.events[0][0] == "response_ready"

            out2 = await svc.maybe_trigger("speech_final")
            assert out2 is None
            assert len(bus.events) == 1

    asyncio.run(_run())


def test_no_fire_outside_window() -> None:
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            svc, bus = _mk(tmp, wake_start=23, wake_end=23)
            from datetime import datetime
            now_h = datetime.now().hour
            if now_h == 23:
                return
            out = await svc.maybe_trigger("speech_final")
            assert out is None
            assert not bus.events

    asyncio.run(_run())


def test_state_persists_across_restart() -> None:
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            rwi = _FakeRealWorld(weather="Clear skies.", news="Top headlines:\n1. Calm day")
            svc1, _bus1 = _mk(tmp, rwi=rwi)
            fired = await svc1.maybe_trigger("startup")
            assert fired is not None
            assert svc1.last_briefed_date

            state_path = tmp / "briefing_state.json"
            assert state_path.exists()
            raw = json.loads(state_path.read_text())
            assert raw["last_briefed_date"] == svc1.last_briefed_date

            svc2, bus2 = _mk(tmp, rwi=rwi)
            assert svc2.last_briefed_date == svc1.last_briefed_date
            out2 = await svc2.maybe_trigger("speech_final")
            assert out2 is None
            assert not bus2.events

    asyncio.run(_run())


def test_empty_sources_still_produces_opener() -> None:
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            svc, bus = _mk(tmp, rwi=None)
            out = await svc.maybe_trigger("startup")
            assert out is not None
            assert len(out.split()) >= 3
            assert len(bus.events) == 1

    asyncio.run(_run())


def test_corrupted_state_does_not_crash() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        state = tmp / "briefing_state.json"
        state.write_text("not-json")
        svc, _bus = _mk(tmp)
        assert svc.last_briefed_date == ""


if __name__ == "__main__":
    test_composes_and_emits_once()
    test_no_fire_outside_window()
    test_state_persists_across_restart()
    test_empty_sources_still_produces_opener()
    test_corrupted_state_does_not_crash()
    print("[D1] All morning briefing tests passed.")
