"""
ATOM -- Phase G2 regression suite for the Presence Sampler.

The sampler must:
  * Run capture + detect on a worker thread and emit ``presence.snapshot``.
  * Skip while ATOM is speaking / listening / thinking.
  * Degrade gracefully when the camera or Vision binding is missing.
  * Back off after repeated capture failures instead of busy-looping.
  * Infer reasonable quality labels from face count + capture latency.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from core.perception.presence_sampler import (
    PresenceSampler,
    PresenceSnapshot,
)


# ── stub helpers ────────────────────────────────────────────────────


@dataclass
class _FakeCapture:
    ok: bool = True
    saved_path: str = "/tmp/atom_presence.jpg"
    capture_ms: float = 80.0
    error: str = ""


@dataclass
class _FakeVision:
    ok: bool = True
    detection_ms: float = 12.0
    faces: int = 1
    face_boxes: list[tuple[float, float, float, float]] = field(
        default_factory=lambda: [(0.1, 0.2, 0.3, 0.4)],
    )
    error: str = ""


def _capture_module(result: _FakeCapture) -> Any:
    class _Mod:
        @staticmethod
        def capture_jpeg(_camera: Any, *, out_path: Any, timeout_s: float) -> _FakeCapture:
            return result

        @staticmethod
        def discover_cameras() -> list[str]:
            return ["camera0"]

        @staticmethod
        def choose_preferred(_cams: list[str], *, preferred: str) -> str:
            return "camera0"

    return _Mod


def _vision_module(result: _FakeVision) -> Any:
    class _Mod:
        @staticmethod
        def detect(_path: str, *, detect_faces: bool, detect_barcodes: bool) -> _FakeVision:
            return result

    return _Mod


class _FakeBus:
    def __init__(self) -> None:
        self.emitted_long: list[tuple[str, dict]] = []

    def on(self, *_a: Any, **_kw: Any) -> None: ...
    def off(self, *_a: Any, **_kw: Any) -> None: ...
    def emit_fast(self, event: str, **payload: Any) -> None: ...
    def emit_long(self, event: str, **payload: Any) -> None:
        self.emitted_long.append((event, payload))


# ── one-shot path ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sample_once_emits_present_snapshot() -> None:
    bus = _FakeBus()
    sampler = PresenceSampler(
        bus,
        capture_fn=_capture_module(_FakeCapture(ok=True, capture_ms=80)),
        detect_fn=_vision_module(_FakeVision(faces=1)),
    )
    snap = await sampler.sample_once()
    assert snap.present is True
    assert snap.face_count == 1
    assert snap.quality == "good"
    assert len(bus.emitted_long) == 1
    event, payload = bus.emitted_long[0]
    assert event == "presence.snapshot"
    assert payload["present"] is True
    assert payload["face_count"] == 1


@pytest.mark.asyncio
async def test_sample_once_with_no_faces_marks_no_face() -> None:
    bus = _FakeBus()
    sampler = PresenceSampler(
        bus,
        capture_fn=_capture_module(_FakeCapture(ok=True, capture_ms=80)),
        detect_fn=_vision_module(_FakeVision(faces=0, face_boxes=[])),
    )
    snap = await sampler.sample_once()
    assert snap.present is False
    assert snap.face_count == 0
    assert snap.quality == "no_face"


@pytest.mark.asyncio
async def test_sample_once_marks_low_light_on_long_capture() -> None:
    bus = _FakeBus()
    sampler = PresenceSampler(
        bus,
        capture_fn=_capture_module(_FakeCapture(ok=True, capture_ms=2200)),
        detect_fn=_vision_module(_FakeVision(faces=1)),
    )
    snap = await sampler.sample_once()
    assert snap.quality == "low_light"


@pytest.mark.asyncio
async def test_sample_once_handles_capture_failure() -> None:
    bus = _FakeBus()
    sampler = PresenceSampler(
        bus,
        capture_fn=_capture_module(
            _FakeCapture(ok=False, error="camera busy", capture_ms=10),
        ),
        detect_fn=_vision_module(_FakeVision()),
    )
    snap = await sampler.sample_once()
    assert snap.present is False
    assert snap.quality == "no_camera"
    assert "camera busy" in snap.error
    payload = bus.emitted_long[-1][1]
    assert payload["present"] is False
    assert payload["error"] == "camera busy"


@pytest.mark.asyncio
async def test_sample_once_degrades_when_modules_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = _FakeBus()
    sampler = PresenceSampler(
        bus,
        capture_fn=None,
        detect_fn=None,
    )

    def _fake_lazy(_self):
        return None, None

    monkeypatch.setattr(PresenceSampler, "_lazy_imports", _fake_lazy)
    snap = await sampler.sample_once()
    assert snap.quality == "no_camera"
    assert snap.present is False
    assert "pyobjc" in snap.error.lower()


# ── periodic loop ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_periodic_loop_emits_then_stops() -> None:
    bus = _FakeBus()
    sampler = PresenceSampler(
        bus,
        interval_s=0.05,
        min_interval_s=0.01,
        capture_fn=_capture_module(_FakeCapture(ok=True)),
        detect_fn=_vision_module(_FakeVision(faces=1)),
    )
    sampler.start()
    await asyncio.sleep(0.16)
    await sampler.stop()
    assert len(bus.emitted_long) >= 2
    assert all(evt == "presence.snapshot" for evt, _ in bus.emitted_long)
    assert sampler.metrics["samples"] >= 2


@pytest.mark.asyncio
async def test_loop_skips_when_state_speaking() -> None:
    bus = _FakeBus()
    sampler = PresenceSampler(
        bus,
        interval_s=0.05,
        min_interval_s=0.01,
        capture_fn=_capture_module(_FakeCapture(ok=True)),
        detect_fn=_vision_module(_FakeVision(faces=1)),
        state_provider=lambda: "speaking",
    )
    sampler.start()
    await asyncio.sleep(0.12)
    await sampler.stop()
    assert bus.emitted_long == []
    assert sampler.metrics["skips"] >= 2
    assert sampler.metrics["samples"] == 0


@pytest.mark.asyncio
async def test_loop_skips_when_busy_provider_true() -> None:
    bus = _FakeBus()
    sampler = PresenceSampler(
        bus,
        interval_s=0.05,
        min_interval_s=0.01,
        capture_fn=_capture_module(_FakeCapture(ok=True)),
        detect_fn=_vision_module(_FakeVision(faces=1)),
        busy_provider=lambda: True,
    )
    sampler.start()
    await asyncio.sleep(0.12)
    await sampler.stop()
    assert bus.emitted_long == []
    assert sampler.metrics["samples"] == 0


@pytest.mark.asyncio
async def test_loop_recovers_after_capture_errors() -> None:
    bus = _FakeBus()
    counter = {"n": 0}

    class _FlakyMod:
        @staticmethod
        def capture_jpeg(*_a: Any, **_kw: Any) -> _FakeCapture:
            counter["n"] += 1
            if counter["n"] <= 3:
                return _FakeCapture(ok=False, error="busy", capture_ms=5)
            return _FakeCapture(ok=True, capture_ms=80)

        @staticmethod
        def discover_cameras() -> list[str]:
            return ["camera0"]

        @staticmethod
        def choose_preferred(_cams: list[str], *, preferred: str) -> str:
            return "camera0"

    sampler = PresenceSampler(
        bus,
        interval_s=0.04,
        min_interval_s=0.01,
        capture_fn=_FlakyMod,
        detect_fn=_vision_module(_FakeVision(faces=1)),
    )
    sampler.start()
    await asyncio.sleep(0.4)
    await sampler.stop()
    # We expect at least one good emission after errors recover.
    good = [p for _, p in bus.emitted_long if p["present"] is True]
    assert good, "sampler never recovered from capture errors"
    assert sampler.metrics["errors"] >= 3


@pytest.mark.asyncio
async def test_metrics_contain_last_snapshot_summary() -> None:
    bus = _FakeBus()
    sampler = PresenceSampler(
        bus,
        capture_fn=_capture_module(_FakeCapture(ok=True)),
        detect_fn=_vision_module(_FakeVision(faces=2)),
    )
    await sampler.sample_once()
    metrics = sampler.metrics
    assert metrics["samples"] == 1
    assert metrics["last_snapshot"]["face_count"] == 2
    assert metrics["last_snapshot"]["present"] is True


# ── snapshot dataclass ─────────────────────────────────────────────


def test_presence_snapshot_as_dict_round_trips() -> None:
    snap = PresenceSnapshot(
        ts=1.0, present=True, face_count=1, quality="good",
        face_boxes=[(0.1, 0.2, 0.3, 0.4)],
        capture_ms=50.0, detection_ms=10.0,
    )
    d = snap.as_dict()
    assert d["face_count"] == 1
    assert d["face_boxes"] == [(0.1, 0.2, 0.3, 0.4)]
    assert d["quality"] == "good"
