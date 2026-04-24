"""
ATOM -- Phase G3 regression suite for the Scene Context engine.

The engine must:
  * Caption only on a meaningful change in presence.
  * Honour the cooldown (default 5 minutes) between VLM calls.
  * Skip silently when no face is present, the captioner is missing,
    or the busy provider says we're in the middle of a turn.
  * Never raise on captioner failure; emit only when a caption was
    produced.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import pytest

from core.perception.scene_context import (
    SceneContext,
    SceneContextEngine,
)


# ── stubs ──────────────────────────────────────────────────────────


@dataclass
class _FakeCapture:
    ok: bool = True
    saved_path: str = "/tmp/atom_scene.jpg"
    capture_ms: float = 80.0
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


class _Captioner:
    def __init__(self, text: str = "User looking at a code editor.") -> None:
        self._text = text
        self.calls = 0

    def describe(self, _path: str) -> str:
        self.calls += 1
        return self._text


class _BrokenCaptioner:
    def __init__(self) -> None:
        self.calls = 0

    def describe(self, _path: str) -> str:
        self.calls += 1
        raise RuntimeError("vlm down")


class _FakeBus:
    def __init__(self) -> None:
        self.handlers: dict[str, list] = {}
        self.emitted_long: list[tuple[str, dict]] = []

    def on(self, event: str, handler: Any) -> None:
        self.handlers.setdefault(event, []).append(handler)

    def off(self, event: str, handler: Any) -> None:
        try:
            self.handlers[event].remove(handler)
        except (KeyError, ValueError):
            pass

    def emit_long(self, event: str, **payload: Any) -> None:
        self.emitted_long.append((event, payload))

    def emit_fast(self, *_a: Any, **_kw: Any) -> None: ...

    async def fire(self, event: str, **payload: Any) -> None:
        for h in list(self.handlers.get(event, ())):
            await h(**payload)


def _make_engine(
    bus: _FakeBus,
    *,
    captioner: Any | None = None,
    cooldown_s: float = 0.5,
    significance_min_seconds: float = 0.0,
    busy_provider: Any | None = None,
    capture_ok: bool = True,
) -> SceneContextEngine:
    captioner = captioner or _Captioner()
    engine = SceneContextEngine(
        bus, captioner,
        cooldown_s=cooldown_s,
        significance_min_seconds=significance_min_seconds,
        capture_fn=_capture_module(_FakeCapture(ok=capture_ok)),
        busy_provider=busy_provider,
    )
    engine.attach()
    return engine


# ── basic firing ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_first_presence_snapshot_triggers_caption() -> None:
    bus = _FakeBus()
    captioner = _Captioner("Boss is at the laptop, looks focused.")
    engine = _make_engine(bus, captioner=captioner)
    await bus.fire("presence.snapshot",
                   present=True, face_count=1, quality="good", ts=time.time())
    await asyncio.sleep(0.05)
    assert captioner.calls == 1
    assert len(bus.emitted_long) == 1
    event, payload = bus.emitted_long[0]
    assert event == "scene.context"
    assert payload["caption"] == "Boss is at the laptop, looks focused."
    assert payload["trigger"] in ("first", "presence_change", "stale")


@pytest.mark.asyncio
async def test_no_face_snapshot_skips_caption() -> None:
    bus = _FakeBus()
    captioner = _Captioner()
    engine = _make_engine(bus, captioner=captioner)
    await bus.fire("presence.snapshot",
                   present=False, face_count=0, quality="no_face", ts=time.time())
    await asyncio.sleep(0.05)
    assert captioner.calls == 0
    assert bus.emitted_long == []
    assert engine.metrics["skips_no_change"] >= 1


@pytest.mark.asyncio
async def test_repeated_identical_snapshots_skip_due_to_no_change() -> None:
    bus = _FakeBus()
    captioner = _Captioner()
    engine = _make_engine(bus, captioner=captioner, cooldown_s=300.0)
    base = {"present": True, "face_count": 1, "quality": "good"}
    await bus.fire("presence.snapshot", **base, ts=time.time())
    await asyncio.sleep(0.05)
    await bus.fire("presence.snapshot", **base, ts=time.time())
    await bus.fire("presence.snapshot", **base, ts=time.time())
    await asyncio.sleep(0.05)
    assert captioner.calls == 1
    assert engine.metrics["skips_no_change"] >= 1 \
        or engine.metrics["skips_cooldown"] >= 1


# ── cooldown ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cooldown_blocks_back_to_back_significant_changes() -> None:
    bus = _FakeBus()
    captioner = _Captioner()
    engine = _make_engine(bus, captioner=captioner, cooldown_s=300.0)
    # Two real changes within a 5-min cooldown -> only first caption.
    await bus.fire("presence.snapshot",
                   present=True, face_count=1, quality="good", ts=time.time())
    await asyncio.sleep(0.05)
    await bus.fire("presence.snapshot",
                   present=True, face_count=2, quality="good", ts=time.time())
    await asyncio.sleep(0.05)
    assert captioner.calls == 1
    assert engine.metrics["skips_cooldown"] >= 1


@pytest.mark.asyncio
async def test_cooldown_releases_after_elapsed_time() -> None:
    bus = _FakeBus()
    captioner = _Captioner()
    engine = _make_engine(
        bus, captioner=captioner,
        cooldown_s=0.05, significance_min_seconds=0.05,
    )
    await bus.fire("presence.snapshot",
                   present=True, face_count=1, quality="good", ts=time.time())
    await asyncio.sleep(0.1)
    await bus.fire("presence.snapshot",
                   present=True, face_count=2, quality="good", ts=time.time())
    await asyncio.sleep(0.1)
    assert captioner.calls >= 2
    assert engine.metrics["emits"] >= 2


# ── busy / capture failure ────────────────────────────────────────


@pytest.mark.asyncio
async def test_busy_provider_skips_caption() -> None:
    bus = _FakeBus()
    captioner = _Captioner()
    engine = _make_engine(
        bus, captioner=captioner, busy_provider=lambda: True,
    )
    await bus.fire("presence.snapshot",
                   present=True, face_count=1, quality="good", ts=time.time())
    await asyncio.sleep(0.05)
    assert captioner.calls == 0
    assert bus.emitted_long == []


@pytest.mark.asyncio
async def test_capture_failure_swallowed() -> None:
    bus = _FakeBus()
    captioner = _Captioner()
    engine = _make_engine(bus, captioner=captioner, capture_ok=False)
    await bus.fire("presence.snapshot",
                   present=True, face_count=1, quality="good", ts=time.time())
    await asyncio.sleep(0.05)
    assert captioner.calls == 0
    assert bus.emitted_long == []
    assert engine.metrics["errors"] >= 1


@pytest.mark.asyncio
async def test_captioner_exception_does_not_emit() -> None:
    bus = _FakeBus()
    broken = _BrokenCaptioner()
    engine = _make_engine(bus, captioner=broken)
    await bus.fire("presence.snapshot",
                   present=True, face_count=1, quality="good", ts=time.time())
    await asyncio.sleep(0.05)
    assert broken.calls == 1
    assert bus.emitted_long == []
    assert engine.metrics["errors"] >= 1


@pytest.mark.asyncio
async def test_empty_caption_does_not_emit() -> None:
    bus = _FakeBus()
    captioner = _Captioner(text="   ")
    engine = _make_engine(bus, captioner=captioner)
    await bus.fire("presence.snapshot",
                   present=True, face_count=1, quality="good", ts=time.time())
    await asyncio.sleep(0.05)
    assert captioner.calls == 1
    assert bus.emitted_long == []


# ── manual trigger ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_manual_caption_now_bypasses_significance_gate() -> None:
    bus = _FakeBus()
    captioner = _Captioner("Manual look.")
    engine = _make_engine(bus, captioner=captioner)
    scene = await engine.caption_now(trigger="manual")
    assert scene is not None
    assert scene.caption == "Manual look."
    assert scene.trigger == "manual"
    assert any(evt == "scene.context" for evt, _ in bus.emitted_long)


# ── attach idempotency ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_attach_is_idempotent() -> None:
    bus = _FakeBus()
    engine = _make_engine(bus)
    engine.attach()
    engine.attach()
    assert len(bus.handlers["presence.snapshot"]) == 1


@pytest.mark.asyncio
async def test_detach_unregisters() -> None:
    bus = _FakeBus()
    engine = _make_engine(bus)
    engine.detach()
    assert bus.handlers.get("presence.snapshot", []) == []


# ── data class ────────────────────────────────────────────────────


def test_scene_context_as_dict_contains_required_fields() -> None:
    s = SceneContext(
        ts=1.0, caption="cap", trigger="first",
        face_count=1, quality="good", elapsed_ms=12.3,
    )
    d = s.as_dict()
    assert d == {
        "ts": 1.0,
        "caption": "cap",
        "trigger": "first",
        "face_count": 1,
        "quality": "good",
        "elapsed_ms": 12.3,
    }
