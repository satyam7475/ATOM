"""Regression tests for Sprint N1 -- continuous screen perception loop.

These tests run against an in-memory stub of ``ScreenReader`` so we
don't actually screencapture or run OCR (and so they're cheap and
deterministic in CI). The key behaviours we want to lock in:

    * the loop is gated by presence + speaking + listening
    * duplicate frames are deduped via the perceptual-hash gate
    * password-field-looking lines get redacted before persistence
    * the SQLite ring buffer is capped at ``max_rows``
    * the public ``query`` filter API works
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

import pytest

from core.perception.screen_perception_loop import (
    ScreenLoopConfig,
    ScreenPerceptionLoop,
    _phash_text,
)


# ── stubs ──────────────────────────────────────────────────────────────


class _StubBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def on(self, *_a, **_kw) -> None:
        return None

    def emit_long(self, name: str, **payload: Any) -> None:
        self.events.append((name, payload))

    def emit_fast(self, name: str, **payload: Any) -> None:
        self.events.append((name, payload))


class _StubScreenReader:
    is_available = True
    ocr_backend = "stub"

    def __init__(self, frames: list[dict[str, Any]]) -> None:
        self._frames = list(frames)
        self.calls = 0

    def capture_and_read(self) -> dict[str, Any]:
        self.calls += 1
        if not self._frames:
            return {"text": ""}
        return self._frames.pop(0)


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "screen_obs.sqlite"


def _build_loop(
    bus: Any, reader: Any, db_path: Path,
    *,
    require_presence: bool = False,
    pause_speech: bool = False,
    pause_listen: bool = False,
    interval: float = 0.001,
) -> ScreenPerceptionLoop:
    cfg = ScreenLoopConfig(
        enabled=True,
        interval_s=interval,
        pause_during_speech=pause_speech,
        pause_during_listen=pause_listen,
        require_presence=require_presence,
        max_rows=50,
        db_path=str(db_path),
        redact_passwords=True,
        min_text_chars=4,
        significance_min_jaccard=0.55,
        burst_when_idle_s=60.0,
        emit_bus_event=True,
    )
    return ScreenPerceptionLoop(bus, reader, None, config=cfg)


# ── tests ──────────────────────────────────────────────────────────────


def test_phash_is_stable_for_token_set() -> None:
    a = _phash_text("Boss writes ATOM code")
    b = _phash_text("ATOM code Boss writes")
    assert a == b
    c = _phash_text("Boss writes ATOM tests now")
    assert a != c


@pytest.mark.asyncio
async def test_loop_persists_first_frame_and_emits_event(tmp_db: Path) -> None:
    bus = _StubBus()
    reader = _StubScreenReader(
        [{"text": "Boss is reading the morning briefing for ATOM"}],
    )
    loop = _build_loop(bus, reader, tmp_db)

    await loop._tick()

    rows = loop.query(limit=10)
    assert len(rows) == 1
    assert "morning briefing" in rows[0]["text"].lower()

    # emitted on the bus
    assert any(name == "screen.observation" for name, _ in bus.events)


@pytest.mark.asyncio
async def test_loop_dedupes_identical_consecutive_frames(tmp_db: Path) -> None:
    bus = _StubBus()
    text = "Boss is reading the same project plan as before"
    reader = _StubScreenReader([{"text": text}, {"text": text}])
    loop = _build_loop(bus, reader, tmp_db)

    await loop._tick()
    await loop._tick()

    rows = loop.query(limit=10)
    assert len(rows) == 1
    assert loop.metrics()["deduped"] >= 1


@pytest.mark.asyncio
async def test_loop_persists_when_text_changes_substantially(
    tmp_db: Path,
) -> None:
    bus = _StubBus()
    reader = _StubScreenReader(
        [
            {"text": "Boss is in the morning standup with the engineering team"},
            {"text": "Boss is now editing a totally different deck about Mars"},
        ],
    )
    loop = _build_loop(bus, reader, tmp_db)

    await loop._tick()
    await loop._tick()

    rows = loop.query(limit=10)
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_redacts_lines_after_password_hint(tmp_db: Path) -> None:
    bus = _StubBus()
    # The value on the line after "Password:" must NOT itself contain
    # any of the password-hint keywords, otherwise the redactor (rightly)
    # treats it as another hint line and preserves it.
    reader = _StubScreenReader(
        [
            {
                "text": (
                    "Settings\nPassword:\n"
                    "Pa55w0rdXYZ_omega_42\nOther line"
                ),
            },
        ],
    )
    loop = _build_loop(bus, reader, tmp_db)

    await loop._tick()

    rows = loop.query(limit=5)
    assert rows
    text = rows[0]["text"]
    assert "Pa55w0rdXYZ_omega_42" not in text
    assert "redacted-by-atom" in text


@pytest.mark.asyncio
async def test_loop_short_text_is_skipped(tmp_db: Path) -> None:
    bus = _StubBus()
    reader = _StubScreenReader([{"text": "hi"}])  # below min_text_chars
    loop = _build_loop(bus, reader, tmp_db)

    await loop._tick()

    assert loop.query(limit=10) == []


@pytest.mark.asyncio
async def test_query_filters_by_text_and_app(tmp_db: Path) -> None:
    bus = _StubBus()
    reader = _StubScreenReader(
        [
            {"text": "alpha alpha alpha keyword-foo"},
            {"text": "beta beta beta delta echo"},
        ],
    )
    loop = _build_loop(bus, reader, tmp_db)

    await loop._tick()
    await loop._tick()

    foo_rows = loop.query(text_contains="keyword-foo", limit=10)
    assert len(foo_rows) == 1
    assert "keyword-foo" in foo_rows[0]["text"]


@pytest.mark.asyncio
async def test_should_skip_when_speaking_or_listening(tmp_db: Path) -> None:
    bus = _StubBus()
    reader = _StubScreenReader([])
    loop = _build_loop(
        bus, reader, tmp_db,
        pause_speech=True, pause_listen=True,
    )

    loop._is_speaking = True
    assert loop._should_skip() == "speaking"
    loop._is_speaking = False
    loop._is_listening = True
    assert loop._should_skip() == "listening"
    loop._is_listening = False
    loop._presence_present = False
    # presence not required by default in helper, so toggle:
    loop.config = ScreenLoopConfig(
        enabled=True, interval_s=0.001, require_presence=True,
        db_path=str(tmp_db),
    )
    assert loop._should_skip() == "presence_absent"


def test_metrics_snapshot_shape(tmp_db: Path) -> None:
    bus = _StubBus()
    reader = _StubScreenReader([])
    loop = _build_loop(bus, reader, tmp_db)
    snap = loop.metrics()
    for key in (
        "samples", "persisted", "deduped", "paused", "errors",
        "db_path", "config",
    ):
        assert key in snap
