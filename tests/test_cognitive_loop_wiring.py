"""Regression: Phase G cognitive-loop wiring is actually attached at boot.

Before this wiring existed, every G module passed its own unit tests but
none of them were ever instantiated by ``main.py``. ATOM booted as a
nice-looking chatbot. These tests pin the wiring so a future refactor
can't silently regress that.

We don't boot ``main.py`` here -- that's heavy and platform-bound. We
exercise ``core.boot.cognitive_loop_wiring.wire_cognitive_loop``
directly with stub bus/state/command_loop and assert:

  * ``turn_complete`` emitter is attached (G6)
  * ``response_ready / tts_complete / speech_final`` subscribers exist
  * ``mood.state`` event has at least one subscriber  (suggester)
  * ``presence.snapshot`` event has subscribers       (mood + suggester)
  * ``command_loop_trace`` event has subscribers      (mood + suggester)
  * ReflectiveLoop wires when a local_brain stub is provided
  * Disabling the loop in config is honoured
  * ``stop()`` cleanly detaches subscribers
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.async_event_bus import AsyncEventBus  # noqa: E402
from core.boot.cognitive_loop_wiring import wire_cognitive_loop  # noqa: E402


# ── stubs ─────────────────────────────────────────────────────────────


class _StubState:
    class _S:
        value = "idle"

    current = _S()


class _StubLock:
    is_busy = False


class _StubCommandLoop:
    """Minimal stand-in for core.command_loop.CommandLoop.

    We only need the methods the wiring touches: ``attach_turn_emitter``
    and the ``is_busy`` property. The wiring also reads
    ``current_trace_id`` indirectly through trace events, but those
    aren't fired by the wiring itself, only by a live loop.
    """

    def __init__(self) -> None:
        self.attach_count = 0
        self._lock = _StubLock()

    def attach_turn_emitter(self) -> None:
        self.attach_count += 1

    @property
    def is_busy(self) -> bool:
        return False


class _StubLocalBrain:
    """Minimal MLXBrain stand-in honouring ``generate(prompt, ...)``."""

    async def generate(self, prompt: str, **kw: Any) -> tuple[str, bool]:
        return ("{\"decision\":\"none\"}", True)


# ── helpers ───────────────────────────────────────────────────────────


def _has_subscriber(bus: AsyncEventBus, event: str) -> bool:
    return bool(bus._subscribers.get(event))


# ── tests ─────────────────────────────────────────────────────────────


def test_wire_cognitive_loop_attaches_turn_emitter() -> None:
    bus = AsyncEventBus()
    state = _StubState()
    cmd = _StubCommandLoop()

    handles = wire_cognitive_loop(
        bus=bus, state=state, command_loop=cmd,
        config={"cognitive_loop": {"enabled": True,
                                    "presence": {"enabled": False},
                                    "scene": {"enabled": False}}},
        local_brain=None,
    )

    assert cmd.attach_count == 1, "turn_complete emitter must be wired exactly once"
    assert handles.enabled_summary is not None


def test_wire_cognitive_loop_attaches_mood_engine() -> None:
    bus = AsyncEventBus()
    handles = wire_cognitive_loop(
        bus=bus, state=_StubState(), command_loop=_StubCommandLoop(),
        config={"cognitive_loop": {"enabled": True}},
        local_brain=None, vision_engine=None, captioner=None,
    )

    assert handles.mood is not None, "MoodInferenceEngine must be instantiated"
    assert handles.enabled_summary["mood"] is True
    # MoodInferenceEngine subscribes to these four events:
    for ev in ("presence.snapshot", "user_emotion_detected",
               "command_loop_trace", "voice_metrics"):
        assert _has_subscriber(bus, ev), f"{ev} must have a subscriber"


def test_wire_cognitive_loop_attaches_jarvis_suggester() -> None:
    bus = AsyncEventBus()
    handles = wire_cognitive_loop(
        bus=bus, state=_StubState(), command_loop=_StubCommandLoop(),
        config={"cognitive_loop": {"enabled": True}},
        local_brain=None, vision_engine=None, captioner=None,
    )

    assert handles.suggester is not None
    assert handles.enabled_summary["suggester"] is True
    assert _has_subscriber(bus, "mood.state")
    assert _has_subscriber(bus, "presence.snapshot")
    assert _has_subscriber(bus, "command_loop_trace")


def test_wire_cognitive_loop_attaches_reflective_when_brain_present() -> None:
    bus = AsyncEventBus()
    handles = wire_cognitive_loop(
        bus=bus, state=_StubState(), command_loop=_StubCommandLoop(),
        config={"cognitive_loop": {"enabled": True}},
        local_brain=_StubLocalBrain(),
    )

    assert handles.reflective is not None
    assert handles.enabled_summary["reflective"] is True
    # ReflectiveLoop subscribes to these four:
    for ev in ("command_loop_trace", "response_ready",
               "tts_complete", "speech_final"):
        assert _has_subscriber(bus, ev), f"{ev} must have a subscriber"


def test_wire_cognitive_loop_skips_reflective_without_brain() -> None:
    bus = AsyncEventBus()
    handles = wire_cognitive_loop(
        bus=bus, state=_StubState(), command_loop=_StubCommandLoop(),
        config={"cognitive_loop": {"enabled": True}},
        local_brain=None,
    )

    assert handles.reflective is None
    assert handles.enabled_summary["reflective"] is False


def test_wire_cognitive_loop_skips_presence_without_vision() -> None:
    bus = AsyncEventBus()
    handles = wire_cognitive_loop(
        bus=bus, state=_StubState(), command_loop=_StubCommandLoop(),
        config={"cognitive_loop": {"enabled": True}},
        local_brain=None, vision_engine=None,
    )

    assert handles.presence is None
    assert handles.enabled_summary["presence"] is False


def test_wire_cognitive_loop_disabled_returns_empty_handles() -> None:
    bus = AsyncEventBus()
    cmd = _StubCommandLoop()
    handles = wire_cognitive_loop(
        bus=bus, state=_StubState(), command_loop=cmd,
        config={"cognitive_loop": {"enabled": False}},
        local_brain=_StubLocalBrain(),
    )

    assert handles.mood is None
    assert handles.suggester is None
    assert handles.reflective is None
    assert handles.presence is None
    assert handles.scene is None
    assert cmd.attach_count == 0, "turn emitter must not wire when loop disabled"


def test_wire_cognitive_loop_subsystem_toggles_independent() -> None:
    bus = AsyncEventBus()
    handles = wire_cognitive_loop(
        bus=bus, state=_StubState(), command_loop=_StubCommandLoop(),
        config={"cognitive_loop": {
            "enabled": True,
            "mood": {"enabled": False},
            "suggester": {"enabled": False},
            "reflective": {"enabled": False},
            "presence": {"enabled": False},
            "scene": {"enabled": False},
        }},
        local_brain=_StubLocalBrain(),
    )

    assert handles.mood is None
    assert handles.suggester is None
    assert handles.reflective is None


def test_wire_cognitive_loop_stop_detaches_subscribers() -> None:
    bus = AsyncEventBus()
    handles = wire_cognitive_loop(
        bus=bus, state=_StubState(), command_loop=_StubCommandLoop(),
        config={"cognitive_loop": {"enabled": True}},
        local_brain=_StubLocalBrain(),
    )

    assert _has_subscriber(bus, "mood.state")
    handles.stop()
    # After stop, the suggester / reflective / mood subscribers should
    # be gone (presence is async so we don't strictly assert it here).
    assert not _has_subscriber(bus, "mood.state"), \
        "stop() must detach JarvisSuggester from mood.state"


def test_main_module_imports_wire_helper() -> None:
    """Pin: main.py must actually import the wiring helper.

    A future refactor that drops the import would silently regress
    the entire Phase G surface area; catch it here without booting
    main.py end-to-end (which costs seconds).
    """
    import importlib
    import inspect

    main_mod = importlib.import_module("main")
    src = inspect.getsource(main_mod)
    assert "wire_cognitive_loop" in src, \
        "main.py must call wire_cognitive_loop() during boot"
    assert "from core.boot.cognitive_loop_wiring import wire_cognitive_loop" in src, \
        "main.py must import the helper"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
