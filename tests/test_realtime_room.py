"""Sprint L1/L2 -- realtime room transport + RoomAgent bridge.

Covers the in-memory :class:`AtomRoom`, :class:`RoomParticipant`
serialization rules, and the :class:`RoomAgent` event mirroring +
chat/say/stt routing without spinning up a real HTTP server.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core.realtime.atom_room import (
    AtomRoom,
    AtomRoomServer,
    Frame,
    FrameKind,
    RoomConfig,
    RoomParticipant,
)
from core.realtime.room_agent import RoomAgent, RoomAgentConfig


pytestmark = pytest.mark.asyncio


class _StubBus:
    """Minimal AsyncEventBus stand-in: ``on``/``emit_fast``/``emit_long``."""

    def __init__(self) -> None:
        self.handlers: dict[str, list[Any]] = {}
        self.emitted_long: list[tuple[str, dict[str, Any]]] = []
        self.emitted_fast: list[tuple[str, dict[str, Any]]] = []

    def on(self, event: str, handler: Any) -> None:
        self.handlers.setdefault(event, []).append(handler)

    def off(self, event: str, handler: Any) -> None:
        try:
            self.handlers.get(event, []).remove(handler)
        except ValueError:
            pass

    def emit_fast(self, event: str, **payload: Any) -> None:
        self.emitted_fast.append((event, payload))

    def emit_long(self, event: str, **payload: Any) -> None:
        self.emitted_long.append((event, payload))

    async def fire(self, event: str, **payload: Any) -> None:
        for h in list(self.handlers.get(event, [])):
            await h(**payload)


class _StubCommandLoop:
    def __init__(self) -> None:
        self.submitted: list[tuple[str, str]] = []

    async def submit(self, text: str, *, priority: str = "voice", **_kw: Any) -> None:
        self.submitted.append((text, priority))


class _StubTTS:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def speak(self, text: str) -> None:
        self.spoken.append(text)


# ── helpers ────────────────────────────────────────────────────────


async def _drain_until(p: RoomParticipant, *, contains: str, max_iter: int = 16) -> str:
    """Drain ``p``'s send queue until a message containing ``contains``."""
    for _ in range(max_iter):
        msg = await asyncio.wait_for(p._drain(), timeout=0.5)
        if isinstance(msg, str) and contains in msg:
            return msg
    raise AssertionError(f"no message containing {contains!r}")


# ── frame protocol ─────────────────────────────────────────────────


async def test_frame_round_trips_through_encode_decode() -> None:
    raw = b"hello world"
    frame = Frame(kind=FrameKind.AUDIO_IN, payload=raw)
    decoded = Frame.decode(frame.encode())
    assert decoded.kind == FrameKind.AUDIO_IN
    assert decoded.payload == raw


async def test_frame_decode_rejects_unknown_tag() -> None:
    with pytest.raises(ValueError):
        Frame.decode(b"\xff stuff")


async def test_frame_decode_rejects_empty() -> None:
    with pytest.raises(ValueError):
        Frame.decode(b"")


# ── room core ──────────────────────────────────────────────────────


async def test_room_dispatches_data_handlers() -> None:
    room = AtomRoom(RoomConfig(name="t", max_participants=3))
    seen: list[tuple[str, str, dict[str, Any]]] = []

    async def handler(p: RoomParticipant, event: str, data: dict[str, Any]) -> None:
        seen.append((p.name, event, data))

    room.on_data(handler)
    p = RoomParticipant(id="p1", name="Boss")
    assert await room.add_participant(p) is True

    await room.dispatch_data(p, "chat.message", {"text": "hi"})
    assert seen == [("Boss", "chat.message", {"text": "hi"})]


async def test_room_server_requires_token_for_non_loopback_bind() -> None:
    room = AtomRoom(RoomConfig(name="t", auth_token=None))
    server = AtomRoomServer(room, host="0.0.0.0", port=0)

    with pytest.raises(RuntimeError, match="auth_token"):
        await server.start()


async def test_room_records_track_on_first_frame() -> None:
    room = AtomRoom()
    p = RoomParticipant(id="p2", name="Boss")
    await room.add_participant(p)

    await room.dispatch_frame(p, Frame(kind=FrameKind.VIDEO_IN, payload=b"xxx"))
    await room.dispatch_frame(p, Frame(kind=FrameKind.VIDEO_IN, payload=b"yyy"))

    track = p.tracks[FrameKind.VIDEO_IN]
    assert track.total_frames == 2
    assert track.total_bytes == 6


async def test_room_capacity_rejects_over_limit() -> None:
    room = AtomRoom(RoomConfig(name="t", max_participants=1))
    a = RoomParticipant(id="a", name="A")
    b = RoomParticipant(id="b", name="B")
    assert await room.add_participant(a) is True
    assert await room.add_participant(b) is False
    assert len(room.participants) == 1


async def test_remove_participant_emits_left_event() -> None:
    room = AtomRoom()
    seen_events: list[tuple[str, dict[str, Any]]] = []

    async def handler(p: RoomParticipant, event: str, data: dict[str, Any]) -> None:
        seen_events.append((event, data))

    room.on_data(handler)
    p1 = RoomParticipant(id="p1", name="Boss")
    p2 = RoomParticipant(id="p2", name="Other")
    await room.add_participant(p1)
    await room.add_participant(p2)

    await room.remove_participant("p2")
    assert any(e[0] == "participant.left" for e in seen_events) or len(room.participants) == 1


async def test_participant_send_data_serializes_json() -> None:
    p = RoomParticipant(id="x", name="Boss")
    await p.send_data("hello", value=42)
    msg = await p._drain()
    assert isinstance(msg, str)
    assert "\"hello\"" in msg and "42" in msg


# ── RoomAgent: chat path ───────────────────────────────────────────


async def test_room_agent_routes_chat_message_to_command_loop() -> None:
    bus = _StubBus()
    room = AtomRoom()
    cl = _StubCommandLoop()
    agent = RoomAgent(room, bus, command_loop=cl)
    agent.attach()

    p = RoomParticipant(id="p", name="Boss")
    await room.add_participant(p)
    await room.dispatch_data(p, "chat.message", {"text": "play music"})

    # let create_task run
    await asyncio.sleep(0)
    assert cl.submitted and cl.submitted[0][0] == "play music"


async def test_room_agent_chat_message_broadcasts_boss_chat() -> None:
    bus = _StubBus()
    room = AtomRoom()
    agent = RoomAgent(room, bus, command_loop=_StubCommandLoop())
    agent.attach()

    p = RoomParticipant(id="boss", name="Boss")
    other = RoomParticipant(id="other", name="Other")
    await room.add_participant(p)
    await room.add_participant(other)

    await room.dispatch_data(p, "chat.message", {"text": "hi"})
    msg = await _drain_until(other, contains="boss.chat")
    assert "boss.chat" in msg


async def test_room_agent_say_text_uses_tts_first() -> None:
    bus = _StubBus()
    room = AtomRoom()
    tts = _StubTTS()
    agent = RoomAgent(room, bus, tts=tts)
    agent.attach()

    p = RoomParticipant(id="p", name="Boss")
    await room.add_participant(p)
    await room.dispatch_data(p, "say.text", {"text": "online, Boss"})
    await asyncio.sleep(0)
    assert tts.spoken == ["online, Boss"]


async def test_room_agent_stt_text_emits_speech_final() -> None:
    bus = _StubBus()
    room = AtomRoom()
    agent = RoomAgent(room, bus)
    agent.attach()

    p = RoomParticipant(id="p", name="Boss")
    await room.add_participant(p)
    await room.dispatch_data(p, "stt.text", {"text": "play music"})

    assert any(ev == "speech_final" for ev, _ in bus.emitted_long)


# ── RoomAgent: bus mirroring ───────────────────────────────────────


async def test_room_agent_mirrors_state_change_to_room() -> None:
    bus = _StubBus()
    room = AtomRoom()
    agent = RoomAgent(room, bus)
    agent.attach()

    p = RoomParticipant(id="p", name="Boss")
    await room.add_participant(p)

    # Drain the welcome / greeting messages before we test the mirror
    drained = 0
    while drained < 2 and p._send_queue and not p._send_queue.empty():
        await p._drain()
        drained += 1

    await bus.fire("state_changed", new_state="THINKING")
    msg = await p._drain()
    assert "atom.state" in msg


async def test_room_agent_forwards_partial_responses_when_enabled() -> None:
    bus = _StubBus()
    room = AtomRoom()
    agent = RoomAgent(room, bus, config=RoomAgentConfig(forward_partial_responses=True))
    agent.attach()

    p = RoomParticipant(id="p", name="Boss")
    await room.add_participant(p)
    while p._send_queue and not p._send_queue.empty():
        await p._drain()

    await bus.fire("partial_response", text="Hel")
    msg = await p._drain()
    assert "atom.partial" in msg


async def test_room_agent_caches_video_frames() -> None:
    bus = _StubBus()
    room = AtomRoom()
    agent = RoomAgent(room, bus)
    agent.attach()
    p = RoomParticipant(id="p", name="Boss")
    await room.add_participant(p)

    await room.dispatch_frame(p, Frame(kind=FrameKind.VIDEO_IN, payload=b"jpeg-bytes"))
    assert agent.latest_video_jpeg() == b"jpeg-bytes"
