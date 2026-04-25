"""ATOM Realtime Room -- LiveKit-style multi-channel WebSocket link.

A *room* is the meeting place between Boss (any browser / iPhone tab)
and ATOM. One participant -- ATOM -- always exists; the rest are
remote clients. Each participant owns a single bidirectional
WebSocket carrying multiplexed frames:

* **Text frames**: JSON control / data messages (chat, state,
  presence, mood updates, custom data events).
* **Binary frames**: media payloads. The first byte is a one-byte
  channel tag (:class:`FrameKind`) so we never need separate WebSocket
  routes for audio/video/screen.

This file ships the *transport layer* only -- it knows nothing about
intent engines, LLMs, or wake words. The :mod:`core.realtime.room_agent`
module wires the room into ATOM's existing CommandLoop / event bus.

Design goals
------------
* **Drop-in dependency surface**: :mod:`aiohttp` only (already in
  requirements.txt). No external SFU. No gstreamer. Works from a
  laptop, an iPhone Safari tab, or a friend's MacBook on the same LAN.
* **Symmetric API to LiveKit**: ``Room.broadcast()``,
  ``Participant.publish_data()``, ``Participant.publish_audio()`` --
  switching to LiveKit-server later is a transport swap, not an API
  rewrite.
* **Backpressure-aware**: every outbound queue is bounded; slow
  participants get dropped frames instead of bloating ATOM's RAM.
* **Test-friendly**: the room can be driven without an HTTP server
  via :class:`AtomRoom` directly, which is what the regression tests
  in ``tests/test_realtime_room.py`` do.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Awaitable, Callable

logger = logging.getLogger("atom.realtime.room")


# ── Frame protocol ──────────────────────────────────────────────────


class FrameKind(IntEnum):
    """One-byte channel tag prefixing every binary frame."""

    AUDIO_IN = 0x01      # browser mic -> ATOM, PCM s16le 16 kHz mono
    AUDIO_OUT = 0x02     # ATOM TTS    -> browser, PCM s16le 22 kHz mono
    VIDEO_IN = 0x03      # browser cam  -> ATOM, JPEG bytes
    SCREEN_IN = 0x04     # browser scrn -> ATOM, JPEG bytes
    AUDIO_OUT_OPUS = 0x05  # ATOM TTS    -> browser, opus/mp3 chunk
    VIDEO_OUT = 0x06     # ATOM cam    -> browser, JPEG (avatar / orb)


@dataclass(slots=True)
class Frame:
    """In-memory representation of a single binary media frame."""

    kind: FrameKind
    payload: bytes
    received_at: float = field(default_factory=time.time)

    def encode(self) -> bytes:
        return bytes([int(self.kind)]) + self.payload

    @classmethod
    def decode(cls, raw: bytes) -> "Frame":
        if not raw:
            raise ValueError("empty binary frame")
        try:
            kind = FrameKind(raw[0])
        except ValueError as exc:
            raise ValueError(f"unknown frame kind 0x{raw[0]:02x}") from exc
        return cls(kind=kind, payload=raw[1:])


# ── Track + participant ─────────────────────────────────────────────


@dataclass(slots=True)
class RoomTrack:
    """A published track on a participant.

    Tracks are created lazily the first time we see a frame of a
    given :class:`FrameKind` from a participant -- mirrors the
    LiveKit ``onTrackPublished`` event without needing an explicit
    SDP-style negotiation.
    """

    kind: FrameKind
    participant_id: str
    started_at: float = field(default_factory=time.time)
    last_frame_at: float = 0.0
    total_frames: int = 0
    total_bytes: int = 0

    def observe(self, frame: Frame) -> None:
        self.last_frame_at = frame.received_at
        self.total_frames += 1
        self.total_bytes += len(frame.payload)


@dataclass(slots=True)
class RoomParticipant:
    """One peer in a room (Boss in a browser tab, ATOM itself, etc.)."""

    id: str
    name: str
    role: str = "human"          # "human" | "atom" | "observer"
    joined_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Outbound queue: bounded to keep ATOM's RAM honest.
    _send_queue: asyncio.Queue[bytes | str] | None = None
    _closed: asyncio.Event | None = None
    _tracks: dict[FrameKind, RoomTrack] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self._send_queue is None:
            self._send_queue = asyncio.Queue(maxsize=256)
        if self._closed is None:
            self._closed = asyncio.Event()

    # --- public API ------------------------------------------------

    @property
    def is_alive(self) -> bool:
        return self._closed is not None and not self._closed.is_set()

    @property
    def tracks(self) -> dict[FrameKind, RoomTrack]:
        return dict(self._tracks)

    async def send_data(self, event: str, **payload: Any) -> bool:
        """Queue a structured data message. Returns ``False`` if dropped."""
        msg = {"type": "data", "event": event, "data": payload, "ts": time.time()}
        return await self._enqueue(json.dumps(msg, default=_json_safe))

    async def send_state(self, state: str, **payload: Any) -> bool:
        """Convenience wrapper for the most common event class."""
        msg = {"type": "state", "state": state, "data": payload, "ts": time.time()}
        return await self._enqueue(json.dumps(msg, default=_json_safe))

    async def send_frame(self, frame: Frame) -> bool:
        return await self._enqueue(frame.encode())

    async def send_audio_out(self, pcm: bytes, *, opus: bool = False) -> bool:
        kind = FrameKind.AUDIO_OUT_OPUS if opus else FrameKind.AUDIO_OUT
        return await self.send_frame(Frame(kind=kind, payload=pcm))

    def close(self) -> None:
        if self._closed is not None:
            self._closed.set()

    # --- internal --------------------------------------------------

    async def _enqueue(self, msg: bytes | str) -> bool:
        if self._send_queue is None or self._closed is None or self._closed.is_set():
            return False
        try:
            self._send_queue.put_nowait(msg)
            return True
        except asyncio.QueueFull:
            # Drop oldest to make room -- degrade gracefully under
            # pressure (slow participant). We log at debug so the live
            # session stays calm.
            try:
                _ = self._send_queue.get_nowait()
                self._send_queue.task_nowait if False else None  # noqa: B015
                self._send_queue.put_nowait(msg)
                logger.debug("room participant=%s send queue full; dropped oldest", self.id)
                return True
            except Exception:
                logger.debug("room participant=%s send queue full; dropped new", self.id)
                return False

    async def _drain(self) -> bytes | str:
        assert self._send_queue is not None
        return await self._send_queue.get()

    def _record_track(self, frame: Frame) -> RoomTrack:
        track = self._tracks.get(frame.kind)
        if track is None:
            track = RoomTrack(kind=frame.kind, participant_id=self.id)
            self._tracks[frame.kind] = track
        track.observe(frame)
        return track


def _json_safe(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return value.hex()
    if hasattr(value, "name") and isinstance(getattr(value, "value", None), int):
        return value.name
    return repr(value)


# ── Room core ───────────────────────────────────────────────────────


DataHandler = Callable[[RoomParticipant, str, dict[str, Any]], Awaitable[None]]
FrameHandler = Callable[[RoomParticipant, Frame], Awaitable[None]]
LifecycleHandler = Callable[[RoomParticipant], Awaitable[None]]


@dataclass(slots=True)
class RoomConfig:
    name: str = "atom-room"
    max_participants: int = 8
    auth_token: str | None = None  # ``None`` = LAN-only / dev mode
    heartbeat_interval_s: float = 15.0


class AtomRoom:
    """Pure in-memory multi-channel room.

    The :class:`AtomRoomServer` wraps this with an HTTP / WebSocket
    front; the tests drive :class:`AtomRoom` directly.
    """

    def __init__(self, config: RoomConfig | None = None) -> None:
        self.config = config or RoomConfig()
        self._participants: dict[str, RoomParticipant] = {}
        self._on_data_handlers: list[DataHandler] = []
        self._on_frame_handlers: list[FrameHandler] = []
        self._on_join_handlers: list[LifecycleHandler] = []
        self._on_leave_handlers: list[LifecycleHandler] = []
        self._lock = asyncio.Lock()

    # --- handler registration -------------------------------------

    def on_data(self, handler: DataHandler) -> None:
        self._on_data_handlers.append(handler)

    def on_frame(self, handler: FrameHandler) -> None:
        self._on_frame_handlers.append(handler)

    def on_join(self, handler: LifecycleHandler) -> None:
        self._on_join_handlers.append(handler)

    def on_leave(self, handler: LifecycleHandler) -> None:
        self._on_leave_handlers.append(handler)

    # --- participants ---------------------------------------------

    @property
    def participants(self) -> list[RoomParticipant]:
        return list(self._participants.values())

    def get(self, participant_id: str) -> RoomParticipant | None:
        return self._participants.get(participant_id)

    async def add_participant(self, participant: RoomParticipant) -> bool:
        async with self._lock:
            if len(self._participants) >= self.config.max_participants:
                logger.warning(
                    "room=%s rejected participant=%s: capacity %d reached",
                    self.config.name,
                    participant.id,
                    self.config.max_participants,
                )
                return False
            self._participants[participant.id] = participant
        for handler in list(self._on_join_handlers):
            try:
                await handler(participant)
            except Exception:
                logger.exception("on_join handler raised for participant=%s", participant.id)
        await self.broadcast_data(
            "participant.joined",
            participant_id=participant.id,
            name=participant.name,
            role=participant.role,
            count=len(self._participants),
            exclude=participant.id,
        )
        logger.info(
            "room=%s join participant=%s name=%s role=%s (%d total)",
            self.config.name, participant.id, participant.name, participant.role,
            len(self._participants),
        )
        return True

    async def remove_participant(self, participant_id: str) -> None:
        async with self._lock:
            participant = self._participants.pop(participant_id, None)
        if participant is None:
            return
        participant.close()
        for handler in list(self._on_leave_handlers):
            try:
                await handler(participant)
            except Exception:
                logger.exception("on_leave handler raised for participant=%s", participant_id)
        await self.broadcast_data(
            "participant.left",
            participant_id=participant.id,
            name=participant.name,
            count=len(self._participants),
        )
        logger.info("room=%s leave participant=%s (%d remain)", self.config.name, participant_id, len(self._participants))

    # --- broadcast helpers ----------------------------------------

    async def broadcast_data(self, event: str, *, exclude: str | None = None, **payload: Any) -> None:
        targets = [p for p in self._participants.values() if p.id != exclude]
        await asyncio.gather(*(p.send_data(event, **payload) for p in targets), return_exceptions=True)

    async def broadcast_state(self, state: str, *, exclude: str | None = None, **payload: Any) -> None:
        targets = [p for p in self._participants.values() if p.id != exclude]
        await asyncio.gather(*(p.send_state(state, **payload) for p in targets), return_exceptions=True)

    async def broadcast_audio_out(self, pcm: bytes, *, exclude: str | None = None, opus: bool = False) -> None:
        targets = [p for p in self._participants.values() if p.id != exclude and p.role == "human"]
        await asyncio.gather(
            *(p.send_audio_out(pcm, opus=opus) for p in targets),
            return_exceptions=True,
        )

    # --- inbound dispatch ----------------------------------------

    async def dispatch_data(self, participant: RoomParticipant, event: str, payload: dict[str, Any]) -> None:
        for handler in list(self._on_data_handlers):
            try:
                await handler(participant, event, payload)
            except Exception:
                logger.exception("on_data handler raised event=%s participant=%s", event, participant.id)

    async def dispatch_frame(self, participant: RoomParticipant, frame: Frame) -> None:
        participant._record_track(frame)
        for handler in list(self._on_frame_handlers):
            try:
                await handler(participant, frame)
            except Exception:
                logger.exception("on_frame handler raised kind=%s participant=%s", frame.kind.name, participant.id)

    # --- snapshot -------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.config.name,
            "participant_count": len(self._participants),
            "participants": [
                {
                    "id": p.id,
                    "name": p.name,
                    "role": p.role,
                    "joined_at": p.joined_at,
                    "tracks": {
                        k.name: {
                            "frames": t.total_frames,
                            "bytes": t.total_bytes,
                            "last_frame_at": t.last_frame_at,
                        }
                        for k, t in p._tracks.items()
                    },
                }
                for p in self._participants.values()
            ],
        }


# ── HTTP / WebSocket server ─────────────────────────────────────────


# Imported lazily so unit tests that only exercise AtomRoom don't pay
# the aiohttp import cost on import-time.
def _lazy_aiohttp():  # pragma: no cover - import shim
    from aiohttp import WSMsgType, web

    return WSMsgType, web


class AtomRoomServer:
    """aiohttp wrapper that exposes :class:`AtomRoom` over HTTP."""

    def __init__(
        self,
        room: AtomRoom,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        playground_dir: str | None = None,
    ) -> None:
        self.room = room
        self.host = host
        self.port = port
        self.playground_dir = playground_dir
        self._runner: Any = None
        self._site: Any = None
        self._app: Any = None

    async def start(self) -> None:
        WSMsgType, web = _lazy_aiohttp()
        app = web.Application()
        app.router.add_get("/healthz", self._handle_healthz)
        app.router.add_get("/room/snapshot", self._handle_snapshot)
        app.router.add_get("/room/ws", self._handle_ws)
        if self.playground_dir:
            from pathlib import Path

            pdir = Path(self.playground_dir)
            if pdir.is_dir():
                app.router.add_static("/play/", path=str(pdir), show_index=True)
                app.router.add_get("/", lambda _r: web.HTTPFound("/play/"))
        self._app = app
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        self._runner = runner
        self._site = site
        logger.info("AtomRoom listening on http://%s:%d (room=%s)", self.host, self.port, self.room.config.name)

    async def stop(self) -> None:
        if self._site is not None:
            await self._site.stop()
            self._site = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        self._app = None

    # ── handlers ---------------------------------------------------

    async def _handle_healthz(self, _request: Any) -> Any:
        WSMsgType, web = _lazy_aiohttp()
        return web.json_response({"ok": True, "room": self.room.config.name})

    async def _handle_snapshot(self, _request: Any) -> Any:
        WSMsgType, web = _lazy_aiohttp()
        return web.json_response(self.room.snapshot())

    async def _handle_ws(self, request: Any) -> Any:
        WSMsgType, web = _lazy_aiohttp()
        # Auth: token via query string, header, or sub-protocol. LAN
        # dev mode allows anonymous when ``auth_token`` is None.
        configured_token = self.room.config.auth_token
        if configured_token:
            supplied = (
                request.query.get("token")
                or request.headers.get("X-Atom-Token")
                or request.headers.get("Sec-WebSocket-Protocol", "").split(",")[0].strip()
            )
            if supplied != configured_token:
                return web.Response(status=401, text="unauthorized")

        ws = web.WebSocketResponse(heartbeat=self.room.config.heartbeat_interval_s, max_msg_size=4 * 1024 * 1024)
        await ws.prepare(request)

        name = (request.query.get("name") or "Boss").strip()[:64] or "Boss"
        role = (request.query.get("role") or "human").strip()[:16] or "human"
        participant = RoomParticipant(id=secrets.token_urlsafe(8), name=name, role=role)

        added = await self.room.add_participant(participant)
        if not added:
            await ws.close(code=1013, message=b"room full")
            return ws

        # Send the welcome snapshot synchronously so the client knows
        # who's in the room before any other event arrives.
        await participant.send_data(
            "room.welcome",
            participant_id=participant.id,
            room=self.room.config.name,
            snapshot=self.room.snapshot(),
        )

        sender_task = asyncio.create_task(self._sender_loop(ws, participant))

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    await self._handle_text(participant, msg.data)
                elif msg.type == WSMsgType.BINARY:
                    await self._handle_binary(participant, msg.data)
                elif msg.type == WSMsgType.ERROR:
                    logger.warning("room ws error participant=%s: %s", participant.id, ws.exception())
                    break
        finally:
            sender_task.cancel()
            try:
                await sender_task
            except Exception:
                pass
            await self.room.remove_participant(participant.id)
            if not ws.closed:
                await ws.close()

        return ws

    async def _sender_loop(self, ws: Any, participant: RoomParticipant) -> None:
        WSMsgType, _web = _lazy_aiohttp()
        try:
            while participant.is_alive and not ws.closed:
                msg = await participant._drain()
                if isinstance(msg, (bytes, bytearray)):
                    await ws.send_bytes(bytes(msg))
                else:
                    await ws.send_str(msg)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("sender loop crashed participant=%s", participant.id)

    async def _handle_text(self, participant: RoomParticipant, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("room participant=%s sent invalid JSON: %r", participant.id, raw[:80])
            return
        if not isinstance(payload, dict):
            return
        msg_type = payload.get("type")
        if msg_type == "data":
            event = str(payload.get("event") or "data")
            data = payload.get("data") or {}
            if isinstance(data, dict):
                await self.room.dispatch_data(participant, event, data)
        elif msg_type == "ping":
            await participant.send_data("pong", t=payload.get("t"))
        elif msg_type == "metadata":
            md = payload.get("data") or {}
            if isinstance(md, dict):
                participant.metadata.update(md)

    async def _handle_binary(self, participant: RoomParticipant, raw: bytes) -> None:
        try:
            frame = Frame.decode(raw)
        except ValueError as exc:
            logger.debug("room participant=%s bad binary frame: %s", participant.id, exc)
            return
        await self.room.dispatch_frame(participant, frame)
