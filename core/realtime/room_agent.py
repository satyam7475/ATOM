"""Bridge between :class:`AtomRoom` and the rest of ATOM.

The :class:`RoomAgent` is the glue layer that lets a browser /
iPhone tab feel like a local mic + speaker:

* **Inbound** (browser -> ATOM)
    * ``chat.message`` data event with ``text`` -> :meth:`CommandLoop.submit`
      (or, if the loop is busy, a polite "still working" reply).
    * ``say.text`` data event with ``text`` -> direct TTS / bus emission
      so Boss can drive ATOM's voice from a remote tab.
    * ``stt.text`` data event -> emitted on the bus as ``speech_final``;
      handy for desktop dictation tools that already do their own STT.
    * Binary ``AUDIO_IN`` frames -> tracked for the perception layer
      (full STT routing lands in Sprint L5; today the agent just
      counts frames and logs the track health).
    * Binary ``VIDEO_IN`` / ``SCREEN_IN`` frames -> latest frame
      cached so the cognitive layer (or Boss via API) can grab a
      JPEG snapshot of what the browser is sending.

* **Outbound** (ATOM -> browser)
    * State machine transitions (``LISTENING``, ``THINKING``, ...)
      mirrored into the room as ``state`` messages so the orb stays
      in sync.
    * ``partial_response`` -> ``atom.partial`` (token streaming).
    * ``response_ready``   -> ``atom.response`` (final spoken text).
    * ``tts_complete``     -> ``atom.spoke``    (audio finished).
    * ``mood_changed``     -> ``atom.mood``     (Boss can show badges).
    * ``suggestion_offer`` -> ``atom.suggestion`` (Sprint J offers).

The agent is designed so that *missing* an upstream component
(no CommandLoop, no TTS, no MoodEngine) degrades gracefully -- the
room still works as a chat passthrough.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from core.realtime.atom_room import (
    AtomRoom,
    Frame,
    FrameKind,
    RoomParticipant,
)

logger = logging.getLogger("atom.realtime.agent")


# Events we mirror from the bus to every room participant. Keeping the
# allow-list explicit prevents the room from drowning in trace spam.
_BUS_EVENT_MIRRORS: tuple[tuple[str, str], ...] = (
    ("state_changed", "atom.state"),
    ("mood_changed", "atom.mood"),
    ("partial_response", "atom.partial"),
    ("response_ready", "atom.response"),
    ("tts_complete", "atom.spoke"),
    ("speech_final", "boss.utterance"),
    ("wake", "atom.wake"),
    ("suggestion_offer", "atom.suggestion"),
    ("atom_thinking", "atom.thinking"),
)


@dataclass(slots=True)
class RoomAgentConfig:
    """Tuning knobs for the bridge."""

    forward_partial_responses: bool = True
    forward_audio_frames: bool = False  # full STT route lands in L5
    cache_video_frames: bool = True
    chat_priority: str = "voice"
    log_event_mirrors: bool = False


@dataclass(slots=True)
class _LatestMedia:
    """Most-recent media frame per channel, used for snapshots."""

    audio_in: Frame | None = None
    video_in: Frame | None = None
    screen_in: Frame | None = None
    counters: dict[str, int] = field(default_factory=dict)


class RoomAgent:
    """Wires a single :class:`AtomRoom` to ATOM's runtime."""

    def __init__(
        self,
        room: AtomRoom,
        bus: Any,
        *,
        command_loop: Any = None,
        state_manager: Any = None,
        tts: Any = None,
        cloud_brain_router: Any = None,
        config: RoomAgentConfig | None = None,
    ) -> None:
        self.room = room
        self.bus = bus
        self.command_loop = command_loop
        self.state = state_manager
        self.tts = tts
        self.cloud_brain_router = cloud_brain_router
        self.config = config or RoomAgentConfig()
        self._latest = _LatestMedia()
        self._attached = False

    # ── lifecycle -------------------------------------------------

    def attach(self) -> None:
        """Hook the room into the bus + register data/frame handlers."""
        if self._attached:
            return
        self._attached = True

        self.room.on_data(self._on_data)
        self.room.on_frame(self._on_frame)
        self.room.on_join(self._on_join)
        self.room.on_leave(self._on_leave)

        for bus_event, room_event in _BUS_EVENT_MIRRORS:
            on = getattr(self.bus, "on", None)
            if not callable(on):
                logger.warning("RoomAgent: bus has no .on() -- mirroring disabled")
                break
            on(bus_event, self._make_mirror(bus_event, room_event))

        logger.info("RoomAgent attached to room=%s", self.room.config.name)

    # ── inbound: data events --------------------------------------

    async def _on_data(self, participant: RoomParticipant, event: str, data: dict[str, Any]) -> None:
        if event == "chat.message":
            await self._handle_chat_message(participant, data)
        elif event == "say.text":
            await self._handle_say_text(participant, data)
        elif event == "stt.text":
            await self._handle_stt_text(participant, data)
        elif event == "snapshot.request":
            await participant.send_data("snapshot.reply", snapshot=self._snapshot_for_client())
        elif event == "ping":
            await participant.send_data("pong", t=data.get("t"))
        else:
            logger.debug(
                "RoomAgent: unhandled data event=%s participant=%s",
                event, participant.id,
            )

    async def _handle_chat_message(self, participant: RoomParticipant, data: dict[str, Any]) -> None:
        text = str(data.get("text") or "").strip()
        if not text:
            return
        priority = str(data.get("priority") or self.config.chat_priority)
        deep_hint = bool(data.get("deep"))

        await self.room.broadcast_data(
            "boss.chat",
            participant_id=participant.id,
            name=participant.name,
            text=text,
            priority=priority,
            deep=deep_hint,
            ts=time.time(),
        )

        # Sprint M1: explicit deep-cloud route when Boss flips the
        # toggle in the playground or prefixes the message with
        # "deep:" / "think hard". The cloud router does its own
        # quota + cooldown checks so this stays cheap for the no-op
        # case.
        if deep_hint and self.cloud_brain_router is not None:
            asyncio.create_task(self._cloud_escalate(text, system_instruction=None))
            return

        if self.command_loop is not None and hasattr(self.command_loop, "submit"):
            try:
                asyncio.create_task(self.command_loop.submit(text, priority=priority))
            except Exception:
                logger.exception("RoomAgent: command_loop.submit failed")
        else:
            await participant.send_data(
                "atom.error",
                error="command_loop_unavailable",
                message="Boss, my command loop isn't wired yet -- chat-only mode.",
            )

    async def _cloud_escalate(self, text: str, *, system_instruction: str | None) -> None:
        try:
            result = await self.cloud_brain_router.maybe_escalate(
                text,
                deep_hint=True,
                system_instruction=system_instruction,
            )
        except Exception:
            logger.exception("RoomAgent: cloud_brain_router escalation failed")
            return
        if result is None or not result.text.strip():
            # Cloud router declined or returned empty; fall back to the
            # local command loop so Boss still gets an answer.
            if self.command_loop is not None and hasattr(self.command_loop, "submit"):
                try:
                    await self.command_loop.submit(text)
                except Exception:
                    logger.exception("RoomAgent: fallback command_loop.submit failed")
            return
        emit = getattr(self.bus, "emit_fast", None)
        if callable(emit):
            emit(
                "response_ready",
                text=result.text,
                source="cloud",
                cloud_provider=result.provider,
                cloud_profile=result.profile,
                latency_ms=result.latency_ms,
            )

    async def _handle_say_text(self, _participant: RoomParticipant, data: dict[str, Any]) -> None:
        text = str(data.get("text") or "").strip()
        if not text:
            return
        if self.tts is not None and hasattr(self.tts, "speak"):
            try:
                asyncio.create_task(self.tts.speak(text))
                return
            except Exception:
                logger.exception("RoomAgent: tts.speak failed")
        # Fallback: emit on the bus so whatever TTS is wired picks it up.
        emit = getattr(self.bus, "emit_fast", None)
        if callable(emit):
            emit("response_ready", text=text)

    async def _handle_stt_text(self, _participant: RoomParticipant, data: dict[str, Any]) -> None:
        text = str(data.get("text") or "").strip()
        if not text:
            return
        emit = getattr(self.bus, "emit_long", None) or getattr(self.bus, "emit_fast", None)
        if callable(emit):
            emit("speech_final", text=text, source="room")

    # ── inbound: media frames -------------------------------------

    async def _on_frame(self, participant: RoomParticipant, frame: Frame) -> None:
        cnt_key = frame.kind.name.lower()
        self._latest.counters[cnt_key] = self._latest.counters.get(cnt_key, 0) + 1

        if frame.kind == FrameKind.AUDIO_IN:
            self._latest.audio_in = frame
            # Future: forward into the STT engine. For now we ack track
            # health so the browser can show a green mic indicator.
            if self._latest.counters[cnt_key] % 50 == 0:
                logger.debug(
                    "room agent received %d audio frames from participant=%s",
                    self._latest.counters[cnt_key], participant.id,
                )
        elif frame.kind == FrameKind.VIDEO_IN and self.config.cache_video_frames:
            self._latest.video_in = frame
        elif frame.kind == FrameKind.SCREEN_IN and self.config.cache_video_frames:
            self._latest.screen_in = frame

    # ── lifecycle hooks -------------------------------------------

    async def _on_join(self, participant: RoomParticipant) -> None:
        await participant.send_data(
            "atom.greeting",
            name="ATOM",
            owner="Satyam",
            persona="Friday-class assistant",
            mood=getattr(self.state, "current_mood", None) if self.state else None,
            state=str(getattr(self.state, "current", "")) if self.state else None,
            snapshot=self._snapshot_for_client(),
        )

    async def _on_leave(self, _participant: RoomParticipant) -> None:
        pass

    # ── outbound: bus -> room mirroring ---------------------------

    def _make_mirror(self, bus_event: str, room_event: str) -> Callable[..., Awaitable[None]]:
        async def _mirror(**payload: Any) -> None:
            if bus_event == "partial_response" and not self.config.forward_partial_responses:
                return
            if self.config.log_event_mirrors:
                logger.debug("RoomAgent mirror %s -> %s payload=%s", bus_event, room_event, list(payload.keys()))
            try:
                await self.room.broadcast_data(room_event, **payload)
            except Exception:
                logger.exception("RoomAgent: mirror failed event=%s", room_event)
        _mirror.__name__ = f"_mirror_{bus_event}"
        return _mirror

    # ── snapshot --------------------------------------------------

    def _snapshot_for_client(self) -> dict[str, Any]:
        snap: dict[str, Any] = {
            "room": self.room.snapshot(),
            "atom": {
                "name": "ATOM",
                "owner": "Satyam",
                "owner_alias": "Boss",
            },
            "media_counts": dict(self._latest.counters),
        }
        if self.state is not None:
            try:
                snap["atom"]["state"] = str(self.state.current)
            except Exception:
                pass
            try:
                snap["atom"]["mood"] = getattr(self.state, "current_mood", None)
            except Exception:
                pass
        return snap

    # ── public helpers --------------------------------------------

    def latest_video_jpeg(self) -> bytes | None:
        return self._latest.video_in.payload if self._latest.video_in else None

    def latest_screen_jpeg(self) -> bytes | None:
        return self._latest.screen_in.payload if self._latest.screen_in else None
