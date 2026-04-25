"""ATOM realtime room: LiveKit-style multi-channel link to browser/mobile clients.

The package ships:

* :mod:`core.realtime.atom_room` -- WebSocket room server with audio,
  video, screen-share and structured-data channels multiplexed over a
  single connection per participant.
* :mod:`core.realtime.room_agent` -- bridges room events into the
  existing :class:`core.command_loop.CommandLoop`, voice pipeline and
  AsyncEventBus so a browser tab feels like a local mic/screen.

Both modules are deliberately framework-light: only :mod:`aiohttp`
is required (already in requirements.txt for the web dashboard).
"""

from core.realtime.atom_room import (
    AtomRoom,
    AtomRoomServer,
    Frame,
    FrameKind,
    RoomConfig,
    RoomParticipant,
    RoomTrack,
)
from core.realtime.room_agent import RoomAgent, RoomAgentConfig

__all__ = [
    "AtomRoom",
    "AtomRoomServer",
    "Frame",
    "FrameKind",
    "RoomAgent",
    "RoomAgentConfig",
    "RoomConfig",
    "RoomParticipant",
    "RoomTrack",
]
