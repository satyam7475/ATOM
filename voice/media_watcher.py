"""
ATOM -- Media Watcher (Real-Time Media Awareness).

Uses Windows SDK (winsdk) to track the currently playing media
(Spotify, YouTube in Chrome, local media players).
This allows ATOM to know what song/video is playing so you can
ask "Do you like this song?" without specifying the title.

Owner: Satyam
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger("atom.media_watcher")


@dataclass
class MediaInfo:
    title: str = ""
    artist: str = ""
    app_name: str = ""
    is_playing: bool = False
    
    @property
    def is_active(self) -> bool:
        return bool(self.title and self.is_playing)
        
    def summary(self) -> str:
        if not self.is_active:
            return "No media playing"
        artist_str = f" by {self.artist}" if self.artist else ""
        app_str = f" (on {self.app_name})" if self.app_name else ""
        return f"Playing: '{self.title}'{artist_str}{app_str}"


class MediaWatcher:
    """Tracks currently playing media on Windows."""

    __slots__ = ("_current_media", "_running", "_task")

    def __init__(self) -> None:
        self._current_media = MediaInfo()
        self._running = False
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._watch_loop())
        logger.info("Media Watcher started")

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("Media Watcher stopped")

    @property
    def current_media(self) -> MediaInfo:
        return self._current_media

    async def _watch_loop(self) -> None:
        """Periodically poll the Windows media controls."""
        try:
            from winsdk.windows.media.control import GlobalSystemMediaTransportControlsSessionManager
        except ImportError:
            logger.error("winsdk not installed. Media awareness disabled. Run: pip install winsdk")
            return

        while self._running:
            try:
                manager = await GlobalSystemMediaTransportControlsSessionManager.request_async()
                if not manager:
                    await asyncio.sleep(5)
                    continue
                    
                session = manager.get_current_session()
                if session is None:
                    self._current_media = MediaInfo()
                    await asyncio.sleep(2)
                    continue

                # Get playback state
                playback_info = session.get_playback_info()
                # 4 = Playing, 5 = Paused
                is_playing = playback_info and playback_info.playback_status == 4

                # Get media properties
                properties = await session.try_get_media_properties_async()
                
                if properties:
                    self._current_media = MediaInfo(
                        title=properties.title or "",
                        artist=properties.artist or "",
                        app_name=session.source_app_user_model_id or "",
                        is_playing=is_playing
                    )
                else:
                    self._current_media = MediaInfo(is_playing=is_playing)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Media watcher error: %s", e)
                
            await asyncio.sleep(2)  # Poll every 2 seconds

# Global singleton
media_watcher = MediaWatcher()
