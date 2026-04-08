"""
ATOM -- Media Watcher (Real-Time Media Awareness).

macOS native: queries Now Playing info via AppleScript for known apps
(Music, Spotify) and a generic system-level media query. Zero external
dependencies — uses osascript which ships with every Mac.

On Windows: falls back to winsdk (legacy path, unchanged).

This allows ATOM to know what song/video is playing so you can
ask "Do you like this song?" without specifying the title.

Owner: Satyam
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from dataclasses import dataclass

logger = logging.getLogger("atom.media_watcher")

_IS_MACOS = sys.platform == "darwin"


@dataclass
class MediaInfo:
    title: str = ""
    artist: str = ""
    album: str = ""
    app_name: str = ""
    is_playing: bool = False
    duration: float = 0.0
    position: float = 0.0

    @property
    def is_active(self) -> bool:
        return bool(self.title and self.is_playing)

    def summary(self) -> str:
        if not self.is_active:
            return "No media playing"
        artist_str = f" by {self.artist}" if self.artist else ""
        album_str = f" from '{self.album}'" if self.album else ""
        app_str = f" (on {self.app_name})" if self.app_name else ""
        return f"Playing: '{self.title}'{artist_str}{album_str}{app_str}"


def _osascript(script: str, timeout: float = 3.0) -> str:
    """Run AppleScript and return stdout, empty string on failure."""
    try:
        r = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.stdout.strip()
    except Exception:
        return ""


def _query_spotify() -> MediaInfo | None:
    """Query Spotify via AppleScript."""
    check = _osascript(
        'tell application "System Events" to '
        '(name of processes) contains "Spotify"'
    )
    if check != "true":
        return None
    raw = _osascript(
        'tell application "Spotify"\n'
        '  if player state is playing then\n'
        '    set t to name of current track\n'
        '    set a to artist of current track\n'
        '    set al to album of current track\n'
        '    set d to duration of current track\n'
        '    set p to player position\n'
        '    return t & "|" & a & "|" & al & "|" & (d / 1000) & "|" & p\n'
        '  else if player state is paused then\n'
        '    set t to name of current track\n'
        '    set a to artist of current track\n'
        '    set al to album of current track\n'
        '    return t & "|" & a & "|" & al & "|paused"\n'
        '  else\n'
        '    return ""\n'
        '  end if\n'
        'end tell'
    )
    if not raw:
        return None
    parts = raw.split("|")
    if len(parts) >= 4 and parts[3] == "paused":
        return MediaInfo(
            title=parts[0], artist=parts[1], album=parts[2],
            app_name="Spotify", is_playing=False,
        )
    if len(parts) >= 5:
        return MediaInfo(
            title=parts[0], artist=parts[1], album=parts[2],
            app_name="Spotify", is_playing=True,
            duration=float(parts[3] or 0), position=float(parts[4] or 0),
        )
    return None


def _query_music() -> MediaInfo | None:
    """Query Apple Music via AppleScript."""
    check = _osascript(
        'tell application "System Events" to '
        '(name of processes) contains "Music"'
    )
    if check != "true":
        return None
    raw = _osascript(
        'tell application "Music"\n'
        '  if player state is playing then\n'
        '    set t to name of current track\n'
        '    set a to artist of current track\n'
        '    set al to album of current track\n'
        '    set d to duration of current track\n'
        '    set p to player position\n'
        '    return t & "|" & a & "|" & al & "|" & d & "|" & p\n'
        '  else if player state is paused then\n'
        '    set t to name of current track\n'
        '    set a to artist of current track\n'
        '    set al to album of current track\n'
        '    return t & "|" & a & "|" & al & "|paused"\n'
        '  else\n'
        '    return ""\n'
        '  end if\n'
        'end tell'
    )
    if not raw:
        return None
    parts = raw.split("|")
    if len(parts) >= 4 and parts[3] == "paused":
        return MediaInfo(
            title=parts[0], artist=parts[1], album=parts[2],
            app_name="Music", is_playing=False,
        )
    if len(parts) >= 5:
        return MediaInfo(
            title=parts[0], artist=parts[1], album=parts[2],
            app_name="Music", is_playing=True,
            duration=float(parts[3] or 0), position=float(parts[4] or 0),
        )
    return None


def _query_generic_browser() -> MediaInfo | None:
    """Detect media playing in browsers via window title heuristics."""
    raw = _osascript(
        'tell application "System Events"\n'
        '  set appList to {"Google Chrome", "Safari", "Firefox", "Arc"}\n'
        '  repeat with appName in appList\n'
        '    if (name of processes) contains appName then\n'
        '      try\n'
        '        tell process appName\n'
        '          set winTitle to name of front window\n'
        '        end tell\n'
        '        if winTitle contains "YouTube" or '
        'winTitle contains "SoundCloud" or '
        'winTitle contains "Bandcamp" then\n'
        '          return winTitle & "|" & appName\n'
        '        end if\n'
        '      end try\n'
        '    end if\n'
        '  end repeat\n'
        '  return ""\n'
        'end tell'
    )
    if not raw or "|" not in raw:
        return None
    title, app = raw.rsplit("|", 1)
    title = title.replace(" - YouTube", "").replace(" — YouTube", "").strip()
    return MediaInfo(title=title, app_name=app.strip(), is_playing=True)


def _poll_macos() -> MediaInfo:
    """Poll all macOS media sources, return best match."""
    for fn in (_query_spotify, _query_music, _query_generic_browser):
        try:
            info = fn()
            if info and info.title:
                return info
        except Exception:
            continue
    return MediaInfo()


class MediaWatcher:
    """Tracks currently playing media. macOS native on darwin, winsdk on Windows."""

    __slots__ = ("_current_media", "_running", "_task")

    def __init__(self) -> None:
        self._current_media = MediaInfo()
        self._running = False
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._watch_loop())
        logger.info("Media Watcher started (%s)", "macOS native" if _IS_MACOS else "winsdk")

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("Media Watcher stopped")

    @property
    def current_media(self) -> MediaInfo:
        return self._current_media

    async def _watch_loop(self) -> None:
        if _IS_MACOS:
            await self._watch_macos()
        else:
            await self._watch_windows()

    async def _watch_macos(self) -> None:
        """Poll macOS media sources via AppleScript every 3 seconds."""
        loop = asyncio.get_running_loop()
        while self._running:
            try:
                info = await loop.run_in_executor(None, _poll_macos)
                self._current_media = info
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("Media watcher error: %s", exc)
            await asyncio.sleep(3)

    async def _watch_windows(self) -> None:
        """Legacy Windows path via winsdk."""
        try:
            from winsdk.windows.media.control import (
                GlobalSystemMediaTransportControlsSessionManager,
            )
        except ImportError:
            logger.error(
                "winsdk not installed. Media awareness disabled. "
                "Run: pip install winsdk"
            )
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
                playback_info = session.get_playback_info()
                is_playing = playback_info and playback_info.playback_status == 4
                properties = await session.try_get_media_properties_async()
                if properties:
                    self._current_media = MediaInfo(
                        title=properties.title or "",
                        artist=properties.artist or "",
                        app_name=session.source_app_user_model_id or "",
                        is_playing=is_playing,
                    )
                else:
                    self._current_media = MediaInfo(is_playing=is_playing)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("Media watcher error: %s", exc)
            await asyncio.sleep(2)


media_watcher = MediaWatcher()
