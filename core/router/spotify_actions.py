"""
ATOM -- Spotify control via osascript (Phase F1 of the Jarvis-OS plan).

Direct AppleScript binding to the local Spotify desktop client. We use
the application's scripting interface instead of the macOS media keys
so the operation is *transactional*: each call returns True/False
based on whether Spotify actually accepted the verb, and we can read
back the current track for confirmation prompts.

The module is offline, side-effect free until a public function is
called, and safe to import on non-darwin (the public API short-circuits
to a typed failure). All AppleScript fragments live as module-level
string templates so tests can assert the exact script we sent without
spinning up a real osascript subprocess.

Public API::

    is_spotify_running() -> bool
    ensure_spotify_running(*, launch_timeout_s: float = 2.0) -> bool
    play() -> bool
    pause() -> bool
    next_track() -> bool
    previous_track() -> bool
    current_track() -> dict | None
    play_search(query: str, *, kind: str = "track") -> bool

Owner: Satyam
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from typing import Any

logger = logging.getLogger("atom.router.spotify")

_IS_MAC: bool = sys.platform == "darwin"


# ── osascript runner (tests monkeypatch this) ────────────────────────


def _run_osascript(script: str, *, timeout_s: float = 4.0) -> tuple[bool, str]:
    """Run an AppleScript snippet and return ``(ok, stdout)``.

    ``ok`` is False on any failure path: not on darwin, osascript
    missing, non-zero exit, timeout, or the script returning a string
    that starts with ``"missing value"`` (Spotify's idiom for "can't
    answer that right now"). ``stdout`` is the raw trimmed text of the
    subprocess so callers can parse rich responses.

    Tests can monkeypatch this whole function to feed deterministic
    fake responses without touching the real Spotify app.
    """
    if not _IS_MAC:
        return False, "non-darwin"
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        logger.warning("spotify osascript timed out after %.1fs", timeout_s)
        return False, "timeout"
    except FileNotFoundError:
        logger.warning("osascript binary not found")
        return False, "no-osascript"
    except Exception:
        logger.debug("spotify osascript raised", exc_info=True)
        return False, "exception"
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        # Surface the first line so callers can decide whether to retry
        # or fall back -- e.g. "Application isn't running" -> launch.
        return False, stderr.split("\n", 1)[0] if stderr else "rc!=0"
    out = (result.stdout or "").strip()
    if out.startswith("missing value"):
        return False, "missing value"
    return True, out


# ── Process-level helpers ────────────────────────────────────────────


_SCRIPT_IS_RUNNING = (
    'tell application "System Events" to '
    '(name of processes) contains "Spotify"'
)

_SCRIPT_LAUNCH = 'tell application "Spotify" to activate'


def is_spotify_running() -> bool:
    """Return True when the Spotify process is alive."""
    ok, out = _run_osascript(_SCRIPT_IS_RUNNING, timeout_s=1.5)
    if not ok:
        return False
    return out.lower() == "true"


def ensure_spotify_running(*, launch_timeout_s: float = 2.0) -> bool:
    """Activate Spotify if it is not already running and wait briefly.

    Returns True iff Spotify is alive at the end of the call. Polls in
    ~120ms increments so a cold launch on a warm app responds in
    <300ms; gives up after ``launch_timeout_s`` and returns False so
    callers can fall back to a friendlier error message.
    """
    if is_spotify_running():
        return True
    ok, _ = _run_osascript(_SCRIPT_LAUNCH, timeout_s=2.0)
    if not ok:
        return False
    deadline = time.monotonic() + max(0.1, launch_timeout_s)
    while time.monotonic() < deadline:
        if is_spotify_running():
            return True
        time.sleep(0.12)
    return is_spotify_running()


# ── Transport verbs ──────────────────────────────────────────────────


_SCRIPT_PLAY = 'tell application "Spotify" to play'
_SCRIPT_PAUSE = 'tell application "Spotify" to pause'
_SCRIPT_NEXT = 'tell application "Spotify" to next track'
_SCRIPT_PREV = 'tell application "Spotify" to previous track'


def _run_with_launch(script: str, *, timeout_s: float = 3.0) -> bool:
    """Run ``script`` against Spotify, launching it first if needed."""
    if not ensure_spotify_running():
        logger.info("Spotify control: app not available -- aborting %s",
                    script.split('"')[-1])
        return False
    ok, err = _run_osascript(script, timeout_s=timeout_s)
    if not ok:
        logger.info("Spotify control: '%s' failed (%s)", script, err)
    return ok


def play() -> bool:
    """Resume playback (or start the current selection)."""
    return _run_with_launch(_SCRIPT_PLAY)


def pause() -> bool:
    """Pause playback. Returns True even if already paused."""
    return _run_with_launch(_SCRIPT_PAUSE)


def next_track() -> bool:
    """Advance to the next track in the current queue/playlist."""
    return _run_with_launch(_SCRIPT_NEXT)


def previous_track() -> bool:
    """Skip back to the previous track."""
    return _run_with_launch(_SCRIPT_PREV)


# ── Now-playing inspection ───────────────────────────────────────────


_SCRIPT_CURRENT = '''
tell application "Spotify"
    if it is running then
        if player state is playing or player state is paused then
            set t_name to name of current track
            set t_artist to artist of current track
            set t_album to album of current track
            set t_dur to duration of current track
            set t_pos to player position
            set t_state to player state as text
            return t_name & "\\t" & t_artist & "\\t" & t_album & "\\t" & t_dur & "\\t" & t_pos & "\\t" & t_state
        end if
    end if
    return ""
end tell
'''.strip()


def current_track() -> dict[str, Any] | None:
    """Read the current track metadata, or ``None`` if nothing is loaded.

    Returned dict shape::

        {
            "name": str,
            "artist": str,
            "album": str,
            "duration_s": float,    # Spotify reports duration in milliseconds
            "position_s": float,    # player_position is already in seconds
            "state": "playing" | "paused" | "stopped",
        }
    """
    if not is_spotify_running():
        return None
    ok, out = _run_osascript(_SCRIPT_CURRENT, timeout_s=2.5)
    if not ok or not out:
        return None
    parts = out.split("\t")
    if len(parts) < 6:
        return None
    name, artist, album, dur_raw, pos_raw, state_raw = parts[:6]
    try:
        # Spotify reports `duration` in milliseconds (per Apple's
        # Spotify scripting dictionary) and `player position` in
        # seconds as a float.
        dur_ms = float(dur_raw or 0.0)
        pos_s = float(pos_raw or 0.0)
    except ValueError:
        return None
    return {
        "name": name.strip(),
        "artist": artist.strip(),
        "album": album.strip(),
        "duration_s": round(dur_ms / 1000.0, 2),
        "position_s": round(pos_s, 2),
        "state": state_raw.strip().lower() or "stopped",
    }


# ── Search & play (URI form) ─────────────────────────────────────────


# Spotify's `play track <uri>` AppleScript verb accepts a Spotify URI
# (spotify:track:..., spotify:album:..., spotify:search:...). We use
# the search URI so the user can speak free-form queries -- Spotify
# resolves them server-side and starts the top result automatically.
_SCRIPT_PLAY_URI_TEMPLATE = 'tell application "Spotify" to play track "{uri}"'

_VALID_KINDS = frozenset({"track", "album", "artist", "playlist"})


def _build_search_uri(query: str, kind: str) -> str:
    """Construct a ``spotify:search:`` URI for free-form playback.

    ``kind`` is one of ``track``/``album``/``artist``/``playlist``;
    invalid kinds are coerced to ``track``. The query is shell-quoted
    so quotes inside the user's request can't break out of the
    AppleScript string.
    """
    safe_kind = kind if kind in _VALID_KINDS else "track"
    cleaned = (query or "").strip().replace('"', "")
    if not cleaned:
        return ""
    return f"spotify:search:{safe_kind}:{cleaned}"


def play_search(query: str, *, kind: str = "track") -> bool:
    """Play the first ``kind`` matching ``query``.

    Returns False on empty/dangerous input, when Spotify won't launch,
    or when the AppleScript verb itself rejects the URI.
    """
    uri = _build_search_uri(query, kind)
    if not uri:
        return False
    if not ensure_spotify_running():
        return False
    # Wrap the URI safely; AppleScript needs literal double quotes,
    # so we escape any embedded quotes by stripping them above.
    script = _SCRIPT_PLAY_URI_TEMPLATE.format(uri=uri)
    ok, err = _run_osascript(script, timeout_s=4.0)
    if not ok:
        logger.info("Spotify play_search('%s', %s) failed: %s",
                    query, kind, err)
    return ok


# ── Diagnostics surface ──────────────────────────────────────────────


def diagnostics() -> dict[str, Any]:
    """Snapshot suitable for /diagnostics or status logs."""
    if not _IS_MAC:
        return {"available": False, "reason": "non-darwin"}
    running = is_spotify_running()
    payload: dict[str, Any] = {"available": True, "running": running}
    if running:
        track = current_track()
        if track:
            payload["track"] = track
    return payload


__all__ = [
    "is_spotify_running",
    "ensure_spotify_running",
    "play",
    "pause",
    "next_track",
    "previous_track",
    "current_track",
    "play_search",
    "diagnostics",
]
