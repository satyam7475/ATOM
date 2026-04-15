"""
ATOM -- Real-Time System State Engine.

Lightweight, sub-second system state snapshot for the command pipeline.
Unlike SystemScanner (deep, every 300s) or SystemWatcher (10s polling),
this engine maintains a cached snapshot refreshed every ~500ms that
the CommandLoop injects before every command.

Tracked state:
  - Active app name + window title (macOS NSWorkspace / Quartz)
  - Media playback status (Now Playing)
  - Battery level + plugged state
  - CPU / RAM pressure (cached psutil)
  - Current task context hint (from timeline)

The observer loop runs in a background asyncio task and emits
``system_state_update`` only when the state actually changes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import psutil

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus

logger = logging.getLogger("atom.system_state")


_APP_HISTORY_MAX = 5
_BROWSER_APPS = frozenset({
    "Safari", "Google Chrome", "Firefox", "Arc", "Brave Browser",
    "Microsoft Edge", "Opera", "Chromium", "Vivaldi",
})


@dataclass
class SystemSnapshot:
    """Lightweight point-in-time system state."""
    active_app: str = ""
    window_title: str = ""
    previous_app: str = ""
    app_history: list[str] = field(default_factory=list)
    media_playing: bool = False
    media_app: str = ""
    battery_pct: int = 100
    battery_plugged: bool = True
    cpu_pct: float = 0.0
    ram_pct: float = 0.0
    clipboard_type: str = ""
    browser_tab_count: int = 0
    timestamp: float = field(default_factory=time.time)

    def context_string(self) -> str:
        """Compact string for LLM context injection."""
        parts = []
        if self.active_app:
            parts.append(f"app:{self.active_app}")
        if self.window_title and self.window_title != self.active_app:
            title = self.window_title[:80]
            parts.append(f"window:{title}")
        if self.media_playing:
            parts.append(f"media:{self.media_app or 'playing'}")
        if not self.battery_plugged:
            parts.append(f"battery:{self.battery_pct}%")
        if self.cpu_pct > 60:
            parts.append(f"cpu:{self.cpu_pct:.0f}%")
        if self.ram_pct > 75:
            parts.append(f"ram:{self.ram_pct:.0f}%")
        if self.browser_tab_count > 0:
            parts.append(f"tabs:{self.browser_tab_count}")
        if self.clipboard_type:
            parts.append(f"clipboard:{self.clipboard_type}")
        return " | ".join(parts) if parts else ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "active_app": self.active_app,
            "window_title": self.window_title,
            "previous_app": self.previous_app,
            "app_history": list(self.app_history),
            "media_playing": self.media_playing,
            "media_app": self.media_app,
            "battery_pct": self.battery_pct,
            "battery_plugged": self.battery_plugged,
            "cpu_pct": self.cpu_pct,
            "ram_pct": self.ram_pct,
            "clipboard_type": self.clipboard_type,
            "browser_tab_count": self.browser_tab_count,
            "timestamp": self.timestamp,
        }


def _get_clipboard_type() -> str:
    """Return a label for the current clipboard content type."""
    try:
        from AppKit import NSPasteboard
        pb = NSPasteboard.generalPasteboard()
        types = pb.types()
        if not types:
            return ""
        t0 = str(types[0])
        if "image" in t0.lower() or "png" in t0.lower() or "tiff" in t0.lower():
            return "image"
        if "rtf" in t0.lower():
            return "rich_text"
        if "string" in t0.lower() or "text" in t0.lower():
            return "text"
        if "url" in t0.lower():
            return "url"
        if "file" in t0.lower():
            return "file"
        return "other"
    except Exception:
        return ""


def _get_browser_tab_count(app_name: str) -> int:
    """Return the number of open tabs for the active browser via AppleScript."""
    if app_name not in _BROWSER_APPS:
        return 0
    script = {
        "Safari": 'tell application "Safari" to count tabs of front window',
        "Google Chrome": 'tell application "Google Chrome" to count tabs of front window',
        "Firefox": 'tell application "Firefox" to count tabs of front window',
        "Arc": 'tell application "Arc" to count tabs of front window',
        "Brave Browser": 'tell application "Brave Browser" to count tabs of front window',
        "Microsoft Edge": 'tell application "Microsoft Edge" to count tabs of front window',
    }.get(app_name)
    if not script:
        return 0
    try:
        import subprocess
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=2,
        )
        return int(result.stdout.strip())
    except Exception:
        return 0


def _get_foreground_app() -> tuple[str, str]:
    """Return (app_name, window_title) using macOS APIs."""
    try:
        from context.context_darwin import get_context as _darwin_ctx
        ctx = _darwin_ctx()
        return (
            ctx.get("active_app", "") or "",
            ctx.get("active_window_title", "") or "",
        )
    except Exception:
        pass

    try:
        import subprocess
        result = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to get name of first process whose frontmost is true'],
            capture_output=True, text=True, timeout=2,
        )
        app = result.stdout.strip()
        return app, ""
    except Exception:
        return "", ""


def _get_media_state() -> tuple[bool, str]:
    """Check if media is currently playing on macOS."""
    try:
        import subprocess
        result = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to get name of every process whose background only is false'],
            capture_output=True, text=True, timeout=2,
        )
        apps = result.stdout.strip()
        media_apps = {"Music", "Spotify", "VLC", "QuickTime Player", "IINA"}
        for app in media_apps:
            if app in apps:
                return True, app
    except Exception:
        pass
    return False, ""


class SystemStateEngine:
    """Real-time system state with sub-second refresh."""

    def __init__(
        self,
        bus: "AsyncEventBus | None" = None,
        *,
        poll_interval_s: float = 0.5,
    ) -> None:
        self._bus = bus
        self._poll_interval = max(0.2, float(poll_interval_s))
        self._snapshot = SystemSnapshot()
        self._previous_app: str = ""
        self._task: asyncio.Task | None = None
        self._shutdown = asyncio.Event()
        self._last_emit_hash: int = 0

    @property
    def snapshot(self) -> SystemSnapshot:
        return self._snapshot

    def get_context(self) -> dict[str, Any]:
        """Return the current snapshot as a dict for intent/router injection."""
        return self._snapshot.as_dict()

    def get_context_string(self) -> str:
        """Compact context string for LLM prompt injection."""
        return self._snapshot.context_string()

    def _capture(self) -> SystemSnapshot:
        """Take a fresh snapshot of system state."""
        app, title = _get_foreground_app()
        media_playing, media_app = _get_media_state()

        try:
            bat = psutil.sensors_battery()
            battery_pct = int(bat.percent) if bat else 100
            battery_plugged = bool(bat.power_plugged) if bat else True
        except Exception:
            battery_pct = 100
            battery_plugged = True

        try:
            cpu_pct = psutil.cpu_percent(interval=0)
            ram_pct = psutil.virtual_memory().percent
        except Exception:
            cpu_pct = 0.0
            ram_pct = 0.0

        prev = self._previous_app
        history = list(self._snapshot.app_history)
        if app and app != self._snapshot.active_app:
            prev = self._snapshot.active_app
            self._previous_app = prev
            if prev and (not history or history[-1] != prev):
                history.append(prev)
                if len(history) > _APP_HISTORY_MAX:
                    history = history[-_APP_HISTORY_MAX:]

        clipboard_type = _get_clipboard_type()
        tab_count = _get_browser_tab_count(app) if app in _BROWSER_APPS else 0

        return SystemSnapshot(
            active_app=app,
            window_title=title,
            previous_app=prev,
            app_history=history,
            media_playing=media_playing,
            media_app=media_app,
            battery_pct=battery_pct,
            battery_plugged=battery_plugged,
            cpu_pct=cpu_pct,
            ram_pct=ram_pct,
            clipboard_type=clipboard_type,
            browser_tab_count=tab_count,
        )

    def _state_hash(self, snap: SystemSnapshot) -> int:
        return hash((
            snap.active_app,
            snap.window_title,
            snap.media_playing,
            snap.battery_pct,
            snap.battery_plugged,
            round(snap.cpu_pct, -1),
            round(snap.ram_pct, -1),
            snap.browser_tab_count,
            snap.clipboard_type,
        ))

    def start(self) -> None:
        """Start the background observer loop."""
        if self._task is not None and not self._task.done():
            return
        self._shutdown.clear()
        self._snapshot = self._capture()
        self._task = asyncio.create_task(self._observer_loop())
        logger.info(
            "SystemStateEngine started (poll=%.1fs, app=%s)",
            self._poll_interval,
            self._snapshot.active_app,
        )

    def stop(self) -> None:
        self._shutdown.set()
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None

    async def _observer_loop(self) -> None:
        """Continuous observer -- captures state and emits on change."""
        while not self._shutdown.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(), timeout=self._poll_interval,
                )
                break
            except asyncio.TimeoutError:
                pass

            try:
                loop = asyncio.get_running_loop()
                snap = await loop.run_in_executor(None, self._capture)
                h = self._state_hash(snap)

                if h != self._last_emit_hash:
                    old_app = self._snapshot.active_app
                    self._snapshot = snap
                    self._last_emit_hash = h

                    if self._bus is not None:
                        self._bus.emit_fast(
                            "system_state_update",
                            snapshot=snap.as_dict(),
                            changed_app=(snap.active_app != old_app),
                        )
                else:
                    self._snapshot = snap

            except asyncio.CancelledError:
                break
            except Exception:
                logger.debug("SystemStateEngine capture error", exc_info=True)
                await asyncio.sleep(1.0)

    def get_live_context(self) -> dict[str, Any]:
        """Structured live context for direct prompt building and fusion."""
        s = self._snapshot
        from datetime import datetime
        now = datetime.now()
        return {
            "active_app": s.active_app,
            "window_title": s.window_title,
            "previous_app": s.previous_app,
            "app_history": list(s.app_history),
            "media": {"playing": s.media_playing, "app": s.media_app},
            "power": {
                "battery_pct": s.battery_pct,
                "plugged": s.battery_plugged,
            },
            "resources": {"cpu_pct": s.cpu_pct, "ram_pct": s.ram_pct},
            "clipboard_type": s.clipboard_type,
            "browser_tab_count": s.browser_tab_count,
            "time": {
                "hour": now.hour,
                "weekday": now.strftime("%A"),
                "time_of_day": (
                    "morning" if 5 <= now.hour < 12
                    else "afternoon" if 12 <= now.hour < 17
                    else "evening" if 17 <= now.hour < 21
                    else "night"
                ),
            },
        }

    def context_hints(self, query: str) -> list[str]:
        """Natural-language hints relevant to the user's query.

        Examples:
          "Chrome is already open with 4 tabs."
          "You were just in VS Code."
        """
        hints: list[str] = []
        s = self._snapshot
        q = query.lower()

        app_keywords = {
            "chrome": "Google Chrome", "safari": "Safari",
            "firefox": "Firefox", "arc": "Arc",
            "vscode": "Code", "vs code": "Code", "visual studio": "Code",
            "terminal": "Terminal", "iterm": "iTerm2",
            "finder": "Finder", "slack": "Slack", "teams": "Teams",
            "spotify": "Spotify", "music": "Music",
        }

        for keyword, app_name in app_keywords.items():
            if keyword not in q:
                continue
            if s.active_app and app_name.lower() in s.active_app.lower():
                hint = f"{s.active_app} is already open"
                if s.browser_tab_count > 0:
                    hint += f" with {s.browser_tab_count} tab{'s' if s.browser_tab_count != 1 else ''}"
                hints.append(hint + ".")
            elif s.previous_app and app_name.lower() in s.previous_app.lower():
                hints.append(f"You were just using {s.previous_app}.")
            elif any(app_name.lower() in h.lower() for h in s.app_history):
                hints.append(f"{app_name} was recently used this session.")

        if "clipboard" in q or "paste" in q:
            if s.clipboard_type:
                hints.append(f"Your clipboard currently holds {s.clipboard_type} content.")

        if "battery" in q or "power" in q or "charge" in q:
            status = "plugged in" if s.battery_plugged else "on battery"
            hints.append(f"Battery is at {s.battery_pct}% ({status}).")

        return hints

    def resolve_reference(self, text: str) -> dict[str, str]:
        """Resolve deictic references like 'this app', 'close that'.

        Returns hints the intent engine / router can use:
          {"this_app": "Safari", "previous_app": "VSCode"}
        """
        hints: dict[str, str] = {}
        if self._snapshot.active_app:
            hints["this_app"] = self._snapshot.active_app
        if self._snapshot.previous_app:
            hints["previous_app"] = self._snapshot.previous_app
        if self._snapshot.media_app:
            hints["media_app"] = self._snapshot.media_app
        return hints


__all__ = ["SystemStateEngine", "SystemSnapshot"]
