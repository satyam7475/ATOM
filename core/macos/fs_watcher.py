"""
ATOM -- macOS FSEvents File System Watcher.

Kernel-level file system monitoring via the FSEvents API.  Near-zero CPU
cost because the kernel pushes events — no polling.  This is the same
mechanism Spotlight uses.

Usage:
    from core.macos.fs_watcher_config import fs_watcher_settings

    watcher = FSWatcher(bus)
    watcher.watch(fs_watcher_settings(config)["paths"])
    watcher.start()
    # ... events emitted on the bus as "fs_event" ...
    watcher.stop()

Events emitted on the bus:
    fs_event: {
        "path": "/Users/.../file.txt",
        "event": "created" | "modified" | "removed" | "renamed",
        "is_dir": bool,
    }

Requires: macOS, pyobjc-framework-FSEvents

Owner: Satyam
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import TYPE_CHECKING, Any

logger = logging.getLogger("atom.fs_watcher")

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus

_HAS_FSEVENTS = False
_FSEvents: Any = None
_CF: Any = None

try:
    import FSEvents as _FSEvents  # type: ignore[import-untyped]
    from CoreFoundation import (  # type: ignore[import-untyped]
        CFRunLoopGetCurrent,
        CFRunLoopRun,
        CFRunLoopStop,
    )
    _CF = True
    _HAS_FSEVENTS = True
except ImportError:
    pass

# Map FSEvent flags to human-readable event types
_FLAG_CREATED = 0x00000100   # kFSEventStreamEventFlagItemCreated
_FLAG_REMOVED = 0x00000200   # kFSEventStreamEventFlagItemRemoved
_FLAG_RENAMED = 0x00000800   # kFSEventStreamEventFlagItemRenamed
_FLAG_MODIFIED = 0x00001000  # kFSEventStreamEventFlagItemModified
_FLAG_IS_DIR = 0x00020000    # kFSEventStreamEventFlagItemIsDir
_FLAG_IS_FILE = 0x00010000   # kFSEventStreamEventFlagItemIsFile

# Directories to ignore (noisy system writes)
_IGNORE_PATTERNS = {
    ".DS_Store", ".Spotlight-V100", ".fseventsd", ".Trashes",
    "__pycache__", ".git", "node_modules", ".venv",
}


def _classify_event(flags: int) -> str:
    """Convert FSEvent flags to a simple event name."""
    if flags & _FLAG_CREATED:
        return "created"
    if flags & _FLAG_REMOVED:
        return "removed"
    if flags & _FLAG_RENAMED:
        return "renamed"
    if flags & _FLAG_MODIFIED:
        return "modified"
    return "changed"


def _should_ignore(path: str) -> bool:
    """Skip noisy system files."""
    basename = os.path.basename(path)
    return basename in _IGNORE_PATTERNS


class FSWatcher:
    """macOS FSEvents file system watcher.

    Runs a CFRunLoop in a dedicated daemon thread. Events are
    forwarded to the ATOM event bus as 'fs_event' messages.
    """

    def __init__(self, bus: AsyncEventBus | None = None) -> None:
        self._bus = bus
        self._paths: list[str] = []
        self._stream: Any = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._runloop_ref: Any = None
        self._event_count = 0
        self._available = _HAS_FSEVENTS and sys.platform == "darwin"

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def event_count(self) -> int:
        return self._event_count

    def watch(self, paths: list[str]) -> None:
        """Set directories to monitor (expands ~ and resolves symlinks)."""
        self._paths = [
            os.path.realpath(os.path.expanduser(p)) for p in paths
        ]
        logger.debug("FSWatcher paths: %s", self._paths)

    def start(self) -> bool:
        """Start watching in a background thread."""
        if not self._available:
            logger.info(
                "FSEvents not available. Install pyobjc-framework-FSEvents."
            )
            return False

        if self._running:
            return True

        if not self._paths:
            logger.warning("FSWatcher: no paths configured, call watch() first")
            return False

        valid_paths = [p for p in self._paths if os.path.isdir(p)]
        if not valid_paths:
            logger.warning("FSWatcher: none of the watched paths exist: %s",
                           self._paths)
            return False

        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(valid_paths,),
            name="fs_watcher",
            daemon=True,
        )
        self._thread.start()
        logger.info("FSWatcher started — watching %d directories", len(valid_paths))
        return True

    def stop(self) -> None:
        """Stop the watcher thread."""
        self._running = False
        if self._runloop_ref is not None:
            try:
                CFRunLoopStop(self._runloop_ref)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        logger.info("FSWatcher stopped (%d events total)", self._event_count)

    def _run_loop(self, paths: list[str]) -> None:
        """Thread target: create FSEventStream and run CFRunLoop."""
        try:
            context = None

            def _callback(
                stream_ref: Any,
                client_info: Any,
                num_events: int,
                event_paths: list,
                event_flags: list,
                event_ids: list,
            ) -> None:
                for i in range(num_events):
                    path = event_paths[i]
                    flags = event_flags[i]

                    if _should_ignore(path):
                        continue

                    event_type = _classify_event(flags)
                    is_dir = bool(flags & _FLAG_IS_DIR)

                    self._event_count += 1
                    logger.debug(
                        "fs_event: %s %s%s",
                        event_type, path, "/" if is_dir else "",
                    )

                    if self._bus is not None:
                        self._bus.emit("fs_event", **{
                            "path": path,
                            "event": event_type,
                            "is_dir": is_dir,
                        })

            stream = _FSEvents.FSEventStreamCreate(
                None,          # allocator
                _callback,     # callback
                context,       # context
                paths,         # pathsToWatch
                _FSEvents.kFSEventStreamEventIdSinceNow,
                1.0,           # latency (seconds) — batch events for efficiency
                (
                    _FSEvents.kFSEventStreamCreateFlagFileEvents
                    | _FSEvents.kFSEventStreamCreateFlagNoDefer
                ),
            )

            self._runloop_ref = CFRunLoopGetCurrent()
            _FSEvents.FSEventStreamScheduleWithRunLoop(
                stream,
                self._runloop_ref,
                _FSEvents.kCFRunLoopDefaultMode,
            )
            _FSEvents.FSEventStreamStart(stream)
            self._stream = stream

            CFRunLoopRun()

            _FSEvents.FSEventStreamStop(stream)
            _FSEvents.FSEventStreamInvalidate(stream)
            _FSEvents.FSEventStreamRelease(stream)
            self._stream = None

        except Exception:
            logger.exception("FSWatcher run loop error")
            self._running = False

    def shutdown(self) -> None:
        """Full shutdown."""
        self.stop()
