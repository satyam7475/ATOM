"""
ATOM -- Atomic Persistence Manager.

Replaces raw JSON file writes across all modules with a crash-safe
atomic persistence layer.

Problems this solves:
  1. CORRUPTION: Previous modules wrote JSON directly -- a crash mid-write
     could leave a truncated/invalid file, losing all data.
  2. CONCURRENCY: Multiple modules wrote to logs/ without coordination.
  3. PERFORMANCE: Full file rewrite on every persist() call.

Solution:
  - Atomic writes: write to temp → fsync → rename (rename is atomic on
    both NTFS and POSIX).
  - Debounced persistence: batch writes every N seconds instead of on
    every single change.
  - Central coordination: all modules register with PersistenceManager
    instead of doing their own file I/O.

Usage:
    pm = PersistenceManager()
    pm.register("second_brain", Path("logs/second_brain.json"))
    pm.save("second_brain", {"facts": [...], "prefs": {...}})
    data = pm.load("second_brain")

Owner: Satyam
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom.persistence")


class PersistenceManager:
    """Crash-safe, debounced persistence for all ATOM modules.

    Features:
        - Atomic writes (temp file → fsync → rename)
        - Debounced batch persistence (configurable interval)
        - Thread-safe (can be called from sync and async contexts)
        - Auto-backup on first write of each session
    """

    __slots__ = (
        "_stores", "_dirty", "_lock", "_debounce_interval",
        "_debounce_task", "_running", "_write_count", "_backup_done",
    )

    def __init__(self, debounce_seconds: float = 5.0) -> None:
        self._stores: dict[str, _StoreEntry] = {}
        self._dirty: set[str] = set()
        self._lock = threading.Lock()
        self._debounce_interval = debounce_seconds
        self._debounce_task: asyncio.Task | None = None
        self._running = False
        self._write_count = 0
        self._backup_done: set[str] = set()

    def register(self, key: str, path: Path) -> None:
        """Register a persistence store.

        Args:
            key: Unique identifier (e.g. 'second_brain', 'goals')
            path: File path to persist to
        """
        with self._lock:
            self._stores[key] = _StoreEntry(path=path, data=None)
        logger.debug("Registered persistence store: %s -> %s", key, path)

    def save(self, key: str, data: Any) -> None:
        """Mark data for persistence. The actual write is debounced.

        Thread-safe. Can be called from sync or async context.
        """
        with self._lock:
            if key not in self._stores:
                logger.warning("Unknown persistence key: %s", key)
                return
            self._stores[key].data = data
            self._stores[key].last_modified = time.time()
            self._dirty.add(key)

    def save_now(self, key: str, data: Any) -> None:
        """Immediately persist data (bypasses debounce).

        Use for critical data that must survive an immediate crash.
        """
        with self._lock:
            if key not in self._stores:
                logger.warning("Unknown persistence key: %s", key)
                return
            self._stores[key].data = data
            self._stores[key].last_modified = time.time()

        self._atomic_write(key)

    def load(self, key: str) -> Any | None:
        """Load data from disk for a registered store.

        Returns None if the file doesn't exist or is corrupted.
        """
        with self._lock:
            entry = self._stores.get(key)
            if entry is None:
                return None

        path = entry.path
        if not path.exists():
            return None

        try:
            text = path.read_text(encoding="utf-8")
            data = json.loads(text)
            with self._lock:
                entry.data = data
            return data
        except json.JSONDecodeError:
            logger.error("Corrupted JSON in %s, attempting backup recovery", path)
            return self._try_backup_recovery(key, path)
        except Exception:
            logger.error("Failed to load %s", path, exc_info=True)
            return None

    def start(self) -> None:
        """Start the debounced persistence background task."""
        if self._running:
            return
        self._running = True
        try:
            loop = asyncio.get_running_loop()
            self._debounce_task = loop.create_task(self._debounce_loop())
        except RuntimeError:
            # No event loop -- caller must call flush_all() manually
            pass
        logger.info(
            "PersistenceManager started (debounce: %.1fs, stores: %d)",
            self._debounce_interval, len(self._stores),
        )

    def stop(self) -> None:
        """Stop debouncing and flush all dirty stores."""
        self._running = False
        if self._debounce_task is not None:
            self._debounce_task.cancel()
            self._debounce_task = None
        self.flush_all()

    def flush_all(self) -> None:
        """Immediately persist all dirty stores."""
        with self._lock:
            dirty_keys = list(self._dirty)
            self._dirty.clear()

        for key in dirty_keys:
            self._atomic_write(key)

    def _atomic_write(self, key: str) -> None:
        """Atomic write: temp file → fsync → rename.

        rename() is atomic on both NTFS (Windows) and POSIX (Linux/Mac).
        """
        with self._lock:
            entry = self._stores.get(key)
            if entry is None or entry.data is None:
                return
            data = entry.data
            path = entry.path
            self._dirty.discard(key)

        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            # Create backup on first write of session
            if key not in self._backup_done and path.exists():
                backup_path = path.with_suffix(".json.bak")
                try:
                    backup_path.write_bytes(path.read_bytes())
                    self._backup_done.add(key)
                except Exception:
                    pass  # Best-effort backup

            # Write to temp file in same directory (same filesystem for rename)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent),
                prefix=f".{path.stem}_",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=str)
                    f.flush()
                    os.fsync(f.fileno())

                # Atomic rename (overwrites existing file)
                os.replace(tmp_path, str(path))
                self._write_count += 1

                logger.debug(
                    "Atomic write: %s (%d total writes)", key, self._write_count
                )
            except Exception:
                # Clean up temp file on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

        except Exception:
            logger.error("Failed to persist %s", key, exc_info=True)

    def _try_backup_recovery(self, key: str, path: Path) -> Any | None:
        """Attempt to recover from a .bak file if the main file is corrupted."""
        backup_path = path.with_suffix(".json.bak")
        if backup_path.exists():
            try:
                data = json.loads(backup_path.read_text(encoding="utf-8"))
                logger.warning("Recovered %s from backup", key)
                # Restore the backup as the main file
                self.save_now(key, data)
                return data
            except Exception:
                logger.error("Backup recovery also failed for %s", key)
        return None

    async def _debounce_loop(self) -> None:
        """Background loop that flushes dirty stores periodically."""
        while self._running:
            try:
                await asyncio.sleep(self._debounce_interval)
                if not self._running:
                    break

                with self._lock:
                    dirty_keys = list(self._dirty)
                    self._dirty.clear()

                for key in dirty_keys:
                    self._atomic_write(key)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.error("Debounce loop error", exc_info=True)

    def get_stats(self) -> dict:
        """Return persistence statistics."""
        with self._lock:
            return {
                "registered_stores": len(self._stores),
                "dirty_stores": len(self._dirty),
                "total_writes": self._write_count,
                "stores": {
                    k: {
                        "path": str(v.path),
                        "has_data": v.data is not None,
                        "last_modified": v.last_modified,
                    }
                    for k, v in self._stores.items()
                },
            }


class _StoreEntry:
    """Internal state for a registered persistence store."""
    __slots__ = ("path", "data", "last_modified")

    def __init__(self, path: Path, data: Any = None) -> None:
        self.path = path
        self.data = data
        self.last_modified: float = 0.0


# Module-level singleton for easy access
persistence_manager = PersistenceManager()
