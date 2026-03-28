"""
ATOM -- System Indexer (Fast Boot-Time Knowledge Graph).

Indexes the local system in the background so ATOM has instant
knowledge of installed applications, running processes, and
recent files. This allows ATOM to execute commands like "open notepad"
or "close chrome" instantly without needing to scan the system each time.

Capabilities:
    - Installed Apps Index: Maps common app names to their executables/paths.
    - Running Processes Index: Maintains a fast lookup of active processes.
    - User Directories: Indexes Desktop, Documents, Downloads.

Owner: Satyam
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.platform_adapter import adapter

logger = logging.getLogger("atom.system_indexer")


@dataclass
class AppIndexEntry:
    name: str
    install_path: str
    executable: str = ""


class SystemIndexer:
    """Background indexer for instant system knowledge."""

    __slots__ = (
        "_executor", "_running", "_task", "_apps_index",
        "_process_index", "_recent_files", "_last_index_time"
    )

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="indexer")
        self._running = False
        self._task: asyncio.Task | None = None
        
        self._apps_index: dict[str, AppIndexEntry] = {}
        self._process_index: dict[str, int] = {}  # name -> pid
        self._recent_files: list[str] = []
        self._last_index_time: float = 0.0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._index_loop())
        logger.info("System Indexer started")

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        self._executor.shutdown(wait=False)
        logger.info("System Indexer stopped")

    async def _index_loop(self) -> None:
        """Periodically update the system index."""
        loop = asyncio.get_running_loop()
        
        # Initial fast index
        await loop.run_in_executor(self._executor, self._build_index)
        
        while self._running:
            try:
                await asyncio.sleep(300)  # Refresh every 5 minutes
                if self._running:
                    await loop.run_in_executor(self._executor, self._build_index)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("System indexer loop error: %s", e)
                await asyncio.sleep(60)

    def _build_index(self) -> None:
        """Build the full system index (runs in thread)."""
        t0 = time.monotonic()
        
        self._index_apps()
        self._index_processes()
        self._index_user_dirs()
        
        self._last_index_time = time.time()
        elapsed = (time.monotonic() - t0) * 1000
        logger.info("System indexed in %.0fms (apps: %d, processes: %d)", 
                    elapsed, len(self._apps_index), len(self._process_index))

    def _index_apps(self) -> None:
        """Index installed applications."""
        try:
            apps = adapter.get_installed_apps()
            new_index = {}
            for app in apps:
                if not app.name:
                    continue
                # Normalize name for searching
                norm_name = app.name.lower()
                new_index[norm_name] = AppIndexEntry(
                    name=app.name,
                    install_path=app.install_path or ""
                )
                
                # Add common aliases
                if "google chrome" in norm_name:
                    new_index["chrome"] = new_index[norm_name]
                elif "visual studio code" in norm_name:
                    new_index["vscode"] = new_index[norm_name]
                    new_index["code"] = new_index[norm_name]
                    
            self._apps_index = new_index
        except Exception as e:
            logger.debug("App indexing failed: %s", e)

    def _index_processes(self) -> None:
        """Index currently running processes."""
        try:
            procs = adapter.list_processes()
            new_index = {}
            for p in procs:
                if p.name:
                    new_index[p.name.lower()] = p.pid
                    # Also index without .exe
                    if p.name.lower().endswith(".exe"):
                        new_index[p.name.lower()[:-4]] = p.pid
            self._process_index = new_index
        except Exception as e:
            logger.debug("Process indexing failed: %s", e)

    def _index_user_dirs(self) -> None:
        """Index recent files in user directories."""
        try:
            home = Path.home()
            dirs_to_scan = [
                home / "Desktop",
                home / "Documents",
                home / "Downloads"
            ]
            
            recent = []
            now = time.time()
            
            for d in dirs_to_scan:
                if not d.exists():
                    continue
                try:
                    # Get files modified in last 7 days
                    for f in d.iterdir():
                        if f.is_file() and not f.name.startswith('.'):
                            mtime = f.stat().st_mtime
                            if now - mtime < 7 * 86400:  # 7 days
                                recent.append((f.name, mtime))
                except (PermissionError, OSError):
                    pass
                    
            # Sort by newest first, keep top 20
            recent.sort(key=lambda x: x[1], reverse=True)
            self._recent_files = [f[0] for f in recent[:20]]
            
        except Exception as e:
            logger.debug("Directory indexing failed: %s", e)

    # ── Public Queries ───────────────────────────────────────────────

    def search_app(self, query: str) -> AppIndexEntry | None:
        """Find an installed app by name."""
        q = query.lower().strip()
        if not q:
            return None
            
        # Exact match
        if q in self._apps_index:
            return self._apps_index[q]
            
        # Substring match
        for name, entry in self._apps_index.items():
            if q in name:
                return entry
                
        return None

    def get_process_pid(self, process_name: str) -> int | None:
        """Get PID for a running process by name."""
        q = process_name.lower().strip()
        if q in self._process_index:
            return self._process_index[q]
        if not q.endswith(".exe") and f"{q}.exe" in self._process_index:
            return self._process_index[f"{q}.exe"]
        return None

    def get_summary_for_llm(self) -> str:
        """Get a compact summary of the system index for the LLM prompt."""
        parts = []
        
        if self._process_index:
            # Get notable running apps
            notable = [p for p in ["chrome", "code", "spotify", "slack", "teams", "discord", "notepad"] 
                      if p in self._process_index]
            if notable:
                parts.append(f"Running apps: {', '.join(notable)}")
                
        if self._recent_files:
            parts.append(f"Recent files: {', '.join(self._recent_files[:5])}")
            
        if not parts:
            return "System index: Ready"
            
        return " | ".join(parts)

# Global singleton
system_indexer = SystemIndexer()
