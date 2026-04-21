"""
ATOM -- Persistent System Profile (System Control v1, Phase B).

A thin, cached view of the host machine that is:
  * Loaded at boot from ``data/system_profile.json``.
  * Refreshed from :class:`SystemScanner` on demand.
  * Exposed as a ~200-char ``[MACHINE] ...`` line that goes into every
    LLM prompt so ATOM can answer "do I have enough disk for X?" or
    "which laptop am I on?" without a tool call.

This module is read-heavy by design: hot-path callers only touch
``get_compact_context()`` which is cached and invalidated when the
scanner emits a new snapshot.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom.system_profile")

_DEFAULT_PATH = Path("data/system_profile.json")
_REFRESH_INTERVAL_SEC = 300.0  # compact context re-derives every 5 min at most


class SystemProfile:
    """Persistent, compact view of the host system for prompt injection."""

    def __init__(
        self,
        config: dict | None = None,
        scanner: Any | None = None,
        path: Path | str | None = None,
    ) -> None:
        self._config = config or {}
        self._scanner = scanner
        cfg_path = (self._config.get("system_profile_path")
                    if isinstance(self._config, dict) else None)
        self._path = Path(path or cfg_path or _DEFAULT_PATH)

        self._lock = threading.Lock()
        self._snapshot: dict[str, Any] = {}
        self._compact_cache: str = ""
        self._compact_mtime: float = 0.0

        self._load()
        if not self._snapshot:
            self._bootstrap_minimal()

    # ── Persistence ────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._path.exists():
                self._snapshot = json.loads(
                    self._path.read_text(encoding="utf-8"),
                )
                logger.debug("Loaded system profile from %s", self._path)
        except Exception:
            logger.debug("System profile load failed", exc_info=True)
            self._snapshot = {}

    def persist(self) -> None:
        with self._lock:
            snap = dict(self._snapshot)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(snap, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            logger.debug("System profile persist failed", exc_info=True)

    # ── Snapshot ingest ─────────────────────────────────────────────

    def _bootstrap_minimal(self) -> None:
        """Fallback when no scanner snapshot exists yet.

        Populates enough data that ``get_compact_context()`` returns a
        useful line even before the first full scan completes.
        """
        try:
            uname = platform.uname()
            snap = {
                "machine": {
                    "os": f"{uname.system} {uname.release}",
                    "arch": uname.machine,
                    "host": uname.node,
                    "user": os.environ.get("USER", ""),
                },
                "refreshed_at": time.time(),
            }
            try:
                import psutil
                snap["machine"]["ram_total_gb"] = round(
                    psutil.virtual_memory().total / (1024 ** 3), 1,
                )
            except Exception:
                pass
            with self._lock:
                self._snapshot = snap
            self.persist()
        except Exception:
            logger.debug("Bootstrap minimal profile failed", exc_info=True)

    def refresh_from_scanner(self) -> bool:
        """Pull the latest scan from :class:`SystemScanner` into this profile.

        Returns True when the snapshot changed, False otherwise.
        """
        scanner = self._scanner
        if scanner is None:
            return False
        try:
            scan = getattr(scanner, "last_scan", None) or {}
            if not scan:
                return False

            system = scan.get("system", {}) or {}
            disks = scan.get("disks", []) or []
            network = scan.get("network", []) or []
            env = scan.get("environment", {}) or {}
            health = scan.get("health", {}) or {}

            primary_disk: dict[str, Any] = {}
            for d in disks:
                if d.get("mount") in ("/", "C:\\"):
                    primary_disk = d
                    break
            if not primary_disk and disks:
                primary_disk = disks[0]

            active_nics = [n for n in network if n.get("is_up")]

            snapshot = {
                "machine": {
                    "os": system.get("os", ""),
                    "os_version": system.get("os_version", ""),
                    "arch": system.get("architecture", ""),
                    "host": system.get("hostname", ""),
                    "user": system.get("username", ""),
                    "cpu": system.get("cpu", ""),
                    "cpu_cores": system.get("cpu_cores", ""),
                    "ram_total_gb": system.get("ram_total_gb"),
                    "ram_available_gb": system.get("ram_available_gb"),
                    "gpu": system.get("gpu", ""),
                    "display": system.get("display", ""),
                    "display_count": system.get("display_count"),
                    "has_battery": system.get("has_battery"),
                    "shell": system.get("shell", ""),
                },
                "storage": {
                    "primary_mount": primary_disk.get("mount", ""),
                    "total_gb": primary_disk.get("total_gb"),
                    "free_gb": primary_disk.get("free_gb"),
                    "percent_used": primary_disk.get("percent_used"),
                    "fs": primary_disk.get("fs", ""),
                    "disk_count": len(disks),
                },
                "network": {
                    "active_interfaces": [
                        n.get("name", "") for n in active_nics
                    ],
                    "primary_ip": next(
                        (n.get("ip", "") for n in active_nics if n.get("ip")),
                        "",
                    ),
                },
                "dev_env": {
                    "languages": list(env.get("languages", []))[:8],
                    "ides": list(env.get("ides", []))[:4],
                    "git": bool(env.get("git")),
                    "docker": bool(env.get("docker")),
                    "node": bool(env.get("node")),
                },
                "health": {
                    "overall": health.get("overall"),
                    "cpu": health.get("cpu"),
                    "ram": health.get("ram"),
                    "disk": health.get("disk"),
                },
                "installed_apps_count": scan.get("installed_apps_count"),
                "refreshed_at": time.time(),
            }

            with self._lock:
                changed = snapshot != self._snapshot
                self._snapshot = snapshot
                if changed:
                    self._compact_cache = ""
                    self._compact_mtime = 0.0

            if changed:
                self.persist()
            return changed
        except Exception:
            logger.debug("Refresh from scanner failed", exc_info=True)
            return False

    # ── Public read API ─────────────────────────────────────────────

    def get_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._snapshot)

    def get_compact_context(self) -> str:
        """Return a ≤230-char ``[MACHINE]`` line for prompt injection.

        This is called on every LLM turn, so the result is cached for
        :data:`_REFRESH_INTERVAL_SEC`. ATOM's spoken latency does not pay
        for it on the hot path.
        """
        now = time.time()
        if (self._compact_cache
                and (now - self._compact_mtime) < _REFRESH_INTERVAL_SEC):
            return self._compact_cache

        self.refresh_from_scanner()

        with self._lock:
            snap = dict(self._snapshot)

        machine = snap.get("machine", {}) or {}
        storage = snap.get("storage", {}) or {}
        health = snap.get("health", {}) or {}

        os_name = (machine.get("os") or "").strip()
        cpu = (machine.get("cpu") or "").strip()
        ram = machine.get("ram_total_gb")
        ram_free = machine.get("ram_available_gb")
        free_gb = storage.get("free_gb")
        total_gb = storage.get("total_gb")
        pct_used = storage.get("percent_used")
        overall = health.get("overall")

        parts: list[str] = []
        if os_name:
            parts.append(os_name)
        if cpu:
            parts.append(cpu.replace("  ", " ")[:40])
        if ram is not None:
            try:
                if ram_free is not None:
                    parts.append(f"RAM {float(ram_free):.1f}/{float(ram):.0f}GB")
                else:
                    parts.append(f"RAM {float(ram):.0f}GB")
            except (TypeError, ValueError):
                pass
        if free_gb is not None and total_gb is not None:
            try:
                parts.append(f"Disk {float(free_gb):.0f}/{float(total_gb):.0f}GB free")
            except (TypeError, ValueError):
                if pct_used is not None:
                    try:
                        parts.append(f"Disk {float(pct_used):.0f}% used")
                    except (TypeError, ValueError):
                        pass
        if overall is not None:
            try:
                parts.append(f"health {int(overall)}/100")
            except (TypeError, ValueError):
                pass

        if not parts:
            compact = ""
        else:
            compact = "[MACHINE] " + " | ".join(parts)
            if len(compact) > 230:
                compact = compact[:227] + "..."

        with self._lock:
            self._compact_cache = compact
            self._compact_mtime = now
        return compact

    def invalidate(self) -> None:
        """Force the next ``get_compact_context()`` call to re-derive."""
        with self._lock:
            self._compact_cache = ""
            self._compact_mtime = 0.0

    def on_scanner_update(self, *_args, **_kwargs) -> None:
        """Bus listener hook: invalidate + refresh when scanner emits new data."""
        if self.refresh_from_scanner():
            logger.debug("System profile updated from scanner event")
