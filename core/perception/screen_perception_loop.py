"""ATOM Sprint N1 — continuous desktop screen perception loop.

The existing :class:`context.screen_reader.ScreenReader` is request-driven
(``screen_describe`` / ``screen_analyze`` intents). This module turns it
into an **always-on** rolling perceptual sense:

    * sample the desktop every ``interval_s`` seconds (default 12 s)
    * skip frames that haven't changed (perceptual hash on a 8x8 thumbnail)
    * OCR via the existing Apple Vision pipeline (Neural Engine, ~150 ms)
    * persist `(timestamp, app, ocr_text, hash, tokens)` rows to a small
      SQLite store at ``data/screen_observations.sqlite`` (rolling cap)
    * emit ``screen.observation`` on the bus so AwarenessLoop, RAG, and
      the realtime room can react

Everything stays local. The loop pauses while ATOM is speaking or
listening — we don't want to burn the GPU during the user's turn — and
when ``presence_present`` is False (Boss isn't at the desk).

Design constraints:
    * single small SQLite file (no schema migrations to maintain)
    * O(1) memory: rows are pruned to ``max_rows`` (default 5_000 ≈ ~16
      hours of 12 s sampling)
    * zero new dependencies (uses ScreenReader + stdlib hashlib)
    * graceful: ``attach()`` survives ScreenReader errors and just logs.

Owner: Boss (Satyam).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom.perception.screen_loop")

_DEFAULT_DB_PATH = "data/screen_observations.sqlite"


# ── front-app helper ──────────────────────────────────────────────────


def _frontmost_app_macos() -> str:
    """Return the active macOS app name, or empty string on any failure."""
    if sys.platform != "darwin":
        return ""
    try:
        out = subprocess.check_output(
            [
                "osascript",
                "-e",
                'tell application "System Events" to get name of '
                "first application process whose frontmost is true",
            ],
            stderr=subprocess.DEVNULL,
            timeout=0.6,
        )
        return out.decode().strip()
    except Exception:
        return ""


def _phash_text(text: str) -> str:
    """Cheap perceptual hash for OCR text (token-set + sliding window)."""
    if not text:
        return "0"
    # Normalise: lowercase, keep alnum tokens >= 3 chars, sort
    tokens = sorted({
        t.strip().lower()
        for t in text.split()
        if len(t.strip()) >= 3
    })
    digest = hashlib.blake2b(
        " ".join(tokens).encode("utf-8"),
        digest_size=12,
    ).hexdigest()
    return digest


# ── config ────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ScreenLoopConfig:
    enabled: bool = True
    interval_s: float = 12.0
    pause_during_speech: bool = True
    pause_during_listen: bool = True
    require_presence: bool = True
    max_rows: int = 5_000
    db_path: str = _DEFAULT_DB_PATH
    redact_passwords: bool = True
    min_text_chars: int = 24
    emit_bus_event: bool = True
    significance_min_jaccard: float = 0.55  # 1.0 = identical
    burst_when_idle_s: float = 60.0
    extra_ignore_apps: tuple[str, ...] = (
        "1Password 7",
        "1Password",
        "KeePassXC",
    )


@dataclass(slots=True)
class _SnapshotMetrics:
    samples: int = 0
    persisted: int = 0
    deduped: int = 0
    paused: int = 0
    errors: int = 0
    last_ok_ts: float = 0.0
    last_text_hash: str = "0"
    last_text_tokens: set[str] = field(default_factory=set)
    last_app: str = ""


# ── core loop ─────────────────────────────────────────────────────────


_PASSWORD_FIELD_HINTS = (
    "password", "secret", "api key", "api_key", "token",
    "ssn", "social security",
)


class ScreenPerceptionLoop:
    """Always-on desktop perception → SQLite + bus.

    Wire it once during boot (after :class:`ScreenReader` is built):

        loop = ScreenPerceptionLoop(bus, screen_reader, state_manager,
                                    config=ScreenLoopConfig(...))
        loop.attach()           # starts background task

    On shutdown:

        await loop.stop()
    """

    def __init__(
        self,
        bus: Any,
        screen_reader: Any,
        state_manager: Any | None = None,
        *,
        config: ScreenLoopConfig | None = None,
    ) -> None:
        self.bus = bus
        self.screen_reader = screen_reader
        self.state = state_manager
        self.config = config or ScreenLoopConfig()
        self._task: asyncio.Task | None = None
        self._shutdown = asyncio.Event()
        self._db_path = Path(self.config.db_path)
        self._metrics = _SnapshotMetrics()
        self._db_lock = threading.Lock()
        self._presence_present: bool | None = None
        self._is_speaking: bool = False
        self._is_listening: bool = False
        self._loop_started_at: float = 0.0
        self._init_db()

    # ── lifecycle ─────────────────────────────────────────────────

    def attach(self) -> None:
        if not self.config.enabled:
            logger.info("Screen perception loop disabled by config")
            return
        if self.screen_reader is None or not getattr(
            self.screen_reader, "is_available", False,
        ):
            logger.info(
                "Screen perception loop: no OCR backend available (skip)",
            )
            return
        # Attach bus subscriptions for state gating
        try:
            self.bus.on("presence.snapshot", self._on_presence)
            self.bus.on("speaking_started", self._on_speak_start)
            self.bus.on("speaking_finished", self._on_speak_end)
            self.bus.on("listening_started", self._on_listen_start)
            self.bus.on("listening_finished", self._on_listen_end)
            self.bus.on("tts_complete", self._on_speak_end)
        except Exception:  # bus might be a stub
            logger.debug("bus subscription failed", exc_info=True)
        self._loop_started_at = time.time()
        self._task = asyncio.create_task(
            self._run(), name="screen_perception_loop",
        )
        logger.info(
            "Screen perception loop attached: interval=%.1fs db=%s "
            "max_rows=%d",
            self.config.interval_s, self._db_path, self.config.max_rows,
        )

    async def stop(self) -> None:
        self._shutdown.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    # ── DB ────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.debug("could not create db parent", exc_info=True)
            return
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS screen_observation (
                        id          INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts          REAL    NOT NULL,
                        app         TEXT,
                        text_hash   TEXT,
                        token_count INTEGER,
                        ocr_text    TEXT,
                        tags        TEXT,
                        created_at  REAL    NOT NULL DEFAULT (strftime('%s','now'))
                    )
                    """,
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_obs_ts "
                    "ON screen_observation(ts)",
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_obs_app "
                    "ON screen_observation(app)",
                )
                conn.commit()
        except Exception:
            logger.exception("Failed to init screen-observation DB")

    # ── bus handlers ──────────────────────────────────────────────

    async def _on_presence(self, *, present: bool | None = None,
                           **_kw: Any) -> None:
        if present is not None:
            self._presence_present = bool(present)

    async def _on_speak_start(self, **_kw: Any) -> None:
        self._is_speaking = True

    async def _on_speak_end(self, **_kw: Any) -> None:
        self._is_speaking = False

    async def _on_listen_start(self, **_kw: Any) -> None:
        self._is_listening = True

    async def _on_listen_end(self, **_kw: Any) -> None:
        self._is_listening = False

    # ── main loop ─────────────────────────────────────────────────

    def _should_skip(self) -> str | None:
        if self.config.require_presence and self._presence_present is False:
            return "presence_absent"
        if self.config.pause_during_speech and self._is_speaking:
            return "speaking"
        if self.config.pause_during_listen and self._is_listening:
            return "listening"
        return None

    def _redact(self, text: str) -> str:
        if not text or not self.config.redact_passwords:
            return text
        out: list[str] = []
        skip_next = False
        for line in text.splitlines():
            lower = line.lower()
            if any(h in lower for h in _PASSWORD_FIELD_HINTS):
                out.append(line)
                skip_next = True
                continue
            if skip_next and line.strip():
                out.append("[redacted-by-atom]")
                skip_next = False
                continue
            out.append(line)
            skip_next = False
        return "\n".join(out)

    async def _run(self) -> None:
        logger.info("Screen perception loop running")
        while not self._shutdown.is_set():
            try:
                await asyncio.sleep(self.config.interval_s)
                if self._shutdown.is_set():
                    break

                skip = self._should_skip()
                if skip is not None:
                    self._metrics.paused += 1
                    continue

                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                self._metrics.errors += 1
                logger.exception(
                    "Screen perception tick failed (continuing)",
                )

    async def _tick(self) -> None:
        self._metrics.samples += 1
        # The OCR is sync + can be heavy; offload to a thread.
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None, self.screen_reader.capture_and_read,
            )
        except Exception:
            self._metrics.errors += 1
            logger.debug("capture_and_read failed", exc_info=True)
            return

        if not isinstance(result, dict):
            return
        text = (result.get("text") or "").strip()
        if len(text) < self.config.min_text_chars:
            return

        app = _frontmost_app_macos()
        if app and app in self.config.extra_ignore_apps:
            return

        text_redacted = self._redact(text)
        text_hash = _phash_text(text_redacted)

        # Significance gate: skip near-identical frames except every ~burst
        # interval so the timeline still has periodic anchors when Boss
        # stares at the same screen.
        last_burst_age = time.time() - self._metrics.last_ok_ts
        same_app = (app == self._metrics.last_app)
        same_hash = (text_hash == self._metrics.last_text_hash)
        if (
            same_hash
            and same_app
            and last_burst_age < self.config.burst_when_idle_s
        ):
            self._metrics.deduped += 1
            return

        # Soft significance: jaccard on token set against last frame.
        new_tokens = {
            t.strip().lower()
            for t in text_redacted.split()
            if len(t.strip()) >= 3
        }
        if self._metrics.last_text_tokens and new_tokens:
            inter = len(new_tokens & self._metrics.last_text_tokens)
            union = len(new_tokens | self._metrics.last_text_tokens)
            if union > 0:
                jac = inter / union
                if (
                    jac >= self.config.significance_min_jaccard
                    and same_app
                    and last_burst_age < self.config.burst_when_idle_s
                ):
                    self._metrics.deduped += 1
                    return

        ts = time.time()
        token_count = len(new_tokens)
        try:
            await loop.run_in_executor(
                None, self._persist, ts, app, text_hash,
                token_count, text_redacted,
            )
        except Exception:
            self._metrics.errors += 1
            logger.debug("screen observation persist failed",
                         exc_info=True)
            return

        self._metrics.persisted += 1
        self._metrics.last_ok_ts = ts
        self._metrics.last_text_hash = text_hash
        self._metrics.last_text_tokens = new_tokens
        self._metrics.last_app = app

        if self.config.emit_bus_event:
            try:
                emit = getattr(self.bus, "emit_long", None) \
                    or getattr(self.bus, "emit_fast", None)
                if emit is not None:
                    emit(
                        "screen.observation",
                        ts=ts,
                        app=app,
                        text=text_redacted[:1500],
                        text_full_chars=len(text_redacted),
                        text_hash=text_hash,
                        token_count=token_count,
                        source="screen_perception_loop",
                    )
            except Exception:
                logger.debug("bus emit failed", exc_info=True)

    # ── persistence ───────────────────────────────────────────────

    def _persist(
        self, ts: float, app: str, text_hash: str,
        token_count: int, text: str,
    ) -> None:
        with self._db_lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT INTO screen_observation "
                "(ts, app, text_hash, token_count, ocr_text, tags) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ts, app, text_hash, token_count, text, json.dumps([])),
            )
            conn.execute(
                "DELETE FROM screen_observation "
                "WHERE id IN ("
                "  SELECT id FROM screen_observation "
                "  ORDER BY ts ASC "
                "  LIMIT MAX(0, (SELECT COUNT(*) FROM screen_observation) - ?)"
                ")",
                (self.config.max_rows,),
            )
            conn.commit()

    # ── public query API ──────────────────────────────────────────

    def query(
        self,
        *,
        since_ts: float | None = None,
        until_ts: float | None = None,
        app: str | None = None,
        text_contains: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Query the screen observation timeline.

        All arguments optional; called from the recall tool / awareness
        loop. Returns most recent first.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if since_ts is not None:
            clauses.append("ts >= ?")
            params.append(float(since_ts))
        if until_ts is not None:
            clauses.append("ts <= ?")
            params.append(float(until_ts))
        if app:
            clauses.append("LOWER(app) = LOWER(?)")
            params.append(app)
        if text_contains:
            clauses.append("LOWER(ocr_text) LIKE ?")
            params.append(f"%{text_contains.lower()}%")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = (
            "SELECT id, ts, app, text_hash, token_count, ocr_text "
            "FROM screen_observation"
            + where +
            " ORDER BY ts DESC LIMIT ?"
        )
        params.append(int(limit))
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(sql, tuple(params)).fetchall()
        except Exception:
            logger.exception("screen observation query failed")
            return []
        return [
            {
                "id": r["id"],
                "ts": r["ts"],
                "app": r["app"] or "",
                "hash": r["text_hash"] or "",
                "tokens": int(r["token_count"] or 0),
                "text": r["ocr_text"] or "",
            }
            for r in rows
        ]

    def metrics(self) -> dict[str, Any]:
        m = self._metrics
        return {
            "samples": m.samples,
            "persisted": m.persisted,
            "deduped": m.deduped,
            "paused": m.paused,
            "errors": m.errors,
            "last_ok_age_s": (
                round(time.time() - m.last_ok_ts, 1)
                if m.last_ok_ts > 0 else None
            ),
            "running_for_s": round(time.time() - self._loop_started_at, 1)
            if self._loop_started_at else 0.0,
            "db_path": str(self._db_path),
            "db_size_bytes": (
                self._db_path.stat().st_size
                if self._db_path.exists() else 0
            ),
            "config": {
                "interval_s": self.config.interval_s,
                "max_rows": self.config.max_rows,
                "require_presence": self.config.require_presence,
                "pause_during_speech": self.config.pause_during_speech,
            },
        }
