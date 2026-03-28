"""Session objects for dashboard / owner authentication."""

from __future__ import annotations

import logging
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom.session")

_CONFIG: dict[str, Any] = {}
_LOCK = threading.RLock()


@dataclass
class Session:
    id: str
    created_at: float
    expires_at: float
    last_activity: float
    privilege_level: str
    revoked: bool = False


class SessionManager:
    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        auth = cfg.get("auth", {})
        self._ttl_s = float(auth.get("session_ttl_s", 86400))
        self._idle_s = float(auth.get("session_max_idle_s", 3600))
        self._persist = bool(auth.get("persist_sessions", False))
        self._path = Path(auth.get("session_db_path", "data/atom_sessions.sqlite"))
        self._sessions: dict[str, Session] = {}
        self._lock = threading.RLock()
        if self._persist:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS revoked (id TEXT PRIMARY KEY)")
            conn.commit()

    def _is_revoked_persisted(self, sid: str) -> bool:
        if not self._persist:
            return False
        try:
            with sqlite3.connect(self._path) as conn:
                cur = conn.execute("SELECT 1 FROM revoked WHERE id = ?", (sid,))
                return cur.fetchone() is not None
        except Exception:
            logger.debug("session persist check failed", exc_info=True)
            return False

    def _persist_revoke(self, sid: str) -> None:
        if not self._persist:
            return
        try:
            with sqlite3.connect(self._path) as conn:
                conn.execute("INSERT OR IGNORE INTO revoked (id) VALUES (?)", (sid,))
                conn.commit()
        except Exception:
            logger.debug("session persist revoke failed", exc_info=True)

    def create_session(self, *, privilege_level: str | None = None) -> Session:
        auth = (_CONFIG.get("auth") or {})
        priv = privilege_level or auth.get("privilege_default", "operate")
        now = time.time()
        sid = secrets.token_urlsafe(32)
        sess = Session(
            id=sid,
            created_at=now,
            expires_at=now + self._ttl_s,
            last_activity=now,
            privilege_level=str(priv),
        )
        with self._lock:
            self._sessions[sid] = sess
        logger.info("Created session %s… (priv=%s)", sid[:8], sess.privilege_level)
        return sess

    def validate(self, sid: str | None) -> Session | None:
        if not sid or not isinstance(sid, str):
            return None
        if self._is_revoked_persisted(sid):
            return None
        with self._lock:
            s = self._sessions.get(sid.strip())
            if s is None:
                return None
            if s.revoked:
                return None
            now = time.time()
            if now > s.expires_at:
                return None
            if now - s.last_activity > self._idle_s:
                return None
            s.last_activity = now
            return s

    def revoke(self, sid: str) -> None:
        with self._lock:
            s = self._sessions.get(sid)
            if s is not None:
                s.revoked = True
                del self._sessions[sid]
        self._persist_revoke(sid)
        logger.info("Revoked session %s…", sid[:8])


_mgr: SessionManager | None = None


def configure(config: dict | None = None) -> None:
    global _CONFIG, _mgr
    with _LOCK:
        _CONFIG = dict(config or {})
        auth = _CONFIG.get("auth") or {}
        if auth.get("sessions_enabled", False):
            _mgr = SessionManager(_CONFIG)
        else:
            _mgr = None


def sessions_enabled() -> bool:
    return bool((_CONFIG.get("auth") or {}).get("sessions_enabled", False))


def get_manager() -> SessionManager | None:
    return _mgr


def create_session(*, privilege_level: str | None = None) -> Session | None:
    m = get_manager()
    if m is None:
        return None
    return m.create_session(privilege_level=privilege_level)


def validate_session(sid: str | None) -> Session | None:
    m = get_manager()
    if m is None:
        return None
    return m.validate(sid)


def revoke_session(sid: str) -> None:
    m = get_manager()
    if m is not None:
        m.revoke(sid)
