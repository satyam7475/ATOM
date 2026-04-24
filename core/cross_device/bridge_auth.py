"""Pre-shared token auth + audit trail for the iPhone Shortcuts bridge.

Design goals
------------
* **Shared-secret only** -- Phase 1 ships HTTP (no TLS). A 32-byte
  `secrets.token_urlsafe` token lives on disk (``config/bridge_token``
  by default) and is copy-pasted into iCloud Keychain once, then
  referenced from every Shortcut via ``Get Password from Keychain``.
* **Constant-time compare** -- `secrets.compare_digest` on every request
  so failed guesses leak no timing.
* **Append-only audit** -- every 401 writes one JSONL line to
  ``logs/atom_bridge_audit.jsonl`` with source IP, endpoint, reason.
  A rolling 60-second window with >= 10 failures raises a spoken
  warning via the event bus (emitted by the caller).
* **Non-destructive on rotation** -- rotating the token is just
  deleting the file; the next boot regenerates it and the Shortcut
  user re-pastes.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger("atom.bridge.auth")


_TOKEN_BYTES = 32
_AUDIT_WINDOW_S = 60.0
_AUDIT_SPOKEN_THRESHOLD = 10


def generate_or_load_token(token_path: str | os.PathLike[str]) -> str:
    """Return the existing token at *token_path* or mint a new one.

    Mode 0600 on disk. If the path already exists and is non-empty we
    re-use it verbatim (important: rotating the file is the owner's
    explicit action, not ATOM's).
    """
    p = Path(token_path).expanduser()
    try:
        if p.exists() and p.stat().st_size > 0:
            raw = p.read_text(encoding="utf-8").strip()
            if raw:
                return raw
    except OSError as exc:
        logger.warning("bridge token read failed (%s); regenerating", exc)

    token = secrets.token_urlsafe(_TOKEN_BYTES)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(token, encoding="utf-8")
    try:
        p.chmod(0o600)
    except OSError:
        logger.debug("could not chmod 0600 on bridge token (%s)", p)
    logger.info("bridge token minted at %s", p)
    return token


def verify_token(expected: str, supplied: str | None) -> bool:
    """Constant-time compare. Any mismatch (including missing) returns False."""
    if not expected or not supplied:
        return False
    if len(expected) != len(supplied):
        # compare_digest requires equal length; short-circuit safely.
        return False
    return secrets.compare_digest(
        expected.encode("utf-8"),
        supplied.encode("utf-8"),
    )


class AuthAuditLog:
    """Append-only JSONL audit log + rolling-window flood detector.

    One instance per bridge. Thread-safe (the aiohttp handlers run on
    the asyncio loop, but nothing stops a future caller from scheduling
    writes from a worker thread).
    """

    __slots__ = ("_path", "_lock", "_recent", "_last_warning_ts")

    def __init__(self, audit_path: str | os.PathLike[str]) -> None:
        self._path = Path(audit_path).expanduser()
        self._lock = Lock()
        self._recent: deque[float] = deque(maxlen=_AUDIT_SPOKEN_THRESHOLD * 4)
        self._last_warning_ts: float = 0.0
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("audit log dir create failed: %s", exc)

    def record_failure(
        self,
        *,
        source_ip: str,
        endpoint: str,
        reason: str,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        """Write one JSONL line. Return True if this pushes the rolling
        window across the spoken-warning threshold (caller should emit
        a bus event in that case)."""
        now = time.time()
        entry: dict[str, Any] = {
            "ts": now,
            "event": "bridge.auth.failure",
            "source_ip": str(source_ip or "")[:64],
            "endpoint": str(endpoint or "")[:64],
            "reason": str(reason or "")[:120],
        }
        if extra:
            for k, v in extra.items():
                entry[str(k)[:32]] = v

        with self._lock:
            self._recent.append(now)
            cutoff = now - _AUDIT_WINDOW_S
            while self._recent and self._recent[0] < cutoff:
                self._recent.popleft()
            flood = len(self._recent) >= _AUDIT_SPOKEN_THRESHOLD
            spoke_this_window = (now - self._last_warning_ts) < _AUDIT_WINDOW_S
            should_warn = flood and not spoke_this_window
            if should_warn:
                self._last_warning_ts = now

        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("audit log write failed: %s", exc)

        return should_warn

    def record_reject(
        self,
        *,
        source_ip: str,
        endpoint: str,
        reason: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Non-auth rejection (e.g. 409 single-device-lock). Audited but
        doesn't count against the auth-flood threshold."""
        entry: dict[str, Any] = {
            "ts": time.time(),
            "event": "bridge.reject",
            "source_ip": str(source_ip or "")[:64],
            "endpoint": str(endpoint or "")[:64],
            "reason": str(reason or "")[:120],
        }
        if extra:
            for k, v in extra.items():
                entry[str(k)[:32]] = v
        try:
            with self._lock, self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("audit log write failed: %s", exc)
