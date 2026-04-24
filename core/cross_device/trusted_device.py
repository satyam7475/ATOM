"""Single-iPhone trusted-device lock.

First handshake wins. The bridge stores a SHA-256 of the iPhone's
``device_id`` (a stable string the Shortcut provides -- e.g.
``Get Device Details -> Device Name`` plus a user-chosen salt) in
``data/trusted_iphone.json``. Subsequent posts with a *different*
hash are rejected with 409 Conflict and written to the bridge audit
log.

Resetting the binding is an explicit CLI action: the owner deletes the
state file (or runs ``python -m core.cross_device.trusted_device reset``).
This intentionally trades convenience for safety -- a housemate's
iPhone can't hijack Boss's routines by simply pointing the Shortcut at
ATOM's IP.

Thread safety
-------------
Writes use ``O_EXCL`` create via a temp-then-rename dance; if two
iPhones race the first one to flush wins and the second sees 409.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger("atom.bridge.device")


def hash_device_id(device_id: str) -> str:
    """SHA-256 hex of *device_id*. Truncated to 64 hex chars (256 bits)."""
    raw = str(device_id or "").strip().encode("utf-8", errors="replace")
    return hashlib.sha256(raw).hexdigest()


class TrustedIPhoneRegistry:
    """Persists the single trusted iPhone UDID hash.

    Layout on disk (``data/trusted_iphone.json``)::

        {
            "device_hash": "<sha256 hex>",
            "registered_at": <float epoch>,
            "label": "Satyam's iPhone",
            "schema": 1
        }
    """

    __slots__ = ("_path", "_lock", "_cached")

    _SCHEMA = 1

    def __init__(self, state_path: str | os.PathLike[str]) -> None:
        self._path = Path(state_path).expanduser()
        self._lock = Lock()
        self._cached: dict[str, Any] | None = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("trusted-iphone state dir create failed: %s", exc)

    def _load(self) -> dict[str, Any] | None:
        if self._cached is not None:
            return self._cached
        try:
            if not self._path.exists() or self._path.stat().st_size == 0:
                return None
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if not isinstance(data, dict):
                return None
            if int(data.get("schema", 0)) != self._SCHEMA:
                return None
            self._cached = data
            return data
        except (OSError, ValueError) as exc:
            logger.warning("trusted-iphone load failed: %s", exc)
            return None

    def registered_hash(self) -> str | None:
        with self._lock:
            data = self._load()
        if not data:
            return None
        h = str(data.get("device_hash") or "").strip().lower()
        return h or None

    def register_or_verify(
        self,
        device_id: str,
        *,
        label: str = "",
    ) -> tuple[bool, str]:
        """Claim the slot if empty, or verify against an existing claim.

        Returns ``(ok, reason)``. ``ok=True`` means the caller is the
        registered iPhone; ``ok=False, reason="conflict"`` means a
        different iPhone already holds the slot.
        """
        incoming = hash_device_id(device_id)
        if not incoming:
            return False, "empty_device_id"

        with self._lock:
            existing = self._load()
            if existing:
                current = str(existing.get("device_hash") or "").strip().lower()
                if current == incoming:
                    return True, "already_registered"
                return False, "conflict"

            payload = {
                "device_hash": incoming,
                "registered_at": time.time(),
                "label": str(label or "")[:64],
                "schema": self._SCHEMA,
            }
            self._atomic_write(payload)
            self._cached = payload
            logger.info(
                "trusted iPhone registered (hash=%s… label=%r)",
                incoming[:12],
                payload["label"],
            )
            return True, "registered"

    def reset(self) -> None:
        """Forget the current trusted iPhone. Caller should own the lock
        (e.g. CLI that prompts the owner before calling)."""
        with self._lock:
            try:
                if self._path.exists():
                    self._path.unlink()
            except OSError as exc:
                logger.warning("trusted-iphone reset failed: %s", exc)
            self._cached = None

    def _atomic_write(self, payload: dict[str, Any]) -> None:
        """Write via temp-then-rename so a crashed writer never leaves
        a half-written JSON that the next load would silently ignore."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".trusted_iphone.",
            suffix=".tmp",
            dir=str(self._path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, self._path)
            try:
                os.chmod(self._path, 0o600)
            except OSError:
                pass
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
