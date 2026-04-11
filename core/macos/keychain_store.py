"""macOS Keychain access for ATOM secrets (generic-password items)."""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("atom.macos.keychain")

_KEYCHAIN_INDEX = Path("data/security/keychain_vault_keys.json")
_MAX_KEY_LEN = 120
_SAFE_KEY = re.compile(r"^[\w.-]{1,%d}$" % _MAX_KEY_LEN)

SecurityRunner = Callable[[list[str], float], Any]


def _default_runner(cmd: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def keychain_set(
    service: str,
    account: str,
    password: str,
    *,
    runner: SecurityRunner | None = None,
    timeout: float = 15.0,
) -> bool:
    """Store or update a generic password (-U update if exists)."""
    if sys.platform != "darwin":
        return False
    run = runner or _default_runner
    proc = run(
        [
            "security",
            "add-generic-password",
            "-s",
            service,
            "-a",
            account,
            "-w",
            password,
            "-U",
        ],
        timeout,
    )
    if proc.returncode != 0:
        logger.debug(
            "keychain_set failed rc=%s stderr=%s",
            proc.returncode,
            (proc.stderr or "")[:300],
        )
    return proc.returncode == 0


def keychain_get(
    service: str,
    account: str,
    *,
    runner: SecurityRunner | None = None,
    timeout: float = 15.0,
) -> str:
    """Return password bytes as UTF-8 string, or empty if missing / error."""
    if sys.platform != "darwin":
        return ""
    run = runner or _default_runner
    proc = run(
        ["security", "find-generic-password", "-s", service, "-a", account, "-w"],
        timeout,
    )
    if proc.returncode != 0:
        return ""
    return (proc.stdout or "").strip()


def keychain_delete(
    service: str,
    account: str,
    *,
    runner: SecurityRunner | None = None,
    timeout: float = 15.0,
) -> bool:
    if sys.platform != "darwin":
        return False
    run = runner or _default_runner
    proc = run(
        ["security", "delete-generic-password", "-s", service, "-a", account],
        timeout,
    )
    return proc.returncode == 0


class KeychainVault:
    """Key-value secrets backed by macOS Keychain (one item per key).

    Compatible with :class:`EncryptedVault` for ``get`` / ``put`` /
    ``delete`` / ``keys`` / ``persist``. Key names are persisted in
    ``data/security/keychain_vault_keys.json`` (names only, no secret values).
    """

    __slots__ = ("_service", "_runner", "_keys", "_index_dirty")

    backend_name = "keychain"

    def __init__(
        self,
        service: str = "com.atom.fortress",
        *,
        runner: SecurityRunner | None = None,
    ) -> None:
        self._service = service
        self._runner = runner
        self._keys: set[str] = set()
        self._index_dirty = False
        self._load_index()

    @property
    def is_encrypted(self) -> bool:
        return True

    def _load_index(self) -> None:
        if not _KEYCHAIN_INDEX.exists():
            return
        try:
            data = json.loads(_KEYCHAIN_INDEX.read_text(encoding="utf-8"))
            keys = data.get("keys")
            if isinstance(keys, list):
                self._keys = {str(k) for k in keys if k}
        except Exception:
            logger.debug("Keychain vault index load failed", exc_info=True)

    def _persist_index(self) -> None:
        if not self._index_dirty:
            return
        try:
            _KEYCHAIN_INDEX.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                {"keys": sorted(self._keys)},
                indent=2,
                sort_keys=True,
            )
            _KEYCHAIN_INDEX.write_text(payload + "\n", encoding="utf-8")
            self._index_dirty = False
        except Exception:
            logger.warning("Keychain vault index persist failed", exc_info=True)

    def persist(self) -> None:
        self._persist_index()

    def _validate_key(self, key: str) -> bool:
        if not key or len(key) > _MAX_KEY_LEN:
            return False
        return bool(_SAFE_KEY.match(key))

    def get(self, key: str, default: str = "") -> str:
        if not self._validate_key(key):
            return default
        val = keychain_get(
            self._service, key, runner=self._runner, timeout=15.0,
        )
        if val and key not in self._keys:
            self._keys.add(key)
            self._index_dirty = True
        return val if val else default

    def put(self, key: str, value: str) -> bool:
        if not self._validate_key(key):
            logger.warning("Rejected vault key (invalid for keychain): %r", key)
            return False
        ok = keychain_set(
            self._service, key, value, runner=self._runner, timeout=15.0,
        )
        if ok:
            self._keys.add(key)
            self._index_dirty = True
        else:
            logger.warning("Keychain store failed for key %r", key)
        return ok

    def delete(self, key: str) -> bool:
        if not self._validate_key(key):
            return False
        existed = key in self._keys or bool(
            keychain_get(self._service, key, runner=self._runner, timeout=5.0),
        )
        keychain_delete(self._service, key, runner=self._runner, timeout=10.0)
        if key in self._keys:
            self._keys.discard(key)
            self._index_dirty = True
        return existed

    def keys(self) -> list[str]:
        return sorted(self._keys)


__all__ = [
    "KeychainVault",
    "SecurityRunner",
    "keychain_delete",
    "keychain_get",
    "keychain_set",
]
