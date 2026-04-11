"""
Tests for macOS Keychain helpers and KeychainVault.

Run: python3 -m tests.test_keychain_store
"""

from __future__ import annotations

import json
import os
import sys
import unittest.mock as mock
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _proc(rc: int, out: str = "", err: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=rc, stdout=out, stderr=err)


class _RecordingRunner:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.responses = responses
        self.calls: list[list[str]] = []
        self._i = 0

    def __call__(self, cmd: list[str], timeout: float) -> SimpleNamespace:
        self.calls.append(cmd)
        if self._i < len(self.responses):
            r = self.responses[self._i]
            self._i += 1
            return r
        return _proc(1, "", "unexpected call")


def test_keychain_set_get_delete_helpers() -> None:
    from core.macos.keychain_store import keychain_delete, keychain_get, keychain_set

    calls: list[list[str]] = []

    def runner(cmd: list[str], timeout: float) -> SimpleNamespace:
        calls.append(cmd)
        joined = " ".join(cmd)
        if "add-generic-password" in joined:
            return _proc(0)
        if "find-generic-password" in joined:
            return _proc(0, "secret-value\n")
        if "delete-generic-password" in joined:
            return _proc(0)
        return _proc(99)

    with mock.patch("core.macos.keychain_store.sys.platform", "darwin"):
        assert keychain_set("svc", "acct", "pw", runner=runner) is True
        assert keychain_get("svc", "acct", runner=runner) == "secret-value"
        assert keychain_delete("svc", "acct", runner=runner) is True
    assert any("add-generic-password" in c for c in calls)
    print("  PASS: keychain_set/get/delete helpers")


def test_keychain_helpers_non_darwin() -> None:
    from core.macos.keychain_store import keychain_get, keychain_set

    with mock.patch("core.macos.keychain_store.sys.platform", "linux"):
        assert keychain_set("s", "a", "p", runner=lambda *_a, **_k: _proc(0)) is False
        assert keychain_get("s", "a", runner=lambda *_a, **_k: _proc(0, "x")) == ""
    print("  PASS: keychain helpers non-darwin")


def test_keychain_vault_put_get_persist_index(tmp_path) -> None:
    from core.macos import keychain_store as ks
    from core.macos.keychain_store import KeychainVault

    idx = tmp_path / "keychain_vault_keys.json"
    runner = _RecordingRunner(
        [_proc(0), _proc(0, "payload\n")],
    )
    with mock.patch.object(ks, "_KEYCHAIN_INDEX", idx):
        with mock.patch("core.macos.keychain_store.sys.platform", "darwin"):
            v = KeychainVault(service="com.test.atom", runner=runner)
            assert v.put("voice_profile", "payload") is True
            assert v.get("voice_profile") == "payload"
            v.persist()
    data = json.loads(idx.read_text(encoding="utf-8"))
    assert data["keys"] == ["voice_profile"]
    assert v.backend_name == "keychain"
    assert v.is_encrypted is True
    print("  PASS: KeychainVault put/get/index")


def test_keychain_vault_delete(tmp_path) -> None:
    from core.macos import keychain_store as ks
    from core.macos.keychain_store import KeychainVault

    idx = tmp_path / "k.json"
    idx.write_text(json.dumps({"keys": ["alpha"]}), encoding="utf-8")
    runner = _RecordingRunner([_proc(0, "x\n"), _proc(0)])
    with mock.patch.object(ks, "_KEYCHAIN_INDEX", idx):
        with mock.patch("core.macos.keychain_store.sys.platform", "darwin"):
            v = KeychainVault(service="com.test.atom", runner=runner)
            assert "alpha" in v.keys()
            assert v.delete("alpha") is True
            v.persist()
    assert json.loads(idx.read_text(encoding="utf-8"))["keys"] == []
    print("  PASS: KeychainVault delete")


def test_use_macos_keychain_flag() -> None:
    from core.security_fortress import _use_macos_keychain

    with mock.patch("sys.platform", "linux"):
        assert _use_macos_keychain({"security_fortress": {}}) is False
    with mock.patch("sys.platform", "darwin"):
        assert _use_macos_keychain({"security_fortress": {"use_macos_keychain": False}}) is False
        assert _use_macos_keychain({"security_fortress": {"use_macos_keychain": True}}) is True
        assert _use_macos_keychain({}) is True
    print("  PASS: _use_macos_keychain")


def test_select_vault_forces_file_when_disabled() -> None:
    from core.security_fortress import EncryptedVault, _select_security_vault

    cfg = {"security_fortress": {"use_macos_keychain": False}}
    with mock.patch("sys.platform", "darwin"):
        v = _select_security_vault(cfg)

    assert isinstance(v, EncryptedVault)
    print("  PASS: _select_security_vault file when disabled")


def test_select_vault_linux_uses_file() -> None:
    from core.security_fortress import EncryptedVault, _select_security_vault

    with mock.patch("sys.platform", "linux"):
        v = _select_security_vault({})
    assert isinstance(v, EncryptedVault)
    print("  PASS: _select_security_vault linux uses EncryptedVault")


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    test_keychain_set_get_delete_helpers()
    test_keychain_helpers_non_darwin()
    with tempfile.TemporaryDirectory() as d1:
        test_keychain_vault_put_get_persist_index(Path(d1))
    with tempfile.TemporaryDirectory() as d2:
        test_keychain_vault_delete(Path(d2))
    test_use_macos_keychain_flag()
    test_select_vault_forces_file_when_disabled()
    test_select_vault_linux_uses_file()
    print("All keychain store tests passed.")