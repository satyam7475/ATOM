"""Face ID freshness + trusted-iPhone helpers.

These tests lock down the contract the router's tier-3 gate
(:py:mod:`core.router.router`) and the proactive engine will rely on:

* `IdentityEngine.is_owner_verified` returns True only for a successful
  Face ID within the freshness window.
* `device_binding.is_trusted_iphone` resolves the single-slot registry
  written by the bridge handler.

Run: ``python3 -m pytest tests/test_identity_freshness.py -v``
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.identity_engine import IdentityEngine  # noqa: E402


# ────────────────────────────────────────────
# IdentityEngine Face ID freshness
# ────────────────────────────────────────────

def test_identity_engine_defaults_to_not_verified() -> None:
    ie = IdentityEngine(config={})
    assert ie.is_owner_verified() is False
    info = ie.faceid_freshness_info()
    assert info["fresh"] is False
    assert info["last_timestamp"] == 0.0
    assert info["age_s"] is None


def test_identity_engine_verified_inside_window() -> None:
    ie = IdentityEngine(config={"cross_device": {"faceid_freshness_s": 60}})
    ie.record_faceid_verification(True, device_id="iphone-boss", label="Boss iPhone")
    assert ie.is_owner_verified() is True
    info = ie.faceid_freshness_info()
    assert info["fresh"] is True
    assert info["freshness_window_s"] == 60.0
    assert info["label"] == "Boss iPhone"
    assert "iphone-boss" not in info["device_prefix"] or info["device_prefix"].startswith("iphone-boss")


def test_identity_engine_expires_after_window() -> None:
    ie = IdentityEngine(config={"cross_device": {"faceid_freshness_s": 1}})
    ie.record_faceid_verification(True, timestamp=time.time() - 5, device_id="d")
    assert ie.is_owner_verified() is False


def test_identity_engine_explicit_window_overrides_config() -> None:
    ie = IdentityEngine(config={"cross_device": {"faceid_freshness_s": 1}})
    ie.record_faceid_verification(True, timestamp=time.time() - 2, device_id="d")
    assert ie.is_owner_verified() is False
    assert ie.is_owner_verified(window_s=60) is True


def test_identity_engine_rejects_failed_verification() -> None:
    ie = IdentityEngine(config={})
    ie.record_faceid_verification(False, device_id="iphone-boss")
    assert ie.is_owner_verified() is False
    info = ie.faceid_freshness_info()
    assert info["last_verified"] is False
    assert info["fresh"] is False


def test_identity_engine_clear_faceid_invalidates() -> None:
    ie = IdentityEngine(config={})
    ie.record_faceid_verification(True, device_id="iphone-boss")
    assert ie.is_owner_verified() is True
    ie.clear_faceid()
    assert ie.is_owner_verified() is False


def test_identity_engine_negative_or_zero_window_is_never_fresh() -> None:
    ie = IdentityEngine(config={"cross_device": {"faceid_freshness_s": -10}})
    ie.record_faceid_verification(True, device_id="d")
    # Negative config value falls back to the default, which is > 0.
    assert ie.is_owner_verified() is True
    # But an explicit window_s <= 0 always blocks.
    assert ie.is_owner_verified(window_s=0) is False
    assert ie.is_owner_verified(window_s=-1) is False


def test_identity_engine_non_numeric_config_falls_back_to_default() -> None:
    ie = IdentityEngine(config={"cross_device": {"faceid_freshness_s": "not a number"}})
    ie.record_faceid_verification(True, device_id="d")
    assert ie.is_owner_verified() is True


def test_identity_engine_device_prefix_never_leaks_full_id() -> None:
    ie = IdentityEngine(config={})
    long_id = "iphone-12345678-extra-secret-stuff"
    ie.record_faceid_verification(True, device_id=long_id)
    info = ie.faceid_freshness_info()
    assert "secret" not in info["device_prefix"]
    assert len(info["device_prefix"]) <= 13  # 12 chars + the "…" ellipsis


# ────────────────────────────────────────────
# device_binding trusted-iPhone helpers
# ────────────────────────────────────────────

@pytest.fixture()
def atom_root(tmp_path: Path) -> Path:
    (tmp_path / "config").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "config" / "settings.json").write_text("{}", encoding="utf-8")
    return tmp_path


def test_device_binding_registered_iphone_hash_none_by_default(atom_root: Path) -> None:
    from core.identity.device_binding import registered_iphone_hash
    state_path = atom_root / "data" / "trusted_iphone.json"
    assert registered_iphone_hash(state_path=state_path) is None


def test_device_binding_is_trusted_iphone_happy_path(atom_root: Path) -> None:
    from core.cross_device.trusted_device import TrustedIPhoneRegistry
    from core.identity.device_binding import (
        is_trusted_iphone,
        registered_iphone_hash,
    )
    state_path = atom_root / "data" / "trusted_iphone.json"
    reg = TrustedIPhoneRegistry(state_path)
    reg.register_or_verify("iphone-boss")

    current = registered_iphone_hash(state_path=state_path)
    assert current is not None
    assert len(current) == 64
    assert is_trusted_iphone("iphone-boss", state_path=state_path) is True
    assert is_trusted_iphone("iphone-housemate", state_path=state_path) is False


def test_device_binding_resolves_path_from_config(atom_root: Path) -> None:
    from core.cross_device.trusted_device import TrustedIPhoneRegistry
    from core.identity.device_binding import is_trusted_iphone

    state_path = atom_root / "data" / "trusted_iphone.json"
    TrustedIPhoneRegistry(state_path).register_or_verify("iphone-boss")

    cfg = {"cross_device": {"trusted_device_path": str(state_path)}}
    assert is_trusted_iphone("iphone-boss", cfg) is True
    assert is_trusted_iphone("iphone-housemate", cfg) is False


def test_device_binding_empty_device_id_is_not_trusted(atom_root: Path) -> None:
    from core.identity.device_binding import is_trusted_iphone
    assert is_trusted_iphone("", state_path=atom_root / "data" / "trusted_iphone.json") is False
