"""Tier-3+ identity-freshness gate.

Validates the pure function in
:py:mod:`core.router.identity_gate` and its coarse integration with
``action_tier`` so tier 1/2 actions always pass and tier 3/4 actions
require a fresh Face ID once an iPhone is paired.

Run: ``python3 -m pytest tests/test_identity_freshness_gate.py -v``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.cross_device.trusted_device import TrustedIPhoneRegistry  # noqa: E402
from core.identity_engine import IdentityEngine  # noqa: E402
from core.router.identity_gate import (  # noqa: E402
    check_identity_freshness,
    gate_requires_fresh_identity,
)


# ────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────

@pytest.fixture()
def atom_root(tmp_path: Path) -> Path:
    (tmp_path / "config").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "config" / "settings.json").write_text("{}", encoding="utf-8")
    return tmp_path


def _cross_device_cfg(root: Path, enabled: bool = True) -> dict:
    return {
        "cross_device": {
            "enabled": enabled,
            "faceid_freshness_s": 60,
            "trusted_device_path": str(root / "data" / "trusted_iphone.json"),
        },
    }


def _register_iphone(root: Path, device_id: str = "iphone-boss") -> None:
    TrustedIPhoneRegistry(root / "data" / "trusted_iphone.json").register_or_verify(device_id)


# ────────────────────────────────────────────
# gate_requires_fresh_identity
# ────────────────────────────────────────────

def test_gate_off_when_cross_device_disabled(atom_root: Path) -> None:
    _register_iphone(atom_root)
    cfg = _cross_device_cfg(atom_root, enabled=False)
    assert gate_requires_fresh_identity(config=cfg, action="shutdown_pc") is False


def test_gate_off_when_no_iphone_registered(atom_root: Path) -> None:
    cfg = _cross_device_cfg(atom_root, enabled=True)
    assert gate_requires_fresh_identity(config=cfg, action="shutdown_pc") is False


def test_gate_off_for_tier_1_even_when_iphone_registered(atom_root: Path) -> None:
    _register_iphone(atom_root)
    cfg = _cross_device_cfg(atom_root)
    assert gate_requires_fresh_identity(config=cfg, action="time") is False
    assert gate_requires_fresh_identity(config=cfg, action="status") is False
    assert gate_requires_fresh_identity(config=cfg, action="greeting") is False


def test_gate_off_for_tier_2(atom_root: Path) -> None:
    _register_iphone(atom_root)
    cfg = _cross_device_cfg(atom_root)
    assert gate_requires_fresh_identity(config=cfg, action="set_assistant_mode") is False


def test_gate_on_for_tier_3_when_iphone_registered(atom_root: Path) -> None:
    _register_iphone(atom_root)
    cfg = _cross_device_cfg(atom_root)
    # tier-3 default action
    assert gate_requires_fresh_identity(config=cfg, action="open_app") is True
    assert gate_requires_fresh_identity(config=cfg, action="move_path") is True


def test_gate_on_for_tier_4(atom_root: Path) -> None:
    _register_iphone(atom_root)
    cfg = _cross_device_cfg(atom_root)
    assert gate_requires_fresh_identity(config=cfg, action="shutdown_pc") is True
    assert gate_requires_fresh_identity(config=cfg, action="kill_process") is True


def test_gate_off_when_config_is_none() -> None:
    assert gate_requires_fresh_identity(config=None, action="shutdown_pc") is False


# ────────────────────────────────────────────
# check_identity_freshness
# ────────────────────────────────────────────

def test_check_allows_tier_1_without_identity_engine(atom_root: Path) -> None:
    _register_iphone(atom_root)
    cfg = _cross_device_cfg(atom_root)
    ok, reason = check_identity_freshness(
        config=cfg, identity_engine=None, action="time",
    )
    assert ok is True
    assert reason == ""


def test_check_blocks_tier_3_without_identity_engine(atom_root: Path) -> None:
    _register_iphone(atom_root)
    cfg = _cross_device_cfg(atom_root)
    ok, reason = check_identity_freshness(
        config=cfg, identity_engine=None, action="open_app",
    )
    assert ok is False
    assert "wired" in reason.lower() or "identity" in reason.lower()


def test_check_blocks_tier_3_when_faceid_stale(atom_root: Path) -> None:
    _register_iphone(atom_root)
    cfg = _cross_device_cfg(atom_root)
    ie = IdentityEngine(config=cfg)
    ok, reason = check_identity_freshness(
        config=cfg, identity_engine=ie, action="shutdown_pc",
    )
    assert ok is False
    assert "iPhone" in reason or "Verify" in reason


def test_check_allows_tier_3_when_faceid_fresh(atom_root: Path) -> None:
    _register_iphone(atom_root)
    cfg = _cross_device_cfg(atom_root)
    ie = IdentityEngine(config=cfg)
    ie.record_faceid_verification(True, device_id="iphone-boss")
    ok, reason = check_identity_freshness(
        config=cfg, identity_engine=ie, action="shutdown_pc",
    )
    assert ok is True
    assert reason == ""


def test_check_faithfully_defers_to_identity_engine_exception(atom_root: Path) -> None:
    """If the engine raises, fail closed with a safe reason."""
    _register_iphone(atom_root)
    cfg = _cross_device_cfg(atom_root)

    class _BrokenEngine:
        def is_owner_verified(self, *_, **__):
            raise RuntimeError("engine is unhappy")

    ok, reason = check_identity_freshness(
        config=cfg, identity_engine=_BrokenEngine(), action="shutdown_pc",
    )
    assert ok is False
    assert reason


# ────────────────────────────────────────────
# Router wiring: attach_identity_engine present + _check_identity_gate works
# ────────────────────────────────────────────

def test_router_attach_identity_engine_is_callable() -> None:
    """Don't construct the full Router (has many dependencies); just
    import the class and check the attach + helper methods exist and
    the defaults make the gate inert."""
    from core.router.router import Router
    assert hasattr(Router, "attach_identity_engine")
    assert callable(Router.attach_identity_engine)
    assert hasattr(Router, "_check_identity_gate")
    assert callable(Router._check_identity_gate)


def test_router_check_identity_gate_passes_when_unregistered(atom_root: Path) -> None:
    """With no iPhone registered the gate must pass for ANY action, even
    tier 4. This ensures fresh installs aren't locked out.

    We stub just enough of Router to exercise ``_check_identity_gate``
    without booting the full AI OS."""
    from core.router.router import Router

    # Bypass __init__ (which pulls in the whole stack) and poke the
    # attributes the helper reads directly.
    r = Router.__new__(Router)
    r._config = _cross_device_cfg(atom_root)
    r._identity_engine = None
    ok, reason = r._check_identity_gate("shutdown_pc")
    assert ok is True
    assert reason == ""


def test_router_check_identity_gate_blocks_stale_when_registered(atom_root: Path) -> None:
    from core.router.router import Router

    _register_iphone(atom_root)
    r = Router.__new__(Router)
    r._config = _cross_device_cfg(atom_root)
    r._identity_engine = IdentityEngine(config=r._config)
    ok, reason = r._check_identity_gate("shutdown_pc")
    assert ok is False
    assert reason


def test_router_check_identity_gate_passes_when_verified(atom_root: Path) -> None:
    from core.router.router import Router

    _register_iphone(atom_root)
    r = Router.__new__(Router)
    r._config = _cross_device_cfg(atom_root)
    r._identity_engine = IdentityEngine(config=r._config)
    r._identity_engine.record_faceid_verification(True, device_id="iphone-boss")
    ok, reason = r._check_identity_gate("shutdown_pc")
    assert ok is True
    assert reason == ""
