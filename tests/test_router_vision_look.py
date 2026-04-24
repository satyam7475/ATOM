"""Router glue for the ``vision_look`` tool.

We don't spin up a full Router here -- the live Router pulls ~30
modules and the vision_look handler only touches ``self._vision_engine``
plus stdlib. So we exercise the unbound method directly with a tiny
stand-in object, which keeps the test fast (~1ms) and lets us assert
on every branch.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from core.perception.apple_vision import VisionResult
from core.perception.camera_capture import CameraInfo, CaptureResult
from core.perception.vision_engine import VisionLookResult
from core.reasoning.tool_registry import get_tool_registry
from core.router.router import Router


# ── tool registry / dispatch wiring ────────────────────────────────


def test_vision_look_tool_is_registered():
    registry = get_tool_registry()
    tool = registry.get("vision_look")
    assert tool is not None, "vision_look tool was never registered"
    assert tool.category == "perception"
    assert tool.safety_level == "moderate"
    # The "focus" parameter must exist with the two enum options
    # documented in the prompt; otherwise the LLM can't ask for a QR
    # scan.
    assert any(p.name == "focus" for p in tool.parameters)
    focus = next(p for p in tool.parameters if p.name == "focus")
    assert focus.enum == ["general", "barcodes"]


def test_vision_look_in_action_dispatch_table():
    assert "vision_look" in Router._ACTION_DISPATCH


def test_vision_look_in_slow_actions_set():
    # AVCaptureSession can take 0.5-2.5s; if it weren't in
    # SLOW_ACTIONS, the event loop would stall during a tool call.
    assert "vision_look" in Router._SLOW_ACTIONS


# ── _do_vision_look behaviour (handler called via unbound method) ──


def _call_handler(self_obj: Any, args: dict) -> str:
    return Router._do_vision_look(self_obj, "vision_look", args)


def _stub_self(engine: Any = None) -> SimpleNamespace:
    return SimpleNamespace(_vision_engine=engine)


def test_vision_look_returns_offline_when_engine_is_none():
    out = _call_handler(_stub_self(None), {})
    assert "Camera is offline" in out


def test_vision_look_returns_offline_when_engine_reports_disabled():
    engine = SimpleNamespace(
        disabled_reason=lambda: "no cameras visible",
        look=lambda **kw: pytest.fail("look() must not be called when disabled"),
    )
    out = _call_handler(_stub_self(engine), {})
    assert "Camera is offline" in out
    assert "no cameras visible" in out


def test_vision_look_happy_path_returns_face_count_summary():
    cam = CameraInfo(
        name="MacBook Air Camera", unique_id="UID", kind="builtin",
        device_type="AVCaptureDeviceTypeBuiltInWideAngleCamera",
    )
    look_result = VisionLookResult(
        ok=True,
        camera=cam,
        capture=CaptureResult(ok=True, saved_path="/tmp/x.jpg", capture_ms=120, camera=cam),
        vision=VisionResult(ok=True, detection_ms=15, faces=1),
        capture_ms=120, detection_ms=15,
    )
    calls: list[dict] = []
    def _fake_look(**kw):
        calls.append(kw)
        return look_result
    engine = SimpleNamespace(disabled_reason=lambda: "", look=_fake_look)
    out = _call_handler(_stub_self(engine), {})
    assert "MacBook Air Camera" in out
    assert "one face" in out
    assert "120ms" in out  # capture timing surfaced
    assert calls[0]["detect_faces"] is True
    assert calls[0]["detect_barcodes"] is False
    assert calls[0]["reason"].startswith("vision_look:")


def test_vision_look_zero_faces_says_no_face_yet():
    cam = CameraInfo(
        name="Camera", unique_id="U", kind="builtin",
        device_type="AVCaptureDeviceTypeBuiltInWideAngleCamera",
    )
    look_result = VisionLookResult(
        ok=True, camera=cam,
        capture=CaptureResult(ok=True, saved_path="/tmp/x.jpg", capture_ms=80, camera=cam),
        vision=VisionResult(ok=True, detection_ms=5, faces=0),
        capture_ms=80, detection_ms=5,
    )
    engine = SimpleNamespace(disabled_reason=lambda: "", look=lambda **kw: look_result)
    out = _call_handler(_stub_self(engine), {})
    assert "no face yet" in out


def test_vision_look_barcode_focus_passes_through():
    cam = CameraInfo(name="C", unique_id="U", kind="builtin", device_type="t")
    look_result = VisionLookResult(
        ok=True, camera=cam,
        capture=CaptureResult(ok=True, saved_path="/tmp/x.jpg", capture_ms=10, camera=cam),
        vision=VisionResult(ok=True, detection_ms=2, faces=0, barcodes=["https://example.com"]),
        capture_ms=10, detection_ms=2,
    )
    calls: list[dict] = []
    def _fake_look(**kw):
        calls.append(kw)
        return look_result
    engine = SimpleNamespace(disabled_reason=lambda: "", look=_fake_look)
    out = _call_handler(_stub_self(engine), {"focus": "barcodes"})
    assert calls[0]["detect_barcodes"] is True
    assert "https://example.com" in out


def test_vision_look_handles_engine_exception_gracefully():
    def _boom(**kw):
        raise RuntimeError("AVCapture exploded")
    engine = SimpleNamespace(disabled_reason=lambda: "", look=_boom)
    out = _call_handler(_stub_self(engine), {})
    assert "Camera glance failed" in out


def test_vision_look_returns_friendly_error_on_unsuccessful_result():
    engine = SimpleNamespace(
        disabled_reason=lambda: "",
        look=lambda **kw: VisionLookResult(ok=False, error="capture timed out after 3.5s"),
    )
    out = _call_handler(_stub_self(engine), {})
    assert "Camera glance came back empty" in out
    assert "timed out" in out
