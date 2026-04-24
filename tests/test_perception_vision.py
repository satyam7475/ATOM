"""Tests for the camera + Apple Vision perception stack.

The real AVFoundation / Vision frameworks aren't called from these
tests -- we monkeypatch the leaf functions and exercise the
``VisionEngine`` glue (audit log, lock, throttle, summary, attach
hooks). A second, smaller block confirms ``CameraInfo``/``choose_preferred``
work even when the pyobjc bindings are absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core.perception import camera_capture
from core.perception.apple_vision import VisionResult
from core.perception.camera_capture import CameraInfo, CaptureResult
from core.perception.vision_audit import VisionAuditLog
from core.perception.vision_engine import VisionEngine, VisionLookResult


# ── camera_capture.choose_preferred ──────────────────────────────────


def _mk(name: str, kind: str, uid: str = "") -> CameraInfo:
    return CameraInfo(
        name=name,
        unique_id=uid or f"uid-{name.lower()}",
        device_type=f"AVCaptureDeviceType{kind.title()}",
        kind=kind,
    )


def test_classify_kind_continuity_type_string():
    assert (
        camera_capture._classify_kind("AVCaptureDeviceTypeContinuityCamera") == "continuity"
    )


def test_classify_kind_builtin_type_string():
    assert (
        camera_capture._classify_kind("AVCaptureDeviceTypeBuiltInWideAngleCamera") == "builtin"
    )


def test_classify_kind_iphone_via_external_type_promoted_to_continuity():
    # Real macOS 15+ behaviour: a Continuity-Camera-paired iPhone
    # reports as ``AVCaptureDeviceTypeExternal`` with a modelID of
    # ``"iPhone15,4"`` (or similar). We must classify it as
    # ``continuity`` so the auto preference picks the iPhone over the
    # built-in MacBook camera.
    kind = camera_capture._classify_kind(
        "AVCaptureDeviceTypeExternal",
        name="iPhone Camera",
        model_id="iPhone15,4",
    )
    assert kind == "continuity"


def test_classify_kind_external_only_via_modelid_promotes_to_continuity():
    # Some pyobjc builds return an empty localizedName but a valid
    # modelID — modelID alone must be enough for the promotion.
    kind = camera_capture._classify_kind(
        "AVCaptureDeviceTypeExternal", name="", model_id="iPhone16,2",
    )
    assert kind == "continuity"


def test_classify_kind_real_external_usb_stays_external():
    # A USB DSLR rig with no iPhone signal must stay external so the
    # iPhone preference doesn't accidentally pick a wired webcam.
    kind = camera_capture._classify_kind(
        "AVCaptureDeviceTypeExternal", name="Logitech BRIO", model_id="C925e",
    )
    assert kind == "external"


def test_choose_preferred_auto_prefers_continuity_over_builtin():
    cams = [_mk("MacBook Air Camera", "builtin"), _mk("Satyam's iPhone", "continuity")]
    chosen = camera_capture.choose_preferred(cams, preferred="auto")
    assert chosen is not None
    assert chosen.kind == "continuity"


def test_choose_preferred_builtin_ignores_iphone_even_when_present():
    cams = [_mk("MacBook Air Camera", "builtin"), _mk("Satyam's iPhone", "continuity")]
    chosen = camera_capture.choose_preferred(cams, preferred="builtin")
    assert chosen is not None
    assert chosen.kind == "builtin"


def test_choose_preferred_continuity_falls_back_to_builtin_when_iphone_absent():
    cams = [_mk("MacBook Air Camera", "builtin")]
    chosen = camera_capture.choose_preferred(cams, preferred="continuity")
    # Falls back rather than returning None so a "prefer iPhone" config
    # still yields *some* camera when the iPhone is asleep.
    assert chosen is not None
    assert chosen.kind == "builtin"


def test_choose_preferred_explicit_uid_wins():
    cams = [
        _mk("MacBook Air Camera", "builtin", uid="UID-A"),
        _mk("Satyam's iPhone", "continuity", uid="UID-B"),
        _mk("USB Webcam", "external", uid="UID-C"),
    ]
    chosen = camera_capture.choose_preferred(cams, preferred="auto", explicit_uid="UID-C")
    assert chosen is not None
    assert chosen.unique_id == "UID-C"


def test_choose_preferred_no_cameras_returns_none():
    assert camera_capture.choose_preferred([], preferred="auto") is None


# ── CGColorSpaceCreateDeviceRGB resolution (live-fix Apr 2026) ───────


def test_cg_color_space_resolver_finds_symbol_in_quartz_umbrella():
    """Regression for the live boot crash on 2026-04-24 where
    ``_Quartz.CGColorSpaceCreateDeviceRGB()`` raised AttributeError and
    the JPEG video delegate logged
    ``video delegate exception: 'CGColorSpaceCreateDeviceRGB'``,
    breaking the boot face check.

    The resolver must accept any of three PyObjC layouts:
      1. symbol re-exported by the ``Quartz`` umbrella
      2. symbol on ``Quartz.CoreGraphics``
      3. symbol on the standalone ``CoreGraphics`` module
    """
    import sys as _sys
    import types as _types

    real_quartz = _sys.modules.get("Quartz")
    real_quartz_cg = _sys.modules.get("Quartz.CoreGraphics")
    real_cg = _sys.modules.get("CoreGraphics")
    real_cap_quartz = camera_capture._Quartz
    try:
        # Layout A: only Quartz umbrella has it.
        sentinel_a = lambda: "umbrella"
        umbrella = _types.SimpleNamespace(CGColorSpaceCreateDeviceRGB=sentinel_a)
        camera_capture._Quartz = umbrella  # type: ignore[attr-defined]
        _sys.modules.pop("CoreGraphics", None)
        _sys.modules.pop("Quartz.CoreGraphics", None)
        fn = camera_capture._resolve_cg_color_space_create_device_rgb()
        assert fn is sentinel_a

        # Layout B: only Quartz.CoreGraphics has it.
        sentinel_b = lambda: "qcg"
        qcg = _types.SimpleNamespace(CGColorSpaceCreateDeviceRGB=sentinel_b)
        umbrella_no_attr = _types.SimpleNamespace(CoreGraphics=qcg)
        camera_capture._Quartz = umbrella_no_attr  # type: ignore[attr-defined]
        fn = camera_capture._resolve_cg_color_space_create_device_rgb()
        assert fn is sentinel_b

        # Layout C: only the standalone CoreGraphics module has it.
        sentinel_c = lambda: "standalone"
        cg_mod = _types.ModuleType("CoreGraphics")
        cg_mod.CGColorSpaceCreateDeviceRGB = sentinel_c  # type: ignore[attr-defined]
        _sys.modules["CoreGraphics"] = cg_mod
        camera_capture._Quartz = _types.SimpleNamespace()  # no attr, no submodule
        fn = camera_capture._resolve_cg_color_space_create_device_rgb()
        assert fn is sentinel_c

        # Layout D: nobody has it (degraded PyObjC) → graceful None.
        _sys.modules.pop("CoreGraphics", None)
        camera_capture._Quartz = _types.SimpleNamespace()
        fn = camera_capture._resolve_cg_color_space_create_device_rgb()
        assert fn is None
    finally:
        camera_capture._Quartz = real_cap_quartz  # type: ignore[attr-defined]
        if real_quartz is not None:
            _sys.modules["Quartz"] = real_quartz
        if real_quartz_cg is not None:
            _sys.modules["Quartz.CoreGraphics"] = real_quartz_cg
        if real_cg is not None:
            _sys.modules["CoreGraphics"] = real_cg
        else:
            _sys.modules.pop("CoreGraphics", None)


# ── VisionAuditLog ────────────────────────────────────────────────────


def test_audit_log_writes_jsonl_with_required_keys(tmp_path: Path):
    log_path = tmp_path / "vision_audit.jsonl"
    log = VisionAuditLog(log_path)
    log.record(
        reason="boot_face_check",
        source="MacBook Air Camera",
        source_kind="builtin",
        capture_ms=420.0,
        detection_ms=18.5,
        faces=1,
        summary="1 face",
        saved_path=str(tmp_path / "frame.jpg"),
    )
    assert log_path.exists()
    line = log_path.read_text(encoding="utf-8").strip()
    record = json.loads(line)
    for key in (
        "ts", "reason", "source", "source_kind", "capture_ms",
        "detection_ms", "faces", "summary", "saved_path", "error",
    ):
        assert key in record, f"missing key {key!r}"
    assert record["faces"] == 1
    assert record["error"] == ""


def test_audit_log_never_raises_when_directory_unwritable(tmp_path: Path):
    # Point at a path inside a *file* (not a directory) — write must
    # be silently swallowed instead of bubbling to the runtime.
    file_blocking_dir = tmp_path / "block.txt"
    file_blocking_dir.write_text("not a directory")
    log = VisionAuditLog(file_blocking_dir / "child" / "audit.jsonl")
    log.record(reason="boot_face_check")  # must not raise


# ── VisionEngine wiring ──────────────────────────────────────────────


@pytest.fixture
def patched_engine(monkeypatch, tmp_path):
    """Engine with capture_jpeg + apple_vision.detect mocked.

    Returns ``(engine, captures, detections)`` so each test can assert
    on what was called and tweak return values in-place.
    """
    captures: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []

    def fake_list_cameras() -> list[CameraInfo]:
        return [_mk("Test Camera", "builtin", uid="UID-T")]

    def fake_capture(camera: CameraInfo, *, out_path=None, timeout_s=3.5) -> CaptureResult:
        captures.append({
            "camera": camera, "out_path": out_path, "timeout_s": timeout_s,
        })
        return CaptureResult(
            ok=True, saved_path=str(tmp_path / "fake.jpg"),
            capture_ms=120.0, camera=camera,
        )

    monkeypatch.setattr(camera_capture, "list_cameras", fake_list_cameras)
    monkeypatch.setattr(camera_capture, "capture_jpeg", fake_capture)
    monkeypatch.setattr(camera_capture, "HAS_AVFOUNDATION", True)

    from core.perception import apple_vision
    monkeypatch.setattr(apple_vision, "HAS_VISION", True)

    def fake_detect(image_path, *, detect_faces=True, detect_barcodes=False) -> VisionResult:
        detections.append({
            "path": str(image_path),
            "detect_faces": detect_faces,
            "detect_barcodes": detect_barcodes,
        })
        return VisionResult(
            ok=True, detection_ms=12.5, faces=1,
            face_boxes=[(0.1, 0.1, 0.4, 0.4)],
        )

    monkeypatch.setattr(apple_vision, "detect", fake_detect)

    engine = VisionEngine(
        enabled=True,
        preferred_camera="auto",
        audit_log_path=tmp_path / "audit.jsonl",
        min_gap_s=0.0,  # disable throttle for these tests
        capture_timeout_s=1.0,
    )
    return engine, captures, detections, tmp_path


def test_engine_disabled_when_config_off(tmp_path):
    engine = VisionEngine(enabled=False, audit_log_path=tmp_path / "a.jsonl")
    assert engine.enabled is False
    assert "vision.enabled=false" in engine.disabled_reason()
    out = engine.look(reason="test")
    assert out.ok is False
    assert "disabled" in out.error


def test_engine_look_happy_path_records_audit(patched_engine):
    engine, captures, detections, tmp_path = patched_engine
    out = engine.look(reason="vision_look:test", detect_barcodes=False)
    assert out.ok is True
    assert out.faces == 1
    assert "Test Camera" in out.summary
    # audit log got one line
    audit_path = tmp_path / "audit.jsonl"
    assert audit_path.exists()
    lines = [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0]["faces"] == 1
    assert lines[0]["reason"] == "vision_look:test"
    # capture + detection were both invoked once
    assert len(captures) == 1
    assert len(detections) == 1
    assert detections[0]["detect_barcodes"] is False


def test_engine_emits_event_with_face_count(patched_engine):
    engine, _, _, _ = patched_engine
    events: list[tuple[str, dict]] = []
    engine._emit = lambda event, **data: events.append((event, data))
    engine.look(reason="manual")
    # exactly one frame.captured event with faces=1
    assert any(name == "vision.frame.captured" for name, _ in events)
    captured_event = next(d for n, d in events if n == "vision.frame.captured")
    assert captured_event["faces"] == 1
    assert captured_event["source"] == "Test Camera"


def test_engine_throttle_blocks_back_to_back_calls(monkeypatch, tmp_path):
    # Same patched stack, but with min_gap_s set to a real value.
    monkeypatch.setattr(camera_capture, "HAS_AVFOUNDATION", True)
    monkeypatch.setattr(camera_capture, "list_cameras", lambda: [_mk("C", "builtin")])
    monkeypatch.setattr(
        camera_capture, "capture_jpeg",
        lambda cam, **kw: CaptureResult(
            ok=True, saved_path="/tmp/x.jpg", capture_ms=10, camera=cam,
        ),
    )
    from core.perception import apple_vision
    monkeypatch.setattr(apple_vision, "HAS_VISION", True)
    monkeypatch.setattr(
        apple_vision, "detect",
        lambda *a, **kw: VisionResult(ok=True, detection_ms=2, faces=0),
    )
    engine = VisionEngine(
        enabled=True, audit_log_path=tmp_path / "a.jsonl",
        min_gap_s=10.0, capture_timeout_s=1.0,
    )
    first = engine.look(reason="r1")
    assert first.ok is True
    second = engine.look(reason="r2")
    assert second.ok is False
    assert "throttled" in second.error


def test_engine_offline_when_no_cameras(monkeypatch, tmp_path):
    monkeypatch.setattr(camera_capture, "HAS_AVFOUNDATION", True)
    monkeypatch.setattr(camera_capture, "list_cameras", lambda: [])
    from core.perception import apple_vision
    monkeypatch.setattr(apple_vision, "HAS_VISION", True)
    engine = VisionEngine(
        enabled=True, audit_log_path=tmp_path / "a.jsonl", min_gap_s=0.0,
    )
    assert "no cameras visible" in engine.disabled_reason()
    out = engine.look(reason="boot")
    assert out.ok is False
    assert "no cameras" in out.error


def test_engine_offline_when_pyobjc_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(camera_capture, "HAS_AVFOUNDATION", False)
    engine = VisionEngine(
        enabled=True, audit_log_path=tmp_path / "a.jsonl", min_gap_s=0.0,
    )
    reason = engine.disabled_reason()
    assert "AVFoundation" in reason
    out = engine.look(reason="boot")
    assert out.ok is False


def test_engine_face_count_zero_propagates_to_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(camera_capture, "HAS_AVFOUNDATION", True)
    monkeypatch.setattr(camera_capture, "list_cameras", lambda: [_mk("C", "builtin")])
    monkeypatch.setattr(
        camera_capture, "capture_jpeg",
        lambda cam, **kw: CaptureResult(
            ok=True, saved_path="/tmp/y.jpg", capture_ms=5, camera=cam,
        ),
    )
    from core.perception import apple_vision
    monkeypatch.setattr(apple_vision, "HAS_VISION", True)
    monkeypatch.setattr(
        apple_vision, "detect",
        lambda *a, **kw: VisionResult(ok=True, detection_ms=3, faces=0),
    )
    engine = VisionEngine(
        enabled=True, audit_log_path=tmp_path / "a.jsonl", min_gap_s=0.0,
    )
    out = engine.look(reason="boot")
    assert out.ok is True
    assert out.faces == 0
    # Summary must still include the camera name + "no faces" so the
    # boot log reads cleanly even when no one's in front of the lens.
    assert "no faces" in out.summary


# ── VLM describe path ────────────────────────────────────────────────


class _FakeCaptioner:
    """Minimal duck-typed captioner for the engine tests.

    Stays in this file so the VLM test module doesn't need to touch
    the engine and vice versa. Matches the interface VisionEngine
    actually calls: ``is_available`` property + ``describe(path)``.
    """

    def __init__(self, caption: str = "A dog on a couch.", available: bool = True) -> None:
        self._caption = caption
        self._available = available
        self.calls: list[str] = []
        self.disabled_reason = lambda: "" if available else "unit-test: disabled"

    @property
    def is_available(self) -> bool:
        return self._available

    def describe(self, jpeg_path: str) -> str:
        self.calls.append(str(jpeg_path))
        return self._caption


def test_engine_describe_populates_result_description(patched_engine):
    engine, _captures, _detections, _tmp = patched_engine
    cap = _FakeCaptioner(caption="The user is holding a coffee mug.")
    engine.attach_captioner(cap)

    out = engine.look(reason="vision_describe:test", describe=True)

    assert out.ok is True
    assert out.description == "The user is holding a coffee mug."
    assert out.description_ms >= 0.0
    assert len(cap.calls) == 1
    # Summary ships the caption so it lands in the audit + log line.
    assert "caption=" in out.summary


def test_engine_describe_false_skips_captioner(patched_engine):
    engine, _c, _d, _t = patched_engine
    cap = _FakeCaptioner()
    engine.attach_captioner(cap)

    out = engine.look(reason="face_only", describe=False)

    assert out.ok is True
    assert out.description == ""
    assert cap.calls == []


def test_engine_recent_caption_returns_empty_without_call(patched_engine):
    engine, _c, _d, _t = patched_engine
    assert engine.recent_caption() == ""


def test_engine_recent_caption_returns_caption_within_window(patched_engine):
    engine, _c, _d, _t = patched_engine
    engine.attach_captioner(_FakeCaptioner(caption="There's a laptop on the desk."))
    engine.look(reason="wake:test", describe=True)

    assert engine.recent_caption() == "There's a laptop on the desk."
    assert engine.recent_caption(max_age_s=120.0) == "There's a laptop on the desk."


def test_engine_recent_caption_respects_stale_window(patched_engine):
    engine, _c, _d, _t = patched_engine
    engine.attach_captioner(_FakeCaptioner(caption="kitchen scene"))
    engine.look(reason="wake:test", describe=True)

    # Age the stashed caption by mutating the engine's private
    # timestamp — avoids having to freeze ``time.monotonic`` globally
    # and exercises the exact stale-window branch we care about.
    engine._last_caption_at = engine._last_caption_at - 10_000.0

    assert engine.recent_caption(max_age_s=60.0) == ""


def test_engine_emits_vision_caption_ready_event(patched_engine):
    engine, _c, _d, _t = patched_engine
    events: list[tuple[str, dict]] = []
    engine._emit = lambda event, **data: events.append((event, data))
    engine.attach_captioner(_FakeCaptioner(caption="a dog on a couch"))

    engine.look(reason="wake:test", describe=True)

    caption_events = [e for e in events if e[0] == "vision.caption.ready"]
    assert len(caption_events) == 1
    payload = caption_events[0][1]
    assert payload["caption"] == "a dog on a couch"
    assert payload["reason"] == "wake:test"


def test_engine_captioner_available_false_when_unavailable(patched_engine):
    engine, _c, _d, _t = patched_engine
    engine.attach_captioner(_FakeCaptioner(caption="x", available=False))
    assert engine.captioner_available is False
    # describe=True silently no-ops (no crash, no caption).
    out = engine.look(reason="test", describe=True)
    assert out.ok is True
    assert out.description == ""


def test_engine_captioner_metrics_returns_empty_when_no_captioner(patched_engine):
    engine, _c, _d, _t = patched_engine
    # No captioner wired -- must return a uniform-shape dict, not raise.
    metrics = engine.captioner_metrics()
    assert metrics == {"available": False, "reason": "captioner not wired"}


def test_engine_captioner_metrics_passthrough_to_real_captioner(patched_engine):
    engine, _c, _d, _t = patched_engine

    class _MetricsCap:
        is_available = True

        def metrics(self) -> dict:
            return {
                "model_path": "models/smolvlm-instruct-4bit",
                "is_loaded": True,
                "inference_count": 7,
                "load_ms": 1850.0,
            }

    engine.attach_captioner(_MetricsCap())
    m = engine.captioner_metrics()
    assert m["model_path"] == "models/smolvlm-instruct-4bit"
    assert m["inference_count"] == 7
    # Engine adds caption_max_age_s + available defaults so the consumer
    # always sees the effective injection budget.
    assert m["caption_max_age_s"] == engine._caption_max_age_s
    assert m["available"] is True


def test_engine_captioner_metrics_handles_metrics_method_failure(patched_engine):
    engine, _c, _d, _t = patched_engine

    class _ExplodingCap:
        is_available = True

        def metrics(self) -> dict:
            raise RuntimeError("metrics blew up")

    engine.attach_captioner(_ExplodingCap())
    m = engine.captioner_metrics()
    # Failure-mode dict -- must not raise upstream; consumer can react.
    assert m["available"] is True
    assert m.get("error") == "metrics_raised"
