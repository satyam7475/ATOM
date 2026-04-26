"""
ATOM -- Presence Sampler (Phase G2).

Periodically captures one webcam frame and runs Apple Vision face
detection so the cognitive layer knows whether the user is *actually
at the laptop* and roughly what their face quality looks like
(present, away, multiple people, low quality / dark room, etc.).

This is intentionally tiny:

* **One frame, every N seconds** (default 30s).
* Uses the existing :func:`core.perception.camera_capture.capture_jpeg`
  + :func:`core.perception.apple_vision.detect` building blocks --
  no new pyobjc surface.
* **Suppressed** while ATOM is speaking, listening, or thinking, and
  while ``vision_look`` (the user-facing camera tool) is in flight, so
  we never steal the camera mid-turn.
* Emits ``presence.snapshot`` on the bus with a stable schema for
  downstream consumers (mood inference, scene context, suggester).

Failure posture: never raises. If the camera or Vision binding is
missing we just log once and stop the sampler.

Owner: Satyam
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus

logger = logging.getLogger("atom.perception.presence")


# ── data classes ────────────────────────────────────────────────────


@dataclass(slots=True)
class PresenceSnapshot:
    """One sample worth of presence/face information."""

    ts: float
    present: bool
    face_count: int
    quality: str  # "good" | "low_light" | "blurry" | "no_camera" | "unknown"
    face_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    capture_ms: float = 0.0
    detection_ms: float = 0.0
    error: str = ""
    camera: str = ""

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["face_boxes"] = [tuple(b) for b in self.face_boxes]
        return d


# ── sampler ────────────────────────────────────────────────────────


class PresenceSampler:
    """Periodic camera-driven presence detector.

    Two cooperating callables are pulled in via DI so unit tests can
    skip pyobjc entirely:

    * ``capture_fn`` -> ``CaptureResult`` (jpeg saved to disk)
    * ``detect_fn``  -> ``VisionResult``  (faces + boxes)

    By default these point at the real ``camera_capture`` /
    ``apple_vision`` modules.
    """

    __slots__ = (
        "_bus", "_interval_s", "_min_interval_s",
        "_capture_fn", "_detect_fn", "_camera_chooser",
        "_state_provider", "_busy_provider",
        "_loop_task", "_running", "_attached",
        "_last_snapshot", "_last_emit_at", "_consecutive_errors",
        "_total_samples", "_total_errors", "_total_skips",
        "_event",
    )

    def __init__(
        self,
        bus: "AsyncEventBus",
        *,
        interval_s: float = 30.0,
        min_interval_s: float = 8.0,
        capture_fn: Callable[..., Any] | None = None,
        detect_fn: Callable[..., Any] | None = None,
        camera_chooser: Callable[[], Any] | None = None,
        state_provider: Callable[[], str] | None = None,
        busy_provider: Callable[[], bool] | None = None,
    ) -> None:
        self._bus = bus
        self._interval_s = max(float(min_interval_s), float(interval_s))
        self._min_interval_s = float(min_interval_s)
        self._capture_fn = capture_fn
        self._detect_fn = detect_fn
        self._camera_chooser = camera_chooser
        self._state_provider = state_provider
        self._busy_provider = busy_provider
        self._loop_task: asyncio.Task[None] | None = None
        self._running = False
        self._attached = False
        self._last_snapshot: PresenceSnapshot | None = None
        self._last_emit_at = 0.0
        self._consecutive_errors = 0
        self._total_samples = 0
        self._total_errors = 0
        self._total_skips = 0
        self._event: asyncio.Event | None = None

    # ── lifecycle ───────────────────────────────────────────────

    def start(self) -> None:
        """Schedule the periodic sampler on the running loop."""
        if self._running:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("PresenceSampler.start() called outside event loop")
            return
        self._event = asyncio.Event()
        self._running = True
        self._loop_task = loop.create_task(self._run())
        logger.info(
            "PresenceSampler started (interval=%.1fs, min=%.1fs)",
            self._interval_s, self._min_interval_s,
        )

    async def stop(self) -> None:
        self._running = False
        if self._event is not None:
            self._event.set()
        task = self._loop_task
        self._loop_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    @property
    def metrics(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "interval_s": self._interval_s,
            "samples": self._total_samples,
            "errors": self._total_errors,
            "skips": self._total_skips,
            "consecutive_errors": self._consecutive_errors,
            "last_snapshot": self._last_snapshot.as_dict() if self._last_snapshot else None,
        }

    @property
    def last_snapshot(self) -> PresenceSnapshot | None:
        return self._last_snapshot

    # ── one-shot path (handy for tests + manual triggers) ──────

    async def sample_once(self) -> PresenceSnapshot:
        """Run a single capture + detect pass and emit on the bus."""
        snapshot = await asyncio.get_running_loop().run_in_executor(
            None, self._sample_blocking,
        )
        self._publish(snapshot)
        return snapshot

    # ── internal: periodic loop ─────────────────────────────────

    async def _run(self) -> None:
        try:
            while self._running:
                if self._should_skip():
                    self._total_skips += 1
                    await self._sleep_interval(short=True)
                    continue
                try:
                    snapshot = await asyncio.get_running_loop() \
                        .run_in_executor(None, self._sample_blocking)
                    self._publish(snapshot)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("presence sampler unexpected failure")
                    self._consecutive_errors += 1
                    self._total_errors += 1
                # Back off when the camera keeps refusing.
                if self._consecutive_errors >= 5:
                    logger.warning(
                        "presence sampler: 5 consecutive errors, pausing 5x",
                    )
                    await self._sleep_interval(multiplier=5)
                    self._consecutive_errors = 0
                    continue
                await self._sleep_interval()
        except asyncio.CancelledError:
            pass

    async def _sleep_interval(
        self, *, short: bool = False, multiplier: float = 1.0,
    ) -> None:
        if not self._event:
            return
        target = self._min_interval_s if short else self._interval_s
        target *= max(0.5, multiplier)
        try:
            await asyncio.wait_for(self._event.wait(), timeout=target)
        except asyncio.TimeoutError:
            return

    # ── internal: sample (blocking) ────────────────────────────

    def _sample_blocking(self) -> PresenceSnapshot:
        """Capture + detect, returning a snapshot. Never raises."""
        start = time.monotonic()
        cap, det = self._lazy_imports()
        if cap is None or det is None:
            self._total_errors += 1
            self._consecutive_errors += 1
            return PresenceSnapshot(
                ts=time.time(), present=False, face_count=0,
                quality="no_camera",
                error="pyobjc not available",
            )

        camera = self._select_camera(cap)
        if camera is None:
            self._total_errors += 1
            self._consecutive_errors += 1
            return PresenceSnapshot(
                ts=time.time(), present=False, face_count=0,
                quality="no_camera",
                error="no camera discovered",
            )

        tmp = Path(tempfile.gettempdir()) / "atom_presence.jpg"
        capture = cap.capture_jpeg(camera, out_path=tmp, timeout_s=2.5)
        capture_ms = float(getattr(capture, "capture_ms", 0.0) or 0.0)
        if not getattr(capture, "ok", False):
            self._total_errors += 1
            self._consecutive_errors += 1
            return PresenceSnapshot(
                ts=time.time(), present=False, face_count=0,
                quality="no_camera",
                capture_ms=capture_ms,
                error=str(getattr(capture, "error", "")) or "capture failed",
                camera=str(camera),
            )

        vision = det.detect(
            getattr(capture, "saved_path", str(tmp)),
            detect_faces=True, detect_barcodes=False,
        )
        detection_ms = float(getattr(vision, "detection_ms", 0.0) or 0.0)
        face_count = int(getattr(vision, "faces", 0) or 0)
        face_boxes = list(getattr(vision, "face_boxes", []) or [])

        quality = self._infer_quality(
            ok=bool(getattr(vision, "ok", False)),
            faces=face_count,
            capture_ms=capture_ms,
        )

        self._total_samples += 1
        self._consecutive_errors = 0
        snapshot = PresenceSnapshot(
            ts=time.time(),
            present=face_count >= 1,
            face_count=face_count,
            face_boxes=face_boxes,
            quality=quality,
            capture_ms=capture_ms,
            detection_ms=detection_ms,
            camera=str(camera),
        )
        logger.debug(
            "presence sample %s in %.0f+%.0fms",
            snapshot.quality, capture_ms, detection_ms,
        )
        # Flag long-running samples so wiring can re-tune interval.
        elapsed_ms = (time.monotonic() - start) * 1000.0
        if elapsed_ms > 4000:
            logger.warning(
                "presence sample took %.0fms -- camera contention?",
                elapsed_ms,
            )
        return snapshot

    # ── helpers ─────────────────────────────────────────────────

    def _lazy_imports(self) -> tuple[Any, Any]:
        cap = self._capture_fn
        det = self._detect_fn
        if cap is None or det is None:
            try:
                from core.perception import apple_vision, camera_capture
                cap = cap or camera_capture
                det = det or apple_vision
            except Exception:
                logger.warning("PresenceSampler: perception modules unavailable")
                return None, None
        return cap, det

    def _select_camera(self, cap_module: Any) -> Any:
        chooser = self._camera_chooser
        if chooser is not None:
            try:
                return chooser()
            except Exception:
                logger.exception("camera_chooser failed")
        # Resolve the camera-discovery callable defensively. The
        # ``camera_capture`` module exposes :func:`list_cameras`; older
        # injection points expected ``discover_cameras``. Accept either
        # so a stub harness or future rename can't silently break the
        # presence sampler.
        discover = getattr(cap_module, "list_cameras", None)
        if discover is None:
            discover = getattr(cap_module, "discover_cameras", None)
        if discover is None:
            return None
        try:
            cams = discover()
        except Exception:
            return None
        try:
            return cap_module.choose_preferred(cams, preferred="builtin")
        except Exception:
            return None

    @staticmethod
    def _infer_quality(*, ok: bool, faces: int, capture_ms: float) -> str:
        if not ok:
            return "unknown"
        if faces == 0:
            return "no_face"
        if capture_ms > 1500:
            return "low_light"
        return "good"

    def _should_skip(self) -> bool:
        if self._busy_provider is not None:
            try:
                if bool(self._busy_provider()):
                    return True
            except Exception:
                pass
        if self._state_provider is None:
            return False
        try:
            state = (self._state_provider() or "").lower()
        except Exception:
            return False
        return state in ("speaking", "thinking", "listening", "error_recovery")

    def _publish(self, snapshot: PresenceSnapshot) -> None:
        self._last_snapshot = snapshot
        self._last_emit_at = time.monotonic()
        try:
            self._bus.emit_long(
                "presence.snapshot",
                **snapshot.as_dict(),
            )
        except Exception:
            logger.exception("presence emit failed")


__all__ = ["PresenceSampler", "PresenceSnapshot"]
