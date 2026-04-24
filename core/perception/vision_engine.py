"""ATOM -- Vision engine (camera capture + Apple Vision detection).

High-level facade the runtime calls. Owns:

* the configured "preferred camera" choice (built-in vs Continuity
  iPhone vs explicit UID),
* a single-slot lock so two callers can't race the AVCapture session
  at the same time (Apple's stack does NOT cope well with concurrent
  ``AVCaptureSession`` startups on the same device),
* the audit log (one JSONL line per capture),
* an optional event-bus hook so other modules can subscribe to
  ``vision.frame.captured`` / ``vision.caption.ready`` without
  depending on this module directly,
* an optional :class:`VLMCaptioner` (SmolVLM-Instruct-4bit via mlx-vlm
  by default, model-agnostic via config). When the captioner is
  attached, ``look(describe=True)`` returns a one-sentence natural-
  language caption alongside the Apple Vision
  face/barcode counts.

Responsibilities
----------------
* :py:meth:`look` -- one synchronous capture+detect(+describe) cycle.
  Used by the boot face check, the on-demand ``vision_look`` and
  ``vision_describe`` tools, and the on-wake ambient capture.
* :py:meth:`list_cameras_human` -- pretty list for the boot banner.
* :py:meth:`disabled_reason` -- short string explaining why the engine
  is offline (config flag off, AVFoundation missing, no cameras), or
  empty when ready.
* :py:meth:`recent_caption` -- most recent natural-language caption
  from ``look(describe=True)`` if it's fresh enough to be relevant.
  Called by the router when it builds ``context_bundle`` for the LLM.

The captioner is wired only when ``vision.vlm.enabled`` is true in
config, so ATOM still boots into its legacy face-detection-only mode
when the user hasn't fetched the VLM weights yet.

Owner: Satyam
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.perception import apple_vision, camera_capture
from core.perception.apple_vision import VisionResult
from core.perception.camera_capture import CameraInfo, CaptureResult
from core.perception.vision_audit import VisionAuditLog

logger = logging.getLogger("atom.perception.engine")


# The ``summary`` string we hand back to TTS / log lines is kept short
# on purpose so it lands well in a spoken sentence ("I see one face")
# and also reads well in the boot banner ("camera ready, no face yet").


@dataclass
class VisionLookResult:
    """Outcome of one ``VisionEngine.look`` call."""

    ok: bool = False
    summary: str = ""
    error: str = ""
    camera: CameraInfo | None = None
    capture: CaptureResult | None = None
    vision: VisionResult | None = None
    capture_ms: float = 0.0
    detection_ms: float = 0.0
    description: str = ""
    description_ms: float = 0.0

    @property
    def faces(self) -> int:
        return self.vision.faces if self.vision else 0

    @property
    def saved_path(self) -> str:
        return self.capture.saved_path if self.capture else ""


class VisionEngine:
    """Thread-safe wrapper around ``camera_capture`` + ``apple_vision``."""

    __slots__ = (
        "_enabled",
        "_preferred",
        "_explicit_uid",
        "_audit",
        "_emit",
        "_capture_lock",
        "_last_capture_at",
        "_min_gap_s",
        "_face_timeout_s",
        "_captioner",
        "_caption_max_age_s",
        "_last_caption",
        "_last_caption_at",
        "_last_caption_reason",
        "_caption_lock",
    )

    def __init__(
        self,
        *,
        enabled: bool = False,
        preferred_camera: str = "auto",
        explicit_uid: str | None = None,
        audit_log_path: str | Path | None = None,
        emit: Any = None,
        min_gap_s: float = 1.0,
        capture_timeout_s: float = 3.5,
        captioner: Any = None,
        caption_max_age_s: float = 60.0,
    ) -> None:
        self._enabled = bool(enabled)
        self._preferred = (preferred_camera or "auto").strip().lower()
        self._explicit_uid = (explicit_uid or "").strip() or None
        self._audit = VisionAuditLog(audit_log_path)
        # ``emit`` is a sync callable: ``emit(event, **data)``.  We
        # accept any duck-typed bus that exposes ``emit_fast`` /
        # ``emit`` / a plain function — the runtime hands us the bus
        # method directly so this module never imports the bus class.
        self._emit = emit
        # Same camera, two near-simultaneous calls — AVCapture will
        # error out. The lock is non-blocking from Python's POV but
        # callers should use ``try_look`` if they want to short-circuit.
        self._capture_lock = threading.Lock()
        self._last_capture_at: float = 0.0
        self._min_gap_s = max(0.0, float(min_gap_s))
        self._face_timeout_s = max(0.5, float(capture_timeout_s))
        # Optional VLM captioner (SmolVLM-Instruct-4bit via mlx-vlm by
        # default; model-agnostic). When ``None``,
        # ``look(describe=True)`` becomes a no-op on the description
        # field but still runs the normal capture + Apple
        # Vision pass. ``_captioner`` is duck-typed (we only require
        # ``describe(jpeg_path) -> str`` and ``is_available``) so
        # tests can swap in a fake without importing mlx-vlm.
        self._captioner = captioner
        self._caption_max_age_s = max(1.0, float(caption_max_age_s))
        self._last_caption: str = ""
        self._last_caption_at: float = 0.0
        self._last_caption_reason: str = ""
        self._caption_lock = threading.Lock()

    # ── status ────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """Allow runtime toggling (e.g. user says "stop watching")."""
        self._enabled = bool(enabled)

    def disabled_reason(self) -> str:
        """Return why the engine is offline; empty string when ready."""
        if not self._enabled:
            return "disabled in config (vision.enabled=false)"
        if not camera_capture.HAS_AVFOUNDATION:
            return "pyobjc-framework-AVFoundation not importable"
        if not apple_vision.HAS_VISION:
            return "pyobjc-framework-Vision not importable"
        cams = camera_capture.list_cameras()
        if not cams:
            return "no cameras visible (try unlocking iPhone for Continuity Camera)"
        return ""

    def attach_captioner(self, captioner: Any) -> None:
        """Wire the VLM captioner after construction.

        Safe to call multiple times; the most recent non-None captioner
        wins.  Pass ``None`` to detach (e.g. when the user disables
        the VLM at runtime).
        """
        self._captioner = captioner

    @property
    def captioner_available(self) -> bool:
        """True if ``look(describe=True)`` will actually produce text."""
        cap = self._captioner
        if cap is None:
            return False
        is_avail = getattr(cap, "is_available", True)
        return bool(is_avail)

    def captioner_disabled_reason(self) -> str:
        """Short reason a describe call will return empty text, or ``""``."""
        cap = self._captioner
        if cap is None:
            return "vlm captioner not wired (vision.vlm.enabled=false)"
        reason_fn = getattr(cap, "disabled_reason", None)
        if callable(reason_fn):
            try:
                return str(reason_fn() or "")
            except Exception:
                logger.debug(
                    "captioner.disabled_reason raised", exc_info=True,
                )
                return ""
        return "" if self.captioner_available else "vlm captioner unavailable"

    def captioner_metrics(self) -> dict[str, Any]:
        """Pass-through to the wired captioner's ``metrics()``.

        Returns ``{"available": False}`` when no captioner is wired so
        callers (status snapshot, web dashboard) get a uniform shape
        regardless of config. Defensive against captioners that don't
        implement ``metrics()`` (e.g. a fake injected from tests).
        """
        cap = self._captioner
        if cap is None:
            return {"available": False, "reason": "captioner not wired"}
        metrics_fn = getattr(cap, "metrics", None)
        if not callable(metrics_fn):
            return {"available": self.captioner_available}
        try:
            data = metrics_fn() or {}
        except Exception:
            logger.debug("captioner.metrics raised", exc_info=True)
            return {"available": self.captioner_available, "error": "metrics_raised"}
        if not isinstance(data, dict):
            return {"available": self.captioner_available}
        # Mirror the engine-side staleness window so the consumer sees
        # the *effective* injection budget, not just the captioner's
        # internal state.
        data.setdefault("caption_max_age_s", self._caption_max_age_s)
        data.setdefault("available", self.captioner_available)
        return data

    def recent_caption(
        self, max_age_s: float | None = None,
    ) -> str:
        """Return the most recent caption if it's still fresh, else ``""``.

        Used by the router's prompt builder to inject ``visual_context``
        into the next LLM turn. ``max_age_s`` defaults to the engine's
        configured staleness window; pass a smaller value for
        interactive paths that want only very-recent perception.
        """
        with self._caption_lock:
            caption = self._last_caption
            captured_at = self._last_caption_at
        if not caption:
            return ""
        window = (
            self._caption_max_age_s if max_age_s is None
            else max(0.0, float(max_age_s))
        )
        if (time.monotonic() - captured_at) > window:
            return ""
        return caption

    def list_cameras_human(self) -> list[str]:
        """Pretty-printed device list for log banners."""
        cams = camera_capture.list_cameras()
        return [f"{c.name} [{c.kind}]" for c in cams]

    def choose_camera(self) -> CameraInfo | None:
        """Return the camera we'd use for the next call, or None."""
        return camera_capture.choose_preferred(
            preferred=self._preferred,
            explicit_uid=self._explicit_uid,
        )

    # ── core call ─────────────────────────────────────────────────

    def look(
        self,
        *,
        reason: str = "manual",
        save_path: str | Path | None = None,
        detect_faces: bool = True,
        detect_barcodes: bool = False,
        describe: bool = False,
    ) -> "VisionLookResult":
        """One synchronous capture + Vision (+ optional VLM) pass.

        Always returns; never raises. Audit-logs every attempt,
        success or failure. When ``describe`` is true *and* a VLM
        captioner is wired and available, the result's ``description``
        field is populated with a one-sentence natural-language caption
        generated from the captured JPEG; the caption is also stashed
        on the engine for :meth:`recent_caption`. If no captioner is
        wired, ``describe`` is a silent no-op (still capture, still
        face-detect) — so this method can be called from ambient
        triggers without defensive checks at the call site.
        """
        if not self._enabled:
            return _disabled("vision engine disabled in config", reason=reason, audit=self._audit)
        block_reason = self.disabled_reason()
        if block_reason:
            return _disabled(block_reason, reason=reason, audit=self._audit)

        # Soft rate-limit on consecutive captures from the same engine
        # — protects against tool-call loops that hit ``vision_look``
        # in a tight retry loop and starve the camera bus.
        now = time.monotonic()
        if (now - self._last_capture_at) < self._min_gap_s:
            wait_for = self._min_gap_s - (now - self._last_capture_at)
            return _disabled(
                f"throttled: try again in {wait_for:.1f}s",
                reason=reason,
                audit=self._audit,
            )

        camera = self.choose_camera()
        if camera is None:
            return _disabled(
                "no camera available", reason=reason, audit=self._audit,
            )

        # Acquire the AVCapture lock without blocking forever — if a
        # previous call is still warming up the sensor, return a soft
        # error so the caller can decide whether to retry.
        if not self._capture_lock.acquire(timeout=self._face_timeout_s + 0.5):
            return _disabled(
                "capture session busy", reason=reason,
                audit=self._audit, source=camera.name, source_kind=camera.kind,
            )

        try:
            self._last_capture_at = time.monotonic()
            cap = camera_capture.capture_jpeg(
                camera, out_path=save_path, timeout_s=self._face_timeout_s,
            )
            if not cap.ok:
                self._audit.record(
                    reason=reason, source=camera.name, source_kind=camera.kind,
                    capture_ms=cap.capture_ms, error=cap.error,
                )
                return VisionLookResult(
                    ok=False, error=cap.error or "capture failed",
                    camera=camera, capture_ms=cap.capture_ms,
                )

            vision = apple_vision.detect(
                cap.saved_path,
                detect_faces=detect_faces,
                detect_barcodes=detect_barcodes,
            )

            description = ""
            description_ms = 0.0
            if describe:
                description, description_ms = self._describe_if_possible(
                    jpeg_path=cap.saved_path, reason=reason,
                )

            summary = self._build_summary(camera, cap, vision, description)
            self._audit.record(
                reason=reason, source=camera.name, source_kind=camera.kind,
                capture_ms=cap.capture_ms,
                detection_ms=vision.detection_ms, faces=vision.faces,
                summary=summary, saved_path=cap.saved_path,
                error=vision.error,
            )
            self._emit_safe(
                "vision.frame.captured",
                reason=reason,
                source=camera.name,
                source_kind=camera.kind,
                faces=vision.faces,
                capture_ms=cap.capture_ms,
                detection_ms=vision.detection_ms,
            )
            return VisionLookResult(
                ok=vision.ok and not vision.error,
                summary=summary,
                camera=camera,
                capture=cap,
                vision=vision,
                capture_ms=cap.capture_ms,
                detection_ms=vision.detection_ms,
                error=vision.error,
                description=description,
                description_ms=description_ms,
            )
        finally:
            self._capture_lock.release()

    def _describe_if_possible(
        self, *, jpeg_path: str, reason: str,
    ) -> tuple[str, float]:
        """Run the VLM captioner on ``jpeg_path`` and stash the result.

        Returns ``(description, elapsed_ms)``. Empty description +
        ``0.0`` ms when the captioner is unavailable. Never raises.
        """
        captioner = self._captioner
        if captioner is None:
            return "", 0.0
        if not getattr(captioner, "is_available", False):
            return "", 0.0
        t0 = time.perf_counter()
        try:
            caption = captioner.describe(jpeg_path)
        except Exception:
            logger.debug("VLM describe raised on %s", jpeg_path, exc_info=True)
            caption = ""
        dt_ms = (time.perf_counter() - t0) * 1000.0

        caption = (caption or "").strip()
        if caption:
            with self._caption_lock:
                self._last_caption = caption
                self._last_caption_at = time.monotonic()
                self._last_caption_reason = reason
            self._emit_safe(
                "vision.caption.ready",
                reason=reason,
                caption=caption,
                description_ms=dt_ms,
            )
        return caption, dt_ms

    # ── helpers ───────────────────────────────────────────────────

    def _build_summary(
        self,
        camera: CameraInfo,
        cap: CaptureResult,
        vision: VisionResult,
        description: str = "",
    ) -> str:
        bits = [f"camera={camera.name} ({camera.kind})"]
        if vision.ok:
            bits.append(vision.summary)
        elif vision.error:
            bits.append(f"vision error: {vision.error}")
        if description:
            # Keep the summary log-friendly by truncating long VLM
            # captions — the full text is still on the result's
            # ``description`` field for any caller that needs it.
            short = description if len(description) <= 120 else description[:117] + "..."
            bits.append(f"caption={short!r}")
        bits.append(f"{cap.capture_ms:.0f}ms capture")
        if vision.detection_ms:
            bits.append(f"{vision.detection_ms:.0f}ms vision")
        return ", ".join(bits)

    def _emit_safe(self, event: str, **data: Any) -> None:
        emit = self._emit
        if emit is None:
            return
        try:
            emit(event, **data)
        except Exception:
            logger.debug("vision emit failed for %s", event, exc_info=True)


def _disabled(
    error: str,
    *,
    reason: str,
    audit: VisionAuditLog,
    source: str = "",
    source_kind: str = "unknown",
) -> VisionLookResult:
    """Helper so the engine never returns a half-populated result."""
    audit.record(
        reason=reason, source=source, source_kind=source_kind,
        error=error,
    )
    return VisionLookResult(ok=False, error=error)
