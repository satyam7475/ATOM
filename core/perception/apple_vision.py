"""ATOM -- Apple Vision detection on a JPEG path.

Thin wrapper over Vision's ``VNDetectFaceRectanglesRequest`` (and
optionally barcode / saliency requests). Runs entirely on the Neural
Engine — same silicon Apple uses for Live Text and the Camera app's
face boxes — so it costs ~10-50ms per frame and **adds zero pressure
to the GPU memory pool the MLX 7B brain occupies**.

We deliberately do *not* run any LLM here. The user's standing
direction is "stay really on one model, 7B"; classifying a frame as
"there is a face" → "Welcome back, Boss" is a tiny rule, not a
visual-LLM job. When the user later asks for richer scene
understanding ("describe this graph"), we'll reach for a real VLM.

Public surface
--------------
* :class:`VisionResult` — dataclass returned by :func:`detect`.
* :func:`detect` — synchronous, runs requested detectors on a JPEG
  file path; never raises.

Owner: Satyam
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom.perception.vision")


HAS_VISION = False
_Vision: Any = None
_Quartz: Any = None
_Foundation: Any = None
try:
    import Vision as _Vision  # type: ignore[import-untyped]
    import Quartz as _Quartz  # type: ignore[import-untyped]
    import Foundation as _Foundation  # type: ignore[import-untyped]
    HAS_VISION = True
except ImportError:
    pass


@dataclass
class VisionResult:
    """Aggregated output of one frame's worth of Apple Vision work."""

    ok: bool = False
    detection_ms: float = 0.0
    faces: int = 0
    face_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    barcodes: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def summary(self) -> str:
        """Short human-readable description; safe for logs and TTS."""
        if not self.ok:
            return f"vision unavailable ({self.error})" if self.error else "vision unavailable"
        bits: list[str] = []
        if self.faces == 0:
            bits.append("no faces")
        elif self.faces == 1:
            bits.append("1 face")
        else:
            bits.append(f"{self.faces} faces")
        if self.barcodes:
            bits.append(f"{len(self.barcodes)} barcodes")
        return ", ".join(bits)


def _load_cgimage(image_path: str) -> Any | None:
    """Load a JPEG/PNG into a CGImage. Returns None on failure."""
    if not HAS_VISION:
        return None
    try:
        url = _Foundation.NSURL.fileURLWithPath_(image_path)
        src = _Quartz.CGImageSourceCreateWithURL(url, None)
        if src is None:
            return None
        return _Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    except Exception:
        logger.debug("CGImage load failed for %s", image_path, exc_info=True)
        return None


def detect(
    image_path: str | Path,
    *,
    detect_faces: bool = True,
    detect_barcodes: bool = False,
) -> VisionResult:
    """Run requested Vision detectors on the JPEG/PNG at *image_path*.

    Returns ``VisionResult`` even when Vision is unavailable; check
    ``ok`` before reading downstream fields. Never raises.
    """
    if not HAS_VISION:
        return VisionResult(error="Vision framework not available")

    p = str(image_path)
    if not Path(p).exists():
        return VisionResult(error=f"image not found: {p}")

    cg_image = _load_cgimage(p)
    if cg_image is None:
        return VisionResult(error="CGImageSource could not decode file")

    requests: list[Any] = []
    face_req: Any = None
    barcode_req: Any = None
    try:
        if detect_faces:
            face_req = _Vision.VNDetectFaceRectanglesRequest.alloc().init()
            requests.append(face_req)
        if detect_barcodes:
            barcode_req = _Vision.VNDetectBarcodesRequest.alloc().init()
            requests.append(barcode_req)
    except Exception as exc:
        return VisionResult(error=f"request alloc failed: {exc}")

    if not requests:
        return VisionResult(ok=True, error="no detectors requested")

    t0 = time.perf_counter()
    try:
        handler = _Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
            cg_image, None,
        )
        success, error = handler.performRequests_error_(requests, None)
        if not success:
            err_str = str(error) if error is not None else "unknown"
            return VisionResult(error=f"performRequests failed: {err_str}")
    except Exception as exc:
        return VisionResult(error=f"handler exception: {exc}")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    result = VisionResult(ok=True, detection_ms=round(elapsed_ms, 1))

    if face_req is not None:
        try:
            observations = list(face_req.results() or [])
        except Exception:
            observations = []
        result.faces = len(observations)
        for obs in observations:
            try:
                rect = obs.boundingBox()
                # Vision returns normalized coords (0..1) with origin
                # bottom-left. Keep as-is so callers can re-project as
                # needed; tests should not assume top-left origin.
                result.face_boxes.append((
                    float(rect.origin.x),
                    float(rect.origin.y),
                    float(rect.size.width),
                    float(rect.size.height),
                ))
            except Exception:
                continue

    if barcode_req is not None:
        try:
            observations = list(barcode_req.results() or [])
        except Exception:
            observations = []
        for obs in observations:
            try:
                payload = obs.payloadStringValue()
                if payload:
                    result.barcodes.append(str(payload))
            except Exception:
                continue

    return result
