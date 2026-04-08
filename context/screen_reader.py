"""
ATOM -- Screen Understanding Engine (Apple Vision + EasyOCR fallback).

Gives ATOM the ability to "see" what's on screen -- like JARVIS reading
Tony Stark's HUD displays.

OCR backends (priority order):
  1. Apple Vision framework (VNRecognizeTextRequest) — Neural Engine,
     ~100-300ms, zero model download, Live-Text quality.  macOS only.
  2. EasyOCR — CPU-based, ~2-5s, needs ~200MB model.  Cross-platform.
  3. Fallback — clipboard + window title.

Screen capture:
  macOS  — screencapture (native)
  Windows — PIL ImageGrab → PowerShell
  Linux  — PIL ImageGrab

Privacy: all processing is local, no screen data leaves the machine.

Owner: Satyam
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("atom.screen")

# ── Try Apple Vision framework ────────────────────────────────────────
_HAS_VISION = False
_Vision: Any = None
_Quartz: Any = None
_Foundation: Any = None

try:
    import Vision as _Vision        # type: ignore[import-untyped]
    import Quartz as _Quartz        # type: ignore[import-untyped]
    import Foundation as _Foundation  # type: ignore[import-untyped]
    _HAS_VISION = True
except ImportError:
    pass


def _vision_ocr(image_path: str) -> list[str]:
    """Extract text from image using Apple Vision framework.

    Runs on the Neural Engine — same engine as Live Text / Camera OCR.
    Returns list of recognized text strings.
    """
    url = _Foundation.NSURL.fileURLWithPath_(image_path)

    image_source = _Quartz.CGImageSourceCreateWithURL(url, None)
    if image_source is None:
        raise RuntimeError(f"Cannot create image source from {image_path}")

    cg_image = _Quartz.CGImageSourceCreateImageAtIndex(image_source, 0, None)
    if cg_image is None:
        raise RuntimeError(f"Cannot create CGImage from {image_path}")

    request = _Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(1)  # VNRequestTextRecognitionLevelAccurate
    request.setUsesLanguageCorrection_(True)

    handler = _Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
        cg_image, None,
    )

    success, error = handler.performRequests_error_([request], None)
    if not success:
        raise RuntimeError(f"Vision OCR failed: {error}")

    texts = []
    for observation in request.results():
        candidates = observation.topCandidates_(1)
        if candidates and len(candidates) > 0:
            text = str(candidates[0].string())
            confidence = float(candidates[0].confidence())
            if confidence > 0.3:
                texts.append(text)

    return texts


class ScreenReader:
    """Local OCR-based screen understanding."""

    def __init__(self, config: dict | None = None) -> None:
        self._config = (config or {}).get("screen_reader", {})
        self._enabled = self._config.get("enabled", True)
        self._ocr_engine: Any = None
        self._ocr_available = False
        self._ocr_backend: str = "none"
        self._last_capture: str = ""
        self._last_capture_time: float = 0.0
        self._init_ocr()

    def _init_ocr(self) -> None:
        if not self._enabled:
            return

        if _HAS_VISION and sys.platform == "darwin":
            self._ocr_available = True
            self._ocr_backend = "apple_vision"
            logger.info(
                "Screen reader: Apple Vision framework (Neural Engine OCR)"
            )
            return

        try:
            import easyocr
            self._ocr_engine = easyocr.Reader(
                ["en"], gpu=False, verbose=False,
            )
            self._ocr_available = True
            self._ocr_backend = "easyocr"
            logger.info("Screen reader: EasyOCR loaded (CPU)")
        except ImportError:
            logger.info(
                "No OCR backend available. "
                "On macOS: pip install pyobjc-framework-Vision. "
                "Cross-platform: pip install easyocr"
            )
        except Exception:
            logger.debug("OCR init failed", exc_info=True)

    @property
    def is_available(self) -> bool:
        return self._ocr_available

    @property
    def ocr_backend(self) -> str:
        return self._ocr_backend

    def capture_and_read(self) -> dict[str, Any]:
        """Take a screenshot and extract text via OCR."""
        if not self._enabled:
            return {"text": "", "error": "Screen reader disabled"}

        try:
            screenshot_path = self._take_screenshot()
            if not screenshot_path:
                return {"text": "", "error": "Screenshot failed"}

            if self._ocr_backend == "apple_vision":
                return self._vision_read(screenshot_path)
            elif self._ocr_backend == "easyocr" and self._ocr_engine is not None:
                return self._easyocr_read(screenshot_path)

            return self._fallback_read()

        except Exception as e:
            logger.debug("Screen capture failed: %s", e)
            return {"text": "", "error": str(e)[:200]}

    def _take_screenshot(self) -> str | None:
        """Capture the current screen to a temporary file."""
        tmp_path = Path("logs/screen_capture.png")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)

        if sys.platform == "darwin":
            try:
                result = subprocess.run(
                    ["screencapture", "-x", "-t", "png", str(tmp_path)],
                    capture_output=True, timeout=5,
                )
                if (result.returncode == 0
                        and tmp_path.exists()
                        and tmp_path.stat().st_size > 0):
                    logger.debug("Screenshot captured via screencapture")
                    return str(tmp_path)
                logger.warning(
                    "screencapture failed (grant Screen Recording permission "
                    "in System Settings > Privacy & Security)"
                )
            except Exception as exc:
                logger.debug("screencapture error: %s", exc)

        try:
            from PIL import ImageGrab
            img = ImageGrab.grab()
            img.save(str(tmp_path))
            logger.debug("Screenshot captured via PIL ImageGrab")
            return str(tmp_path)
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("PIL ImageGrab failed: %s", exc)

        if sys.platform == "win32":
            try:
                subprocess.run(
                    ["powershell", "-Command",
                     "Add-Type -AssemblyName System.Windows.Forms; "
                     "[System.Windows.Forms.Screen]::PrimaryScreen | "
                     "ForEach-Object { $bitmap = New-Object System.Drawing.Bitmap("
                     "$_.Bounds.Width, $_.Bounds.Height); "
                     "$graphics = [System.Drawing.Graphics]::FromImage($bitmap); "
                     "$graphics.CopyFromScreen($_.Bounds.Location, "
                     "[System.Drawing.Point]::Empty, $_.Bounds.Size); "
                     "$bitmap.Save('" + str(tmp_path) + "') }"],
                    capture_output=True, timeout=5,
                )
                if tmp_path.exists():
                    logger.debug("Screenshot captured via PowerShell")
                    return str(tmp_path)
            except Exception:
                pass

        return None

    def _vision_read(self, image_path: str) -> dict[str, Any]:
        """Extract text using Apple Vision framework (Neural Engine)."""
        t0 = time.monotonic()
        try:
            texts = _vision_ocr(image_path)
            full_text = "\n".join(texts)
            elapsed = (time.monotonic() - t0) * 1000

            self._last_capture = full_text
            self._last_capture_time = time.time()

            logger.info(
                "Vision OCR: %d text regions, %d chars in %.0fms",
                len(texts), len(full_text), elapsed,
            )
            return {
                "text": full_text[:5000],
                "regions": len(texts),
                "time_ms": elapsed,
                "method": "apple_vision",
            }
        except Exception as e:
            logger.warning("Vision OCR failed: %s", e)
            return self._fallback_read()

    def _easyocr_read(self, image_path: str) -> dict[str, Any]:
        """Extract text from screenshot using EasyOCR."""
        t0 = time.monotonic()
        try:
            results = self._ocr_engine.readtext(image_path)
            texts = [r[1] for r in results if r[2] > 0.3]
            full_text = " ".join(texts)

            elapsed = (time.monotonic() - t0) * 1000
            self._last_capture = full_text
            self._last_capture_time = time.time()

            logger.info(
                "EasyOCR: %d text regions, %d chars in %.0fms",
                len(texts), len(full_text), elapsed,
            )
            return {
                "text": full_text[:3000],
                "regions": len(texts),
                "time_ms": elapsed,
                "method": "easyocr",
            }
        except Exception as e:
            logger.debug("EasyOCR read failed: %s", e)
            return {"text": "", "error": str(e)[:200]}

    def _fallback_read(self) -> dict[str, Any]:
        """Fallback: read clipboard + active window title."""
        parts = []

        try:
            from context.context_engine import ContextEngine
            ctx = ContextEngine({})
            bundle = ctx.get_bundle()
            if bundle:
                if bundle.get("active_window"):
                    parts.append(f"Active window: {bundle['active_window']}")
                if bundle.get("clipboard"):
                    parts.append(f"Clipboard: {bundle['clipboard'][:500]}")
        except Exception:
            pass

        text = " | ".join(parts) if parts else "Unable to read screen content."
        return {
            "text": text,
            "method": "fallback",
            "regions": 0,
        }

    def get_screen_summary(self, max_words: int = 100) -> str:
        """Get a brief summary of what's currently on screen."""
        result = self.capture_and_read()
        text = result.get("text", "")
        if not text:
            return "I can't see the screen right now, Boss."

        words = text.split()
        if len(words) > max_words:
            text = " ".join(words[:max_words]) + "..."

        return f"On your screen I can see: {text}"

    @property
    def last_capture(self) -> str:
        return self._last_capture

    def shutdown(self) -> None:
        self._ocr_engine = None
        self._ocr_available = False
