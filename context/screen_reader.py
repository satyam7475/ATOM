"""
ATOM -- Screen Understanding Engine (Local OCR).

Gives ATOM the ability to "see" what's on screen -- like JARVIS reading
Tony Stark's HUD displays. Uses local OCR (EasyOCR or Tesseract) on
screenshots to extract text and understand screen content.

Capabilities:
    - "What's on my screen?" -> OCR + summarize visible text
    - "Read the error on screen" -> Extract error messages
    - Contextual awareness of active applications

Falls back to clipboard + window title when OCR is not available.

Privacy: all processing is local, no screen data leaves the machine.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any
import ctypes

logger = logging.getLogger("atom.screen")


class ScreenReader:
    """Local OCR-based screen understanding."""

    def __init__(self, config: dict | None = None) -> None:
        self._config = (config or {}).get("screen_reader", {})
        self._enabled = self._config.get("enabled", True)
        self._ocr_engine: Any = None
        self._ocr_available = False
        self._last_capture: str = ""
        self._last_capture_time: float = 0.0
        self._init_ocr()

    def _init_ocr(self) -> None:
        if not self._enabled:
            return
        try:
            import easyocr
            self._ocr_engine = easyocr.Reader(
                ["en"], gpu=False, verbose=False,
            )
            self._ocr_available = True
            logger.info("Screen reader: EasyOCR loaded (CPU)")
        except ImportError:
            logger.info(
                "EasyOCR not installed -- screen reading limited to clipboard. "
                "Install with: pip install easyocr"
            )
        except Exception:
            logger.debug("OCR init failed", exc_info=True)

    @property
    def is_available(self) -> bool:
        return self._ocr_available

    def capture_and_read(self) -> dict[str, Any]:
        """Take a screenshot and extract text via OCR."""
        if not self._enabled:
            return {"text": "", "error": "Screen reader disabled"}

        try:
            screenshot_path = self._take_screenshot()
            if not screenshot_path:
                return {"text": "", "error": "Screenshot failed"}

            if self._ocr_available and self._ocr_engine is not None:
                return self._ocr_read(screenshot_path)

            return self._fallback_read()

        except Exception as e:
            logger.debug("Screen capture failed: %s", e)
            return {"text": "", "error": str(e)[:200]}

    def _take_screenshot(self) -> str | None:
        """Capture the current screen to a temporary file."""
        try:
            from PIL import ImageGrab
            img = ImageGrab.grab()
            tmp_path = Path("logs/screen_capture.png")
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(tmp_path))
            return str(tmp_path)
        except ImportError:
            try:
                import subprocess
                tmp_path = Path("logs/screen_capture.png")
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    ["powershell", "-Command",
                     f"Add-Type -AssemblyName System.Windows.Forms; "
                     f"[System.Windows.Forms.Screen]::PrimaryScreen | "
                     f"ForEach-Object {{ $bitmap = New-Object System.Drawing.Bitmap($_.Bounds.Width, $_.Bounds.Height); "
                     f"$graphics = [System.Drawing.Graphics]::FromImage($bitmap); "
                     f"$graphics.CopyFromScreen($_.Bounds.Location, [System.Drawing.Point]::Empty, $_.Bounds.Size); "
                     f"$bitmap.Save('{tmp_path}') }}"],
                    capture_output=True, timeout=5,
                )
                if tmp_path.exists():
                    return str(tmp_path)
            except Exception:
                pass
        except Exception:
            pass
        return None

    def _ocr_read(self, image_path: str) -> dict[str, Any]:
        """Extract text from screenshot using OCR."""
        t0 = time.monotonic()
        try:
            results = self._ocr_engine.readtext(image_path)
            texts = [r[1] for r in results if r[2] > 0.3]
            full_text = " ".join(texts)

            elapsed = (time.monotonic() - t0) * 1000
            self._last_capture = full_text
            self._last_capture_time = time.time()

            logger.info("Screen OCR: %d text regions, %d chars in %.0fms",
                        len(texts), len(full_text), elapsed)

            return {
                "text": full_text[:3000],
                "regions": len(texts),
                "time_ms": elapsed,
                "method": "easyocr",
            }
        except Exception as e:
            logger.debug("OCR read failed: %s", e)
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
