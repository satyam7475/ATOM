"""ATOM perception -- physical screen capture + VLM/Cloud description.

Two backends in priority order:
  1. **Cloud Vision** (``gemini_client``) -- when configured, deepest layout
     understanding. Used for screen-content reasoning queries.
  2. **Local VLM** (``vlm_captioner``, default SmolVLM-Instruct-4bit on
     MLX) -- always-available on-device fallback. Sprint A4 wired this
     in; before Sprint A4 the fallback was a hard-coded
     ``"Vision subsystem fallback: Screen captured locally, but Gemini
     Client offline."`` string that bypassed the SmolVLM that was
     already loaded at boot (atomLogs.txt L417, L464, L485).

Both paths are best-effort -- when neither is available we return a
single, plainly-worded sentence so downstream TTS does not speak a stack
trace.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time

logger = logging.getLogger("atom.perception.vision")


class ScreenReader:
    """Mac-native screencapture pipe to either a Cloud Vision client
    (Gemini) or an on-device VLM captioner (SmolVLM by default)."""

    def __init__(self, gemini_client=None, vlm_captioner=None) -> None:
        self.gemini = gemini_client
        # New in Sprint A4: when ``gemini`` is None we fall through to
        # the local VLM captioner (already loaded by main.py at boot).
        self.vlm_captioner = vlm_captioner
        self.temp_dir = "/tmp/atom_vision"
        os.makedirs(self.temp_dir, exist_ok=True)

    # ── helpers ──────────────────────────────────────────────────

    def capture_screen(self) -> str:
        """Take a physical macOS screenshot; return the file path or
        ``""`` on failure. ``-x`` silences the shutter, ``-C`` includes
        the cursor for richer context."""
        ts = int(time.time())
        filepath = f"{self.temp_dir}/screen_{ts}.png"
        try:
            subprocess.run(["screencapture", "-x", "-C", filepath], check=True)
            logger.info("Screen physically captured -> %s", filepath)
            return filepath
        except Exception as exc:
            logger.error("Failed to capture screen: %s", exc)
            return ""

    def _captioner_available(self) -> bool:
        cap = self.vlm_captioner
        if cap is None:
            return False
        # ``VLMCaptioner.is_available`` is the canonical predicate; fall
        # back to a callable-shaped duck-type so test doubles work.
        is_avail = getattr(cap, "is_available", None)
        if callable(is_avail):
            try:
                return bool(is_avail())
            except Exception:
                logger.debug("vlm_captioner.is_available raised", exc_info=True)
                return False
        if isinstance(is_avail, bool):
            return is_avail
        return hasattr(cap, "describe") or hasattr(cap, "describe_image")

    # ── public API ───────────────────────────────────────────────

    async def analyze_screen(
        self,
        query: str = "Analyze the UI layout and read the text on the screen.",
    ) -> str:
        """Pipe a fresh screenshot to whichever vision backend is
        available. Cleans up the screenshot file on success.

        Returns a natural-language description of the screen. NEVER
        returns the legacy ``"Gemini Client offline"`` string -- if no
        backend is reachable we return a short plain sentence the TTS
        can speak as-is.
        """
        filepath = self.capture_screen()
        if not filepath or not os.path.exists(filepath):
            return "Vision subsystem failed: Could not capture physical display."

        # ── 1. Cloud Vision (when configured) ─────────────────
        if self.gemini is not None:
            try:
                result = await self.gemini.ask(query, image_path=filepath)
                self._cleanup(filepath)
                return result
            except Exception as exc:
                logger.error("Cloud Vision inference failed: %s", exc)
                # Don't return yet -- try the local VLM next.

        # ── 2. Local VLM captioner (always available) ─────────
        if self._captioner_available():
            try:
                caption = await self._describe_local(filepath, query)
                self._cleanup(filepath)
                if caption:
                    return caption
            except Exception as exc:
                logger.warning("Local VLM caption failed: %s", exc)

        # ── 3. Hard fallback -- speakable, no stack-trace ─────
        self._cleanup(filepath)
        return (
            "I can see the screen, Boss, but the description model "
            "isn't loaded right now."
        )

    # ── internals ────────────────────────────────────────────────

    async def _describe_local(self, image_path: str, query: str) -> str:
        """Ask the local VLM captioner for a description. Tolerates
        sync OR async ``describe`` / ``describe_image`` methods so we
        match every captioner shape currently in tree (SmolVLM via
        MLX-VLM, Apple-Vision wrapper, test doubles)."""
        cap = self.vlm_captioner
        # Prefer ``describe`` (newer interface), fall back to
        # ``describe_image`` (legacy shape).
        method = getattr(cap, "describe", None) or getattr(
            cap, "describe_image", None,
        )
        if method is None:
            return ""
        try:
            res = method(image_path, prompt=query)
        except TypeError:
            res = method(image_path)
        # async result?
        import inspect
        if inspect.isawaitable(res):
            res = await res
        if isinstance(res, dict):
            res = res.get("text") or res.get("caption") or ""
        return (res or "").strip()

    @staticmethod
    def _cleanup(filepath: str) -> None:
        try:
            os.remove(filepath)
        except OSError:
            logger.debug("screen capture cleanup failed", exc_info=True)
