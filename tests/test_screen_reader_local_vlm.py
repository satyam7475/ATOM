"""Regression tests for Sprint A4 -- screen_reader VLM fallback.

Pins:

* ``ScreenReader`` no longer returns the legacy
  ``"Vision subsystem fallback: ... Gemini Client offline."`` string.
* When ``vlm_captioner`` is supplied, the captioner is asked.
* When no backend is available, a short speakable sentence is returned.
* ``LocalBrainController.attach_vlm_captioner`` plumbs the captioner
  through to the PERCEPTION branch.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from core.perception.screen_reader import ScreenReader


# ── helpers ────────────────────────────────────────────────────────


class _FakeCaptioner:
    """Minimal stand-in for ``VLMCaptioner``."""

    def __init__(self, caption: str = "A code editor with a Python file.") -> None:
        self.caption = caption
        self.calls: list[tuple[str, str]] = []
        self.is_available_calls = 0

    def is_available(self) -> bool:
        self.is_available_calls += 1
        return True

    def describe(self, image_path: str, prompt: str = "") -> str:
        self.calls.append((image_path, prompt))
        return self.caption


class _UnavailableCaptioner:
    def is_available(self) -> bool:
        return False

    def describe(self, image_path: str, prompt: str = "") -> str:
        raise AssertionError("captioner.describe must not be called when unavailable")


# ── fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def fake_screenshot(tmp_path):
    """Patch ``ScreenReader.capture_screen`` to skip the real
    ``screencapture`` and return a tiny dummy file path."""
    fake = tmp_path / "screen.png"
    fake.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    with patch.object(ScreenReader, "capture_screen", lambda self: str(fake)):
        yield str(fake)


# ── core fallback behaviour ────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_backend_returns_speakable_string_not_legacy(fake_screenshot):
    sr = ScreenReader(gemini_client=None, vlm_captioner=None)
    out = await sr.analyze_screen("what is on my screen")
    assert "Gemini Client offline" not in out
    assert "Gemini" not in out
    assert "Boss" in out or "screen" in out.lower()
    # Cleanup happened
    assert not os.path.exists(fake_screenshot)


@pytest.mark.asyncio
async def test_vlm_captioner_used_when_gemini_missing(fake_screenshot):
    cap = _FakeCaptioner(caption="VS Code is open with main.py focused.")
    sr = ScreenReader(gemini_client=None, vlm_captioner=cap)
    out = await sr.analyze_screen("describe my screen")
    assert out == "VS Code is open with main.py focused."
    assert len(cap.calls) == 1
    assert cap.calls[0][0] == fake_screenshot
    assert cap.is_available_calls >= 1


@pytest.mark.asyncio
async def test_vlm_captioner_skipped_when_unavailable(fake_screenshot):
    cap = _UnavailableCaptioner()
    sr = ScreenReader(gemini_client=None, vlm_captioner=cap)
    out = await sr.analyze_screen("describe my screen")
    # Should not crash, should not echo legacy string
    assert "Gemini" not in out


@pytest.mark.asyncio
async def test_capture_failure_returns_speakable_error(monkeypatch):
    sr = ScreenReader()
    monkeypatch.setattr(sr, "capture_screen", lambda: "")
    out = await sr.analyze_screen("describe")
    assert "Could not capture" in out
    assert "Gemini" not in out


@pytest.mark.asyncio
async def test_gemini_failure_falls_through_to_local_vlm(fake_screenshot):
    """If the cloud client raises, we must try the on-device captioner."""

    class _BrokenGemini:
        async def ask(self, query, image_path=None):
            raise RuntimeError("network down")

    cap = _FakeCaptioner(caption="A terminal showing build output.")
    sr = ScreenReader(gemini_client=_BrokenGemini(), vlm_captioner=cap)
    out = await sr.analyze_screen("what's on screen")
    assert out == "A terminal showing build output."
    assert len(cap.calls) == 1


# ── controller plumbing ───────────────────────────────────────────


def test_local_brain_controller_exposes_attach_vlm_captioner():
    from cursor_bridge.local_brain_controller import LocalBrainController

    assert hasattr(LocalBrainController, "attach_vlm_captioner")


def test_attach_vlm_captioner_stores_reference():
    from cursor_bridge.local_brain_controller import LocalBrainController

    obj = LocalBrainController.__new__(LocalBrainController)
    obj._vlm_captioner = None
    cap = _FakeCaptioner()
    LocalBrainController.attach_vlm_captioner(obj, cap)
    assert obj._vlm_captioner is cap


def test_screen_reader_constructor_accepts_vlm_captioner():
    cap = _FakeCaptioner()
    sr = ScreenReader(vlm_captioner=cap)
    assert sr.vlm_captioner is cap
