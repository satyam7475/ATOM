"""Regression tests for the boot face-check log demotion (C12).

Live log evidence (atom_log.txt L233):

    2026-04-25 00:43:28 | atom.main | INFO | Boot face check: video
    delegate exception: 'CMSampleBufferGetImageBuffer'

The boot face check is an *optional* feature (Continuity Camera or
local webcam, gated behind ``vision.boot_face_check``). When it fails
— camera dozed off, video delegate raised, AVCapture never produced
a frame — the user said it himself: "don't make startup noise about
an optional feature."

These tests prove that:

1. The failure-path log lands at DEBUG, not INFO (so a missing camera
   stays out of the boot log unless the user is debugging).
2. The "camera up but no face" path also lands at DEBUG (common case
   when the user isn't looking at the lens).
3. The positive "face detected" path is *still* INFO (that's the
   only branch that contributes a feature signal worth surfacing).

We do NOT actually boot ATOM (AVCaptureSession requires a real Mac
runloop). Instead we read main.py and assert the literal logger
levels around the boot-face-check branches.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


_MAIN_PY = Path(__file__).resolve().parents[1] / "main.py"


@pytest.fixture(scope="module")
def main_source() -> str:
    return _MAIN_PY.read_text(encoding="utf-8")


def _slice_boot_face_block(src: str) -> str:
    """Return the substring of ``main.py`` that contains the boot face
    check result handler, so the assertions below operate on the right
    region instead of pattern-matching the whole 3000-line file."""
    start = src.find('if face_result is not None:')
    assert start != -1, "boot face check block not found in main.py"
    end = src.find("logger.info(\"Startup greeting:", start)
    assert end != -1, "end-of-block sentinel not found"
    return src[start:end]


def test_failure_path_logs_at_debug(main_source: str) -> None:
    """The ``else`` branch (face_result.ok == False) must demote to DEBUG.

    This is the branch the live log hit at L233 — anything except
    DEBUG here would re-introduce the boot-log noise the user
    explicitly asked us to silence.
    """
    block = _slice_boot_face_block(main_source)
    assert "Boot face check unavailable" in block, (
        "expected DEBUG-level 'Boot face check unavailable' message"
    )
    failure_pattern = re.compile(
        r"logger\.debug\(\s*\n?\s*['\"]Boot face check unavailable",
        re.MULTILINE,
    )
    assert failure_pattern.search(block), (
        "Boot face check failure path must use logger.debug — found:\n"
        + block,
    )


def test_no_face_path_logs_at_debug(main_source: str) -> None:
    """The ``camera ready but no face`` branch must also be DEBUG so a
    user not currently looking at the lens isn't logged at INFO every
    boot."""
    block = _slice_boot_face_block(main_source)
    no_face_pattern = re.compile(
        r"logger\.debug\(\s*\n?\s*['\"]Boot face check: camera ready",
        re.MULTILINE,
    )
    assert no_face_pattern.search(block), (
        "'camera ready but no face yet' must log at DEBUG"
    )


def test_face_detected_path_still_info(main_source: str) -> None:
    """The positive branch (face actually seen) is the *only* one
    worth surfacing in a normal boot log — keep it at INFO."""
    block = _slice_boot_face_block(main_source)
    detected_pattern = re.compile(
        r"logger\.info\(\s*\n?\s*['\"]Boot face check: detected",
        re.MULTILINE,
    )
    assert detected_pattern.search(block), (
        "Positive face-detected branch must remain at logger.info"
    )


def test_failure_path_does_not_use_info(main_source: str) -> None:
    """Defensive: regardless of message wording, ``Boot face check``
    must never appear with ``logger.info(... %s, ... face_result.error``
    pattern again."""
    block = _slice_boot_face_block(main_source)
    leaked = re.compile(
        r"logger\.info\([^)]*face_result\.error",
        re.DOTALL,
    )
    assert leaked.search(block) is None, (
        "Found logger.info call passing face_result.error — that path "
        "must use logger.debug to keep boot logs quiet about an "
        "optional feature."
    )


def test_result_handling_wrapped_in_try_except(main_source: str) -> None:
    """A defensive try/except guards the result-handling block so a
    malformed face_result (e.g. missing .ok after a refactor) cannot
    crash the greeting."""
    block = _slice_boot_face_block(main_source)
    assert "try:" in block
    assert "except Exception" in block
    assert "Boot face check result handling raised" in block
