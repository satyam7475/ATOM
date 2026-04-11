"""
Tests for FSWatcher configuration helpers (step 5.6).

Run: python3 -m tests.test_fs_watcher_config
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_fs_watcher_settings_defaults() -> None:
    from core.macos.fs_watcher_config import DEFAULT_PATHS, fs_watcher_settings

    s = fs_watcher_settings({})
    assert s["enabled"] is True
    assert s["paths"] == list(DEFAULT_PATHS)
    assert s["hints_enabled"] is True
    assert ".pdf" in s["hint_extensions"]
    assert s["hint_cooldown_s"] == 120.0
    assert s["emit_voice"] is False
    print("  PASS: fs_watcher_settings defaults")


def test_notable_hint_downloads_pdf() -> None:
    from core.macos.fs_watcher_config import notable_file_hint

    p = "/Users/boss/Downloads/report.pdf"
    h = notable_file_hint(path=p, event="created", is_dir=False, config={})
    assert h and "PDF" in h and "report.pdf" in h
    assert notable_file_hint(path=p, event="removed", is_dir=False, config={}) is None
    assert notable_file_hint(path="/tmp/x.pdf", event="created", is_dir=False, config={}) is None
    print("  PASS: notable_file_hint downloads PDF")


def test_notable_hint_disabled() -> None:
    from core.macos.fs_watcher_config import notable_file_hint

    cfg = {"macos": {"fs_watcher": {"hints_enabled": False}}}
    assert (
        notable_file_hint(
            path="/Users/x/Downloads/a.pdf",
            event="created",
            is_dir=False,
            config=cfg,
        )
        is None
    )
    print("  PASS: notable_file_hint respects hints_enabled")


def test_classify_and_ignore_importable() -> None:
    from core.macos.fs_watcher import _classify_event, _should_ignore

    assert _classify_event(0x00000100) == "created"
    assert _should_ignore("/a/.DS_Store") is True
    assert _should_ignore("/a/readme.txt") is False
    print("  PASS: fs_watcher classify / ignore")


if __name__ == "__main__":
    test_fs_watcher_settings_defaults()
    test_notable_hint_downloads_pdf()
    test_notable_hint_disabled()
    test_classify_and_ignore_importable()
    print("All fs_watcher config tests passed.")
