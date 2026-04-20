from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cursor_bridge.local_brain_controller import LocalBrainController  # noqa: E402


class _FakeBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, event: str, **data) -> None:
        self.events.append((event, data))

    def emit_long(self, event: str, **data) -> None:
        self.events.append((event, data))


class _FakePromptBuilder:
    pass


class _FakeSecondBrain:
    def __init__(self) -> None:
        self.saved: list[tuple[str, str, str]] = []

    def remember_report(self, topic: str, summary: str, path: str) -> None:
        self.saved.append((topic, summary, path))


def test_local_brain_exports_long_report(tmp_path: Path) -> None:
    controller = LocalBrainController(
        _FakeBus(),
        _FakePromptBuilder(),
        {
            "brain": {
                "mlx_primary_model": "models/qwen3-8b-mlx-4bit",
                "mlx_fast_model": "models/qwen3-8b-mlx-4bit",
            },
        },
    )
    controller._report_dir = tmp_path
    memory = _FakeSecondBrain()
    controller.attach_second_brain(memory)

    report = (
        "Safari is best for battery life and text rendering. "
        "Chrome and Arc win on extension compatibility and dev tools. "
        "If you want the best balance on a MacBook Air, Safari or Arc are usually "
        "the strongest picks depending on workflow. "
        "For long coding sessions, memory pressure and battery life matter as much "
        "as extension depth. "
    ) * 4

    spoken_text, saved_path = controller._maybe_export_report(
        "research best browser for coding on mac",
        report,
    )

    assert saved_path is not None
    assert Path(saved_path).exists()
    assert "saved the full report" in spoken_text.lower()
    assert memory.saved
    controller.close()
