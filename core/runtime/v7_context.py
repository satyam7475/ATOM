"""
V7RuntimeContext — shared snapshot for mode, system, feedback, and timeline.

Passed across LocalBrainController, RuntimeModeResolver, RagEngine, PrefetchEngine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class V7RuntimeContext:
    """Per-query (or periodic) carrier; avoid heavy work in __init__."""

    system_state: dict[str, Any] = field(default_factory=dict)
    feedback_metrics: dict[str, Any] = field(default_factory=dict)
    runtime_mode: str = "SMART"
    mode_info: dict[str, Any] = field(default_factory=dict)
    timeline_summary: str = ""
    gpu_util_pct: float = 0.0
    prediction_accuracy: float | None = None
    last_retrieval_source: str = ""

    def with_mode(self, mode: str, info: dict[str, Any] | None = None) -> V7RuntimeContext:
        return V7RuntimeContext(
            system_state=dict(self.system_state),
            feedback_metrics=dict(self.feedback_metrics),
            runtime_mode=mode,
            mode_info=dict(info or {}),
            timeline_summary=self.timeline_summary,
            gpu_util_pct=self.gpu_util_pct,
            prediction_accuracy=self.prediction_accuracy,
            last_retrieval_source=self.last_retrieval_source,
        )
