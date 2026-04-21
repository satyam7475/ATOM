"""core.latency_timeline — v3 Phase 6: end-to-end per-turn latency telemetry.

What this gives us
------------------
Existing ``MetricsCollector`` tracks averages and p95s for each named
pipeline stage independently. That's good for trend dashboards but bad
for diagnosing a single bad turn -- you can't see WHICH stage stalled
on the turn where the user complained.

``LatencyTimeline`` solves this by stamping every turn through every
critical stage and writing one JSON line per turn to
``logs/atom_latency.jsonl``. Each line includes the absolute wall-clock
of every stage relative to the start of the turn, so a 4.2-second turn
can be unambiguously decomposed into "STT 320ms, LLM TTFT 1900ms,
TTS first-audio 1980ms" the next morning.

Stages we care about for Jarvis-grade voice
-------------------------------------------
``mic_open``       -- mic actually started capturing this utterance
``vad_endpoint``   -- silence after speech detected
``stt_final``      -- streaming STT emitted the final transcript
``stt_confirm``    -- WhisperConfirmer decided/corrected (if wired)
``router_route``   -- CognitiveKernel chose an ExecPath
``llm_first_token``-- first emittable token from the brain (LOCAL or CLOUD)
``llm_complete``   -- full LLM response done
``tts_first_audio``-- TTS started audible playback
``tts_complete``   -- TTS finished playback

Usage
-----
    timeline = LatencyTimeline()
    turn = timeline.begin_turn(turn_id="abc")
    turn.mark("mic_open")
    ...
    turn.mark("tts_first_audio")
    timeline.commit(turn)

Each ``commit()`` writes a line to ``logs/atom_latency.jsonl`` AND
records the totals into the optional MetricsCollector for live
dashboard surfacing.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("atom.latency_timeline")

# Order matters for both human readability and the nightly eval's
# "first stage to slip" heuristic. New stages should be appended.
STAGES: tuple[str, ...] = (
    "mic_open",
    "vad_endpoint",
    "stt_final",
    "stt_confirm",
    "router_route",
    "llm_first_token",
    "llm_complete",
    "tts_first_audio",
    "tts_complete",
)


@dataclass
class TurnTimeline:
    """Stage timestamps for a single user → ATOM turn."""

    turn_id: str
    started_at_mono: float = field(default_factory=time.monotonic)
    started_at_wall: float = field(default_factory=time.time)
    stages: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def mark(self, stage: str) -> None:
        """Record ``stage`` as having occurred now (relative ms from start)."""
        if stage not in STAGES:
            logger.debug("Unknown latency stage: %s", stage)
        self.stages[stage] = (time.monotonic() - self.started_at_mono) * 1000.0

    def annotate(self, **kw: Any) -> None:
        """Attach metadata (path taken, transcript len, model used, etc.)."""
        self.meta.update(kw)

    def total_ms(self) -> float:
        if not self.stages:
            return 0.0
        return max(self.stages.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "ts": self.started_at_wall,
            "total_ms": round(self.total_ms(), 1),
            "stages_ms": {k: round(v, 1) for k, v in self.stages.items()},
            "meta": self.meta,
        }


class LatencyTimeline:
    """Per-turn latency recorder + JSONL writer.

    Thread-safe. Designed to be cheap on the hot path -- ``begin_turn``
    and ``mark`` do no I/O; only ``commit`` writes a line.
    """

    def __init__(
        self,
        *,
        log_path: str | Path = "logs/atom_latency.jsonl",
        metrics: Any = None,
        max_log_bytes: int = 5_000_000,
        backup_count: int = 3,
    ) -> None:
        self._log_path = Path(log_path)
        self._metrics = metrics
        self._lock = threading.Lock()
        self._writer: Optional[logging.Logger] = None
        self._max_log_bytes = int(max_log_bytes)
        self._backup_count = int(backup_count)

    # ── public API ─────────────────────────────────────────────────

    def begin_turn(self, turn_id: Optional[str] = None) -> TurnTimeline:
        """Start a new turn. ``turn_id`` defaults to a short hex uuid."""
        return TurnTimeline(turn_id=turn_id or uuid.uuid4().hex[:8])

    def commit(self, turn: TurnTimeline) -> None:
        """Persist a turn timeline as one JSONL line and feed metrics."""
        if turn is None:
            return
        record = turn.to_dict()
        try:
            self._get_writer().info(json.dumps(record, separators=(",", ":")))
        except Exception:
            logger.debug("LatencyTimeline.commit write failed", exc_info=True)

        # Feed the live MetricsCollector so the existing health log
        # surface picks up totals + per-stage averages.
        if self._metrics is not None:
            try:
                self._metrics.record_latency("perceived", record["total_ms"])
                for stage, ms in record["stages_ms"].items():
                    self._metrics.record_latency(f"pipeline_{stage}", ms)
            except Exception:
                logger.debug("MetricsCollector feed failed", exc_info=True)

    # ── writer plumbing ────────────────────────────────────────────

    def _get_writer(self) -> logging.Logger:
        if self._writer is not None:
            return self._writer
        with self._lock:
            if self._writer is not None:
                return self._writer
            try:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                logger.debug("Could not create latency log dir", exc_info=True)
            log = logging.getLogger("atom.latency_timeline.jsonl")
            log.setLevel(logging.INFO)
            log.propagate = False
            # Idempotent handler attach (avoid duplicate writes if the
            # module is re-imported in tests).
            if not log.handlers:
                handler = RotatingFileHandler(
                    self._log_path,
                    maxBytes=self._max_log_bytes,
                    backupCount=self._backup_count,
                    encoding="utf-8",
                )
                handler.setFormatter(logging.Formatter("%(message)s"))
                log.addHandler(handler)
            self._writer = log
            return log


__all__ = ["STAGES", "LatencyTimeline", "TurnTimeline"]
