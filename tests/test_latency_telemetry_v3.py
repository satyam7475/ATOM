"""v3 Phase 6 — LatencyTimeline + jarvis_eval regression tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.latency_timeline import STAGES, LatencyTimeline, TurnTimeline


# ── LatencyTimeline ───────────────────────────────────────────────────


def test_turn_timeline_marks_stages_and_totals(tmp_path: Path) -> None:
    tl = LatencyTimeline(log_path=tmp_path / "atom_latency.jsonl")
    turn = tl.begin_turn(turn_id="t1")
    assert isinstance(turn, TurnTimeline)
    turn.mark("mic_open")
    turn.mark("stt_final")
    turn.mark("router_route")
    turn.mark("llm_first_token")
    turn.mark("tts_first_audio")

    rec = turn.to_dict()
    assert rec["turn_id"] == "t1"
    assert rec["total_ms"] >= 0.0
    assert "mic_open" in rec["stages_ms"]
    assert "tts_first_audio" in rec["stages_ms"]


def test_turn_timeline_stages_are_monotonically_non_decreasing() -> None:
    """Stage timestamps are relative ms-from-start, so they must form a
    non-decreasing sequence in the order they were marked."""
    tl = LatencyTimeline()
    turn = tl.begin_turn()
    turn.mark("mic_open")
    turn.mark("stt_final")
    turn.mark("router_route")
    turn.mark("llm_first_token")
    turn.mark("tts_first_audio")
    seq = [turn.stages[s] for s in (
        "mic_open", "stt_final", "router_route",
        "llm_first_token", "tts_first_audio",
    )]
    assert seq == sorted(seq)


def test_commit_writes_one_jsonl_line(tmp_path: Path) -> None:
    log_path = tmp_path / "atom_latency.jsonl"
    tl = LatencyTimeline(log_path=log_path)
    for i in range(3):
        turn = tl.begin_turn(turn_id=f"t{i}")
        turn.mark("mic_open")
        turn.mark("tts_first_audio")
        turn.annotate(path="DIRECT")
        tl.commit(turn)
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    for i, line in enumerate(lines):
        rec = json.loads(line)
        assert rec["turn_id"] == f"t{i}"
        assert "stages_ms" in rec
        assert rec["meta"]["path"] == "DIRECT"


def test_commit_feeds_metrics_collector(tmp_path: Path) -> None:
    """When a MetricsCollector is wired, commit() also records each
    stage as ``pipeline_<stage>`` so the existing health log surfaces
    them automatically."""

    class _FakeMetrics:
        def __init__(self) -> None:
            self.samples: list[tuple[str, float]] = []

        def record_latency(self, name: str, ms: float) -> None:
            self.samples.append((name, ms))

    metrics = _FakeMetrics()
    tl = LatencyTimeline(log_path=tmp_path / "atom_latency.jsonl", metrics=metrics)
    turn = tl.begin_turn()
    turn.mark("mic_open")
    turn.mark("router_route")
    turn.mark("tts_first_audio")
    tl.commit(turn)

    names = [n for n, _ in metrics.samples]
    assert "perceived" in names
    assert "pipeline_mic_open" in names
    assert "pipeline_router_route" in names
    assert "pipeline_tts_first_audio" in names


def test_commit_is_resilient_to_logger_io_errors(tmp_path: Path, monkeypatch) -> None:
    """A broken filesystem must not propagate up and stall the turn."""
    tl = LatencyTimeline(log_path=tmp_path / "atom_latency.jsonl")

    class _BrokenLogger:
        def info(self, *_a, **_kw):
            raise RuntimeError("disk full")

    tl._writer = _BrokenLogger()  # type: ignore[assignment]
    turn = tl.begin_turn()
    turn.mark("mic_open")
    tl.commit(turn)  # must not raise


def test_unknown_stage_does_not_raise() -> None:
    tl = LatencyTimeline()
    turn = tl.begin_turn()
    turn.mark("not_a_real_stage")  # logged at debug, no exception
    assert "not_a_real_stage" in turn.stages


def test_stages_is_immutable_tuple_with_expected_order() -> None:
    assert isinstance(STAGES, tuple)
    expected = (
        "mic_open", "vad_endpoint", "stt_final", "stt_confirm",
        "router_route", "llm_first_token", "llm_complete",
        "tts_first_audio", "tts_complete",
    )
    assert STAGES == expected


# ── jarvis_eval harness ──────────────────────────────────────────────


def test_jarvis_eval_runs_end_to_end(tmp_path: Path) -> None:
    """Run the eval as a subprocess and assert it produces the report
    files. Using a fresh out-dir keeps it isolated from real logs/."""
    repo_root = Path(__file__).resolve().parent.parent
    cmd = [
        sys.executable, str(repo_root / "tests" / "jarvis_eval.py"),
        "--out-dir", str(tmp_path),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=60,
    )
    # We do NOT require --strict so the harness doesn't gate this test
    # on tuning thresholds.
    assert proc.returncode == 0, (
        f"jarvis_eval exited {proc.returncode}\nstderr:\n{proc.stderr}"
    )
    md = tmp_path / "JARVIS_EVAL_REPORT.md"
    assert md.exists(), "report markdown missing"
    md_text = md.read_text(encoding="utf-8")
    assert "Jarvis Eval Report" in md_text
    assert "Overall:" in md_text

    json_files = list(tmp_path.glob("jarvis_eval_*.json"))
    assert json_files, "no jarvis_eval_*.json file written"
    payload = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert "axes" in payload
    assert "overall_ok" in payload
    axis_names = {a["name"] for a in payload["axes"]}
    expected_axes = {
        "prompt_leak_guard",
        "reasoning_leak_guard",
        "smart_route",
        "tool_grammar",
        "latency_telemetry",
        "whisper_confirmer",
    }
    missing = expected_axes - axis_names
    assert not missing, f"expected axes missing: {missing}"


def test_jarvis_eval_strict_mode_returns_nonzero_when_axis_fails(tmp_path: Path, monkeypatch) -> None:
    """If we monkey-patch the prompt-leak axis to obviously fail, --strict
    must return 1. Sanity check on the gating path."""
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

    from tests import jarvis_eval as je

    bad_axis = je.AxisResult(
        name="prompt_leak_guard",
        passed=0,
        total=4,
        score_pct=0.0,
        threshold_pct=100.0,
        failures=["forced miss"],
    )

    def _fake_prompt_axis():
        return bad_axis

    monkeypatch.setattr(je, "_axis_prompt_leak", _fake_prompt_axis)
    rc = je.main(["--out-dir", str(tmp_path), "--strict"])
    assert rc == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
