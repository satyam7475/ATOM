#!/usr/bin/env python3
"""tests.jarvis_eval — v3 Phase 6.2 nightly Jarvis-grade scorecard.

What this is
------------
A *short*, *deterministic*, *no-network* eval that produces a one-page
scorecard against a fixed set of axes that matter for the Jarvis-grade
v3 release:

  * **prompt-leak rate**       (regression on Phase 1.2/1.3 sanitisers)
  * **CoT/reasoning-leak rate**(regression on the local-brain sanitiser)
  * **direct-intent hit rate** (regression on the intent engine cold path)
  * **smart-route accuracy**   (Phase 3 cloud routing decisions)
  * **tool-call validation**   (Phase 5 grammar+validator)
  * **latency telemetry parses** (Phase 6.1 timeline JSONL is well-formed)

Designed to run in <5 seconds in CI / nightly cron. Writes:

  logs/jarvis_eval_<timestamp>.json
  logs/JARVIS_EVAL_REPORT.md

Usage:
  python -m tests.jarvis_eval          # writes report files
  python -m tests.jarvis_eval --print  # also prints the markdown to stdout
  python -m tests.jarvis_eval --strict # exit 1 on any axis below threshold
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ── Per-axis scorers ──────────────────────────────────────────────────


@dataclass
class AxisResult:
    name: str
    passed: int
    total: int
    score_pct: float
    threshold_pct: float
    failures: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.score_pct >= self.threshold_pct


def _run_axis(
    name: str,
    cases: list[tuple[str, Callable[[], bool]]],
    threshold_pct: float,
) -> AxisResult:
    t0 = time.perf_counter()
    passed = 0
    failures: list[str] = []
    for label, fn in cases:
        try:
            ok = bool(fn())
        except Exception as exc:  # axis-fail rather than test crash
            ok = False
            failures.append(f"{label}: raised {type(exc).__name__}: {exc}")
        if ok:
            passed += 1
        elif label not in (f for f, _ in cases if False):
            failures.append(label)
    total = len(cases)
    score = (passed / total * 100.0) if total else 100.0
    return AxisResult(
        name=name,
        passed=passed,
        total=total,
        score_pct=round(score, 1),
        threshold_pct=threshold_pct,
        failures=failures,
        elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 1),
    )


# ── Axis 1: prompt-leak detection ────────────────────────────────────


def _axis_prompt_leak() -> AxisResult:
    from cursor_bridge.local_brain_controller import _looks_like_prompt_leak
    from voice.tts_macos import _is_prompt_leak

    # Only phrases known from production atomlogs.txt to have been
    # parroted by the LLM verbatim. We do NOT add speculative ones --
    # false positives here would silence legitimate user replies.
    leaky_phrases = [
        "The final answer only. One short line.",
        "If the question is a simple, short, or info query, give one short sentence",
        "two short sentences max.",
        "Output only the final answer.",
    ]
    safe_phrases = [
        "Yes, Boss.",
        "It's 3:45 PM.",
        "I've opened Chrome.",
        "Sure, what would you like to play?",
    ]
    cases: list[tuple[str, Callable[[], bool]]] = []
    for p in leaky_phrases:
        cases.append((f"controller flags: {p[:40]}", lambda p=p: _looks_like_prompt_leak(p)))
        cases.append((f"tts flags: {p[:40]}", lambda p=p: _is_prompt_leak(p)))
    for p in safe_phrases:
        cases.append((f"controller passes: {p[:40]}", lambda p=p: not _looks_like_prompt_leak(p)))
        cases.append((f"tts passes: {p[:40]}", lambda p=p: not _is_prompt_leak(p)))
    return _run_axis("prompt_leak_guard", cases, threshold_pct=100.0)


# ── Axis 2: CoT / reasoning-leak detection ──────────────────────────


def _axis_reasoning_leak() -> AxisResult:
    from cursor_bridge.local_brain_controller import LocalBrainController

    raw_safe_outputs = [
        "Hello Boss, opening Chrome now.",
        "It's 32 degrees in Mumbai right now.",
        "Sure, what should I play next?",
    ]
    # Whole-string CoT/reasoning leaks observed in production atomlogs.
    # The sanitiser must drop ALL of these to empty.
    whole_string_leaks = [
        "Since I can't browse the internet directly, I should rely on my existing knowledge.",
        "The answer needs to be concise and in plain text.",
        "Based on the response contract, I should confirm my activity.",
    ]
    # Mixed-line leaks: a CoT preface + an actual answer. The sanitiser
    # should KEEP the answer line and DROP the preface.
    mixed_leaks = [
        ("Let me think step by step. First I will check the weather.",
         "First I will check the weather."),
    ]
    inst = LocalBrainController.__new__(LocalBrainController)
    cases: list[tuple[str, Callable[[], bool]]] = []
    for s in raw_safe_outputs:
        cases.append((
            f"keeps: {s[:40]}",
            lambda s=s: bool(inst._sanitize_emittable_text(s).strip()),
        ))
    for s in whole_string_leaks:
        cases.append((
            f"drops whole: {s[:40]}",
            lambda s=s: not inst._sanitize_emittable_text(s).strip(),
        ))
    for raw, expected_substring in mixed_leaks:
        cases.append((
            f"strips preface: {raw[:40]}",
            lambda raw=raw, want=expected_substring: (
                want.lower() in inst._sanitize_emittable_text(raw).lower()
                and "step by step" not in inst._sanitize_emittable_text(raw).lower()
            ),
        ))
    return _run_axis("reasoning_leak_guard", cases, threshold_pct=85.0)


# ── Axis 3: smart-route routing accuracy ─────────────────────────────


def _axis_smart_route() -> AxisResult:
    from core.cognitive_kernel import CognitiveKernel, ExecPath

    class _GeminiStub:
        is_available = True

    cfg = {
        "cloud": {
            "enabled": True,
            "daily_budget_calls": 100,
            "smart_route_keywords": ["explain properly", "research this", "deep analysis"],
            "smart_route_min_query_words": 25,
        },
    }
    k = CognitiveKernel(config=cfg)
    k.attach_cloud_intelligence(gemini_client=_GeminiStub())

    expects_cloud = [
        "Boss, please research this for me",
        "explain properly why the sky is blue",
        "do a deep analysis of this trade-off and walk me through the math",
        " ".join(["topic"] * 30),  # long form
    ]
    expects_local = [
        "what time is it",
        "open chrome",
        "play music",
        "are you there",
    ]
    cases: list[tuple[str, Callable[[], bool]]] = []
    for q in expects_cloud:
        cases.append((
            f"cloud routes: {q[:40]}",
            lambda q=q: k.route(q).path == ExecPath.CLOUD_REASON,
        ))
    for q in expects_local:
        cases.append((
            f"stays local: {q[:40]}",
            lambda q=q: k.route(q).path != ExecPath.CLOUD_REASON,
        ))
    return _run_axis("smart_route", cases, threshold_pct=85.0)


# ── Axis 4: tool-call grammar + validator ────────────────────────────


def _axis_tool_grammar() -> AxisResult:
    from core.reasoning.tool_grammar import (
        build_tool_call_prompt_grammar, validate_tool_call,
    )
    from core.reasoning.tool_parser import ToolCall, parse_tool_calls
    from core.reasoning.tool_registry import Tool, ToolParameter, ToolRegistry

    reg = ToolRegistry()
    reg.register(Tool(
        name="open_app",
        description="Open an app",
        parameters=[ToolParameter(name="app_name", type="string", required=True)],
    ))

    cases: list[tuple[str, Callable[[], bool]]] = [
        ("parses canonical", lambda: parse_tool_calls(
            '<tool_call>{"name":"open_app","arguments":{"app_name":"Chrome"}}</tool_call>'
        ).has_tool_calls),
        ("recovers naked json", lambda: parse_tool_calls(
            '{"name":"open_app","arguments":{"app_name":"Safari"}}'
        ).has_tool_calls),
        ("ignores prose json", lambda: not parse_tool_calls(
            'the city is {"city":"Mumbai"}'
        ).has_tool_calls),
        ("validator accepts good", lambda: validate_tool_call(
            ToolCall(name="open_app", arguments={"app_name": "Chrome"}),
            reg,
        ).ok),
        ("validator rejects unknown", lambda: not validate_tool_call(
            ToolCall(name="nope", arguments={}), reg,
        ).ok),
        ("validator rejects missing arg", lambda: not validate_tool_call(
            ToolCall(name="open_app", arguments={}), reg,
        ).ok),
        ("grammar lists tool", lambda: "open_app" in build_tool_call_prompt_grammar(reg)),
    ]
    return _run_axis("tool_grammar", cases, threshold_pct=100.0)


# ── Axis 5: latency telemetry round-trip ─────────────────────────────


def _axis_latency_telemetry(tmp_dir: Path) -> AxisResult:
    from core.latency_timeline import STAGES, LatencyTimeline

    log_path = tmp_dir / "atom_latency_eval.jsonl"
    if log_path.exists():
        log_path.unlink()
    tl = LatencyTimeline(log_path=log_path)

    cases: list[tuple[str, Callable[[], bool]]] = []

    def _full_turn() -> bool:
        turn = tl.begin_turn(turn_id="evaltest")
        turn.mark("mic_open")
        turn.mark("stt_final")
        turn.mark("router_route")
        turn.mark("llm_first_token")
        turn.mark("tts_first_audio")
        turn.annotate(path="DIRECT", text_len=12)
        tl.commit(turn)
        # Read back and confirm well-formed JSON line.
        if not log_path.exists():
            return False
        line = log_path.read_text(encoding="utf-8").strip().splitlines()[-1]
        rec = json.loads(line)
        return (
            rec["turn_id"] == "evaltest"
            and "stages_ms" in rec
            and "tts_first_audio" in rec["stages_ms"]
            and rec["meta"]["path"] == "DIRECT"
        )

    cases.append(("full turn round-trips", _full_turn))
    cases.append(("STAGES has expected stages", lambda: all(
        s in STAGES for s in (
            "mic_open", "stt_final", "router_route",
            "llm_first_token", "tts_first_audio",
        )
    )))
    return _run_axis("latency_telemetry", cases, threshold_pct=100.0)


# ── Axis 6: WhisperConfirmer disabled-by-default + suspect detection ─


def _axis_whisper_confirmer() -> AxisResult:
    from voice.whisper_confirmer import WhisperConfirmer

    wc_default = WhisperConfirmer({})
    wc_on = WhisperConfirmer({"whisper_confirm": {"enabled": True}})

    cases: list[tuple[str, Callable[[], bool]]] = [
        ("disabled-by-default", lambda: wc_default.is_enabled() is False),
        ("disabled noop confirm", lambda: wc_default.confirm("uh", 0.1).used_whisper is False),
        ("flags blank as suspect", lambda: wc_on._is_suspect("", 0.99)[0]),
        ("flags noise token as suspect", lambda: wc_on._is_suspect("uh", 0.99)[0]),
        ("passes healthy text", lambda: not wc_on._is_suspect("what time is it", 0.92)[0]),
    ]
    return _run_axis("whisper_confirmer", cases, threshold_pct=100.0)


# ── Axis 7: System Control v1 — voice-to-handler reachability ───────


def _axis_system_control_reachability() -> AxisResult:
    """Verify every System Control v1 voice command reaches a real handler.

    Two-layer check per case:
      1. :class:`IntentEngine` must classify the phrase to the expected
         ``action`` string (not ``fallback``).
      2. That ``action`` must be resolvable via ``Router._ACTION_DISPATCH``
         or ``Router._LATE_DISPATCH`` (bus-handled intents are checked
         separately in the router/intents unit tests).
    """
    from core.intent_engine import IntentEngine
    from core.router.router import Router
    from core.security_tiers import action_tier

    engine = IntentEngine()
    dispatch = dict(Router._ACTION_DISPATCH)
    late = dict(Router._LATE_DISPATCH)

    cases_map: list[tuple[str, str, int]] = [
        # (utterance, expected action, expected security tier)
        ("is chrome running", "find_process_by_name", 1),
        ("find process called spotify", "find_process_by_name", 1),
        ("show me details for pid 4521", "get_process_details", 1),
        ("show me the open ports", "get_open_ports", 1),
        ("scan wifi networks", "get_wifi_networks", 1),
        ("find the biggest files", "find_large_files", 1),
        ("analyze temp files", "analyze_temp_files", 1),
        ("describe the focused element", "describe_focused_element", 1),
        ("read the focused field", "read_focused_text", 1),
        ("type hello into the focused field", "set_focused_text", 3),
        ("click the submit button", "click_ui_element", 3),
        ("make pid 1234 high priority", "set_process_priority", 4),
        ("optimize for atom", "optimize_for_atom", 4),
    ]

    def _check(text: str, expected_action: str, expected_tier: int) -> bool:
        result = engine.classify(text)
        if result.action != expected_action:
            return False
        if expected_action not in dispatch and expected_action not in late:
            return False
        return action_tier(expected_action) == expected_tier

    cases: list[tuple[str, Callable[[], bool]]] = [
        (
            f"{utt!r} -> {act} (tier {tier})",
            lambda u=utt, a=act, t=tier: _check(u, a, t),
        )
        for utt, act, tier in cases_map
    ]
    return _run_axis("system_control_reachability", cases, threshold_pct=100.0)


# ── Report rendering ─────────────────────────────────────────────────


def _render_markdown(results: list[AxisResult]) -> str:
    overall_ok = all(r.ok for r in results)
    avg_score = sum(r.score_pct for r in results) / max(1, len(results))
    badge = "PASS" if overall_ok else "FAIL"

    lines = [
        "# ATOM v3 — Jarvis Eval Report",
        "",
        f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        f"**Overall: {badge}**  -- average score {avg_score:.1f}%  ",
        f"({sum(1 for r in results if r.ok)}/{len(results)} axes above threshold)",
        "",
        "| Axis | Score | Threshold | Pass | Time | Status |",
        "|------|------:|----------:|------|-----:|--------|",
    ]
    for r in results:
        status = "OK" if r.ok else "MISS"
        lines.append(
            f"| {r.name} | {r.score_pct:.1f}% | {r.threshold_pct:.0f}% | "
            f"{r.passed}/{r.total} | {r.elapsed_ms:.0f}ms | {status} |"
        )

    failing = [r for r in results if not r.ok]
    if failing:
        lines.append("")
        lines.append("## Failures")
        for r in failing:
            lines.append(f"### {r.name}  (score {r.score_pct:.1f}% < {r.threshold_pct:.0f}%)")
            for f in r.failures[:20]:
                lines.append(f"  - {f}")
            if len(r.failures) > 20:
                lines.append(f"  - ...and {len(r.failures) - 20} more")
    return "\n".join(lines) + "\n"


# ── main ─────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print", action="store_true", help="echo report to stdout")
    parser.add_argument("--strict", action="store_true", help="exit 1 if any axis below threshold")
    parser.add_argument("--out-dir", default="logs", help="report output dir")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = [
        _axis_prompt_leak(),
        _axis_reasoning_leak(),
        _axis_smart_route(),
        _axis_tool_grammar(),
        _axis_latency_telemetry(out_dir),
        _axis_whisper_confirmer(),
        _axis_system_control_reachability(),
    ]

    md = _render_markdown(results)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"jarvis_eval_{ts}.json"
    md_path = out_dir / "JARVIS_EVAL_REPORT.md"

    json_path.write_text(json.dumps(
        {
            "generated_at": datetime.now().isoformat(),
            "axes": [asdict(r) for r in results],
            "overall_ok": all(r.ok for r in results),
            "avg_score_pct": round(
                sum(r.score_pct for r in results) / max(1, len(results)), 2,
            ),
        },
        indent=2,
    ))
    md_path.write_text(md)

    if args.print:
        sys.stdout.write(md)

    print(f"Wrote {md_path}", file=sys.stderr)
    print(f"Wrote {json_path}", file=sys.stderr)

    if args.strict and not all(r.ok for r in results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
