#!/usr/bin/env python3
"""ATOM log triage — read-only structured summary of atomlogs.txt.

Usage:
    python3 .cursor/skills/atom-systems-engineer/scripts/triage_log.py [path/to/atomlogs.txt]

Exits 0 if no P0 issues, 1 if P0 issues detected (useful in CI / pre-commit hooks).
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
LEVEL_RE = re.compile(r"\| (DEBUG|INFO|WARNING|ERROR|CRITICAL|FATAL)\s*\|")
MODEL_LOAD_RE = re.compile(r"CognitiveKernel: quick=(\S+?),\s*full=(\S+?),")
MLX_LABEL_RE = re.compile(r"Agentic MLX LLM \((.+?)\)")
MLX_PEAK_RE = re.compile(r"MLX \[[^\]]+\]: (\d+)ms, (\d+) tokens, ~(\d+) words, ([\d.]+) tok/s, peak ([\d.]+)GB")
TTS_UTTERANCE_RE = re.compile(r"TTS \[NSSpeechSynthesizer[^]]*\]: '(.+?)'")
TTS_STREAM_RE = re.compile(r"TTS stream slice \(\d+/\d+ words\): '(.+?)'")
STT_FINAL_RE = re.compile(r"partial stable for [\d.]+s — promoting to final: '(.+?)'")
ECHO_SUPPRESSED_RE = re.compile(r"self-echo detected on stable partial — promotion suppressed: '(.+?)'")
INTENT_TIMEOUT_RE = re.compile(r"Runtime budget exceeded: intent_engine")
WATCHDOG_WARN_RE = re.compile(r"Runtime budget exceeded: (\S+)")
FIRST_TOKEN_RE = re.compile(r"Brain: (\d+)ms total, (\d+)ms first-token, (\d+) words")
PIPELINE_TOTAL_RE = re.compile(r"PIPELINE \| id=\S+ \| Query: '.+?' \| Intent: \d+ms \| Decision: \S+ \| Action: \d+ms \| TTS: \d+ms \| Total: (\d+)ms")
PROMPT_LEAK_FINGERPRINTS = [
    "the final answer only",
    "one short line",
    "if the question is a simple",
    "final-answer rules",
    "no preface, no rules",
]
COT_PREFACE_FINGERPRINTS = [
    re.compile(r"\b(okay|so|hmm|alright|well|let me see|let'?s see)\b[\s,.]", re.I),
    re.compile(r"\bthe user is (asking|saying|wondering|wants)\b", re.I),
    re.compile(r"\bi (should|need to|will) (think|analyze|consider|parse)\b", re.I),
]
PRESSURE_RE = re.compile(r"Memory pressure tier (\d) -> (\d) \(memory_pct=([\d.]+)%\)")
ERROR_LINE_RE = re.compile(r"\| (ERROR|CRITICAL|FATAL)\s*\|")
TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\)")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class TriageReport:
    log_path: Path
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    total_lines: int = 0
    level_counts: Counter = field(default_factory=Counter)

    model_labels: set[str] = field(default_factory=set)
    mlx_peak_ram_gb: list[float] = field(default_factory=list)
    mlx_first_token_ms: list[int] = field(default_factory=list)

    tts_utterances: list[str] = field(default_factory=list)
    stt_finals: list[str] = field(default_factory=list)
    echo_suppressions: int = 0

    prompt_leaks: list[str] = field(default_factory=list)
    cot_leaks: list[str] = field(default_factory=list)

    watchdog_breaches: Counter = field(default_factory=Counter)
    intent_boot_grace_breaches: int = 0
    intent_late_breaches: int = 0

    pipeline_totals_ms: list[int] = field(default_factory=list)
    memory_pressure_events: list[tuple[int, int, float]] = field(default_factory=list)

    errors: list[str] = field(default_factory=list)
    tracebacks: int = 0

    p0_findings: list[str] = field(default_factory=list)
    p1_findings: list[str] = field(default_factory=list)
    p2_findings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def parse_log(path: Path) -> TriageReport:
    rep = TriageReport(log_path=path)
    boot_ts: datetime | None = None

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            rep.total_lines += 1
            line = line.rstrip("\n")

            ts_match = TIMESTAMP_RE.match(line)
            if ts_match:
                ts_str = ts_match.group(1)
                if rep.first_timestamp is None:
                    rep.first_timestamp = ts_str
                    try:
                        boot_ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        boot_ts = None
                rep.last_timestamp = ts_str

            lv = LEVEL_RE.search(line)
            if lv:
                rep.level_counts[lv.group(1)] += 1

            m = MODEL_LOAD_RE.search(line)
            if m:
                rep.model_labels.add(m.group(1))
                rep.model_labels.add(m.group(2))
            m = MLX_LABEL_RE.search(line)
            if m:
                rep.model_labels.add(m.group(1))
            m = MLX_PEAK_RE.search(line)
            if m:
                rep.mlx_first_token_ms.append(int(m.group(1)))
                rep.mlx_peak_ram_gb.append(float(m.group(5)))

            m = TTS_UTTERANCE_RE.search(line) or TTS_STREAM_RE.search(line)
            if m:
                utter = m.group(1).strip()
                rep.tts_utterances.append(utter)
                low = utter.lower()
                for fp in PROMPT_LEAK_FINGERPRINTS:
                    if fp in low:
                        rep.prompt_leaks.append(utter)
                        break
                for fp in COT_PREFACE_FINGERPRINTS:
                    if fp.search(utter):
                        rep.cot_leaks.append(utter)
                        break

            m = STT_FINAL_RE.search(line)
            if m:
                rep.stt_finals.append(m.group(1).strip())

            if ECHO_SUPPRESSED_RE.search(line):
                rep.echo_suppressions += 1

            m = WATCHDOG_WARN_RE.search(line)
            if m:
                subsystem = m.group(1)
                rep.watchdog_breaches[subsystem] += 1
                if subsystem == "intent_engine" and boot_ts and rep.last_timestamp:
                    try:
                        evt_ts = datetime.strptime(rep.last_timestamp, "%Y-%m-%d %H:%M:%S")
                        if (evt_ts - boot_ts).total_seconds() <= 30:
                            rep.intent_boot_grace_breaches += 1
                        else:
                            rep.intent_late_breaches += 1
                    except ValueError:
                        rep.intent_late_breaches += 1
                elif subsystem == "intent_engine":
                    rep.intent_late_breaches += 1

            m = FIRST_TOKEN_RE.search(line)
            if m:
                rep.mlx_first_token_ms.append(int(m.group(2)))

            m = PIPELINE_TOTAL_RE.search(line)
            if m:
                rep.pipeline_totals_ms.append(int(m.group(1)))

            m = PRESSURE_RE.search(line)
            if m:
                rep.memory_pressure_events.append((int(m.group(1)), int(m.group(2)), float(m.group(3))))

            if ERROR_LINE_RE.search(line):
                rep.errors.append(line[:220])
            if TRACEBACK_RE.search(line):
                rep.tracebacks += 1

    _rank_findings(rep)
    return rep


def _rank_findings(rep: TriageReport) -> None:
    if rep.prompt_leaks:
        rep.p0_findings.append(
            f"Prompt-text leaks in TTS ({len(rep.prompt_leaks)}): see PB-01"
        )
    if rep.cot_leaks:
        rep.p0_findings.append(
            f"Chain-of-thought preface leaks ({len(rep.cot_leaks)}): see PB-02"
        )
    if rep.tracebacks:
        rep.p0_findings.append(f"Python tracebacks in log: {rep.tracebacks}")

    ratio_echo_to_finals = (
        rep.echo_suppressions / max(len(rep.stt_finals), 1) if rep.stt_finals else 0
    )
    if rep.tts_utterances and rep.stt_finals:
        self_talk = _count_self_talk(rep.tts_utterances, rep.stt_finals)
        if self_talk >= 2:
            rep.p0_findings.append(
                f"Possible self-talk: {self_talk} STT finals closely match recent TTS — see PB-03"
            )

    if rep.intent_late_breaches:
        rep.p1_findings.append(
            f"Intent-engine budget exceeded {rep.intent_late_breaches}x after boot grace — see PB-06"
        )
    if rep.intent_boot_grace_breaches:
        rep.p2_findings.append(
            f"{rep.intent_boot_grace_breaches} intent-engine breaches within 30s boot grace (expected)"
        )

    for subsystem, n in rep.watchdog_breaches.items():
        if subsystem == "intent_engine":
            continue
        rep.p1_findings.append(f"Watchdog breach: {subsystem} x{n}")

    if rep.mlx_peak_ram_gb:
        max_peak = max(rep.mlx_peak_ram_gb)
        if max_peak >= 4.0:
            rep.p1_findings.append(
                f"Peak MLX RAM {max_peak:.1f}GB suggests a ≥7B model loaded (Phi-3.5-4bit is ~2.5GB)"
            )

    if rep.errors:
        rep.p1_findings.append(f"{len(rep.errors)} ERROR-level log line(s)")

    if rep.memory_pressure_events:
        worst = max(e[2] for e in rep.memory_pressure_events)
        if worst >= 85:
            rep.p1_findings.append(f"Memory pressure peaked at {worst:.0f}%")

    if rep.pipeline_totals_ms:
        avg = sum(rep.pipeline_totals_ms) / len(rep.pipeline_totals_ms)
        if avg >= 5000:
            rep.p2_findings.append(
                f"Avg turn latency {avg:.0f}ms (target ≤ 3000ms for voice UX)"
            )

    if ratio_echo_to_finals >= 3.0:
        rep.p2_findings.append(
            f"Echo suppressions outnumber finals {ratio_echo_to_finals:.1f}:1 — TTS is chatty"
        )


def _count_self_talk(tts: list[str], finals: list[str]) -> int:
    tts_normalized = [_normalize(t) for t in tts]
    hits = 0
    for f in finals:
        nf = _normalize(f)
        if not nf:
            continue
        for tnorm in tts_normalized:
            if tnorm and (nf in tnorm or tnorm in nf):
                hits += 1
                break
    return hits


_NORM_RE = re.compile(r"[^a-z0-9 ]+")


def _normalize(s: str) -> str:
    return _NORM_RE.sub(" ", s.lower()).strip()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _fmt_ms_bucket(values: list[int]) -> str:
    if not values:
        return "n/a"
    s = sorted(values)
    n = len(s)
    p50 = s[n // 2]
    p95 = s[min(n - 1, int(0.95 * n))]
    return f"min={min(s)} p50={p50} p95={p95} max={max(s)} n={n}"


def render(rep: TriageReport) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f" ATOM TRIAGE  ·  {rep.log_path}")
    lines.append("=" * 72)
    lines.append(
        f" Boot:       {rep.first_timestamp}   →   last: {rep.last_timestamp}"
    )
    lines.append(f" Lines:      {rep.total_lines:,}")
    lines.append(
        "  Level:      "
        + "  ".join(
            f"{lv}={rep.level_counts.get(lv, 0)}"
            for lv in ("INFO", "WARNING", "ERROR", "CRITICAL", "FATAL")
        )
    )
    lines.append("")
    lines.append(" MODEL")
    lines.append("  labels:     " + (", ".join(sorted(rep.model_labels)) or "n/a"))
    if rep.mlx_peak_ram_gb:
        lines.append(
            f"  peak RAM:   max={max(rep.mlx_peak_ram_gb):.2f}GB  avg={sum(rep.mlx_peak_ram_gb)/len(rep.mlx_peak_ram_gb):.2f}GB"
        )
    lines.append(f"  first-tok:  {_fmt_ms_bucket(rep.mlx_first_token_ms)}")

    lines.append("")
    lines.append(" VOICE")
    lines.append(
        f"  TTS utterances:       {len(rep.tts_utterances)}"
        f"   (prompt-leaks: {len(rep.prompt_leaks)}  cot-leaks: {len(rep.cot_leaks)})"
    )
    lines.append(
        f"  STT finals:           {len(rep.stt_finals)}"
        f"   (echo suppressions: {rep.echo_suppressions})"
    )

    lines.append("")
    lines.append(" PIPELINE")
    lines.append(f"  turn total ms:  {_fmt_ms_bucket(rep.pipeline_totals_ms)}")
    if rep.watchdog_breaches:
        breaches = ", ".join(f"{k}={v}" for k, v in rep.watchdog_breaches.most_common())
        lines.append(f"  watchdog:       {breaches}")
    if rep.memory_pressure_events:
        lines.append(f"  mem pressure:   {len(rep.memory_pressure_events)} event(s)")

    lines.append("")
    lines.append(" FINDINGS")

    def _section(label: str, items: list[str]) -> None:
        if not items:
            return
        lines.append(f"  [{label}]")
        for it in items:
            lines.append(f"    - {it}")

    _section("P0", rep.p0_findings)
    _section("P1", rep.p1_findings)
    _section("P2", rep.p2_findings)
    if not (rep.p0_findings or rep.p1_findings or rep.p2_findings):
        lines.append("  clean · no flagged issues")

    if rep.prompt_leaks:
        lines.append("")
        lines.append(" PROMPT LEAK SAMPLES")
        for s in rep.prompt_leaks[:5]:
            lines.append(f"    · {s[:140]}")
    if rep.cot_leaks:
        lines.append("")
        lines.append(" COT LEAK SAMPLES")
        for s in rep.cot_leaks[:5]:
            lines.append(f"    · {s[:140]}")
    if rep.errors:
        lines.append("")
        lines.append(" ERROR SAMPLES")
        for s in rep.errors[:5]:
            lines.append(f"    · {s}")

    lines.append("")
    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("atomlogs.txt")
    if not path.exists():
        print(f"triage_log: file not found: {path}", file=sys.stderr)
        return 2
    rep = parse_log(path)
    print(render(rep))
    return 1 if rep.p0_findings else 0


if __name__ == "__main__":
    sys.exit(main())
