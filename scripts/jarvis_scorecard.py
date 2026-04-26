#!/usr/bin/env python3
"""ATOM Jarvis scorecard from a boot/runtime log.

This is intentionally log-only and side-effect free. Run it after each fresh
boot to keep ATOM optimised around the owner experience instead of feature
count.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_MS_PATTERNS: dict[str, re.Pattern[str]] = {
    "tts_ready_ms": re.compile(r"TTS ready \((?P<v>\d+(?:\.\d+)?)ms"),
    "local_brain_ready_ms": re.compile(r"Local brain ready in (?P<v>\d+(?:\.\d+)?)ms"),
    "cold_start_ms": re.compile(r"Cold-start bootstrap: (?P<v>\d+(?:\.\d+)?)ms"),
    "whisperkit_preload_ms": re.compile(r"WhisperKitSTT preloaded .* in (?P<v>\d+(?:\.\d+)?) ms"),
    "stt_pipeline_ready_ms": re.compile(r"STT pipeline ready \((?P<v>\d+(?:\.\d+)?)ms"),
    "voice_loop_ready_ms": re.compile(r"VOICE_LOOP_READY: (?P<v>\d+(?:\.\d+)?)ms"),
}

_BOOT_TIMELINE_RE = re.compile(
    r"Boot timeline: total=(?P<boot>\d+(?:\.\d+)?)ms .*?"
    r"tts_init=(?P<tts>\d+(?:\.\d+)?)ms "
    r"cold_start=(?P<cold>\d+(?:\.\d+)?)ms .*?"
    r"stt_preload=(?P<stt>\d+(?:\.\d+)?)ms"
)
_MEMORY_PCT_RE = re.compile(
    r"memory(?:_pct|_percent)?['\"]?[=:\s]+(?P<v>\d+(?:\.\d+)?)%?"
)
_MEMORY_WARNING_RE = re.compile(r"Memory pressure warning: (?P<v>\d+(?:\.\d+)?)%")

_PROMPT_LEAK_NEEDLES = (
    "final answer only",
    "tool call format",
    "available tools",
    "system prompt",
)
_COT_NEEDLES = (
    "okay, let's see",
    "let me think",
    "we need answer",
    "chain of thought",
)
_POLITE_INTERRUPT_RE = re.compile(
    r"Voice interrupt partial detected: ['\"](?P<t>thank you|thanks|okay|ok|go ahead)\.?",
    re.I,
)


@dataclass
class JarvisScorecard:
    log_path: str
    boot_total_ms: float | None = None
    stt_pipeline_ready_ms: float | None = None
    whisperkit_preload_ms: float | None = None
    tts_ready_ms: float | None = None
    local_brain_ready_ms: float | None = None
    cold_start_ms: float | None = None
    voice_loop_ready_ms: float | None = None
    voice_pipeline_active: bool = False
    stt_listening_active: bool = False
    max_memory_pct: float | None = None
    memory_pressure_events: int = 0
    prompt_leak_candidates: int = 0
    cot_leak_candidates: int = 0
    echo_suppressions: int = 0
    polite_interrupt_candidates: int = 0
    adam_to_atom_corrections: int = 0
    realtime_room_started: bool = False
    iphone_bridge_started: bool = False
    screen_loop_started: bool = False
    score: int = 0
    grade: str = "unknown"
    findings: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _update_ms(scorecard: JarvisScorecard, key: str, value: float) -> None:
    old = getattr(scorecard, key)
    if old is None or value > old:
        setattr(scorecard, key, value)


def parse_scorecard(path: str | Path) -> JarvisScorecard:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    scorecard = JarvisScorecard(log_path=str(p))

    for line in text.splitlines():
        for key, pattern in _MS_PATTERNS.items():
            match = pattern.search(line)
            if match:
                _update_ms(scorecard, key, float(match.group("v")))

        boot = _BOOT_TIMELINE_RE.search(line)
        if boot:
            scorecard.boot_total_ms = float(boot.group("boot"))
            scorecard.tts_ready_ms = float(boot.group("tts"))
            scorecard.cold_start_ms = float(boot.group("cold"))
            scorecard.stt_pipeline_ready_ms = float(boot.group("stt"))

        mem = _MEMORY_WARNING_RE.search(line) or _MEMORY_PCT_RE.search(line)
        if mem:
            val = float(mem.group("v"))
            if scorecard.max_memory_pct is None or val > scorecard.max_memory_pct:
                scorecard.max_memory_pct = val

        lower = line.lower()
        if "voice_loop_ready" in lower:
            scorecard.voice_pipeline_active = True
        if "whisperkitstt listening" in lower or "stt ready -- atom fully operational" in lower:
            scorecard.stt_listening_active = True
        if "memory pressure" in lower:
            scorecard.memory_pressure_events += 1
        if "echo suppressed" in lower:
            scorecard.echo_suppressions += 1
        if "adam" in lower and "->" in line and "atom" in lower:
            scorecard.adam_to_atom_corrections += 1
        if "atomroom listening" in lower or "realtime room ready" in lower:
            scorecard.realtime_room_started = True
        if "iphone bridge listening" in lower or "iphone bridge online" in lower:
            scorecard.iphone_bridge_started = True
        if "screen perception loop running" in lower:
            scorecard.screen_loop_started = True
        if _POLITE_INTERRUPT_RE.search(line):
            scorecard.polite_interrupt_candidates += 1
        if "tts [" in lower or "tts ack" in lower:
            if any(needle in lower for needle in _PROMPT_LEAK_NEEDLES):
                scorecard.prompt_leak_candidates += 1
            if any(needle in lower for needle in _COT_NEEDLES):
                scorecard.cot_leak_candidates += 1

    _score(scorecard)
    return scorecard


def _score(card: JarvisScorecard) -> None:
    score = 100
    findings: list[str] = []

    def penalty(points: int, msg: str) -> None:
        nonlocal score
        score -= points
        findings.append(msg)

    if not card.voice_pipeline_active:
        penalty(20, "Voice loop did not reach VOICE_LOOP_READY.")
    if not card.stt_listening_active:
        penalty(20, "STT never reached listening/fully-operational state.")

    if card.boot_total_ms is None:
        penalty(8, "Boot timeline missing; cannot track total readiness.")
    elif card.boot_total_ms > 25_000:
        penalty(12, f"Boot readiness is high: {card.boot_total_ms:.0f} ms.")
    elif card.boot_total_ms > 18_000:
        penalty(7, f"Boot readiness above target: {card.boot_total_ms:.0f} ms.")

    if card.stt_pipeline_ready_ms is None:
        penalty(8, "STT readiness metric missing.")
    elif card.stt_pipeline_ready_ms > 20_000:
        penalty(14, f"STT ready time is high: {card.stt_pipeline_ready_ms:.0f} ms.")
    elif card.stt_pipeline_ready_ms > 12_000:
        penalty(7, f"STT ready time above target: {card.stt_pipeline_ready_ms:.0f} ms.")

    if card.max_memory_pct is None:
        penalty(6, "Memory percentage missing from log.")
    elif card.max_memory_pct >= 82:
        penalty(14, f"Memory pressure hit {card.max_memory_pct:.1f}%.")
    elif card.max_memory_pct > 75:
        penalty(7, f"Idle/runtime memory above target: {card.max_memory_pct:.1f}%.")

    if card.memory_pressure_events:
        penalty(min(10, card.memory_pressure_events * 3), "Memory pressure events occurred.")
    if card.prompt_leak_candidates:
        penalty(15, "Prompt text may have leaked into TTS.")
    if card.cot_leak_candidates:
        penalty(15, "Chain-of-thought style text may have leaked into TTS.")
    if card.polite_interrupt_candidates:
        penalty(10, "Polite phrase partials triggered interrupt handling.")
    if card.realtime_room_started:
        penalty(3, "Realtime room started; keep disabled unless actively needed.")
    if card.iphone_bridge_started:
        penalty(3, "iPhone bridge started; keep disabled unless actively needed.")

    card.score = max(0, min(100, score))
    if card.score >= 90:
        card.grade = "A"
    elif card.score >= 80:
        card.grade = "B"
    elif card.score >= 70:
        card.grade = "C"
    elif card.score >= 60:
        card.grade = "D"
    else:
        card.grade = "Needs work"
    card.findings = findings


def _print_human(card: JarvisScorecard) -> None:
    print(f"ATOM Jarvis scorecard: {card.score}/100 ({card.grade})")
    print(f"  log: {card.log_path}")
    print(f"  voice active: {card.voice_pipeline_active}")
    print(f"  stt listening: {card.stt_listening_active}")
    print(f"  boot total: {_fmt_ms(card.boot_total_ms)}")
    print(f"  stt ready: {_fmt_ms(card.stt_pipeline_ready_ms)}")
    print(f"  whisperkit preload: {_fmt_ms(card.whisperkit_preload_ms)}")
    print(f"  tts ready: {_fmt_ms(card.tts_ready_ms)}")
    print(f"  local brain ready: {_fmt_ms(card.local_brain_ready_ms)}")
    print(f"  cold start: {_fmt_ms(card.cold_start_ms)}")
    print(f"  max memory: {_fmt_pct(card.max_memory_pct)}")
    print(f"  memory pressure events: {card.memory_pressure_events}")
    print(f"  prompt leaks / CoT leaks: {card.prompt_leak_candidates} / {card.cot_leak_candidates}")
    print(f"  polite interrupt candidates: {card.polite_interrupt_candidates}")
    print(f"  Adam->Atom corrections: {card.adam_to_atom_corrections}")
    print(
        "  optional loops started: "
        f"realtime={card.realtime_room_started}, "
        f"iphone={card.iphone_bridge_started}, "
        f"screen_loop={card.screen_loop_started}"
    )
    if card.findings:
        print("  findings:")
        for finding in card.findings:
            print(f"    - {finding}")


def _fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f} ms"


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score ATOM owner-experience readiness from a log")
    parser.add_argument("log", nargs="?", default="atomCurrentLogs.txt")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true", help="exit 2 when score is below --min-score")
    parser.add_argument("--min-score", type=int, default=80)
    args = parser.parse_args(argv)

    card = parse_scorecard(args.log)
    if args.json:
        print(json.dumps(card.to_dict(), indent=2))
    else:
        _print_human(card)

    if args.strict and card.score < args.min_score:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
