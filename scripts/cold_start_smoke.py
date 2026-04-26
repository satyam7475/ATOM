#!/usr/bin/env python3
"""S1 -- Cold-start smoke test (Sprint P1/P2/P3 verification).

Exercises the same import + boot pipeline ``main.py`` does on a real
launch, *without* opening the dashboard or microphone. The point is to
prove ATOM reaches a "READY" state -- config parsed, MLX brain
discoverable, embedding engine usable, voice pipeline factory healthy --
in the budget the plan claims.

Pass criteria (per docs/ATOM_NEXT_STEPS_PLAN.md §4):
    * ``READY`` (full boot path) reached in <= 8 s warm, <= 12 s cold.
    * No exception raised.

Exit codes:
    0  OK and within budget
    1  exception during boot
    2  boot exceeded the budget (only with --strict)

Usage::

    python scripts/cold_start_smoke.py
    python scripts/cold_start_smoke.py --json
    python scripts/cold_start_smoke.py --strict --budget-warm 8.0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from typing import Any


def _phase(label: str, fn: Any) -> dict[str, Any]:
    t0 = time.perf_counter()
    error: str | None = None
    result: Any = None
    try:
        result = fn()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = (time.perf_counter() - t0) * 1000.0
    return {
        "label": label,
        "elapsed_ms": round(elapsed, 1),
        "ok": error is None,
        "error": error,
        "result": result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ATOM S1 cold-start probe")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--strict", action="store_true",
        help="fail when total elapsed exceeds the budget",
    )
    parser.add_argument(
        "--budget-warm", type=float, default=8.0,
        help="seconds; warm-cache total budget (default 8.0)",
    )
    parser.add_argument(
        "--budget-cold", type=float, default=12.0,
        help="seconds; cold-cache total budget (default 12.0)",
    )
    parser.add_argument(
        "--cold", action="store_true",
        help="treat this run as a cold start for budget purposes",
    )
    args = parser.parse_args()

    phases: list[dict[str, Any]] = []
    t_total = time.perf_counter()

    def _import_config() -> Any:
        from core.boot.config_loader import load_config
        return load_config()

    cfg_phase = _phase("config_load", _import_config)
    phases.append(cfg_phase)
    cfg = cfg_phase["result"]
    if not cfg_phase["ok"] or cfg is None:
        _summarise(phases, args, t_total)
        return 1

    def _build_brain() -> dict[str, Any]:
        from brain.mlx_llm import MLXBrain
        b = MLXBrain(cfg)
        return {
            "available": bool(b.is_available()),
            "model_path": b._model_path,
            "speculative_enabled": bool(b._speculative_enabled),
            "mx_compile_enabled": bool(b._mx_compile_enabled),
        }

    phases.append(_phase("brain_construct", _build_brain))

    def _build_embed() -> dict[str, Any]:
        from core.embedding_engine import EmbeddingEngine
        eng = EmbeddingEngine(cfg)
        meta = eng.provider_metadata()
        # Issue one embed to validate the path. Cached after first call.
        vec = eng.embed_sync("atom cold start probe")
        meta["embed_ok"] = bool(vec)
        return meta

    phases.append(_phase("embedding_engine", _build_embed))

    def _voice_pipeline_imports() -> dict[str, Any]:
        # Don't actually instantiate a VoicePipeline (it wants a bus, mic,
        # router). Just prove the imports + the factory aliases are sound;
        # this is the part the plan calls out.
        from voice.voice_pipeline import VoicePipeline
        return {
            "pipeline_class_loaded": True,
            "whisperkit_aliases": list(VoicePipeline._WHISPERKIT_ALIASES),
            "whisper_cpp_aliases": list(VoicePipeline._WHISPER_CPP_ALIASES),
        }

    phases.append(_phase("voice_pipeline_imports", _voice_pipeline_imports))

    def _stt_factory_probe() -> dict[str, Any]:
        from voice.stt_whisperkit import is_whisperkit_available
        return {
            "whisperkit_cli_present": bool(is_whisperkit_available(cfg)),
        }

    phases.append(_phase("stt_factory_probe", _stt_factory_probe))

    return _summarise(phases, args, t_total)


def _summarise(
    phases: list[dict[str, Any]],
    args: argparse.Namespace,
    t_total: float,
) -> int:
    total_ms = (time.perf_counter() - t_total) * 1000.0
    total_s = total_ms / 1000.0
    budget_s = args.budget_cold if args.cold else args.budget_warm
    over_budget = total_s > budget_s

    any_failed = any(not p["ok"] for p in phases)
    summary: dict[str, Any] = {
        "status": (
            "boot_failed" if any_failed else
            "over_budget" if over_budget and args.strict else
            "ok"
        ),
        "total_ms": round(total_ms, 1),
        "total_s": round(total_s, 2),
        "budget_s": budget_s,
        "over_budget": over_budget,
        "phases": phases,
    }

    if args.json:
        print(json.dumps(summary, default=str))
    else:
        print(
            f"ATOM cold start: total={summary['total_s']:.2f}s "
            f"(budget={budget_s:.1f}s) status={summary['status']}",
        )
        for p in phases:
            mark = "OK" if p["ok"] else "FAIL"
            print(
                f"  [{mark}] {p['label']:<26} {p['elapsed_ms']:>8.1f} ms "
                + (f"-- {p['error']}" if p["error"] else ""),
            )
            if p["ok"] and isinstance(p["result"], dict):
                for k, v in p["result"].items():
                    print(f"      {k}: {v}")

    if any_failed:
        return 1
    if over_budget and args.strict:
        return 2
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
