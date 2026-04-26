"""Sprint Ω.9 quick audit (no VLM, no full boot, no embedding load).

Captures the metrics that the sprint moved:
  - quick-reply gate behavior on explanatory queries
  - semantic-cache class gate spot checks
  - cloud rotation slot health (cerebras gpt-oss-120b live)
  - executor pool health (light/heavy split)
  - WhisperKit kick_serve_async surface check
  - TTS preflight_speak surface check
  - Boot-warm pass: structural call-sites in main.py

Skips the embedding engine cold-load test because the local ``mlx-embeddings``
install is broken against the current ``huggingface_hub`` (see
ModuleNotFoundError on ``huggingface_hub.utils._errors``). The warmup-encode
change still lands -- it activates the next time a working provider loads.

Writes ``audit_omega9_report.json`` next to itself.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    report: dict = {"sprint": "omega.9", "ts": time.time()}
    cfg = json.loads((ROOT / "config/settings.json").read_text())

    # ── 1. Config drift verification ──────────────────────────────────
    print("[1/6] config drift...", flush=True)
    cerebras = next(
        (
            p
            for p in cfg["cloud"]["rotation"]["providers"]
            if p.get("name") == "cerebras"
        ),
        {},
    )
    report["config"] = {
        "cross_device.enabled": cfg.get("cross_device", {}).get("enabled"),
        "cognitive_kernel.quick_model": cfg["cognitive_kernel"]["quick_model"],
        "cognitive_kernel.full_model": cfg["cognitive_kernel"]["full_model"],
        "cerebras.deep_model": cerebras.get("deep_model"),
        "brain.mlx_model": cfg["brain"]["mlx_model"],
    }
    drift_ok = (
        report["config"]["cross_device.enabled"] is True
        and report["config"]["cognitive_kernel.quick_model"]
        == "qwen3-4b-instruct-4bit"
        and report["config"]["cognitive_kernel.full_model"]
        == "qwen3-4b-instruct-4bit"
        and report["config"]["cerebras.deep_model"] == "gpt-oss-120b"
    )
    print(f"   drift cleared: {drift_ok}", flush=True)
    report["config_drift_cleared"] = drift_ok

    # ── 2. Quick-reply explanatory gate ───────────────────────────────
    print("[2/6] quick-reply explanatory gate...", flush=True)
    from core.quick_replies import try_quick_reply

    cases = [
        ("hi", "filler-fires"),
        ("how are you", "filler-fires"),
        ("tell me about yourself", "domain-fires"),
        ("how does unified memory work in detail", "domain-fires"),
        (
            "explain the difference between optimal and full performance",
            "domain-fires",
        ),
        (
            "explain how WhisperKit runs on the Apple Neural Engine",
            "explain-gate-blocks",
        ),
        ("compare safari and arc for coding on macbook air", "domain-fires"),
        ("walk me through how speculative decoding works", "explain-gate-blocks"),
        (
            "what is the latency difference between optimal mode and full performance mode",
            "domain-fires",
        ),
        (
            "describe how Apple Silicon and the Neural Engine cooperate during inference",
            "explain-gate-blocks",
        ),
        (
            "elaborate on how prompt caching reduces first-token latency",
            "explain-gate-blocks",
        ),
    ]
    qr_results = []
    for q, kind in cases:
        out = try_quick_reply(q, cfg)
        fired = out is not None
        ok = (kind.endswith("fires") and fired) or (
            kind.endswith("blocks") and not fired
        )
        qr_results.append(
            {
                "query": q,
                "expected": kind,
                "fired": fired,
                "ok": ok,
                "head": (out or "")[:100],
            }
        )
    report["quick_reply"] = qr_results
    qr_ok = all(c["ok"] for c in qr_results)
    print(f"   quick-reply gates correct: {qr_ok}", flush=True)
    report["quick_reply_gates_ok"] = qr_ok

    # ── 3. Semantic-cache class gate ──────────────────────────────────
    print("[3/6] semantic-cache class gate...", flush=True)
    from core.cognitive_kernel import _force_exact_semantic_cache

    sem_cases = [
        ("who are you", True),
        ("hi", True),
        ("what time is it", True),
        ("hello", True),
        ("namaste", True),
        ("how are you", True),
        ("good morning", True),
        ("how does Apple Silicon use the Neural Engine", False),
        ("can you draft a thank-you email to Riya", False),
        ("set a 25 minute focus timer", False),
    ]
    sem_results = []
    for q, expect_exact in sem_cases:
        got = _force_exact_semantic_cache(q)
        sem_results.append(
            {
                "query": q,
                "expected_exact_only": expect_exact,
                "got": got,
                "ok": got == expect_exact,
            }
        )
    report["semantic_cache_gate"] = sem_results
    sem_ok = all(c["ok"] for c in sem_results)
    print(f"   class-gate matches expectations: {sem_ok}", flush=True)
    report["semantic_cache_gate_ok"] = sem_ok

    # ── 4. Cloud rotation slots ───────────────────────────────────────
    print("[4/6] cloud rotation slots...", flush=True)
    try:
        from core.cloud.rotating_openai_client import RotatingOpenAIClient

        client = RotatingOpenAIClient(cfg)
        diag = client.diagnostics()
        slots = diag.get("slots", []) or []
        # Sprint Ω.10 (Apr 27 2026): the per-slot key in the diagnostics
        # payload is ``has_key`` (RotatingOpenAIClient.diagnostics), not
        # ``api_key_present``. The audit also surfaces whether the
        # encrypted vault was unlocked in the audit env so a "no ready
        # slots" reading can be distinguished from "vault sealed".
        report["cloud_rotation"] = {
            "enabled": diag.get("enabled"),
            "provider": diag.get("provider"),
            "available": diag.get("available"),
            "vault_unlocked_in_audit_env": bool(
                os.environ.get("ATOM_MASTER_PASSWORD"),
            ),
            "slot_count": len(slots),
            "slot_names": [s.get("name") for s in slots],
            "tiers": [s.get("tier") for s in slots],
            "ready_slots": [
                s["name"] for s in slots if s.get("has_key")
            ],
            "deep_models": {s.get("name"): s.get("deep_model") for s in slots},
        }
        print(
            f"   ready_slots = {report['cloud_rotation']['ready_slots']}",
            flush=True,
        )
        print(
            f"   cerebras deep_model = "
            f"{report['cloud_rotation']['deep_models'].get('cerebras')}",
            flush=True,
        )
    except Exception as exc:
        report["cloud_rotation"] = {
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(f"   cloud rotation probe failed: {exc}", flush=True)

    # ── 5. Executor split + voice surface ─────────────────────────────
    print("[5/6] executor pools + voice surface...", flush=True)
    from core.async_event_bus import (
        get_heavy_executor,
        get_light_executor,
        shutdown_bus_executors,
    )
    from voice.stt_whisperkit import WhisperKitSTT
    from voice.tts_macos import MacOSTTSAsync

    light = get_light_executor()
    heavy = get_heavy_executor()
    report["executor_pools"] = {
        "light_max_workers": light._max_workers,
        "heavy_max_workers": heavy._max_workers,
        "light_prefix": light._thread_name_prefix,
        "heavy_prefix": heavy._thread_name_prefix,
        "light_alive": not getattr(light, "_shutdown", False),
        "heavy_alive": not getattr(heavy, "_shutdown", False),
    }
    report["voice_surface"] = {
        "stt.kick_serve_async": callable(
            getattr(WhisperKitSTT, "kick_serve_async", None)
        ),
        "tts.preflight_speak": callable(
            getattr(MacOSTTSAsync, "preflight_speak", None)
        ),
    }
    print(
        "   pools light/heavy = "
        f"{report['executor_pools']['light_max_workers']}/"
        f"{report['executor_pools']['heavy_max_workers']}",
        flush=True,
    )
    print(
        "   voice surface kick="
        f"{report['voice_surface']['stt.kick_serve_async']} "
        f"preflight={report['voice_surface']['tts.preflight_speak']}",
        flush=True,
    )
    shutdown_bus_executors()

    # ── 6. Boot warm wiring in main.py ────────────────────────────────
    print("[6/6] main.py boot-warm wiring...", flush=True)
    main_src = (ROOT / "main.py").read_text()
    report["boot_warm_wiring"] = {
        "_background_boot_warm_defined": "_background_boot_warm" in main_src,
        "kick_serve_async_call": "stt, \"kick_serve_async\"" in main_src
        or 'getattr(stt, "kick_serve_async"' in main_src,
        "preflight_speak_call": 'tts, "preflight_speak"' in main_src
        or "preflight_speak()" in main_src,
        "embedding_seed_call": "seed_warm_cache" in main_src,
    }
    fp_src = (ROOT / "core/fast_path.py").read_text()
    report["fast_path_wiring"] = {
        "uses_get_light_executor": "get_light_executor" in fp_src,
        "has_partial_prefetch_timeout": (
            "_PARTIAL_PREFETCH_TIMEOUT_S" in fp_src
        ),
        "has_diagnostics_timeouts": "prefetch_timeouts" in fp_src,
    }
    print(
        f"   main.py boot warm wired: "
        f"{report['boot_warm_wiring']['_background_boot_warm_defined']}",
        flush=True,
    )
    print(
        f"   fast_path uses light pool: "
        f"{report['fast_path_wiring']['uses_get_light_executor']}",
        flush=True,
    )

    # ── Summary scoring ───────────────────────────────────────────────
    report["summary"] = {
        "config_drift_cleared": drift_ok,
        "quick_reply_gates_ok": qr_ok,
        "semantic_cache_gate_ok": sem_ok,
        "executor_split_alive": (
            report["executor_pools"]["light_alive"]
            and report["executor_pools"]["heavy_alive"]
            and report["executor_pools"]["light_max_workers"] == 2
            and report["executor_pools"]["heavy_max_workers"] == 3
        ),
        "voice_surface_present": all(report["voice_surface"].values()),
        "boot_warm_wired": all(report["boot_warm_wiring"].values()),
        "fast_path_wired": all(report["fast_path_wiring"].values()),
    }
    overall = all(report["summary"].values())
    report["summary"]["overall_pass"] = overall

    out = ROOT / "audit_omega9_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\noverall pass: {overall}", flush=True)
    print(f"report -> {out}", flush=True)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
