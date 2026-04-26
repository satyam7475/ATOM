"""Focused live audit of ATOM's brain stack on the current 4B config.

Measures (against the *running* code, no full app boot):
  1. Cold MLX load time (4B-4bit on M5)
  2. First-token latency on a short voice-style prompt
  3. Sustained tokens/sec on a 80-token generation
  4. Second-turn prefill speed (prompt cache hit)
  5. Long-context behavior (6144 ctx target with ~1800-token prompt)
  6. Memory footprint delta (rss MB)

Writes a JSON report next to itself.
"""

from __future__ import annotations

import asyncio
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import psutil  # noqa: E402

PROC = psutil.Process()


def _mem_mb() -> float:
    return PROC.memory_info().rss / (1024 * 1024)


def _system_mem_pct() -> float:
    return psutil.virtual_memory().percent


async def _main() -> int:
    report: dict = {
        "cwd": os.getcwd(),
        "system": {
            "python": sys.version.split()[0],
            "cpu_count": psutil.cpu_count(),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
            "ram_used_pct_start": _system_mem_pct(),
        },
    }

    cfg_path = ROOT / "config/settings.json"
    cfg = json.loads(cfg_path.read_text())
    brain_cfg = cfg["brain"]
    mlx_path = brain_cfg.get("mlx_model")
    report["config"] = {
        "mlx_model": mlx_path,
        "n_ctx": brain_cfg.get("n_ctx"),
        "max_tokens": brain_cfg.get("max_tokens"),
        "single_resident": brain_cfg.get("single_resident"),
    }

    print("== ATOM brain audit ==")
    print(json.dumps(report, indent=2))

    from brain.mlx_llm import MLXBrain

    cfg_min = {
        "brain": {
            "mlx_model": mlx_path,
            "single_resident": True,
            "max_tokens": 64,
            "temperature": 0.7,
            "n_ctx": brain_cfg.get("n_ctx", 6144),
        }
    }

    brain = MLXBrain(cfg_min)

    gc.collect()
    rss_pre = _mem_mb()
    t0 = time.perf_counter()
    ok = brain.preload(load_all=False)
    cold_ms = (time.perf_counter() - t0) * 1000.0
    rss_post = _mem_mb()
    print(f"  MLX cold load: {cold_ms:.0f}ms  ok={ok}  rss {rss_pre:.0f} -> {rss_post:.0f}MB (+{rss_post-rss_pre:.0f})")
    report["mlx_cold_load_ms"] = round(cold_ms, 1)
    report["mlx_load_ok"] = bool(ok)
    report["rss_after_load_mb"] = round(rss_post, 1)
    report["rss_load_delta_mb"] = round(rss_post - rss_pre, 1)

    if not ok:
        report["error"] = "model_load_failed"
        (ROOT / "audit_brain_report.json").write_text(json.dumps(report, indent=2))
        return 1

    short_prompt = (
        "<|im_start|>system\nYou are ATOM. Reply tersely.<|im_end|>\n"
        "<|im_start|>user\nWhat time is it where you are?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    print("  >> turn 1 (cold prefill)")
    t1 = time.perf_counter()
    out1, ok1 = await brain.generate(short_prompt, max_tokens_override=48)
    t1_ms = (time.perf_counter() - t1) * 1000.0
    tokens1 = max(1, len(out1.split()))
    tps1 = tokens1 * 1000.0 / max(1.0, t1_ms)
    print(f"     {t1_ms:.0f}ms  ok={ok1}  ~{tokens1} words  {tps1:.1f} wps  -> {out1[:140]!r}")
    report["turn1_ms"] = round(t1_ms, 1)
    report["turn1_words"] = tokens1
    report["turn1_words_per_sec"] = round(tps1, 2)
    report["turn1_text_head"] = out1[:200]

    short_prompt_2 = (
        "<|im_start|>system\nYou are ATOM. Reply tersely.<|im_end|>\n"
        "<|im_start|>user\nName the largest moon of Jupiter in one word.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    print("  >> turn 2 (warm prefill / prompt cache)")
    t2 = time.perf_counter()
    out2, ok2 = await brain.generate(short_prompt_2, max_tokens_override=24)
    t2_ms = (time.perf_counter() - t2) * 1000.0
    tokens2 = max(1, len(out2.split()))
    tps2 = tokens2 * 1000.0 / max(1.0, t2_ms)
    print(f"     {t2_ms:.0f}ms  ok={ok2}  ~{tokens2} words  {tps2:.1f} wps  -> {out2[:140]!r}")
    report["turn2_ms"] = round(t2_ms, 1)
    report["turn2_words"] = tokens2
    report["turn2_words_per_sec"] = round(tps2, 2)
    report["turn2_text_head"] = out2[:200]

    print("  >> turn 3 (longer 80-token target, sustained tps)")
    long_prompt = (
        "<|im_start|>system\nYou are ATOM, Satyam's calm cognitive AI. Speak in 3 short sentences.<|im_end|>\n"
        "<|im_start|>user\nGive a status report: what is unified memory and why does it matter on M-series Macs?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    t3 = time.perf_counter()
    out3, ok3 = await brain.generate(long_prompt, max_tokens_override=96)
    t3_ms = (time.perf_counter() - t3) * 1000.0
    tokens3 = max(1, len(out3.split()))
    tps3 = tokens3 * 1000.0 / max(1.0, t3_ms)
    print(f"     {t3_ms:.0f}ms  ok={ok3}  ~{tokens3} words  {tps3:.1f} wps  -> {out3[:140]!r}")
    report["turn3_ms"] = round(t3_ms, 1)
    report["turn3_words"] = tokens3
    report["turn3_words_per_sec"] = round(tps3, 2)
    report["turn3_text_head"] = out3[:300]

    # Synthetic ~1800-token prompt to validate voice-mode cap survives.
    bulk = (" ".join(["status"] * 220)).strip()
    bulk_prompt = (
        "<|im_start|>system\nYou are ATOM, terse.<|im_end|>\n"
        f"<|im_start|>user\nIgnore this fluff: {bulk}\nIn one sentence, what is 2+2?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    print("  >> turn 4 (~1800-tok prefill, voice-cap stress)")
    t4 = time.perf_counter()
    out4, ok4 = await brain.generate(bulk_prompt, max_tokens_override=24)
    t4_ms = (time.perf_counter() - t4) * 1000.0
    print(f"     {t4_ms:.0f}ms  ok={ok4}  -> {out4[:140]!r}")
    report["turn4_long_prefill_ms"] = round(t4_ms, 1)
    report["turn4_text_head"] = out4[:200]

    rss_final = _mem_mb()
    sys_pct_final = _system_mem_pct()
    report["rss_final_mb"] = round(rss_final, 1)
    report["system_ram_pct_final"] = sys_pct_final
    print(f"  final rss {rss_final:.0f}MB | system RAM {sys_pct_final:.1f}%")

    out_path = ROOT / "audit_brain_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n  report -> {out_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(_main()))
    except KeyboardInterrupt:
        sys.exit(130)
