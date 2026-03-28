#!/usr/bin/env python3
"""
ATOM -- End-to-end module timing benchmark (no full mic/TTS playback).

Measures: config load, intent classification, memory recall, prompt build,
local LLM inference for daily-work / buddy / learning style queries.

Run from ATOM root:
    py -3.11 tools/e2e_benchmark.py
    py -3.11 tools/e2e_benchmark.py --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def _pct(samples: list[float], p: float) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    idx = min(len(s) - 1, int(len(s) * p))
    return s[idx]


def bench_intent(engine, phrases: list[str], iterations: int = 3) -> dict:
    all_ms: list[float] = []
    per_phrase: dict[str, list[float]] = {}
    for _ in range(iterations):
        for p in phrases:
            t0 = time.perf_counter()
            engine.classify(p)
            ms = (time.perf_counter() - t0) * 1000
            all_ms.append(ms)
            per_phrase.setdefault(p, []).append(ms)
    return {
        "samples": len(all_ms),
        "avg_ms": round(statistics.mean(all_ms), 3),
        "p50_ms": round(_pct(all_ms, 0.5), 3),
        "p95_ms": round(_pct(all_ms, 0.95), 3),
        "max_ms": round(max(all_ms), 3),
        "per_phrase_avg_ms": {
            k[:48]: round(statistics.mean(v), 3) for k, v in per_phrase.items()
        },
    }


async def bench_llm(llm, prompts: list[tuple[str, str]]) -> list[dict]:
    out = []
    for label, prompt in prompts:
        t0 = time.perf_counter()
        text, _ = await llm.generate(prompt)
        elapsed = (time.perf_counter() - t0) * 1000
        out.append({
            "scenario": label,
            "prompt_chars": len(prompt),
            "response_chars": len(text),
            "elapsed_ms": round(elapsed, 1),
            "response_preview": (text[:180] + "…") if len(text) > 180 else text,
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    args = parser.parse_args()

    results: dict = {
        "atom_root": str(ROOT),
        "benchmark": "e2e_module_timings_v15",
    }

    # 1) Config load
    t0 = time.perf_counter()
    with open(ROOT / "config" / "settings.json", encoding="utf-8") as f:
        config = json.load(f)
    results["config_load_ms"] = round((time.perf_counter() - t0) * 1000, 3)

    # 2) Intent engine -- daily work + buddy + learning style utterances
    from core.intent_engine import IntentEngine

    intent_engine = IntentEngine()
    daily_work = [
        "what time is it",
        "open notepad",
        "close chrome",
        "how much ram am I using",
        "set a timer for five minutes",
        "show my reminders",
        "lock the screen",
    ]
    buddy_chat = [
        "hey atom how's your day",
        "thanks buddy you're the best",
        "I'm exhausted today",
        "tell me a joke",
    ]
    learning = [
        "explain recursion in simple terms",
        "what is the difference between list and tuple in python",
        "how does a CPU cache work",
    ]
    fallback_like = [
        "what would you do if you could feel emotions",
        "help me plan my study schedule for SSC exam",
    ]
    all_phrases = daily_work + buddy_chat + learning + fallback_like
    results["intent_classification"] = bench_intent(intent_engine, all_phrases, iterations=5)

    # 3) Router compress_query (sync path cost)
    from core.router.router import compress_query

    cq_samples = []
    for p in all_phrases:
        t0 = time.perf_counter()
        compress_query(p)
        cq_samples.append((time.perf_counter() - t0) * 1000)
    results["compress_query"] = {
        "avg_ms": round(statistics.mean(cq_samples), 4),
        "max_ms": round(max(cq_samples), 4),
    }

    # 4) Memory engine retrieve (async API; keyword overlap loop)
    from core.memory_engine import MemoryEngine

    mem = MemoryEngine(config)

    async def _mem_bench():
        t0 = time.perf_counter()
        for _ in range(50):
            await mem.retrieve("python system cpu reminder study", k=3)
        return (time.perf_counter() - t0) * 1000 / 50

    results["memory_retrieve_50x_avg_ms"] = round(asyncio.run(_mem_bench()), 4)

    # 5) Cache engine
    from core.cache_engine import CacheEngine

    cache = CacheEngine(config)
    t0 = time.perf_counter()
    for i in range(200):
        cache.get(f"test query number {i} unique")
    results["cache_miss_get_200_avg_ms"] = round(
        (time.perf_counter() - t0) * 1000 / 200, 4,
    )

    # 6) Structured prompt build (same path as local brain)
    from cursor_bridge.structured_prompt_builder import StructuredPromptBuilder

    spb = StructuredPromptBuilder(config)
    pb_samples = []
    queries = [
        ("daily", "Remind me to send the weekly report before 5pm."),
        ("buddy", "Atom, I've had a rough day. Can we just talk for a minute?"),
        ("learning", "Teach me the Feynman technique for learning hard topics."),
    ]
    for tag, q in queries:
        t0 = time.perf_counter()
        spb.build(
            q,
            memory_summaries=["User prefers short answers.", "Boss works in tech."],
            history=[("previous", "ATOM helped with RAM tips.")],
            context={"active_app": "Cursor", "window_title": "ATOM main.py"},
        )
        pb_samples.append((tag, (time.perf_counter() - t0) * 1000))
    results["structured_prompt_build_ms"] = {
        k: round(v, 3) for k, v in pb_samples
    }

    # 7) Local LLM (if available) — use longer timeout for benchmark stability
    from brain.mini_llm import MiniLLM

    _bench_cfg = json.loads(json.dumps(config))
    _bench_cfg["brain"] = {**config.get("brain", {}), "timeout_seconds": 90}
    mini = MiniLLM(_bench_cfg)
    llama_prompts = []
    if mini.is_available():
        system = (
            "You are ATOM, Satyam's personal AI OS. Address him as Boss. "
            "Be warm, concise (2-4 sentences unless asked for detail)."
        )
        for tag, user_q in [
            ("work_daily", "Boss needs to prep for a standup in 10 minutes. What should he jot down?"),
            ("buddy", "Boss says he's lonely working from home tonight. Respond like a close friend who cares."),
            ("learning", "Boss is learning async/await in Python. Explain it with one tiny code example."),
            ("learning_hard", "Compare gradient descent and stochastic gradient descent for someone new to ML."),
        ]:
            prompt = (
                f"<|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n{user_q}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n"
            )
            llama_prompts.append((tag, prompt))

        async def _run():
            mini.preload()
            return await bench_llm(mini, llama_prompts)

        llm_rows = asyncio.run(_run())
        results["local_llm_inference"] = llm_rows
        mini.shutdown()
    else:
        results["local_llm_inference"] = []
        results["local_llm_note"] = "Model missing or llama_cpp not installed — skipped."

    # 8) Edge TTS: time to produce first audio chunk (network; optional)
    try:
        import edge_tts

        async def _tts_first_chunk():
            text = "Boss, quick status check. All systems nominal."
            communicate = edge_tts.Communicate(text, "en-GB-RyanNeural")
            t0 = time.perf_counter()
            first = True
            async for _chunk in communicate.stream():
                if first:
                    return (time.perf_counter() - t0) * 1000
            return None

        tts_ms = asyncio.run(_tts_first_chunk())
        results["edge_tts_first_chunk_ms"] = round(tts_ms, 1) if tts_ms else None
    except Exception as e:
        results["edge_tts_first_chunk_ms"] = None
        results["edge_tts_note"] = str(e)[:120]

    results["stt_reference"] = {
        "engine": "faster-whisper",
        "model_size": config.get("stt", {}).get("whisper_model_size", "small"),
        "typical_decode_ms": "300-500 (CPU int8), 100-200 (GPU float16)",
        "note": "Full mic→speech_final timing requires live run; see logs PIPELINE lines.",
    }

    # 10) Synthetic end-to-end estimate (local path with LLM)
    if results.get("local_llm_inference"):
        avg_llm = statistics.mean(r["elapsed_ms"] for r in results["local_llm_inference"])
        intent_avg = results["intent_classification"]["avg_ms"]
        tts_ms = results.get("edge_tts_first_chunk_ms") or 400.0
        stt_mid = 150.0
        prompt_avg = statistics.mean(results["structured_prompt_build_ms"].values())
        results["estimated_voice_round_trip_ms"] = {
            "stt_decode_typical_mid_ms": stt_mid,
            "intent_plus_router_typical_ms": intent_avg,
            "structured_prompt_build_ms": prompt_avg,
            "local_llm_avg_ms_this_run": round(avg_llm, 1),
            "edge_tts_first_chunk_ms": tts_ms,
            "sum_typical_llm_path_ms": round(
                stt_mid + intent_avg + prompt_avg + avg_llm + tts_ms, 1,
            ),
            "disclaimer": "Sum is indicative; parallel work and caching change real runs.",
        }

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
