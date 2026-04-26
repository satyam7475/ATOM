"""Voice-pipeline latency audit.

Splits the user-perceived 'speak then listen' delay into its real
components so we can see exactly where the 9/10 gap lives.

Phase 1 — local brain (4B-4bit) on the steady, persona-pinned hot path:
  • cold prefill ms (turn 1, freshly loaded)
  • warm prefill ms (turn 2+, persona pinned, prompt cache warm)
  • first-token ms (streaming) on a typical voice query
  • decode tokens/sec on a 32-token target
  • long prefill ms (~1800 token voice-cap stress)

Phase 2 — soft path round-trip:
  • intent classify
  • quick reply lookup
  • prompt builder
  • semantic cache (semantic + exact_only)
  • vector store search
  • memory engine retrieve

Writes audit_voice_pipeline_report.json.
"""

from __future__ import annotations

import asyncio
import gc
import json
import os
import statistics
import sys
import time
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import psutil  # noqa: E402

PROC = psutil.Process()


def _rss_mb() -> float:
    return PROC.memory_info().rss / (1024 * 1024)


async def _phase_brain(report: dict, cfg: dict) -> None:
    print("\n== Phase 1: local brain (4B-4bit) ==")
    brain_cfg = cfg["brain"]
    mlx_path = brain_cfg.get("mlx_model")

    from brain.mlx_llm import MLXBrain

    cfg_min = {
        "brain": {
            "mlx_model": mlx_path,
            "single_resident": True,
            "max_tokens": 64,
            "temperature": 0.7,
            "n_ctx": brain_cfg.get("n_ctx", 6144),
            "prompt_cache_enabled": True,
            "prompt_cache_persist": True,
        },
    }
    brain = MLXBrain(cfg_min)

    gc.collect()
    rss_pre = _rss_mb()
    t0 = time.perf_counter()
    ok = brain.preload(load_all=False)
    cold_ms = (time.perf_counter() - t0) * 1000.0
    rss_post = _rss_mb()
    print(f"  load: {cold_ms:.0f}ms ok={ok} rss +{rss_post - rss_pre:.0f}MB")
    report["brain"] = {
        "load_ms": round(cold_ms, 1),
        "rss_load_mb": round(rss_post - rss_pre, 1),
    }

    if not ok:
        report["brain"]["error"] = "load_failed"
        return

    persona = (
        "<|im_start|>system\n"
        "You are ATOM, a personal AI assistant created by Satyam Yadav. "
        "You call him 'Boss'. Friendly, witty, concise. Keep voice replies "
        "to 1-2 sentences unless asked for detail.\n"
        "<|im_end|>\n"
    )

    voice_qs = [
        "what time is it",
        "how are you",
        "what is unified memory in one sentence",
        "tell me a quick fact about Mars",
        "remind me to call mom tomorrow at 9 am",
    ]

    # Streaming first-token probe on the FIRST query (turn 1 cold).
    print("  >> turn1 streaming (cold)")
    p1 = persona + (
        f"<|im_start|>user\n{voice_qs[0]}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    first_ms_list = []
    decode_wps_list = []
    full_ms_list = []

    for i, q in enumerate(voice_qs):
        prompt = persona + (
            f"<|im_start|>user\n{q}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        t0 = time.perf_counter()
        first_t = None
        chunks: list[str] = []
        # MLXBrain.generate_stream may exist; fall back to generate
        gen_stream = getattr(brain, "generate_stream", None)
        if callable(gen_stream):
            try:
                async for ch in gen_stream(prompt, max_tokens_override=24):
                    if first_t is None:
                        first_t = time.perf_counter()
                    chunks.append(str(ch))
            except TypeError:
                # Some impls return a sync iterator
                it = gen_stream(prompt, max_tokens_override=24)
                for ch in it:
                    if first_t is None:
                        first_t = time.perf_counter()
                    chunks.append(str(ch))
            text = "".join(chunks).strip()
            ok_g = True
        else:
            t_gen0 = time.perf_counter()
            text, ok_g = await brain.generate(prompt, max_tokens_override=24)
            first_t = time.perf_counter()  # whole answer arrives at once
            ok_g = bool(ok_g)
        end_t = time.perf_counter()
        full_ms = (end_t - t0) * 1000.0
        first_ms = (first_t - t0) * 1000.0 if first_t else full_ms
        words = max(1, len((text or "").split()))
        decode_window_ms = max(1.0, (end_t - first_t) * 1000.0) if first_t else full_ms
        wps = words * 1000.0 / decode_window_ms

        first_ms_list.append(first_ms)
        decode_wps_list.append(wps)
        full_ms_list.append(full_ms)

        print(
            f"  q{i+1}: first_token={first_ms:.0f}ms full={full_ms:.0f}ms "
            f"words={words} decode={wps:.1f} wps -> {(text or '')[:80]!r}"
        )

    report["brain"]["first_token_cold_ms"] = round(first_ms_list[0], 1)
    if len(first_ms_list) > 1:
        warm = first_ms_list[1:]
        report["brain"]["first_token_warm_median_ms"] = round(
            statistics.median(warm), 1,
        )
        report["brain"]["first_token_warm_min_ms"] = round(min(warm), 1)
        report["brain"]["first_token_warm_max_ms"] = round(max(warm), 1)
    report["brain"]["full_warm_median_ms"] = round(
        statistics.median(full_ms_list[1:] or full_ms_list), 1,
    )
    report["brain"]["decode_wps_median"] = round(
        statistics.median(decode_wps_list), 2,
    )

    # Long prefill stress (~1800 tokens).
    bulk = (" ".join(["status"] * 220)).strip()
    bulk_prompt = persona + (
        f"<|im_start|>user\nIgnore: {bulk}\nIn one sentence, what is 2+2?"
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    t0 = time.perf_counter()
    out, ok_g = await brain.generate(bulk_prompt, max_tokens_override=24)
    long_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  long_prefill: {long_ms:.0f}ms -> {(out or '')[:80]!r}")
    report["brain"]["long_prefill_ms"] = round(long_ms, 1)

    report["brain"]["prompt_cache_hits"] = brain._prompt_cache_hits
    report["brain"]["prompt_cache_misses"] = brain._prompt_cache_misses
    report["brain"]["rss_final_mb"] = round(_rss_mb(), 1)


async def _phase_soft(report: dict, cfg: dict) -> None:
    print("\n== Phase 2: soft path ==")

    # 1. Intent
    from core.intent_engine import IntentEngine
    ie = IntentEngine()
    phrases = [
        "what time is it", "play some music", "remind me about goals",
        "open chrome", "tell me about quantum physics",
        "set a timer for 5 minutes", "draft an email to my brother",
        "show me my reminders", "take a screenshot",
        "what is on my screen", "close all apps", "volume up",
    ]
    ie.classify_silent(phrases[0])
    ts = []
    for p in phrases:
        t0 = time.perf_counter()
        ie.classify_silent(p)
        ts.append((time.perf_counter() - t0) * 1000)
    print(
        f"  intent: median={statistics.median(ts):.2f}ms "
        f"max={max(ts):.2f}ms (n={len(ts)})"
    )
    report["intent"] = {
        "median_ms": round(statistics.median(ts), 3),
        "max_ms": round(max(ts), 3),
    }

    # 2. Quick replies
    from core.quick_replies import try_quick_reply
    qr_phrases = [
        "hi", "hello", "thanks", "what is your name",
        "good morning", "bye", "how are you",
    ]
    for p in qr_phrases:
        try_quick_reply(p, cfg)
    qts = []
    for p in qr_phrases:
        t0 = time.perf_counter()
        try_quick_reply(p, cfg)
        qts.append((time.perf_counter() - t0) * 1000)
    print(
        f"  quick_reply: median={statistics.median(qts):.3f}ms "
        f"max={max(qts):.3f}ms"
    )
    report["quick_reply"] = {
        "median_ms": round(statistics.median(qts), 4),
        "max_ms": round(max(qts), 4),
    }

    # 3. Prompt builder voice-mode
    from cursor_bridge.structured_prompt_builder import StructuredPromptBuilder
    pb = StructuredPromptBuilder(cfg)
    mem = [f"mem-{i}: " + ("blah " * 30) for i in range(8)]
    hist = [(f"u{i}", f"a{i} " + ("yada " * 15)) for i in range(6)]
    docs = [("rag chunk " * 40)[:500] for _ in range(4)]
    pb.build(
        query="hello", memory_summaries=mem, history=hist,
        document_context=docs, voice_mode=True,
    )
    pbts = []
    last_p = ""
    for q in [
        "what time is it", "play some music", "draft email to brother",
        "tell me about quantum physics", "remind me to call mom",
    ]:
        t0 = time.perf_counter()
        last_p = pb.build(
            query=q, memory_summaries=mem, history=hist,
            document_context=docs, voice_mode=True,
        )
        pbts.append((time.perf_counter() - t0) * 1000)
    print(
        f"  prompt_builder voice: median={statistics.median(pbts):.2f}ms "
        f"approx_tokens={len(last_p)//4}"
    )
    report["prompt_builder"] = {
        "median_ms": round(statistics.median(pbts), 3),
        "max_ms": round(max(pbts), 3),
        "voice_approx_tokens": len(last_p) // 4,
    }

    # 4. Embedding (post-Ω.9 warmup)
    from core.embedding_engine import get_embedding_engine
    emb = get_embedding_engine(cfg)
    emb.embed_sync("warmup")
    emb.embed_sync("a longer phrase that has more tokens to encode")
    em_ts = []
    em_phrases = [
        "what time is it",
        "play some music",
        "what is the difference between optimism and realism",
        "briefly summarize the meeting and highlight action items I owe",
        "how are you", "tell me about quantum physics today",
        "hello", "tomorrow at 5 pm",
    ]
    for q in em_phrases:
        t0 = time.perf_counter()
        emb.embed_sync(q)
        em_ts.append((time.perf_counter() - t0) * 1000)
    print(
        f"  embedding: median={statistics.median(em_ts):.2f}ms "
        f"max={max(em_ts):.2f}ms (n={len(em_ts)})"
    )
    report["embedding"] = {
        "median_ms": round(statistics.median(em_ts), 3),
        "max_ms": round(max(em_ts), 3),
    }

    # 5. Vector store
    from core.vector_store import VectorStore
    vs = VectorStore(cfg)
    coll_counts: dict[str, int] = {}
    for c, coll in (vs._collections or {}).items():
        try:
            coll_counts[c] = coll.count()
        except Exception:
            coll_counts[c] = -1
    chosen = max(coll_counts, key=lambda c: coll_counts[c]) if coll_counts else None
    if chosen and coll_counts[chosen] > 0:
        qv = emb.embed_sync("what time is it")
        vs.search(chosen, qv, k=5, min_score=0.0)
        vts = []
        for q in ["what time", "play music", "goals", "physics", "draft email"]:
            qv = emb.embed_sync(q)
            t0 = time.perf_counter()
            vs.search(chosen, qv, k=5, min_score=0.0)
            vts.append((time.perf_counter() - t0) * 1000)
        print(
            f"  vector_store: median={statistics.median(vts):.2f}ms "
            f"max={max(vts):.2f}ms backend={vs._backend} "
            f"counts={coll_counts}"
        )
        report["vector_store"] = {
            "median_ms": round(statistics.median(vts), 3),
            "max_ms": round(max(vts), 3),
            "backend": vs._backend,
            "counts": coll_counts,
        }
    else:
        print(f"  vector_store: empty collections {coll_counts}")
        report["vector_store"] = {
            "backend": vs._backend, "counts": coll_counts,
        }

    # 6. Memory engine retrieve
    from core.memory_engine import MemoryEngine
    me = MemoryEngine(cfg)
    await me.retrieve("warmup query", k=5)
    mts = []
    for q in [
        "my goals", "recent work", "project status",
        "what i said earlier", "remind me about goals",
    ]:
        t0 = time.perf_counter()
        await me.retrieve(q, k=5)
        mts.append((time.perf_counter() - t0) * 1000)
    print(
        f"  memory_retrieve: median={statistics.median(mts):.2f}ms "
        f"max={max(mts):.2f}ms"
    )
    report["memory_retrieve"] = {
        "median_ms": round(statistics.median(mts), 3),
        "max_ms": round(max(mts), 3),
    }


async def _main() -> int:
    cfg = json.loads((ROOT / "config/settings.json").read_text())
    report: dict[str, Any] = {
        "system": {
            "python": sys.version.split()[0],
            "cpu_count": psutil.cpu_count(),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
            "ram_used_pct_start": psutil.virtual_memory().percent,
        },
        "config": {
            "mlx_model": cfg["brain"].get("mlx_model"),
            "n_ctx": cfg["brain"].get("n_ctx"),
        },
    }

    await _phase_soft(report, cfg)  # do soft first so brain has full RSS budget
    await _phase_brain(report, cfg)

    out = ROOT / "audit_voice_pipeline_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nreport -> {out}")
    print(f"system RAM end: {psutil.virtual_memory().percent:.1f}%")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(_main()))
    except KeyboardInterrupt:
        sys.exit(130)
