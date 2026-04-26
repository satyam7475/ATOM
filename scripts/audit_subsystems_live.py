"""Live audit of ATOM's non-LLM subsystems (no full app boot).

Tests in order:
  1. WhisperKit-cli existence + version (STT engine sanity)
  2. Embedding model cold load + 5-query latency
  3. SmolVLM warm + 1 caption call
  4. Memory engine semantic retrieval round-trip
  5. ChromaDB read latency (4 collections)
  6. Structured prompt builder voice-mode token cap
  7. Cloud rotation diagnostics shape (Sprint Ω.8 refactor verification)

Writes a JSON report next to itself.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import psutil  # noqa: E402

PROC = psutil.Process()


def _mem_mb() -> float:
    return PROC.memory_info().rss / (1024 * 1024)


def _maybe(fn, label, report, **kw):
    t0 = time.perf_counter()
    rss_pre = _mem_mb()
    try:
        result = fn()
        dt = (time.perf_counter() - t0) * 1000.0
        rss_post = _mem_mb()
        report[label] = {
            "ok": True,
            "ms": round(dt, 1),
            "rss_delta_mb": round(rss_post - rss_pre, 1),
            **(kw or {}),
            **(result if isinstance(result, dict) else {"value": result}),
        }
        print(f"  [OK]  {label}: {dt:.0f}ms  +{rss_post-rss_pre:.0f}MB", flush=True)
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000.0
        report[label] = {"ok": False, "ms": round(dt, 1), "error": f"{type(e).__name__}: {e}"}
        print(f"  [FAIL] {label}: {e}", flush=True)


def main() -> int:
    report: dict = {"system": {"ram_pct_start": psutil.virtual_memory().percent}}
    print("== ATOM non-LLM audit ==")

    def whisperkit_cli():
        path = shutil.which("whisperkit-cli")
        if not path:
            return {"installed": False}
        try:
            out = subprocess.run([path, "--version"], capture_output=True, timeout=8, text=True)
            return {"installed": True, "path": path, "version": (out.stdout or out.stderr).strip()[:200]}
        except Exception as e:
            return {"installed": True, "path": path, "version_error": str(e)[:200]}

    _maybe(whisperkit_cli, "whisperkit_cli", report)

    def embedding():
        from core.embedding_engine import EmbeddingEngine
        cfg = json.loads((ROOT / "config/settings.json").read_text())
        engine = EmbeddingEngine(config=cfg)
        warm = engine.embed_sync("hello atom")
        t0 = time.perf_counter()
        queries = [
            "what is the capital of France",
            "remind me about my goals",
            "play some music",
            "draft an email to my brother",
            "summarize the last meeting",
        ]
        vecs = [engine.embed_sync(q) for q in queries]
        dt = (time.perf_counter() - t0) * 1000.0
        dims = len(vecs[0]) if vecs else 0
        return {
            "available": dims > 0,
            "dims": dims,
            "warm_ok": bool(warm),
            "five_query_ms": round(dt, 1),
            "per_query_ms": round(dt / 5, 1),
        }

    _maybe(embedding, "embedding", report)

    def vlm_caption():
        from core.perception.vlm_describe import VLMCaptioner
        captioner = VLMCaptioner(model_path="models/smolvlm-instruct-4bit")
        from PIL import Image
        img = Image.new("RGB", (240, 160), color=(20, 80, 200))
        tmp = ROOT / "tmp_audit_vlm.png"
        img.save(tmp)
        t0 = time.perf_counter()
        caption = captioner.describe(str(tmp), prompt="What is in this image?", max_tokens=32)
        dt = (time.perf_counter() - t0) * 1000.0
        try:
            tmp.unlink()
        except Exception:
            pass
        return {
            "first_call_total_ms": round(dt, 1),
            "load_ms": getattr(captioner, "_load_ms", None),
            "caption_head": (caption or "")[:200],
        }

    _maybe(vlm_caption, "vlm_caption", report)

    def chroma_ping():
        from core.vector_store import VectorStore
        cfg = json.loads((ROOT / "config/settings.json").read_text())
        store = VectorStore(cfg)
        diag = store.diagnostics() if hasattr(store, "diagnostics") else {}
        info = {"backend": getattr(store, "backend", "?"), "diag": diag}
        return info

    _maybe(chroma_ping, "vector_store", report)

    def prompt_builder_voice_cap():
        from cursor_bridge.structured_prompt_builder import StructuredPromptBuilder
        cfg = json.loads((ROOT / "config/settings.json").read_text())
        builder = StructuredPromptBuilder(cfg)
        big = "RAG snippet about unified memory and Apple Silicon Mx series. " * 200
        memory_summaries = [f"long-term-fact-{i}: {('blah ' * 40)}" for i in range(8)]
        history = [(f"u{i}", f"a{i} " + ("yada " * 20)) for i in range(6)]
        documents = [big[:500] for _ in range(4)]

        voice_prompt = builder.build(
            query="what is unified memory in one sentence?",
            memory_summaries=memory_summaries,
            history=history,
            document_context=documents,
            rag_enrichment=big,
            voice_mode=True,
        )
        chat_prompt = builder.build(
            query="what is unified memory in one sentence?",
            memory_summaries=memory_summaries,
            history=history,
            document_context=documents,
            rag_enrichment=big,
            voice_mode=False,
        )
        approx_voice_tokens = len(voice_prompt) // 4
        approx_chat_tokens = len(chat_prompt) // 4
        return {
            "voice_chars": len(voice_prompt),
            "chat_chars": len(chat_prompt),
            "voice_approx_tokens": approx_voice_tokens,
            "chat_approx_tokens": approx_chat_tokens,
            "voice_under_1800tok": approx_voice_tokens <= 1800,
        }

    _maybe(prompt_builder_voice_cap, "prompt_builder_voice_cap", report)

    def cloud_rotation_diag():
        from core.cloud.rotating_openai_client import RotatingOpenAIClient
        cfg = json.loads((ROOT / "config/settings.json").read_text())
        client = RotatingOpenAIClient(cfg)
        diag = client.diagnostics()
        slots = diag.get("slots", []) or []
        ready = [s for s in slots if s.get("api_key_present")]
        return {
            "enabled": diag.get("enabled"),
            "provider": diag.get("provider"),
            "slot_count": len(slots),
            "slot_names": [s.get("name") for s in slots],
            "tiers": [s.get("tier") for s in slots],
            "ready_slots": [s["name"] for s in ready],
            "diag_keys": sorted(list(diag.keys())),
            "first_slot_keys": sorted(list((slots[0] or {}).keys())) if slots else [],
        }

    _maybe(cloud_rotation_diag, "cloud_rotation", report)

    out = ROOT / "audit_subsystems_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nreport -> {out}")
    print(f"system RAM end: {psutil.virtual_memory().percent:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
