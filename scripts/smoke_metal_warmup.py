"""Live smoke for the cold-start Metal-warmup race.

Originally written to reproduce the live boot crash on 2026-04-24
(MLX command-buffer race) when the brain was Qwen2.5-7B-Instruct-MLX-4bit.
After the JARVIS-grade rewrite the brain is Qwen3-4B-Instruct-2507-4bit,
which the smoke now exercises end-to-end:
  1. Cold-start warmup with real MLX (Qwen3-4B-Instruct-2507-4bit) +
     real torch.mps SentenceTransformer. VLM warmup is now lazy by
     default (vision.vlm.warm_at_boot=false) so the smoke skips it
     unless the operator explicitly opts in via SMOKE_WARM_VLM=1.
  2. Two consecutive MLX inferences (the user's "second turn" was the
     SIGABRT trigger in the live log, not the first).

Exit code:
  0 -- cold start completed AND both inferences returned non-empty text.
  1 -- something else failed (printed inline).
  134 -- SIGABRT, the original Metal completion-handler race.

Usage:
  python scripts/smoke_metal_warmup.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

# Must be set BEFORE any HF tokenizer touches the process.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class _BusStub:
    def emit(self, *_a, **_kw) -> None:
        pass

    def emit_fast(self, *_a, **_kw) -> None:
        pass

    def emit_long(self, *_a, **_kw) -> None:
        pass

    def on(self, *_a, **_kw) -> None:
        pass


class _StateStub:
    class _S:
        value = "idle"

    current = _S()


class _SystemMonitorStub:
    def get_system_state(self) -> dict:
        return {
            "cpu_percent": 10.0,
            "ram_percent": 50.0,
            "foreground_window_title": "smoke",
            "ts": time.time(),
        }


async def _make_real_brain() -> object:
    """Real MLX brain wrapped to look like a LocalBrainController for
    the cold-start optimizer's warm_up contract.
    """
    from brain.mlx_llm import MLXBrain

    cfg = {
        "brain": {
            "model_paths": {
                "fast": "models/qwen3-4b-instruct-4bit",
                "primary": "models/qwen3-4b-instruct-4bit",
            },
            "max_tokens": 64,
            "temperature": 0.7,
        },
    }
    llm = MLXBrain(cfg)

    class _BrainShim:
        @property
        def available(self) -> bool:
            return llm.is_available()

        async def warm_up(
            self,
            *,
            model_role: str | None = None,
            load_all: bool = False,
        ) -> bool:
            loop = asyncio.get_running_loop()
            from functools import partial as _partial
            return await loop.run_in_executor(
                None,
                _partial(llm.preload, model_role=model_role, load_all=load_all),
            )

        async def generate(self, prompt: str) -> str:
            text, _ok = await llm.generate(prompt, max_tokens_override=32)
            return text

    return _BrainShim()


async def _main() -> int:
    print("== ATOM Metal warmup smoke ==")
    print(f"  cwd          : {os.getcwd()}")
    print(f"  models LLM   : {(ROOT / 'models/qwen3-4b-instruct-4bit').exists()}")
    print(f"  models VLM   : {(ROOT / 'models/smolvlm-instruct-4bit').exists()}")
    warm_vlm = os.environ.get("SMOKE_WARM_VLM", "0") in {"1", "true", "yes"}
    print(f"  warm_vlm     : {warm_vlm}")

    from core.boot.cold_start import ColdStartOptimizer
    from core.conversation_memory import ConversationMemory
    from core.memory_engine import MemoryEngine
    from core.perception.vlm_describe import VLMCaptioner

    brain = await _make_real_brain()
    memory = MemoryEngine(config={})
    captioner = VLMCaptioner(model_path="models/smolvlm-instruct-4bit")

    cold_start = ColdStartOptimizer(
        config={"vision": {"vlm": {"warm_at_boot": warm_vlm}}},
        bus=_BusStub(),
        state_manager=_StateStub(),
        local_brain=brain,
        memory_store=memory,
        conversation_memory=ConversationMemory(),
        intent_engine=None,
        system_monitor=_SystemMonitorStub(),
        vlm_captioner=captioner,
    )

    print("  >> warm_up() begin")
    t0 = time.monotonic()
    report = await cold_start.warm_up()
    elapsed = (time.monotonic() - t0) * 1000.0
    print(
        f"  >> warm_up() done in {elapsed:.0f}ms  "
        f"fast={report.fast_model_ready} emb={report.embeddings_ready} "
        f"vlm={report.vlm_ready} (vlm_ms={report.vlm_warmup_ms:.0f})"
    )

    if not report.fast_model_ready:
        print("  !! fast model didn't warm; cannot run inference smoke")
        return 1

    print("  >> inference #1 (mimics the boot greeting)")
    t1 = time.monotonic()
    out1 = await brain.generate(  # type: ignore[attr-defined]
        "You are ATOM. Reply with the single sentence: I'm here, Boss."
    )
    print(f"     {(time.monotonic()-t1)*1000:.0f}ms -> {out1!r}"[:200])

    print("  >> inference #2 (mimics the user's first question -- "
          "this is the turn that crashed in the live log)")
    t2 = time.monotonic()
    out2 = await brain.generate(  # type: ignore[attr-defined]
        "What two active goals does ATOM have right now? Reply in one short sentence."
    )
    print(f"     {(time.monotonic()-t2)*1000:.0f}ms -> {out2!r}"[:200])

    if not out1 or not out2:
        print("  !! one of the inferences returned empty text")
        return 1

    print("  == OK ==  cold start + 2 inferences completed without SIGABRT")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(_main()))
    except KeyboardInterrupt:
        sys.exit(130)
