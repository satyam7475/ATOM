#!/usr/bin/env python3
"""
ATOM — Jarvis-level manual benchmark (≥30 scenarios).

Runs from repo root (cwd must be ATOM/). Measures:
  - CognitiveKernel.route() latency and ExecPath
  - Production-like answers: DIRECT/cache/kernel skip_llm vs LocalBrainController
  - Prompt-layer audit: unwanted [WORLD]/weather leakage on small-talk (static)

Env:
  ATOM_BENCH_DELAY_SEC — pause between cases (default 2.0)
  ATOM_BENCH_MAX_CASES — limit scenarios for quick smoke (default all)
  ATOM_BENCH_START_INDEX — 0-based offset into SCENARIOS (default 0)
  ATOM_BENCH_SKIP_LLM — if 1, never call LocalBrain (route + prompt audit only)

Writes:
  logs/jarvis_benchmark_<timestamp>.json
  logs/JARVIS_BENCHMARK_REPORT.md
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("atom.bench")


@dataclass
class Scenario:
    id: int
    category: str
    text: str
    expect_no_world_block: bool  # prompt should not inject fused world for this


SCENARIOS: list[Scenario] = [
    Scenario(1, "small_talk", "How are you?", True),
    Scenario(2, "small_talk", "Are you there?", True),
    Scenario(3, "small_talk", "Hello ATOM", True),
    Scenario(4, "small_talk", "Good morning", True),
    Scenario(5, "small_talk", "Thanks buddy", True),
    Scenario(6, "media", "Play Tera Hua on YouTube", False),
    Scenario(7, "media", "Open YouTube and search for lo-fi music", False),
    Scenario(8, "time", "What time is it?", False),
    Scenario(9, "time", "What is the date today?", False),
    Scenario(10, "world", "What's the weather in New Delhi?", False),
    Scenario(11, "world", "What season is it in Tokyo right now?", False),
    Scenario(12, "reasoning", "What is 15 times 7?", False),
    Scenario(13, "reasoning", "What is 2 plus 2?", False),
    Scenario(14, "chit_chat", "Tell me a very short joke", True),
    Scenario(15, "identity", "Are you smart?", True),
    Scenario(16, "identity", "Are you like Jarvis?", True),
    Scenario(17, "system", "What is my battery level?", False),
    Scenario(18, "system", "Check CPU usage", False),
    Scenario(19, "system", "Give me a quick system diagnostic", False),
    Scenario(20, "perception", "Describe what is on my screen", False),
    Scenario(21, "control", "Set volume to 40 percent", False),
    Scenario(22, "control", "Mute the audio", False),
    Scenario(23, "meta", "Stop listening", True),
    Scenario(24, "meta", "Cancel that", True),
    Scenario(25, "reasoning", "Explain quantum entanglement in one sentence", False),
    Scenario(26, "reasoning", "Write a minimal Python hello world", False),
    Scenario(27, "world", "Current time in London?", False),
    Scenario(28, "small_talk", "Goodnight", True),
    Scenario(29, "edge", "   ", True),
    Scenario(30, "long", "Test " * 25, False),
    Scenario(31, "small_talk", "Hey, you okay?", True),
    Scenario(32, "media", "Search YouTube for Tera Hua song", False),
]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _prompt_audit(build_text: str, scenario: Scenario) -> dict[str, object]:
    """Heuristic: small-talk should not carry heavy [WORLD] / location blocks."""
    t = build_text or ""
    has_world = "[WORLD]" in t or "[WEATHER]" in t or "[LOCATION]" in t
    has_delhi = "delhi" in t.lower() or "new delhi" in t.lower()
    has_spring = "spring" in t.lower()
    leak = False
    if scenario.expect_no_world_block and (has_world or (has_delhi and has_spring)):
        leak = True
    return {
        "prompt_chars": len(t),
        "has_world_markers": has_world,
        "suspected_smalltalk_leak": leak,
    }


def _response_heuristic(query: str, response: str) -> dict[str, object]:
    """Catch the old failure mode: answering everything with time + Delhi + season."""
    q = (query or "").lower()
    r = (response or "").lower()
    issues: list[str] = []
    small = any(
        x in q
        for x in (
            "how are you",
            "are you there",
            "hello",
            "good morning",
            "thanks",
            "goodnight",
            "you okay",
        )
    )
    if small and "new delhi" in r and "spring" in r:
        issues.append("delhi_spring_on_smalltalk")
    if small and r.count("current time") >= 1 and "?" not in q:
        issues.append("time_recital_on_smalltalk")
    return {"issues": issues, "response_chars": len(response or "")}


async def _run() -> int:
    delay = _env_float("ATOM_BENCH_DELAY_SEC", 2.0)
    max_cases = _env_int("ATOM_BENCH_MAX_CASES", len(SCENARIOS))
    start_idx = max(0, _env_int("ATOM_BENCH_START_INDEX", 0))
    skip_llm = os.environ.get("ATOM_BENCH_SKIP_LLM", "").strip() in ("1", "true", "yes")
    selected = SCENARIOS[start_idx : start_idx + max_cases]
    if not selected:
        logger.error("ATOM_BENCH_START_INDEX out of range")
        return 2

    from core.boot.config_loader import load_config
    from core.cognitive_kernel import CognitiveKernel
    from core.security_gateway import SecurityGateway
    from core.confidence_engine import ConfidenceEngine
    from core.decision_engine import DecisionEngine
    from core.tools.search_tool import SearchTool
    from core.memory.preference_store import PreferenceStore
    from core.semantic_cache import SemanticCache
    from cursor_bridge.structured_prompt_builder import StructuredPromptBuilder
    from core.async_event_bus import AsyncEventBus
    from cursor_bridge.local_brain_controller import LocalBrainController

    config = load_config()
    cloud_on = bool(config.get("cloud", {}).get("enabled", True))

    kernel = CognitiveKernel(config=config)
    security_gateway = SecurityGateway(config)
    confidence_engine = ConfidenceEngine(config)
    decision_engine = DecisionEngine(config)
    semantic_cache = SemanticCache(config)
    preference_store = PreferenceStore(config)

    gemini_client = None
    if cloud_on:
        from core.cloud.gemini_client import GeminiClient
        from core.secrets_manager import get_gemini_fast_key

        gemini_client = GeminiClient(config, security_gateway=security_gateway)
        k = get_gemini_fast_key()
        if k:
            gemini_client.configure_api_key(k)

    search_tool = SearchTool(
        config, security_gateway=security_gateway, gemini_client=gemini_client
    )
    kernel.attach_cloud_intelligence(
        confidence_engine=confidence_engine,
        search_tool=search_tool,
        gemini_client=gemini_client,
        semantic_cache=semantic_cache,
    )

    prompt_builder = StructuredPromptBuilder(config)
    prompt_builder.set_preference_store(preference_store)

    bus = AsyncEventBus()
    bus.start()

    local_brain: LocalBrainController | None = None
    if not skip_llm:
        local_brain = LocalBrainController(bus, prompt_builder, config)
        local_brain.attach_cloud_intelligence(
            confidence_engine=confidence_engine,
            decision_engine=decision_engine,
            gemini_client=gemini_client,
            semantic_cache=semantic_cache,
            preference_store=preference_store,
        )

    mlx_note = ""
    if local_brain is not None and hasattr(local_brain, "_llm"):
        mlx_note = (
            "mlx_installed"
            if local_brain._llm.is_available()
            else "mlx_or_model_dir_missing"
        )

    results: list[dict[str, object]] = []
    t_session = time.perf_counter()

    for sc in selected:
        text = sc.text.strip()
        if not text:
            text = " "

        # --- Route timing ---
        t0 = time.perf_counter()
        plan = kernel.route(text)
        route_ms = (time.perf_counter() - t0) * 1000.0

        # --- Prompt build (always; cheap) ---
        try:
            built = prompt_builder.build(text)
        except Exception as e:
            built = f"[prompt build error: {e}]"

        audit = _prompt_audit(built, sc)

        row: dict[str, object] = {
            "id": sc.id,
            "category": sc.category,
            "query": sc.text,
            "route_ms": round(route_ms, 3),
            "exec_path": str(getattr(plan, "path", "")),
            "skip_llm": bool(getattr(plan, "skip_llm", False)),
            "route_reason": str(getattr(plan, "reason", "") or ""),
            "prompt_audit": audit,
        }

        direct = getattr(plan, "direct_response", None)
        if getattr(plan, "skip_llm", False) and direct:
            row["source"] = "kernel_direct"
            row["response_preview"] = (direct or "")[:600]
            row["llm_ms"] = 0.0
            row["response_heuristic"] = _response_heuristic(sc.text, str(direct))
        elif skip_llm or local_brain is None:
            row["source"] = "skipped_llm"
            row["response_preview"] = ""
            row["llm_ms"] = 0.0
        else:
            captures: list[tuple[str, str]] = []

            async def _on_cursor(**kw: object) -> None:
                r = kw.get("response")
                if isinstance(r, str) and r:
                    captures.append(("cursor", r))

            async def _on_ready(**kw: object) -> None:
                t = kw.get("text")
                if isinstance(t, str) and t:
                    captures.append(("ready", t))

            bus.on("cursor_response", _on_cursor)
            bus.on("response_ready", _on_ready)

            t_llm = time.perf_counter()
            await local_brain.on_query(text, query_plan=plan)
            await asyncio.sleep(0.85)
            llm_ms = (time.perf_counter() - t_llm) * 1000.0

            resp = ""
            if captures:
                resp = captures[-1][1]
            row["source"] = "local_brain"
            row["llm_ms"] = round(llm_ms, 2)
            row["response_preview"] = resp[:600]
            row["response_heuristic"] = _response_heuristic(sc.text, resp)

        results.append(row)
        await asyncio.sleep(delay)

    total_wall_s = time.perf_counter() - t_session

    # Aggregate
    prompt_leaks = sum(
        1
        for r in results
        if (r.get("prompt_audit") or {}).get("suspected_smalltalk_leak")
    )
    heur_fails = sum(
        1
        for r in results
        if (r.get("response_heuristic") or {}).get("issues")
    )
    direct_hits = sum(1 for r in results if r.get("source") == "kernel_direct")
    llm_calls = sum(1 for r in results if r.get("source") == "local_brain")

    report_md = _render_report(
        config_cloud=cloud_on,
        delay_sec=delay,
        max_cases=len(selected),
        start_idx=start_idx,
        skip_llm=skip_llm,
        mlx_note=mlx_note,
        total_wall_s=total_wall_s,
        results=results,
        prompt_leaks=prompt_leaks,
        heur_fails=heur_fails,
        direct_hits=direct_hits,
        llm_calls=llm_calls,
    )

    logs_dir = _REPO_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = logs_dir / f"jarvis_benchmark_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": ts,
                "delay_sec": delay,
                "max_cases": len(selected),
                "start_index": start_idx,
                "skip_llm": skip_llm,
                "cloud_enabled": cloud_on,
                "mlx": mlx_note,
                "total_wall_seconds": round(total_wall_s, 2),
                "summary": {
                    "prompt_smalltalk_leak_flags": prompt_leaks,
                    "response_heuristic_issue_rows": heur_fails,
                    "kernel_direct_responses": direct_hits,
                    "local_brain_calls": llm_calls,
                },
                "results": results,
            },
            f,
            indent=2,
        )

    report_path = logs_dir / "JARVIS_BENCHMARK_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(report_md)
    print(f"\nJSON: {json_path}")
    print(f"Report: {report_path}")
    return 0 if prompt_leaks == 0 and heur_fails == 0 else 1


def _render_report(
    *,
    config_cloud: bool,
    delay_sec: float,
    max_cases: int,
    start_idx: int = 0,
    skip_llm: bool,
    mlx_note: str,
    total_wall_s: float,
    results: list[dict[str, object]],
    prompt_leaks: int,
    heur_fails: int,
    direct_hits: int,
    llm_calls: int,
) -> str:
    lines: list[str] = []
    lines.append("# ATOM Jarvis benchmark report\n")
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Config `cloud.enabled`: **{config_cloud}**")
    lines.append(f"- Delay between cases: **{delay_sec}s** (set `ATOM_BENCH_DELAY_SEC`)")
    lines.append(
        f"- Cases run: **{max_cases}** starting at index **{start_idx}** "
        f"(`ATOM_BENCH_MAX_CASES` / `ATOM_BENCH_START_INDEX`)"
    )
    lines.append(f"- Skip LLM path: **{skip_llm}** (`ATOM_BENCH_SKIP_LLM`)")
    lines.append(f"- MLX / model availability note: **{mlx_note or 'n/a'}**")
    lines.append(f"- Total wall time (script): **{total_wall_s:.1f}s**")
    lines.append("")
    lines.append("## Summary counts")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"| --- | --- |")
    lines.append(f"| Kernel direct (skip_llm) answers | {direct_hits} |")
    lines.append(f"| Local brain invocations | {llm_calls} |")
    lines.append(f"| Prompt small-talk leak flags | {prompt_leaks} |")
    lines.append(f"| Response heuristic issue rows | {heur_fails} |")
    lines.append("")
    lines.append("## Time & throughput (planning)")
    lines.append("")
    lines.append(
        f"- **Between-question delay budget**: ~{max_cases * delay_sec:.0f}s "
        f"({max_cases} × {delay_sec}s)."
    )
    lines.append(
        "- **Local LLM inference (Apple Silicon MLX)**: typically **2–15s** per query "
        f"({llm_calls} calls in this run); first load can add **10–60s** once."
    )
    lines.append(
        "- **Cloud (Gemini)**: **disabled** in your current settings — when enabled, "
        "add **0.5–3s** per cloud call + quota/network variance."
    )
    lines.append(
        f"- **Rough total for interactive 32-case session**: delay + MLX; "
        f"this run spent **{total_wall_s:.1f}s** wall time."
    )
    lines.append("")
    lines.append("## Bottlenecks vs “Jarvis level”")
    lines.append("")
    lines.append(
        "1. **Local MLX generation** — largest variable; 4B+ models on-device dominate "
        "latency vs cloud APIs."
    )
    lines.append(
        "2. **Prompt + context assembly** — Layer 3b fusion, RAG, and history grow "
        "`n_ctx` work; long prompts slow prefill."
    )
    lines.append(
        "3. **ReAct / tools** — each tool round-trip adds one more LLM pass "
        "(up to `MAX_REACT_STEPS`)."
    )
    lines.append(
        "4. **Voice stack (not in this script)** — STT + VAD + TTS streaming often "
        "match or exceed LLM time in real use."
    )
    lines.append(
        "5. **Cloud off** — no Gemini fallback; hard reasoning and fresh web facts "
        "depend on local model quality or search tools."
    )
    lines.append("")
    lines.append("## Per-case results")
    lines.append("")
    lines.append(
        "| # | Category | Path | Route ms | Source | Prompt leak? | Heuristic issues | Preview |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in results:
        pa = r.get("prompt_audit") or {}
        rh = r.get("response_heuristic") or {}
        issues = rh.get("issues") or []
        prev = (r.get("response_preview") or "").replace("\n", " ")[:80]
        lines.append(
            f"| {r.get('id')} | {r.get('category')} | `{r.get('exec_path')}` | "
            f"{r.get('route_ms')} | {r.get('source')} | "
            f"{pa.get('suspected_smalltalk_leak')} | {issues} | {prev} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
