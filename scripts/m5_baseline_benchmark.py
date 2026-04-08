"""
ATOM M5 Baseline Benchmark — Step 0.3

Measures latency, memory, and CPU for each module stage on macOS M5.
Outputs a structured report to stdout and saves to docs/ATOM_BASELINE_METRICS.md.

Usage: python scripts/m5_baseline_benchmark.py
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psutil

PROCESS = psutil.Process()


def mem_mb() -> float:
    return PROCESS.memory_info().rss / (1024 * 1024)


def cpu_pct() -> float:
    return PROCESS.cpu_percent(interval=0.1)


class BenchmarkResult:
    def __init__(self, name: str, elapsed_ms: float, mem_before: float, mem_after: float, error: str = ""):
        self.name = name
        self.elapsed_ms = elapsed_ms
        self.mem_before = mem_before
        self.mem_after = mem_after
        self.mem_delta = mem_after - mem_before
        self.error = error

    def __repr__(self):
        status = "FAIL" if self.error else "OK"
        return (f"  {self.name:<45} {self.elapsed_ms:>8.1f}ms  "
                f"mem: {self.mem_after:>7.1f}MB (+{self.mem_delta:>5.1f}MB)  [{status}]")


def bench(name: str, fn):
    gc.collect()
    mem_before = mem_mb()
    t0 = time.perf_counter()
    error = ""
    try:
        fn()
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    elapsed = (time.perf_counter() - t0) * 1000
    mem_after = mem_mb()
    return BenchmarkResult(name, elapsed, mem_before, mem_after, error)


def run_benchmarks():
    results: list[BenchmarkResult] = []
    system_info = {}

    print("=" * 80)
    print("  ATOM M5 BASELINE BENCHMARK")
    print("=" * 80)

    # ── System info ──
    mem = psutil.virtual_memory()
    system_info["platform"] = sys.platform
    system_info["python"] = sys.version.split()[0]
    system_info["cpu_count"] = psutil.cpu_count()
    system_info["ram_total_gb"] = round(mem.total / (1024**3), 1)
    system_info["ram_available_gb"] = round(mem.available / (1024**3), 1)
    battery = psutil.sensors_battery()
    if battery:
        system_info["battery_pct"] = battery.percent
        system_info["plugged"] = battery.power_plugged

    print(f"\n  Platform: {sys.platform} | Python: {system_info['python']}")
    print(f"  CPU cores: {system_info['cpu_count']} | RAM: {system_info['ram_available_gb']}GB free / {system_info['ram_total_gb']}GB")
    if battery:
        print(f"  Battery: {battery.percent}% | Plugged: {battery.power_plugged}")

    baseline_mem = mem_mb()
    print(f"\n  Baseline RSS: {baseline_mem:.1f} MB")
    print("-" * 80)

    # ── Stage 1: Config loading ──
    print("\n[STAGE 1] Config & Logging")
    config = {}

    r = bench("Config load (settings.json)", lambda: None)
    from core.boot.config_loader import load_config
    r = bench("load_config()", lambda: None)
    config = load_config()
    r = bench("Config load + parse", lambda: load_config())
    results.append(r)
    print(r)

    from core.logging_setup import setup_logging
    r = bench("setup_logging()", setup_logging)
    results.append(r)
    print(r)

    from core.config_schema import validate_and_log
    r = bench("validate_and_log(config)", lambda: validate_and_log(config))
    results.append(r)
    print(r)

    # ── Stage 2: Core module imports ──
    print("\n[STAGE 2] Core Module Imports")

    def import_core_modules():
        from core.state_manager import StateManager, AtomState  # noqa: F401
        from core.cache_engine import CacheEngine  # noqa: F401
        from core.memory_engine import MemoryEngine  # noqa: F401
        from core.intent_engine import IntentEngine  # noqa: F401
        from core.router import Router  # noqa: F401
        from context.context_engine import ContextEngine  # noqa: F401
        from core.metrics import MetricsCollector  # noqa: F401
        from core.command_registry import get_registry  # noqa: F401

    r = bench("Import 8 core modules", import_core_modules)
    results.append(r)
    print(r)

    # ── Stage 3: Core module initialization ──
    print("\n[STAGE 3] Core Module Initialization")

    from core.async_event_bus import AsyncEventBus
    from core.state_manager import StateManager
    from core.cache_engine import CacheEngine
    from core.memory_engine import MemoryEngine
    from core.intent_engine import IntentEngine
    from core.metrics import MetricsCollector
    from core.command_registry import get_registry

    bus = AsyncEventBus()
    state = StateManager(bus)
    metrics = MetricsCollector()

    r = bench("AsyncEventBus()", lambda: AsyncEventBus())
    results.append(r)
    print(r)

    r = bench("StateManager(bus)", lambda: StateManager(bus))
    results.append(r)
    print(r)

    r = bench("CacheEngine()", lambda: CacheEngine(max_size=128, ttl=300, metrics=metrics))
    results.append(r)
    print(r)

    r = bench("MemoryEngine(config)", lambda: MemoryEngine(config))
    results.append(r)
    print(r)

    intent_engine = IntentEngine()
    r = bench("IntentEngine()", lambda: IntentEngine())
    results.append(r)
    print(r)

    r = bench("CommandRegistry (get_registry)", lambda: get_registry())
    results.append(r)
    print(r)

    from context.context_engine import ContextEngine
    r = bench("ContextEngine(config)", lambda: ContextEngine(config))
    results.append(r)
    print(r)

    # ── Stage 4: Intent engine latency ──
    print("\n[STAGE 4] Intent Engine Latency (hot path)")

    test_queries = [
        "open chrome",
        "what time is it",
        "set volume to 50",
        "play some music",
        "take a screenshot",
        "tell me a joke",
        "what's the weather like",
        "search for python tutorials",
        "how does quantum computing work",
        "remind me to call mom at 5pm",
    ]

    intent_times = []
    for query in test_queries:
        t0 = time.perf_counter()
        result = intent_engine.classify(query)
        elapsed = (time.perf_counter() - t0) * 1000
        intent_times.append((query, elapsed, result))

    avg_intent = sum(t for _, t, _ in intent_times) / len(intent_times)
    max_intent = max(t for _, t, _ in intent_times)
    min_intent = min(t for _, t, _ in intent_times)

    r = BenchmarkResult(f"IntentEngine.match() avg ({len(test_queries)} queries)",
                        avg_intent, mem_mb(), mem_mb())
    results.append(r)
    print(f"  Intent engine: avg={avg_intent:.3f}ms  min={min_intent:.3f}ms  max={max_intent:.3f}ms")
    for query, elapsed, result in intent_times:
        match_str = result.intent if hasattr(result, 'intent') else str(result)
        print(f"    {query:<45} {elapsed:.3f}ms  → {match_str}")

    # ── Stage 5: Context engine ──
    print("\n[STAGE 5] Context Engine")

    ctx_engine = ContextEngine(config)
    t0 = time.perf_counter()
    ctx = ctx_engine.get_bundle()
    ctx_elapsed = (time.perf_counter() - t0) * 1000
    r = BenchmarkResult("ContextEngine.get_bundle()", ctx_elapsed, mem_mb(), mem_mb())
    results.append(r)
    print(r)
    print(f"    Context keys: {list(ctx.keys()) if isinstance(ctx, dict) else type(ctx)}")

    # ── Stage 6: Heavy modules ──
    print("\n[STAGE 6] Heavy Module Initialization")

    r = bench("SecurityPolicy(config)", lambda: __import__('core.security_policy', fromlist=['SecurityPolicy']).SecurityPolicy(config))
    results.append(r)
    print(r)

    r = bench("SecurityFortress(config)", lambda: __import__('core.security_fortress', fromlist=['SecurityFortress']).SecurityFortress(config))
    results.append(r)
    print(r)

    def init_code_introspector():
        from core.code_introspector import CodeIntrospector
        ci = CodeIntrospector()
        ci.scan()
        return ci

    r = bench("CodeIntrospector.scan()", init_code_introspector)
    results.append(r)
    print(r)

    def init_system_scanner():
        from core.system_scanner import SystemScanner
        ss = SystemScanner(bus, config)
        return ss

    r = bench("SystemScanner(bus, config)", init_system_scanner)
    results.append(r)
    print(r)

    def init_system_indexer():
        from core.system_indexer import system_indexer
        system_indexer.start()
        time.sleep(0.5)
        return system_indexer

    r = bench("SystemIndexer.start()", init_system_indexer)
    results.append(r)
    print(r)

    # ── Stage 7: Tool registry ──
    print("\n[STAGE 7] Tool & Reasoning")

    r = bench("ToolRegistry (get_tool_registry)", lambda: __import__('core.reasoning.tool_registry', fromlist=['get_tool_registry']).get_tool_registry())
    results.append(r)
    print(r)

    r = bench("ReasoningPlanner(config)", lambda: __import__('core.reasoning.planner', fromlist=['ReasoningPlanner']).ReasoningPlanner(config))
    results.append(r)
    print(r)

    # ── Stage 8: Cognitive layer ──
    print("\n[STAGE 8] Cognitive Layer")

    def init_cognitive():
        from core.cognitive.second_brain import SecondBrain
        from core.cognitive.goal_engine import GoalEngine
        from core.cognitive.behavior_model import BehaviorModel
        from core.cognitive.prediction_engine import PredictionEngine
        from core.cognitive.self_optimizer import SelfOptimizer
        from core.cognitive.dream_engine import DreamEngine
        from core.cognitive.curiosity_engine import CuriosityEngine
        from core.behavior_tracker import BehaviorTracker

        behavior = BehaviorTracker(config)
        memory = MemoryEngine(config)
        sb = SecondBrain(memory, behavior, config)
        GoalEngine(bus, sb, config)
        bm = BehaviorModel(bus, config)
        PredictionEngine(bus, behavior, memory, bm, config)
        SelfOptimizer(bus, metrics, config)
        DreamEngine(bus, config)
        CuriosityEngine(bus, config)

    r = bench("Cognitive layer (7 modules)", init_cognitive)
    results.append(r)
    print(r)

    # ── Stage 9: TTS / STT availability ──
    print("\n[STAGE 9] Voice Pipeline Availability")

    def check_stt():
        try:
            import speech_recognition  # noqa: F401
            return "speech_recognition OK"
        except ImportError:
            raise ImportError("speech_recognition not installed")

    r = bench("speech_recognition import", check_stt)
    results.append(r)
    print(r)

    def check_whisper():
        try:
            import faster_whisper  # noqa: F401
            return "faster_whisper OK"
        except ImportError:
            raise ImportError("faster-whisper not installed")

    r = bench("faster_whisper import", check_whisper)
    results.append(r)
    print(r)

    def check_edge_tts():
        try:
            import edge_tts  # noqa: F401
            return "edge_tts OK"
        except ImportError:
            raise ImportError("edge-tts not installed")

    r = bench("edge_tts import", check_edge_tts)
    results.append(r)
    print(r)

    def check_pygame():
        try:
            import pygame  # noqa: F401
            return "pygame OK"
        except ImportError:
            raise ImportError("pygame not installed")

    r = bench("pygame import", check_pygame)
    results.append(r)
    print(r)

    def check_llm():
        try:
            import llama_cpp  # noqa: F401
            return "llama_cpp OK"
        except ImportError:
            raise ImportError("llama-cpp-python not installed")

    r = bench("llama_cpp import", check_llm)
    results.append(r)
    print(r)

    # ── Stage 10: Memory snapshot ──
    print("\n[STAGE 10] Memory Snapshot")

    final_mem = mem_mb()
    system_mem = psutil.virtual_memory()
    print(f"  Process RSS:       {final_mem:.1f} MB")
    print(f"  RSS delta from baseline: +{final_mem - baseline_mem:.1f} MB")
    print(f"  System RAM used:   {system_mem.used / (1024**3):.2f} GB / {system_mem.total / (1024**3):.1f} GB ({system_mem.percent}%)")
    print(f"  System RAM free:   {system_mem.available / (1024**3):.2f} GB")

    # ── Summary ──
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    ok_results = [r for r in results if not r.error]
    fail_results = [r for r in results if r.error]

    total_time = sum(r.elapsed_ms for r in results)
    print(f"\n  Total benchmark time: {total_time:.0f}ms")
    print(f"  Modules OK: {len(ok_results)} / {len(results)}")
    print(f"  Modules FAIL: {len(fail_results)}")
    print(f"  Final RSS: {final_mem:.1f} MB")
    print(f"  Intent engine avg: {avg_intent:.3f}ms")

    if fail_results:
        print(f"\n  FAILURES:")
        for r in fail_results:
            print(f"    {r.name}: {r.error}")

    # ── Generate markdown report ──
    report_lines = [
        "## STEP 0.3 — BASELINE PERFORMANCE METRICS",
        "",
        f"> **Date:** {time.strftime('%Y-%m-%d')}",
        f"> **Platform:** {sys.platform} / Python {system_info['python']}",
        f"> **CPU cores:** {system_info['cpu_count']} | **RAM:** {system_info['ram_available_gb']}GB free / {system_info['ram_total_gb']}GB",
        "",
        "### Module Initialization Latency",
        "",
        "| Module | Time (ms) | Memory After (MB) | Memory Delta (MB) | Status |",
        "|--------|-----------|-------------------|-------------------|--------|",
    ]

    for r in results:
        status = "FAIL: " + r.error[:50] if r.error else "OK"
        report_lines.append(
            f"| {r.name} | {r.elapsed_ms:.1f} | {r.mem_after:.1f} | +{r.mem_delta:.1f} | {status} |"
        )

    report_lines.extend([
        "",
        "### Intent Engine Latency (per query)",
        "",
        "| Query | Time (ms) | Match |",
        "|-------|-----------|-------|",
    ])

    for query, elapsed, result in intent_times:
        match_str = result.intent if hasattr(result, 'intent') else str(result)
        report_lines.append(f"| {query} | {elapsed:.3f} | {match_str} |")

    report_lines.extend([
        "",
        f"**Intent avg:** {avg_intent:.3f}ms | **min:** {min_intent:.3f}ms | **max:** {max_intent:.3f}ms",
        "",
        "### Memory Footprint",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Baseline RSS (Python startup) | {baseline_mem:.1f} MB |",
        f"| Final RSS (all modules loaded) | {final_mem:.1f} MB |",
        f"| RSS delta | +{final_mem - baseline_mem:.1f} MB |",
        f"| System RAM total | {system_info['ram_total_gb']} GB |",
        f"| System RAM available | {system_info['ram_available_gb']} GB |",
        "",
        "### Voice Pipeline Status",
        "",
        "| Component | Status |",
        "|-----------|--------|",
    ])

    voice_modules = ["speech_recognition import", "faster_whisper import",
                     "edge_tts import", "pygame import", "llama_cpp import"]
    for vm in voice_modules:
        matching = [r for r in results if r.name == vm]
        if matching:
            status = "MISSING" if matching[0].error else "OK"
            report_lines.append(f"| {vm} | {status} |")

    report_lines.extend([
        "",
        "### Performance Targets vs Current",
        "",
        "| Metric | Current | Target | Gap |",
        "|--------|---------|--------|-----|",
        f"| Intent match latency | {avg_intent:.3f}ms | <100ms | {'MET' if avg_intent < 100 else 'MISS'} |",
        f"| Module load (all) | {total_time:.0f}ms | <5000ms | {'MET' if total_time < 5000 else 'MISS'} |",
        f"| Memory (steady state) | {final_mem:.0f}MB | <3072MB | {'MET' if final_mem < 3072 else 'MISS'} |",
        "| STT latency | N/A (not installed) | <300ms | BLOCKED |",
        "| LLM latency | N/A (not installed) | <2000ms | BLOCKED |",
        "| TTS latency | N/A (not installed) | <100ms | BLOCKED |",
        "| E2E voice-to-voice | N/A | <3000ms | BLOCKED |",
        "",
    ])

    report_text = "\n".join(report_lines)

    print("\n" + "=" * 80)
    print("  Benchmark complete. Writing report...")
    print("=" * 80)

    return report_text


if __name__ == "__main__":
    report = run_benchmarks()

    from pathlib import Path
    state_path = Path("docs/ATOM_CURRENT_STATE.md")
    if state_path.exists():
        content = state_path.read_text()
        marker = "## NEXT STEPS"
        if marker in content:
            parts = content.split(marker, 1)
            new_content = parts[0] + "---\n\n" + report + "\n---\n\n" + marker + parts[1]
            state_path.write_text(new_content)
            print(f"\n  Report appended to {state_path}")
        else:
            out_path = Path("docs/ATOM_BASELINE_METRICS.md")
            out_path.write_text(report)
            print(f"\n  Report saved to {out_path}")
    else:
        out_path = Path("docs/ATOM_BASELINE_METRICS.md")
        out_path.write_text(report)
        print(f"\n  Report saved to {out_path}")
