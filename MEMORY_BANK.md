# ATOM MEMORY BANK

> **Purpose:** Single source of truth for evolution progress. Read this FIRST in every new session.
> **Updated:** 2026-04-11 (Prediction Preload Enabled)
> **Hardware:** MacBook Air M5 (Apple Silicon, Unified Memory, Neural Engine, Metal GPU)

---

## CURRENT STATUS

```
CURRENT_STEP  = 4.3
OVERALL_PHASE = PHASE 4 — INTELLIGENCE UPGRADE
BLOCKER       = None — MLX installs in the current project venv; abbreviated integration smoke found non-fatal runtime defects only
LAST_ACTION   = Step 4.2 DONE — upgraded the prediction engine to warm prompt/RAG/app resources for high-confidence predictions with cooldowns and degraded-mode safety.
NEXT_ACTION   = Execute Step 4.3 — enhance `core/rag/rag_engine.py` with temporal decay, owner-priority, usage boosts, and staleness awareness
```

### COMPLETED PHASES SUMMARY

| Phase | Status | Key Result |
|-------|--------|------------|
| **Phase 0** (Baseline) | **DONE** (4/4 steps) | `docs/ATOM_CURRENT_STATE.md` — 49 Windows-only paths found, 879ms boot, 44.8MB RSS |
| **Phase 1** (Mac Survival) | **DONE** (10/10 steps) | ATOM boots on M5 with zero crashes. All Windows code has macOS equivalents. Silicon Refactoring: ~1000 lines NVIDIA dead code removed. |
| **Native macOS Stack** | **DONE** (6 modules) | pyobjc bridge to Apple frameworks. STT/TTS/OCR/Media/FSEvents/Embeddings all native. Phase 5 steps 5.6+5.7 done early. |
| **Model Strategy** | **CONFIRMED** | Qwen3-4B primary (3GB, thinking mode) + Qwen3-1.7B fast (1.2GB). Total 4.2GB, 5.8GB headroom. |

---

## QUICK CONTEXT (for AI sessions)

**What is ATOM?** Satyam's JARVIS-style voice AI OS. ~51,400 lines Python. Local LLM now runs on MLX/Qwen3 dual-models on Apple Silicon, with the older llama-cpp/GGUF path retained only as a legacy baseline/reference path. Native macOS STT/TTS/OCR (pyobjc), aiohttp dashboard, 40+ tools, ReAct loop, autonomous behavior.

**What are we doing?** Evolving ATOM from a Windows-built prototype to a production-grade cognitive OS optimized for Apple Silicon M5. The full plan is in `docs/ATOM_M5_EVOLUTION_PLAN.md`.

**Key files to read for context:**
- `MEMORY_BANK.md` (this file) — current step, what's done, what's next
- `CODEBASE_INDEX.md` — every file in the project with descriptions
- `docs/ATOM_M5_EVOLUTION_PLAN.md` — full evolution plan with technical details

---

## EVOLUTION STEPS

### PHASE 0 — BASELINE & MAC TRIAGE

| Step | Description | Status | Report |
|------|------------|--------|--------|
| 0.1 | Run `main.py`, capture all import errors and crashes on macOS | **DONE** | `docs/ATOM_CURRENT_STATE.md` |
| 0.2 | Scan all files for Windows-only code paths (winreg, WMIC, SAPI, ctypes.windll) | **DONE** | `docs/ATOM_CURRENT_STATE.md` (Step 0.2 section) |
| 0.3 | Measure baseline: latency per stage, memory footprint, CPU usage | **DONE** | `docs/ATOM_CURRENT_STATE.md` (Step 0.3 section) |
| 0.4 | Create `docs/ATOM_CURRENT_STATE.md` with full baseline report | **DONE** | Merged into 0.1-0.3 — doc is complete |

**Phase 0 Deliverable:** `docs/ATOM_CURRENT_STATE.md` — complete baseline snapshot.

---

### PHASE 1 — MAC SURVIVAL (make it boot without crashing)

| Step | Description | Files | Status | Report |
|------|------------|-------|--------|--------|
| 1.1 | Fix `platform_adapter.py` — GPU info, display info, service listing for macOS | `core/platform_adapter.py` | **DONE** | Tested: all 5 methods working |
| 1.2 | Create `core/apple_silicon_monitor.py` — replace pynvml with macOS hardware monitoring | NEW: `core/apple_silicon_monitor.py`, UPDATE: `core/gpu_governor.py` | **DONE** | Tested: all stats working |
| 1.3 | Fix GPU resource manager — unified memory model (no VRAM budgets on Apple Silicon) | `core/gpu_resource_manager.py`, `core/gpu_execution_coordinator.py` | **DONE** | Tested: memory pressure model working |
| 1.4 | Create `voice/tts_macos.py` — native macOS TTS via `say` command | NEW: `voice/tts_macos.py`, UPDATE: TTS selection in `main.py` | **DONE** | Tested: speak, barge-in, auto-detect |
| 1.5 | Fix `screen_reader.py` — macOS screenshot via `screencapture` command | `context/screen_reader.py` | **DONE** | Tested: screencapture works, 8.7MB Retina PNG |
| 1.6 | Fix `system_control.py` — macOS implementations for volume, brightness, wifi, startup | `core/system_control.py` | **DONE** | Tested: 10/10 methods working |
| 1.7 | Fix `desktop_control.py` — ensure pyautogui works on macOS or add AppleScript fallback | `core/desktop_control.py`, `core/security_policy.py` | **DONE** | Tested: all 11 functions working via AppleScript |
| 1.8 | Install llama-cpp-python with Metal: `CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python` | Terminal | **DONE** | Metal build confirmed, import OK |
| 1.8A | Silicon Refactoring: strip NVIDIA/CUDA, create Apple Silicon-native compute layer | `core/` | **DONE** | 3 files deleted, 2 created, 8 updated, ~1200 lines removed |
| 1.9 | Run `main.py` again — verify zero crashes on macOS | Terminal | **DONE** | Zero crashes, health 75/100, all modules OK |

**Phase 1 Deliverable:** ATOM boots and runs on M5 without crashes. All Windows-only code paths have macOS equivalents or graceful fallbacks.

---

### PHASE 2 — STABILITY & COGNITIVE KERNEL

| Step | Description | Files | Status | Report |
|------|------------|-------|--------|--------|
| 2.1 | Create `core/cognitive_kernel.py` — central brain coordinator (mode selection, query routing, resource allocation). Note: GPU stack consolidation (InferenceGuard) already done in Step 1.8A. | NEW: `core/cognitive_kernel.py` | **DONE** | 5 paths, circuit breakers, system-aware routing, wired to Router + main.py |
| 2.2 | Add error isolation wrappers to: `router.py`, `local_brain_controller.py`, `action_executor.py`, `wiring.py`, `stt_async.py`, `tts_edge.py` | 6 files | **DONE** | Top-level error boundaries + guarded wiring + smoke-tested forced failures |
| 2.3 | Upgrade `runtime_watchdog.py` — per-module execution budgets with timeout kill | `core/runtime_watchdog.py` + integration call sites | **DONE** | Active budgets for intent/cache/LLM/tools, RAG cap, TTS timeout supervision, wired + smoke-tested |
| 2.4 | Add memory pressure protection — limits on vector results, RAG snippets, graph nodes + periodic memory check | Multiple core files | **DONE** | Vector/RAG/graph caps + `silicon_stats_update` pressure shedding + smoke-tested |
| 2.5 | Integration test — run ATOM for 30 min, verify no crashes, no memory leaks | Terminal | **DONE** | Abbreviated smoke/integration pass per user request: no full-process crash, RSS stayed flat, several non-fatal runtime defects logged |

**Phase 2 Deliverable:** ATOM is unbreakable. Cognitive Kernel routes all decisions. No single module can crash the system.

---

### PHASE 3 — SPEED (MLX + Dual-Model Architecture)

**Dual-Model Strategy (confirmed 2026-04-09):**
- **Primary brain:** Qwen3-4B-Q4_K_M (~3.0 GB RAM, 50-70 tok/s on M5)
  - Conversation, reasoning, complex tool calls, multi-step ReAct
  - Thinking mode ON for complex queries (matches 7B quality), OFF for speed
  - 32K native context window, excellent tool calling, bilingual EN/HI
- **Fast brain:** Qwen3-1.7B-Q4_K_M (~1.2 GB RAM, 120-160 tok/s on M5)
  - Quick acks, simple tool calls, summaries, short answers
  - Same Qwen3 family = identical prompt format + tool call syntax (zero adaptation)
  - Thinking mode always OFF (speed priority)
- **Total model RAM:** ~4.2 GB (leaves ~5.8 GB headroom on 16 GB M5)
- **Routing:** Cognitive Kernel picks which model based on task complexity:
  - Intent match / known command → skip LLM entirely (sub-5ms)
  - Simple query / quick tool call → Qwen3-1.7B fast brain (80-150ms)
  - Conversation / reasoning → Qwen3-4B thinking OFF (300-600ms)
  - Complex reasoning / multi-step ReAct → Qwen3-4B thinking ON (800-2000ms)
- **Priority:** Fast brain handles 70%+ of interactions. Primary brain only for real thinking.
- **Why Qwen3 family:** Same prompt template (ChatML), same tool call format, same tool_parser.py — zero code adaptation. Thinking mode toggle gives two speeds from one model.

| Step | Description | Files | Status | Report |
|------|------------|-------|--------|--------|
| 3.1 | Install MLX: `pip install mlx mlx-lm` | Terminal | **DONE** | `mlx` + `mlx-lm` installed in project venv; functional imports verified |
| 3.2 | Download MLX models: Qwen3-4B-Q4_K_M (primary) + Qwen3-1.7B-Q4_K_M (fast) | `models/` | **DONE** | Downloaded `mlx-community/Qwen3-4B-4bit` + `mlx-community/Qwen3-1.7B-4bit` into project-local model dirs |
| 3.3 | Create `brain/mlx_llm.py` — MLX-native LLM wrapper with streaming | NEW: `brain/mlx_llm.py` | **DONE** | Added `MLXBrain` compatibility wrapper with async streaming, preempt, primary/fast roles, and local-model defaults |
| 3.4 | Update `local_brain_controller.py` — use MLX brain instead of llama-cpp | `cursor_bridge/local_brain_controller.py` | **DONE** | Swapped controller backend to `MLXBrain`; warm-up and unload smoke verified through the controller layer |
| 3.5 | Implement dual-model routing in Cognitive Kernel (Qwen3-1.7B for simple, Qwen3-4B for complex, thinking mode toggle) | `core/cognitive_kernel.py` + LLM handoff files | **DONE** | Kernel now emits `model_role` / `runtime_mode` / prompt hints, and the local-brain path honors them |
| 3.6 | Create `core/runtime/latency_controller.py` — dynamic latency budgets | NEW: `core/runtime/latency_controller.py` + routing integrations | **DONE** | Dynamic latency/context budgets now flow through `QueryPlan`, router memory gating, and local-brain RAG/prompt trimming |
| 3.7 | Update `config/settings.json` — MLX model paths, Apple Silicon tuning | `config/settings.json` + schema alignment | **DONE** | Explicit MLX paths, latency/kernel defaults, watchdog budgets, and schema validation updated |
| 3.8 | Benchmark: measure MLX vs llama-cpp inference speed on M5 | Terminal + `tools/mlx_vs_llamacpp_benchmark.py` | **DONE** | MLX benchmark recorded on local Qwen3-4B; matching GGUF baseline downloaded but current llama-cpp 0.3.2 fails to load it |

**Phase 3 Deliverable:** ATOM uses MLX for inference. Qwen3-4B (primary, ~3GB) + Qwen3-1.7B (fast, ~1.2GB) loaded simultaneously in Unified Memory. Sub-150ms via fast brain, sub-600ms via primary brain (thinking OFF), sub-2s for deep reasoning (thinking ON).

---

### PHASE 4 — INTELLIGENCE UPGRADE

| Step | Description | Files | Status | Report |
|------|------------|-------|--------|--------|
| 4.1 | Implement cognitive budget system in Cognitive Kernel | `core/cognitive_kernel.py` | **DONE** | Added concrete budget tiers, tier-aware routing, and low-pressure degradation |
| 4.2 | Upgrade prediction engine — resource preloading for high-confidence predictions | `core/cognitive/prediction_engine.py` | **DONE** | Added lightweight prediction-driven prompt/RAG/app warming with cooldowns |
| 4.3 | Enhance RAG — temporal decay, owner-priority, usage-frequency boost | `core/rag/rag_engine.py` | NOT_STARTED | — |
| 4.4 | Create `core/identity_engine.py` — ATOM self-identity + owner relationship model | NEW: `core/identity_engine.py` | NOT_STARTED | — |
| 4.5 | Integration test — verify intelligence routing, RAG quality, personality adaptation | Terminal | NOT_STARTED | — |

**Phase 4 Deliverable:** ATOM intelligently routes queries, predicts and preloads, retrieves contextually relevant memories.

---

### PHASE 5 — DEEP macOS INTEGRATION

| Step | Description | Files | Status | Report |
|------|------------|-------|--------|--------|
| 5.1 | Create `core/macos/applescript_engine.py` — deep macOS automation | NEW: `core/macos/applescript_engine.py` | NOT_STARTED | — |
| 5.2 | Integrate Accessibility API via pyobjc — read/control app UI elements | NEW: `core/macos/accessibility.py` | NOT_STARTED | — |
| 5.3 | Add Spotlight search tool (`mdfind`) to tool registry | `core/reasoning/tool_registry.py` | NOT_STARTED | — |
| 5.4 | Add macOS Keychain integration for secure credential storage | `core/security_fortress.py` or NEW file | NOT_STARTED | — |
| 5.5 | Create `launchd` plist for background agent mode | NEW: `scripts/com.atom.agent.plist` | NOT_STARTED | — |
| 5.6 | Add FSEvents file monitoring for proactive suggestions | `core/macos/fs_watcher.py` | **DONE** (Native Stack) | FSEvents kernel-level, near-zero CPU, watches ~/Desktop,Downloads,Documents |
| 5.7 | Add native macOS screen OCR via Vision framework | `context/screen_reader.py` | **DONE** (Native Stack) | Vision OCR 109ms avg, 89 regions, Neural Engine. EasyOCR fallback preserved |
| 5.8 | Integration test — full macOS control suite | Terminal | NOT_STARTED | — |

**Phase 5 Deliverable:** ATOM controls macOS like JARVIS controls the lab. AppleScript, Accessibility API, Spotlight, Keychain, background agent all working.

---

### PHASE 6 — AUTONOMY & PROACTIVE BEHAVIOR

| Step | Description | Files | Status | Report |
|------|------------|-------|--------|--------|
| 6.1 | Upgrade proactive engine — enhanced trigger system (time, system, behavioral, context-aware) | `core/cognitive/proactive_engine.py`, `core/proactive_awareness.py` | NOT_STARTED | — |
| 6.2 | Upgrade goal engine — real execution pipeline (goal → steps → tool calls → tracking) | `core/cognitive/goal_engine.py` | NOT_STARTED | — |
| 6.3 | Enhance dream mode for M5 — idle memory consolidation, pattern summaries, low-power model | `core/cognitive/dream_engine.py` | NOT_STARTED | — |
| 6.4 | Integration test — proactive suggestions, goal tracking, dream mode cycle | Terminal | NOT_STARTED | — |

**Phase 6 Deliverable:** ATOM works autonomously. Proactive suggestions, real goal execution, dream mode consolidation.

---

### PHASE 7 — TESTING & HARDENING

| Step | Description | Files | Status | Report |
|------|------------|-------|--------|--------|
| 7.1 | Run stress tests: `v7_stress_test.py`, `v7_chaos_test.py`, `v7_long_run.py` | `scripts/` | NOT_STARTED | — |
| 7.2 | macOS-specific tests: AirPods disconnect, sleep/wake, memory pressure, battery transition | NEW test scripts | NOT_STARTED | — |
| 7.3 | Performance validation against targets (see table below) | Terminal | NOT_STARTED | — |
| 7.4 | Create final `docs/ATOM_M5_PRODUCTION_REPORT.md` | NEW doc | NOT_STARTED | — |

**Performance Targets:**

| Metric | Target |
|--------|--------|
| Known command (intent match) | < 100ms |
| Simple query (small model) | < 500ms |
| Full conversation (large model) | < 2s |
| TTS first word | < 100ms |
| End-to-end (voice in → voice out) | < 3s |
| Memory (steady state) | < 3GB |
| Crash rate | 0 |

**Phase 7 Deliverable:** ATOM is production-grade on M5. All targets met. Final report generated.

---

## COMPLETION REPORTS

> Each completed step gets a report entry here. This is how future sessions know exactly what was done.

### --- TEMPLATE (copy for each completed step) ---
```
### Step X.Y — [Title]
**Date:** YYYY-MM-DD
**Status:** DONE
**Files Modified:** [list]
**Files Created:** [list]
**What Changed:**
  - [bullet points of actual changes made]
**Test Result:**
  - [how it was verified]
**Issues Found:**
  - [any problems discovered during this step]
**Notes for Next Step:**
  - [anything the next session needs to know]
```

### Step 3.1 — Install MLX
**Date:** 2026-04-11
**Status:** DONE
**Files Modified:** none
**Files Created:** none
**What Changed:**
  - Installed `mlx` and `mlx-lm` into the active project virtual environment via `python3 -m pip install mlx mlx-lm`
  - This also pulled the required Apple Silicon runtime package `mlx-metal`
  - Confirmed the environment can import `mlx.core`, `mlx.nn`, and `mlx_lm`, and execute a basic MLX tensor operation
**Test Result:**
  - `python3 -m pip install mlx mlx-lm` completed successfully
  - Functional import check passed:
    - `import mlx.core as mx`
    - `import mlx.nn as nn`
    - `import mlx_lm`
    - `mx.array([1, 2, 3]).tolist()` returned `[1, 2, 3]`
**Issues Found:**
  - Importing the Hugging Face dependency stack surfaces a `urllib3` LibreSSL warning on this Python build, but MLX itself imports and basic ops work correctly
  - The earlier Phase 3 blocker claiming Python 3.11+ was required for MLX is no longer accurate for this environment
**Notes for Next Step:**
  - Step 3.2 should download the two MLX model repos into `models/` and record their on-disk locations
  - Step 3.3 can now assume `mlx` and `mlx_lm` are available in the project venv

### Step 3.2 — Download MLX Models
**Date:** 2026-04-11
**Status:** DONE
**Files Modified:** none
**Files Created:**
  - `models/qwen3-4b-mlx/` — downloaded `mlx-community/Qwen3-4B-4bit` snapshot (config, tokenizer, sharded safetensors, README, cache metadata)
  - `models/qwen3-1.7b-mlx/` — downloaded `mlx-community/Qwen3-1.7B-4bit` snapshot (config, tokenizer, sharded safetensors, README, cache metadata)
**What Changed:**
  - Downloaded the primary MLX model into `models/qwen3-4b-mlx`
  - Downloaded the fast MLX model into `models/qwen3-1.7b-mlx`
  - Kept both models project-local so Phase 3 code can reference stable filesystem paths instead of ad-hoc Hugging Face cache locations
**Test Result:**
  - Download completed successfully for both repos using `huggingface_hub.snapshot_download()`
  - Verified on-disk sizes:
    - `models/qwen3-4b-mlx` ≈ 2.1 GB
    - `models/qwen3-1.7b-mlx` ≈ 939 MB
  - Verified expected MLX model files in both directories:
    - `config.json`
    - `model.safetensors` + `model.safetensors.index.json`
    - `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`
    - vocabulary / merge files and README
**Issues Found:**
  - Hugging Face download stack still emits the LibreSSL / `urllib3` warning on this Python build, but it did not block download or local verification
**Notes for Next Step:**
  - Step 3.3 should use these local model directories directly in the MLX wrapper
  - Suggested config paths for the wrapper:
    - primary: `models/qwen3-4b-mlx`
    - fast: `models/qwen3-1.7b-mlx`

### Step 3.3 — Create MLX LLM Wrapper
**Date:** 2026-04-11
**Status:** DONE
**Files Modified:** none
**Files Created:**
  - `brain/mlx_llm.py` — new `MLXBrain` wrapper for MLX model loading, async generation, streaming callbacks, preemption, and unload/close handling
**What Changed:**
  - Created `MLXBrain` as a MiniLLM-compatible bridge so `LocalBrainController` can swap from llama-cpp to MLX without redesigning its async contract first
  - Added lazy model loading for two roles: `primary` (`models/qwen3-4b-mlx`) and `fast` (`models/qwen3-1.7b-mlx`)
  - Implemented synchronous MLX streaming under a single-worker executor, preserving ATOM's existing async `generate()` / `generate_streaming()` surface and `(text, preempted)` return shape
  - Added `request_abort_preempt()` handling, timeout-triggered preemption, and unload helpers that clear MLX cache so later runtime pressure or power policies can reclaim memory
  - Preserved compatibility stubs for unused MiniLLM KV-cache methods so the replacement can be integrated incrementally
**Test Result:**
  - `python3 -m py_compile brain/mlx_llm.py` passed
  - IDE lint check for `brain/mlx_llm.py` returned no diagnostics
  - Focused smoke test passed with the fast local model:
    - `MLXBrain(...).set_model_role("fast")`
    - `preload()` returned `True`
    - `generate_streaming("Say hello to Boss in one short sentence.")` returned a streamed response with `preempted=False`
    - End-to-end wrapper smoke completed in ~3.4s on the local environment
**Issues Found:**
  - Importing the MLX/Hugging Face stack still emits the known LibreSSL / `urllib3` warning on this Python build, but it did not block load or generation
  - The wrapper is not wired into `LocalBrainController` yet, so this step validates the MLX layer in isolation rather than the full ATOM response path
**Notes for Next Step:**
  - Step 3.4 should replace the `MiniLLM` import/instantiation in `cursor_bridge/local_brain_controller.py` with `MLXBrain`
  - The wrapper already supports explicit `primary` vs `fast` roles; Step 3.4 can start with `primary` as the default and let Step 3.5 drive routing decisions
  - `shutdown()` is unload-oriented and keeps the executor reusable; `close()` performs the final executor teardown if a hard close path is needed later

### Step 3.4 — Wire LocalBrainController to MLX
**Date:** 2026-04-11
**Status:** DONE
**Files Modified:**
  - `cursor_bridge/local_brain_controller.py` — replaced `MiniLLM` usage with `MLXBrain`, updated lifecycle messaging, and used `close()` when available for final teardown
**Files Created:** none
**What Changed:**
  - Swapped the controller's local-brain backend import from `brain.mini_llm.MiniLLM` to `brain.mlx_llm.MLXBrain`
  - Kept the controller's existing async inference flow unchanged so the new MLX backend fits behind the same `generate_streaming()` / preempt / warm-up surface
  - Updated warm-up and unavailable-response text to reference MLX model directories and `mlx` / `mlx_lm` instead of GGUF / llama-cpp
  - Kept `unload_llm_for_power()` as an unload-only path while using `close()` for final teardown when the backend supports it
**Test Result:**
  - `python3 -m py_compile brain/mlx_llm.py cursor_bridge/local_brain_controller.py` passed
  - IDE lint check for both edited files returned no diagnostics
  - Focused controller smoke passed with lightweight stubs:
    - `LocalBrainController(...).available` returned `True`
    - `await warm_up()` completed successfully against the local MLX primary model
    - `is_loaded` became `True`
    - `unload_llm_for_power()` dropped `is_loaded` back to `False`
**Issues Found:**
  - The known LibreSSL / `urllib3` warning still appears when the MLX stack imports, but it did not block controller warm-up or unload
  - Some non-controller diagnostics and helper scripts still mention GGUF / llama-cpp paths and wording; runtime controller wiring now uses MLX, but ancillary messaging cleanup is still pending
**Notes for Next Step:**
  - Step 3.5 should start routing queries onto `fast` vs `primary` model roles by pushing that decision into the Cognitive Kernel and then into the MLX backend
  - The controller currently defaults to the wrapper's `primary` role; no query-aware model selection is wired yet
  - Once routing exists, re-check whether startup should preload both roles or keep the fast model lazy-loaded until first use

### Step 3.5 — Dual-Model Routing in Cognitive Kernel
**Date:** 2026-04-11
**Status:** DONE
**Files Modified:**
  - `core/cognitive_kernel.py` — extended `QueryPlan` with `model_role`, `runtime_mode`, and prompt-hint fields; quick/full/deep routes now map directly onto fast vs primary MLX roles
  - `cursor_bridge/local_brain_controller.py` — consumes `query_plan` hints for MLX role selection, runtime-mode alignment, RAG gating, prompt guidance, and late-RAG retry continuity
  - `cursor_bridge/structured_prompt_builder.py` — injects a routing hint into the prompt context layer so `DEEP` queries explicitly bias toward careful reasoning
  - `core/llm_inference_queue.py` — preserves `query_plan` when the in-process queue coalesces and forwards work
  - `core/boot/wiring.py` — forwards `query_plan` into the queue path and updates the local-brain unavailable messaging to MLX wording
**Files Created:** none
**What Changed:**
  - Made the Cognitive Kernel the source of truth for model-role selection: `QUICK` now targets MLX role `fast`, while `FULL` and `DEEP` target MLX role `primary`
  - Added `runtime_mode` hints to query plans so the controller's existing `FAST` / `SMART` / `DEEP` logic stays aligned with kernel routing instead of making an independent conflicting decision
  - Added route-specific prompt guidance so fast-path prompts bias toward concise answers and deep-path prompts bias toward careful reasoning without redesigning the prompt-builder surface
  - Taught `LocalBrainController` to honor `use_rag` / `use_memory` from the route plan, and to preserve the plan through late-RAG restart so deep queries do not silently fall back to default routing on retry
  - Fixed the in-process LLM queue so `query_plan` survives queue submission; without this, queue-backed inference would have dropped the kernel's model-role decision before generation
**Test Result:**
  - `python3 -m py_compile core/cognitive_kernel.py cursor_bridge/local_brain_controller.py cursor_bridge/structured_prompt_builder.py core/boot/wiring.py core/llm_inference_queue.py` passed
  - IDE lint check returned no diagnostics for all modified files
  - Focused routing propagation smoke passed:
    - Kernel classified a simple query as `quick` with `model_role=fast`, `runtime_mode=FAST`
    - Kernel classified a medium-complexity query as `full` with `model_role=primary`, `runtime_mode=SMART`
    - Kernel classified a long complex query as `deep` with `model_role=primary`, `runtime_mode=DEEP`, `thinking=True`
    - Stubbed `LocalBrainController` consumed those plans and forwarded `fast` vs `primary` into `generate_streaming(...)`
    - `LLMInferenceQueue` preserved `query_plan` and delivered it intact to `on_query(...)`
**Issues Found:**
  - The known LibreSSL / `urllib3` warning still appears on imports that touch the MLX/Hugging Face stack, but it did not block the routing smoke
  - Initial Step 3.5 wiring still left router-side memory retrieval ahead of plan consumption, but that follow-up optimization was completed immediately afterward in Step 3.6
**Notes for Next Step:**
  - Step 3.6 should turn the kernel's path choice plus live system state into dynamic stage budgets instead of relying only on the current static per-path defaults
  - Follow-up completed in Step 3.6: query planning now happens early enough in the router for quick-path queries to skip unnecessary semantic-memory retrieval
  - Startup still only warms the default role; decide later whether Phase 3 should preload both MLX roles or keep the fast role lazy for better cold-start behavior

### Step 3.6 — Latency Controller
**Date:** 2026-04-11
**Status:** DONE
**Files Modified:**
  - `core/cognitive_kernel.py` — applies dynamic latency decisions onto `QueryPlan` (`budget_ms`, `rag_budget_ms`, `reduce_context`, memory/history limits) and emits richer routing diagnostics
  - `core/router/router.py` — plans before memory retrieval, serves direct/cache responses earlier, and skips semantic-memory work for quick plans that do not need it
  - `cursor_bridge/local_brain_controller.py` — trims prompt memory/history from the plan, caps RAG budget from the plan, and keeps MLX role routing aligned with the new latency policy
  - `core/runtime/__init__.py` — exports the latency controller for runtime modules
**Files Created:**
  - `core/runtime/latency_controller.py` — dynamic budget engine for total query budget, RAG budget, context trimming, and low-latency degradation
**What Changed:**
  - Added a dedicated `LatencyController` that converts path + system state into concrete limits: total budget, optional RAG budget, prompt context reduction, memory snippet cap, and history turn cap
  - Integrated that controller into `CognitiveKernel` so every `QueryPlan` now carries not just the chosen path/model role, but also the latency envelope that path should run under right now
  - Closed the outstanding router optimization gap from Step 3.5 by moving Cognitive Kernel planning ahead of semantic-memory retrieval in the fallback path
  - Quick/direct/cache plans can now avoid unnecessary memory work earlier, while full/deep plans carry explicit context and RAG limits into `LocalBrainController`
  - `LocalBrainController` now consumes those plan limits to trim prompt memory/history and to cap RAG retrieval time instead of relying only on static defaults
**Test Result:**
  - `python3 -m py_compile core/runtime/latency_controller.py core/runtime/__init__.py core/cognitive_kernel.py core/router/router.py cursor_bridge/local_brain_controller.py` passed
  - IDE lint check returned no diagnostics for all modified files
  - Focused smoke checks passed:
    - Neutral-state plans now carry dynamic latency fields (`budget_ms`, `rag_budget_ms`, memory/history limits)
    - Memory-pressure routing produced a degraded quick plan with `reduce_context=True` and `rag_budget_ms=0`
    - Router quick-path fallback skipped semantic-memory retrieval entirely (`memory_calls=0`) while still forwarding the `query_plan`
    - Stubbed `LocalBrainController` honored `history_turn_limit`, dropped memory context when `use_memory=False`, and still forwarded MLX role `fast`
**Issues Found:**
  - The known LibreSSL / `urllib3` warning still appears on imports that touch the MLX/Hugging Face stack, but it did not block latency-controller or router smoke checks
  - Dynamic latency budgets currently shape router memory retrieval, prompt trimming, and RAG capping, but watchdog stage timeouts still remain the hard upper bound for actual execution
**Notes for Next Step:**
  - Step 3.7 should add the explicit MLX model path defaults and any latency-controller tuning overrides into `config/settings.json`
  - After Step 3.7, benchmarking can compare whether the new quick-path memory skip and dynamic context trimming materially improve first-token latency on M5

### Step 3.7 — Config Defaults for MLX + Apple Silicon
**Date:** 2026-04-11
**Status:** DONE
**Files Modified:**
  - `config/settings.json` — added explicit MLX model paths, latency-controller defaults, cognitive-kernel defaults, watchdog budgets, and Apple Silicon-friendly embedding config
  - `core/config_schema.py` — added schema coverage for the new MLX/config sections and cleaned up adjacent schema drift so the checked-in settings validate cleanly
**Files Created:** none
**What Changed:**
  - Added explicit `brain.mlx_primary_model`, `brain.mlx_fast_model`, and `brain.mlx_default_role` defaults so the MLX runtime no longer depends on implicit code fallbacks
  - Added explicit `cognitive_kernel` and `latency_controller` sections to `config/settings.json` so the routing/budget heuristics introduced in Steps 3.5-3.6 are now visible, configurable defaults
  - Added explicit watchdog stage budgets to `performance` and switched `embedding.device` from `cpu` to `auto` so Apple Silicon can use MPS when available instead of being pinned to CPU by config
  - Added `rag.adaptive` defaults so the existing adaptive RAG-budget code has checked-in values instead of relying on internal fallbacks
  - Expanded the config schema for MLX brain keys (`mlx_primary_model`, `mlx_fast_model`, `mlx_default_role`), existing brain tuning keys (`n_batch`, `top_p`, `repeat_penalty`, `n_gpu_layers=-1`), and the new top-level `cognitive_kernel` / `latency_controller` sections
  - Fixed nearby schema drift for existing checked-in settings (`stt.bilingual`, cognitive dream/curiosity keys), which removed false validation errors unrelated to runtime behavior
**Test Result:**
  - `python3 -m py_compile core/config_schema.py` passed
  - IDE lint check for `core/config_schema.py` returned no diagnostics
  - Config validation passed cleanly:
    - `validate_config(config/settings.json)` returned `0` errors
**Issues Found:**
  - None blocking for this step; the known LibreSSL / `urllib3` warning is unrelated to config/schema validation and did not affect the result
**Notes for Next Step:**
  - Step 3.8 should benchmark end-to-end MLX behavior now that model paths and tuning defaults are explicit and stable
  - If benchmarking reveals first-token regressions, the new `latency_controller` and `cognitive_kernel` settings can be tuned without additional code changes

### Step 3.8 — Benchmark MLX vs llama-cpp on M5
**Date:** 2026-04-11
**Status:** DONE
**Files Modified:** none
**Files Created:**
  - `tools/mlx_vs_llamacpp_benchmark.py` — isolated-subprocess benchmark for preload time, RSS delta, first-token latency, and total generation time across MLX vs llama-cpp
  - `models/Qwen3-4B-Q4_K_M.gguf` — downloaded GGUF baseline for same-family Qwen3-4B comparison (~2.33 GB)
**What Changed:**
  - Added a dedicated benchmark script that runs each backend in a fresh subprocess so load time and RSS are not polluted by the other backend
  - Chose a same-family backend comparison target: MLX Qwen3-4B local model vs official Qwen3-4B `Q4_K_M` GGUF baseline
  - Downloaded the GGUF baseline into the workspace and ran the benchmark with two measured runs per scenario after warm-up
  - Verified that MLX loads and serves the target Qwen3-4B model locally on this M5 environment
  - Verified that the current llama-cpp baseline in this environment fails before inference: it sees the file but cannot load the target Qwen3-4B GGUF at all
**Test Result:**
  - `python3 -m py_compile tools/mlx_vs_llamacpp_benchmark.py` passed
  - IDE lint check for `tools/mlx_vs_llamacpp_benchmark.py` returned no diagnostics
  - Benchmark run completed:
    - `python3 tools/mlx_vs_llamacpp_benchmark.py --runs 2 --json`
  - MLX (`models/qwen3-4b-mlx`) results:
    - preload: `607.75ms`
    - RSS delta after preload: `+2318.3 MB`
    - `quick_summary`: avg first token `439.9ms`, avg total `4200.64ms`
    - `technical_reasoning`: avg first token `430.69ms`, avg total `4092.4ms`
  - llama-cpp baseline (`models/Qwen3-4B-Q4_K_M.gguf`) result:
    - matching GGUF file downloaded successfully (~`2.33 GB`)
    - current `llama-cpp-python 0.3.2` backend reported `ok=false` because preload failed before any generation run
**Issues Found:**
  - The direct backend speed comparison is blocked in the current environment because `llama-cpp-python 0.3.2` cannot load `Qwen3-4B-Q4_K_M.gguf`, even though the file exists and MLX runs the corresponding Qwen3-4B model correctly
  - Because the baseline backend never reached inference, there is no valid first-token or total-latency speedup ratio to report yet
**Notes for Next Step:**
  - Phase 4 can proceed; the important Phase 3 takeaway is that MLX is not just configured, but actually running the target Qwen3-4B model locally on M5
  - If a future session needs a true apples-to-apples throughput comparison, the likely next move is upgrading/rebuilding the llama-cpp baseline to a version that can load Qwen3 GGUF, or benchmarking a different supported GGUF family separately
  - The new benchmark script is reusable for reruns once the llama-cpp baseline compatibility issue is resolved

### Step 4.2 — Prediction Preload
**Date:** 2026-04-11
**Status:** DONE
**Files Modified:**
  - `core/cognitive/prediction_engine.py` — added target extraction, live `llm_query` learning, high-confidence resource preloading, cooldowns, and conservative degraded-mode gating
  - `cursor_bridge/structured_prompt_builder.py` — added lightweight prompt precompile support and a small query-hint cache so predicted LLM work can warm prompt resources without building a full prompt
  - `main.py` — attached the prediction engine to the prompt builder, shared RAG prefetch engine, and cognitive kernel so predictions can warm real downstream resources
  - `core/config_schema.py` — added schema for prediction preload tuning knobs
  - `config/settings.json` — enabled conservative defaults for preload confidence, item cap, cooldown, and timeout
**Files Created:**
  - `tests/test_prediction_preload.py` — focused regression coverage for multi-resource warming, cooldown behavior, degraded-mode skipping, and live `llm_query` learning
**What Changed:**
  - Prediction cycles now do more than publish guesses: high-confidence predictions can warm lightweight app-resolution metadata, schedule RAG prefetch for likely knowledge queries, and precompile prompt-builder layers for likely LLM work
  - The preload path is intentionally MacBook-safe: it is capped by confidence, limited to a small number of items per cycle, guarded by cooldowns, and skips heavier RAG work when the routed plan is already degraded or context-reduced
  - Added live `cursor_query` tracking so the prediction engine can learn recurring `llm_query` patterns, not just direct intent-engine actions
  - Kept the design aligned with the new cognitive-budget layer from Step 4.1 by routing predicted queries through the cognitive kernel before deciding what to warm
**Test Result:**
  - `python3 -m py_compile core/cognitive/prediction_engine.py cursor_bridge/structured_prompt_builder.py core/config_schema.py main.py tests/test_prediction_preload.py` passed
  - `python3 -m tests.test_prediction_preload` passed:
    - prompt, RAG, and app warm-up all triggered for the right high-confidence predictions
    - preload cooldown prevented repeated warming of the same prediction
    - degraded plans skipped heavier prefetch work while still allowing cheap prompt warming
    - live `llm_query` traffic became predictable
  - `python3 -m tests.test_cognitive_kernel` passed
  - `python3 -m tests.test_state_machine` passed
  - IDE lint check returned no diagnostics for all edited files
**Issues Found:**
  - The app preload path intentionally warms app-resolution metadata rather than launching apps in the background; that keeps the feature useful on macOS without paying a RAM/CPU penalty from speculative app startups
**Notes for Next Step:**
  - Step 4.3 can now reuse the prediction preload path once smarter RAG scoring exists; better retrieval ranking will directly improve the value of prefetch hits
  - If future profiling shows prediction preload waste is too high, the next tuning lever is tightening `prediction_preload_min_confidence` or lowering `prediction_preload_max_items`, not adding heavier preload types

### Step 4.1 — Cognitive Budget System
**Date:** 2026-04-11
**Status:** DONE
**Files Modified:**
  - `core/cognitive_kernel.py` — added concrete cognitive budget tiers, requested-vs-applied budget tracking, tier-aware routing, and richer route observability/diagnostics
  - `core/runtime/latency_controller.py` — added base-budget overrides so tier budgets now drive the final dynamic latency decision instead of only the execution path defaults
  - `cursor_bridge/local_brain_controller.py` — expanded query-plan logging to surface the new budget-tier metadata during local-brain execution
**Files Created:**
  - `tests/test_cognitive_kernel.py` — focused regression coverage for command/info/simple/complex/creative routing and laptop-safe degradation on low battery
**What Changed:**
  - Added concrete budget tiers for `command`, `info`, `simple`, `complex`, and `creative` work, instead of routing only by raw path names like `quick`/`full`/`deep`
  - The kernel now records both the query's requested tier and the actually applied tier, so ATOM can preserve intent while still degrading execution safely when the machine is under battery, thermal, or memory pressure
  - Latency budgeting now honors the tier's base budget (`100ms`, `500ms`, `1500ms`, `5000ms`, `10000ms`) before applying dynamic scaling, which keeps route timing aligned with the evolution plan
  - Added budget-tier observability to the `cognitive_route` event and diagnostics, making it easier to see when ATOM chose to protect the laptop by downshifting a demanding query into a cheaper execution tier
**Test Result:**
  - `python3 -m py_compile core/cognitive_kernel.py core/runtime/latency_controller.py cursor_bridge/local_brain_controller.py tests/test_cognitive_kernel.py` passed
  - `python3 -m tests.test_cognitive_kernel` passed:
    - command/info fast paths received the expected concrete budgets
    - simple/complex/creative queries mapped to the intended execution paths
    - low-battery creative work degraded to a laptop-safe quick/simple budget
  - `python3 -m tests.test_local_brain_streaming` passed
  - `python3 -m tests.test_voice_interrupt` passed
  - `python3 -m tests.test_state_machine` passed
  - IDE lint check returned no diagnostics for all edited files
**Issues Found:**
  - None blocking. For direct no-LLM fast paths, the new budget metadata is exposed for observability even though the router still returns immediately instead of doing extra memory enrichment work, which keeps the fast path cheap on MacBook hardware
**Notes for Next Step:**
  - Phase 4.2 can now build on this budget layer: high-confidence predictions should preload only the resources justified by the current tier, instead of warming everything equally
  - The new requested-vs-applied budget fields are also a good hook for future dashboard or telemetry displays if deeper observability is needed

### Step 3.11 — Native macOS Streaming TTS
**Date:** 2026-04-11
**Status:** DONE
**Files Modified:**
  - `voice/tts_macos.py` — replaced end-of-response buffering with incremental streamed chunk playback, preserved the spoken-word cap, and added stale-stream guarding for interrupted responses
  - `cursor_bridge/local_brain_controller.py` — tagged each streamed `partial_response` turn with a lightweight `stream_id` so downstream consumers can reject stale chunks after preemption
  - `core/boot/wiring.py` — made dashboard stream aggregation aware of `stream_id` so stale interrupted chunks do not leak into the visible streaming transcript
  - `tests/test_local_brain_streaming.py` — extended the focused local-brain streaming regression to assert `stream_id` propagation
**Files Created:**
  - `tests/test_tts_macos_streaming.py` — focused regression coverage for first-chunk playback and stale-stream suppression on the native macOS TTS path
**What Changed:**
  - Converted the native macOS `partial_response` path into a true incremental stream: ATOM now starts speaking on the first streamed sentence chunk instead of buffering the whole answer and only speaking when the final chunk arrives
  - Kept the implementation MacBook-friendly by using the existing event-driven TTS path rather than adding a continuous VAD/inference loop; there is no new always-on background model load from this step
  - Added per-response `stream_id` propagation so when a user interrupts ATOM and a new answer starts, stale chunks from the older interrupted stream are ignored cleanly by both TTS and the UI transcript
  - Preserved the existing audio budget behavior: ATOM still caps spoken output to the first ~45 words and sends the remainder to screen-only display instead of over-talking
**Test Result:**
  - `python3 -m py_compile voice/tts_macos.py cursor_bridge/local_brain_controller.py core/boot/wiring.py tests/test_local_brain_streaming.py tests/test_tts_macos_streaming.py` passed
  - `python3 -m tests.test_tts_macos_streaming` passed:
    - macOS TTS started speaking on the first streamed chunk before the last chunk arrived
    - stale chunks from an older interrupted stream were ignored
  - `python3 -m tests.test_local_brain_streaming` passed
  - `python3 -m tests.test_voice_interrupt` passed
  - `python3 -m tests.test_state_machine` passed
  - IDE lint check returned no diagnostics for all edited files
**Issues Found:**
  - This step optimizes the native macOS TTS path used on the MacBook. The distributed/proxy TTS path still handles chunks more simply and was not expanded here because the goal was the lowest-overhead local path
**Notes for Next Step:**
  - Phase 4 can proceed; this step improves perceived responsiveness without increasing steady-state laptop load
  - If future tuning is needed, the next low-overhead refinement is better chunk boundary shaping or faster first-ack timing, not a continuous VAD loop

### Step 3.10 — Voice Interrupt System
**Date:** 2026-04-11
**Status:** DONE
**Files Modified:**
  - `core/boot/wiring.py` — inserted voice-interrupt coordination ahead of `router.on_speech`, routed `resume_listening` through the new coordinator, and replaced direct TTS `speech_partial` handling with centralized interrupt logic
  - `core/ipc/interrupt_manager.py` — made the global interrupt manager safe for both in-process `AsyncEventBus` and distributed ZMQ buses, and switched broadcasts to the fast emit path when available
  - `core/async_event_bus.py` — raised `resume_listening`, `user_interrupt`, and `INTERRUPT_ALL` to high-priority events so barge-in work is dispatched ahead of normal background traffic
  - `services/tts_worker.py` — awaited async TTS `stop()` during distributed interrupt handling instead of dropping the coroutine
**Files Created:**
  - `voice/interrupt_handler.py` — new voice interrupt coordinator for STT partial/final detection, TTS stop, brain preemption, and clean state handoff back to `LISTENING`
  - `tests/test_voice_interrupt.py` — focused regression coverage for speaking/thinking interrupts, partial-status heuristics, and `resume_listening` emission
**What Changed:**
  - Added a dedicated `VoiceInterruptHandler` so barge-in is no longer just "stop TTS if `speech_partial` happens"; it now coordinates global interrupt broadcast, local-brain preemption, TTS stop, and legal state transition back to `LISTENING`
  - Fixed the stale-state failure mode where ATOM could keep processing a new utterance while still technically in `SPEAKING`, which blocked legal `THINKING` transitions and could cause follow-up TTS output to be ignored
  - Moved interrupt preparation ahead of `router.on_speech(...)` for `speech_final`, so even if early partial detection does not fire, the final recognized utterance still forces a clean interrupt before the new turn is routed
  - Added a lightweight partial-text heuristic that ignores STT arming status (`Listening...`) but treats `Processing...` or real partial transcript text as a barge-in signal, avoiding the old self-cutoff risk
**Test Result:**
  - `python3 -m py_compile core/boot/wiring.py core/async_event_bus.py core/ipc/interrupt_manager.py voice/interrupt_handler.py services/tts_worker.py tests/test_voice_interrupt.py` passed
  - `python3 -m tests.test_voice_interrupt` passed:
    - speaking interrupt stopped TTS, preempted the brain, and returned to `LISTENING`
    - plain `Listening...` status did not trigger a false interrupt
    - `Processing...` emitted `resume_listening` through the voice-interrupt path
    - new user speech cleanly preempted `THINKING` before routing
  - `python3 -m tests.test_state_machine` passed, including the existing barge-in lifecycle checks
  - IDE lint check returned no diagnostics for all edited files
**Issues Found:**
  - The current faster-whisper path still does not expose a separate low-level VAD signal, so in that path the interrupt fires when STT reaches `Processing...` (captured audio) or `speech_final`, not at the very first phoneme like a dedicated VAD loop would
**Notes for Next Step:**
  - The natural continuation is the streaming-response step: once interruption is reliable, streamed sentence chunks into TTS will make ATOM feel much more immediate
  - If future profiling shows barge-in still feels late on the faster-whisper path, the next refinement is adding a true low-level voice-activity detector instead of relying on STT partial/final milestones

### Step 3.9 — Cold Start Optimization
**Date:** 2026-04-11
**Status:** DONE
**Files Modified:**
  - `main.py` — replaced the old one-off local-brain warm-up block with a boot-time cold-start optimizer, restored last known context after wiring, and persisted a next-boot snapshot on shutdown
  - `cursor_bridge/local_brain_controller.py` — extended MLX warm-up to accept explicit model-role targets so startup can preload the fast role instead of only the default role
  - `core/memory_engine.py` — added public helpers for eager embedding-model warm-up and hot-command lookup from successful interaction history
**Files Created:**
  - `core/boot/cold_start.py` — new cold-start orchestrator for fast-role preload, embedding warm-up, session restore, hot command-cache seeding, and lightweight system-context replay
  - `tests/test_cold_start.py` — focused regression coverage for restore/cache/preload behavior and persisted snapshot writing
**What Changed:**
  - Added a dedicated `ColdStartOptimizer` that runs during boot and preloads the fast MLX role, eagerly warms the embedding model, restores a short slice of recent conversation history, and seeds the command cache from real successful commands
  - Kept the implementation aligned with the existing architecture instead of inventing new stores: startup now reuses `ConversationMemory`, `MemoryEngine` interaction history, the existing command cache singleton, and `PersistenceManager`
  - Added a lightweight boot snapshot (`logs/cold_start_snapshot.json`) so shutdown persists recent conversation pairs plus last known system context, and the next boot can replay that context before the first periodic health poll lands
  - Closed the known startup gap noted after earlier Phase 3 work: boot no longer only warms the default MLX role, and simple first-query traffic now has the fast role available immediately when cold-start warm-up succeeds
**Test Result:**
  - `python3 -m py_compile main.py core/boot/cold_start.py core/memory_engine.py cursor_bridge/local_brain_controller.py tests/test_cold_start.py` passed
  - `python3 -m tests.test_cold_start` passed:
    - session restore repopulated `ConversationMemory`
    - top commands seeded the shared command cache, including `info:*` reuse keys
    - fast-role warm-up targeted `model_role="fast"`
    - shutdown snapshot persistence wrote the expected conversation/system payload
  - IDE lint check returned no diagnostics for all edited files
**Issues Found:**
  - The focused test environment cannot import the full intent package because `psutil` is missing there, so the new cold-start test uses a tiny local intent-result stub instead of importing the live package tree
**Notes for Next Step:**
  - Phase 4 can still proceed; this step mainly closes the startup-latency gap left behind after MLX routing/latency work
  - If first-response profiling later shows the primary path is still too cold for your workload, the next tuning decision is whether to preload both MLX roles at boot or continue warming only the fast role

### Step 2.5 — Integration Test (Abbreviated by User Request)
**Date:** 2026-04-11
**Status:** DONE
**Files Modified:** none
**Files Created:** none
**What Changed:**
  - Replaced the planned 30-minute soak with a lightweight validation pass per user direction
  - Ran `scripts/v7_stress_test.py --n 1000` to pressure the priority scheduler
  - Ran `scripts/v7_chaos_test.py` to validate RecoveryManager replay and worker-crash handling
  - Ran a live `main.py` runtime monitor and sampled resident memory while watching for full-process crash / leak behavior
**Test Result:**
  - `scripts/v7_stress_test.py --n 1000` passed: `completed=1000`, `jobs_dropped=0`, queue drained successfully
  - `scripts/v7_chaos_test.py` passed: ring replay worked and worker-crash handling executed cleanly
  - Abbreviated live runtime stayed up for ~6.5 minutes until manually stopped after the user shortened the testing scope
  - No full-process crash observed
  - Sampled tree RSS stayed effectively flat: ~118.1 MB at 10s, ~104.5 MB at 251s, ~105.4 MB at 372s
  - No obvious memory leak signal in the abbreviated sample window
**Issues Found:**
  - `voice/stt_async.py`: missing `speech_recognition` dependency triggers repeated STT failures / backoff
  - `core/macos/fs_watcher.py`: `emit()` argument collision (`multiple values for argument 'event'`)
  - `core/cognitive/proactive_engine.py`: `_on_system_light_scan()` references missing `_last_scan`
  - TTS runtime is degraded in this environment (`pygame` / mixer / Edge TTS init warnings), but the process stayed alive and fell back without crashing
**Notes for Next Step:**
  - Full 30-minute soak was intentionally skipped per user request; this step should be treated as abbreviated validation, not final hardening
  - The important Phase 2 signal is that ATOM stayed alive despite repeated component failures, and RSS did not trend upward in the short run
  - The runtime defects found here are good cleanup candidates, but they are not blockers for starting Phase 3

### Step 2.4 — Memory Pressure Protection
**Date:** 2026-04-11
**Status:** DONE
**Files Modified:**
  - `core/memory_engine.py` — added pressure-mode hooks, vector-result cap, and keyword-only degradation when unified memory is high
  - `core/rag/rag_engine.py` — capped RAG snippets, added low-memory mode, and clears/embed-unloads on pressure transitions
  - `core/rag/rag_cache.py` — added cache clear helpers so RAG can shed retrieval/embed caches under pressure
  - `brain/memory_graph.py` — added hard node cap with pruning, semantic-result cap, pressure-mode query reduction, and vector cleanup for pruned nodes
  - `cursor_bridge/local_brain_controller.py` — forwards memory pressure updates to the attached RAG engine and MemoryGraph
  - `main.py` — wires periodic `silicon_stats_update` handling to MemoryEngine, LocalBrainController, and embedding unload signaling
  - `core/silicon_governor.py` — aligned default memory warning threshold to 85% unified-memory usage
  - `core/config_schema.py` — added config schema keys for memory/RAG pressure thresholds and caps
  - `config/settings.json` — set explicit M5-friendly defaults: vector result cap, graph node cap, RAG snippet cap, and 85% pressure threshold
**Files Created:** none
**What Changed:**
  - **Vector result protection:** Semantic memory retrieval now clamps vector fan-out to 5 results max and falls back to keyword-only retrieval when memory pressure mode is active.
  - **RAG snippet protection:** RAG now defaults to 3 snippets max, then degrades to a smaller snippet budget during pressure mode so prompt context does not balloon while unified memory is tight.
  - **MemoryGraph growth cap:** Graph writes now enforce a hard node ceiling (default 1000). Low-value / old episodic nodes are pruned first, and their vector entries are removed too.
  - **Periodic memory check:** Instead of adding a second poller, ATOM now uses the existing periodic `silicon_stats_update` telemetry stream from `SiliconGovernor` to push live memory percentages into the memory-heavy modules.
  - **Embedding shedding:** On pressure entry, semantic-heavy modules clear caches and shut down the shared embedding engine so it can be reloaded later instead of staying resident during high-memory periods.
  - **Pressure recovery:** When memory usage drops below the relief threshold, normal retrieval limits are restored automatically.
**Test Result:**
  - `python3 -m py_compile` passed for all changed files: memory engine, RAG engine/cache, MemoryGraph, local brain, main, config schema, silicon governor
  - Focused smoke test passed:
    - MemoryGraph prunes to configured node cap
    - MemoryGraph pressure mode reduces query result count
    - MemoryEngine enters/exits pressure mode with reduced retrieval budget
    - RAG enters/exits pressure mode with reduced snippet budget
  - Zero lint errors across all modified files
**Issues Found:**
  - The local embedding model did not have a pre-existing bus-driven unload path for in-process memory pressure, so the pressure hooks now call embedding shutdown directly and also emit the inference-guard unload signal for coordination
  - `MemoryGraph` originally had no built-in node cap, so pruning policy had to be introduced in this step (episodic / low-importance / low-access nodes pruned first)
**Notes for Next Step:**
  - Step 2.5 should run the full 30-minute integration test and specifically watch unified-memory behavior around repeated RAG + LLM + TTS usage
  - Pressure handling now depends on `SiliconGovernor` telemetry being enabled, which it is in the current config
  - If long-run testing still shows memory creep, the next likely hotspots are document ingestion/vector persistence growth and prompt-history accumulation rather than hot-path retrieval fan-out

### Step 2.3 — RuntimeWatchdog Upgrade (Per-Module Execution Budgets)
**Date:** 2026-04-11
**Status:** DONE
**Files Modified:**
  - `core/runtime_watchdog.py` — upgraded from state-dwell watchdog to active budget enforcer with intent/cache/RAG/LLM/TTS/tool budgets, async helpers, and TTS timeout supervision
  - `core/router/router.py` — wired RuntimeWatchdog into intent classification and LLM cache lookup so timeouts force fallback/skip behavior instead of hanging the turn
  - `cursor_bridge/local_brain_controller.py` — capped RAG budget via watchdog, wrapped LLM streaming with 30s timeout handling, and moved tool execution onto async path for watchdog enforcement
  - `core/reasoning/action_executor.py` — added `execute_async()` so tool calls can use Router’s async dispatch path and participate in watchdog timeouts
  - `main.py` — attaches RuntimeWatchdog to Router and LocalBrainController at startup
  - `core/config_schema.py` — added explicit config keys for intent/cache/RAG/LLM/TTS/tool watchdog budgets
**Files Created:** none
**What Changed:**
  - **Active watchdog helpers:** `RuntimeWatchdog` now exposes `run_sync()` and `run_async()` wrappers that apply per-stage budgets and return structured timeout metadata instead of just logging after the fact.
  - **Intent/cache budgets:** Router intent classification is now budgeted at 50ms and cache lookup at 100ms. Timeout behavior degrades cleanly: forced `fallback` intent or skip-cache path.
  - **RAG cap:** Local brain now clamps adaptive RAG time budgets to the watchdog ceiling (500ms max), preserving the existing late-result flow while preventing runaway retrieval budgets.
  - **LLM timeout recovery:** Local brain streaming generation is wrapped in a 30s watchdog budget. On timeout, the watchdog emits `llm_error`, requests preemption, unloads the local model for reset, and triggers the normal recovery burst.
  - **TTS timeout supervision:** Watchdog now listens for `response_ready` / `partial_response` and `tts_complete` to detect TTS synthesis stalls. If speech exceeds budget (15s default), it skips audio and returns to listening.
  - **Tool execution path:** ReAct tool calls now use `ActionExecutor.execute_async()` so watchdog budgets can wrap tool execution without pushing router/bus work onto unsafe background threads.
  - **Startup recovery fix:** Smoke testing exposed that the old cooldown initialization could suppress the first recovery burst right after startup because `time.monotonic()` starts near zero in this environment. `_last_recovery` now initializes relative to the current monotonic clock so early watchdog recoveries still fire.
**Test Result:**
  - `python3 -m py_compile` passed for all changed files: watchdog, router, local brain, action executor, main, config schema
  - Watchdog smoke test passed:
    - sync budget timeout (`intent_engine`) returns fallback
    - async budget timeout (`llm_inference`) returns timeout default
    - watchdog emits `llm_error` + `watchdog_recoveries` on LLM timeout
  - Zero lint errors across all modified files
**Issues Found:**
  - The original cooldown logic in `RuntimeWatchdog` could suppress the first recovery burst during early process startup on this Python build; fixed during this step
  - Tool timeout enforcement is strongest on async/slow-dispatch paths. If future inline router actions prove capable of blocking unexpectedly, they may need deeper async refactoring for fully preemptive budgets
**Notes for Next Step:**
  - Step 2.4 should now add memory pressure protection around vector results, RAG snippet counts, and memory graph growth
  - The new watchdog config knobs live under `performance.*` and default to the M5 evolution-plan budgets
  - RAG already has an adaptive early-return path; memory protection should complement it rather than duplicate timeout logic

### Step 2.2 — Error Isolation Wrappers (Critical Runtime Boundaries)
**Date:** 2026-04-11
**Status:** DONE
**Files Modified:**
  - `core/router/router.py` — added a top-level router error boundary around `on_speech()` with timeline logging, metrics increment, fallback response, and state recovery hook
  - `cursor_bridge/local_brain_controller.py` — split `on_query()` into wrapper + `_on_query_impl()` so all unhandled LLM/ReAct failures are isolated and converted into `llm_error` + user-safe fallback
  - `core/reasoning/action_executor.py` — wrapped the full `execute()` pipeline so registry/security/validation failures return `ActionResult` instead of escaping
  - `core/boot/wiring.py` — added `_guard_handler()` for critical event registrations and fixed `shutdown_event` plumbing by passing it explicitly from `main.py`
  - `voice/stt_async.py` — added outer runtime guards around `start_listening()`, `on_state_changed()`, and `shutdown()`
  - `voice/tts_edge.py` — added outer runtime guards around `speak()`, `on_response()`, `on_partial_response()`, and `shutdown()`
  - `main.py` — passes `shutdown_event` into `wire_events()`
**Files Created:** none
**What Changed:**
  - **Router boundary:** If routing explodes at the top level, ATOM now logs the fault, increments error metrics, records an error event to the timeline, tells Boss something went wrong, and triggers state recovery instead of dropping the task.
  - **Local brain boundary:** Any failure in prompt build, ReAct loop, RAG, tool execution, or streaming now gets caught by the `on_query()` wrapper. ATOM emits `llm_error` and responds with a safe fallback instead of letting the query task die.
  - **Action executor boundary:** Tool execution was already protected at dispatch time, but registry / validation / security-path failures could still leak. The outer wrapper now collapses those into structured `ActionResult` failures.
  - **Guarded event wiring:** Critical bus registrations now go through `_guard_handler()` so component handler failures are consistently logged and counted at the wiring boundary.
  - **STT isolation:** Mic/listen state transitions and listen loop startup failures now back off and emit a recovery timeout instead of crashing the speech pipeline task.
  - **TTS isolation:** Response playback and streaming entry points now fall back to on-screen text and emit `tts_complete` when needed so ATOM does not get stuck in `SPEAKING`.
  - **Wiring bug fix:** `core/boot/wiring.py` referenced `shutdown_event` without owning it. The event is now explicitly passed from `main.py`, removing a hidden runtime failure path in the extracted wiring layer.
**Test Result:**
  - `python -m py_compile` passed for all changed files: router, local brain, action executor, wiring, STT, Edge TTS, main
  - Forced-failure smoke tests passed:
    - Router top-level failure → caught, logged, no crash
    - Local brain top-level failure → caught, logged, no crash
    - ActionExecutor internal failure → returns failed `ActionResult`, no exception escape
    - STT top-level failure → caught, error counter increments, no crash
    - Edge TTS top-level failure → caught, fallback path used, no crash
  - Zero lint errors across all modified files
**Issues Found:**
  - AsyncEventBus already isolates handler exceptions, so `wiring.py` wrappers focus on critical entry points rather than every single event registration
  - STT smoke test used a forced missing attribute (`_executor`) on a manually constructed object to validate the outer boundary; the important result is that the wrapper contained the failure
**Notes for Next Step:**
  - Step 2.3 should add per-module execution budgets to `core/runtime_watchdog.py`
  - The new error boundaries reduce crash risk, but timeout enforcement still needs the watchdog step to turn hangs into recoveries
  - `core/boot/wiring.py` now depends on `shutdown_event` being passed explicitly; `main.py` already does this

---

### Step 2.1 — Create core/cognitive_kernel.py (Cognitive Kernel)
**Date:** 2026-04-11
**Status:** DONE
**Files Created:**
  - `core/cognitive_kernel.py` (~310 lines) — Central brain coordinator. 5 execution paths: DIRECT (intent/quick-reply, sub-5ms), CACHE (cached LLM response, sub-10ms), QUICK (fast 1.7B model, 80-150ms), FULL (4B model thinking OFF, 300-600ms), DEEP (4B model thinking ON + RAG, 800-2000ms).
**Files Modified:**
  - `main.py` — Imports CognitiveKernel + ExecPath. Creates cognitive_kernel with full dependency wiring (config, bus, intent_engine, cache, metrics, inference_guard, silicon_governor, state_manager). Calls `router.attach_cognitive_kernel()`.
  - `core/router/router.py` — Added `_cognitive_kernel` slot. Added `attach_cognitive_kernel()` method. LLM fallback path (`_handle_llm_fallback`) now calls `cognitive_kernel.route()` and passes `query_plan` in the `cursor_query` event payload.
**What Changed:**
  - **QueryPlan dataclass:** Describes how to process a query — path, model, RAG/memory flags, thinking toggle, latency budget, pre-resolved responses.
  - **5 execution paths:** DIRECT (skip LLM entirely), CACHE (cached response), QUICK (fast brain), FULL (primary brain), DEEP (primary brain + thinking + RAG). Each has a calibrated latency budget.
  - **System-aware routing:** Reads SiliconGovernor for thermal pressure, memory usage, battery state. Degrades to QUICK path on low battery (<20%), thermal throttling, or high memory pressure (>85%).
  - **Circuit breakers:** Per-module failure tracking (intent, cache, llm_quick, llm_full, rag). After 3 failures → circuit opens for 30s → module bypassed. Auto-recovers on success.
  - **LLM model selection:** Routes simple queries to Qwen3-1.7B (quick_model), complex queries to Qwen3-4B (full_model). Thinking mode ON only for DEEP path.
  - **Metrics integration:** Records routing latency via MetricsCollector. Emits `cognitive_route` bus event with path, model, reason, timing.
  - **Diagnostics API:** `get_diagnostics()` returns total routed, path distribution percentages, LLM skip rate, circuit breaker states.
  - **User overrides:** FAST → QUICK path, DEEP → DEEP path with full RAG + thinking, SMART → FULL path.
  - **Router integration:** Router's LLM fallback path now gets a QueryPlan attached to the `cursor_query` event. Future phases will use this to select which model to invoke.
**Test Result:**
  - All imports clean (95ms)
  - 5 execution paths verified: DIRECT, CACHE, QUICK, FULL, DEEP
  - Intent-matched commands (open chrome, what time, play music) → DIRECT path (skip LLM)
  - Greetings (hello boss, hi) → DIRECT path via quick_reply
  - Simple queries (set volume to 50) → QUICK path (fast model)
  - Complex queries (explain quantum computing) → FULL path (primary model + RAG)
  - Very long recall queries (>120 chars, complex) → DEEP path with thinking ON
  - Cache path: cached queries served from cache
  - Circuit breaker: opens after 3 failures, bypasses module for 30s
  - LLM skip rate: 80% on test set (4/5 common queries skip LLM entirely)
  - Full integration test with IntentEngine + CacheEngine: all paths correct
  - Zero lint errors
**Issues Found:**
  - Python 3.9 doesn't support `@dataclass(slots=True)` — removed, using regular `@dataclass`
  - Router integration is lightweight (passes QueryPlan as metadata in cursor_query event). Full model-selection integration requires Phase 3 (MLX dual-model loader).
**Notes for Next Step:**
  - Step 2.2 adds error isolation wrappers to 6 critical files
  - When Phase 3 (MLX) is done, the `cursor_query` handler in LocalBrainController should read `query_plan.model` to select Qwen3-1.7B vs Qwen3-4B
  - The `query_plan.thinking` flag maps to Qwen3-4B's thinking mode toggle
  - The `query_plan.use_rag` and `query_plan.use_memory` flags should gate RAG/memory retrieval

---

### Native macOS Stack — Apple Framework Integration (pyobjc)
**Date:** 2026-04-09
**Status:** DONE
**Files Created:**
  - `voice/stt_macos.py` (~280 lines) — native STT via SFSpeechRecognizer + AVAudioEngine. On-device, Neural Engine, ~50ms commands, built-in wake word, HW noise suppression via Voice Processing I/O.
  - `core/macos/__init__.py` — macOS-native modules package
  - `core/macos/fs_watcher.py` (~200 lines) — FSEvents kernel-level file watcher. Near-zero CPU, same mechanism as Spotlight.
**Files Modified:**
  - `voice/media_watcher.py` — REWRITTEN. Replaced winsdk (broken) with macOS AppleScript queries for Spotify, Music, and browser media. Also queries album, duration, position. Windows winsdk path preserved.
  - `voice/tts_macos.py` — REWRITTEN. Dual backend: NSSpeechSynthesizer (no subprocess, ~0ms spawn, instant barge-in 4.4ms) with `say` subprocess fallback. Premium/enhanced voice auto-detection. 184 voices available.
  - `context/screen_reader.py` — UPGRADED. Apple Vision framework OCR (VNRecognizeTextRequest) as primary on macOS. Neural Engine powered, ~109ms avg (vs EasyOCR ~3000ms). 89 text regions from full-screen capture. EasyOCR preserved as cross-platform fallback.
  - `core/embedding_engine.py` — Added MPS (Apple Silicon GPU) device detection. `auto` → MPS when torch.backends.mps.is_available().
  - `main.py` — Wired FSWatcher (watches ~/Desktop, ~/Downloads, ~/Documents), updated screen_reader log to show backend name, added FSWatcher cleanup on shutdown.
  - `requirements.txt` — v19 "Native Edition": added 8 pyobjc packages (~18MB total), moved 7 third-party voice deps to optional comments (faster-whisper, SpeechRecognition, PyAudio, edge-tts, pygame, openwakeword, easyocr).
**Packages Installed:**
  - pyobjc-core 11.1
  - pyobjc-framework-Cocoa 11.1 (Foundation + AppKit)
  - pyobjc-framework-AVFoundation 11.1 (AVAudioEngine mic)
  - pyobjc-framework-Speech 11.1 (SFSpeechRecognizer STT)
  - pyobjc-framework-Vision 11.1 (VNRecognizeTextRequest OCR)
  - pyobjc-framework-Quartz 11.1 (CGImage for Vision)
  - pyobjc-framework-FSEvents 11.1 (kernel file watcher)
  - pyobjc-framework-NaturalLanguage 11.1 (NLEmbedding)
  - pyobjc-framework-CoreML 11.1 (CoreML inference)
  - pyobjc-framework-CoreAudio 11.1 (audio)
  - pyobjc-framework-CoreMedia 11.1 (media)
**What Changed:**
  - **Philosophy:** Same as MLX for LLM — use what macOS already has loaded in RAM. pyobjc bridges Python to Apple frameworks that are ALREADY in memory as part of the OS.
  - **STT:** SFSpeechRecognizer replaces faster-whisper + SpeechRecognition + PyAudio (3 deps → 1). On-device, Neural Engine, ~50ms for commands. Also replaces OpenWakeWord — built-in keyword detection.
  - **TTS:** NSSpeechSynthesizer replaces `say` subprocess — no process spawn, direct API call, instant barge-in (4.4ms vs 6ms+). 184 voices available. Premium neural voices available when user downloads them (System Settings > Accessibility > Spoken Content).
  - **Media Watcher:** AppleScript replaces winsdk. Queries Spotify, Music, and browser media. Works immediately, zero deps. Previously completely broken on macOS.
  - **Screen OCR:** Vision framework replaces EasyOCR. 109ms avg vs ~3000ms. 89 text regions from screen. Neural Engine powered. No model download needed.
  - **File Watcher:** FSEvents — kernel-level, no polling, near-zero CPU. Watches Desktop/Downloads/Documents for proactive awareness.
  - **Embeddings:** MPS device auto-detection for Apple Silicon GPU acceleration.
  - **RAM impact:** pyobjc adds ~18MB. The frameworks it replaces would have needed ~2.8GB (faster-whisper ~550MB, PyTorch for sentence-transformers ~2GB, EasyOCR ~200MB, OpenWakeWord ~80MB).
**Test Result:**
  - 6/6 modules PASSED verification:
    - Media Watcher: macOS native, 520ms poll (AppleScript)
    - TTS NSSpeech: 184 voices, Eddy (US) auto-selected, 4.4ms barge-in
    - STT SFSpeech: available, on-device supported, auth=notDetermined (needs user grant)
    - Screen OCR: Vision framework, 89 regions, 109ms avg, 1911 chars extracted
    - FSEvents: 2 events captured from test file creation
    - Embedding MPS: auto→cpu (correct, torch not installed), mps→mps
  - All imports clean (295ms for all 6 modules)
  - All public APIs backward compatible (12 TTS methods, 5 ScreenReader methods)
  - Zero lint errors across all 7 modified/created files
  - TTS live speech: "Hello Boss, I am ATOM. All native modules are operational." — spoke successfully
**Issues Found:**
  - STT auth is `notDetermined` — first use will trigger macOS authorization dialog. User must grant Speech Recognition + Microphone permissions.
  - No premium/enhanced TTS voices installed on this Mac. User can download them: System Settings > Accessibility > Spoken Content > Manage Voices. When installed, the voice selector auto-picks premium British male voices.
  - FSEvents callback uses bus.emit which may need loop thread safety if bus is asyncio-only. Current implementation handles this.
  - Media Watcher AppleScript poll takes ~520ms (osascript subprocess). Acceptable at 3s intervals.
**Notes for Next Step:**
  - Step 2.1 (Cognitive Kernel) is unchanged — proceed as planned
  - STT native will need integration into the voice pipeline loop (currently a standalone module)
  - TTS native is already wired via MacOSTTSAsync in main.py — works as drop-in replacement
  - For optimal TTS quality, download premium voices: System Settings > Accessibility > Spoken Content > System Voice > Manage Voices > search "Daniel" or "Evan" > Download
  - Phase 3 (MLX): mlx-whisper can be a secondary STT for long-form/multilingual speech alongside SFSpeechRecognizer for commands

---

### Step 1.9 — Verify zero crashes on macOS (PHASE 1 COMPLETE)
**Date:** 2026-04-09
**Status:** DONE
**Files Modified:** none (verification only)
**Files Created:** none
**What Changed:** nothing — this was a verification step
**Test Result:**
  - ATOM boots on macOS in ~1.5 seconds
  - **Zero crashes, zero unhandled exceptions**
  - All Silicon Refactoring modules loaded correctly:
    - `InferenceGuard`: "Unified Memory mode (Apple Silicon)"
    - `SiliconGovernor`: "Apple M5 (10 GPU cores, Unified Memory)"
    - `GPUStallWatchdog`: started
  - All core systems initialized: config, security, state, intent, router, brain, RAG, scheduler
  - All JARVIS modules: platform, scanner, indexer, media watcher, owner understanding
  - Cognitive layer: 8 modules (dream + curiosity) all started
  - Web dashboard: http://127.0.0.1:8765/
  - System health: 75/100
  - Boot greeting spoke successfully
  - Expected optional dep warnings (not crashes): speech_recognition, pygame, winsdk, keyboard, chromadb, cryptography
**Issues Found:**
  - STT: `speech_recognition` not installed — backoff loop works correctly (2s → 4s → 8s → 16s → 30s cap)
  - TTS: `pygame` not installed — Edge TTS skipped gracefully, text shown on screen
  - Media Watcher: `winsdk` not installed — disabled gracefully (expected, Windows-only)
  - Integrity check: 88 violations (new pip packages in .venv — cosmetic, not real)
**Notes for Next Step:**
  - **PHASE 1 IS COMPLETE.** ATOM boots and runs on M5 without crashes.
  - Phase 2 starts with Step 2.1: Create `core/cognitive_kernel.py`
  - Voice pipeline needs: SpeechRecognition, PyAudio, faster-whisper, pygame (install when ready)
  - LLM inference needs a GGUF model file in models/

---

### Step 1.8A — Silicon Refactoring (Apple Silicon-native compute layer)
**Date:** 2026-04-09
**Status:** DONE
**Files Deleted:**
  - `core/gpu_governor.py` (331 lines) — multi-backend GPU monitor with NVIDIA/pynvml paths
  - `core/gpu_resource_manager.py` (387 lines) — VRAM budget model, slot allocation, load grant tokens
  - `core/gpu_execution_coordinator.py` (593 lines) — VRAM admission control, fragmentation estimation, priority queue
**Files Created:**
  - `core/silicon_governor.py` (~145 lines) — Apple Silicon-only hardware monitor. Single backend via AppleSiliconMonitor. Emits silicon_stats_update + legacy gpu_stats_update events for backward compat. Thermal + memory pressure warnings.
  - `core/inference_guard.py` (~155 lines) — Simplified model lifecycle manager. Model slot tracking, Unified Memory pressure check (<90%), idle unload policy, power modes. No VRAM budgets, no slot allocations, no fragmentation heuristics.
**Files Modified:**
  - `core/gpu_watchdog.py` — removed CUDA reset (`_maybe_cuda_reset`, `allow_cuda_reset`), removed `_config`/`_allow_reset` slots. Pure stall detection only. 89→73 lines.
  - `main.py` — replaced GPUGovernor → SiliconGovernor, GPUResourceManager → InferenceGuard, removed GPUExecutionCoordinator from RAG wiring. Updated init, start, and shutdown sections.
  - `cursor_bridge/local_brain_controller.py` — replaced `_gpu_resource_mgr` → `_inference_guard`, added `attach_inference_guard()` with legacy `attach_gpu_resource_manager()` shim.
  - `core/rag/rag_engine.py` — removed GPUExecutionCoordinator TYPE_CHECKING import, removed coordinator submit_task embed path (~35 lines), simplified `_get_or_embed` to sync/async without coordinator queue.
  - `services/llm_worker.py` — import events from inference_guard instead of gpu_resource_manager.
  - `services/stt_worker.py` — import events from inference_guard instead of gpu_resource_manager.
  - `services/gpu_cognition_worker.py` — import InferenceGuard instead of GPUResourceManager, use attach_inference_guard.
  - `core/apple_silicon_monitor.py` — updated docstrings (GPUGovernor → SiliconGovernor).
  - `scripts/v7_long_run.py` — replaced get_nvml_vram_mb import with get_apple_silicon_memory_mb.
  - `requirements.txt` — removed comtypes, winsdk, truststore (Windows-only), removed pynvml comment, added Metal build instructions.
**What Changed:**
  - Removed ~1,311 lines of NVIDIA/CUDA/multi-backend dead code across 3 deleted files
  - Created ~300 lines of clean Apple Silicon-native code across 2 new files
  - Net reduction: ~1,000 lines removed from the compute stack
  - Old stack: 5 files, ~1,765 lines, 3 backends (NVIDIA + Apple Silicon + CPU fallback), 3 state dataclasses
  - New stack: 3 files, ~370 lines, 1 backend (Apple Silicon), 1 state class (AppleSiliconStats)
  - Event names preserved for backward compatibility (gpu_stats_update, gpu_thermal_warning, gpu_memory_warning, v7_gpu_* events)
  - Legacy `attach_gpu_resource_manager()` shim preserved in LocalBrainController
**Test Result:**
  - All 4 new modules import correctly (silicon_governor, inference_guard, apple_silicon_monitor, gpu_watchdog)
  - All 3 deleted modules confirmed unimportable (ImportError)
  - SiliconGovernor: Apple M5, 10416/16384MB Unified Memory
  - InferenceGuard: memory_available=True (63.6% usage, under 90% threshold)
  - Consumer imports: llm_worker, stt_worker, gpu_cognition_worker all OK
  - Zero lint errors across all 9 modified files
**Issues Found:**
  - None — all changes are clean
**Notes for Next Step:**
  - Step 1.9: run main.py, verify zero crashes
  - The event names are backward-compatible (existing bus listeners still work)
  - Phase 2 Cognitive Kernel can use InferenceGuard directly (no further consolidation needed)

---

### Step 1.8 — Install llama-cpp-python with Metal
**Date:** 2026-04-09
**Status:** DONE
**Files Modified:** none (pip install only)
**Files Created:** none
**What Changed:**
  - Installed cmake 4.3.1 via pip (Homebrew unavailable)
  - Built llama-cpp-python 0.3.2 from source with `CMAKE_ARGS="-DGGML_METAL=on"`
  - Metal shader library embedded directly into .dylib (GGML_METAL_EMBED_LIBRARY)
  - Build linked against Metal.framework + MetalKit.framework
  - Architecture: arm64-apple-darwin25 (native M5)
**Test Result:**
  - `import llama_cpp` — OK, version 0.3.2
  - Build log confirms GGML_METAL_EMBED_LIBRARY flag
  - Metal and MetalKit frameworks linked
**Issues Found:**
  - None
**Notes for Next Step:**
  - Step 1.8A performs Silicon Refactoring (strip NVIDIA dead code)
  - When a GGUF model is placed in models/ and n_gpu_layers > 0, inference runs on M5 GPU via Metal

---

### GPU Stack Cleanup — Dead Code Removal (Option A)
**Date:** 2026-04-09
**Status:** DONE
**Files Deleted:**
  - `core/gpu_scheduler.py` (104 lines) — dead code: created in main.py but `submit_gpu_task()` never called by any file. Was assigned to `_` to suppress unused variable warning.
  - `brain/gpu_pipeline.py` (107 lines) — 107-line file wrapping 1 useful line (`refresh_vram()`). `run_retrieval_stage()` and `build_context_parallel()` never called from outside the class.
**Files Modified:**
  - `main.py` — removed GPUScheduler import, creation, and `_ = (gpu_sched_v7, ...)` suppression line
  - `cursor_bridge/local_brain_controller.py` — replaced `self._gpu_pipeline` (GPUPipeline wrapper) with direct `self._gpu_resource_mgr` reference. `refresh_gpu_budget()` wrapper replaced with direct `refresh_vram()` call.
**Files Created:**
  - `.cursor/rules/GPU_STACK_RULE.md` — GPU stack architecture rule with DO/DON'T guidelines and Phase 2 consolidation plan
**What Changed:**
  - Removed 211 lines of dead/unnecessary code across 2 deleted files
  - Eliminated GPUPipeline indirection layer (3 levels of wrapping reduced to 1 direct call)
  - Documented target architecture for Phase 2: merge gpu_resource_manager + gpu_execution_coordinator into ~150-line InferenceGuard
  - Created guardrail rules: no new GPU state dataclasses, no VRAM budgets, no pynvml imports, no torch.cuda without MPS guards
**Test Result:**
  - All 4 remaining GPU modules import and initialize correctly
  - Both deleted modules confirmed unimportable (ImportError)
  - LocalBrainController no longer references gpu_pipeline
  - Zero lint errors
**Issues Found:**
  - None
**Notes for Next Step:**
  - Phase 2 Step 2.1 will do the full consolidation (gpu_resource_manager + gpu_execution_coordinator → InferenceGuard)
  - For now, the remaining files work correctly on Apple Silicon with the patches from Steps 1.2-1.3

---

### Step 1.7 — Fix desktop_control.py for macOS
**Date:** 2026-04-09
**Status:** DONE
**Files Modified:**
  - `core/desktop_control.py` — full rewrite with AppleScript fallback + macOS key mapping
  - `core/security_policy.py` — added macOS hotkey equivalents to SAFE_HOTKEYS
**Files Created:** none
**What Changed:**
  - Rewrote `desktop_control.py` with dual-backend architecture: pyautogui (primary) + AppleScript (macOS fallback)
  - All 11 public functions now work on macOS without pyautogui installed
  - Added `_macos_combo()` — auto-maps `ctrl` → `command`, `alt` → `option`, `win` → `command` on macOS
  - Added `_applescript_keystroke()` — sends keystrokes via `osascript` with modifier support, full key code map (36 keys)
  - Added `_applescript_type()` — types arbitrary text (including Unicode) via System Events keystroke
  - Added `_applescript_scroll()` — scrolls via Option+Arrow key simulation
  - `take_screenshot()` — uses `screencapture` on macOS (7MB Retina PNG), falls back to pyautogui
  - `get_screen_size()` — falls back to system_profiler display resolution when pyautogui unavailable (1710x1107 detected)
  - `type_text()` — uses `gui.write()` (not `typewrite()`) on macOS for proper Unicode handling
  - `hotkey_combo()` — dual label map for both ctrl+* and command+* versions (e.g., both "ctrl+c" and "command+c" → "Copied")
  - `click_center()` / `click_at()` — AppleScript fallback via System Events click
  - Mouse movement (`move_mouse`) — still requires pyautogui (AppleScript can't do relative mouse movement easily)
  - `SAFE_HOTKEYS` in security_policy.py — added 13 macOS equivalents: command+c/v/x/z/a/s/f/p/n/t/tab/shift+tab, option+tab, command+q/w
  - pyautogui availability cached after first check (no repeated import attempts)
**Test Result:**
  - All imports: OK
  - pyautogui: NOT available (expected — not installed)
  - Hotkey mapping: ctrl+c→command+c, ctrl+v→command+v, alt+tab→option+tab, ctrl+shift+t→command+shift+t
  - Screen size: 1710x1107 (via system_profiler fallback)
  - Screenshot: 7MB PNG via screencapture
  - press_key("escape"): "Pressed escape." (via AppleScript key code 53)
  - Security tiers: command+c=safe, command+q=confirm, win+r=block, ctrl+alt+delete=block
  - scroll_down(2): "Scrolled down 2 clicks" (via AppleScript Option+Arrow)
  - Zero lint errors
**Issues Found:**
  - `move_mouse()` and `double_click_center()` still require pyautogui — AppleScript can't do relative mouse movement or precise double-clicks. Phase 5 (Accessibility API) will provide full mouse control.
  - AppleScript click_at/click_center is approximate (System Events click is unreliable for some apps)
  - AppleScript scroll uses Option+Arrow which scrolls by line, not by pixel — different feel than pyautogui
**Notes for Next Step:**
  - Step 1.8 installs llama-cpp-python with Metal: `CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python`
  - Consider installing pyautogui later: `pip install pyautogui` (works on macOS with Accessibility permission)
  - Phase 5.2 (Accessibility API via pyobjc) will provide pixel-perfect mouse/keyboard control

---

### Step 1.6 — Fix system_control.py for macOS
**Date:** 2026-04-09
**Status:** DONE
**Files Modified:**
  - `core/system_control.py` — 6 methods fixed + 5 new methods added for macOS
**Files Created:** none
**What Changed:**
  - `flush_dns()` — removed `sudo` from macOS path (dscacheutil works without it), added `killall -HUP mDNSResponder` for complete flush
  - `get_wifi_networks()` — added macOS via `system_profiler SPAirPortDataType -json`. Parses SSID, channel, band (2/5GHz), security mode, PHY mode. Handles macOS SSID redaction (shows "Hidden Network" with metadata)
  - `list_startup_programs()` — added macOS: scans `~/Library/LaunchAgents/`, `/Library/LaunchAgents/`, `/Library/LaunchDaemons/` for plist files + queries login items via AppleScript
  - `set_power_plan()` — added macOS via `pmset -a lowpowermode 0/1`. Maps plan names: power_saver/low_power → LPM on, balanced/high_performance → LPM off
  - `analyze_temp_files()` — added `~/Library/Caches/` to macOS temp dirs (in addition to /tmp, /var/tmp)
  - `get_full_status()` — fixed GPU label: says "Unified Memory" instead of "VRAM" on macOS
  - NEW: `get_power_status()` — battery %, plugged status, Low Power Mode state, display sleep timeout via pmset
  - NEW: `get_volume()` — reads output/input volume and mute state via `osascript 'get volume settings'`
  - NEW: `set_volume(level)` — sets output volume 0-100 via `osascript 'set volume output volume N'`
  - NEW: `toggle_mute()` — toggles output mute via osascript
  - NEW: `set_brightness(level)` — approximate brightness via key code simulation (144/145 for brightness up/down)
  - All Windows and Linux code paths preserved — zero regression
**Test Result:**
  - `flush_dns()`: success=True, DNS cache flushed
  - `get_wifi_networks()`: 22 networks found (SSIDs redacted by macOS privacy, metadata preserved)
  - `list_startup_programs()`: 0 programs (fresh Mac, correct)
  - `get_power_status()`: Battery 100%, plugged in, low_power_mode=True, display_sleep=10min
  - `get_volume()`: output=56%, input=50%, muted=False
  - `set_volume(50)`: success=True
  - `toggle_mute()`: success=True (muted, then unmuted)
  - `analyze_temp_files()`: 3038 files, 245.1MB (includes ~/Library/Caches/)
  - `get_full_status()`: "...Apple M5 (10-core GPU) with 16 gigs Unified Memory..."
  - `get_system_uptime()`: "System uptime: 2h 26m"
  - Zero lint errors
**Issues Found:**
  - macOS redacts WiFi SSIDs via system_profiler for privacy (all show as `<redacted>`). Network metadata (channel, band, security) still available. Full SSID visibility requires CoreWLAN via pyobjc (Phase 5).
  - `set_brightness()` uses key code simulation — approximate, not absolute. True brightness control requires CoreDisplay private API via pyobjc (Phase 5).
  - `set_power_plan()` with pmset needs sudo for writing. Falls back gracefully with user-friendly message.
**Notes for Next Step:**
  - Step 1.7 fixes `desktop_control.py` — ensure pyautogui works on macOS or add AppleScript fallback
  - The new volume/brightness methods should be wired into the router (system_actions.py) in Phase 1.6 or later
  - Phase 5.1 (AppleScript Engine) will provide deeper macOS control for all these operations

---

### Step 1.5 — Fix screen_reader.py for macOS screenshot
**Date:** 2026-04-09
**Status:** DONE
**Files Modified:**
  - `context/screen_reader.py` — platform-aware screenshot capture + cleanup
**Files Created:** none
**What Changed:**
  - Removed unused `import ctypes` (leftover from Windows code)
  - Added `import subprocess` and `import sys` at module level (subprocess was previously imported inline)
  - Rewrote `_take_screenshot()` with platform-aware fallback chain:
    1. macOS: `screencapture -x -t png <path>` (native, no dependencies, silent capture)
    2. Cross-platform: `PIL.ImageGrab.grab()` (works on Windows + macOS via CoreGraphics)
    3. Windows fallback: PowerShell `System.Windows.Forms.Screen` bitmap capture
  - Added file size validation for screencapture output (rejects zero-byte files from permission failures)
  - Added descriptive warning when Screen Recording permission is missing on macOS
  - Guarded PowerShell fallback behind `sys.platform == "win32"` (was always attempted before)
  - Added `logger.debug` for each successful capture path (helps identify which method was used)
  - All existing public API unchanged: `capture_and_read()`, `get_screen_summary()`, `is_available`, `shutdown()`
**Test Result:**
  - `screencapture` succeeded: 8.7MB PNG at 3420x2214 (Retina 2x of 1710x1107 logical)
  - PIL ImageGrab also works as fallback (6.7MB PNG, same resolution)
  - EasyOCR not installed → graceful fallback to clipboard + window title
  - `capture_and_read()`: returns fallback result with context when OCR unavailable
  - `get_screen_summary()`: "On your screen I can see: Unable to read screen content." (expected without OCR)
  - `shutdown()`: clean teardown
  - Zero lint errors
**Issues Found:**
  - None — `screencapture` works out of the box in Python subprocess context
  - Screen Recording permission may need to be granted to Terminal.app or ATOM's parent process for first use
**Notes for Next Step:**
  - Step 1.6 fixes `system_control.py` for macOS volume, brightness, wifi, startup programs
  - OCR (EasyOCR) is not installed — text extraction from screenshots will be limited to fallback mode until Phase 5 (Vision framework OCR via Neural Engine, Step 5.7)
  - The `screencapture` command supports region capture (`-R x,y,w,h`) and window capture (`-l <window_id>`) for future targeted reads

---

### Step 1.4 — Create voice/tts_macos.py (native macOS TTS)
**Date:** 2026-04-09
**Status:** DONE
**Files Created:**
  - `voice/tts_macos.py` — native macOS TTS engine via `say` command (~310 lines)
**Files Modified:**
  - `main.py` — added `macos_native` engine to TTS selection cascade with auto-detection on macOS
**What Changed:**
  - Created `MacOSTTSAsync` class with full interface compatibility (drop-in replacement for EdgeTTSAsync / TTSAsync)
  - Uses `asyncio.create_subprocess_exec("say", ...)` for non-blocking speech — ~5ms spawn overhead
  - Sentence-by-sentence streaming for natural pacing on multi-sentence responses
  - Barge-in support via subprocess termination — ~6ms stop latency
  - Configurable voice (`macos_voice`, default "Daniel" — British male, closest to JARVIS) and rate (`macos_rate`, default 200 wpm)
  - `list_voices()` utility — parses `say -v '?'` output, returns all available voices with locale
  - Voice validation at `init_voice()` — falls back to system default if configured voice not found
  - Markdown cleanup (code blocks, bold, italic, headers, bullets stripped before speaking)
  - Word limit cap (45 words spoken, overflow shown on screen — same as EdgeTTSAsync)
  - Ack phrase cycling with 26 pre-defined phrases (same variety as EdgeTTSAsync)
  - Governor hooks (`set_postprocess`, `restore_postprocess`) are no-ops — native TTS has no post-processing
  - `refresh_output_device()` is no-op — macOS `say` routes through system audio automatically
  - Full event handler suite: `on_response`, `on_partial_response`, `on_speech_partial` (barge-in)
  - Emits `tts_complete` event (consistent with EdgeTTSAsync and TTSAsync wiring)
  - Auto-detection in main.py: when engine is "sapi" (default) and platform is darwin, auto-switches to "macos_native"
  - Explicit engine configs ("edge", "kokoro") are respected — auto-detection only overrides the SAPI default
  - Dashboard label shows "macOS Native (Daniel)" with configured voice name
**Test Result:**
  - `list_voices()`: 74 voices found, 29 English
  - `_voice_exists("Daniel")`: True
  - `_voice_exists("Samantha")`: True
  - `_voice_exists("NonExistent")`: False
  - All 13 public methods present (speak, speak_ack, stop, on_response, on_partial_response, on_speech_partial, init_voice, shutdown, next_ack_phrase, set_postprocess, restore_postprocess, refresh_output_device)
  - Short phrase ("Hello Boss, I am ATOM."): spoke successfully in ~2.4s
  - Multi-sentence (3 sentences): spoke with natural pauses in ~6s
  - Ack phrase ("Yes, Boss?"): spoke in ~1.9s
  - Barge-in stop: 306ms (300ms sleep + 6ms kill) — near-instant interruption
  - Markdown cleanup: strips **bold**, _italic_, `code` correctly
  - Auto-detection: sapi on darwin → macos_native; edge stays edge; explicit macos_native works
  - Zero lint errors
**Issues Found:**
  - None — fully functional
**Notes for Next Step:**
  - Step 1.5 fixes `screen_reader.py` for macOS screenshot via `screencapture` command
  - The native TTS has no pre-cached audio (unlike EdgeTTSAsync) — `say` is fast enough that caching isn't needed
  - Future enhancement (Phase 3): use native TTS for quick acks (<50 chars), Edge TTS for longer/richer speech
  - Voice quality is functional but not neural-network quality — Daniel voice is clear and JARVIS-like
  - The `say` command uses system audio routing, so AirPods/Bluetooth headphones work automatically

---

### Step 1.3 — Fix GPU resource manager + execution coordinator for Unified Memory
**Date:** 2026-04-09
**Status:** DONE
**Files Modified:**
  - `core/gpu_resource_manager.py` — unified memory model, relaxed admission, Apple Silicon detection
  - `core/gpu_execution_coordinator.py` — `_nvml_snapshot()` + `_torch_mem_mb()` Apple Silicon paths
**Files Created:** none
**What Changed:**
  - `get_nvml_vram_mb()` now tries Apple Silicon first (via `get_apple_silicon_memory_mb()`), then NVML. Same function signature — zero breakage for callers (`v7_long_run.py`, `gpu_cognition_worker.py`, etc.)
  - `GPUResourceManager.__init__` detects Apple Silicon and sets `_unified_memory = True`
  - `enough_vram_for()` on Apple Silicon: checks system memory pressure (<90% used) instead of per-slot VRAM budgets. On NVIDIA: unchanged (slot budget + reserve heuristic)
  - `issue_load_grant()` warning says "memory" instead of "VRAM" on Apple Silicon
  - `emit_status()` includes `unified_memory` flag in the event payload
  - `_nvml_snapshot()` in coordinator tries Apple Silicon first — returns (used, total, cpu%, mem%) using `psutil`
  - `_torch_cuda_mem_mb()` renamed to `_torch_mem_mb()` — tries MPS (Apple Silicon) first via `torch.mps.current_allocated_memory()`, then CUDA. Falls back gracefully when torch is not installed.
  - `refresh_gpu_state()` updated to call `_torch_mem_mb()`
  - Added `_UNIFIED_MEMORY_PRESSURE_THRESHOLD = 0.90` class constant
  - All NVIDIA code paths preserved — zero regression
**Test Result:**
  - `get_nvml_vram_mb()`: (10198, 16384) — Unified Memory stats
  - `GPUResourceManager._unified_memory`: True
  - `enough_vram_for()`: True for all slots (llm, stt, embeddings) at 62% memory usage
  - Memory pressure: 62.25% (well under 90% threshold)
  - `_nvml_snapshot()`: (10206, 16384, 0.0, 62.3) — system memory + CPU%
  - `estimate_fragmentation()`: 0.623 — healthy
  - `GPUState.can_fit(512, 512)`: True
  - `GPUState.can_fit(4096, 512)`: True
  - `_torch_mem_mb()`: (None, None) — torch not installed yet, expected
**Issues Found:**
  - None — all changes are backward compatible
**Notes for Next Step:**
  - Step 1.4 creates `voice/tts_macos.py` — native macOS TTS via `say` command
  - The entire GPU stack (governor, resource manager, execution coordinator) is now Apple Silicon-aware
  - `gpu_watchdog.py` has a CUDA-specific `_maybe_cuda_reset()` — safe as-is (guarded by try/except, no-ops on Apple Silicon)

---

### Step 1.2 — Create apple_silicon_monitor.py + update gpu_governor.py
**Date:** 2026-04-09
**Status:** DONE
**Files Created:**
  - `core/apple_silicon_monitor.py` — Apple Silicon hardware monitoring module (~240 lines)
**Files Modified:**
  - `core/gpu_governor.py` — added Apple Silicon backend, refactored to multi-backend architecture
  - `main.py` — fixed GPU Governor log message for non-NVIDIA platforms
**What Changed:**
  - Created `AppleSiliconMonitor` class: reads Unified Memory (psutil), thermal pressure (pmset -g therm), approximate CPU temp (sysctl), battery status (psutil), CPU utilization as chip load proxy
  - Created `AppleSiliconStats` dataclass with `is_throttled`, `unified_memory`, `summary()` properties
  - Added `is_apple_silicon()` utility function (arm64 + Darwin detection)
  - Added `get_apple_silicon_memory_mb()` drop-in replacement for `get_nvml_vram_mb()` (for Step 1.3)
  - GPU name + core count cached from `system_profiler SPDisplaysDataType -json` (one-time ~350ms)
  - Refactored `GPUGovernor.__init__` to call `_init_backend()` which tries Apple Silicon first, then NVML
  - Separated `get_stats()` into `_get_apple_silicon_stats()` and `_get_nvml_stats()` backends
  - Added `unified_memory` and `thermal_pressure` fields to `GPUStats` dataclass
  - `GPUStats.summary()` now shows "Unified" instead of "VRAM" on Apple Silicon
  - `GPUStats.is_throttled` now checks `thermal_pressure` in addition to temperature
  - `GPUGovernor.is_available` returns True for Apple Silicon (was always False before)
  - `GPUGovernor.start()` now starts monitoring on Apple Silicon (was blocked by NVML check)
  - Apple Silicon thermal threshold defaults to 95°C (vs 85°C for NVIDIA) — Apple chips run hotter by design
  - Monitor loop thermal check uses both temperature AND thermal_pressure for Apple Silicon
  - All NVIDIA code paths preserved — zero regression for Windows/Linux
**Test Result:**
  - `is_apple_silicon()`: True
  - GPU Name: "Apple M5" (10 GPU cores)
  - Unified Memory: 10084/16384 MB (61.5%)
  - Thermal pressure: "nominal"
  - CPU temp: 0.0°C (sysctl sensor not exposed on this OS version — pmset pressure used instead)
  - CPU utilization: 17.6%
  - Battery: 100%, plugged in
  - `is_throttled`: False
  - `GPUGovernor.is_available`: True (was False before this change)
  - Thermal threshold: 95°C (Apple Silicon default)
  - Summary: "GPU: Apple M5 | Util: 18% | Unified: 10084/16384MB (62%) | Temp: 0°C | Power: 0W"
  - Backward compat: default GPUStats still shows "GPU: not available (CPU-only mode)"
**Issues Found:**
  - `sysctl machdep.xcpm.cpu_thermal_level` returns empty on this macOS version — cpu_temp_c is 0.0°C. Not a problem: thermal_pressure from pmset is the reliable source. Temperature field is a "nice to have".
  - `power_watts` is 0 — psutil doesn't expose wattage on macOS. Battery percentage and plugged-in status are available. Actual power draw would require `sudo powermetrics`.
  - `gpu_execution_coordinator.py` also uses `_nvml_snapshot()` — will need similar update in Step 1.3
**Notes for Next Step:**
  - Step 1.3 fixes `gpu_resource_manager.py` — use `get_apple_silicon_memory_mb()` instead of `get_nvml_vram_mb()`, treat all memory as shared (no VRAM eviction on Apple Silicon)
  - The `_nvml_snapshot()` in `gpu_execution_coordinator.py` should also be updated to use Apple Silicon monitor
  - `gpu_watchdog.py` has a `_maybe_cuda_reset()` that's CUDA-specific — safe to leave (guarded by try/except)

---

### Step 1.1 — Fix platform_adapter.py for macOS
**Date:** 2026-04-09
**Status:** DONE
**Files Modified:**
  - `core/platform_adapter.py` — 5 methods fixed/added for macOS
**Files Created:** none
**What Changed:**
  - `_get_gpu_info()` — added `_get_gpu_info_macos()` using `system_profiler SPDisplaysDataType -json`. Returns GPU name (e.g. "Apple M5 (10-core GPU)") and total system RAM as shared memory (Unified Memory).
  - `_get_display_info()` — added `_get_display_info_macos()` using `system_profiler SPDisplaysDataType -json`. Returns display count, resolution with Retina info (e.g. "1710 x 1107 @ 60.00Hz (2880x1864Retina)").
  - `list_services()` — added macOS branch using `launchctl list`. Parses PID/status/label from tab-delimited output. Returns running services with PID.
  - `_fg_window_macos()` — enhanced AppleScript to return app name, PID, and window title (via pipe-delimited string). Gracefully handles Accessibility permission denial for window title.
  - `get_system_summary()` — added `is_apple_silicon` property. GPU line now shows "Unified Memory" instead of "VRAM" on Apple Silicon.
**Test Result:**
  - GPU: "Apple M5 (10-core GPU)" with 16.0GB Unified Memory
  - Display: "1710 x 1107 @ 60.00Hz (2880x1864Retina)", 1 monitor
  - Services: 100 running services listed
  - Foreground window: "Cursor" (PID 1797), title "ATOM_RULES.md — ATOM"
  - System summary now shows full Unified Memory info
  - All methods return real data instead of empty strings / "unknown"
**Issues Found:**
  - Window title via AppleScript requires Accessibility permissions. If not granted, falls back to app name only. Will be fully enabled in Phase 5 (Accessibility API).
  - `system_profiler` calls take ~350ms each. Two calls in `get_system_profile()` add ~700ms. Consider caching the JSON result.
**Notes for Next Step:**
  - Step 1.2 creates `core/apple_silicon_monitor.py` — the pynvml replacement for real-time GPU/thermal monitoring
  - The `_get_gpu_info_macos()` method here gives static info; the monitor in 1.2 gives dynamic stats

---

### Step 0.3 — Baseline Performance Metrics
**Date:** 2026-04-09
**Status:** DONE
**Files Modified:**
  - `docs/ATOM_CURRENT_STATE.md` — appended Step 0.3 benchmark results (~90 lines)
**Files Created:**
  - `scripts/m5_baseline_benchmark.py` — reusable benchmark script (475 lines)
**What Changed:**
  - Created benchmark script measuring 10 stages: config, imports, module init, intent engine, context engine, heavy modules, tools, cognitive layer, voice pipeline, memory snapshot
  - Ran benchmark on M5 with Python 3.9.6 / macOS 26.4 / arm64
  - Appended structured results to `docs/ATOM_CURRENT_STATE.md`
**Test Result:**
  - **Total module load: 879ms** (well under 5s target)
  - **Intent engine: 0.214ms avg** (0.080ms min, 0.348ms max) — 467x faster than 100ms target
  - **Memory: 44.8 MB RSS** (all modules loaded, no LLM) — extremely light
  - **CodeIntrospector.scan(): 642ms** — the single slowest init (scans 228 files)
  - **MemoryEngine: 31.4ms + 7.3MB** — second heaviest init
  - **Config validation: 92.7ms + 9.7MB** — jsonschema overhead
  - 20/26 modules OK, 6 FAIL (5 missing voice/LLM deps, 1 needs event loop)
  - System: 7.0GB free / 16.0GB, Battery 98%, Plugged In
**Issues Found:**
  - SystemIndexer.start() fails outside async context (needs `asyncio.run`) — not a real bug, just can't bench it synchronously
  - 5 voice/LLM packages missing: speech_recognition, faster-whisper, edge-tts, pygame, llama-cpp-python
  - All measurable targets are MET (intent, module load, memory)
  - STT/LLM/TTS/E2E targets are BLOCKED until voice deps are installed
**Notes for Next Step:**
  - Step 0.4 (finalize baseline report) is effectively DONE — ATOM_CURRENT_STATE.md now has all 3 sections (Step 0.1 errors, Step 0.2 audit, Step 0.3 metrics)
  - Can merge 0.4 into 0.3 and proceed directly to Phase 1
  - The benchmark script is reusable — run again after each phase to track improvement

---

### Step 0.2 — Scan All Files for Windows-Only Code Paths
**Date:** 2026-04-09
**Status:** DONE
**Files Modified:**
  - `docs/ATOM_CURRENT_STATE.md` — added comprehensive Step 0.2 audit section (~200 lines)
**Files Created:** none
**What Changed:**
  - Recursive grep scan for: `ctypes.windll`, `winreg`, `pynvml`, `wmic`, `SAPI`, `comtypes`, `winsdk`, `powershell`, `taskkill`, `win_service_iter`, `ImageGrab`, `keyboard.add_hotkey`
  - Classified every hit into 6 severity groups (CRASH / SILENT FAIL / GUARDED / NVIDIA-ONLY / WINDOWS-ONLY MODULE / SUBPROCESS COMMAND)
  - Also scanned for platform guards (`_is_windows`, `sys.platform`, `platform.system`)
  - Read all 5 critical files in full to verify wrapping status line by line
**Test Result:**
  - **49 total Windows-only code paths identified**
  - 18 functions UNGUARDED (will crash) across 5 files
  - 6 functions SILENT FAIL (wrapped, returns empty) across 5 files
  - 6 areas GUARDED (have platform check) across 3 files
  - 3 files NVIDIA-only (gpu_governor, gpu_resource_manager, gpu_execution_coordinator)
  - 2 entire modules Windows-only (tts_async/SAPI, media_watcher/winsdk)
  - 14 Windows subprocess commands across 6 files
**Issues Found:**
  - `core/router/media_actions.py` — ENTIRE file has zero platform guards. Every function crashes on macOS.
  - `core/router/system_actions.py` — ENTIRE file is Windows-only (ctypes.windll + powershell). All 9 functions crash.
  - `core/router/utility_actions.py` — 3 of 5 functions crash (minimize, maximize, switch_window). Only `read_clipboard_text` has try/except.
  - `core/router/app_actions.py` — `close_app()` uses `taskkill`, `list_installed_apps()` uses `powershell Get-StartApps`. Both crash.
  - `play_youtube()` uses `cmd /c start URL` which doesn't exist on macOS.
  - Step 0.1 reported "16 Windows-only paths" — actual count after deep audit is **49**.
**Notes for Next Step:**
  - Steps 0.3 (latency measurement) and 0.4 (finalize baseline report) can likely be merged since the current state doc already has boot metrics
  - Phase 1 fix priority table created — P0 files are media_actions, system_actions, utility_actions (most common user commands)
  - Python 3.11+ blocker remains — needs admin for Homebrew or standalone installer from python.org

---

### Step 0.1 — Run ATOM and Capture All macOS Errors
**Date:** 2026-04-09
**Status:** DONE
**Files Modified:**
  - `core/router/router.py` — fixed `_do_remember` NameError (class body ordering)
  - `core/gpu_resource_manager.py` — added `_last_power_unload` to `__slots__`
  - `core/system_indexer.py` — fixed `from core.platform_adapter import adapter` → `get_platform_adapter`
  - `core/runtime_config.py` — added `from __future__ import annotations`
  - `core/reasoning/action_executor.py` — `str | None` → `Optional[str]` for Py3.9 compat
  - `brain/intent_engine.py` — added `from __future__ import annotations`
  - `brain/simulation_engine.py` — added `from __future__ import annotations`
  - `core/cognition/preemption.py` — added `from __future__ import annotations`
  - `core/router/app_actions.py` — added `from __future__ import annotations`
  - `core/router/conversation_manager.py` — added `from __future__ import annotations`
  - `core/config_schema.py` — made validation non-fatal (warnings only)
  - `config/settings.json` — changed `ui.mode` from "desktop" to "web"
**Files Created:**
  - `docs/ATOM_CURRENT_STATE.md` — full baseline report (300+ lines)
  - `.venv/` — Python 3.9.6 virtual environment (temporary, will upgrade to 3.11+)
**What Changed:**
  - Installed Xcode Command Line Tools (920MB, gives git + Python 3.9.6 + C compiler)
  - Initialized git repo with remote origin
  - Created Python venv, installed core deps (psutil, aiohttp, numpy, Pillow, jsonschema)
  - Fixed 5 code bugs that blocked boot on ANY platform (not just macOS)
  - Added `from __future__ import annotations` to 5 files for Py3.10+ type syntax
  - Ran main.py successfully — ATOM BOOTS ON MACOS
  - Captured 16 Windows-only code paths across 10 files
  - Captured 12 missing package issues
  - Captured 5 config validation warnings
**Test Result:**
  - ATOM boots on macOS in ~2 seconds
  - Web dashboard starts at http://127.0.0.1:8765/
  - Startup greeting speaks: "Hey Boss... Apple M5 (10C/10T)... 16GB RAM..."
  - 43 tools, 28 commands, 8 cognitive modules all initialized
  - System health: 58/100
  - Clean shutdown after 25s test run
**Issues Found:**
  - Python 3.9.6 (CLT) works but project targets 3.11+
  - Homebrew install failed — needs admin/sudo (user `satyamyadav` not admin)
  - `winsdk` and `comtypes` are Windows-only packages in requirements.txt
  - 16 code paths use `ctypes.windll` which crashes on macOS (most wrapped in try/except)
  - `media_actions.py` and `system_actions.py` have UNWRAPPED `ctypes.windll` calls
  - STT pipeline completely broken (missing speech_recognition, pyaudio, faster-whisper)
  - TTS can init but not play audio (missing pygame)
**Notes for Next Step:**
  - Step 0.2 (scan for Windows-only code) is largely DONE as part of this step — see `docs/ATOM_CURRENT_STATE.md` Category 3
  - Priority: install Python 3.11+ (need admin for Homebrew, or use python.org installer)
  - The 5 code bugs fixed here would also crash on Windows — they're genuine bugs
  - ATOM is surprisingly portable — most Windows code has try/except guards

---

### Step 0.0 — Project Setup
**Date:** 2026-04-09
**Status:** DONE
**Files Modified:** none (fresh clone)
**Files Created:**
  - `CODEBASE_INDEX.md` — full codebase map (548 lines)
  - `MEMORY_BANK.md` — this file
  - `docs/ATOM_M5_EVOLUTION_PLAN.md` — rated + enhanced evolution plan (907 lines)
  - `.cursor/rules/ATOM_RULES.md` — Cursor rule for auto-reading memory bank
**What Changed:**
  - Cloned ATOM from https://github.com/satyam7475/ATOM.git (via zip, git CLT not installed)
  - Analyzed all 158 Python files, read key modules in depth
  - Created comprehensive codebase index with every file, line count, and purpose
  - Rated original evolution plan (7.2/10) and created enhanced M5-specific plan
  - Identified ~40 Windows-only code paths that will crash on macOS
  - Mapped 16 Apple-native technologies to ATOM modules they can replace/enhance
**Test Result:**
  - All files extracted and verified present in workspace
  - File count matches GitHub repo (158 Python, 6 config, 22 docs)
**Issues Found:**
  - Xcode Command Line Tools not installed (git unavailable, triggered install dialog)
  - Workspace is NOT a git repo yet (cloned via zip download)
**Notes for Next Step:**
  - If Xcode CLT installed, run: `git init && git remote add origin https://github.com/satyam7475/ATOM.git && git add -A && git commit -m "Initial clone"`
  - Next step is 0.1: Run main.py and capture all macOS errors

---

## FILES CHANGELOG

> Tracks every file modified/created during the evolution. Append-only.

| Date | Step | Action | File | Notes |
|------|------|--------|------|-------|
| 2026-04-09 | 0.0 | CREATED | `CODEBASE_INDEX.md` | Full codebase map |
| 2026-04-09 | 0.0 | CREATED | `MEMORY_BANK.md` | This file |
| 2026-04-09 | 0.0 | CREATED | `docs/ATOM_M5_EVOLUTION_PLAN.md` | Evolution plan |
| 2026-04-09 | 0.0 | CREATED | `.cursor/rules/ATOM_RULES.md` | Cursor auto-rule |
| 2026-04-09 | 0.1 | MODIFIED | `core/router/router.py` | Fixed _do_remember NameError |
| 2026-04-09 | 0.1 | MODIFIED | `core/gpu_resource_manager.py` | Fixed __slots__ |
| 2026-04-09 | 0.1 | MODIFIED | `core/system_indexer.py` | Fixed adapter import |
| 2026-04-09 | 0.1 | MODIFIED | `core/runtime_config.py` | Added future annotations |
| 2026-04-09 | 0.1 | MODIFIED | `core/reasoning/action_executor.py` | Fixed Py3.9 type syntax |
| 2026-04-09 | 0.1 | MODIFIED | `core/config_schema.py` | Made validation non-fatal |
| 2026-04-09 | 0.1 | MODIFIED | `config/settings.json` | ui.mode: desktop → web |
| 2026-04-09 | 0.1 | MODIFIED | 5 files | Added from __future__ import annotations |
| 2026-04-09 | 0.1 | CREATED | `docs/ATOM_CURRENT_STATE.md` | Full macOS baseline report |
| 2026-04-09 | 0.2 | MODIFIED | `docs/ATOM_CURRENT_STATE.md` | Added Step 0.2 audit: 49 Windows-only code paths across 18 files |
| 2026-04-09 | 0.3 | CREATED | `scripts/m5_baseline_benchmark.py` | Reusable benchmark script (10 stages) |
| 2026-04-09 | 0.3 | MODIFIED | `docs/ATOM_CURRENT_STATE.md` | Added Step 0.3 metrics: 879ms load, 0.21ms intent, 44.8MB RSS |
| 2026-04-09 | 1.1 | MODIFIED | `core/platform_adapter.py` | Added macOS GPU info, display info, services, window title, Unified Memory |
| 2026-04-09 | 1.2 | CREATED | `core/apple_silicon_monitor.py` | Apple Silicon hardware monitor (Unified Memory, thermal, battery) |
| 2026-04-09 | 1.2 | MODIFIED | `core/gpu_governor.py` | Multi-backend: Apple Silicon + NVIDIA, unified_memory field, thermal_pressure |
| 2026-04-09 | 1.2 | MODIFIED | `main.py` | Fixed GPU Governor log message for non-NVIDIA platforms |
| 2026-04-09 | 1.3 | MODIFIED | `core/gpu_resource_manager.py` | Unified Memory model: Apple Silicon detection, memory pressure admission, no VRAM budgets |
| 2026-04-09 | 1.3 | MODIFIED | `core/gpu_execution_coordinator.py` | Apple Silicon _nvml_snapshot, MPS torch path, renamed _torch_mem_mb |
| 2026-04-09 | 1.3A | DELETED | `core/gpu_scheduler.py` | Dead code: created but never called (104 lines) |
| 2026-04-09 | 1.3A | DELETED | `brain/gpu_pipeline.py` | 107 lines wrapping 1 useful line — inlined into local_brain_controller |
| 2026-04-09 | 1.3A | MODIFIED | `main.py` | Removed GPUScheduler import and dead variable |
| 2026-04-09 | 1.3A | MODIFIED | `cursor_bridge/local_brain_controller.py` | Replaced GPUPipeline wrapper with direct gpu_resource_mgr reference |
| 2026-04-09 | 1.3A | CREATED | `.cursor/rules/GPU_STACK_RULE.md` | GPU stack architecture rule + Phase 2 consolidation plan |
| 2026-04-09 | 1.4 | CREATED | `voice/tts_macos.py` | Native macOS TTS via `say` command (~310 lines, full interface compat) |
| 2026-04-09 | 1.4 | MODIFIED | `main.py` | Added `macos_native` TTS engine with auto-detection on darwin |
| 2026-04-09 | 1.5 | MODIFIED | `context/screen_reader.py` | Platform-aware screenshot: screencapture (macOS) + PIL + PowerShell (Windows) |
| 2026-04-09 | 1.6 | MODIFIED | `core/system_control.py` | macOS: wifi scan, startup list, DNS flush, power mgmt, volume, brightness |
| 2026-04-09 | 1.7 | MODIFIED | `core/desktop_control.py` | Dual-backend: pyautogui + AppleScript fallback, ctrl→cmd mapping |
| 2026-04-09 | 1.7 | MODIFIED | `core/security_policy.py` | Added 13 macOS hotkey equivalents (command+c/v/x/z/a/s/f/p/n/t/tab/q/w) |
| 2026-04-09 | 1.8A | CREATED | `core/silicon_governor.py` | Apple Silicon-only hardware monitor (~145L) |
| 2026-04-09 | 1.8A | CREATED | `core/inference_guard.py` | Model lifecycle + memory pressure guard (~155L) |
| 2026-04-09 | 1.8A | DELETED | `core/gpu_governor.py` | Multi-backend GPU monitor (331L) — NVIDIA/pynvml dead code |
| 2026-04-09 | 1.8A | DELETED | `core/gpu_resource_manager.py` | VRAM budgets, slot allocation (387L) — wrong model for Unified Memory |
| 2026-04-09 | 1.8A | DELETED | `core/gpu_execution_coordinator.py` | VRAM admission control, fragmentation (593L) — unnecessary on Apple Silicon |
| 2026-04-09 | 1.8A | MODIFIED | `core/gpu_watchdog.py` | Removed CUDA reset, simplified to pure stall detection |
| 2026-04-09 | 1.8A | MODIFIED | `main.py` | SiliconGovernor + InferenceGuard replacing GPU modules |
| 2026-04-09 | 1.8A | MODIFIED | `cursor_bridge/local_brain_controller.py` | attach_inference_guard + legacy shim |
| 2026-04-09 | 1.8A | MODIFIED | `core/rag/rag_engine.py` | Removed GPUExecutionCoordinator dependency |
| 2026-04-09 | 1.8A | MODIFIED | `services/llm_worker.py` | Events from inference_guard |
| 2026-04-09 | 1.8A | MODIFIED | `services/stt_worker.py` | Events from inference_guard |
| 2026-04-09 | 1.8A | MODIFIED | `services/gpu_cognition_worker.py` | InferenceGuard replacing GPUResourceManager |
| 2026-04-09 | 1.8A | MODIFIED | `requirements.txt` | Removed Windows-only deps, added Metal build notes |
| 2026-04-09 | Native | REWRITTEN | `voice/media_watcher.py` | macOS AppleScript (Spotify, Music, browser) replacing broken winsdk |
| 2026-04-09 | Native | REWRITTEN | `voice/tts_macos.py` | NSSpeechSynthesizer + say fallback, 184 voices, 4.4ms barge-in |
| 2026-04-09 | Native | CREATED | `voice/stt_macos.py` | SFSpeechRecognizer + AVAudioEngine, on-device Neural Engine STT |
| 2026-04-09 | Native | MODIFIED | `context/screen_reader.py` | Vision framework OCR primary (109ms avg), EasyOCR fallback |
| 2026-04-09 | Native | CREATED | `core/macos/__init__.py` | macOS-native modules package |
| 2026-04-09 | Native | CREATED | `core/macos/fs_watcher.py` | FSEvents kernel-level file watcher |
| 2026-04-09 | Native | MODIFIED | `core/embedding_engine.py` | MPS (Apple Silicon GPU) device auto-detection |
| 2026-04-09 | Native | MODIFIED | `main.py` | Wired FSWatcher, updated screen_reader log |
| 2026-04-09 | Native | MODIFIED | `requirements.txt` | v19: 8 pyobjc packages, 7 third-party voice deps → optional |
| 2026-04-11 | 2.1 | CREATED | `core/cognitive_kernel.py` | Central brain coordinator: 5 paths, circuit breakers, system-aware routing (~310L) |
| 2026-04-11 | 2.1 | MODIFIED | `main.py` | CognitiveKernel init + wiring to Router |
| 2026-04-11 | 2.1 | MODIFIED | `core/router/router.py` | attach_cognitive_kernel + query_plan in cursor_query event |
| 2026-04-11 | 2.2 | MODIFIED | `core/router/router.py` | Added top-level router error boundary + recovery hook |
| 2026-04-11 | 2.2 | MODIFIED | `cursor_bridge/local_brain_controller.py` | Wrapped on_query with outer LLM failure boundary |
| 2026-04-11 | 2.2 | MODIFIED | `core/reasoning/action_executor.py` | Full execute() isolation, no registry/security failure escapes |
| 2026-04-11 | 2.2 | MODIFIED | `core/boot/wiring.py` | Guarded critical event handlers + explicit shutdown_event parameter |
| 2026-04-11 | 2.2 | MODIFIED | `voice/stt_async.py` | Error isolation for start_listening/on_state_changed/shutdown |
| 2026-04-11 | 2.2 | MODIFIED | `voice/tts_edge.py` | Error isolation for speak/on_response/on_partial_response/shutdown |
| 2026-04-11 | 2.2 | MODIFIED | `main.py` | Pass shutdown_event into wire_events |
| 2026-04-11 | 2.3 | MODIFIED | `core/runtime_watchdog.py` | Active per-module budgets, async helpers, TTS timeout supervision, startup cooldown fix |
| 2026-04-11 | 2.3 | MODIFIED | `core/router/router.py` | RuntimeWatchdog budgets for intent classification + cache lookup |
| 2026-04-11 | 2.3 | MODIFIED | `cursor_bridge/local_brain_controller.py` | RuntimeWatchdog wired into RAG cap, LLM streaming timeout, async tool execution |
| 2026-04-11 | 2.3 | MODIFIED | `core/reasoning/action_executor.py` | Added `execute_async()` for watchdog-compatible tool execution |
| 2026-04-11 | 2.3 | MODIFIED | `main.py` | Attached RuntimeWatchdog to Router + LocalBrainController |
| 2026-04-11 | 2.3 | MODIFIED | `core/config_schema.py` | Added config schema entries for watchdog stage budgets |
| 2026-04-11 | 2.4 | MODIFIED | `core/memory_engine.py` | Vector-result cap + pressure-mode keyword-only retrieval |
| 2026-04-11 | 2.4 | MODIFIED | `core/rag/rag_engine.py` | RAG snippet cap + low-memory mode + embed/cache shedding |
| 2026-04-11 | 2.4 | MODIFIED | `core/rag/rag_cache.py` | Added cache clear helpers for pressure shedding |
| 2026-04-11 | 2.4 | MODIFIED | `brain/memory_graph.py` | Node cap + pruning + pressure-mode query reduction + vector cleanup |
| 2026-04-11 | 2.4 | MODIFIED | `cursor_bridge/local_brain_controller.py` | Forward memory pressure updates to RAG + MemoryGraph |
| 2026-04-11 | 2.4 | MODIFIED | `main.py` | Periodic silicon_stats_update memory-pressure coordination |
| 2026-04-11 | 2.4 | MODIFIED | `core/silicon_governor.py` | Lowered default unified-memory warning threshold to 85% |
| 2026-04-11 | 2.4 | MODIFIED | `core/config_schema.py` | Added memory/RAG pressure config keys |
| 2026-04-11 | 2.4 | MODIFIED | `config/settings.json` | Explicit M5 memory-pressure caps and thresholds |
| 2026-04-11 | 2.5 | VERIFIED | `Terminal` | Abbreviated integration smoke: stress + chaos harnesses + short live runtime, no full-process crash |
| 2026-04-11 | 3.1 | INSTALLED | `Terminal` | Installed `mlx`, `mlx-lm`, and `mlx-metal` into project venv; functional imports OK |
| 2026-04-11 | 3.2 | DOWNLOADED | `models/qwen3-4b-mlx/` | Local snapshot of `mlx-community/Qwen3-4B-4bit` (~2.1 GB) |
| 2026-04-11 | 3.2 | DOWNLOADED | `models/qwen3-1.7b-mlx/` | Local snapshot of `mlx-community/Qwen3-1.7B-4bit` (~939 MB) |
| 2026-04-11 | 3.3 | CREATED | `brain/mlx_llm.py` | Added `MLXBrain` compatibility wrapper with MLX load/stream/preempt support and primary/fast model roles |
| 2026-04-11 | 3.4 | MODIFIED | `cursor_bridge/local_brain_controller.py` | Swapped controller backend to `MLXBrain`, updated MLX lifecycle messaging, and verified warm-up/unload path |
| 2026-04-11 | 3.5 | MODIFIED | `core/cognitive_kernel.py` | Added `model_role` / `runtime_mode` / prompt-hint routing for quick/full/deep MLX paths |
| 2026-04-11 | 3.5 | MODIFIED | `cursor_bridge/local_brain_controller.py` | Consumes route-plan hints for MLX role selection, RAG gating, and late-RAG retry continuity |
| 2026-04-11 | 3.5 | MODIFIED | `cursor_bridge/structured_prompt_builder.py` | Added prompt context support for route-specific inference hints |
| 2026-04-11 | 3.5 | MODIFIED | `core/llm_inference_queue.py` | Preserves `query_plan` through queue submission/worker delivery |
| 2026-04-11 | 3.5 | MODIFIED | `core/boot/wiring.py` | Forwarded `query_plan` into queue-backed local-brain execution and refreshed MLX wording |
| 2026-04-11 | 3.6 | CREATED | `core/runtime/latency_controller.py` | Added dynamic latency/controller policy for per-plan total budget, RAG cap, and context trimming |
| 2026-04-11 | 3.6 | MODIFIED | `core/runtime/__init__.py` | Exported latency controller runtime helpers |
| 2026-04-11 | 3.6 | MODIFIED | `core/cognitive_kernel.py` | Applies latency policy onto `QueryPlan` and emits richer budget diagnostics |
| 2026-04-11 | 3.6 | MODIFIED | `core/router/router.py` | Plans before memory retrieval and skips unnecessary semantic-memory work for quick/direct/cache fallbacks |
| 2026-04-11 | 3.6 | MODIFIED | `cursor_bridge/local_brain_controller.py` | Uses plan-based context trimming and RAG budget caps from the latency controller |
| 2026-04-11 | 3.7 | MODIFIED | `config/settings.json` | Added explicit MLX model paths, cognitive-kernel/latency defaults, watchdog budgets, and Apple Silicon-friendly runtime settings |
| 2026-04-11 | 3.7 | MODIFIED | `core/config_schema.py` | Added schema coverage for MLX config and latency sections; validation now passes for checked-in settings |
| 2026-04-11 | 3.8 | CREATED | `tools/mlx_vs_llamacpp_benchmark.py` | Added isolated-subprocess benchmark for MLX vs llama-cpp Qwen3-4B load/inference timing |
| 2026-04-11 | 3.8 | DOWNLOADED | `models/Qwen3-4B-Q4_K_M.gguf` | Same-family GGUF baseline for Qwen3-4B benchmark (~2.33 GB) |
| 2026-04-11 | 3.9 | CREATED | `core/boot/cold_start.py` | Added cold-start orchestration for fast-role preload, embedding warm-up, session restore, command-cache seeding, and restored system context |
| 2026-04-11 | 3.9 | MODIFIED | `main.py` | Wired cold-start bootstrap into boot/shutdown and replayed restored context before listening starts |
| 2026-04-11 | 3.9 | MODIFIED | `cursor_bridge/local_brain_controller.py` | Extended MLX warm-up to target specific model roles during startup |
| 2026-04-11 | 3.9 | MODIFIED | `core/memory_engine.py` | Added eager embedding warm-up and top-command lookup helpers for cold-start seeding |
| 2026-04-11 | 3.9 | CREATED | `tests/test_cold_start.py` | Added focused regression coverage for cold-start restore/cache/persist behavior |
| 2026-04-11 | 3.10 | CREATED | `voice/interrupt_handler.py` | Added centralized voice-interrupt coordination for barge-in state handoff, TTS stop, and brain preemption |
| 2026-04-11 | 3.10 | MODIFIED | `core/boot/wiring.py` | Routed `speech_partial`/`resume_listening` through the interrupt coordinator and prepared interrupts before routing new speech |
| 2026-04-11 | 3.10 | MODIFIED | `core/ipc/interrupt_manager.py` | Made global interrupts compatible with both AsyncEventBus and ZMQ workers and used fast emit when available |
| 2026-04-11 | 3.10 | MODIFIED | `core/async_event_bus.py` | Promoted interrupt-related events to high priority for lower-latency barge-in dispatch |
| 2026-04-11 | 3.10 | MODIFIED | `services/tts_worker.py` | Awaited async TTS stop calls during worker interrupt handling |
| 2026-04-11 | 3.10 | CREATED | `tests/test_voice_interrupt.py` | Added focused regression tests for voice interrupt heuristics and state transitions |

---

## ARCHITECTURE DECISIONS LOG

> Key decisions made during evolution. Helps future sessions understand WHY, not just WHAT.

| # | Decision | Rationale | Date |
|---|----------|-----------|------|
| 1 | macOS survival before optimization | Codebase has ~40 Windows-only paths. It will crash before we can benchmark. | 2026-04-09 |
| 2 | MLX over llama-cpp-python for inference | MLX is native Apple Silicon: unified memory, zero-copy, 30-40% faster on M-series | 2026-04-09 |
| 3 | Dual-model: Qwen3-4B + Qwen3-1.7B | **Confirmed best combo for M5 Air.** Primary: Qwen3-4B (3GB, 50-70 tok/s, thinking mode). Fast: Qwen3-1.7B (1.2GB, 120-160 tok/s). Same family = identical prompt/tool format. Total 4.2GB leaves 5.8GB headroom. 4B chosen over 8B: 40% faster, 45% less RAM, no thermal throttle on fanless Air. Thinking mode closes quality gap with 8B when needed. | 2026-04-09 |
| 4 | Native macOS TTS for quick replies | `say` command is instant (~5ms). Edge TTS requires network. Use native for short, Edge for quality. | 2026-04-09 |
| 5 | Cognitive Kernel as central coordinator | Single point for mode selection, model routing, resource allocation, circuit breaking. | 2026-04-09 |
| 6 | Unified Memory: no VRAM budget model | Apple Silicon shares all memory. The NVIDIA VRAM budget model is wrong. Treat as single pool. | 2026-04-09 |
| 7 | GPU stack: remove dead code, defer full consolidation | gpu_scheduler.py (dead) and gpu_pipeline.py (1 useful line) deleted. Full consolidation (resource_manager + execution_coordinator → InferenceGuard) planned for Phase 2.1 after Cognitive Kernel. | 2026-04-09 |
| 8 | Apple Silicon-only compute layer | Stripped all NVIDIA/CUDA/pynvml code. Created SiliconGovernor + InferenceGuard to replace multi-backend GPU stack. No VRAM budgets — Unified Memory with pressure-based admission. Net ~1,000 lines removed. | 2026-04-09 |
| 9 | Native macOS stack via pyobjc | Same philosophy as MLX for LLM — use frameworks already loaded in macOS RAM. pyobjc (~18MB) bridges to Apple frameworks that replace ~2.8GB of third-party deps. STT: SFSpeechRecognizer (Neural Engine, ~50ms), TTS: NSSpeechSynthesizer (no subprocess, 4.4ms barge-in), OCR: Vision (109ms vs 3000ms), Media: AppleScript, FileWatch: FSEvents. | 2026-04-09 |
| 10 | Cognitive Kernel as 5-path router | Single coordinator for all query routing: DIRECT (intent/quick-reply, no LLM), CACHE (cached response), QUICK (fast 1.7B), FULL (primary 4B), DEEP (4B+thinking+RAG). Circuit breakers auto-bypass failing modules. System-aware: degrades on low battery, thermal throttling, memory pressure. 80% of common queries skip LLM entirely. | 2026-04-11 |
| 11 | Add top-level runtime boundaries before watchdog budgets | AsyncEventBus already isolates handler tasks, but critical component entry points still needed explicit outer guards so Router, LocalBrain, STT, Edge TTS, and ActionExecutor fail soft with fallback behavior instead of dying mid-task. Wiring also now receives `shutdown_event` explicitly to avoid hidden cross-module global dependency. | 2026-04-11 |
| 12 | RuntimeWatchdog should actively budget hot paths, not just detect stuck states | Passive THINKING/SPEAKING dwell checks are too coarse for intent/cache/RAG/LLM/tool stages. The watchdog now wraps hot operations directly where possible, caps RAG budgets, and keeps state-dwell supervision as a last-resort safety net. | 2026-04-11 |
| 13 | Use SiliconGovernor telemetry as the single periodic memory-pressure signal | Apple Silicon already has one unified-memory monitor loop. Rather than add another poller, ATOM now applies retrieval/pruning/embed-shedding behavior from the existing `silicon_stats_update` stream so every memory-heavy module degrades off the same signal. | 2026-04-11 |
| 14 | Phase 3 can start on the current project venv without waiting for a Python upgrade | Installing `mlx` + `mlx-lm` succeeded on the active Python 3.9 project environment, so the old "Python 3.11+ required for Phase 3 kickoff" blocker is stale. Remove the blocker and continue with model download / MLX integration. | 2026-04-11 |
| 15 | Store MLX model snapshots under project-local `models/` paths | Keeping both MLX repos inside the workspace (`models/qwen3-4b-mlx`, `models/qwen3-1.7b-mlx`) makes Phase 3 wiring deterministic, avoids depending on ephemeral global cache paths, and matches the existing project convention of local model assets. | 2026-04-11 |
| 16 | Build the MLX migration as a MiniLLM-compatible wrapper before swapping controller wiring | Preserving the existing async/preempt/streaming contract in `brain/mlx_llm.py` lets Phase 3 switch inference backends with smaller, safer controller changes. Dual-model role support can be added now without forcing a full LocalBrainController redesign in the same step. | 2026-04-11 |
| 17 | Swap LocalBrainController to MLX before adding dual-model routing | Moving the controller onto the MLX backend first isolates the backend migration from the upcoming routing logic. That keeps Step 3.5 focused on decision policy and model-role selection instead of mixing backend replacement with routing bugs. | 2026-04-11 |
| 18 | Treat `QueryPlan` as the authoritative handoff for model role, runtime mode, and reasoning bias | Once the kernel chooses quick/full/deep, downstream code should consume that plan instead of recomputing its own conflicting routing. Propagating `query_plan` through the queue and retry path keeps MLX role selection, RAG behavior, and prompt style aligned. | 2026-04-11 |
| 19 | Attach dynamic latency/context policy to `QueryPlan` instead of creating a parallel budget channel | The same plan that chooses model/path should also carry the active budget envelope. That lets the router skip pre-LLM memory work earlier, lets LocalBrainController trim memory/history and cap RAG consistently, and avoids scattering separate latency heuristics across the stack. | 2026-04-11 |
| 20 | Make MLX/kernel/latency defaults explicit in checked-in config instead of relying on code fallbacks | Once Phase 3 runtime behavior depends on MLX model paths and latency heuristics, those values should live visibly in `config/settings.json` and validate against schema. That makes benchmarking reproducible and keeps future tuning as config changes instead of hidden code defaults. | 2026-04-11 |
| 21 | Benchmark backend speed with same-family Qwen3 assets in isolated subprocesses | For backend comparison, the benchmark should keep model family aligned and avoid shared-process memory contamination. On this machine the current llama-cpp baseline cannot load the target Qwen3-4B GGUF, which is itself a meaningful migration result: MLX is operational for the planned Qwen3 target while the legacy backend is not. | 2026-04-11 |

---

## HOW TO USE THIS MEMORY BANK

### For a NEW chat session:

1. AI reads `MEMORY_BANK.md` (this file) — knows exactly where we are
2. AI reads `CURRENT_STEP` at the top — knows what to do next
3. AI reads the completion report for the LAST completed step — knows what was just done
4. AI executes the next step
5. After completing the step, AI updates this file:
   - Moves `CURRENT_STEP` forward
   - Writes a completion report
   - Appends to FILES CHANGELOG
   - Updates the step status to DONE

### For Satyam:

Tell the AI: **"Read MEMORY_BANK.md and continue from where we left off"**
or: **"Execute next evolution step"**
or: **"Execute step 1.3"** (to target a specific step)

The AI will:
1. Read this file
2. Know the current step
3. Know what files to modify
4. Execute the step
5. Update this file with the completion report
6. Tell you what to do in the next session
