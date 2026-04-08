# ATOM MEMORY BANK

> **Purpose:** Single source of truth for evolution progress. Read this FIRST in every new session.
> **Updated:** 2026-04-09
> **Hardware:** MacBook Air M5 (Apple Silicon, Unified Memory, Neural Engine, Metal GPU)

---

## CURRENT STATUS

```
CURRENT_STEP  = 0.1
OVERALL_PHASE = PHASE 0 — BASELINE & MAC TRIAGE
BLOCKER       = Xcode Command Line Tools install pending (needed for git)
LAST_ACTION   = Created Memory Bank, Codebase Index, Evolution Plan
NEXT_ACTION   = Execute Step 0.1 — Run ATOM and capture all errors/crashes on macOS
```

---

## QUICK CONTEXT (for AI sessions)

**What is ATOM?** Satyam's JARVIS-style voice AI OS. ~51,400 lines Python. Local LLM (llama-cpp-python/GGUF), faster-whisper STT, Edge TTS, aiohttp dashboard, 40+ tools, ReAct loop, autonomous behavior.

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
| 0.1 | Run `main.py`, capture all import errors and crashes on macOS | NOT_STARTED | — |
| 0.2 | Scan all files for Windows-only code paths (winreg, WMIC, SAPI, ctypes.windll) | NOT_STARTED | — |
| 0.3 | Measure baseline: latency per stage, memory footprint, CPU usage | NOT_STARTED | — |
| 0.4 | Create `docs/ATOM_CURRENT_STATE.md` with full baseline report | NOT_STARTED | — |

**Phase 0 Deliverable:** `docs/ATOM_CURRENT_STATE.md` — complete baseline snapshot.

---

### PHASE 1 — MAC SURVIVAL (make it boot without crashing)

| Step | Description | Files | Status | Report |
|------|------------|-------|--------|--------|
| 1.1 | Fix `platform_adapter.py` — GPU info, display info, service listing for macOS | `core/platform_adapter.py` | NOT_STARTED | — |
| 1.2 | Create `core/apple_silicon_monitor.py` — replace pynvml with macOS hardware monitoring | NEW: `core/apple_silicon_monitor.py`, UPDATE: `core/gpu_governor.py` | NOT_STARTED | — |
| 1.3 | Fix GPU resource manager — unified memory model (no VRAM budgets on Apple Silicon) | `core/gpu_resource_manager.py` | NOT_STARTED | — |
| 1.4 | Create `voice/tts_macos.py` — native macOS TTS via `say` command | NEW: `voice/tts_macos.py`, UPDATE: TTS selection in `main.py` | NOT_STARTED | — |
| 1.5 | Fix `screen_reader.py` — macOS screenshot via `screencapture` command | `context/screen_reader.py` | NOT_STARTED | — |
| 1.6 | Fix `system_control.py` — macOS implementations for volume, brightness, wifi, startup | `core/system_control.py` | NOT_STARTED | — |
| 1.7 | Fix `desktop_control.py` — ensure pyautogui works on macOS or add AppleScript fallback | `core/desktop_control.py` | NOT_STARTED | — |
| 1.8 | Install llama-cpp-python with Metal: `CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python` | Terminal | NOT_STARTED | — |
| 1.9 | Run `main.py` again — verify zero crashes on macOS | Terminal | NOT_STARTED | — |

**Phase 1 Deliverable:** ATOM boots and runs on M5 without crashes. All Windows-only code paths have macOS equivalents or graceful fallbacks.

---

### PHASE 2 — STABILITY & COGNITIVE KERNEL

| Step | Description | Files | Status | Report |
|------|------------|-------|--------|--------|
| 2.1 | Create `core/cognitive_kernel.py` — central brain coordinator (mode selection, query routing, resource allocation) | NEW: `core/cognitive_kernel.py` | NOT_STARTED | — |
| 2.2 | Add error isolation wrappers to: `router.py`, `local_brain_controller.py`, `action_executor.py`, `wiring.py`, `stt_async.py`, `tts_edge.py` | 6 files | NOT_STARTED | — |
| 2.3 | Upgrade `runtime_watchdog.py` — per-module execution budgets with timeout kill | `core/runtime_watchdog.py` | NOT_STARTED | — |
| 2.4 | Add memory pressure protection — limits on vector results, RAG snippets, graph nodes + periodic memory check | Multiple core files | NOT_STARTED | — |
| 2.5 | Integration test — run ATOM for 30 min, verify no crashes, no memory leaks | Terminal | NOT_STARTED | — |

**Phase 2 Deliverable:** ATOM is unbreakable. Cognitive Kernel routes all decisions. No single module can crash the system.

---

### PHASE 3 — SPEED (MLX + Apple Native)

| Step | Description | Files | Status | Report |
|------|------------|-------|--------|--------|
| 3.1 | Install MLX: `pip install mlx mlx-lm` | Terminal | NOT_STARTED | — |
| 3.2 | Download MLX models: Qwen2.5-7B-4bit + Qwen2.5-1.5B-4bit | `models/` | NOT_STARTED | — |
| 3.3 | Create `brain/mlx_llm.py` — MLX-native LLM wrapper with streaming | NEW: `brain/mlx_llm.py` | NOT_STARTED | — |
| 3.4 | Update `local_brain_controller.py` — use MLX brain instead of llama-cpp | `cursor_bridge/local_brain_controller.py` | NOT_STARTED | — |
| 3.5 | Implement dual-model routing in Cognitive Kernel (1.5B for simple, 7B for complex) | `core/cognitive_kernel.py` | NOT_STARTED | — |
| 3.6 | Create `core/runtime/latency_controller.py` — dynamic latency budgets | NEW: `core/runtime/latency_controller.py` | NOT_STARTED | — |
| 3.7 | Update `config/settings.json` — MLX model paths, Apple Silicon tuning | `config/settings.json` | NOT_STARTED | — |
| 3.8 | Benchmark: measure MLX vs llama-cpp inference speed on M5 | Terminal | NOT_STARTED | — |

**Phase 3 Deliverable:** ATOM uses MLX for inference. Sub-500ms for simple queries, sub-2s for complex. Dual-model architecture working.

---

### PHASE 4 — INTELLIGENCE UPGRADE

| Step | Description | Files | Status | Report |
|------|------------|-------|--------|--------|
| 4.1 | Implement cognitive budget system in Cognitive Kernel | `core/cognitive_kernel.py` | NOT_STARTED | — |
| 4.2 | Upgrade prediction engine — resource preloading for high-confidence predictions | `core/cognitive/prediction_engine.py` | NOT_STARTED | — |
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
| 5.6 | Add FSEvents file monitoring for proactive suggestions | NEW: `core/macos/file_watcher.py` | NOT_STARTED | — |
| 5.7 | Add native macOS screen OCR via Vision framework | `context/screen_reader.py` | NOT_STARTED | — |
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

---

## ARCHITECTURE DECISIONS LOG

> Key decisions made during evolution. Helps future sessions understand WHY, not just WHAT.

| # | Decision | Rationale | Date |
|---|----------|-----------|------|
| 1 | macOS survival before optimization | Codebase has ~40 Windows-only paths. It will crash before we can benchmark. | 2026-04-09 |
| 2 | MLX over llama-cpp-python for inference | MLX is native Apple Silicon: unified memory, zero-copy, 30-40% faster on M-series | 2026-04-09 |
| 3 | Dual-model (1.5B + 7B) architecture | Small model for fast commands, large for deep reasoning. Both fit in unified memory. | 2026-04-09 |
| 4 | Native macOS TTS for quick replies | `say` command is instant (~5ms). Edge TTS requires network. Use native for short, Edge for quality. | 2026-04-09 |
| 5 | Cognitive Kernel as central coordinator | Single point for mode selection, model routing, resource allocation, circuit breaking. | 2026-04-09 |
| 6 | Unified Memory: no VRAM budget model | Apple Silicon shares all memory. The NVIDIA VRAM budget model is wrong. Treat as single pool. | 2026-04-09 |

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
