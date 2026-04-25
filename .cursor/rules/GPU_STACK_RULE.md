# Compute Stack — Architecture Rule (Apple Silicon)

> **Scope:** All files related to hardware monitoring, memory management, model
> loading, and inference scheduling.
> **Hardware:** Apple Silicon M5 (Unified Memory, Metal GPU, Neural Engine).
> **Updated:** 2026-04-09

---

## Current Files (after Step 1.8A Silicon Refactoring)

| File | Lines | Role | Status |
|------|-------|------|--------|
| `core/apple_silicon_monitor.py` | ~294 | Hardware data source (memory, thermal, battery, GPU info) | KEEP |
| `core/silicon_governor.py` | ~145 | Monitoring loop + thermal/memory event emission | KEEP |
| `core/inference_guard.py` | ~155 | Model slot tracking, memory pressure, idle unload | KEEP |
| `core/gpu_watchdog.py` | ~73 | Inference stall detection | KEEP |

**Deleted in Step 1.8A (Silicon Refactoring):**
- `core/gpu_governor.py` (331L) — multi-backend monitor (NVIDIA + Apple Silicon + CPU fallback)
- `core/gpu_resource_manager.py` (387L) — VRAM budgets, slot allocation, load grant tokens
- `core/gpu_execution_coordinator.py` (593L) — VRAM admission, fragmentation estimation, priority queue

**Previously deleted in Phase 1:**
- `core/gpu_scheduler.py` — dead code (created but never called)
- `brain/gpu_pipeline.py` — 107 lines wrapping 1 useful line (inlined)

---

## Core Principle

**Apple Silicon has Unified Memory.** CPU, GPU, and Neural Engine share the
same RAM pool. There is no VRAM. There is no discrete GPU. There is no
CPU-vs-GPU decision. The SoC handles scheduling.

ATOM needs to know:
1. **When to throttle** (thermal pressure via pmset)
2. **When memory is tight** (system RAM pressure — because it IS the GPU memory)
3. **Which models are loaded** (slot tracking for lifecycle management)
4. **When to idle-unload** (power/battery awareness)

ATOM does NOT need:
- VRAM budgets or slot allocations
- CUDA/NVML/pynvml imports
- Fragmentation heuristics
- Multi-backend detection logic
- Separate CPU-only fallback mode

---

## Rules for AI Assistants

### DO NOT

1. **Do not create new GPU/compute files.** The 4-file stack is complete.
   Add logic to existing files.

2. **Do not add VRAM budget logic.** Unified Memory = system RAM = GPU RAM.
   Use `InferenceGuard.memory_available()` for pressure checks.

3. **Do not import pynvml, torch.cuda, or any NVIDIA libraries.**
   This codebase is Apple Silicon-only.

4. **Do not create new state dataclasses.** `AppleSiliconStats` is the single
   hardware state class. `InferenceGuard` tracks model slots internally.

5. **Do not add "CPU-only mode" fallbacks.** Apple Silicon always has a GPU.

### DO

1. **Use `SiliconGovernor.get_stats()`** for hardware telemetry (thermal
   pressure, memory, CPU utilization, battery).

2. **Use `InferenceGuard.memory_available()`** to check if system memory
   can handle loading another model (<90% pressure threshold).

3. **Use `GPUStallWatchdog`** for inference timeout detection. Simple and
   platform-agnostic.

4. **Use `get_apple_silicon_memory_mb()`** from `apple_silicon_monitor`
   for direct memory stats anywhere.

---

## Wiring in main.py

```python
# Silicon Governor (hardware monitoring)
from core.silicon_governor import SiliconGovernor
silicon_governor = SiliconGovernor(bus, config)
silicon_governor.start()

# Inference Guard (model lifecycle)
from core.inference_guard import InferenceGuard
inference_guard = InferenceGuard(bus, config)
inference_guard.start_power_task()
local_brain.attach_inference_guard(inference_guard)

# Stall Watchdog (inference timeout)
from core.gpu_watchdog import GPUStallWatchdog
gpu_stall_wd = GPUStallWatchdog(bus, config)
gpu_stall_wd.start()
```

## Event Names (backward compatible)

| Event | Emitter | Purpose |
|-------|---------|---------|
| `silicon_stats_update` | SiliconGovernor | Periodic hardware telemetry |
| `gpu_stats_update` | SiliconGovernor | Legacy compat alias |
| `silicon_thermal_warn` | SiliconGovernor | Thermal throttling detected |
| `silicon_memory_warn` | SiliconGovernor | Memory pressure >90% |
| `v7_gpu_request_load` | InferenceGuard | Model load signal |
| `v7_gpu_unload` | InferenceGuard | Model unload signal |
| `v7_gpu_status` | InferenceGuard | Model slot status |
| `v7_gpu_power` | InferenceGuard | Power mode change |
| `gpu_stall` | GPUStallWatchdog | Inference hung |
