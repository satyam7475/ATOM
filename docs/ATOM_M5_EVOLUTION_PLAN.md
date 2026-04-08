# ATOM M5 Evolution Plan — Rated & Enhanced

> **For:** Satyam (Boss)
> **Hardware:** MacBook Air M5 (Apple Silicon, Unified Memory, Neural Engine, Metal GPU)
> **Date:** April 2026

---

## Plan Rating: 7.2 / 10

### What the original plan gets RIGHT

| Aspect | Score | Notes |
|--------|-------|-------|
| Phase ordering (stability first) | 9/10 | Correct. You don't optimize something that crashes. |
| MLX adoption | 9/10 | The single biggest performance win on Apple Silicon. |
| Cognitive Kernel concept | 8/10 | Central brain-of-brain is the right pattern. |
| Latency Controller | 8/10 | Smart routing to skip LLM when unnecessary. |
| Phase 0 baseline | 10/10 | Essential. No optimization without measurement. |
| Multi-model strategy | 8/10 | Small model for fast, big model for deep — correct. |
| Autonomy upgrade | 7/10 | Good direction but needs more concrete execution plan. |

### What the original plan gets WRONG or MISSES

| Issue | Impact | Details |
|-------|--------|---------|
| macOS integration is Phase 5 (last) | CRITICAL | It should be Phase 1-2. Your code will CRASH on Mac before you can optimize it. The codebase has ~40 Windows-only code paths (winreg, WMIC, PowerShell, SAPI COM, ctypes.windll). You must fix these FIRST. |
| GPU Governor is NVIDIA-only | CRITICAL | `gpu_governor.py` uses `pynvml` exclusively. Apple Silicon has no NVIDIA GPU. This module silently fails and provides zero telemetry. |
| Missing Unified Memory understanding | HIGH | Apple Silicon has NO separate VRAM. CPU and GPU share the SAME memory pool. The entire `gpu_resource_manager.py` VRAM budget model is wrong for M5. You don't "offload" — everything is already in shared memory. |
| Missing Apple Neural Engine (ANE) | HIGH | M5 has 16+ ANE cores that run ML models at ~4x efficiency vs GPU with near-zero power. Perfect for: embedding model, intent classification, small inference. The plan doesn't mention it. |
| Missing native Apple STT | HIGH | macOS has `SFSpeechRecognizer` (on-device, ~50ms latency for short phrases). For commands, this is 5-10x faster than faster-whisper. Use native for commands, whisper for conversation. |
| Missing native Apple TTS | MEDIUM | macOS `say` command / `NSSpeechSynthesizer` is instant, zero-latency, fully offline. Edge TTS needs network. Use native for fast acknowledgements, Edge for quality responses. |
| Missing macOS Accessibility API | HIGH | This is HOW you get JARVIS-level control. The Accessibility API can read/control ANY app's UI elements — buttons, text fields, menus. Way more powerful than pyautogui blind clicking. |
| Missing Apple Keychain | MEDIUM | `security_fortress.py` builds a custom EncryptedVault. macOS Keychain is hardware-backed (Secure Enclave on M5), zero-effort, already integrated with the OS. |
| Missing Vision framework for OCR | MEDIUM | `screen_reader.py` uses EasyOCR (slow, CPU-heavy). Apple's Vision framework does OCR natively on the Neural Engine — 10x faster, zero dependencies. |
| Missing IOKit for hardware monitoring | MEDIUM | `system_scanner.py` and `gpu_governor.py` can't read Apple Silicon thermals, power draw, or SSD health. IOKit provides all of this natively. |
| Screen capture is Windows-only | HIGH | `screen_reader.py` uses PIL ImageGrab with PowerShell fallback. macOS needs `screencapture` command or CGWindowListCreateImage. |
| Timelines too optimistic | LOW | 1-2 days for Phase 0 baseline with a codebase this size is tight. Plan for 2-3 days. Phase 1 stability is 5-8 days realistically. |
| Missing `launchd` for background agent | MEDIUM | The plan mentions "background service" but doesn't specify `launchd` plist — the ONLY proper way to run background agents on macOS. |

---

## Enhanced Evolution Plan

### PHASE 0 — BASELINE & MAC TRIAGE (2-3 days)

**Goal:** Know exactly what works, what crashes, what's slow.

#### 0.1 — Run and Record

```
python main.py 2>&1 | tee logs/m5_first_run.log
```

Capture every error, crash, and import failure. Create:

```
docs/ATOM_CURRENT_STATE.md
```

Must contain:
- Import errors (which modules fail on macOS)
- Runtime crashes (which Windows-only paths fire)
- Latency measurements (STT, LLM, TTS, router, end-to-end)
- Memory baseline (RSS at start, after 5min, after 30min)
- CPU/GPU usage during inference
- List of every `if self._is_windows` or Windows-only code path

#### 0.2 — Crash Audit

Scan for every Windows-only dependency. These files WILL crash or silently fail:

| File | Problem | Severity |
|------|---------|----------|
| `core/gpu_governor.py` | pynvml (NVIDIA-only) | Module provides zero data on Mac |
| `core/gpu_resource_manager.py` | VRAM budgets assume discrete GPU | Wrong model for Unified Memory |
| `core/gpu_watchdog.py` | pynvml dependency | Silent failure |
| `core/gpu_scheduler.py` | NVIDIA assumptions | Silent failure |
| `core/gpu_execution_coordinator.py` | NVIDIA assumptions | Silent failure |
| `voice/tts_async.py` | SAPI COM (Windows-only) | Complete crash |
| `core/system_control.py` | PowerShell/winreg/WMI calls in many methods | Partial crash |
| `core/platform_adapter.py` | GPU info returns ('', 0.0) on Mac | Missing data |
| `core/desktop_control.py` | pyautogui works but is limited on Mac | Degraded |
| `context/screen_reader.py` | ImageGrab + PowerShell fallback | Needs macOS screencapture |
| `core/system_scanner.py` | Many WMI-based detection paths | Partial failure |
| `brain/mini_llm.py` | llama-cpp-python (works on Mac but needs Metal build) | Needs correct install |

#### 0.3 — Output

`docs/ATOM_CURRENT_STATE.md` with:
- Every failing module + error
- Avg latency per stage
- Memory footprint
- List of all 40+ Windows code paths that need macOS equivalents

---

### PHASE 1 — MAC SURVIVAL (3-5 days)

**Goal:** ATOM boots, runs, and doesn't crash on M5.

This is NOT optimization. This is making the skeleton work.

#### 1.1 — Fix Platform Adapter for macOS

**File:** `core/platform_adapter.py`

Current state: macOS support exists but is thin. Missing:
- GPU info (returns empty on Apple Silicon)
- Display info (returns "unknown")
- Service management (no macOS impl)
- Audio device enumeration

Add to `_get_gpu_info()`:
```python
if self.os_type == OSType.MACOS:
    result = subprocess.run(
        ["system_profiler", "SPDisplaysDataType", "-json"],
        capture_output=True, text=True, timeout=10,
    )
    data = json.loads(result.stdout)
    gpu = data["SPDisplaysDataType"][0]
    name = gpu.get("sppci_model", "Apple Silicon GPU")
    # Unified Memory: report system RAM as shared GPU memory
    mem = psutil.virtual_memory()
    vram_gb = round(mem.total / (1024**3), 2)
    return name, vram_gb
```

Add to `_get_display_info()`:
```python
if self.os_type == OSType.MACOS:
    result = subprocess.run(
        ["system_profiler", "SPDisplaysDataType", "-json"],
        capture_output=True, text=True, timeout=10,
    )
    # parse resolution from JSON output
```

Add macOS service listing:
```python
elif self.os_type == OSType.MACOS:
    result = subprocess.run(
        ["launchctl", "list"],
        capture_output=True, text=True, timeout=10,
    )
    # parse launchctl output
```

#### 1.2 — Fix GPU Stack for Apple Silicon

**Files:** `core/gpu_governor.py`, `core/gpu_resource_manager.py`

The entire GPU stack assumes NVIDIA. On Apple Silicon:

1. There is NO discrete GPU — CPU, GPU, and Neural Engine share Unified Memory
2. There is NO "VRAM" — it's all system RAM
3. There is NO `pynvml` — use `powermetrics` or `IOKit` via `pyobjc`

Create **`core/apple_silicon_monitor.py`**:
```python
"""Apple Silicon hardware monitoring via IOKit/powermetrics."""

import subprocess
import json

def get_apple_gpu_stats() -> dict:
    """Get Apple Silicon GPU utilization and thermals."""
    try:
        # powermetrics requires sudo, but system_profiler doesn't
        result = subprocess.run(
            ["sudo", "powermetrics", "--samplers", "gpu_power",
             "-i", "1000", "-n", "1", "--format", "plist"],
            capture_output=True, timeout=5,
        )
        # parse plist for GPU active %, frequency, power
    except Exception:
        pass
    
    # Fallback: psutil for memory pressure (since GPU shares RAM)
    import psutil
    mem = psutil.virtual_memory()
    return {
        "available": True,
        "name": "Apple M5 GPU",
        "memory_used_mb": (mem.total - mem.available) / (1024**2),
        "memory_total_mb": mem.total / (1024**2),
        "memory_pct": mem.percent,
        "unified_memory": True,  # KEY: no separate VRAM
    }
```

Update `gpu_governor.py` to use `apple_silicon_monitor` when on macOS.
Update `gpu_resource_manager.py` to treat all memory as shared on Apple Silicon (no eviction needed — models stay in unified memory at no extra cost).

#### 1.3 — Fix TTS for macOS

**File:** `voice/tts_async.py` (SAPI COM — Windows-only, will crash on Mac)

The plan correctly identifies Edge TTS but misses the native option.

macOS has TWO native TTS engines:
1. **`say` command** — instant, zero-latency, offline. Good for acknowledgements.
2. **`AVSpeechSynthesizer`** via pyobjc — programmatic control, voice selection, rate/pitch control.

Create **`voice/tts_macos.py`**:
```python
"""macOS native TTS via NSSpeechSynthesizer (pyobjc) or `say` command."""

import subprocess
import asyncio

class MacOSTTS:
    """Native macOS TTS. Zero latency for short phrases."""
    
    VOICE = "Daniel"  # British male, closest to JARVIS feel
    # Other options: "Alex", "Samantha", "Ava (Premium)"
    
    async def speak(self, text: str, rate: int = 200):
        """Speak using macOS native TTS. ~5ms overhead."""
        proc = await asyncio.create_subprocess_exec(
            "say", "-v", self.VOICE, "-r", str(rate), text,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
    
    def stop(self):
        subprocess.run(["killall", "say"], capture_output=True)
```

Strategy: Use native `say` for quick replies (< 50 chars), Edge TTS for rich responses.

#### 1.4 — Fix Screen Capture for macOS

**File:** `context/screen_reader.py`

Replace PIL/PowerShell screenshot with:
```python
if sys.platform == "darwin":
    subprocess.run(
        ["screencapture", "-x", "-t", "png", str(tmp_path)],
        timeout=5,
    )
```

For OCR, use Apple's Vision framework instead of EasyOCR:
```python
# Via pyobjc-framework-Vision
import Vision
import Quartz

def native_ocr(image_path: str) -> str:
    """On-device OCR via Apple Vision framework. Runs on Neural Engine."""
    image = Quartz.CGImageCreateWithPNGDataProvider(...)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(image, None)
    handler.performRequests_error_([request], None)
    return "\n".join(obs.topCandidates_(1)[0].string() for obs in request.results())
```

This runs on the Neural Engine — 10x faster than EasyOCR, zero external dependencies.

#### 1.5 — Fix System Control for macOS

**File:** `core/system_control.py`

Add macOS implementations for:
- WiFi scan: `airport -s` (at `/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport`)
- Startup programs: parse `~/Library/LaunchAgents/` and `/Library/LaunchAgents/`
- Power plan equivalent: `pmset` command
- Volume control: `osascript -e 'set volume output volume 50'`
- Brightness: `brightness` CLI tool or IOKit

#### 1.6 — Install llama-cpp-python with Metal Support

**Critical:** The default pip install of llama-cpp-python does NOT enable Metal on Apple Silicon. You must build with:

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

This gives full Metal GPU acceleration for LLM inference on M5.

---

### PHASE 2 — STABILITY & COGNITIVE KERNEL (5-7 days)

**Goal:** ATOM is unbreakable. The "brain of the brain" exists.

#### 2.1 — Cognitive Kernel (plan's Phase 1.1 — GOOD idea)

Create **`core/cognitive_kernel.py`**:

```python
"""The Cognitive Kernel — central coordinator for all intelligence decisions.

Responsibilities:
  1. Mode selection (FAST / SMART / DEEP) based on query + system state
  2. Resource allocation (which models to load, memory budget)
  3. Query routing override (skip LLM for known commands)
  4. Model selection (small model vs large model)
  5. Power-aware degradation (battery → reduce everything)
  6. Health-based circuit breaking (if module failing → bypass it)
"""

class CognitiveKernel:
    def route_query(self, query: str, context: dict) -> QueryPlan:
        """Decide the optimal execution path for a query."""
        intent = self.intent_engine.quick_match(query)
        
        if intent:
            return QueryPlan(path="direct", skip_llm=True)
        
        if self.cache.has(query):
            return QueryPlan(path="cache", skip_llm=True)
        
        complexity = classify_query(query)
        if complexity == "SIMPLE":
            return QueryPlan(path="small_model", model="1B")
        
        return QueryPlan(path="full_brain", model="8B")
```

#### 2.2 — Error Isolation (plan's Phase 1.2 — CORRECT)

Wrap every module entry point with structured error handling.
Focus files (exactly as the plan says):
- `core/router/router.py`
- `cursor_bridge/local_brain_controller.py`
- `core/reasoning/action_executor.py`

But ALSO add to:
- `core/boot/wiring.py` (event handler registration — a single bad handler kills the bus)
- `voice/stt_async.py` (mic errors crash the pipeline)
- `voice/tts_edge.py` (network errors during TTS)

#### 2.3 — Watchdog Upgrade (plan's Phase 1.3 — CORRECT)

Add per-module execution budgets:

| Module | Max Time | Action on Timeout |
|--------|----------|-------------------|
| Intent Engine | 50ms | Force return "fallback" |
| Cache Lookup | 100ms | Skip cache, go to LLM |
| RAG Retrieval | 500ms | Skip RAG enrichment |
| LLM Inference | 30s | Kill, return error, restart model |
| TTS Synthesis | 15s | Skip TTS, log error |
| Tool Execution | 10s | Abort tool, return error to LLM |

#### 2.4 — Memory Leak Protection (plan's Phase 1.4 — CORRECT)

Apple Silicon has generous RAM but Unified Memory means LLM + STT + embeddings ALL compete for the same pool.

Critical limits:
```python
MAX_CONVERSATION_TURNS = 20   # already set, correct
MAX_VECTOR_RESULTS = 5        # reduce from default
MAX_RAG_SNIPPETS = 3          # limit RAG context
MAX_MEMORY_GRAPH_NODES = 1000 # cap graph growth
EMBEDDING_CACHE_MAX_MB = 256  # disk cache limit
```

Add periodic memory pressure check:
```python
import psutil
if psutil.virtual_memory().percent > 85:
    # Unload embedding model, reduce caches
    self.degrade_to_minimal()
```

---

### PHASE 3 — SPEED (MLX + Apple Native) (7-10 days)

**Goal:** ATOM feels instant. Sub-second responses for commands.

#### 3.1 — MLX Inference (plan's Phase 2.1 — THE BIGGEST WIN)

Replace `brain/mini_llm.py` (llama-cpp-python) with MLX-based inference.

```bash
pip install mlx mlx-lm
```

Create **`brain/mlx_llm.py`**:

```python
"""ATOM -- MLX-native LLM inference for Apple Silicon.

Why MLX over llama-cpp-python:
  1. Native Apple Silicon optimization (Unified Memory, Metal, ANE-aware)
  2. No memory copies between CPU and GPU (unified memory = zero-copy)
  3. Lazy evaluation — only computes what's needed
  4. ~30-40% faster token generation vs llama.cpp on M-series
  5. Better memory efficiency (shared memory model)
"""

import mlx.core as mx
from mlx_lm import load, generate, stream_generate

class MLXBrain:
    def __init__(self, config: dict):
        self.model_path = config.get("brain", {}).get("mlx_model_path",
            "mlx-community/Qwen2.5-7B-Instruct-4bit")
        self.model = None
        self.tokenizer = None
    
    def load(self):
        self.model, self.tokenizer = load(self.model_path)
    
    async def generate(self, prompt: str, max_tokens: int = 512):
        """Stream tokens from MLX model."""
        for token in stream_generate(
            self.model, self.tokenizer, prompt,
            max_tokens=max_tokens,
            temp=0.7,
        ):
            yield token
```

MLX model options for M5 (ranked by speed/quality tradeoff):

| Model | Size | Speed (M5 est.) | Use Case |
|-------|------|-----------------|----------|
| Qwen2.5-1.5B-Instruct-4bit | ~1GB | ~80 tok/s | Fast path: simple queries, acknowledgements |
| Qwen2.5-7B-Instruct-4bit | ~4GB | ~30 tok/s | Smart path: conversation, reasoning |
| Qwen2.5-14B-Instruct-4bit | ~8GB | ~15 tok/s | Deep path: complex analysis (if RAM allows) |

Download MLX models:
```bash
pip install huggingface_hub
huggingface-cli download mlx-community/Qwen2.5-7B-Instruct-4bit --local-dir models/qwen-7b-mlx
huggingface-cli download mlx-community/Qwen2.5-1.5B-Instruct-4bit --local-dir models/qwen-1.5b-mlx
```

#### 3.2 — Dual-Model Architecture (plan's Phase 2.2 — ENHANCED)

The Cognitive Kernel routes queries to the right model:

```
FAST path:  Intent match → direct action (0ms LLM)
QUICK path: Simple query → 1.5B model (~80 tok/s, ~200ms first token)
SMART path: Conversation → 7B model (~30 tok/s, ~500ms first token)
DEEP path:  Complex reasoning → 7B model + RAG + tools (full pipeline)
```

Both models stay loaded in Unified Memory simultaneously. On Apple Silicon this costs NO extra overhead because there's no GPU↔CPU memory copy.

#### 3.3 — Native Apple STT (NEW — not in original plan)

macOS has `SFSpeechRecognizer` — on-device speech recognition using the Neural Engine.

For COMMAND recognition (< 5 words), native STT is ~50ms vs ~300ms for faster-whisper.

Strategy: Use native STT as a "pre-filter". If it matches a known command with high confidence, skip faster-whisper entirely.

Create **`voice/stt_macos.py`**:
```python
"""Native macOS STT via SFSpeechRecognizer (runs on Neural Engine)."""

# Via pyobjc-framework-Speech
# Falls back to faster-whisper for longer utterances
```

Pipeline becomes:
```
Mic → Native STT (50ms) → Intent match?
  ├── YES → Direct action (skip whisper entirely)
  └── NO  → faster-whisper (300ms) → full pipeline
```

This gives sub-100ms command response for "open chrome", "what time is it", etc.

#### 3.4 — Latency Controller (plan's Phase 2.3 — GOOD, enhanced)

Create **`core/runtime/latency_controller.py`**:

```python
class LatencyController:
    """Dynamic latency management based on system state and query type."""
    
    def get_budget(self, query: str, system_state: dict) -> LatencyBudget:
        on_battery = system_state.get("on_battery", False)
        memory_pressure = system_state.get("memory_pct", 0) > 80
        
        if self.is_cached(query):
            return LatencyBudget(total_ms=50, skip_llm=True)
        
        if on_battery:
            return LatencyBudget(total_ms=2000, model="small", skip_rag=True)
        
        if memory_pressure:
            return LatencyBudget(total_ms=3000, model="small", reduce_context=True)
        
        return LatencyBudget(total_ms=5000, model="large")
```

#### 3.5 — Apple Silicon Tuning

**settings.json updates for M5:**
```json
{
    "brain": {
        "backend": "mlx",
        "mlx_model_path": "models/qwen-7b-mlx",
        "mlx_small_model_path": "models/qwen-1.5b-mlx",
        "n_ctx": 8192,
        "temperature": 0.7
    },
    "stt": {
        "engine": "faster_whisper",
        "use_native_prescan": true,
        "whisper_model_size": "small",
        "compute_type": "int8"
    },
    "tts": {
        "engine": "macos_native",
        "native_voice": "Daniel",
        "edge_voice": "en-GB-RyanNeural",
        "use_edge_for_long": true
    },
    "gpu": {
        "backend": "apple_silicon",
        "unified_memory": true,
        "thermal_threshold": 95
    }
}
```

---

### PHASE 4 — INTELLIGENCE UPGRADE (7-10 days)

**Goal:** ATOM thinks, predicts, and remembers like JARVIS.

#### 4.1 — Cognitive Budget System (plan's Phase 3.1 — CORRECT)

Already partially exists in `core/runtime/modes.py` (FAST/SMART/DEEP/SECURE).
Enhance with concrete execution budgets:

```python
COGNITIVE_BUDGETS = {
    "command":  {"llm": False, "rag": False, "memory": False, "budget_ms": 100},
    "info":     {"llm": False, "rag": False, "memory": True,  "budget_ms": 500},
    "simple":   {"llm": "small", "rag": False, "memory": True,  "budget_ms": 1500},
    "complex":  {"llm": "large", "rag": True,  "memory": True,  "budget_ms": 5000},
    "creative": {"llm": "large", "rag": True,  "memory": True,  "budget_ms": 10000},
}
```

#### 4.2 — Prediction Preload (plan's Phase 3.2 — ENHANCED)

Upgrade `core/cognitive/prediction_engine.py` to preload not just predictions but actual resources:

```python
async def preload_predicted(self, predictions: list[PredictionResult]):
    for pred in predictions:
        if pred.confidence > 0.8:
            if pred.action == "open_app":
                # Pre-warm the app process
                await self.app_preloader.warm(pred.target)
            elif pred.action == "search":
                # Pre-fetch likely search results
                await self.rag_engine.prefetch(pred.target)
            elif pred.action == "llm_query":
                # Pre-load the prompt template
                await self.prompt_builder.precompile(pred.target)
```

#### 4.3 — Smart RAG (plan's Phase 3.3 — CORRECT, add details)

The RAG engine already has most of this. Enhance with:
1. **Temporal decay** — recent memories score higher
2. **Owner-priority** — facts told directly by Boss score 2x
3. **Usage-frequency boost** — frequently retrieved facts score higher
4. **Staleness detection** — mark facts that might be outdated

#### 4.4 — Identity Engine (plan's Phase 3.4 — ENHANCED)

Don't just replace `adaptive_personality.py`. Build on it:

Create **`core/identity_engine.py`**:
```python
class IdentityEngine:
    """ATOM's self-identity and owner relationship model.
    
    ATOM knows:
      - Who it is (ATOM, Satyam's AI OS)
      - Who Boss is (preferences, expertise, mood patterns)
      - What state Boss is in (working, relaxed, frustrated, creative)
      - How to adapt (tone, verbosity, proactivity level)
    """
    
    def get_voice_profile(self, context: dict) -> dict:
        hour = context.get("hour", 12)
        boss_mood = context.get("owner_emotion", "neutral")
        
        if boss_mood == "frustrated":
            return {"tone": "calm", "verbosity": "minimal", "proactive": False}
        if hour < 8:
            return {"tone": "gentle", "verbosity": "brief", "proactive": False}
        if context.get("in_flow_state"):
            return {"tone": "minimal", "verbosity": "terse", "proactive": False}
        
        return {"tone": "confident", "verbosity": "normal", "proactive": True}
```

---

### PHASE 5 — DEEP macOS INTEGRATION (5-8 days)

**Goal:** ATOM controls your Mac like JARVIS controls the lab.

#### 5.1 — AppleScript Engine (plan mentions this — needs depth)

Create **`core/macos/applescript_engine.py`**:

```python
"""AppleScript execution engine for deep macOS control."""

import subprocess

class AppleScriptEngine:
    def run(self, script: str) -> str:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip()
    
    def open_app(self, name: str) -> bool:
        return self.run(f'tell application "{name}" to activate') != ""
    
    def get_frontmost_app(self) -> dict:
        app = self.run('tell application "System Events" to get name of first process whose frontmost is true')
        title = self.run('tell application "System Events" to get title of front window of first process whose frontmost is true')
        return {"app": app, "title": title}
    
    def set_volume(self, level: int) -> bool:
        return self.run(f"set volume output volume {level}") is not None
    
    def get_safari_url(self) -> str:
        return self.run('tell application "Safari" to get URL of current tab of front window')
    
    def get_chrome_url(self) -> str:
        return self.run('tell application "Google Chrome" to get URL of active tab of front window')
    
    def send_notification(self, title: str, message: str) -> bool:
        return self.run(f'display notification "{message}" with title "{title}" sound name "Glass"') is not None
    
    def type_text(self, text: str) -> bool:
        return self.run(f'tell application "System Events" to keystroke "{text}"') is not None
    
    def press_key(self, key: str, modifiers: list = None) -> bool:
        mod_str = ""
        if modifiers:
            mod_str = " using {" + ", ".join(f"{m} down" for m in modifiers) + "}"
        return self.run(f'tell application "System Events" to key code {key}{mod_str}') is not None
```

#### 5.2 — Accessibility API (plan mentions pyobjc — needs specifics)

This is the JARVIS-level capability. The Accessibility API can:
- Read any app's UI element tree (buttons, labels, text fields)
- Click buttons by name ("click the Send button in Slack")
- Read text from any text field
- Monitor UI changes in real-time

Requires: `pip install pyobjc-framework-ApplicationServices`

```python
"""macOS Accessibility API for deep app control."""

import AppKit
import ApplicationServices

def get_focused_element() -> dict:
    """Get the currently focused UI element and its properties."""
    system = ApplicationServices.AXUIElementCreateSystemWide()
    focused = ApplicationServices.AXUIElementCopyAttributeValue(
        system, "AXFocusedUIElement", None
    )
    # Returns: role, title, value, position, size
```

**This is what separates ATOM from every chatbot.** With Accessibility API, ATOM can:
- "Fill in the email field with satyam@example.com"
- "Click the Submit button"
- "Read what's in the error dialog"
- "Switch to the Code tab in VS Code"

#### 5.3 — Spotlight Integration (NEW)

macOS Spotlight can search files, apps, contacts, mail — everything.

```python
def spotlight_search(query: str, limit: int = 10) -> list[dict]:
    """Search using macOS Spotlight."""
    result = subprocess.run(
        ["mdfind", "-limit", str(limit), query],
        capture_output=True, text=True, timeout=10,
    )
    return [{"path": p} for p in result.stdout.strip().split("\n") if p]
```

#### 5.4 — macOS Keychain Integration (NEW)

Replace `security_fortress.py`'s custom EncryptedVault with Keychain:

```python
import subprocess

def keychain_set(service: str, account: str, password: str) -> bool:
    subprocess.run([
        "security", "add-generic-password",
        "-s", service, "-a", account, "-w", password,
        "-U",  # update if exists
    ], capture_output=True)
    return True

def keychain_get(service: str, account: str) -> str:
    result = subprocess.run([
        "security", "find-generic-password",
        "-s", service, "-a", account, "-w",
    ], capture_output=True, text=True)
    return result.stdout.strip()
```

Backed by the M5's Secure Enclave hardware. Way more secure than Fernet encryption in a JSON file.

#### 5.5 — Background Agent via launchd (plan mentions this)

Create **`scripts/com.atom.agent.plist`**:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "...">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.atom.agent</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/python</string>
        <string>/path/to/ATOM/main.py</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/atom.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/atom.err</string>
</dict>
</plist>
```

Install: `launchctl load ~/Library/LaunchAgents/com.atom.agent.plist`

#### 5.6 — FSEvents for File Monitoring (NEW)

macOS has kernel-level file change notifications. Way better than polling.

```python
"""macOS FSEvents for real-time file system monitoring."""

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ATOM can watch Downloads folder, Desktop, project dirs
# and proactively suggest: "Boss, you just downloaded a PDF. Want me to summarize it?"
```

---

### PHASE 6 — AUTONOMY & PROACTIVE BEHAVIOR (5-7 days)

**Goal:** ATOM works without being told.

#### 6.1 — Proactive Engine (plan's Phase 4.1 — CORRECT)

Already exists in `core/cognitive/proactive_engine.py` and `core/proactive_awareness.py`.
Enhance triggers:

```python
PROACTIVE_TRIGGERS = {
    # Time-based
    "morning_briefing": lambda ctx: ctx.hour in (7, 8, 9) and not ctx.briefed_today,
    "end_of_day": lambda ctx: ctx.hour == 18 and ctx.was_active_today,
    
    # System-based
    "low_battery": lambda ctx: ctx.battery < 20 and not ctx.plugged,
    "high_memory": lambda ctx: ctx.memory_pct > 85,
    "disk_full": lambda ctx: ctx.disk_free_gb < 10,
    
    # Behavioral
    "long_idle": lambda ctx: ctx.idle_minutes > 30 and ctx.has_pending_goals,
    "habit_time": lambda ctx: ctx.habit_confidence > 0.9,
    "project_stale": lambda ctx: ctx.days_since_project > 3,
    
    # Context-aware (macOS)
    "new_download": lambda ctx: ctx.new_files_in_downloads,
    "meeting_starting": lambda ctx: ctx.calendar_event_in_5min,
}
```

#### 6.2 — Goal Execution (plan's Phase 4.2 — needs detail)

Upgrade `core/cognitive/goal_engine.py`:
- Goals should generate actionable steps
- Steps should map to tool calls
- Progress should be tracked automatically
- Daily briefing should summarize goal progress

#### 6.3 — Dream Mode (plan's Phase 4.3 — GOOD for idle time)

Already exists at `core/cognitive/dream_engine.py`. Enhance for M5:
- When Mac is idle for 30+ min, run memory consolidation
- Prune low-value memories
- Generate pattern summaries
- Pre-embed frequently accessed documents
- Use the small model (1.5B) to save power during idle

---

### PHASE 7 — TESTING & HARDENING (3-5 days)

#### 7.1 — Performance Targets

| Metric | Current (est.) | Target | How |
|--------|---------------|--------|-----|
| Known command (intent match) | ~200ms | <100ms | Native STT pre-scan |
| Simple query (small model) | ~2s | <500ms | MLX 1.5B model |
| Full conversation (large model) | ~5s | <2s | MLX 7B model |
| TTS first word | ~500ms | <100ms | Native TTS for short replies |
| End-to-end (voice in → voice out) | ~8s | <3s | All optimizations combined |
| Memory (steady state) | ~4GB | <3GB | Cache limits, model optimization |
| Crash rate | Unknown | 0 | Error isolation, watchdog |

#### 7.2 — Stress Tests

Use existing scripts:
```bash
python scripts/v7_stress_test.py
python scripts/v7_chaos_test.py
python scripts/v7_long_run.py
```

Add macOS-specific tests:
- Bluetooth AirPods connect/disconnect during STT
- System sleep/wake cycle
- Memory pressure simulation
- Battery transition (plugged → unplugged)

---

## Apple-Native Technologies Summary

| Apple Technology | ATOM Module It Replaces/Enhances | Benefit |
|-----------------|----------------------------------|---------|
| **MLX** | `brain/mini_llm.py` (llama-cpp-python) | 30-40% faster inference, native unified memory |
| **Metal** | GPU compute (via MLX internally) | Zero-copy GPU access |
| **Unified Memory** | `gpu_resource_manager.py` VRAM model | No memory copies, both models stay loaded |
| **Neural Engine (ANE)** | Embedding model, intent classifier | 4x efficiency, near-zero power |
| **SFSpeechRecognizer** | `voice/stt_async.py` (pre-scan layer) | 50ms command recognition vs 300ms |
| **NSSpeechSynthesizer** | `voice/tts_async.py` | Instant offline TTS, ~5ms overhead |
| **Vision framework** | `context/screen_reader.py` OCR | 10x faster, runs on ANE |
| **Accessibility API** | `core/desktop_control.py` | Read/control ANY app's UI elements |
| **AppleScript/osascript** | `core/platform_adapter.py`, `core/system_control.py` | Deep system automation |
| **Keychain** | `core/security_fortress.py` EncryptedVault | Hardware-backed security (Secure Enclave) |
| **IOKit/powermetrics** | `core/gpu_governor.py` | Apple Silicon thermals, power, health |
| **Spotlight (mdfind)** | File search tool | System-wide instant search |
| **FSEvents** | File monitoring | Kernel-level file change notifications |
| **launchd** | Background service | Proper macOS daemon management |
| **CoreAudio** | Audio device management | Low-latency audio I/O |
| **pmset** | Power management | Battery optimization, sleep control |

---

## Recommended Execution Order

```
Week 1:  PHASE 0 (baseline) + PHASE 1 (Mac survival — fix crashes)
Week 2:  PHASE 2 (stability + cognitive kernel)
Week 3:  PHASE 3 (MLX + speed) — the BIG payoff week
Week 4:  PHASE 4 (intelligence)
Week 5:  PHASE 5 (deep macOS integration)
Week 6:  PHASE 6 (autonomy) + PHASE 7 (testing)
```

Total: ~6 weeks for a single developer working focused evenings/weekends.

---

## Final Verdict

Your original plan is a solid 7.2/10 — the right ideas, mostly the right order. The critical fix is: **macOS compatibility is not a nice-to-have feature for Phase 5. It is the FOUNDATION that must come first, because your entire codebase was built for Windows and will crash before you can even begin optimizing.**

The enhanced plan makes three strategic shifts:
1. **Platform survival first** — fix the 40+ crash points before adding anything new
2. **Apple-native everything** — use what M5 gives you for free instead of fighting it
3. **Unified Memory awareness** — stop thinking in CPU/GPU terms, think in shared memory terms

When you execute this, ATOM won't just run on M5. It will run BETTER than it ever did on any NVIDIA Windows machine, because Apple Silicon's architecture is uniquely suited to an always-on AI OS with shared memory, low power, and instant model switching.
