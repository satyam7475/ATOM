# ATOM M5 Evolution Plan — Rated & Enhanced

> **For:** Satyam (Boss)
> **Hardware:** MacBook Air M5 (Apple Silicon, Unified Memory, Neural Engine, Metal GPU)
> **Date:** April 2026

---

## Plan Rating: 9.3 / 10

> Near production-grade system architecture thinking.

---

### What the Enhanced Plan Gets EXTREMELY RIGHT

#### 1. macOS Compatibility in Phase 0-1 (THE Critical Fix)
Moving macOS compatibility from Phase 5 to Phase 0-1 is what upgraded this plan from 7.2 to 9+. The codebase was Windows-native with ~40 Windows-only code paths — none of the optimizations matter if the system crashes on boot.

#### 2. Apple Silicon Understanding (Senior-Level Clarity)
Correctly identified the fundamental architectural differences:
- No VRAM — Unified Memory model
- No NVIDIA stack — Metal / MLX instead
- ANE (Neural Engine) — free ML acceleration
- Shared memory = zero-copy model loading

This level of hardware awareness is rare and directly impacts every architecture decision.

#### 3. Native Apple Stack Integration
Using the OS itself as the intelligence substrate:
- `SFSpeechRecognizer` (STT) — 50ms command recognition
- `NSSpeechSynthesizer` (TTS) — instant offline speech
- Vision OCR — Neural Engine powered, 10x faster than EasyOCR
- Accessibility API — read/control ANY app's UI
- Keychain — hardware-backed Secure Enclave security
- Spotlight — system-wide instant search
- `launchd` — proper background daemon management

This transforms ATOM from "AI project" into an **Operating Intelligence System**.

#### 4. Dual-Model + MLX Strategy
**Confirmed (2026-04-09): Qwen3-4B + Qwen3-1.7B — best combo for MacBook Air M5.**

| Role | Model | RAM | Speed (M5) | Use |
|------|-------|-----|-----------|-----|
| Primary brain | **Qwen3-4B-Q4_K_M** | ~3.0 GB | 50-70 tok/s | Conversation, reasoning, complex tool calls |
| Fast brain | **Qwen3-1.7B-Q4_K_M** | ~1.2 GB | 120-160 tok/s | Quick acks, simple tool calls, summaries |
| **Total** | | **4.2 GB** | | **5.8 GB headroom on 16 GB** |

Why this combo wins:
- **Same Qwen3 family** = identical ChatML prompt template + tool call format. Zero adaptation to prompt_builder or tool_parser.
- **Qwen3-4B thinking mode** = toggleable deep reasoning. With thinking ON, matches Qwen2.5-7B quality. With thinking OFF, runs at 50-70 tok/s (speed).
- **4B over 8B** = 40% faster, 45% less RAM, no thermal throttle on fanless Air.
- **1.7B over 0.6B** = reliable tool calling. 0.6B too weak for structured JSON output.
- **Bilingual EN/HI** = both models trained on Hindi data (required for Satyam).

Routing (Cognitive Kernel):
- Intent match / known command → skip LLM (sub-5ms)
- Simple tool call / short answer → Qwen3-1.7B (80-150ms)
- Conversation → Qwen3-4B thinking OFF (300-600ms)
- Complex reasoning / ReAct → Qwen3-4B thinking ON (800-2000ms)

#### 5. Cognitive Kernel + Latency Budget
The brain-of-the-brain layer means ATOM is no longer calling LLMs blindly. It decides WHEN to think, HOW MUCH to think, and WHICH model to use based on query complexity, system state, and power budget.

---

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

### PHASE 0 — BASELINE & MAC TRIAGE (2-3 days) ✅ COMPLETE

**Goal:** Know exactly what works, what crashes, what's slow.

**Result:** `docs/ATOM_CURRENT_STATE.md` created. 49 Windows-only code paths identified. Baseline: 879ms boot, 0.21ms intent, 44.8MB RSS. All 4 steps (0.1–0.4) done on 2026-04-09.

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

### PHASE 1 — MAC SURVIVAL (3-5 days) ✅ COMPLETE

**Goal:** ATOM boots, runs, and doesn't crash on M5.

**Result:** All 10 steps (1.1–1.9) done on 2026-04-09. ATOM boots with zero crashes. All Windows-only code has macOS equivalents. Silicon Refactoring removed ~1000 lines of NVIDIA dead code. Native macOS Stack (pyobjc) added STT/TTS/OCR/Media/FSEvents — pulling forward Phase 5 steps 5.6+5.7. System health: 75/100. See `MEMORY_BANK.md` for detailed completion reports.

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

#### 1.7 — Silicon Refactoring (Step 1.8A)

**DONE.** Stripped all NVIDIA/CUDA/pynvml dead code. Created Apple Silicon-native compute layer:

| Old File (DELETED) | Lines | Why Removed |
|---|---|---|
| `core/gpu_governor.py` | 331 | Multi-backend (NVIDIA + Apple Silicon + CPU fallback) — only Apple Silicon needed |
| `core/gpu_resource_manager.py` | 387 | VRAM budgets, slot allocation, load grant tokens — wrong model for Unified Memory |
| `core/gpu_execution_coordinator.py` | 593 | VRAM admission, fragmentation heuristics — unnecessary on Apple Silicon |

| New File (CREATED) | Lines | Purpose |
|---|---|---|
| `core/silicon_governor.py` | 145 | Apple Silicon-only hardware monitor + thermal/memory events |
| `core/inference_guard.py` | 155 | Model lifecycle: slot tracking + Unified Memory pressure + idle unload |

**Net result:** ~1,000 lines removed. 5-file, 3-backend, 3-state-class stack → 3-file, 1-backend, 1-state-class stack.

**Architectural principle:** Apple Silicon has Unified Memory. CPU, GPU, and Neural Engine share the same RAM pool. There is no VRAM, no discrete GPU, no CPU-vs-GPU decision. ATOM monitors thermal pressure and memory usage — the SoC handles the rest.

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

#### 2.2 — Event Priority System (NEW — CRITICAL)

The event bus exists but has no priority scheduling. When multiple events arrive simultaneously (voice input, system trigger, background task), the system needs deterministic ordering to prevent lag, race conditions, and chaos.

Create **`core/event_priority.py`**:

```python
"""Event Priority System — deterministic scheduling for concurrent events.

Without this, simultaneous events (voice command + system alert + background task)
compete randomly, causing lag and unpredictable behavior.
"""

from enum import IntEnum
import heapq
import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable
import time

class EventPriority(IntEnum):
    CRITICAL = 0   # voice commands, security alerts — always first
    HIGH = 1       # system alerts, user-triggered actions
    NORMAL = 2     # background tasks, scheduled jobs
    LOW = 3        # learning, dream mode, memory consolidation

@dataclass(order=True)
class PrioritizedEvent:
    priority: int
    timestamp: float = field(compare=True)
    event_type: str = field(compare=False)
    payload: Any = field(compare=False)
    callback: Callable = field(compare=False, default=None)

class EventScheduler:
    """Priority queue for ATOM events. Higher priority = processed first.
    
    Prevents:
      - Voice commands waiting behind background tasks
      - Race conditions between concurrent event sources
      - System alerts getting buried under low-priority work
    """
    
    def __init__(self):
        self._queue: list[PrioritizedEvent] = []
        self._lock = asyncio.Lock()
    
    async def submit(self, event_type: str, payload: Any,
                     priority: EventPriority = EventPriority.NORMAL,
                     callback: Callable = None):
        async with self._lock:
            event = PrioritizedEvent(
                priority=priority.value,
                timestamp=time.monotonic(),
                event_type=event_type,
                payload=payload,
                callback=callback,
            )
            heapq.heappush(self._queue, event)
    
    async def next(self) -> PrioritizedEvent | None:
        async with self._lock:
            if self._queue:
                return heapq.heappop(self._queue)
            return None
    
    async def process_loop(self):
        while True:
            event = await self.next()
            if event and event.callback:
                await event.callback(event.payload)
            else:
                await asyncio.sleep(0.01)
```

Integration points:
- Voice input events → `CRITICAL` priority
- System alerts (low battery, high memory) → `HIGH` priority
- Proactive suggestions, scheduled tasks → `NORMAL` priority
- Dream mode, memory consolidation → `LOW` priority

#### 2.3 — State Manager (NEW — VERY IMPORTANT)

Context and memory exist, but there is no single source of truth for the global system state. The State Manager tracks real-time awareness across all dimensions — user, system, and environment.

Create **`core/state_manager.py`**:

```python
"""Global State Manager — real-time system awareness.

This is what makes ATOM truly context-aware. Every module reads from here,
and relevant modules write to it. The Cognitive Kernel uses this to make
intelligent routing decisions.
"""

import psutil
import time
import asyncio
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ATOMState:
    # Cognitive state
    mode: str = "FAST"            # FAST / SMART / DEEP
    active_model: str = "none"    # which LLM is currently loaded
    last_intent: str = ""
    last_query_time: float = 0.0
    
    # User state
    user_state: str = "idle"      # working / idle / focused / away
    owner_emotion: str = "neutral"
    idle_minutes: float = 0.0
    
    # System state
    system_load: float = 0.0
    memory_pct: float = 0.0
    battery: int = 100
    on_battery: bool = False
    thermal_pressure: str = "nominal"  # nominal / fair / serious / critical
    
    # Environment state
    active_app: str = ""
    active_window: str = ""
    hour: int = 0
    
    # Session state
    session_start: float = field(default_factory=time.time)
    queries_this_session: int = 0
    errors_this_session: int = 0

class StateManager:
    """Centralized state for all ATOM modules.
    
    Read by: Cognitive Kernel, Latency Controller, Proactive Engine, Identity Engine
    Written by: System monitor, Voice pipeline, App tracker, Battery monitor
    """
    
    def __init__(self):
        self._state = ATOMState()
        self._listeners: list = []
    
    @property
    def state(self) -> ATOMState:
        return self._state
    
    async def update(self, **kwargs):
        changed = {}
        for key, value in kwargs.items():
            if hasattr(self._state, key):
                old = getattr(self._state, key)
                if old != value:
                    setattr(self._state, key, value)
                    changed[key] = (old, value)
        if changed:
            await self._notify_listeners(changed)
    
    async def refresh_system_state(self):
        """Poll system metrics. Called periodically by watchdog."""
        mem = psutil.virtual_memory()
        battery = psutil.sensors_battery()
        
        await self.update(
            system_load=psutil.cpu_percent(interval=0.1),
            memory_pct=mem.percent,
            battery=int(battery.percent) if battery else 100,
            on_battery=not battery.power_plugged if battery else False,
            hour=time.localtime().tm_hour,
        )
    
    def on_change(self, callback):
        self._listeners.append(callback)
    
    async def _notify_listeners(self, changes: dict):
        for listener in self._listeners:
            await listener(changes)
    
    def snapshot(self) -> dict:
        """Return full state as dict for logging/debugging."""
        return {
            "mode": self._state.mode,
            "user_state": self._state.user_state,
            "system_load": self._state.system_load,
            "memory_pct": self._state.memory_pct,
            "battery": self._state.battery,
            "active_app": self._state.active_app,
            "last_intent": self._state.last_intent,
            "queries": self._state.queries_this_session,
            "errors": self._state.errors_this_session,
        }
```

The Cognitive Kernel, Latency Controller, and Proactive Engine all read from this single state object to make decisions. No more scattered `psutil` calls across modules.

#### 2.4 — Security Execution Sandbox (NEW — CRITICAL)

The security fortress validates permissions, but tools can still execute risky commands without a dry-run check. The sandbox simulates actions before executing them.

Create **`core/execution_sandbox.py`**:

```python
"""Execution Sandbox — simulate-before-execute safety layer.

Flow:
  LLM proposes action → sandbox simulates → safe? execute : ask user

Prevents:
  - Accidental file deletion
  - Dangerous system commands
  - Unintended privilege escalation
"""

from dataclasses import dataclass
from enum import Enum

class RiskLevel(Enum):
    SAFE = "safe"               # read-only, no side effects
    LOW = "low"                 # reversible side effects
    MEDIUM = "medium"           # significant but recoverable
    HIGH = "high"               # destructive or irreversible
    CRITICAL = "critical"       # system-level, requires explicit approval

DANGEROUS_PATTERNS = [
    "rm -rf", "sudo rm", "mkfs", "dd if=", "chmod 777",
    "kill -9", "shutdown", "reboot", "launchctl unload",
    "> /dev/", "curl | sh", "eval(", "exec(",
]

SENSITIVE_PATHS = [
    "/System", "/Library", "/usr", "/bin", "/sbin",
    "~/.ssh", "~/.gnupg", "~/Library/Keychains",
]

@dataclass
class SandboxResult:
    allowed: bool
    risk_level: RiskLevel
    reason: str
    requires_confirmation: bool = False

class ExecutionSandbox:
    """Pre-execution safety check for all tool actions."""
    
    def evaluate(self, action: str, args: dict) -> SandboxResult:
        command = args.get("command", "")
        path = args.get("path", "")
        
        # Check for dangerous command patterns
        for pattern in DANGEROUS_PATTERNS:
            if pattern in command:
                return SandboxResult(
                    allowed=False,
                    risk_level=RiskLevel.CRITICAL,
                    reason=f"Blocked dangerous pattern: {pattern}",
                    requires_confirmation=True,
                )
        
        # Check for sensitive path access
        for sensitive in SENSITIVE_PATHS:
            if sensitive in path:
                return SandboxResult(
                    allowed=False,
                    risk_level=RiskLevel.HIGH,
                    reason=f"Sensitive path access: {sensitive}",
                    requires_confirmation=True,
                )
        
        # Read-only actions are always safe
        if action in ("read_file", "search", "list_dir", "get_info"):
            return SandboxResult(
                allowed=True, risk_level=RiskLevel.SAFE,
                reason="Read-only operation"
            )
        
        # Write actions need medium caution
        if action in ("write_file", "create_file", "move_file"):
            return SandboxResult(
                allowed=True, risk_level=RiskLevel.LOW,
                reason="Write operation — reversible",
                requires_confirmation=False,
            )
        
        # Unknown actions default to requiring confirmation
        return SandboxResult(
            allowed=False, risk_level=RiskLevel.MEDIUM,
            reason="Unknown action type — requesting confirmation",
            requires_confirmation=True,
        )
```

Integration: The `action_executor.py` calls `sandbox.evaluate()` before every tool execution. If `requires_confirmation` is True, ATOM asks: "Boss, this looks risky. Should I proceed?"

#### 2.5 — Error Isolation (plan's Phase 1.2 — CORRECT)

Wrap every module entry point with structured error handling.
Focus files (exactly as the plan says):
- `core/router/router.py`
- `cursor_bridge/local_brain_controller.py`
- `core/reasoning/action_executor.py`

But ALSO add to:
- `core/boot/wiring.py` (event handler registration — a single bad handler kills the bus)
- `voice/stt_async.py` (mic errors crash the pipeline)
- `voice/tts_edge.py` (network errors during TTS)

#### 2.6 — Watchdog Upgrade (plan's Phase 1.3 — CORRECT)

Add per-module execution budgets:

| Module | Max Time | Action on Timeout |
|--------|----------|-------------------|
| Intent Engine | 50ms | Force return "fallback" |
| Cache Lookup | 100ms | Skip cache, go to LLM |
| RAG Retrieval | 500ms | Skip RAG enrichment |
| LLM Inference | 30s | Kill, return error, restart model |
| TTS Synthesis | 15s | Skip TTS, log error |
| Tool Execution | 10s | Abort tool, return error to LLM |

#### 2.7 — Memory Leak Protection (plan's Phase 1.4 — CORRECT)

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
        self.primary_path = config.get("brain", {}).get("mlx_primary_model",
            "mlx-community/Qwen3-4B-4bit")
        self.fast_path = config.get("brain", {}).get("mlx_fast_model",
            "mlx-community/Qwen3-1.7B-4bit")
        self.primary_model = None
        self.fast_model = None
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

**Confirmed models for M5 (2026-04-09):**

| Role | Model | Size (RAM) | Speed (M5) | Use Case |
|------|-------|-----------|-----------|----------|
| **Fast brain** | Qwen3-1.7B-Q4_K_M | ~1.2 GB | 120-160 tok/s | Quick acks, simple tool calls, summaries |
| **Primary brain** | Qwen3-4B-Q4_K_M | ~3.0 GB | 50-70 tok/s | Conversation, reasoning, complex tools |
| **Total** | | **~4.2 GB** | | **5.8 GB headroom on 16 GB M5** |

Why Qwen3 family: same ChatML prompt template, same tool_call format, zero adaptation to structured_prompt_builder.py or tool_parser.py. Thinking mode on Qwen3-4B toggles deep reasoning (matches 7B quality) vs speed.

Download MLX models:
```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-4B-GGUF qwen3-4b-q4_k_m.gguf --local-dir models/
huggingface-cli download Qwen/Qwen3-1.7B-GGUF qwen3-1.7b-q4_k_m.gguf --local-dir models/
# MLX format (when Phase 3 MLX migration happens):
# huggingface-cli download mlx-community/Qwen3-4B-4bit --local-dir models/qwen3-4b-mlx
# huggingface-cli download mlx-community/Qwen3-1.7B-4bit --local-dir models/qwen3-1.7b-mlx
```

#### 3.2 — Dual-Model Architecture (plan's Phase 2.2 — ENHANCED)

The Cognitive Kernel routes queries to the right model:

```
FAST path:   Intent match → direct action (skip LLM, sub-5ms)
QUICK path:  Simple query → Qwen3-1.7B (120-160 tok/s, 80-150ms response)
SMART path:  Conversation → Qwen3-4B thinking OFF (50-70 tok/s, 300-600ms)
DEEP path:   Complex reasoning → Qwen3-4B thinking ON + RAG + tools (800-2000ms)
```

Both models stay loaded in Unified Memory simultaneously. On Apple Silicon this costs NO extra overhead because there's no GPU↔CPU memory copy. Thinking mode toggle gives two speeds from the same model.

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

#### 3.5 — Cold Start Optimization (NEW — UX CRITICAL)

First response delay is the biggest UX killer. If ATOM takes 5 seconds to respond after boot, it feels dead. Cold start optimization ensures the first command is instant.

Add to **`core/boot/cold_start.py`**:

```python
"""Cold Start Optimization — preload essentials at boot for instant first response.

At ATOM boot, the following are preloaded into memory:
  1. Small LLM (1.5B) — ready for fast-path queries immediately
  2. Embeddings model — ready for RAG/memory lookups
  3. Last session memory — conversational continuity
  4. Top commands cache — most frequent intents pre-matched
  5. State Manager snapshot — last known system state

Result: First command after boot = instant response.
"""

import asyncio
import time

class ColdStartOptimizer:
    def __init__(self, config: dict, state_manager, model_loader, memory_store):
        self.config = config
        self.state = state_manager
        self.loader = model_loader
        self.memory = memory_store
        self._boot_time = None
    
    async def warm_up(self):
        """Run at ATOM boot. Preloads everything needed for instant first response."""
        self._boot_time = time.monotonic()
        
        await asyncio.gather(
            self._preload_small_model(),
            self._preload_embeddings(),
            self._restore_session(),
            self._cache_top_commands(),
            self._restore_state_snapshot(),
        )
        
        elapsed = time.monotonic() - self._boot_time
        await self.state.update(mode="FAST")
        return elapsed
    
    async def _preload_small_model(self):
        """Load 1.5B model first — it's fast and handles 70% of queries."""
        await self.loader.load("small")
    
    async def _preload_embeddings(self):
        """Load embedding model for memory/RAG lookups."""
        await self.loader.load("embeddings")
    
    async def _restore_session(self):
        """Restore last conversation context for continuity."""
        await self.memory.restore_last_session()
    
    async def _cache_top_commands(self):
        """Pre-fill intent cache with the 50 most frequent commands."""
        top = await self.memory.get_top_intents(limit=50)
        for intent in top:
            self.loader.intent_cache.put(intent.pattern, intent.action)
    
    async def _restore_state_snapshot(self):
        """Restore last known system state to avoid cold polling delay."""
        snapshot = await self.memory.get_last_state_snapshot()
        if snapshot:
            await self.state.update(**snapshot)
```

Boot sequence becomes:
```
ATOM boot → ColdStartOptimizer.warm_up() → [parallel preload all 5 items]
           → "Ready, Boss." (first response available in <2s)
```

#### 3.6 — Voice Interrupt System (NEW — JARVIS FEATURE)

Currently ATOM speaks and the user waits. Real JARVIS-level interaction means the user can interrupt ATOM mid-speech and ATOM immediately switches to listening.

Add to **`voice/interrupt_handler.py`**:

```python
"""Voice Interrupt System — JARVIS-level conversational flow.

Problem: ATOM speaks, user waits. Not natural.
Solution: If user speaks while ATOM is talking, immediately:
  1. Stop TTS playback
  2. Switch to listening mode
  3. Process the new input

This makes ATOM feel alive and responsive.
"""

import asyncio
from typing import Optional

class VoiceInterruptHandler:
    def __init__(self, tts_engine, stt_engine, vad_detector):
        self.tts = tts_engine
        self.stt = stt_engine
        self.vad = vad_detector
        self._speaking = False
        self._monitor_task: Optional[asyncio.Task] = None
    
    async def speak_interruptible(self, text: str):
        """Speak text but allow user to interrupt at any point."""
        self._speaking = True
        self._monitor_task = asyncio.create_task(self._monitor_for_interrupt())
        
        try:
            await self.tts.speak(text)
        except asyncio.CancelledError:
            pass
        finally:
            self._speaking = False
            if self._monitor_task:
                self._monitor_task.cancel()
    
    async def _monitor_for_interrupt(self):
        """Continuously check for voice activity during TTS playback."""
        while self._speaking:
            if await self.vad.detect_voice():
                await self._handle_interrupt()
                return
            await asyncio.sleep(0.05)  # 50ms polling
    
    async def _handle_interrupt(self):
        """User spoke while ATOM was talking. Switch immediately."""
        self.tts.stop()
        self._speaking = False
        # Pipeline now flows back to STT → processing
```

Flow:
```
ATOM speaking → user starts talking
  → VAD detects voice → TTS stops immediately
  → STT activates → new query processed
  → feels like natural conversation
```

#### 3.7 — Streaming Response System (NEW — PERCEIVED SPEED)

JARVIS doesn't wait for the full response — it streams. Token-by-token generation piped directly to TTS gives the perception of instant response even when the LLM is still thinking.

Add to **`core/streaming_pipeline.py`**:

```python
"""Streaming Response System — LLM tokens → TTS chunks → live speech.

Instead of: LLM generates full response → TTS synthesizes → play audio
Stream:     LLM generates token → buffer sentence → TTS chunk → play immediately

Result: User hears the first word within 200ms of LLM start,
        even if the full response takes 5 seconds to generate.
"""

import asyncio
from typing import AsyncIterator

class StreamingPipeline:
    SENTENCE_DELIMITERS = {'.', '!', '?', ',', ';', ':'}
    MIN_CHUNK_CHARS = 20  # don't send tiny fragments to TTS
    
    def __init__(self, tts_engine, interrupt_handler):
        self.tts = tts_engine
        self.interrupt = interrupt_handler
    
    async def stream_to_speech(self, token_stream: AsyncIterator[str]):
        """Pipe LLM token stream directly to TTS in sentence chunks."""
        buffer = ""
        
        async for token in token_stream:
            buffer += token
            
            if (len(buffer) >= self.MIN_CHUNK_CHARS and
                    buffer[-1] in self.SENTENCE_DELIMITERS):
                chunk = buffer.strip()
                buffer = ""
                
                # Speak this chunk while LLM continues generating
                await self.interrupt.speak_interruptible(chunk)
        
        # Flush remaining buffer
        if buffer.strip():
            await self.interrupt.speak_interruptible(buffer.strip())
```

Pipeline becomes:
```
LLM → token stream → sentence buffer → TTS chunk → speak live
  ↕ (parallel)         ↕ (parallel)
  generating...        user hearing first words
```

Combined with the Voice Interrupt System, this creates a fully bidirectional, streaming conversation loop — the core of JARVIS-level interaction.

#### 3.8 — Apple Silicon Tuning

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
| Simple query (Qwen3-1.7B) | ~2s | <150ms | MLX fast brain, thinking OFF |
| Full conversation (Qwen3-4B) | ~5s | <600ms | MLX primary brain, thinking OFF |
| Complex reasoning (Qwen3-4B) | ~5s | <2s | MLX primary brain, thinking ON |
| TTS first word | ~500ms | <100ms | Native TTS for short replies |
| End-to-end (voice in → voice out) | ~8s | <3s | All optimizations combined |
| Memory (steady state) | ~4GB | <3GB | Cache limits, model optimization |
| Crash rate | Unknown | 0 | Error isolation, watchdog |

#### 7.2 — Observability Dashboard (NEW — PRODUCTION REQUIREMENT)

For a system this complex, you NEED visibility into what's happening in real time. Without it, debugging is guesswork and performance regressions go unnoticed.

Create **`tools/observability_dashboard.py`**:

```python
"""Observability Dashboard — real-time ATOM system visibility.

Shows:
  - Latency per module (STT, LLM, TTS, Router, Tools)
  - Memory usage breakdown (models, caches, system)
  - Active model and mode
  - Event flow and priority queue depth
  - Error rate and recent failures
  - System state snapshot
"""

import time
import asyncio
from collections import deque
from dataclasses import dataclass, field

@dataclass
class ModuleMetrics:
    name: str
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    call_count: int = 0
    error_count: int = 0
    last_call_time: float = 0.0
    _latencies: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def record(self, latency_ms: float, error: bool = False):
        self._latencies.append(latency_ms)
        self.call_count += 1
        self.last_call_time = time.time()
        if error:
            self.error_count += 1
        
        latencies = sorted(self._latencies)
        self.avg_latency_ms = sum(latencies) / len(latencies)
        self.p95_latency_ms = latencies[int(len(latencies) * 0.95)] if latencies else 0

class ObservabilityDashboard:
    """Collects and exposes metrics from all ATOM modules."""
    
    TRACKED_MODULES = [
        "stt", "intent_engine", "router", "llm_small", "llm_large",
        "rag", "memory", "tts", "tool_executor", "state_manager",
    ]
    
    def __init__(self, state_manager):
        self.state = state_manager
        self.modules = {name: ModuleMetrics(name) for name in self.TRACKED_MODULES}
        self._event_log: deque = deque(maxlen=500)
    
    def record_module_call(self, module: str, latency_ms: float, error: bool = False):
        if module in self.modules:
            self.modules[module].record(latency_ms, error)
    
    def log_event(self, event_type: str, details: str):
        self._event_log.append({
            "time": time.time(),
            "type": event_type,
            "details": details,
        })
    
    def get_dashboard_data(self) -> dict:
        """Full dashboard snapshot for UI rendering."""
        return {
            "system_state": self.state.snapshot(),
            "modules": {
                name: {
                    "avg_latency_ms": m.avg_latency_ms,
                    "p95_latency_ms": m.p95_latency_ms,
                    "calls": m.call_count,
                    "errors": m.error_count,
                }
                for name, m in self.modules.items()
            },
            "recent_events": list(self._event_log)[-20:],
            "health": self._compute_health(),
        }
    
    def _compute_health(self) -> str:
        total_errors = sum(m.error_count for m in self.modules.values())
        total_calls = sum(m.call_count for m in self.modules.values())
        if total_calls == 0:
            return "idle"
        error_rate = total_errors / total_calls
        if error_rate > 0.1:
            return "degraded"
        if error_rate > 0.01:
            return "warning"
        return "healthy"
```

Optional: Expose as a local web UI on `localhost:9090` using a simple FastAPI/websocket server, or render in the terminal using `rich` tables. Even a simple `GET /health` endpoint is invaluable for monitoring.

Dashboard displays:
| Metric | Source |
|--------|--------|
| Latency per module | `ModuleMetrics` recorded at each call site |
| Memory usage | State Manager + `psutil` |
| Active model | State Manager (`active_model` field) |
| Event flow | Event Priority System queue depth |
| Error rate | Per-module error counters |
| System health | Computed from error rate + latency thresholds |

#### 7.3 — Stress Tests

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
| **MLX** | `brain/mini_llm.py` (llama-cpp-python) | 30-40% faster inference, native unified memory. Planned: Qwen3-4B + Qwen3-1.7B dual-model |
| **Metal** | GPU compute (via MLX internally) | Zero-copy GPU access |
| **Unified Memory** | `inference_guard.py` (replaced `gpu_resource_manager.py`) | No memory copies, Qwen3-4B (3GB) + Qwen3-1.7B (1.2GB) stay loaded simultaneously |
| **Neural Engine (ANE)** | Embedding model, intent classifier | 4x efficiency, near-zero power |
| **SFSpeechRecognizer** | `voice/stt_async.py` (pre-scan layer) | 50ms command recognition vs 300ms |
| **NSSpeechSynthesizer** | `voice/tts_async.py` | Instant offline TTS, ~5ms overhead |
| **Vision framework** | `context/screen_reader.py` OCR | 10x faster, runs on ANE |
| **Accessibility API** | `core/desktop_control.py` | Read/control ANY app's UI elements |
| **AppleScript/osascript** | `core/platform_adapter.py`, `core/system_control.py` | Deep system automation |
| **Keychain** | `core/security_fortress.py` EncryptedVault | Hardware-backed security (Secure Enclave) |
| **IOKit/powermetrics** | `core/silicon_governor.py` + `core/apple_silicon_monitor.py` (replaced `gpu_governor.py`) | Apple Silicon thermals, power, health |
| **Spotlight (mdfind)** | File search tool | System-wide instant search |
| **FSEvents** | File monitoring | Kernel-level file change notifications |
| **launchd** | Background service | Proper macOS daemon management |
| **CoreAudio** | Audio device management | Low-latency audio I/O |
| **pmset** | Power management | Battery optimization, sleep control |

## New Core Systems Summary (9.3/10 Additions)

| System | File | Phase | Purpose |
|--------|------|-------|---------|
| **Event Priority System** | `core/event_priority.py` | Phase 2 | Deterministic scheduling — voice commands always first, dream mode always last |
| **State Manager** | `core/state_manager.py` | Phase 2 | Single source of truth for system, user, and environment state |
| **Execution Sandbox** | `core/execution_sandbox.py` | Phase 2 | Simulate-before-execute safety — blocks dangerous commands, asks for confirmation |
| **Cold Start Optimizer** | `core/boot/cold_start.py` | Phase 3 | Preloads models, caches, session memory at boot — first command is instant |
| **Voice Interrupt Handler** | `voice/interrupt_handler.py` | Phase 3 | User can interrupt ATOM mid-speech — immediate switch to listening |
| **Streaming Pipeline** | `core/streaming_pipeline.py` | Phase 3 | Token-to-speech streaming — user hears first word within 200ms |
| **Observability Dashboard** | `tools/observability_dashboard.py` | Phase 7 | Real-time visibility: latency, memory, errors, event flow, system health |

---

## ATOM Final Architecture — 7 Core Systems

After all enhancements, ATOM is composed of 7 interconnected core systems. Every module, file, and feature maps into one of these:

```
┌─────────────────────────────────────────────────────────────┐
│                    ATOM CORE SYSTEMS                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. PERCEPTION LAYER                                        │
│     ├── STT (SFSpeechRecognizer + faster-whisper)           │
│     ├── Screen Reader (Vision OCR + screencapture)          │
│     ├── Context Awareness (active app, window, clipboard)   │
│     ├── Voice Activity Detection (interrupt system)         │
│     └── FSEvents (file system monitoring)                   │
│                                                             │
│  2. COGNITIVE KERNEL (Decision Brain)                       │
│     ├── Query Router (FAST / QUICK / SMART / DEEP)          │
│     ├── Intent Engine (pattern match → skip LLM)            │
│     ├── Latency Controller (budget per query type)          │
│     ├── Event Priority Scheduler (CRITICAL → LOW)           │
│     └── Cognitive Budget System (resource allocation)       │
│                                                             │
│  3. STATE MANAGER (Real-time Awareness)                     │
│     ├── System State (CPU, memory, battery, thermal)        │
│     ├── User State (working, idle, focused, away)           │
│     ├── Environment State (active app, hour, location)      │
│     ├── Session State (queries, errors, uptime)             │
│     └── Mode Controller (FAST / SMART / DEEP switching)     │
│                                                             │
│  4. INTELLIGENCE LAYER (LLM + RAG + Memory)                │
│     ├── MLX Brain (Qwen3-1.7B fast + Qwen3-4B smart)        │
│     ├── RAG Engine (temporal decay, owner-priority)         │
│     ├── Memory Graph (episodic + semantic + procedural)     │
│     ├── Prediction Engine (preload predicted actions)       │
│     ├── Identity Engine (personality, owner model)          │
│     └── Streaming Response Pipeline (token → TTS live)      │
│                                                             │
│  5. EXECUTION LAYER (Tools + Sandbox)                       │
│     ├── Tool Executor (file, web, system, app tools)        │
│     ├── Execution Sandbox (simulate → safe? → execute)      │
│     ├── Action Executor (LLM decisions → real actions)      │
│     └── Security Fortress (encryption, access control)      │
│                                                             │
│  6. OS INTEGRATION (macOS Native APIs)                      │
│     ├── AppleScript Engine (deep app control)               │
│     ├── Accessibility API (UI element read/write)           │
│     ├── Spotlight Integration (system-wide search)          │
│     ├── Keychain (Secure Enclave credential storage)        │
│     ├── Apple Silicon Monitor (IOKit, powermetrics)         │
│     ├── launchd Agent (background daemon)                   │
│     ├── Native TTS (NSSpeechSynthesizer + say)              │
│     ├── Native STT (SFSpeechRecognizer)                     │
│     └── Vision OCR (Neural Engine powered)                  │
│                                                             │
│  7. AUTONOMY LAYER (Prediction + Goals + Self-Improvement)  │
│     ├── Proactive Engine (trigger-based suggestions)        │
│     ├── Goal Engine (multi-step autonomous execution)       │
│     ├── Dream Mode (idle-time memory consolidation)         │
│     ├── Cold Start Optimizer (instant boot readiness)       │
│     └── Observability Dashboard (self-monitoring)           │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  CROSS-CUTTING: Voice Interrupt System, Event Bus,          │
│  Watchdog, Error Isolation, Memory Leak Protection          │
└─────────────────────────────────────────────────────────────┘
```

Data flow through the 7 systems:

```
User speaks
  → [1. Perception] STT captures → VAD detects speech
  → [3. State Manager] updates user_state, last_query_time
  → [2. Cognitive Kernel] classifies intent, picks route + model
  → [4. Intelligence] LLM generates (streaming tokens)
  → [5. Execution] sandbox checks → tool runs if needed
  → [6. OS Integration] AppleScript / Accessibility if needed
  → [7. Autonomy] prediction engine preloads next likely action
  → [1. Perception] streaming TTS speaks response (interruptible)
  → [3. State Manager] records query, updates metrics
```

---

## Recommended Execution Order

```
Week 1:  PHASE 0 (baseline) + PHASE 1 (Mac survival — fix crashes)
Week 2:  PHASE 2 (stability + cognitive kernel + state manager + event priority + sandbox)
Week 3:  PHASE 3 (MLX + Qwen3-4B/1.7B dual-model + cold start + streaming) — BIG payoff
Week 4:  PHASE 4 (intelligence + identity + prediction)
Week 5:  PHASE 5 (deep macOS integration — Accessibility, AppleScript, Keychain)
Week 6:  PHASE 6 (autonomy + proactive engine)
Week 7:  PHASE 7 (testing + observability dashboard + hardening)
```

Total: ~7 weeks for a single developer working focused evenings/weekends.

The extra week accounts for the three new Phase 2 systems (Event Priority, State Manager, Execution Sandbox) and the three new Phase 3 systems (Cold Start, Voice Interrupt, Streaming Response) which are essential for the 10/10 architecture.

---

## Final Verdict

**Rating: 9.3/10** — near production-grade system architecture.

The enhanced plan now covers all 7 core systems needed for a true Operating Intelligence System. The critical additions that push from 7.2 to 9.3:

1. **Platform survival first** — macOS compatibility in Phase 0-1, not Phase 5
2. **Apple-native everything** — use what M5 gives you for free instead of fighting it
3. **Unified Memory awareness** — stop thinking in CPU/GPU terms, think in shared memory
4. **Event Priority System** — deterministic scheduling prevents chaos under load
5. **State Manager** — single source of truth for real-time context awareness
6. **Cold Start Optimization** — instant first response eliminates the biggest UX killer
7. **Execution Sandbox** — simulate-before-execute safety layer for all tool actions
8. **Voice Interrupt System** — JARVIS-level conversational flow (user can interrupt ATOM)
9. **Streaming Response** — token-to-speech pipeline for perceived instant responses
10. **Observability Dashboard** — production-grade visibility into system health

When you execute this, ATOM won't just run on M5. It will run BETTER than it ever did on any NVIDIA Windows machine, because Apple Silicon's architecture is uniquely suited to an always-on AI OS with shared memory, low power, and instant model switching.
