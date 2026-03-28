# ATOM Architecture Blueprint v3.0

> **Owner:** Satyam | **System:** ATOM v19 (Cognitive AI Operating System — JARVIS-Level Intelligence, Cross-Platform, GPU-Accelerated)
> **Purpose:** The single source of truth for understanding, building, and evolving ATOM.
> **Philosophy:** Every module is a replaceable organ. ATOM is a living system — it grows, it adapts, it never stops evolving.

---

## Table of Contents

1. [System Identity](#1-system-identity)
2. [Architecture Layers (The 7 Rings)](#2-architecture-layers-the-7-rings)
3. [Module Registry — Every Organ Mapped](#3-module-registry--every-organ-mapped)
4. [The Nervous System — Event Bus Contract](#4-the-nervous-system--event-bus-contract)
5. [The Brain Pipeline — How ATOM Thinks](#5-the-brain-pipeline--how-atom-thinks)
6. [State Machine — The Heartbeat](#6-state-machine--the-heartbeat)
7. [Interface Contracts — Upgrade Without Breaking](#7-interface-contracts--upgrade-without-breaking)
8. [Data Flow Maps](#8-data-flow-maps)
9. [Configuration Architecture](#9-configuration-architecture)
10. [Security Architecture](#10-security-architecture)
11. [Performance Architecture](#11-performance-architecture)
12. [Evolution Roadmap — The ATOM Infinity Path](#12-evolution-roadmap--the-atom-infinity-path)
13. [Module Upgrade Playbook](#13-module-upgrade-playbook)
14. [Build Order — What to Build First](#14-build-order--what-to-build-first)

---

## 1. System Identity

**ATOM** is not an assistant. It is a **Personal Cognitive AI Operating System** — a software entity that:
- **Perceives** (voice, system state, user behavior, context)
- **Thinks** (intent classification, LLM reasoning, prediction)
- **Acts** (desktop control, app management, system commands)
- **Learns** (behavior tracking, habit formation, self-optimization)
- **Evolves** (self-diagnostics, pattern detection, architecture improvement)

### Core Principles

| Principle | Implementation |
|-----------|---------------|
| **Offline-First** | GPU-accelerated GGUF LLM (9B/13B), faster-whisper STT, SAPI TTS — zero cloud dependency |
| **Owner-Centric** | Single owner (Satyam), addressed as "Boss", deep personal understanding (JARVIS-level) |
| **Event-Driven** | Every module communicates through AsyncEventBus — zero coupling |
| **Security-Gated** | Every action passes through SecurityPolicy before execution |
| **Self-Improving** | SelfEvolutionEngine + SelfOptimizer + BehaviorTracker |
| **Modular Organs** | Every subsystem can be replaced without touching others |
| **Cross-Platform** | PlatformAdapter abstracts Windows/Linux/macOS — runs anywhere |
| **System-Aware** | SystemScanner profiles and continuously monitors the entire host |
| **Proactive** | JarvisCore anticipates needs, generates briefings, fuses all intelligence |

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ (async/await) |
| Concurrency | asyncio + ThreadPoolExecutor(3 workers) |
| STT | faster-whisper small (offline, bilingual en+hi, CPU/GPU) |
| TTS | Windows SAPI (offline primary), Edge Neural TTS (online option) |
| LLM | llama-cpp-python with GGUF models (9B/13B, full GPU offload, true streaming) |
| UI | aiohttp WebSocket + Three.js JARVIS dashboard |
| Monitoring | psutil, custom MetricsCollector, HealthMonitor |
| Persistence | JSON files (logs/, config/) — no database dependency |

---

## 2. Architecture Layers (The 7 Rings)

ATOM is structured in **8 concentric rings**, from outermost (user-facing) to innermost (JARVIS mind):

```
┌─────────────────────────────────────────────────────────────────────┐
│  RING 1: PERCEPTION LAYER                                          │
│  ┌─────────┐  ┌───────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │STTAsync  │  │MicManager │  │SystemWatcher  │  │ContextEngine  │  │
│  └─────────┘  └───────────┘  └──────────────┘  └───────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│  RING 2: UNDERSTANDING LAYER                                       │
│  ┌──────────────┐  ┌────────────┐  ┌──────────────────┐            │
│  │IntentEngine   │  │CommandCache │  │SkillsRegistry    │            │
│  │(12 sub-modules│  │(O(1) lookup)│  │(phrase expansion) │            │
│  └──────────────┘  └────────────┘  └──────────────────┘            │
├─────────────────────────────────────────────────────────────────────┤
│  RING 3: DECISION LAYER                                            │
│  ┌──────┐  ┌───────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │Router│  │CacheEngine │  │MemoryEngine   │  │ConversationMem   │  │
│  │(3-tier│  │(LRU+Jaccard│  │(keyword+JSON) │  │(rolling window)  │  │
│  │ intel)│  │ similarity)│  │               │  │                  │  │
│  └──────┘  └───────────┘  └──────────────┘  └──────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│  RING 4: EXECUTION LAYER                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │system_actions │  │app_actions    │  │media_actions  │              │
│  │network_actions│  │file_actions   │  │utility_actions│              │
│  │desktop_control│  │SystemControl  │  │               │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
├─────────────────────────────────────────────────────────────────────┤
│  RING 5: EXPRESSION LAYER                                          │
│  ┌───────────┐  ┌───────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │TTSAsync    │  │EdgeTTSAsync│  │WebDashboard   │  │Personality   │  │
│  │(SAPI)      │  │(Neural)    │  │(JARVIS UI)    │  │(tone/style)  │  │
│  └───────────┘  └───────────┘  └──────────────┘  └─────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│  RING 6: COGNITION LAYER                                           │
│  ┌───────────┐  ┌──────────────┐  ┌──────────────────┐             │
│  │GoalEngine  │  │PredictionEng │  │PersonalityModes   │             │
│  │SecondBrain │  │BehaviorModel │  │SelfOptimizer      │             │
│  └───────────┘  └──────────────┘  └──────────────────┘             │
├─────────────────────────────────────────────────────────────────────┤
│  RING 7: AUTONOMY + SELF-AWARENESS LAYER                           │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────────┐ │
│  │AutonomyEngine │  │SelfEvolutionEngine│  │HealthMonitor+Governor│ │
│  │(habit→action)  │  │(diagnose+improve) │  │(CPU gov, watchdog)   │ │
│  └──────────────┘  └──────────────────┘  └───────────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│  RING 8: JARVIS INTELLIGENCE LAYER (v19 — The Mind)                │
│  ┌───────────────┐ ┌──────────────────┐ ┌────────────────────────┐ │
│  │ JarvisCore     │ │OwnerUnderstanding│ │ SystemScanner          │ │
│  │ (intelligence  │ │(deep personal    │ │ (deep hardware/        │ │
│  │  fusion,       │ │ model, emotion,  │ │  software/network      │ │
│  │  proactive     │ │ communication,   │ │  profiling, health     │ │
│  │  anticipation, │ │ topics, temporal │ │  scoring, bottleneck   │ │
│  │  contextual    │ │ patterns, project│ │  detection)            │ │
│  │  inference,    │ │ tracking)        │ │                        │ │
│  │  briefings)    │ │                  │ │                        │ │
│  └───────────────┘ └──────────────────┘ └────────────────────────┘ │
│  ┌───────────────┐ ┌──────────────────┐                            │
│  │PlatformAdapter│ │ SystemControl    │                            │
│  │(cross-platform│ │ (full system     │                            │
│  │ abstraction,  │ │  authority:      │                            │
│  │ Win/Linux/Mac)│ │  processes,      │                            │
│  │               │ │  services,       │                            │
│  │               │ │  network, power) │                            │
│  └───────────────┘ └──────────────────┘                            │
└─────────────────────────────────────────────────────────────────────┘

              ┌──────────────────────────────────────┐
              │  BACKBONE: AsyncEventBus              │
              │  (connects ALL rings, zero coupling)  │
              └──────────────────────────────────────┘
```

---

## 3. Module Registry — Every Organ Mapped

### 3.1 Perception Layer (Ring 1)

| Module | File | Purpose | Dependencies | Interface |
|--------|------|---------|-------------|-----------|
| **STTAsync** | `voice/stt_async.py` | Multi-engine speech-to-text orchestrator | MicManager, IntentEngine, EventBus | Emits: `speech_final`, `speech_partial`, `stt_did_not_catch`, `stt_too_noisy` |
| **MicManager** | `voice/mic_manager.py` | Microphone device detection, BT preference, hot-swap | PyAudio | Returns device index + name |
| **SystemWatcher** | `core/system_watcher.py` | Network, power, Bluetooth event detection | psutil | Emits: `system_event` (kind=network_lost/power_unplugged/bt_connected/...) |
| **ContextEngine** | `context/context_engine.py` | Active window + clipboard context bundle | ctypes (Win32) | Returns: `{clipboard, active_window, ...}` |
| **PrivacyFilter** | `context/privacy_filter.py` | Redacts PII/secrets before storage or prompts | regex | `redact(text) -> clean_text` |

### 3.2 Understanding Layer (Ring 2)

| Module | File | Purpose | Latency | Interface |
|--------|------|---------|---------|-----------|
| **IntentEngine** | `core/intent_engine/__init__.py` | Ultra-fast regex intent classifier | <5ms | `classify(text) -> IntentResult` |
| **IntentResult** | `core/intent_engine/base.py` | Classification output DTO | — | `{intent, response, action, action_args}` |
| **12 Intent Sub-Modules** | `core/intent_engine/*.py` | Categorized regex patterns | <1ms each | `check(text) -> IntentResult or None` |
| **CommandCache** | `core/command_cache.py` | Intent result caching | O(1) | `get(text) -> IntentResult`, `put(text, result)` |
| **SkillsRegistry** | `core/skills_registry.py` | Phrase expansion + multi-step chains | O(n) | `try_expand_full(text) -> SkillMatch or None` |

#### Intent Sub-Module Map

| Module | Handles | Example Commands |
|--------|---------|-----------------|
| `meta_intents` | greeting, thanks, exit, go_silent, confirm/deny | "hi", "bye", "yes", "no" |
| `info_intents` | time, date, CPU, RAM, battery, disk, IP, uptime | "what time", "check CPU" |
| `app_intents` | open/close/list apps | "open chrome", "close notepad" |
| `media_intents` | volume, mute, play, stop | "set volume 50", "play jazz" |
| `system_intents` | lock, screenshot, shutdown, brightness | "lock screen", "take screenshot" |
| `desktop_intents` | scroll, click, press key, type text | "scroll down", "press enter" |
| `file_intents` | create folder, move, copy | "create folder test" |
| `network_intents` | search, open URL, weather, research | "search Python tutorial" |
| `os_intents` | self-check, diagnostics, kill process | "self check", "kill process" |
| `cognitive_intents` | goals, predictions, brain recall, optimize | "set a goal", "show predictions" |
| `runtime_mode_intents` | perf mode, brain profile, assistant mode | "switch to lite mode" |

### 3.3 Decision Layer (Ring 3)

| Module | File | Purpose | Key Design |
|--------|------|---------|-----------|
| **Router** | `core/router/router.py` | 3-layer agentic routing | Intent fast-path → Cache/Memory → LLM Agent |
| **CacheEngine** | `core/cache_engine.py` | TTL-aware LRU + Jaccard similarity | O(1) exact + O(32) fuzzy scan |
| **MemoryEngine** | `core/memory_engine.py` | Keyword-overlap Q&A memory + interaction logger | JSON persistence, top-k retrieval |
| **ConversationMemory** | `core/conversation_memory.py` | Rolling conversation window + session topics | 20-turn window, topic extraction |
| **ToolRegistry** | `core/reasoning/tool_registry.py` | Formal definitions of 40+ tools | Auto-generates LLM prompt + function schemas |
| **ToolParser** | `core/reasoning/tool_parser.py` | Extracts tool calls from LLM output | 4 formats: ATOM JSON, Qwen ✿FUNCTION✿, simple, bare |
| **ActionExecutor** | `core/reasoning/action_executor.py` | Security-gated tool execution bridge | Registry → param → security → dispatch → result |
| **ReasoningPlanner** | `core/reasoning/planner.py` | Multi-step task decomposition | Template plans + LLM-generated plans |
| **CodeSandbox** | `core/reasoning/code_sandbox.py` | Safe Python eval for math/calculations | Restricted builtins, 5s timeout, human math |

#### Router 3-Layer Intelligence (v18 — Agentic)

```
User Speech
    │
    ▼
┌─────────────────────────────────────────────┐
│ Layer 1: Intent Engine (<5ms, fast-path)     │
│ ┌─────────────────────────────────────────┐ │
│ │ 12 regex sub-modules → IntentResult     │ │
│ │ If high-confidence → direct action      │ │
│ │ "open chrome", "set volume 50", "time"  │ │
│ └─────────────────────────────────────────┘ │
│ If fallback / ambiguous ↓                   │
├─────────────────────────────────────────────┤
│ Layer 2: Cache + Memory (instant)           │
│ ┌──────────────┐  ┌──────────────────────┐ │
│ │CacheEngine    │  │MemoryEngine          │ │
│ │(exact+Jaccard)│  │(keyword overlap)     │ │
│ │if hit → serve │  │provides context      │ │
│ └──────────────┘  └──────────────────────┘ │
│ If miss ↓                                   │
├─────────────────────────────────────────────┤
│ Layer 3: LLM Reasoning Agent (agentic)      │
│ ┌──────────────────────────────────────┐    │
│ │ 7-Layer Prompt + 40+ ToolRegistry   │    │
│ │ → LLM reasons + calls tools         │    │
│ │ → ActionExecutor (security-gated)   │    │
│ │ → ReAct loop (act → observe → act)  │    │
│ │ → Text response streamed to TTS     │    │
│ └──────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

### 3.4 Execution Layer (Ring 4)

| Module | File | Actions | Safety |
|--------|------|---------|--------|
| **system_actions** | `core/router/system_actions.py` | lock, screenshot, brightness, shutdown, restart, sleep, logoff, flush DNS, empty recycle bin | SecurityPolicy gate |
| **app_actions** | `core/router/app_actions.py` | open, close, list apps | Executable allowlist |
| **media_actions** | `core/router/media_actions.py` | volume, mute, play YouTube, stop music | Feature flag |
| **network_actions** | `core/router/network_actions.py` | search, open URL, weather, WiFi status | Feature flag |
| **file_actions** | `core/router/file_actions.py` | create folder, move path, copy path | Path allowlist |
| **utility_actions** | `core/router/utility_actions.py` | minimize/maximize/switch window, timer, clipboard | — |
| **desktop_control** | `core/desktop_control.py` | scroll, click, press key, type text, hotkey combos | Hotkey safety tiers |

### 3.5 Expression Layer (Ring 5)

| Module | File | Purpose | Key Feature |
|--------|------|---------|------------|
| **TTSAsync** | `voice/tts_async.py` | Windows SAPI TTS (offline) | Barge-in support |
| **EdgeTTSAsync** | `voice/tts_edge.py` | Edge Neural TTS (online) | Audio postprocessing, ack cache |
| **WebDashboard** | `ui/web_dashboard.py` | JARVIS-style web UI + WebSocket | Three.js orb, real-time panels |
| **FloatingIndicator** | `ui/floating_indicator.py` | Tkinter fallback UI | System tray icon |
| **Personality** | `core/personality.py` | Response tone/style generator | Owner-aware responses |
| **PersonalityModes** | `core/personality_modes.py` | work/focus/chill/sleep modes | Voice profile overrides |

### 3.6 Cognition Layer (Ring 6)

| Module | File | Purpose | Data Source |
|--------|------|---------|-----------|
| **SecondBrain** | `core/cognitive/second_brain.py` | Unified knowledge store (facts + preferences + corrections) | MemoryEngine + BehaviorTracker |
| **GoalEngine** | `core/cognitive/goal_engine.py` | Goal → Plan → Execute → Evaluate | LLM decomposition, step tracking |
| **PredictionEngine** | `core/cognitive/prediction_engine.py` | Predict next user action before they speak | Time-slot frequency + transition probability |
| **BehaviorModel** | `core/cognitive/behavior_model.py` | Personal behavior profile + energy inference | Action timestamps, app switch tracking |
| **SelfOptimizer** | `core/cognitive/self_optimizer.py` | Suggest improvements to ATOM itself | Feature usage analysis, fallback pattern detection |

### 3.7 Autonomy + Self-Awareness Layer (Ring 7)

| Module | File | Purpose | Safety |
|--------|------|---------|--------|
| **AutonomyEngine** | `core/autonomy_engine.py` | Autonomous habit detection + auto-execution | Never auto-executes destructive actions |
| **SelfEvolutionEngine** | `core/self_evolution.py` | Performance diagnostics + improvement suggestions | Suggest-only, never auto-changes |
| **HealthMonitor** | `core/health_monitor.py` | CPU governor + stuck-state recovery + context snapshots | Auto-recovery from stuck states |
| **RuntimeWatchdog** | `core/runtime_watchdog.py` | Thinking/Speaking timeout enforcement | Forces state recovery |
| **BehaviorTracker** | `core/behavior_tracker.py` | Habit formation from action patterns | Configurable decay + confidence |
| **SecurityPolicy** | `core/security_policy.py` | Central security gate for all actions | 7-layer enforcement |

### 3.8 JARVIS Intelligence Layer (Ring 8 — v19)

| Module | File | Purpose | Key Feature |
|--------|------|---------|------------|
| **JarvisCore** | `core/jarvis_core.py` | Intelligence fusion — the mind of ATOM | Proactive anticipation, contextual inference, situational awareness, intelligent briefings |
| **OwnerUnderstanding** | `core/owner_understanding.py` | Deep owner intelligence — JARVIS-level personal model | Communication style, emotional trajectory, topic expertise, temporal patterns, project tracking |
| **SystemScanner** | `core/system_scanner.py` | Deep system intelligence — knows everything about the host | Hardware/software profiling, health scoring, bottleneck detection, environment analysis |
| **PlatformAdapter** | `core/platform_adapter.py` | Cross-platform OS abstraction | Windows/Linux/macOS unified API for all OS interactions |
| **SystemControl** | `core/system_control.py` | Full system authority — JARVIS-level control | Advanced process management, network control, storage analysis, startup management, optimization |

### Infrastructure Modules

| Module | File | Purpose |
|--------|------|---------|
| **AsyncEventBus** | `core/async_event_bus.py` | Fire-and-forget pub/sub backbone with timeout/isolation |
| **StateManager** | `core/state_manager.py` | 6-state machine with validated transitions |
| **MetricsCollector** | `core/metrics.py` | Counters, latency histograms, gauges |
| **PipelineTimer** | `core/pipeline_timer.py` | End-to-end latency measurement |
| **PriorityScheduler** | `core/priority_scheduler.py` | Single-worker priority queue (voice > LLM > background) |
| **LLMInferenceQueue** | `core/llm_inference_queue.py` | Serial LLM queue with request coalescing |
| **BrainModeManager** | `core/brain_mode_manager.py` | Brain profile switching (atom/balanced/brain) |
| **AssistantModeManager** | `core/assistant_mode_manager.py` | Assistant mode switching (hybrid/command_only) |
| **ConfigSchema** | `core/config_schema.py` | Settings validation at startup |
| **LoggingSetup** | `core/logging_setup.py` | Structured logging configuration |

---

## 4. The Nervous System — Event Bus Contract

The **AsyncEventBus** is ATOM's nervous system. Every module communicates through events. This is the **complete event catalog**:

### 4.1 Core Pipeline Events

| Event | Emitted By | Consumed By | Payload | Tier |
|-------|-----------|------------|---------|------|
| `speech_partial` | STTAsync | WebDashboard | `{text}` | fast |
| `speech_final` | STTAsync | Router, Metrics, Dashboard | `{text}` | long |
| `intent_classified` | Router | Metrics, Autonomy, Behavior, Dashboard | `{intent, ms, text, action_args}` | fast |
| `thinking_ack` | Router | TTS, Dashboard | `{text}` | long |
| `cursor_query` | Router | LocalBrainController | `{text, memory_context, context, history}` | long |
| `cursor_response` | LocalBrainController | Cache, Memory, Router, SecondBrain | `{query, response}` | emit |
| `response_ready` | Router/Autonomy | TTS, Dashboard | `{text}` | long |
| `partial_response` | LocalBrainController | TTS, Dashboard, Metrics | `{text, is_first, is_last, source}` | long |
| `text_display` | Router | Dashboard | `{text}` | long |
| `tts_complete` | TTS | StateManager | `{}` | emit |

### 4.2 State Events

| Event | Emitted By | Consumed By | Payload |
|-------|-----------|------------|---------|
| `state_changed` | StateManager | STT, Dashboard, HealthMonitor, Watchdog | `{old, new}` (AtomState enums) |
| `resume_listening` | Hotkey/Dashboard | StateManager, STT | `{}` |
| `enter_sleep_mode` | Router | STT, StateManager | `{}` |
| `restart_listening` | StateManager | STT | `{}` |
| `silence_timeout` | STT | StateManager | `{}` |

### 4.3 System Events

| Event | Emitted By | Consumed By | Payload |
|-------|-----------|------------|---------|
| `system_event` | SystemWatcher | main.py handlers | `{kind, app, message, level, device}` |
| `media_started` | Router | STT | `{}` |
| `mic_changed` | HealthMonitor | Dashboard | `{name}` |
| `llm_error` | LocalBrainController | StateManager | `{source}` |
| `shutdown_requested` | UI/Dashboard | main.py | `{}` |

### 4.4 Autonomy + Cognitive Events

| Event | Emitted By | Consumed By | Payload |
|-------|-----------|------------|---------|
| `context_snapshot` | HealthMonitor | BehaviorModel, AutonomyEngine | `{time_of_day, hour, cpu, ram, idle_minutes, active_app}` |
| `habit_suggestion` | AutonomyEngine | main.py → TTS | `{text, habit_id, confidence}` |
| `autonomous_action` | AutonomyEngine | main.py → Router | `{action, target, habit_id, confidence}` |
| `user_feedback` | main.py | AutonomyEngine | `{habit_id, accepted}` |
| `goal_update` | GoalEngine | — | `{goal_id, action, title}` |
| `goal_briefing` | GoalEngine | main.py → TTS | `{text}` |
| `prediction_ready` | PredictionEngine | Dashboard | `{predictions: []}` |
| `user_energy_state` | BehaviorModel | PersonalityModes | `{energy, idle_minutes, active_app}` |
| `mode_changed` | PersonalityModes | main.py → TTS/Dashboard | `{mode, old_mode, voice_rate_adj, ...}` |
| `optimization_suggestions` | SelfOptimizer | main.py → Dashboard | `{suggestions: []}` |
| `idle_detected` | HealthMonitor | — | `{idle_minutes}` |

### 4.5 JARVIS Intelligence Events (v19)

| Event | Emitted By | Consumed By | Payload |
|-------|-----------|------------|---------|
| `system_intelligence` | SystemScanner | JarvisCore, Dashboard | `{scan, health}` |
| `system_light_scan` | SystemScanner | JarvisCore | `{scan}` |
| `jarvis_insight` | JarvisCore | main.py → TTS | `{message, category, priority, action, action_args}` |
| `system_scan_request` | Router/Intent | main.py → SystemScanner | `{}` |
| `system_control_request` | Router/Intent | main.py → SystemControl | `{action, **kwargs}` |
| `owner_summary_request` | Router/Intent | main.py → OwnerUnderstanding | `{}` |

### 4.6 Performance Events

| Event | Emitted By | Consumed By | Payload |
|-------|-----------|------------|---------|
| `governor_throttle` | HealthMonitor | TTS, Dashboard, AutonomyEngine | `{cpu}` |
| `governor_normal` | HealthMonitor | TTS, Dashboard, AutonomyEngine | `{cpu}` |
| `set_performance_mode` | Router/Dashboard | main.py | `{mode}` |
| `runtime_settings_changed` | Router | Dashboard | `{brain_profile, assistant_mode}` |
| `metrics_event` | various | MetricsCollector | `{counter}` |
| `metrics_latency` | various | MetricsCollector | `{name, ms}` |
| `reminder_due` | TaskScheduler | main.py → TTS | `{label, task_id}` |
| `intent_chain_suggestion` | Router | main.py → Dashboard | `{suggestion, action}` |

### 4.6 Event Emission Tiers

| Tier | Method | Timeout | Use Case |
|------|--------|---------|----------|
| **fast** | `emit_fast()` | None | Metrics, logging, UI state (<1ms guaranteed) |
| **normal** | `emit()` | 10s | Standard handlers, warn if >5s |
| **long** | `emit_long()` | 60s | TTS playback, LLM inference, network I/O |

---

## 5. The Brain Pipeline — How ATOM Thinks

### 5.1 Complete Processing Pipeline

```
┌──────────┐   ┌───────────┐   ┌────────────┐   ┌──────────┐   ┌──────┐
│Microphone│──▶│STT Engine │──▶│Text Correct│──▶│Noise     │──▶│Intent│
│(PyAudio) │   │(Whisper   │   │(known      │   │Filter    │   │Quick │
│          │   │bilingual) │   │ corrections│   │(is_noise)│   │Match │
└──────────┘   └───────────┘   └────────────┘   └──────────┘   └──┬───┘
                                                                   │
                                              ┌────────────────────┘
                                              ▼
                                    ┌───────────────┐
                                    │ speech_final   │ (event on bus)
                                    └───────┬───────┘
                                            ▼
                              ┌──────────────────────────┐
                              │ Router._route(text)       │
                              │ 1. Security sanitize      │
                              │ 2. Filler word removal    │
                              │ 3. Skill expansion        │
                              │ 4. Pronoun resolution     │
                              │ 5. Clipboard injection    │
                              └─────────┬────────────────┘
                                        ▼
                              ┌──────────────────┐
                              │ IntentEngine      │
                              │ .classify(text)   │
                              │ (<5ms, 12 modules)│
                              └────────┬─────────┘
                                       │
                       ┌───────────────┼────────────────┐
                       ▼               ▼                ▼
                 ┌──────────┐   ┌──────────┐    ┌────────────────┐
                 │ MATCHED  │   │ COGNITIVE │    │ FALLBACK →     │
                 │ (70%)    │   │ (goals,  │    │ LLM REASONING  │
                 │Fast-path │   │  predict) │    │ AGENT          │
                 └─────┬────┘   └─────┬────┘    └─────┬──────────┘
                       │              │               │
                       ▼              ▼               ▼
                 ┌──────────┐   ┌──────────┐    ┌────────────────┐
                 │Confirm?  │   │Dispatched│    │Quick Reply?    │
                 │Security  │   │via bus to │    │Cache hit?      │
                 │Gate      │   │cognitive  │    │Memory ctx?     │
                 └─────┬────┘   │handlers  │    └─────┬──────────┘
                       │        └──────────┘          │
                       ▼                              ▼
                 ┌──────────┐                   ┌────────────────┐
                 │ Execute  │                   │ AGENTIC LLM    │
                 │ Action   │                   │ 7-layer prompt │
                 │ (dispatch│                   │ + 40+ tools    │
                 │  table)  │                   │ + ToolRegistry │
                 └─────┬────┘                   └─────┬──────────┘
                       │                              │
                       ▼                        ┌─────┴──────────┐
                 ┌──────────┐                   │ ReAct Loop:    │
                 │ TTS      │                   │ 1. LLM output  │
                 │ Response │                   │ 2. Parse tools  │
                 │          │                   │ 3. Execute via  │
                 └──────────┘                   │    ActionExec.  │
                                                │ 4. Observe      │
                                                │ 5. Reason again │
                                                │ → Text + TTS    │
                                                └─────────────────┘
```

### 5.2 Latency Budget by Layer

| Stage | Target | Actual | Notes |
|-------|--------|--------|-------|
| STT (Whisper base.en) | <500ms | ~300ms | Offline, CPU (GPU upgrade planned) |
| Intent Classification | <5ms | 1-3ms | Pure regex, 12 modules |
| Cache Lookup | <1ms | <0.5ms | O(1) exact + O(32) Jaccard |
| Memory Retrieval | <5ms | 2-5ms | Keyword overlap scan (vector upgrade planned) |
| Action Dispatch | <10ms | 1-5ms | O(1) dispatch table |
| LLM Inference (GPU 9B) | <4s | 1-4s | Full GPU offload, Q4_K_M quantized |
| LLM First Token | <500ms | 300-500ms | True streaming, sentence buffer |
| KV Cache Restore | 0ms | ~0ms | System prompt cached in memory |
| TTS (SAPI) | <100ms | ~50ms | Instant start, offline |
| **End-to-End (local)** | **<200ms** | **50-100ms** | Intent → Action → TTS |
| **End-to-End (LLM)** | **<2s** | **300ms-1s** | Speech → first spoken sentence |

---

## 6. State Machine — The Heartbeat

### 6.1 States

| State | Description | CPU Usage |
|-------|-------------|-----------|
| `SLEEP` | Fully shut down, no audio processing | ~0% |
| `IDLE` | Resting, minimal background work | <0.5% |
| `LISTENING` | STT active, processing microphone input | 2-5% |
| `THINKING` | LLM query in flight or action processing | 5-100% |
| `SPEAKING` | TTS playing audio output | 1-3% |
| `ERROR_RECOVERY` | Transient error, auto-recovers to IDLE | <0.5% |

### 6.2 Transition Map

```
              ┌──────────────────────┐
              │       SLEEP          │
              │  (fully stopped)     │
              └─────┬──────▲─────────┘
                    │      │
              resume│      │shutdown (from ANY state)
                    ▼      │
              ┌──────────────────────┐
              │       IDLE           │◄─── tts_complete (if !always_listen)
              │  (resting)           │◄─── auto-recover from ERROR_RECOVERY
              └─────┬──────▲─────────┘
                    │      │
          start     │      │ silence_timeout
          listening │      │
                    ▼      │
              ┌──────────────────────┐
              │     LISTENING        │──── speech_final ────▶ THINKING
              │  (STT active)        │──── fast-path ───────▶ SPEAKING
              └──────────────────────┘
                                          │
              ┌──────────────────────┐    │
              │     THINKING         │◄───┘
              │  (LLM processing)    │──── response ────────▶ SPEAKING
              └──────────────────────┘──── error ──────────▶ ERROR_RECOVERY
                                          │
              ┌──────────────────────┐    │
              │     SPEAKING         │◄───┘
              │  (TTS playing)       │──── tts_complete ───▶ IDLE/LISTENING
              └──────────────────────┘──── barge-in ───────▶ LISTENING

              ┌──────────────────────┐
              │  ERROR_RECOVERY      │──── auto ───────────▶ IDLE
              │  (transient error)   │
              └──────────────────────┘
```

### 6.3 Always-Listen Mode

When `state.always_listen = True` (default):
- `SPEAKING → tts_complete → LISTENING` (not IDLE)
- `IDLE → auto-recover → LISTENING` (background task)
- `silence_timeout → restart_listening` (not IDLE)

---

## 7. Interface Contracts — Upgrade Without Breaking

This is the **contract registry** — the key to modular evolution. Each contract defines what a module MUST provide. Replace the implementation, keep the contract.

### 7.1 STT Contract

Any STT implementation MUST:

```
class STTContract:
    mic_name: str                           # Current microphone name
    async preload() -> None                 # Pre-load STT model
    async start_listening() -> None         # Begin continuous listening
    def stop() -> None                      # Stop listening
    def shutdown() -> None                  # Release all resources
    def on_state_changed(old, new) -> None  # React to state changes
    def on_media_started() -> None          # Pause during media playback
    def refresh_mic() -> bool               # Hot-swap microphone

    # MUST emit via bus:
    #   speech_final(text=str)
    #   speech_partial(text=str)
    #   stt_did_not_catch()
    #   stt_too_noisy()
    #   silence_timeout()
```

### 7.2 TTS Contract

```
class TTSContract:
    async init_voice() -> None              # Initialize TTS engine
    async on_response(text: str) -> None    # Speak full response
    async on_partial_response(text, is_first, is_last) -> None
    async speak_ack(text: str) -> None      # Quick acknowledgment
    async stop() -> None                    # Interrupt playback
    async shutdown() -> None                # Release resources

    # MUST emit via bus:
    #   tts_complete()
```

### 7.3 Brain Contract (LLM — GPU, Agentic Tool Use, ReAct Loop)

```
class BrainContract:
    available: bool                         # Model file exists + library installed
    is_loaded: bool                         # Model in memory (GPU)
    def set_action_executor(executor)       # Wire ActionExecutor for tool execution
    def request_preempt() -> None           # Cancel current generation
    async warm_up() -> None                 # Pre-load model to GPU
    async on_query(text, memory_context, context, history) -> None
    def close() -> None                     # Unload model + free GPU memory
    def get_stats() -> dict                 # Calls, tokens, tool_calls, react_loops

    # MUST emit via bus:
    #   partial_response(text, is_first, is_last, source)  — TRUE streaming per sentence
    #   cursor_response(query, response)
    #   tool_executed(tool, success, elapsed_ms)            — NEW v18
    #   pending_tool_confirmation(tool_call, result)        — NEW v18
    #   metrics_latency(name="llm", ms=float)
    #   metrics_latency(name="llm_first_token", ms=float)
    #   llm_error(source) [on failure]

    # Agentic Flow (v18):
    #   1. Build 7-layer prompt with ToolRegistry definitions
    #   2. LLM generates response (streamed)
    #   3. Parse response for <tool_call> tags (4 format support)
    #   4. If tool calls: ActionExecutor validates → security → dispatch → result
    #   5. Feed [OBSERVATION] back to LLM (ReAct loop, max 3 steps)
    #   6. LLM generates final text response for TTS
```

### 7.4 UI/Indicator Contract

```
class IndicatorContract:
    def start() -> None
    def shutdown() -> None
    def on_state_changed(old, new) -> None
    def add_log(category: str, text: str) -> None
    def show_hearing(text: str) -> None
    def clear_hearing() -> None
    def set_mic_name(name: str) -> None
    def set_shutdown_callback(cb) -> None
    def set_mode_change_callback(cb) -> None
```

### 7.5 Intent Engine Contract

```
class IntentEngineContract:
    def classify(text: str) -> IntentResult  # <5ms
    def quick_match(text: str) -> str | None # Used by STT for early exit
```

### 7.6 Cognitive Module Contract

```
class CognitiveModuleContract:
    def start() -> None                      # Begin background task
    def stop() -> None                       # Cancel background task + persist
    def persist() -> None                    # Save state to disk
```

---

## 8. Data Flow Maps

### 8.1 Voice Command Flow (Local — 85% of queries)

```
Mic → STT → speech_final → Router → IntentEngine → Action Dispatch → TTS
                                                  ↗
                              SecurityPolicy ─────┘
Total time: 50-200ms
```

### 8.2 Agentic LLM Query Flow (Complex — 30% of queries)

```
Mic → STT → speech_final → Router → IntentEngine (fallback/ambiguous)
                                   → thinking_ack → TTS ("Working on it")
                                   → Cache lookup (parallel)
                                   → Memory retrieval (parallel)
                                   → cursor_query → LLMInferenceQueue
                                                  → PriorityScheduler
                                                  → LocalBrainController (AGENTIC)
                                                  → 7-Layer PromptBuilder
                                                     + ToolRegistry (40+ tools)
                                                  → MiniLLM (GPU, 9B/13B)
                                   ← LLM Response (streamed):
                                      ├── Text only → TTS speaks
                                      └── Tool call(s) detected:
                                          → ToolParser.parse_tool_calls()
                                          → ActionExecutor.execute()
                                            → Registry validate
                                            → Param validate
                                            → SecurityPolicy gate
                                            → Router dispatch
                                          → Result as [OBSERVATION]
                                          → Re-prompt LLM (ReAct step 2)
                                          → Final text response → TTS
                                   ← cursor_response → Cache.put + Memory.add
Total time: 300ms first audio (text), 1-5s with tool execution
```

### 8.3 Autonomy Decision Flow

```
HealthMonitor → context_snapshot → BehaviorModel (energy inference)
                                 → AutonomyEngine (habit check)
                                 → PredictionEngine (next action guess)

AutonomyEngine Decision:
  confidence >= 0.95 + not destructive → autonomous_action (auto-execute)
  confidence >= 0.72                   → habit_suggestion (ask user)
  user confirms                        → user_feedback(accepted=True) → confidence +0.1
  user denies                          → user_feedback(accepted=False) → confidence -0.15
```

### 8.4 Goal Intelligence Flow

```
Voice: "set a goal to learn Rust"
  → cognitive_intents → goal_create
  → GoalEngine.create_goal("learn Rust")
  → TTS: "Goal set. Say 'break down this goal' for a step plan."

Voice: "break down this goal"
  → cognitive_intents → goal_decompose
  → GoalEngine.decompose_with_llm()
  → cursor_query → LLM generates 5-8 steps
  → cursor_response → parse steps → add to goal
  → TTS: "Added 6 steps to 'learn Rust'."

Daily (7-10 AM): GoalEngine._check_daily_briefing()
  → goal_briefing → TTS: "Good morning Boss. 2 active goals..."

Hourly: GoalEngine._evaluate_all_goals()
  → trajectory = ahead / on_track / behind / stalled
  → suggestions generated for behind/stalled goals
```

---

## 9. Configuration Architecture

### 9.1 Config Hierarchy

```
config/settings.json           ← Main config (ALL settings)
config/commands.json            ← Action registry (confirm flags)
config/skills.json              ← Phrase expansion skills
config/settings.corporate.example.json  ← Corporate laptop template
```

### 9.2 Config Section Map

| Section | Controls | Hot-Reloadable |
|---------|----------|---------------|
| `owner` | Name, title (Boss) | No (startup) |
| `vision` | Camera, face recognition | No |
| `mic` | Device preference, BT priority | Partial (refresh_mic) |
| `stt` | Engine, model, sample rate, thresholds | No |
| `tts` | Engine, voice, rate, postprocessing | Partial (mode switch) |
| `brain` | Model path, n_ctx, threads, tokens, temp | Partial (profile switch) |
| `assistant_brain` | Profiles (atom/balanced/brain), quick replies | Yes (runtime) |
| `cache` | Max size, TTL | Yes (self-tune) |
| `memory` | Max entries, top-k | No |
| `context` | Clipboard, active window | No |
| `ui` | Web port, auto-open | No |
| `security` | Mode, audit, confirmation list | No |
| `features` | Desktop control, file ops, LLM, web research | No |
| `control` | Lock mode, assistant mode, runtime switching | Yes (runtime) |
| `performance` | Mode, governor, intervals, watchdog timeouts | Partial (mode switch restarts) |
| `autonomy` | Enabled, thresholds, interval, decay | No |
| `cognitive` | Goals, predictions, behavior, optimizer, modes | No |
| `deployment` | Profile badge | No |

---

## 10. Security Architecture

### 10.1 The 7 Security Gates

```
┌─────────────────────────────────────────────────────────────┐
│ Gate 1: INPUT SANITIZATION                                   │
│   - Max 2000 chars                                           │
│   - Strip shell injection chars (;&|`$)                      │
│   - Strip XSS patterns (<script, javascript:)                │
├─────────────────────────────────────────────────────────────┤
│ Gate 2: ACTION-LEVEL GATE (allow_action)                     │
│   - Lock mode enforcement (safe_only blocks non-safe)        │
│   - Feature flag checking                                    │
│   - Executable allowlist (open_app)                          │
│   - Close target allowlist (close_app)                       │
│   - Power action blocking in strict mode                     │
├─────────────────────────────────────────────────────────────┤
│ Gate 3: CONFIRMATION FLOW                                    │
│   - Configurable per-action (security.require_confirmation)  │
│   - 25-second timeout for pending confirmations              │
│   - Voice confirm/deny handling                              │
├─────────────────────────────────────────────────────────────┤
│ Gate 4: SHELL COMMAND BLOCKLIST                              │
│   - Blocks format, del, reg delete, cipher, diskpart, etc.   │
│   - Blocks encoded PowerShell, Invoke-Expression             │
├─────────────────────────────────────────────────────────────┤
│ Gate 5: HOTKEY SAFETY TIERS                                  │
│   - safe: Ctrl+C, Ctrl+V, Alt+Tab, etc.                     │
│   - confirm: Alt+F4, Ctrl+W                                 │
│   - block: Win+R, Ctrl+Alt+Delete                            │
├─────────────────────────────────────────────────────────────┤
│ Gate 6: PATH ALLOWLIST                                       │
│   - Blocks: Windows, System32, ProgramData                   │
│   - Allows: User home, current working directory             │
├─────────────────────────────────────────────────────────────┤
│ Gate 7: PRIVACY FILTER                                       │
│   - Clipboard redaction before LLM prompts                   │
│   - PII stripping before memory storage                      │
│   - Audit logging (logs/audit.log, chmod 600)                │
└─────────────────────────────────────────────────────────────┘
```

### 10.2 Autonomy Safety

```
NEVER auto-execute (regardless of confidence):
  shutdown_pc, restart_pc, logoff, sleep_pc, close_app, kill_process,
  empty_recycle_bin, create_folder, move_path, copy_path,
  type_text, hotkey_combo, press_key

Auto-execute ONLY when:
  1. Confidence >= 0.95 (from BehaviorTracker)
  2. Action NOT in NEVER_AUTO_EXECUTE set
  3. SecurityPolicy.allow_action() returns True
  4. Not executed in the last 3600 seconds
```

---

## 11. Performance Architecture

### 11.1 Performance Modes

| Mode | Health Check | System Watch | Maintenance | Use Case |
|------|-------------|-------------|-------------|----------|
| `full` | 60s | 10s | 120s | Dedicated PC, full features |
| `lite` | 120s | 30s | 180s | Corporate laptop, balanced |
| `ultra_lite` | 300s | 60s | 300s | Low-resource, minimal overhead |
| `auto` | Adaptive | Adaptive | Adaptive | Latency-driven switching |

### 11.2 CPU Governor

```
CPU > threshold (75%) for 2 consecutive checks:
  → governor_throttle event
  → Health check interval × 2.5
  → TTS postprocessing disabled
  → Dashboard shows throttle badge

CPU < threshold for 3 consecutive checks:
  → governor_normal event
  → Intervals restored
  → TTS postprocessing restored
```

### 11.3 Priority Scheduler

```
Priority 0 (VOICE):     Speech processing — never delayed
Priority 1 (LLM):       LLM inference — after voice
Priority 2 (BACKGROUND): Autonomy checks, maintenance — lowest priority
```

### 11.4 Self-Tuning

```
Cache hit rate > 65%  → TTL × 1.2 (keep hits longer)
Cache hit rate < 15%  → TTL × 0.8 (reduce stale entries)
Cooldown: 3 cycles between adjustments
```

---

## 12. Evolution Roadmap — The ATOM Infinity Path

ATOM evolves in **Phases**. Each phase is independent and can be built incrementally.

### Phase 1: PERCEPTION EVOLUTION (Next)

| Upgrade | Current | Target | Effort | Impact |
|---------|---------|--------|--------|--------|
| **Wake Word Detection** | Always listening | "Hey ATOM" trigger word (Porcupine/OpenWakeWord) | Medium | Saves CPU, natural activation |
| **Emotion Detection** | None | Voice tone analysis (pitch, speed, energy) | Medium | Empathetic responses |
| **Multi-Language STT** | English only | Hindi + English code-switching | Medium | Natural for Indian users |
| **Ambient Sound Awareness** | Noise filter only | Classify: music/talking/silence/typing | Hard | Context-aware behavior |
| **Screen Understanding** | Screenshot only | Local vision model for screen OCR | Hard | "What's on my screen?" |

### Phase 2: INTELLIGENCE EVOLUTION

| Upgrade | Current | Target | Effort | Impact |
|---------|---------|--------|--------|--------|
| **RAG Pipeline** | Keyword memory | Vector embeddings + FAISS/ChromaDB | Medium | 10x better memory retrieval |
| **Multi-Turn Reasoning** | 5-turn window | Chain-of-thought with scratchpad | Medium | Complex problem solving |
| **Tool Use (Function Calling)** | Action dispatch table | LLM can choose and chain tools | Hard | Dynamic problem solving |
| **Code Execution Sandbox** | None | Safe Python/JS eval for calculations | Medium | "Calculate 15% of 2300" |
| **Document Ingestion** | None | PDF/DOCX → knowledge base | Medium | "What does this document say?" |

### Phase 3: AUTONOMY EVOLUTION

| Upgrade | Current | Target | Effort | Impact |
|---------|---------|--------|--------|--------|
| **Workflow Automation** | Single actions | Multi-step recorded workflows | Hard | "Do my morning routine" |
| **Calendar Integration** | None | Outlook/Google Calendar read/write | Medium | Meeting prep, schedule awareness |
| **Email Triage** | None | Summarize + prioritize inbox | Medium | "Any urgent emails?" |
| **Proactive Research** | Disabled (offline) | Background web research on topics | Medium | "You might find this interesting" |
| **Cross-Device Sync** | Single machine | ATOM state sync across devices | Hard | Seamless multi-device |

### Phase 4: EXPRESSION EVOLUTION

| Upgrade | Current | Target | Effort | Impact |
|---------|---------|--------|--------|--------|
| **Voice Cloning** | Fixed TTS voice | Custom ATOM voice (XTTS/Bark) | Hard | Unique identity |
| **3D Avatar** | Three.js orb | Animated face/character in dashboard | Hard | Visual personality |
| **Spatial Audio** | Mono output | Directional audio with head tracking | Hard | Immersive experience |
| **Multi-Modal Response** | Text + voice | Voice + images + diagrams + code blocks | Medium | Richer answers |

### Phase 5: META-COGNITIVE EVOLUTION

| Upgrade | Current | Target | Effort | Impact |
|---------|---------|--------|--------|--------|
| **Dream Mode** | None | Offline consolidation — replay, compress, learn from day's interactions | Hard | Deeper learning |
| **Epistemic Humility** | None | Confidence scoring on every answer + "I'm not sure" | Medium | Trust calibration |
| **Curiosity Engine** | None | ATOM asks questions to learn about you | Medium | Proactive relationship |
| **Personality Evolution** | 4 fixed modes | Personality drift based on long-term behavior | Hard | Grows with you |
| **Attention Economy** | Process all | Priority-weighted attention — ignore noise, amplify signal | Hard | True intelligence |

### Phase 6: DISTRIBUTED ATOM

| Upgrade | Current | Target | Effort | Impact |
|---------|---------|--------|--------|--------|
| **ATOM Mesh** | Single instance | Multiple ATOMs coordinating (home, work, phone) | Very Hard | Omnipresent AI |
| **Plugin Architecture** | Hardcoded modules | Dynamic plugin loading with capability declaration | Hard | Community extensions |
| **Agent Protocol** | None | ATOM as MCP/A2A server + client | Hard | Interop with other AI systems |
| **Hardware Brain** | CPU-only | Dedicated NPU/GPU inference card | Hardware | 10x faster LLM |

---

## 13. Module Upgrade Playbook

### How to Upgrade ANY Module Without Breaking ATOM

#### Step 1: Identify the Contract
Look up the module in Section 7 (Interface Contracts). Your new implementation MUST satisfy every method and event in the contract.

#### Step 2: Check Event Dependencies
Look up the module in Section 4 (Event Bus Contract). Your module MUST:
- Emit the same events with the same payload shapes
- Handle the same incoming events

#### Step 3: Implement the Replacement
Create your new module. It must pass the contract checklist:

```
[ ] All contract methods implemented
[ ] All required events emitted
[ ] All consumed events handled
[ ] Shutdown/cleanup works properly
[ ] Persistence (if any) is backward-compatible
[ ] Config section (if any) is backward-compatible
```

#### Step 4: Wire It In
Modify `main.py` to import your new module instead of the old one. The wiring in `_wire_events()` should work unchanged if your contract is satisfied.

#### Step 5: Test
Run the existing test suite:
```bash
python -m pytest tests/ -v
```

### Example: Replacing STT Engine

```python
# 1. Create voice/stt_new_engine.py
# 2. Implement STTContract (see Section 7.1)
# 3. In main.py, change:
#    from voice.stt_async import STTAsync
#    to:
#    from voice.stt_new_engine import NewSTT as STTAsync
# 4. Everything else stays the same
```

### Example: Adding a New Action

```python
# 1. Register tool in core/reasoning/tool_registry.py (_register_builtin_tools)
#    - Define name, description, parameters, safety_level, category
# 2. Add action handler to router/[category]_actions.py
# 3. Add dispatch entry to Router._ACTION_DISPATCH
# 4. (Optional) Add regex pattern to intent_engine for <5ms fast-path
# 5. The LLM will automatically see the tool via ToolRegistry prompt generation
# 6. ActionExecutor handles security + confirmation automatically
```

### Example: Adding a New Cognitive Module

```python
# 1. Create core/cognitive/new_module.py
# 2. Implement CognitiveModuleContract (start, stop, persist)
# 3. Subscribe to events via bus.on() in start()
# 4. In main.py, instantiate and wire in the cognitive section
# 5. Add intent patterns to cognitive_intents.py
# 6. Add event handler in the cognitive handler block in main.py
```

---

## 14. Build Order — What to Build First

When resuming ATOM development, follow this priority order:

### Tier 1: Foundation Hardening (build confidence)

| # | Task | Why | Files |
|---|------|-----|-------|
| 1 | **Remove kill switch** | Enable execution | `main.py` line 1661-1663 |
| 2 | **Run on target machine** | Validate current state | `python main.py` |
| 3 | **Fix any startup errors** | Baseline stability | Various |
| 4 | **Test all 27 registered commands** | Ensure execution layer works | `config/commands.json` |
| 5 | **Verify dashboard WebSocket** | UI must work | `ui/web_dashboard.py` |

### Tier 2: Quick Wins (high impact, low effort)

| # | Task | Impact | Effort |
|---|------|--------|--------|
| 1 | **Add 20 more intent patterns** | Reduce LLM fallback rate | 2-3 hours |
| 2 | **Add quick replies for common phrases** | Instant responses | 1 hour |
| 3 | **Tune STT noise thresholds** | Better recognition | 1-2 hours |
| 4 | **Add more skill chains** | Multi-step automation | 1 hour |
| 5 | **Edge TTS voice selection** | Better voice quality | 30 min |

### Tier 3: Core Intelligence (medium effort, transformative)

| # | Task | Impact | Effort |
|---|------|--------|--------|
| 1 | **RAG with vector embeddings** | 10x memory quality | 1-2 days |
| 2 | **Wake word detection** | Natural activation | 1 day |
| 3 | **Calendar integration** | Schedule awareness | 1-2 days |
| 4 | **Code sandbox** | Calculation + code exec | 1 day |
| 5 | **Workflow recording** | Multi-step automation | 2-3 days |

### Tier 4: Advanced (high effort, next-level)

| # | Task | Impact | Effort |
|---|------|--------|--------|
| 1 | **Plugin architecture** | Extensibility | 3-5 days |
| 2 | **Multi-language (Hindi+English)** | Natural interaction | 2-3 days |
| 3 | **Tool use / function calling** | Dynamic intelligence | 3-5 days |
| 4 | **Voice cloning** | ATOM identity | 2-3 days |
| 5 | **Dream mode** | Deep learning | 3-5 days |

---

## Appendix A: File Tree with Descriptions

```
ATOM/
├── main.py                         # Entry point — wires ALL modules through event bus
├── requirements.txt                # Pinned dependencies (Python 3.11+)
├── README.md                       # User-facing quick start guide
│
├── config/
│   ├── settings.json               # THE config file — all behavior controlled here
│   ├── commands.json               # Action registry (27 commands, confirm flags)
│   ├── skills.json                 # Phrase expansion skills (10 skills)
│   └── settings.corporate.example.json  # Corporate laptop template
│
├── core/
│   ├── __init__.py
│   ├── async_event_bus.py          # BACKBONE: pub/sub with 3 emission tiers
│   ├── state_manager.py            # 6-state machine with validated transitions
│   ├── config_schema.py            # Settings.json validation
│   ├── logging_setup.py            # Structured logging
│   ├── metrics.py                  # Counters, latencies, gauges
│   ├── pipeline_timer.py           # End-to-end latency measurement
│   ├── fast_path.py                # Startup warm-up + latency budget
│   ├── personality.py              # Response tone/style (owner-aware)
│   ├── personality_modes.py        # work/focus/chill/sleep modes
│   ├── security_policy.py          # THE security gate (7 layers)
│   ├── deployment_profile.py       # Corporate vs personal deployment
│   │
│   ├── router/
│   │   ├── __init__.py
│   │   ├── router.py               # 3-layer intelligence (intent→cache→LLM)
│   │   ├── system_actions.py       # Lock, screenshot, shutdown, etc.
│   │   ├── app_actions.py          # Open, close, list apps
│   │   ├── media_actions.py        # Volume, mute, YouTube
│   │   ├── network_actions.py      # Search, URL, weather
│   │   ├── file_actions.py         # Create folder, move, copy
│   │   └── utility_actions.py      # Window mgmt, timer, clipboard
│   │
│   ├── intent_engine/
│   │   ├── __init__.py             # IntentEngine class (orchestrator)
│   │   ├── base.py                 # IntentResult DTO + grammar words
│   │   ├── meta_intents.py         # greeting, thanks, exit, confirm/deny
│   │   ├── info_intents.py         # time, CPU, RAM, battery, disk, IP
│   │   ├── app_intents.py          # open/close/list apps
│   │   ├── media_intents.py        # volume, mute, play, stop
│   │   ├── system_intents.py       # lock, screenshot, shutdown, brightness
│   │   ├── desktop_intents.py      # scroll, click, press key, type
│   │   ├── file_intents.py         # create folder, move, copy
│   │   ├── network_intents.py      # search, URL, weather, research
│   │   ├── os_intents.py           # self-check, diagnostics, kill
│   │   ├── cognitive_intents.py    # goals, predictions, brain recall
│   │   └── runtime_mode_intents.py # perf mode, brain profile, assistant mode
│   │
│   ├── cognitive/
│   │   ├── __init__.py
│   │   ├── second_brain.py         # Unified knowledge store
│   │   ├── goal_engine.py          # Goal → Plan → Execute → Evaluate
│   │   ├── prediction_engine.py    # Predict next user action
│   │   ├── behavior_model.py       # Personal profile + energy inference
│   │   └── self_optimizer.py       # Suggest improvements to ATOM itself
│   │
│   ├── reasoning/                   # v18 Agentic Reasoning Engine
│   │   ├── __init__.py
│   │   ├── tool_registry.py        # 40+ tool definitions (params, safety, schemas)
│   │   ├── tool_parser.py          # Multi-format tool call parser (4 formats)
│   │   ├── action_executor.py      # Security-gated tool execution bridge
│   │   ├── planner.py              # Multi-step planning (templates + LLM)
│   │   └── code_sandbox.py         # Safe Python eval (restricted builtins)
│   │
│   ├── cache_engine.py             # TTL LRU + Jaccard similarity
│   ├── memory_engine.py            # Keyword Q&A memory + interaction log
│   ├── conversation_memory.py      # Rolling conversation window
│   ├── command_cache.py            # Intent result caching
│   ├── command_registry.py         # commands.json loader
│   ├── command_filter.py           # Pre-classification filtering
│   ├── skills_registry.py          # skills.json phrase expansion
│   ├── quick_replies.py            # Instant replies (no LLM needed)
│   ├── behavior_tracker.py         # Habit formation from patterns
│   ├── autonomy_engine.py          # Autonomous habit execution
│   ├── self_evolution.py           # Performance diagnostics
│   ├── health_monitor.py           # CPU governor + stuck recovery
│   ├── runtime_watchdog.py         # Thinking/Speaking timeout
│   ├── priority_scheduler.py       # Voice > LLM > Background queue
│   ├── llm_inference_queue.py      # Serial LLM with coalescing
│   ├── brain_mode_manager.py       # Brain profile switching
│   ├── assistant_mode_manager.py   # Assistant mode switching
│   ├── task_scheduler.py           # Reminder system
│   ├── process_manager.py          # System process monitoring
│   ├── system_watcher.py           # Network/power/BT events
│   ├── proactive_awareness.py      # Time-of-day greetings
│   ├── desktop_control.py          # Mouse/keyboard automation
│   └── web_researcher.py           # DuckDuckGo lookup (offline: disabled)
│
├── voice/
│   ├── __init__.py
│   ├── stt_async.py                # Multi-engine STT orchestrator
│   ├── tts_async.py                # Windows SAPI TTS
│   ├── tts_edge.py                 # Edge Neural TTS
│   ├── mic_manager.py              # Microphone device management
│   ├── audio_preprocessor.py       # Audio conditioning (noise gate, normalization)
│   ├── speech_detector.py          # Noise filtering, text corrections
│   └── voice_profiles.py           # Voice configuration profiles
│
├── brain/
│   ├── __init__.py
│   └── mini_llm.py                 # GPU-accelerated LLM (llama.cpp, single 9B/13B model, true streaming)
│
├── cursor_bridge/
│   ├── __init__.py
│   ├── local_brain_controller.py   # Agentic brain controller (ReAct loop, tool-use, streaming)
│   └── structured_prompt_builder.py # 7-layer agentic prompt (System/Tools/Context/Memory/History/Observations/Query)
│
├── context/
│   ├── context_engine.py           # Active window + clipboard context
│   └── privacy_filter.py           # PII/secret redaction
│
├── ui/
│   ├── __init__.py
│   ├── web_dashboard.py            # JARVIS web UI (aiohttp + WebSocket)
│   ├── floating_indicator.py       # Tkinter fallback UI
│   └── dashboard/
│       └── index.html              # Three.js JARVIS dashboard
│
├── scripts/
│   └── enroll_owner_face.py        # One-time face enrollment
│
├── tests/
│   ├── test_all_components.py
│   ├── test_context_engine.py
│   ├── test_deployment_profile.py
│   ├── test_heavy_deployment.py
│   ├── test_jarvis_upgrades.py
│   ├── test_mic_manager.py
│   ├── test_session_and_skills.py
│   └── test_state_machine.py
│
├── tools/
│   └── e2e_benchmark.py            # End-to-end performance benchmark
│
├── docs/                           # Architecture docs, reviews, reports
│   ├── README.md
│   ├── ATOM_Full_System_Review.md
│   ├── ATOM_OS_Review.md
│   ├── ATOM_Production_Readiness_Scorecard.md
│   └── ... (8 more docs)
│
└── logs/                           # Runtime data (gitignored)
    ├── memory.json                 # Q&A memory
    ├── interactions.json           # Full interaction history
    ├── second_brain.json           # Facts + preferences + corrections
    ├── goals.json                  # Goal engine state
    ├── user_profile.json           # Behavior model profile
    ├── optimizer.json              # Self-optimizer data
    ├── evolution.json              # Evolution history
    ├── autonomy.log                # Autonomy decision log
    └── audit.log                   # Security audit trail
```

## Appendix B: Module Count Summary

| Category | Count |
|----------|-------|
| Core modules | 30 |
| Intent sub-modules | 12 |
| Router action modules | 6 |
| Reasoning modules | 5 |
| Voice modules | 7 |
| Cognitive modules | 5 |
| Brain modules (GPU) | 2 |
| **JARVIS Intelligence modules (v19)** | **5** |
| UI modules | 3 |
| Context modules | 2 |
| Config files | 4 |
| Test files | 8 |
| **Total source files** | **~90** |
| **Total events** | **~55** |
| **Total registered tools** | **40+** |
| **Total commands** | **40+** |
| **Total intent patterns** | **200+** |

## Appendix C: Quick Reference — What Controls What

| Want to change... | Edit this... |
|-------------------|-------------|
| How ATOM speaks | `core/personality.py`, `core/personality_modes.py` |
| What commands exist | `core/reasoning/tool_registry.py` + `core/intent_engine/*.py` + `core/router/*.py` |
| How fast ATOM thinks | `config/settings.json` → `brain.*` (GPU layers, ctx, batch), `performance.*` |
| Security rules | `core/security_policy.py`, `config/settings.json` → `security.*` |
| Dashboard look | `ui/dashboard/index.html` |
| Which mic is used | `config/settings.json` → `mic.*` |
| Wake word behavior | `voice/stt_async.py` (future: dedicated wake word module) |
| Goal tracking | `core/cognitive/goal_engine.py` |
| Habit learning | `core/behavior_tracker.py` + `core/autonomy_engine.py` |
| Performance tuning | `config/settings.json` → `performance.*` |
| What gets cached | `core/cache_engine.py`, `core/command_cache.py` |
| What gets remembered | `core/memory_engine.py`, `core/cognitive/second_brain.py` |
| How ATOM understands you | `core/owner_understanding.py` — emotional, communication, topic profiles |
| How ATOM scans the system | `core/system_scanner.py` — hardware, software, health, bottlenecks |
| JARVIS-level intelligence | `core/jarvis_core.py` — proactive insights, contextual inference, briefings |
| Cross-platform behavior | `core/platform_adapter.py` — Windows/Linux/macOS abstraction |
| Advanced system control | `core/system_control.py` — processes, services, network, power, storage |

---

> **ATOM is not finished. ATOM is never finished. It is a living system that grows with its owner.**
> **Every module is an organ. Every event is a nerve signal. Every config is a gene.**
> **v19 brings JARVIS-level understanding: ATOM now knows its owner, knows its system, anticipates needs, and runs on any platform.**
> **Build one module at a time. Test. Evolve. Repeat.**
>
> — *Satyam, ATOM Creator*
