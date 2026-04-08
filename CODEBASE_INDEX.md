# ATOM Codebase Index

> **Quick-reference map for AI assistants and developers.**
> Read this file FIRST before exploring or modifying ATOM.
> Total: ~51,400 lines Python + ~1,850 lines HTML/JSON config across 155 files.

---

## 1. What ATOM Is

ATOM is Satyam's **personal cognitive AI operating system** — a local-first, JARVIS-style voice OS that perceives (voice, system state), reasons (intent → cache/memory → RAG/graph → local LLM), acts (security-gated tools), and learns (behavior, habits, feedback metrics). It is **not** a chatbot — it is an always-listening AI OS with a 9-layer LLM prompt, GPU-accelerated inference, and autonomous decision-making.

**Owner:** Satyam ("Boss")
**Language:** Python 3.11+ (98.4%), HTML (1.6%)
**LLM:** Offline via llama-cpp-python (current: Qwen3-8B-Q4_K_M GGUF). Planned: MLX dual-model — Qwen3-4B (primary) + Qwen3-1.7B (fast brain)
**Voice:** faster-whisper STT + Edge Neural TTS (en-GB-RyanNeural)
**UI:** aiohttp web dashboard with Three.js animated orb + WebSocket push

---

## 2. Pipeline (end-to-end data flow)

```
Mic → MicManager → STT (faster-whisper) → IntentEngine (<5ms regex)
  ├── KNOWN COMMAND → Router direct dispatch (sub-ms)
  └── FALLBACK → Cache/Memory lookup → LocalBrainController (LLM)
        → StructuredPromptBuilder (9-layer prompt + ContextFusion + RealWorldIntel)
        → MiniLLM (GPU inference, ReAct loop, tool use)
        → ActionExecutor (40+ tools, security-gated)
        → partial_response streaming → TTS (Edge Neural, sentence-by-sentence)
        → WebDashboard (WebSocket push)

Background:
  AutonomyEngine → BehaviorTracker → PredictionEngine → ProactiveEngine
  SelfHealingEngine monitors everything; SecurityFortress gates every action
```

---

## 3. Entry Points

| File | Purpose | Run with |
|------|---------|----------|
| `main.py` (1534L) | Primary entry. Async event loop, wires all modules, runs voice pipeline | `python main.py` |
| `main.py --v3` | Multi-process mode via ZMQ broker | `python main.py --v3` |
| `main.py --v4` | V4 Cognitive OS mode (brain orchestrator) | `python main.py --v4` |
| `run_v3.py` (72L) | V3 multi-process orchestrator (spawns broker + workers) | `python run_v3.py` |
| `run_v4.py` (98L) | V4 orchestrator (broker + STT/TTS/LLM + brain workers) | `python run_v4.py` |
| `atom_cli.py` (110L) | CLI interface for text-based interaction | `python atom_cli.py` |
| `refactor.py` (48L) | One-shot refactoring utility | `python refactor.py` |

---

## 4. Directory Map

### `core/` — Engine (87 files, ~25,000 lines)

The nervous system. Everything flows through here.

#### State & Events
| File | Lines | What it does |
|------|-------|-------------|
| `state_manager.py` | 224 | 6-state machine: SLEEP→IDLE→LISTENING→THINKING→SPEAKING→ERROR_RECOVERY. Thread-safe, transition metrics |
| `async_event_bus.py` | 311 | PriorityEventBus: fire-and-forget async pub/sub. All inter-module communication. Priority queues, 10s handler timeout |
| `event_ring.py` | ~80 | Circular event buffer for debugging |

#### Routing & Intent
| File | Lines | What it does |
|------|-------|-------------|
| `router/router.py` | 1065 | 3-layer routing: Intent→Cache→LLM. Security-gated dispatch to action sub-modules |
| `router/system_actions.py` | ~300 | OS control: shutdown, restart, volume, brightness |
| `router/app_actions.py` | ~250 | App launch/close via ProcessManager |
| `router/media_actions.py` | ~200 | YouTube, Spotify, media control |
| `router/file_actions.py` | ~200 | File/folder CRUD operations |
| `router/network_actions.py` | ~150 | Network diagnostics, speedtest |
| `router/utility_actions.py` | ~250 | Timer, calculator, clipboard, type-text |
| `router/confirmation_manager.py` | ~120 | Dangerous-action confirmation flow |
| `router/diagnostics_handler.py` | ~100 | System diagnostics dispatch |
| `router/conversation_manager.py` | ~150 | Conversation compression, context tracking |
| `intent_engine/__init__.py` | 100 | IntentEngine: <5ms regex classifier. Chain of: meta→runtime→info→world→system→media→desktop→file→network→os→cognitive→app |
| `intent_engine/base.py` | 105 | IntentResult dataclass, grammar words, slot cleaning |
| `intent_engine/app_intents.py` | ~180 | "open chrome", "close notepad" patterns |
| `intent_engine/system_intents.py` | ~200 | "cpu usage", "battery", "shutdown" patterns |
| `intent_engine/media_intents.py` | ~150 | "play", "pause", "volume" patterns |
| `intent_engine/desktop_intents.py` | ~150 | "screenshot", "minimize", "lock screen" |
| `intent_engine/file_intents.py` | ~120 | "create folder", "move file" patterns |
| `intent_engine/info_intents.py` | ~120 | "what time", "what date", "weather" |
| `intent_engine/network_intents.py` | ~100 | "my ip", "ping", "speedtest" |
| `intent_engine/os_intents.py` | ~100 | "self check", OS-level queries |
| `intent_engine/meta_intents.py` | ~100 | "stop listening", "go to sleep", "thank you" |
| `intent_engine/cognitive_intents.py` | ~120 | "set goal", "show predictions", "dream mode" |
| `intent_engine/runtime_mode_intents.py` | ~80 | "command mode", "hybrid mode", "brain mode" |
| `intent_engine/world_intents.py` | ~80 | "news", "world time" patterns |

#### Intelligence & Cognition
| File | Lines | What it does |
|------|-------|-------------|
| `jarvis_core.py` | 924 | The "mind" — proactive intelligence, 12 insight categories, 4-tier briefings, deep contextual inference |
| `context_fusion.py` | 376 | "The Thalamus" — fuses 7 layers: situation, conversation, owner, system, memory, action, meta |
| `owner_understanding.py` | 741 | Owner model: expertise, preferences, communication style, emotional state, energy |
| `real_world_intel.py` | 651 | Weather (wttr.in), news (RSS), calendar, world clock, location, temporal awareness |
| `proactive_awareness.py` | ~300 | Workflow/behavioral/temporal trigger engine |
| `adaptive_personality.py` | ~250 | Context-aware, emotion-responsive expression style |
| `personality_modes.py` | ~150 | Personality mode definitions |

#### Memory & Knowledge
| File | Lines | What it does |
|------|-------|-------------|
| `memory_engine.py` | ~400 | Keyword+semantic memory retrieval, top-K ranking |
| `conversation_memory.py` | ~200 | Sliding window of conversation turns (max 20) |
| `cache_engine.py` | ~250 | LRU + Jaccard similarity cache for repeated queries |
| `command_cache.py` | ~150 | Fast command-to-response cache |
| `l1_cache.py` | ~100 | Hot-path L1 cache layer |
| `embedding_engine.py` | ~200 | Sentence-transformer embeddings (all-MiniLM-L6-v2) |
| `vector_store.py` | ~300 | ChromaDB vector store backend |
| `document_ingestion.py` | ~250 | Document chunking and vector indexing |

#### RAG (Retrieval-Augmented Generation)
| File | Lines | What it does |
|------|-------|-------------|
| `rag/rag_engine.py` | 588 | GPU-aware RAG: query classifier → embed → retrieve → re-rank (vector+keyword+recency) |
| `rag/graph_rag.py` | 65 | Graph-augmented RAG via MemoryGraph |
| `rag/query_classifier.py` | ~100 | Classifies query complexity: SIMPLE/MODERATE/COMPLEX |
| `rag/context_builder.py` | ~100 | Builds RAG enrichment blocks for prompts |
| `rag/prefetch_engine.py` | ~200 | Speculative prefetch based on prediction confidence |
| `rag/adaptive_budget.py` | ~80 | Dynamic time budget for RAG retrieval |
| `rag/rag_cache.py` | ~100 | Embed + retrieval TTL caches |
| `rag/embedding_disk_cache.py` | ~120 | Persistent on-disk embedding cache |
| `rag/qdrant_backend.py` | ~150 | Optional Qdrant vector backend |

#### Security
| File | Lines | What it does |
|------|-------|-------------|
| `security_policy.py` | 534 | 11-layer security gate: action allow/deny, shell blocklist, path allowlist, rate limiting, injection defense |
| `security_fortress.py` | 977 | 7-gate fortress: OwnerAuth, EncryptedVault, IntegrityMonitor, SessionManager, AuditTrail, ThreatDetector |
| `security_context.py` | ~150 | Security context propagation |
| `security/action_signing.py` | ~100 | Cryptographic action signing |
| `action_safety.py` | ~100 | Action safety classification |
| `owner_gate.py` | ~100 | Owner identity gate |
| `auth/voice_auth.py` | 684 | VoicePrint biometric authentication |
| `auth/behavior_auth.py` | ~200 | Behavioral authentication (typing patterns, usage patterns) |

#### V7 Runtime Intelligence
| File | Lines | What it does |
|------|-------|-------------|
| `runtime/modes.py` | 248 | 4 modes: FAST/SMART/DEEP/SECURE. Auto-selects based on query complexity, CPU/GPU load |
| `runtime/v7_context.py` | 37 | V7RuntimeContext dataclass — shared snapshot across all V7 modules |
| `cognition/feedback_engine.py` | ~300 | Runtime feedback: tracks accuracy, learns from user corrections |
| `cognition/predictor.py` | ~200 | Query-level prediction for prefetch |
| `cognition/preemption.py` | ~150 | Mid-query preemption logic |
| `cognition/suggester.py` | ~100 | Proactive suggestion engine |
| `observability/debug_snapshot.py` | ~100 | Periodic V7 debug snapshots |
| `observability/warnings.py` | ~100 | V7 health warnings |

#### Cognitive Layer (Ring 6)
| File | Lines | What it does |
|------|-------|-------------|
| `cognitive/second_brain.py` | 340 | Vector-enhanced knowledge store: facts, preferences, corrections |
| `cognitive/goal_engine.py` | 502 | Goal lifecycle: create→decompose→track→evaluate→briefing. Max 20 goals |
| `cognitive/prediction_engine.py` | 294 | Predictive action engine: time-slot frequency + transition probability |
| `cognitive/behavior_model.py` | ~250 | User state model: work/relax/focus/creative modes |
| `cognitive/self_optimizer.py` | ~200 | Self-optimization: parameter tuning based on feedback |
| `cognitive/dream_engine.py` | ~200 | "Dream mode" — offline consolidation and insight generation |
| `cognitive/curiosity_engine.py` | ~150 | Autonomous curiosity: asks questions about gaps in knowledge |
| `cognitive/proactive_engine.py` | ~200 | Proactive intelligence triggers |

#### Autonomy & Scheduling
| File | Lines | What it does |
|------|-------|-------------|
| `autonomy_engine.py` | 353 | Auto-executes trusted habits (>=0.95 confidence), suggests others |
| `behavior_tracker.py` | ~350 | Habit frequency tracking with confidence decay |
| `task_scheduler.py` | ~250 | Async task scheduling with priorities |
| `priority_scheduler.py` | ~200 | Priority-based execution queue |

#### Apple Silicon Compute Layer
| File | Lines | What it does |
|------|-------|-------------|
| `apple_silicon_monitor.py` | 294 | Hardware data source: Unified Memory, thermal pressure, battery, GPU info |
| `silicon_governor.py` | 145 | Monitoring loop + thermal/memory event emission (Apple Silicon-only) |
| `inference_guard.py` | 155 | Model slot tracking, memory pressure admission, idle unload policy |
| `gpu_watchdog.py` | 73 | Inference stall detection (platform-agnostic) |

#### macOS Native Modules (`core/macos/`)
| File | Lines | What it does |
|------|-------|-------------|
| `__init__.py` | 1 | Package init |
| `fs_watcher.py` | ~200 | **FSEvents** kernel-level file watcher. Near-zero CPU, same mechanism as Spotlight. Watches Desktop/Downloads/Documents for proactive awareness |

#### System & Health
| File | Lines | What it does |
|------|-------|-------------|
| `self_healing.py` | 1060 | 5 sub-engines: ExceptionTracker, ModuleHealthChecker, FailureAnalyzer, FixEngine, StartupValidator |
| `health_monitor.py` | ~300 | Periodic health checks + stuck state detection |
| `system_scanner.py` | 1032 | Deep hardware profiling: CPU, GPU, RAM, disk, battery, network |
| `system_control.py` | 611 | OS control: volume, brightness, power, display |
| `system_watcher.py` | ~200 | Background system state monitoring |
| `system_indexer.py` | ~250 | Installed app discovery and indexing |
| `system_health_score.py` | ~100 | Composite health score (0-100) |
| `code_introspector.py` | 650 | Self-aware codebase analysis (ATOM can read its own code) |
| `process_manager.py` | ~350 | Process launch/kill with safety checks |
| `platform_adapter.py` | 840 | Cross-platform abstraction (Windows/macOS/Linux) |
| `desktop_control.py` | ~200 | Mouse/keyboard automation |
| `power_governor.py` | ~150 | Battery-aware power management |
| `watchdog_service.py` | ~200 | Process-level watchdog |
| `runtime_watchdog.py` | ~150 | Runtime state watchdog |

#### Reasoning & Tools
| File | Lines | What it does |
|------|-------|-------------|
| `reasoning/tool_registry.py` | 302 | Formal tool registration: name, params, safety level. 40+ tools |
| `reasoning/action_executor.py` | ~350 | Executes tool calls from LLM with safety checks |
| `reasoning/tool_parser.py` | ~150 | Parses LLM output for tool call JSON |
| `reasoning/planner.py` | ~200 | Multi-step plan generation |
| `reasoning/workflow_engine.py` | ~250 | Named workflow execution |
| `reasoning/code_sandbox.py` | ~150 | Sandboxed Python code execution |

#### Boot & Wiring
| File | Lines | What it does |
|------|-------|-------------|
| `boot/config_loader.py` | ~150 | Loads and validates config/settings.json |
| `boot/wiring.py` | 486 | Wires all event bus handlers. The "nervous system" connection layer |
| `boot/services.py` | 85 | AtomServices dataclass — dependency container |

#### IPC (Inter-Process Communication)
| File | Lines | What it does |
|------|-------|-------------|
| `ipc/zmq_bus.py` | ~200 | ZeroMQ event bus for multi-process mode |
| `ipc/zmq_broker.py` | ~150 | ZMQ broker process |
| `ipc/interrupt_manager.py` | ~150 | Cross-process interrupt handling |
| `ipc/proxies.py` | ~100 | IPC proxy objects |

#### Other Core
| File | Lines | What it does |
|------|-------|-------------|
| `config_schema.py` | 1068 | JSON Schema for settings.json validation |
| `contracts.py` | ~80 | CognitiveModuleContract protocol |
| `metrics.py` | ~200 | Metrics collection and health logging |
| `logging_setup.py` | ~100 | Logging configuration |
| `persistence_manager.py` | ~150 | JSON file persistence with atomic writes |
| `profiler.py` | ~80 | @profile decorator for timing |
| `unified_trace.py` | ~100 | Distributed tracing |
| `telemetry_engine.py` | ~200 | Telemetry collection and batching |
| `pipeline_timer.py` | ~80 | Pipeline stage timing |
| `web_researcher.py` | ~250 | Web search integration |
| `skills_registry.py` | ~200 | Skill definition and lookup |
| `command_registry.py` | ~150 | Static command registration |
| `command_filter.py` | ~100 | Command filtering and normalization |
| `lock_modes.py` | ~80 | Lock mode definitions |
| `deployment_profile.py` | ~150 | Corporate vs home deployment profiles |
| `runtime_config.py` | ~100 | Runtime configuration helpers |
| `quick_replies.py` | ~80 | Fast static reply lookup |
| `self_evolution.py` | ~300 | Self-evolution engine (code generation) |
| `recovery_manager.py` | ~150 | Crash recovery |
| `fast_path.py` | ~100 | Ultra-fast command dispatch |
| `llm_inference_queue.py` | ~150 | LLM inference request queuing |
| `brain_mode_manager.py` | ~200 | Brain profile switching (atom/balanced/brain) |
| `assistant_mode_manager.py` | ~200 | Assistant mode switching (command/hybrid/brain) |

#### Wiring Handlers
| File | Lines | What it does |
|------|-------|-------------|
| `wiring/cognitive_handlers.py` | ~200 | Event handlers for cognitive modules |
| `wiring/feature_handlers.py` | ~200 | Event handlers for feature modules |
| `wiring/intelligence_handlers.py` | ~200 | Event handlers for intelligence modules |

---

### `brain/` — Cognitive Brain (19 files, ~4,500 lines)

Higher-order reasoning, planning, and learning.

| File | Lines | What it does |
|------|-------|-------------|
| `mini_llm.py` | 376 | Offline LLM wrapper: llama-cpp-python, GPU offload, KV cache, token streaming |
| `memory_graph.py` | 495 | SQLite + ChromaDB graph: MemoryNode (episodic/semantic/procedural), relationship edges, hybrid query |
| `intent_engine.py` | ~200 | Brain-level intent classification (separate from core fast-path) |
| `context_router.py` | ~200 | Routes queries to appropriate brain subsystem |
| `planning_engine.py` | ~300 | Multi-step plan generation with LLM |
| `execution_engine.py` | ~250 | Plan step execution with rollback |
| `learning_engine.py` | ~250 | Experience-based learning from outcomes |
| `reflection_engine.py` | ~200 | Post-action reflection and improvement |
| `simulation_engine.py` | ~200 | Plan outcome simulation |
| `plan_evaluator.py` | ~150 | Plan quality scoring |
| `plan_registry.py` | ~150 | Named plan templates |
| `exploration_engine.py` | ~150 | Knowledge gap exploration |
| `goal_engine.py` | ~300 | V4 goal management (separate from core cognitive) |
| `behavior_model.py` | ~200 | V4 behavior model |
| `proactive_engine.py` | ~200 | V4 proactive intelligence |
| `skill_engine.py` | ~200 | Skill definition and execution |
| ~~`gpu_pipeline.py`~~ | — | DELETED (Step 1.3A — 1 useful line inlined) |
| `local_cognitive_pipeline.py` | ~250 | In-process cognitive pipeline (V4) |

---

### `voice/` — Perception & Expression (13 files, ~5,400 lines)

Audio input/output pipeline.

| File | Lines | What it does |
|------|-------|-------------|
| `stt_async.py` | 969 | faster-whisper STT: bilingual (EN+HI), audio preprocessing, noise gate, hallucination detection |
| `stt_macos.py` | ~280 | **macOS native STT**: SFSpeechRecognizer + AVAudioEngine. On-device Neural Engine, ~50ms commands, built-in wake word, HW noise suppression |
| `tts_async.py` | 252 | Windows SAPI TTS: COM-based, async speak, barge-in, markdown cleanup |
| `tts_edge.py` | 784 | Edge Neural TTS: Microsoft neural voices, SSML, sentence streaming, emotion profiles, RMS normalization |
| `tts_macos.py` | ~350 | **macOS native TTS**: NSSpeechSynthesizer (direct API, 4.4ms barge-in, 184 voices) + `say` fallback. Premium neural voice auto-detection |
| `tts_kokoro.py` | 146 | Kokoro TTS alternative backend |
| `mic_manager.py` | 350 | Microphone ownership lock: device profiling, auto-selection, quality scoring |
| `speech_detector.py` | ~250 | VAD, energy detection, silence timeout, noise word filtering |
| `audio_preprocessor.py` | ~200 | DC removal, pre-emphasis, spectral noise gate |
| `voice_profiles.py` | 160 | Voice emotion profiles: rate/pitch/volume per context |
| `emotion_detector.py` | ~150 | Audio emotion detection from speech patterns |
| `wake_word.py` | 192 | Wake word detection ("Hey ATOM") — superseded by stt_macos.py keyword detection |
| `media_watcher.py` | ~230 | **macOS native**: AppleScript queries for Spotify, Music, browser media. Winsdk fallback on Windows |

---

### `ui/` — Dashboard (3 files, ~1,700 lines)

| File | Lines | What it does |
|------|-------|-------------|
| `web_dashboard.py` | 683 | aiohttp server: HTTP + WebSocket, system status push, conversation log, mode switcher, V7 health endpoint |
| `floating_indicator.py` | 785 | Tkinter floating indicator (legacy/optional) |
| `dashboard/index.html` | ~900 | Three.js animated orb, real-time panels, activity monitor, conversation log |

---

### `cursor_bridge/` — LLM Brain Controller (3 files, ~1,050 lines)

| File | Lines | What it does |
|------|-------|-------------|
| `local_brain_controller.py` | 853 | Agentic brain: Query→Prompt→LLM→Parse tool calls→ActionExecutor→ReAct loop (max 3 steps)→Stream response |
| `structured_prompt_builder.py` | ~200 | 9-layer prompt construction: system persona, tools, context fusion, real-world intel, conversation history |

---

### `context/` — Context Awareness (4 files, ~600 lines)

| File | Lines | What it does |
|------|-------|-------------|
| `context_engine.py` | ~300 | Clipboard + active window context capture |
| `privacy_filter.py` | ~200 | Redacts sensitive data (passwords, keys, PII) from context |
| `screen_reader.py` | ~270 | **Apple Vision OCR** (Neural Engine, 109ms avg) + EasyOCR fallback. Screenshot via screencapture |

---

### `services/` — V4 Multi-Process Workers (9 files, ~2,500 lines)

Each runs as a separate process, communicating via ZMQ.

| File | Lines | What it does |
|------|-------|-------------|
| `brain_orchestrator.py` | 369 | Central brain process: goals, planning, execution, learning, reflection, simulation |
| `intent_worker.py` | ~200 | Intent classification worker |
| `stt_worker.py` | ~200 | STT processing worker |
| `tts_worker.py` | ~200 | TTS synthesis worker |
| `llm_worker.py` | ~300 | LLM inference worker |
| `context_worker.py` | ~200 | Context gathering worker |
| `decision_worker.py` | ~200 | Decision-making worker |
| `memory_worker.py` | ~200 | Memory operations worker |
| `gpu_cognition_worker.py` | ~250 | GPU-accelerated cognition worker |

---

### `config/` — Configuration (6 files)

| File | What it does |
|------|-------------|
| `settings.json` | Active configuration (379 lines). All tunable parameters |
| `settings.desktop.example.json` | Desktop-optimized config template |
| `settings.corporate.example.json` | Corporate/restricted config template |
| `commands.json` | Static command definitions |
| `skills.json` | Skill definitions |
| `plan_registry.json` | Named plan templates |

---

### `scripts/` — Testing & Utilities (5 files)

| File | What it does |
|------|-------------|
| `test_v4_system.py` | V4 architecture integration test |
| `v7_stress_test.py` | V7 intelligence stress test |
| `v7_chaos_test.py` | V7 chaos/fault injection test |
| `v7_long_run.py` | V7 long-running stability test |
| `enroll_owner_face.py` | Owner face enrollment for vision auth |

---

### `tests/` — Test Suite (9 files)

| File | What it tests |
|------|--------------|
| `test_all_components.py` | Smoke test for all module imports |
| `test_state_machine.py` | State transitions and edge cases |
| `test_context_engine.py` | Context capture and fusion |
| `test_deployment_profile.py` | Deployment profile loading |
| `test_heavy_deployment.py` | Full deployment simulation (1523L) |
| `test_jarvis_upgrades.py` | JarvisCore intelligence |
| `test_mic_manager.py` | Microphone management |
| `test_session_and_skills.py` | Session and skill lifecycle |

---

### `docs/` — Documentation (22 files)

| File | Content |
|------|---------|
| `ATOM_ARCHITECTURE_BLUEPRINT.md` | Full system blueprint (72K) |
| `ATOM_HLD.md` | High-level design |
| `ATOM_LLD.md` | Low-level design |
| `ATOM_CODE_REVIEW_AND_DESKTOP_PLAN.md` | Code review + desktop migration plan |
| `ATOM_V4_Cognitive_OS_Blueprint.md` | V4 cognitive architecture |
| `ATOM_Deployment_Profiles.md` | Corporate vs home hardware |
| `ATOM_Full_System_Review.md` | Complete system review |
| `ATOM_DEEP_TECHNICAL_REVIEW.md` | Deep technical analysis |
| `ATOM_Production_Readiness_Rating.md` | Production readiness assessment |
| `architecture/INDEX.md` | 14-chapter architecture guide |
| `architecture/00-13_*.md` | System identity through upgrade playbook |

---

## 5. Key Configuration (settings.json)

| Section | Key settings |
|---------|-------------|
| `owner` | name="Satyam", title="Boss" |
| `brain` | model_path="models/qwen3-8b-q4_k_m.gguf", n_ctx=8192, n_gpu_layers=-1, temperature=0.7. **Planned:** Qwen3-4B primary + Qwen3-1.7B fast brain (dual-model, Phase 3) |
| `stt` | engine="faster_whisper", model="small", bilingual=true, sample_rate=16000 |
| `tts` | engine="edge", voice="en-GB-RyanNeural" |
| `ui` | web_port=8765, mode="desktop" |
| `security` | mode="strict", require_confirmation_for=[shutdown, restart, kill, ...] |
| `v7_intelligence` | default_mode="SMART", auto_mode=true, 4 modes: FAST/SMART/DEEP/SECURE |
| `cognitive` | goals, predictions, behavior model, dream mode, curiosity all enabled |
| `autonomy` | auto_execute_threshold=0.95, suggest_threshold=0.72 |
| `assistant_brain.profiles` | atom (fast, 4K ctx), balanced (8K ctx), brain (deep, 16K ctx) |

---

## 6. State Machine

```
SLEEP ──→ IDLE ──→ LISTENING ──→ THINKING ──→ SPEAKING
  ↑         ↑         │              │            │
  │         └─────────←┘              │            │
  │         └─────────────────────────←┘           │
  │         └──────────────────────────────────────←┘
  │                                                │
  └── ERROR_RECOVERY ←────────────────────────────←┘
            │
            └──→ IDLE (auto-recovery)
```

---

## 7. Event Bus Events (key ones)

| Event | Emitted by | Consumed by |
|-------|-----------|-------------|
| `state_changed` | StateManager | STT, TTS, Dashboard, Watchdog |
| `speech_final` | STT | Router (query processing) |
| `response_ready` | Router/Brain | TTS (speak), Dashboard (display) |
| `partial_response` | Brain | TTS (streaming), Dashboard |
| `tool_executed` | ActionExecutor | Brain (ReAct loop), Metrics |
| `metrics_latency` | Various | Metrics, Dashboard |
| `goal_update` | GoalEngine | Dashboard, SecondBrain |
| `prediction` | PredictionEngine | AutonomyEngine, Dashboard |
| `habit_suggestion` | AutonomyEngine | Dashboard, TTS |

---

## 8. Security Layers

1. **SecurityPolicy** — action allow/deny, shell blocklist, path allowlist
2. **SecurityFortress** — OwnerAuth, EncryptedVault, IntegrityMonitor
3. **VoicePrintAuth** — voiceprint biometric verification
4. **BehavioralAuth** — typing/usage pattern matching
5. **ActionSigning** — cryptographic action verification
6. **OwnerGate** — owner identity verification
7. **PrivacyFilter** — PII/credential redaction from context

---

## 9. Dependencies

| Package | Purpose |
|---------|---------|
| `llama-cpp-python` | Local LLM inference (GGUF models, Metal GPU offload) |
| `pyobjc-framework-*` | macOS native bridge (~18MB): STT, TTS, OCR, FSEvents, AVFoundation |
| `chromadb` | Vector store |
| `sentence-transformers` | Embedding model (MPS-accelerated on Apple Silicon) |
| `aiohttp` | Web dashboard server |
| `psutil` | System monitoring |
| `numpy` | Audio processing |
| `pyzmq` | IPC (V3/V4 multi-process) |
| `faster-whisper` | (optional) Offline STT — replaced by native SFSpeechRecognizer on macOS |
| `edge-tts` | (optional) Neural TTS — replaced by NSSpeechSynthesizer on macOS |
| `easyocr` | (optional) OCR — replaced by Apple Vision on macOS |

---

## 10. macOS / Apple Silicon Notes

- **GPU:** llama-cpp-python uses Metal (MPS) on Apple Silicon. Set `n_gpu_layers=-1` for full offload.
- **Planned LLM (Phase 3):** Dual-model via MLX — Qwen3-4B primary (3GB, 50-70 tok/s, thinking mode) + Qwen3-1.7B fast brain (1.2GB, 120-160 tok/s). Total 4.2GB, leaves 5.8GB headroom on 16GB M5. Same Qwen3 family = zero prompt/tool-call adaptation.
- **STT:** `voice/stt_macos.py` — SFSpeechRecognizer + AVAudioEngine. On-device Neural Engine, ~50ms. Replaces faster-whisper + PyAudio on macOS.
- **TTS:** `voice/tts_macos.py` — NSSpeechSynthesizer (no subprocess, 4.4ms barge-in). Download premium voices: System Settings > Accessibility > Spoken Content > Manage Voices.
- **OCR:** `context/screen_reader.py` — Apple Vision framework. Neural Engine, 109ms avg. Replaces EasyOCR.
- **Media:** `voice/media_watcher.py` — AppleScript queries for Spotify, Music, browser. Replaces broken winsdk.
- **FileWatch:** `core/macos/fs_watcher.py` — FSEvents kernel-level monitoring. Near-zero CPU.
- **Mic:** AVAudioEngine with Voice Processing I/O (hardware echo cancellation + noise suppression). No PyAudio/PortAudio needed.
- **Embeddings:** `core/embedding_engine.py` auto-detects MPS for Apple Silicon GPU acceleration
- **Platform:** `core/platform_adapter.py` handles OS differences. Most Windows paths now have macOS equivalents.
- **Dependencies:** pyobjc (~18MB) bridges to Apple frameworks already in macOS RAM. Replaces ~2.8GB of third-party voice/OCR deps.

---

## 11. Quick Navigation Guide

**"I want to understand how ATOM processes a voice command"**
→ `voice/stt_macos.py` (macOS) or `voice/stt_async.py` (cross-platform) → `core/intent_engine/` → `core/router/router.py` → `cursor_bridge/local_brain_controller.py`

**"I want to understand the LLM brain"**
→ `brain/mini_llm.py` → `cursor_bridge/structured_prompt_builder.py` → `cursor_bridge/local_brain_controller.py`

**"I want to understand memory/knowledge"**
→ `core/memory_engine.py` → `brain/memory_graph.py` → `core/cognitive/second_brain.py` → `core/rag/rag_engine.py`

**"I want to understand security"**
→ `core/security_policy.py` → `core/security_fortress.py` → `core/auth/`

**"I want to understand the dashboard"**
→ `ui/web_dashboard.py` → `ui/dashboard/index.html`

**"I want to understand V7 intelligence modes"**
→ `core/runtime/modes.py` → `core/runtime/v7_context.py` → `core/cognition/`

**"I want to understand startup/wiring"**
→ `main.py` → `core/boot/config_loader.py` → `core/boot/wiring.py` → `core/boot/services.py`

**"I want to understand autonomous behavior"**
→ `core/autonomy_engine.py` → `core/behavior_tracker.py` → `core/cognitive/prediction_engine.py`

---

## 12. File Count Summary

| Directory | Python files | Lines (approx) |
|-----------|-------------|----------------|
| `core/` | 87 | ~25,000 |
| `voice/` | 12 | ~4,800 |
| `brain/` | 19 | ~4,500 |
| `services/` | 9 | ~2,500 |
| `ui/` | 3 | ~1,700 |
| `cursor_bridge/` | 3 | ~1,050 |
| `context/` | 4 | ~600 |
| `tests/` | 9 | ~4,500 |
| `scripts/` | 5 | ~1,200 |
| `tools/` | 1 | ~300 |
| Root `.py` | 6 | ~5,400 |
| **Total** | **158** | **~51,400** |
