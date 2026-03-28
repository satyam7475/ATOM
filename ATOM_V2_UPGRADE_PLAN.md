# ATOM V2 -- Cognitive OS Upgrade Plan

> **Author:** ATOM Architecture Review  
> **For:** Satyam (Owner & Creator)  
> **Purpose:** Transform ATOM from a voice-controlled assistant into a real Cognitive Operating System that genuinely thinks, reasons, learns, and acts autonomously -- at JARVIS/FRIDAY level.  
> **Constraint:** Runs on a personal system with 9B or 13B parameter local LLM on GPU. Owner-exclusive. Fully offline-capable.

---

## Current State Assessment (Post v18)

### What ATOM v18 Has (Operational)

| Component | Status | Notes |
|---|---|---|
| 7-Ring Architecture | Excellent | Best-in-class modular design |
| AsyncEventBus backbone | Excellent | Zero-coupling nervous system |
| Interface Contracts | Excellent | Enables modular upgrades |
| SecurityPolicy (7 gates) | Strong | 7-layer enforcement |
| StateManager (6 states) | Solid | SLEEP/IDLE/LISTENING/THINKING/SPEAKING/ERROR_RECOVERY |
| IntentEngine (12 modules, <5ms) | Solid | Fast-path for obvious commands |
| GPU LLM Brain (9B/13B) | v17 | Full GPU offload, 1-4s inference |
| True Token Streaming | v17 | First sentence in <500ms |
| KV Cache Persistence | v17 | ~200-400ms savings per query |
| Expanded Context (8192-16384) | v17 | 20-turn history, context budget |
| **Agentic Tool Use (40+ tools)** | **NEW v18** | **LLM dynamically calls tools, replaces hardcoded dispatch for ambiguous queries** |
| **ReAct Reasoning Loop** | **NEW v18** | **LLM acts → observes results → reasons again (up to 3 steps)** |
| **7-Layer Prompt Architecture** | **NEW v18** | **Added observations layer, auto-generated tools from ToolRegistry** |
| **ActionExecutor (security-gated)** | **NEW v18** | **Validates, security-checks, and dispatches tool calls from LLM** |
| **Multi-Step Planner** | **NEW v18** | **Template plans + LLM-generated plans for complex requests** |
| **Code Sandbox** | **NEW v18** | **Safe Python eval for math/calculations** |
| Router 3-layer intelligence | Upgraded | Intent fast-path → Cache/Memory → LLM Reasoning Agent |
| CPU Governor + HealthMonitor | Good | Adapted for GPU monitoring |
| BehaviorTracker + AutonomyEngine | Good | Habit detection + auto-execution |
| Cognitive Layer (5 modules) | Good | Goals, predictions, behavior, optimizer, modes |

### What Was Removed in v17

| Removed | Reason |
|---|---|
| 1B model (`Llama-3.2-1B-Instruct`) | Single 9B/13B model handles everything better |
| Dual-model routing | Complexity heuristic unnecessary with powerful single model |
| `_is_complex_query` heuristic | No longer needed — one model for all |
| Fake streaming (sentence-split post-gen) | Replaced by true token-by-token streaming |
| `truststore` SSL injection | Fully offline — no cloud dependencies |
| Kill switch | Ready for deployment on target machine |
| Cloud model references | All messaging updated for offline-only |

### What Must Still Change

| Problem | Current State | Why It Blocks JARVIS-Level |
|---|---|---|
| **Memory is primitive** | Keyword-overlap JSON, no semantic search | Cannot recall by meaning, cannot learn deeply |
| ~~**No real reasoning**~~ | ✅ SOLVED v18 — ReAct loop, multi-step planning | ~~Cannot solve multi-step problems~~ |
| ~~**No tool use**~~ | ✅ SOLVED v18 — LLM calls 40+ tools dynamically | ~~LLM cannot dynamically choose actions~~ |
| **No document knowledge** | Zero file/document ingestion | Cannot learn from PDFs, docs, notes |
| **No wake word** | Always-on STT burns CPU constantly | Unnatural interaction, wastes resources |
| **Context window not semantic** | Large but still keyword-based memory | Forgets meaning, only recalls exact matches |
| **No owner verification** | Config-based only | Anyone who runs it can use it |

---

## Hardware Requirements (Personal System)

| Component | Minimum Spec | Recommended Spec | Role |
|---|---|---|---|
| **GPU** | RTX 3060 12GB VRAM | RTX 4070 Ti 12GB+ / RTX 3090 24GB | LLM inference, STT acceleration, TTS |
| **CPU** | Ryzen 5 5600 / i5-12400 | Ryzen 7 7700X / i7-13700 | STT Whisper, async tasks, system monitoring |
| **RAM** | 16GB DDR4 | 32GB DDR5 | Model loading, vector DB, OS overhead |
| **Storage** | 512GB NVMe SSD | 1TB NVMe SSD | Models (~15GB), vector DB, audio cache, documents |
| **Microphone** | Any USB mic | Blue Yeti / HyperX QuadCast | Voice quality directly affects STT accuracy |

### Model Recommendations

| Use Case | Model | Size (GGUF Q4_K_M) | VRAM | Quality |
|---|---|---|---|---|
| **Primary (8B)** | Qwen3-8B-Q4_K_M | ~5.0GB | ~6GB | Native tool calling, dual thinking modes, 32K context |
| **Primary (13B)** | Mistral-Nemo-12B-Instruct | ~7GB | ~9GB | Superior reasoning, long context (128K native) |
| **Alternative (9B)** | Gemma-2-9B-it | ~5.5GB | ~7GB | Google quality, strong instruction following |
| **Alternative (13B)** | Llama-3.1-8B-Instruct | ~4.5GB | ~6GB | Meta's best, strong general reasoning |
| **Embedding** | nomic-embed-text-v1.5 (GGUF) | ~260MB | CPU | Semantic search, runs on CPU, no GPU needed |
| **Wake Word** | OpenWakeWord | ~5MB | CPU | Custom "Hey ATOM" detection |

**Why Qwen3-8B:** Native tool calling with dual thinking modes (/think for complex reasoning, /no_think for fast replies). 32K context window, GGUF quantized to ~5GB. Best-in-class for agentic AI at the 8B parameter scale.

**Why Mistral-Nemo-12B for 13B slot:** 128K native context window (no rope scaling needed), excellent reasoning, strong instruction following, and good speed on GPU.

---

## Upgrade Phases (Ordered by Dependency)

---

### ✅ PHASE 1: Brain Transplant — COMPLETED

**Status:** DONE (v17 deployed)  
**Goal:** Replace the 1B+3B CPU dual-model system with a single 9B/13B GPU-accelerated model.

#### What Was Built

| Sub-task | Status | Key Changes |
|---|---|---|
| 1.1 GPU-Accelerated Inference | ✅ Done | `n_gpu_layers=-1`, `n_batch=512`, `n_ctx=8192`, single model, removed all 1B code |
| 1.2 True Token Streaming | ✅ Done | Sentence buffer accumulates tokens, emits `partial_response` at sentence boundaries, TTS speaks while LLM generates |
| 1.3 KV Cache Persistence | ✅ Done | `save_kv_cache()` / `restore_kv_cache()` for system prompt, invalidates on model/prompt change |
| 1.4 Expanded Context Window | ✅ Done | `n_ctx=8192` (brain mode: 16384), `_conv_window_max=20`, `max_turns=20`, context budget system |
| 1.5 Prompt Architecture Rebuild | ✅ Done | 6-layer architecture: System Identity → Tools → Dynamic Context → Memory → History → Query |

#### Files Modified

| File | Changes |
|---|---|
| `brain/mini_llm.py` | Complete rewrite — single model, GPU offload, streaming callback, KV cache, stop tokens for Qwen |
| `cursor_bridge/local_brain_controller.py` | Complete rewrite — true streaming with sentence buffer, first-token latency tracking |
| `cursor_bridge/structured_prompt_builder.py` | Complete rewrite — 6-layer prompt, context budget, tool definitions, system prompt hash |
| `config/settings.json` | New brain config (GPU, 8192 ctx, 512 tokens), updated profiles, removed 1B model |
| `core/router/router.py` | `_conv_window_max` 5→20, updated docstrings |
| `core/brain_mode_manager.py` | Updated defaults (GPU layers, batch, top_p, repeat_penalty) |
| `main.py` | Removed kill switch, removed truststore, updated version to v17, updated greeting |

#### Expected Performance Gains

| Metric | Before (v15) | After (v17) | Improvement |
|---|---|---|---|
| LLM inference | 5-25s (CPU) | 1-4s (GPU) | **5-8x faster** |
| First audio response | 5-25s (wait for full gen) | 300-500ms (first sentence) | **10-50x faster perceived** |
| Context window | 2048 tokens (5 turns) | 8192-16384 tokens (20 turns) | **4-8x more context** |
| Max response length | 80-100 tokens | 512-768 tokens | **5-8x richer answers** |
| System prompt re-processing | ~400ms every query | ~0ms (cached) | **Eliminated** |
| Model quality | 1B/3B (limited reasoning) | 9B/13B (strong reasoning + tool-use) | **Qualitative leap** |

---

### PHASE 2: Memory Revolution

**Status:** NOT STARTED  
**Goal:** Replace keyword-overlap JSON memory with a semantic vector memory system. This transforms ATOM from "forgetting everything" to "remembering everything by meaning."

**Files to modify:** `core/memory_engine.py`, `core/cognitive/second_brain.py`, `core/cache_engine.py`, new file `core/vector_store.py`

#### 2.1 Local Embedding Engine

**New file:** `core/embedding_engine.py`

Create a lightweight embedding engine that runs on CPU (no GPU needed):
- Use `sentence-transformers` with `nomic-embed-text-v1.5` or `all-MiniLM-L6-v2`
- Embed text into 384-dimensional vectors in <10ms per query
- Batch embedding for bulk operations (document ingestion)
- Singleton pattern -- one model instance shared across all modules
- Lazy loading -- don't load until first use
- Interface:
  ```
  class EmbeddingEngine:
      async def embed(text: str) -> list[float]
      async def embed_batch(texts: list[str]) -> list[list[float]]
      def embed_sync(text: str) -> list[float]
  ```

#### 2.2 Vector Store

**New file:** `core/vector_store.py`

Build a vector store on top of ChromaDB (pure Python, serverless, persistent to disk):
- Collections: `conversations`, `facts`, `documents`, `interactions`
- Operations: `add`, `search`, `search_temporal`, `delete_collection`, `get_stats`
- Storage: `data/vector_db/` directory (persistent across restarts)

#### 2.3 Memory Engine Rebuild

**Modify:** `core/memory_engine.py`

Replace keyword-overlap with vector similarity search:
- Embed query+response → store in vector store
- Retrieve by semantic similarity, not keyword overlap
- Migrate existing `logs/memory.json` on first startup

#### 2.4 SecondBrain Migration

**Modify:** `core/cognitive/second_brain.py`

- Replace JSON storage with vector store `facts` collection
- Semantic retrieval instead of keyword matching

#### 2.5 Document Ingestion Pipeline

**New file:** `core/document_ingestion.py`

- Support: `.txt`, `.md`, `.pdf`, `.docx`, `.py`, `.json`, `.csv`
- Chunking: ~500 token chunks with 50 token overlap
- Voice commands: "learn this document", "what does [file] say about [topic]?"

#### 2.6 Cache Engine Enhancement

**Modify:** `core/cache_engine.py`

- Add vector similarity as secondary lookup (cosine > 0.92 = cache hit)
- Fuzzy memory for paraphrased questions

---

### ✅ PHASE 3: Reasoning Engine (Tool Use + Planning) — COMPLETED

**Status:** DONE (v18 deployed)  
**Goal:** Transform ATOM from "match intent → execute action" to "LLM reasons about what to do and calls tools dynamically."

#### What Was Built

| Sub-task | Status | Key Changes |
|---|---|---|
| 3.1 Tool Registry System | ✅ Done | `core/reasoning/tool_registry.py` — 40+ tools registered with name, description, parameters, safety levels, categories |
| 3.2 Tool-Use Response Parser | ✅ Done | `core/reasoning/tool_parser.py` — Parses 4 formats: ATOM native JSON, Qwen3 tool_call, simple `<tool>`, bare JSON |
| 3.3 Action Executor | ✅ Done | `core/reasoning/action_executor.py` — Security-gated bridge: ToolRegistry validation → param validation → SecurityPolicy → confirmation gate → dispatch |
| 3.4 Router Architecture Rebuild | ✅ Done | IntentEngine kept as <5ms fast-path, LLM is now the reasoning engine with full tool access |
| 3.5 Agentic Brain Controller | ✅ Done | `local_brain_controller.py` rewritten with ReAct loop: LLM → tool calls → execute → observe → re-reason (up to 3 steps) |
| 3.6 Agentic Prompt Architecture | ✅ Done | `structured_prompt_builder.py` upgraded to 7-layer: added observations layer, auto-generated tools from ToolRegistry |
| 3.7 Multi-Step Planning | ✅ Done | `core/reasoning/planner.py` — Template plans + LLM-generated plans, step tracking, failure recovery |
| 3.8 Code Execution Sandbox | ✅ Done | `core/reasoning/code_sandbox.py` — Safe Python eval, restricted builtins, 5s timeout, human math preprocessing |

#### Files Modified/Created

| File | Changes |
|---|---|
| `core/reasoning/tool_registry.py` | **NEW** — 40+ tool definitions with parameters, safety levels, categories, function schemas |
| `core/reasoning/tool_parser.py` | **NEW** — Multi-format tool call parser (4 formats), clean text extraction |
| `core/reasoning/action_executor.py` | **NEW** — Security-gated execution bridge with parameter validation and alt-key resolution |
| `core/reasoning/planner.py` | **NEW** — Multi-step planning with templates, step tracking, failure recovery |
| `core/reasoning/code_sandbox.py` | **NEW** — Sandboxed Python execution with restricted builtins |
| `core/reasoning/__init__.py` | **NEW** — Module exports |
| `cursor_bridge/local_brain_controller.py` | **Rewritten** — Agentic ReAct loop: LLM generates → parse tool calls → execute → feed observations → re-reason |
| `cursor_bridge/structured_prompt_builder.py` | **Upgraded** — 7-layer architecture, auto-generated tools from ToolRegistry, observations layer for ReAct |
| `core/router/router.py` | **Upgraded** — ActionExecutor integration, tool confirmation handling, agentic docstrings |
| `main.py` | **Updated** — ActionExecutor wiring, pending_tool_confirmation event handler |

#### Architecture: The Agentic Pipeline

```
User speaks → STT → Router._route()
  │
  ├─► IntentEngine.classify() (<5ms regex fast-path)
  │   ├─ High confidence match? → Execute instantly via dispatch table
  │   └─ "open chrome", "what time is it", "set volume 50" → <5ms
  │
  └─► Fallback / Ambiguous → LLM Reasoning Agent
      │
      ├─ LLM sees: System Identity + 40+ Tool Definitions + Context + Memory + History
      │
      ├─ LLM responds with:
      │   (A) Text only → TTS speaks it
      │   (B) Tool call(s) → ActionExecutor validates + executes
      │   (C) Text + Tool call(s) → TTS speaks text, tools execute
      │
      └─ ReAct Loop (if tools called):
          Step 1: LLM outputs tool_call → ActionExecutor executes → result
          Step 2: Result fed back as [OBSERVATION] → LLM reasons again
          Step 3: LLM outputs final response or more tool calls
          (max 3 ReAct steps per query)
```

#### Why This Matters (99% accuracy target)

| Before (v17) | After (v18) | Impact |
|---|---|---|
| Regex matching only | LLM understands natural language intent | "I need the internet" → open_app(chrome) |
| Hardcoded dispatch table | LLM dynamically selects from 40+ tools | Handles any phrasing |
| Single action per query | Multi-step ReAct reasoning | "Set up my workspace" → 4 actions |
| No feedback loop | Observes tool results, adapts | Self-corrects on failure |
| Falls back to text-only | Falls back to text + explains capabilities | Always useful |

---

### PHASE 4: Perception Upgrade

**Status:** NOT STARTED  
**Goal:** Make ATOM truly aware of its environment.

#### 4.1 Wake Word Engine (`voice/wake_word.py`)
- OpenWakeWord, custom "Hey ATOM" trigger, <1% CPU
- Hotkey (Ctrl+Alt+A) bypasses wake word

#### 4.2 GPU-Accelerated STT
- Switch to `faster-whisper` with GPU backend
- Use `whisper-small.en` for better accuracy
- VAD filter for efficiency

#### 4.3 Emotion Detection from Voice (`voice/emotion_detector.py`)
- Pitch, speed, volume analysis
- Classify: neutral, stressed, excited, tired, frustrated, happy

#### 4.4 Screen Understanding (`context/screen_reader.py`)
- Local OCR (Tesseract/EasyOCR) on screenshots
- "What's on my screen?" support

#### 4.5 Ambient Audio Classification (`voice/ambient_classifier.py`)
- Classify: silence, typing, talking, music, noise
- Adjust STT sensitivity and TTS volume based on environment

---

### PHASE 5: Owner Security & Exclusivity

**Status:** NOT STARTED  
**Goal:** Make ATOM accessible only to Satyam via biometric verification.

#### 5.1 Voice Print Authentication (`core/auth/voice_auth.py`)
- Enroll 10 phrases, create voice fingerprint
- Cosine similarity > 0.75 = verified owner

#### 5.2 Behavioral Authentication (`core/auth/behavior_auth.py`)
- Passive verification from usage patterns
- Anomaly detection triggers re-verification

#### 5.3 Encrypted Storage
- AES-256-GCM for all sensitive data at rest
- Key in Windows Credential Manager

#### 5.4 Session Lock
- Auto-lock after 30 min inactivity
- Voice verification to unlock

---

### PHASE 6: Speed & Performance Engineering

**Status:** NOT STARTED  
**Goal:** <200ms perceived latency for commands, <1s first-audio for LLM queries.

#### 6.1 Speculative Pipeline
- Begin processing on partial speech (before user finishes talking)
- If high-confidence intent from partial → prepare action in advance

#### 6.2 Response Pre-computation
- Pre-compute predictable responses (time, greetings, system status)

#### 6.3 TTS Pipeline Optimization
- Pre-generate common acknowledgments at startup
- Cache top 20 TTS outputs

#### 6.4 GPU Resource Management (`core/gpu_governor.py`)
- Monitor GPU via pynvml
- Priority: LLM > STT > background tasks

---

### PHASE 7: Autonomy & Intelligence Deepening

**Status:** NOT STARTED  
**Goal:** Make ATOM genuinely proactive and intelligent.

#### 7.1 Workflow Recording & Replay (`core/reasoning/workflow_engine.py`)
- "Watch what I do" → record → name → replay

#### 7.2 Strengthened Prediction Engine
- Sequence-based prediction instead of frequency counting

#### 7.3 Proactive Intelligence
- "You've been coding for 3 hours. Want a break timer?"
- "Meeting in 10 minutes. Open Teams?"

#### 7.4 Dream Mode (`core/cognitive/dream_engine.py`)
- Offline memory consolidation during idle periods

---

### PHASE 8: Expression & Personality

**Status:** NOT STARTED  
**Goal:** Give ATOM a unique, recognizable identity.

#### 8.1 Custom Voice (Piper TTS or XTTS v2)
#### 8.2 Advanced Personality Engine (evolving style)
#### 8.3 Dashboard Enhancement (predictions, goals, GPU, real-time tool chains)

---

### PHASE 9: Integration Layer

**Status:** NOT STARTED  
**Goal:** Connect ATOM to external services and data sources.

#### 9.1 Calendar Integration
#### 9.2 Email Integration
#### 9.3 File System Intelligence
#### 9.4 Notification Hub

---

## Full Offline Pipeline (v18 — Agentic)

The complete pipeline from voice input to voice output, fully offline, with agentic tool use:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  VOICE INPUT PIPELINE                                                        │
│                                                                              │
│  Microphone → PyAudio → faster-whisper (STT, CPU/GPU)                       │
│  → speech_final event on AsyncEventBus                                       │
│  → Noise filter + text corrections + filler word removal                    │
└─────────────────┬───────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  UNDERSTANDING + ROUTING                                                     │
│                                                                              │
│  Router._route(text)                                                         │
│    1. Security sanitize (injection protection)                              │
│    2. Skill expansion (multi-step phrase chains)                             │
│    3. Pronoun resolution (conversational continuity)                         │
│    4. Clipboard injection (implicit context)                                │
│    5. Intent Engine classify (<5ms, 12 regex modules, FAST PATH)            │
│                                                                              │
│  ┌──────────────────┐  ┌──────────────┐  ┌────────────────────────────────┐│
│  │ 70%: INTENT MATCH │  │ Cache/Memory │  │ 30%: LLM REASONING AGENT      ││
│  │ → Direct action   │  │ → Instant    │  │ → Agentic 7-layer prompt      ││
│  │ → Security gate   │  │   response   │  │ → Tool calls + text response  ││
│  │ → Execute + TTS   │  │              │  │ → ReAct loop (act→observe→    ││
│  │   (<5ms)          │  │              │  │   reason again, up to 3 steps)││
│  └──────────────────┘  └──────────────┘  └────────────────────────────────┘│
└─────────────────┬───────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  AGENTIC LLM BRAIN (v18)                                                     │
│                                                                              │
│  StructuredPromptBuilder assembles 7-layer agentic prompt:                   │
│    Layer 1: System Identity + Agentic Instructions — CACHED                 │
│    Layer 2: 40+ Tool Definitions (auto-generated from ToolRegistry) — CACHED│
│    Layer 3: Dynamic Context (time, app, clipboard, topics)                  │
│    Layer 4: Memory Context (past conversations, RAG results)                │
│    Layer 5: Conversation History (up to 10 turns, budget-trimmed)           │
│    Layer 6: ReAct Observations (tool results from current turn)             │
│    Layer 7: Current User Query                                              │
│                                                                              │
│  MiniLLM (llama.cpp, GPU):                                                   │
│    - Model: Qwen3-8B-Q4_K_M (native tool calling, dual thinking modes)     │
│    - Full GPU offload, 8192-16384 token context                             │
│    - KV cache for system prompt, 1-4s inference                             │
│    - TRUE STREAMING: tokens → sentence buffer → TTS in real-time           │
│                                                                              │
│  LocalBrainController (Agentic ReAct Loop):                                  │
│    1. LLM generates response with optional <tool_call> tags                 │
│    2. ToolParser extracts tool calls (4 format support)                     │
│    3. ActionExecutor validates → security check → dispatch → result         │
│    4. Result fed back to LLM as [OBSERVATION]                               │
│    5. LLM reasons again: more tools, or final text response                │
│    6. Repeat up to 3 ReAct steps per query                                  │
│                                                                              │
│  ActionExecutor (security-gated):                                            │
│    ToolCall → Registry validation → Param validation → SecurityPolicy       │
│    → Confirmation gate (dangerous actions) → Router dispatch → Result       │
└─────────────────┬───────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  VOICE OUTPUT PIPELINE                                                       │
│                                                                              │
│  TTS receives partial_response events:                                       │
│    - Windows SAPI (offline, instant start, ~50ms)                           │
│    - OR Edge Neural TTS (online option, higher quality)                     │
│    - Speaks each sentence as it arrives (true streaming)                    │
│    - Barge-in support: user can interrupt at any time                       │
│                                                                              │
│  State: THINKING → SPEAKING → LISTENING (always-listen cycle)               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Execution Order Summary

| Order | Phase | Key Deliverables | Dependencies | Status |
|---|---|---|---|---|
| **1** | ✅ Phase 1: Brain Transplant | GPU inference, true streaming, expanded context | Hardware with GPU | **COMPLETED** |
| **2** | Phase 2: Memory Revolution | Vector store, RAG pipeline, document ingestion | Phase 1 | NOT STARTED |
| **3** | ✅ Phase 3: Reasoning Engine | Agentic tool use, ReAct loop, planner, sandbox | Phase 1 | **COMPLETED** |
| **4** | Phase 5: Owner Security | Voice auth, encrypted storage, session lock | Independent | NOT STARTED |
| **5** | Phase 4: Perception Upgrade | Wake word, emotion detection, screen reading | Phase 1 | NOT STARTED |
| **6** | Phase 6: Speed Engineering | Speculative pipeline, pre-computation, GPU governor | Phase 1-3 | NOT STARTED |
| **7** | Phase 7: Autonomy Deepening | Workflow recording, proactive intelligence, dream mode | Phase 2 + 3 | NOT STARTED |
| **8** | Phase 8: Expression | Custom voice, personality evolution, dashboard upgrade | Phase 1 | NOT STARTED |
| **9** | Phase 9: Integration | Calendar, email, file intelligence | Phase 2 + 3 | NOT STARTED |

**Total estimated time: ~20-25 days of focused development (Phase 1 + 3 complete saves ~7-8 days)**

---

## New Dependencies to Add

```
# requirements-v2.txt (add to existing requirements.txt)

# GPU LLM Inference (ALREADY USING — rebuild with CUDA)
llama-cpp-python          # Rebuild with: CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# Vector Memory (Phase 2)
chromadb>=0.4.0           # Serverless vector database
sentence-transformers     # Embedding models

# Wake Word (Phase 4)
openwakeword              # Custom wake word detection

# Voice Authentication (Phase 5)
resemblyzer               # Voice embedding for speaker verification

# Screen Understanding (Phase 4)
easyocr                   # Local OCR for screen reading

# Document Ingestion (Phase 2)
pymupdf                   # PDF text extraction (fitz)
python-docx               # DOCX text extraction

# GPU Monitoring (Phase 6)
pynvml                    # NVIDIA GPU monitoring

# Local TTS (Phase 8, choose one)
piper-tts                 # Fast local TTS (CPU)
# OR
TTS                       # Coqui XTTS v2 (GPU, voice cloning)

# Enhanced STT (Phase 4)
faster-whisper            # GPU-accelerated Whisper (already in stack)

# Encryption (Phase 5)
cryptography              # AES-256-GCM for encrypted storage

# Calendar/Email (Phase 9)
google-api-python-client  # Google Calendar
imapclient                # Email reading
```

---

## Projected JARVIS Rating

| Dimension | v15 (Before) | v17 (Now) | After Full v2 | JARVIS Standard |
|---|---|---|---|---|
| **Conversational Intelligence** | 5/10 | 7/10 | 8.5/10 | 10/10 |
| **Response Speed** | 5/10 | 8/10 | 9/10 | 10/10 |
| **Knowledge & Memory** | 4/10 | 5/10 | 8.5/10 | 10/10 |
| **System Control** | 6/10 | 6.5/10 | 8/10 | 9/10 |
| **Autonomy** | 5/10 | 5/10 | 8/10 | 9/10 |
| **Security** | 7/10 | 7/10 | 9.5/10 | 10/10 |
| **Personality** | 5/10 | 6/10 | 8/10 | 9/10 |
| **Proactivity** | 4/10 | 4/10 | 8/10 | 9/10 |
| **Learning** | 4/10 | 4.5/10 | 8/10 | 9/10 |
| **Overall** | **5.5/10** | **6.5/10** | **8.5/10** | **10/10** |

v17 gains come from: GPU speed (8/10 on response), richer prompts (7/10 on conversation), expanded context (better follow-ups), and true streaming (feels much faster).

---

> **Phase 1 is DONE. The brain is transplanted. ATOM v17 is ready for deployment.**  
> **Next: Phase 2 (Memory Revolution) -- this gives ATOM semantic recall and document knowledge.**  
> **Every phase keeps the existing architecture intact -- ATOM's 7-ring design is the foundation, not the problem.**
