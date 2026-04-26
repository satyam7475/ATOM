# ATOM Module 00: System Identity

> Read this before making ANY change to ATOM.

## What ATOM Is

**ATOM** is a Personal Cognitive AI Operating System — not an assistant. It:
- **Perceives** (voice, system state, user behavior, context)
- **Thinks** (intent classification, LLM reasoning, prediction)
- **Acts** (desktop control, app management, system commands)
- **Learns** (behavior tracking, habit formation, self-optimization)
- **Evolves** (self-diagnostics, pattern detection, architecture improvement)

## Core Principles

| Principle | Rule |
|-----------|------|
| **Offline-First** | Zero cloud dependency. Local LLM, local STT, local TTS. |
| **Owner-Centric** | Single owner (Satyam), addressed as "Boss". |
| **Event-Driven** | ALL modules communicate through AsyncEventBus — zero direct coupling. |
| **Security-Gated** | EVERY action passes through SecurityPolicy before execution. |
| **Self-Improving** | SelfEvolutionEngine + SelfOptimizer + BehaviorTracker. |
| **Modular Organs** | Every subsystem can be replaced without touching others. |

## Tech Stack (Sprint Ω.7, 2026-04-26)

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ (async/await) — `dataclass(slots=True)` + PEP 604 unions; pinned via `.python-version` |
| Concurrency | asyncio + ThreadPoolExecutor (CPU/Metal/MLX work runs via `loop.run_in_executor`) |
| STT (primary) | **WhisperKit CoreML on Apple Neural Engine** (`whisperkit-cli serve`, OpenAI-compat `/v1/audio/transcriptions`, model: `whisper-large-v3-v20240930_turbo_632MB`) |
| STT (fallbacks) | `whisper.cpp` Metal (pywhispercpp) → SFSpeechRecognizer (macos_native) |
| TTS | macOS native `NSSpeechSynthesizer` (`voice/tts_macos.py`); Kokoro + Edge are opt-in only |
| LLM | **MLX single-resident Qwen3-8B-4bit** (Apple Silicon, `brain.single_resident=true`, no draft, no whisper-confirmer); Gemini cloud via `cognitive_kernel` Path 2.65 (off by default) |
| VLM | mlx-vlm SmolVLM-Instruct-4bit (lazy-loaded, `vision.vlm.warm_at_boot=false`) |
| Embeddings | SentenceTransformer on torch.mps (Phase B.2) |
| RAG | ChromaDB (local, no network) |
| UI | aiohttp WebSocket + Three.js JARVIS dashboard |
| Monitoring | `core/health_monitor.py` (CPU/RAM/STT-watchdog), `core/runtime_watchdog.py`, `voice/stt_watchdog.py`, `voice/audio_intelligence.py` |
| Persistence | JSON files (`logs/`, `config/`, `data/`) — no DB dependency |

## The 8 Rings (Architecture Layers)

```
Ring 1: PERCEPTION    — STT, Mic, SystemWatcher, ContextEngine
Ring 2: UNDERSTANDING — IntentEngine (12 sub-modules), CommandCache, Skills
Ring 3: DECISION      — Router (3-tier), CacheEngine, MemoryEngine
Ring 4: EXECUTION     — system/app/media/network/file/utility actions
Ring 5: EXPRESSION    — TTS, WebDashboard, Personality, PersonalityModes
Ring 6: COGNITION     — SecondBrain, GoalEngine, PredictionEngine, BehaviorModel
Ring 7: AUTONOMY      — AutonomyEngine, SelfEvolution, HealthMonitor, Security
BACKBONE:             — AsyncEventBus (connects ALL rings)
```

## Entry Point

`main.py` — wires all modules, loads config, registers event handlers, starts background tasks.

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point, all module wiring |
| `config/settings.json` | ALL behavior configuration |
| `config/commands.json` | Action registry (27 commands) |
| `config/skills.json` | Phrase expansion skills |

## Module Count (live as of 2026-04-26)

~155 Python source files in `core/`, `brain/`, `voice/`, `vision/`, `cognitive_kernel/`, `device/`, plus 1929 passing tests in `tests/`.

## Verified Working End-to-End (Sprint Ω.6.B / Ω.7 demo gate)

- Voice: WhisperKit STT → MLX Qwen3-8B-4bit (single-resident, no speculative) → macOS native TTS, with barge-in and audio-device hot-swap.
- Brain: prompt cache persistence, Metal-serial cold start, ~7-8 GB peak RAM on M-series 16 GB.
- Test suite: 1929 passed, 2 skipped, 0 failed on `.venv/bin/python` (3.11.15).
- Disabled-by-default for stability: `cloud.enabled`, `cloud_brain_router.enabled`, `whisper_confirm.enabled`, `speculative.enabled`, `vision.vlm.warm_at_boot`.
