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

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ (async/await) |
| Concurrency | asyncio + ThreadPoolExecutor(3 workers) |
| STT | faster-whisper small (offline, bilingual en+hi, CPU/GPU) |
| TTS | Edge Neural TTS (primary), Windows SAPI (offline fallback) |
| LLM | llama-cpp-python with GGUF models (1B + 3B dual routing) |
| UI | aiohttp WebSocket + Three.js JARVIS dashboard |
| Monitoring | psutil, custom MetricsCollector, HealthMonitor |
| Persistence | JSON files (logs/, config/) — no database dependency |

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

## Module Count

~80 source files, ~45 events, 27+ commands, 200+ intent patterns.
