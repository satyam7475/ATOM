# ATOM Architecture Map

Subsystem → file → responsibility. Read this when you need to find the owner of a symptom. Kept deliberately wide (many files, short descriptions) so Grep on any keyword lands in the right place.

## Top-level entry

| File | Responsibility |
|---|---|
| `main.py` | Boot sequence: config → security → state → memory → router → voice → watchdog → scanner → cognitive layers → UI. The spine. |
| `atom_cli.py` | CLI entry for non-voice usage + smoke tests. |
| `Run ATOM.command` | macOS bundle launcher (calls `scripts/atom_app_bundle_launcher.sh`). |
| `config/settings.json` | **Single source of truth** for every tunable. |
| `config/commands.json` | Hotword command registry. |
| `config/skills.json` | Skill triggers + chains. |

## Brain / LLM

| File | Responsibility |
|---|---|
| `brain/mlx_llm.py` | MLX inference, CoT/prompt-leak stripping at the LLM-output layer, special-token removal. |
| `brain/mini_llm.py` | Tiny local fallback model (currently unused in happy path, kept as emergency). |
| `brain/memory_graph.py` | `v22_confidence` scoring, short-term memory. |
| `cursor_bridge/local_brain_controller.py` | Streaming LLM controller, response sanitization, guardrails, strict recovery, prompt-leak fingerprinting. |
| `cursor_bridge/structured_prompt_builder.py` | System + context + query prompt assembly. System profile injection. Anti-CoT rules. |

## Router / Intent / Cognitive Kernel

| File | Responsibility |
|---|---|
| `core/router/router.py` | Query intake, intent classification, action dispatch, TTS echo guard, guardrail rewrite, bare-wake ack. |
| `core/cognitive_kernel.py` | Tier / path / model / mode / RAG-budget selection per query. |
| `core/llm_inference_queue.py` | Single-slot LLM queue with coalescing; plumbs `repeat_hint`. |
| `core/intent_engine/__init__.py` | Aggregator, fast-path warm-up. |
| `core/intent_engine/os_intents.py` | Time, weather, presence-check, boss-opener, greeting, clarifiers. |
| `core/intent_engine/system_intents.py` | `find_process_by_name`, `kill_process`, `get_open_ports`, `set_process_priority`, system-control actions. |
| `core/intent_engine/desktop_intents.py` | Open app, close window, focus app, switch workspace, `click_ui_element`. |
| `core/intent_engine/info_intents.py` | Definitions, knowledge lookups, trivia. |

## Voice Pipeline

| File | Responsibility |
|---|---|
| `voice/voice_pipeline.py` | Orchestrates STT + TTS + listening modes. Defers STT init until boot TTS done. |
| `voice/stt_macos.py` | Native STT (SFSpeechRecognizer), partial/final gating by state, echo detection, trivial-final guard. |
| `voice/tts_macos.py` | Native TTS (NSSpeechSynthesizer), streaming slice TTS, echo ring buffer, prompt-leak fingerprint. |
| `voice/listening_modes.py` | Wake-word filter, direct-address phrases, always-on bypass. |
| `voice/interrupt_handler.py` | Barge-in detection, partial gating via echo guard. |
| `voice/wiring.py` | Event bus wiring for voice subsystems. |
| `voice/whisper_confirmer.py` | Optional Whisper re-transcription layer (disabled by default). |
| `voice/mic_manager.py` | Device selection, sample rate negotiation, audio path health. |
| `voice/audio_preprocessor.py` | VAD, noise gate, RMS. |
| `voice/ack_engine.py` | "On it.", "Working on it.", "Here, Boss." micro-acks. |
| `voice/earcons.py` | Sound effects. Currently heartbeat disabled. |
| `voice/emotion_detector.py` | Speech-level emotion/urgency signal for adaptive TTS rate. |
| `voice/media_watcher.py` | Pause TTS during external media playback. |

## Security / Safety

| File | Responsibility |
|---|---|
| `core/security_policy.py` | `allow_action()` gate, tier enforcement, rate limit, lock. |
| `core/security_tiers.py` | Action → tier mapping and escalation prompts. |
| `core/security_fortress.py` | Defense-in-depth (env scrub, FS deny, runtime hardening). |
| `core/security_secret_scrub.py` | Clears sensitive env vars after boot. |
| `core/owner_gate.py` | Owner binding ("Satyam"), unknown-voice lockout. |
| `core/action_safety.py` | Destructive-action preflight. |

## Memory / State / Context

| File | Responsibility |
|---|---|
| `core/state_manager.py` | State machine (idle → listening → thinking → speaking → error_recovery). |
| `core/async_event_bus.py` | Pub/sub spine. |
| `core/conversation/conversation_memory.py` | Per-session dialog buffer. |
| `core/memory/*.py` | Long-term memory (SQLite + vectors). |
| `core/embedding/*.py` | Embedding model, warm-file, batch writer. |
| `core/rag/*.py` | Retrieval, semantic cache, RAG budget. |
| `core/context_layer.py` | Session / user / state context assembly for prompt. |
| `core/system_profile.py` | Persistent system snapshot (`[MACHINE]` line in prompt). |

## Boot / Runtime Control

| File | Responsibility |
|---|---|
| `core/boot/cold_start.py` | Cold-start snapshot, regex priming, boot-TTS greeting. |
| `core/runtime_watchdog.py` | Budget enforcement (intent 50ms, LLM 20s, TTS 15s), auto-demote on repeated breach, TTS stop on timeout. |
| `core/silicon_governor.py` | Apple Silicon thermal + GPU coordination. |
| `core/apple_silicon_monitor.py` | M-series performance counters. |
| `core/power_governor.py` | Plugged-in vs battery adaptive mode. |
| `core/health_watchdog.py` | Periodic health scan (120 s). |
| `core/system_watcher.py` | File/app watcher (30 s). |
| `core/inference_guard.py` | Unified Memory mode guard. |
| `core/gpu_stall_watchdog.py` | Metal/GPU stall recovery. |

## Scheduling / Autonomy / Cognitive

| File | Responsibility |
|---|---|
| `core/scheduler/task_scheduler.py` | Cron-like task scheduling. |
| `core/autonomy_engine.py` | Proactive action scoring (auto ≥ 0.95, suggest ≥ 0.72). |
| `core/goal_engine.py` | User goals evaluation (hourly). |
| `core/prediction_engine.py` | 5-minute prediction cycles. |
| `core/self_optimizer.py` | 2-hour tuning pass. |
| `core/modes/personality_modes.py` | Work / focus / bedtime / meeting modes. |
| `core/background/background_task_manager.py` | Low-priority background jobs. |
| `core/cognitive/*.py` | Dream + curiosity engines. |
| `core/understanding/owner_understanding.py` | Owner model (88+ interactions). |
| `core/proactive/routine_engine.py` | Routines (bedtime, deep_work, meeting). |
| `core/proactive/proactive_engine.py` | 15-minute proactive scan. |
| `core/jarvis_core.py` | JARVIS v21 proactive interval + fusion. |
| `core/real_world/real_world_intel.py` | Weather, news, briefing, world clock. |

## System Control

| File | Responsibility |
|---|---|
| `core/system_control/system_scanner.py` | macOS introspection (procs, network, disks, apps). |
| `core/system_control/system_indexer.py` | Searchable index of system state. |
| `core/system_control/process_control.py` | Kill, prioritize, inspect processes. |
| `core/system_control/network_control.py` | Wi-Fi, ports, DNS. |
| `core/system_control/ui_control.py` | AppleScript, accessibility API, click UI element. |
| `core/system_control/siri_shortcuts.py` | Bridge to Siri Shortcuts (permissions-gated). |

## UI / Telemetry

| File | Responsibility |
|---|---|
| `ui/dashboard.py` | `http://127.0.0.1:8765/` web dashboard. |
| `core/obs/error_rate_monitor.py` | Rolling error-rate alarm (5/60 s). |
| `core/obs/pipeline_budget.py` | Per-turn pipeline latency accounting. |
| `core/timeline/timeline.py` | v7 timeline events. |
| `core/observability/*.py` | Metrics & logs plumbing. |

## Tests / Evaluation

| File | Responsibility |
|---|---|
| `tests/test_atom_smoke.py` | 60-second smoke (boot + one turn). |
| `tests/test_voice_pipeline_critical.py` | Voice pipeline end-to-end. |
| `tests/test_system_control_v1.py` | 52-test system-control regression suite. |
| `tests/test_prompt_leak_fix.py` | Prompt-leak regression. |
| `tests/test_echo_loop_fix.py` | Self-echo regression. |
| `tests/test_jarvis_stream_sanitizer.py` | CoT / preface / token stripping. |
| `tests/test_optimization_stability_security.py` | Perf + security gates. |
| `tests/jarvis_eval.py` | Scorecard harness (ATOM vs Jarvis axes). |
| `tests/jarvis_e2e_benchmark.py` | End-to-end latency benchmark. |

## Utility Scripts (top-level)

| File | Responsibility |
|---|---|
| `scripts/atom_app_bundle_launcher.sh` | Launch via `.app` bundle (captures mic/speech permissions). |
| `scripts/install_atom_launchagent.sh` | LaunchAgent KeepAlive install. |
| `scripts/uninstall_atom_launchagent.sh` | Remove LaunchAgent. |
| `scripts/enroll_owner_face.py` | Face enrollment for tier-4 unlock. |
| `scripts/setup_api_keys.py` | Cloud key provisioning (opt-in). |
| `scripts/v7_chaos_test.py` | Chaos / fault injection. |
| `scripts/v7_long_run.py` | 24-hour stability soak. |
| `scripts/m5_baseline_benchmark.py` | M5 perf baseline (Metal + ANE). |

## Data Layout

```
data/
  atom_memory.db            # Long-term memory (SQLite)
  semantic_cache.db         # Semantic response cache
  rag_embedding_cache.sqlite  # Embedding cache
  vector_db/                # ChromaDB collections (4 collections)
  embeddings_warm.npz       # Warm-file for embeddings (cold-start speedup)
  behavior_profile.json     # Adaptive behavior state
  user_profile.json         # Owner profile
  real_world_cache.json     # Weather/news cache
  system_profile.json       # System snapshot for [MACHINE] line
  security/                 # Biometric enrollments, passphrase hashes
```

## How to use this map

1. **Symptom hunt.** Grep the log for unique keywords (module names like `atom.stt_macos`, action names like `MLX`, error codes like `kLSRErrorDomain 301`). They map directly to files above.
2. **Dependency trace.** When touching a hot-path module, scan downstream subscribers (voice/wiring.py, core/router/router.py attach-style wiring).
3. **Config-first.** Before changing code, check if the behavior can be tuned in `config/settings.json`. If yes, pref that.
4. **New module placement.** Match the naming conventions above (`core/<domain>/<module>.py`, `voice/<name>.py`, `brain/<name>.py`).
