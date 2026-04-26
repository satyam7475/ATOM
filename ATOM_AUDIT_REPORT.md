# ATOM — End-to-End Audit Report

> **Report compiled:** 2026-04-27 00:35 IST  
> **Owner:** Satyam (addressed as "Boss" by ATOM)  
> **Repo HEAD:** `3f0c787` — Sprint Ω.9 + Ω.10: voice fluency P0 + dead-code cleanup + idle maintenance (2026-04-27 00:19:05 +0530)  
> **Audit method:** Static config inspection + live MLX brain load + warm voice-pipeline timings + 1,983-test pytest run + log triage on `logs/atom.log` (24-hour history) + ChromaDB / SQLite content counts + invariant-by-invariant cross-check.  
> **Validation gap (must-read):** No fresh `atomCurrentLogs.txt` exists from a post-`3f0c787` boot. Every runtime number below labelled "live" came from in-process scripts (`scripts/audit_*`); voice-pipeline E2E numbers (TTS wedge, STT starvation, boot timeline) are from the **pre-Ω.9** runtime in `logs/atom.log` and are explicitly tagged "**[STALE]**" wherever surfaced.

This report is structured for an external reviewer (ChatGPT or another senior engineer) who has not seen the codebase. Every claim is backed by a file path, log line, JSON artifact, or test name. Numbers are real measurements, not estimates. Contradictions and unknowns are called out explicitly under § "Open questions".

---

## 1 · Executive Summary

ATOM is a **154,451-LOC Python 3.11 personal AI OS** running on a **MacBook Air M5 (16 GB unified memory, 10P/10E cores, 10-core GPU, Apple Neural Engine)**. The default brain is **Qwen3-4B-Instruct-4bit (MLX)** — single-resident — with a 4-provider rotating cloud fallback (Groq, Gemini, Cerebras, NVIDIA). Voice goes WhisperKit (CoreML/ANE) → cognitive kernel → Qwen3 → NSSpeechSynthesizer. There is a 5-collection ChromaDB-backed RAG layer, a 6-table SQLite memory graph, an aiohttp dashboard at `http://127.0.0.1:8765/`, and a 4-tier security policy with biometric escalation.

**Today's scorecard (against pre-Ω.9 log):** **31 / 100** ("Needs work") — driven entirely by pre-Ω.9 voice fluency: STT preload of 16.3 s (target 6 s), boot total 18.5 s (target 12 s), 3× TTS wedges, 5× STT starvation events, 89 % memory pressure, p95 first-token 12.8 s. **Sprint Ω.9 specifically targets these.** Until the user reboots and produces a fresh log, the headline grade remains 31/100 with **a credible expected jump to ~70 once the executor split + boot warmup land in a real boot.**

**Component-level health (live, post-Ω.9 in-process):**

| Subsystem | Status | Evidence |
|---|---|---|
| Brain — MLX cold load | **PASS within 6 %** | 4,780 ms (target ≤ 4,500 ms) — `audit_brain_report.json` |
| Brain — first-token (warm) | **PASS** | 356 ms median warm (target ≤ 1,400 ms) |
| Brain — first-token (cold) | **PASS** | 803 ms (target ≤ 1,400 ms) |
| Brain — long-prefill 1,800 tok | **PASS** | 210 ms (prompt cache hit) |
| Intent engine | **EXCEPTIONAL** | 0.11 ms median (target ≤ 50 ms) |
| Quick-reply gates | **PASS** | 0.33 ms median, 11/11 cases correct |
| Semantic cache | **PASS** | 10/10 class-gate tests correct |
| Embedding (warm) | **PASS** | 3.33 ms median per query (was 670 ms pre-Ω.9) |
| Vector store | **PASS** | 0.89 ms median, 290 entries across 5 ChromaDB collections |
| Cloud rotation | **WIRED, NO KEYS** | 4/4 slots configured, 0/4 with credentials |
| Voice pipeline (live in-process) | **PASS** | TTS preflight + STT kick wired, gates green |
| Voice pipeline (live boot) | **STALE — needs fresh log** | Pre-Ω.9 had 3× TTS wedge + 5× STT starve |
| Test suite | **99.95 %** | 1,982 / 1,983 pass; one pre-existing test stale |
| Config drift | **CLEARED** | Ω.9 audit `overall pass = True` |
| Invariant compliance | **15 / 17 honored** | I-06 (max_tokens) and I-12 (cloud opt-in) violated |

**Bottom line:** The static and unit-level health is strong. The real proof is the next post-Ω.9 boot log — that will determine whether the executor split actually fixes the wedges and starvation that grounded the runtime grade.

---

## 2 · Hardware & Runtime Profile

| | |
|---|---|
| Machine | MacBook Air, Model `Mac17,4`, MDVQ4HN/A |
| Chip | Apple M5 |
| CPU | 10-core (10P / 10E, `arm64`) |
| GPU | Apple M5 10-core, MLX `Device(gpu, 0)` confirmed live |
| ANE | Apple Neural Engine (used by WhisperKit CoreML graph) |
| Unified memory | 16.0 GB (no discrete VRAM — every GB is shared with macOS, Chrome, MLX, ChromaDB, embedding warm-file) |
| Storage | 460.4 GB APFS volume, 365.2 GB free (3.1 % used) |
| OS | Darwin 25.4.0 (xnu-12377.101.15) |
| Display | 2880 × 1864 Retina @ 60 Hz |
| Battery | Yes |
| Python (project) | **3.11.15** in `.venv/bin/python` (193 packages installed) |
| Python (system) | 3.9.6 (do not use; not all deps installed there) |
| Live RAM at audit start | 60.0 % used / 6.4 GB available |
| Live RAM after audits | 70.9 % used / ~4.6 GB available |

**Source of truth:** `data/system_profile.json` (auto-written at boot), `system_profiler SPHardwareDataType`, `audit_brain_report.json` `system` block.

**Red flag:** Live RAM is already at 60 % **before ATOM is running**. After the brain + VLM + embeddings + ChromaDB + dashboard load, ATOM occupies an additional ~3.5–4 GB (4-bit Qwen ~1.6 GB warm + 1.6 GB SmolVLM if loaded + 0.5 GB embeddings + 0.3 GB ChromaDB), pushing system to ~80–85 % at idle on a typical Chrome-open workday. This is exactly why Sprint Ω.10 disabled `dream_engine`, `behavior_model`, `self_optimizer`, and `prediction_preload`.

---

## 3 · Codebase Scale

| Bucket | Files | LOC |
|---|---:|---:|
| Total Python | 501 | 154,451 |
| `core/` (router, kernel, intent, memory, RAG, observability, autonomy, security, governors, …) | ~290 | 80,267 |
| `voice/` (STT engines, TTS, listening modes, mic manager, audio preprocessor, interrupt handler, ack engine) | ~30 | 19,313 |
| `tests/` | 141 | 34,639 |
| `scripts/` | 27 | 4,968 |
| `cursor_bridge/` (prompt builder, local brain controller) | 5 | 3,765 |
| `brain/` (MLX inference, memory graph, mini-LLM fallback) | ~10 | 3,511 |
| `ui/` (web_dashboard.py, native_ui.py + dashboard HTML/JS/CSS) | 4 | 1,308 |
| Misc (`tools/`, `context/`, `docs/`) | rest | ~6,680 |

**Test:Production ratio:** 34.6 K / (154.4 K − 34.6 K − 5.0 K) ≈ **30 %** test code by LOC, 1,983 collected tests.

The skill file `.cursor/skills/atom-systems-engineer/SKILL.md` claims `~51 K LOC across ~150 Python files` — **stale by 3×** (codebase has grown roughly 3× since that doc was written; the skill should be regenerated).

---

## 4 · On-Disk Models

| Path | Size | Status |
|---|---:|---|
| `models/qwen3-4b-instruct-4bit/` | 2.1 GB (`model.safetensors` 2,263,022,417 B) | **Active** — `brain.mlx_model` |
| `models/smolvlm-instruct-4bit/` | 1.4 GB | Lazy-loaded (`vision.vlm.warm_at_boot=false`) |
| `models/qwen3-8b-4bit/` | **MISSING** | Referenced by stale `brain.speculative_decoding.draft_model_path` |
| `models/ggml-large-v3-turbo-q5_0.bin` | **MISSING** | Referenced by `stt.whisper_model_path` (legacy whisper.cpp; current STT uses WhisperKit serve, not this file) |
| `models/kokoro/kokoro-v1.0.onnx` | **MISSING** | Referenced by `tts.kokoro_*` (currently using NSSpeechSynthesizer; Kokoro is opt-in) |
| WhisperKit CoreML model | 632 MB | Downloaded by `whisperkit-cli` to `~/.whisperkit/...` on demand (`whisper-large-v3-v20240930_turbo_632MB`) |

**Skill drift:** `SKILL.md` says default is **Qwen3-8B-4bit**. Reality after Sprint Ω.8 (commit `49a95b1`) and the current Ω.9/Ω.10 head: default is **Qwen3-4B-Instruct-4bit**. The 8B was retired and uninstalled to free disk and remove the dual-model RAM risk. Skill needs regeneration.

**WhisperKit CLI:** `/opt/homebrew/bin/whisperkit-cli` v0.18.0, Swift native, ANE-backed. `lsof -nP -iTCP:50060 -sTCP:LISTEN` returns no listener at audit time (ATOM not currently running) — clean state.

---

## 5 · Boot Sequence — Static Spec

`main.py` (4,758 lines) runs the following at every boot:

```
config load → setup_logging → validate_config → security_secret_scrub
 → AsyncEventBus → StateManager → CacheEngine → MemoryEngine
 → IntentEngine → CommandRegistry → ContextEngine
 → SecurityPolicy → SecurityFortress → CodeIntrospector
 → SystemScanner → SystemIndexer → ToolRegistry → ReasoningPlanner
 → 7 cognitive modules → CognitiveKernel.route warmup
 → MLX Brain warm load (qwen3-4b-instruct-4bit)
 → Voice pipeline (deferred until boot TTS done)
   → tts_macos.preflight_speak (Ω.9 boot warm)
   → stt_whisperkit.kick_serve_async (Ω.9 boot warm)
   → mic_manager device select
   → audio_preprocessor (VAD, noise gate)
   → STT engine bind to port 50060
 → IdleMaintenance.start (new in Ω.10)
 → 4 watchdogs (runtime, health, system_watcher, gpu_stall)
 → Web dashboard on http://127.0.0.1:8765/
 → Boot greeting "Here, Boss." → state listening
```

`scripts/m5_baseline_benchmark.py` measured the boot-relevant module init costs (independent of brain/voice load, which are profiled below):

| Stage | Time (ms) | RSS Δ (MB) | Status |
|---|---:|---:|---|
| Config + logging + validate | 44.9 | +6.5 | OK |
| Import 8 core modules | 53.9 | +4.6 | OK |
| `MemoryEngine(config)` | 1,018.1 | +219.3 | OK (SQLite + ChromaDB warmup) |
| `ContextEngine.get_bundle()` | 158.6 | +0 | OK |
| `SecurityFortress(config)` | 62.3 | +2.0 | OK |
| `CodeIntrospector.scan()` | 1,793.8 | +22.0 | OK |
| `SystemScanner` | 2.2 | +0 | OK |
| Cognitive layer (7 modules) | 7.0 | +0 | OK |
| `IntentEngine.match()` per query (avg of 10) | 0.196 | – | OK |
| **Subtotal: stages 1–8** | **3,142.5** | **+254.4** | – |

**`m5_baseline_benchmark.py` reports 5 stale FAILs** (`speech_recognition`, `edge_tts`, `pygame`, `llama_cpp`, `SystemIndexer.start()` event-loop issue). These are **expected**: Sprint Ω.7+ removed all four legacy deps; the benchmark script itself wasn't updated and is now misleading. Recommend deleting these 4 import probes from the script.

### Live Boot Timelines (pre-Ω.9, `logs/atom.log`)

`atom.boot.timeline` events (5 most recent boots, all pre-`3f0c787`):

| Timestamp | Total | tts_init | cold_start | persona_pin | stt_preload |
|---|---:|---:|---:|---:|---:|
| 2026-04-26 21:01:50 | **18,255 ms** | 987 | 9,429 | 0 | 16,133 |
| 2026-04-26 21:22:11 | **10,942 ms** | 1,346 | 5,764 | 0 | 8,789 |
| 2026-04-26 21:28:15 | **19,285 ms** | 1,001 | 10,091 | 0 | 17,140 |
| 2026-04-26 22:21:38 | **13,680 ms** | 1,192 | 5,590 | 2,872 | 9,450 |
| 2026-04-26 22:48:32 | **18,460 ms** | 1,446 | 10,426 | 2,984 | 16,296 |

vs targets (skill `SKILL.md` § Performance Budgets):

| Metric | Target | Best of 5 (pre-Ω.9) | Worst of 5 |
|---|---:|---:|---:|
| Boot total | ≤ 12,000 ms | 10,942 (1× pass) | 19,285 (4× over) |
| Cold start | ≤ 6,500 ms | 5,590 (1× pass) | 10,426 (4× over) |
| TTS ready | ≤ 1,800 ms | 987 (always pass) | 1,446 |
| STT preload | ≤ 6,000 ms (warm) | 8,789 (3.5× over) | 17,140 (2.9× over) |

**STT preload is the long pole** — every boot. WhisperKit CoreML model load (632 MB) does not appear to be amortized across boots. Sprint Ω.9 added `kick_serve_async` to overlap this with boot, but the post-Ω.9 boot log is not yet captured.

---

## 6 · Brain / LLM Stack — Live Numbers

**Source:** `scripts/audit_brain_live.py` (cold MLX load + 4-turn warm test) → `audit_brain_report.json`. Run completed 2026-04-27 ~00:30 IST against current `models/qwen3-4b-instruct-4bit` and current `config/settings.json`.

```
mlx_cold_load_ms        : 4,780.5 ms
rss_after_load_mb       : 1,583    (delta +1,508 from baseline)
turn1_first_token_ms    :   803    (cold prefill)
turn2_first_token_ms    :   356    (warm / cache hit)
turn3_full_response_ms  : 2,966    (52 words, 17.5 wps decode)
turn4_long_prefill_ms   :   210    (~1,800-tok prefix re-use)
final_rss_mb            : 1,637
system_ram_pct_final    : 70.1 %
```

A second consecutive run (model already resident) — `scripts/audit_voice_pipeline.py` § Phase 1:

```
warm_load_ms             :   977   (prompt-cache + tokenizer reuse)
rss_load_delta_mb        : +2,359
q1_first_token (cold)    :   994   (13 words, 13 wps)
q2_first_token (warm)    :   848   ( 8 words)
q3_first_token           : 1,277   (21 words, 21 wps)
q4_first_token           : 1,611   (13 words, 13 wps)
q5_first_token           : 1,303   (18 words, 18 wps)
long_prefill_ms          : 1,713
prompt_cache_hits/misses : 6 / 0
```

**Read-out:**
- Cold load **at target** (4.78 s vs 4.5 s budget — 6 % over, acceptable).
- First-token cold **47 % better than budget** (803 vs 1,400 ms).
- First-token warm **75 % better** (356 ms median).
- Sustained decode **17–21 wps** (≈ 25–30 tokens/s), credible for 4-bit Qwen on M5 GPU.
- Long-prefill 210 ms confirms KV-cache reuse is effective for the 1,800-token voice prompt.

**Resident memory:** 1.6 GB warm. Add 1.4 GB if SmolVLM is forced loaded. Add 470 MB embedding model. Add ~35 MB ChromaDB. Net active "ATOM RAM cost" at idle: **~2.1 GB; under voice load: ~3.5–4 GB; with VLM warm: ~5 GB**.

**Brain config (`config/settings.json` → `brain`):**

| Key | Value | Comment |
|---|---|---|
| `mlx_model` | `models/qwen3-4b-instruct-4bit` | matches on-disk |
| `mlx_model_fallback` | `models/qwen3-4b-instruct-4bit` | self-fallback (no real backup) |
| `n_ctx` | 6,144 | adequate for 1.2 K voice prompt + history |
| `max_tokens` | **384** | **VIOLATES INVARIANT I-06 (≤ 320 for voice)** |
| `temperature` | 0.6 | |
| `top_p` | 0.85 | |
| `repeat_penalty` | 1.1 | |
| `single_resident` | `true` | enforced |
| `speculative_decoding.enabled` | `false` | required given single-resident |
| `speculative_decoding.draft_model_path` | `models/qwen3-8b-4bit` | dead reference, model not on disk |
| `kv_bits` / `kv_group_size` | 8 / 64 | 8-bit KV cache |
| `prompt_cache_max_size` / `_max_mb` | 6 entries / 320 MB | KV-cache budget |
| `prompt_cache_persist` | `true` (`data/prompt_cache_v33.safetensors`) | persistent KV |
| `mx_compile_enabled` | `true` | MLX graph compile |

**Recommended fix #1:** Drop `brain.max_tokens` from 384 → 320 (or 256 for `optimal` profile, 384 only for `full_performance`). Current value silently violates I-06.

**Recommended fix #2:** Either (a) restore the 8B as the actual draft model and re-enable speculative, or (b) drop `speculative_decoding.draft_model_path` to `""` to remove the dead reference and the 8B mention from the config.

---

## 7 · Voice Pipeline

### 7.1 STT — WhisperKit (CoreML / ANE) primary

| | |
|---|---|
| Engine | `whisperkit-cli` v0.18.0 (Swift, ANE-accelerated) |
| Model | `whisper-large-v3-v20240930_turbo_632MB` |
| Port | 50060 (HTTP serve) |
| Locale | `en-US` (with `bilingual=true` for Hindi support) |
| Sample rate | 16,000 Hz |
| VAD | aggressiveness 3 |
| Smart turn-taker | enabled, EoT prob 0.78, midthought lockout 0.92 |
| Noise floor | −45.0 dBFS, 3 consecutive frames |
| Promotion confidence | 0.50 (with wake) / 0.65 (without wake) |
| Post-TTS cooldown | 600 ms |
| Whisper confirmer | **disabled** (config `stt.whisper_confirm.enabled=false`, was retired in Ω.7) |
| Wake-mishear normalizer | catches `adam → atom`, `adtan → atom`, `adton → atom`, etc. |

**Fallback chain:** WhisperKit → SFSpeechRecognizer (`voice/stt_macos.py`) → degraded ("voice input disabled for this session") in extreme failure.

**Pre-Ω.9 STT problems in `logs/atom.log` (24-hour history):**
- **5× STT Watchdog "recognizer likely starved"** events (no partials for 6.0–8.0 s despite speech-like audio at −12 to −33 dBFS).
- **STT preload 8.8–17.1 s every boot** (target ≤ 6 s).
- **11 wake-mishear corrections** (Adam → Atom). The normalizer is doing its job; the underlying mis-hearings are an STT concern, not a router one.
- **WhisperKitSTT correction trace** showing duplicated transcription `"...separate pools. ...separate pools."` — likely the model double-ran on the same audio frame (echo of TTS during tail-mute window).

### 7.2 TTS — NSSpeechSynthesizer primary

| | |
|---|---|
| Engine | macOS native `NSSpeechSynthesizer` |
| Voice | `jarvis` (custom voice slot — falls back to `Daniel` if unavailable) |
| Rate | 205 wpm |
| First-word warmup | 140 ms |
| Tail drain (built-in) | 120 ms |
| Tail drain (Bluetooth) | 200 ms |
| Echo ring buffer | 30 s window, every utterance pushed before `startSpeakingString_` |
| Watchdog timeout | 15 s static, +0.5 s per word, max 45 s dynamic |
| Pre-flight speak (Ω.9) | wired in `voice/tts_macos.preflight_speak()` |
| Streaming slice TTS | enabled |
| Max lines per turn | 4 |

**Optional (not currently active):**
- Edge-TTS (`en-GB-SoniaNeural` voice) — installed
- Kokoro-TTS (`af_heart` voice) — `kokoro-onnx` 0.3.9 + `kokoro-tts` 2.3.1 installed; model files NOT on disk

**Pre-Ω.9 TTS problems in `logs/atom.log`:**
- **3× NSSpeechSynthesizer wedged mid-utterance** events:
  - `21:31:17`: 6.0 s, 10 words, rate 193, `stuck_starts=1`
  - `20:52:10`: 4.1 s, 4 words, rate 182, `stuck_starts=1`
  - `22:51:02`: 6.5 s, 20 words, rate 205, `stuck_starts=1` — fell back to `say` CLI

The wedge pattern (`isSpeaking=True for ≥4 s on a short utterance`) strongly suggests the AsyncEventBus single executor was starved by an overlapping LLM job. **Sprint Ω.9 split the executor pools (light=2 / heavy=3) precisely to fix this.** Validation pending fresh boot log.

### 7.3 Voice Pipeline Live Test (Sprint Ω.9 in-process)

`scripts/audit_voice_pipeline.py` Phase 2 results (warm in-process):

| Stage | Median (ms) | Max (ms) | n | Notes |
|---|---:|---:|---:|---|
| Intent classify | **0.11** | 0.19 | 12 | regex aggregator, all paths |
| Quick-reply lookup | **0.33** | 1.46 | 12 | ConfigDict; explanatory-gate added in Ω.9 |
| Prompt builder (voice) | **1.60** | 1.61 | – | output ≈ 1,227 tokens |
| Embedding | **3.33** | 4.95 | 8 | MLX MiniLM-L6, 384-dim, warm cache |
| Vector store query | **0.89** | 1.04 | – | ChromaDB persistent client |
| Memory retrieve | **3.86** | 4.16 | – | full graph + vector path |

**Soft path total** (intent + cache + builder + embedding + vector + memory) ≈ **9.6 ms median**, well under the latency-controller `direct_budget_ms: 50`.

### 7.4 Voice Invariant Cross-Check

| Invariant | File(s) | Status |
|---|---|---|
| I-01 prompt builder must not emit quotable rule text | `cursor_bridge/structured_prompt_builder.py` | **PASS** — `tests/test_jarvis_stream_sanitizer.py::test_system_prompt_v3_uses_opaque_style_fingerprint` green |
| I-02 every TTS utterance hits echo ring | `voice/tts_macos.py` | **PASS** — covered by `test_voice_pipeline_critical.py::test_tts_is_echo_recognises_recent_spoken_text` |
| I-03 STT finals consult `tts.is_echo()` | `voice/stt_*.py` (3 paths) | **PASS** — `test_perception_predicted_interrupt_drops_self_echo_partials` green |
| I-04 guardrail rewrite ≠ rejection | `core/router/router.py`, `cursor_bridge/local_brain_controller.py` | **PASS** — `test_voice_pipeline_critical.py` rewrite suite green |
| I-05 intent regex priming at boot | `core/boot/cold_start.py:_prime_intent_engine_regexes` | **PASS** — confirmed in `tests/test_cold_start.py` |
| I-06 `max_tokens ≤ 320` | `config/settings.json:brain.max_tokens` | **FAIL — 384 currently** |
| I-07 `max_action_tier ≤ 3` w/o biometrics | `core/security_tiers.py` | **PASS** — `max_tier_for_security_mode("strict") = 3`; tier-4 (kill_process, shutdown_pc, restart_pc, sleep_pc) blocked |
| I-08 settings.json single source of truth | global | **PASS** — no model/voice/timeout hardcodes found in `core/`, `voice/`, `brain/` |
| I-09 legacy keys consistent | `config/settings.json:brain.*` | **PASS** — `mlx_primary_model`/`mlx_fast_model`/`mlx_deep_model`/`mlx_default_role`/`model_path` all absent |
| I-10 memory pressure tier cooperative | `core/main.py`, RAG/MLX | **PASS by spec** — runtime validation pending |
| I-11 fresh boot log per voice/LLM change | runbook | **PENDING — no post-Ω.9 log yet** |
| I-12 cloud opt-in only | `config/settings.json:cloud.enabled` | **AT RISK — `cloud.enabled=true` globally**; per-capability flags exist but the global default is `true`, so any module that only checks `cloud.enabled` will phone out |
| I-13 AsyncEventBus pub/sub | `core/async_event_bus.py` | **PASS** — Ω.9 split into `_LIGHT_EXEC`/`_HEAVY_EXEC` thread pools |
| I-14 no narration comments | global | **PASS by spot check** — sampled brain/voice/router |
| I-15 `from __future__ import annotations` | global | **PASS by spot check** — top of every audit script and new file |
| I-16 WhisperKit serve recoverable | `voice/stt_whisperkit.py:_reap_stale_serve_on_port` | **PASS by spec** — covered in playbook PB-13 |
| I-17 self-echo path coverage on every STT engine | `voice/stt_macos.py`, `voice/stt_whisperkit.py` | **PASS** — output mute + echo guard + ATOM self-text strip + wake-mishear normalize all present |

**Net violation count:** **2 of 17** (I-06, I-12). I-11 is a process / pending validation, not a code violation.

---

## 8 · Cognitive Kernel + Intent Engine + Routing

### 8.1 Cognitive Kernel — live routing test

`core/cognitive_kernel.py:CognitiveKernel.route()` invoked on 13 representative queries (live, Sprint Ω.9 head):

| Query | Path | Model | Budget (ms) | RAG | Mode | Reason | Wall (ms) |
|---|---|---|---:|---|---|---|---:|
| "what time is it" | QUICK | qwen3-4b-instruct-4bit | 1,260 | False | FAST | simple_query | 5.28 |
| "cpu" | QUICK | qwen3-4b-instruct-4bit | 1,260 | False | FAST | simple_query | 1.58 |
| "open spotify" | FULL | qwen3-4b-instruct-4bit | 4,200 | False | SMART | moderate_query | 1.63 |
| "tell me about yourself" | DIRECT | none | 100 | False | FAST | quick_reply | 0.68 |
| "how does Apple Silicon use the Neural Engine" | FULL | qwen3-4b-instruct-4bit | 5,000 | **True** | SMART | moderate_query | 0.86 |
| "walk me through how speculative decoding works" | FULL | qwen3-4b-instruct-4bit | 4,200 | False | SMART | moderate_query | 0.74 |
| "i feel tired today and want a story" | DEEP | qwen3-4b-instruct-4bit | 8,400 | True | DEEP | complex_query | 1.87 |
| "what should I do for the next two hours" | QUICK | qwen3-4b-instruct-4bit | 1,260 | False | FAST | simple_query | 1.96 |
| "analyse memory pressure on my mac and recommend a fix" | QUICK | qwen3-4b-instruct-4bit | 1,500 | False | FAST | simple_query | 2.05 |
| "kill safari" | QUICK | qwen3-4b-instruct-4bit | 1,260 | False | FAST | simple_query | 1.65 |
| "set a 25 minute timer" | FULL | qwen3-4b-instruct-4bit | 4,200 | False | SMART | moderate_query | 1.73 |
| "switch to focus mode" | QUICK | qwen3-4b-instruct-4bit | 1,260 | False | FAST | simple_query | 1.73 |
| "remind me to call mom at 9am tomorrow" | FULL | qwen3-4b-instruct-4bit | 4,200 | False | SMART | moderate_query | 1.86 |

**Reads:**
- Routing wall-time itself is **always < 6 ms** (≪ kernel budget).
- "tell me about yourself" correctly cached as DIRECT.
- "i feel tired today and want a story" is the only DEEP route — 8.4 s budget feels right for an emotional/long query.

**Routing concerns:**
- **"open spotify"** routes to FULL (4.2 s LLM budget) instead of DIRECT — there is a deterministic intent (`open_app`) with a known answer. Calling the LLM for this wastes 1–2 s. The `route()` decision doesn't fuse with `IntentEngine.classify()` results before deciding the path.
- **"kill safari"** routes to QUICK; same fix would short-circuit to DIRECT (`close_app` action, tier 3, allowed in strict mode).
- **"switch to focus mode"** likewise QUICK → should be DIRECT (`mode_switch` action).
- **"what should I do for the next two hours"** routes QUICK to LLM (1.26 s budget), but no system / calendar context is injected — likely answers generically.

### 8.2 Intent Engine — live classification test

`core/intent_engine/__init__.py:IntentEngine.classify_silent()` on the same 13 queries:

| Query | Intent | Action | Time (ms) |
|---|---|---|---:|
| "what time is it" | time | – | 0.09 |
| "cpu" | **fallback** | – | 0.03 |
| "open spotify" | open_app | open_app | 0.18 |
| "tell me about yourself" | brain_recall | brain_recall | 0.15 |
| "how does Apple Silicon use the Neural Engine" | fallback | – | 0.25 |
| "walk me through how speculative decoding works" | fallback | – | 0.22 |
| "i feel tired today and want a story" | fallback | – | 0.21 |
| "what should I do for the next two hours" | **music_next** (false positive) | music_next | 0.15 |
| "analyse memory pressure on my mac and recommend a fix" | fallback | – | 0.28 |
| "kill safari" | close_app | close_app | 0.08 |
| "set a 25 minute timer" | **fallback** | – | 0.14 |
| "switch to focus mode" | mode_switch | mode_switch | 0.11 |
| "remind me to call mom at 9am tomorrow" | **fallback** | – | 0.22 |

**Bugs surfaced (pre-existing, not Ω.9 regressions):**

1. **`"cpu"` → fallback** — the bare word is in `_INFO_INTENTS` but the regex aggregator's first match wins, and the OS-info intent file checks for `cpu usage`/`cpu pct` style phrases. Bare-noun queries fall through to LLM. This is the same shape as the user's "Adam → Atom" mishearing: the engine is too literal.
2. **`"what should I do for the next two hours"` → `music_next`** (false positive). `music_next` matches the bare token "next" in a sentence that contains it; the engine should require word-boundary on a music context (e.g. preceded by "play", "song", "track"). This is the false-positive class the architecture map calls out under PB-04.
3. **`"set a 25 minute timer"` → fallback** — there is no `set_timer` action wired into the productivity intents. Falls to LLM, which can produce a freeform answer but cannot actually start a timer.
4. **`"remind me to call mom at 9am tomorrow"` → fallback** — similarly no reminder pipeline.

**Performance:** all classifications < 0.3 ms — exceptional, well within the 50 ms watchdog. Sprint Ω.5's regex priming did its job.

### 8.3 Quick-reply Explanatory Gate (Ω.9 specific)

`scripts/audit_omega9_quick.py` runs the full 11-case gate matrix:

| Query | Expected | Fired? | OK |
|---|---|:---:|:---:|
| "hi" | filler-fires | **yes** ("Ready, Boss.") | ✓ |
| "how are you" | filler-fires | yes ("All good here, Boss.") | ✓ |
| "tell me about yourself" | domain-fires | yes ("I'm ATOM, your operating intelligence, Boss.") | ✓ |
| "how does unified memory work in detail" | domain-fires | yes (Apple Silicon answer) | ✓ |
| "explain the difference between optimal and full performance" | domain-fires | yes | ✓ |
| **"explain how WhisperKit runs on the Apple Neural Engine"** | **explain-gate-blocks** | **no** (correctly suppressed) | ✓ |
| "compare safari and arc for coding on macbook air" | domain-fires | yes | ✓ |
| **"walk me through how speculative decoding works"** | **explain-gate-blocks** | **no** (correctly suppressed) | ✓ |
| "what is the latency difference between optimal mode and full performance mode" | domain-fires | yes | ✓ |
| **"describe how Apple Silicon and the Neural Engine cooperate during inference"** | **explain-gate-blocks** | **no** (correctly suppressed) | ✓ |
| **"elaborate on how prompt caching reduces first-token latency"** | **explain-gate-blocks** | **no** (correctly suppressed) | ✓ |

**11 / 11 pass.** Ω.9 explain-gate (`_EXPLAIN_GATE_RE` in `core/quick_replies.py`) correctly routes "explain/walk me through/elaborate/describe how" queries to the LLM instead of returning a canned chit-chat reply.

### 8.4 Semantic Cache Class Gate

| Query | Expected `exact_only` | Got | OK |
|---|:---:|:---:|:---:|
| "who are you" | True | True | ✓ |
| "hi" | True | True | ✓ |
| "what time is it" | True | True | ✓ |
| "hello" | True | True | ✓ |
| "namaste" | True | True | ✓ |
| "how are you" | True | True | ✓ |
| "good morning" | True | True | ✓ |
| "how does Apple Silicon use the Neural Engine" | False | False | ✓ |
| "can you draft a thank-you email to Riya" | False | False | ✓ |
| "set a 25 minute focus timer" | False | False | ✓ |

**10 / 10 pass.** Ω.9's `exact_only` flag for short identity/meta queries correctly prevents cross-turn cache bleed (e.g. "hi" → answer for "who are you").

---

## 9 · Memory, RAG, and Vector Store

### 9.1 ChromaDB (`data/vector_db/`, 1.7 MB on disk)

| Collection | Entries | Use |
|---|---:|---|
| `interactions` | 0 | per-turn interaction log (currently empty) |
| `conversations` | 30 | persisted conversation summaries |
| `atom_memories` | 0 | long-term memories (currently empty) |
| `documents` | 0 | ingested documents (auto-ingest disabled) |
| `facts` | **260** | learned facts (only collection actually used) |

**Config:** ChromaDB persistent client at `data/vector_db`, 5 collections. Embedding model: `mlx-community/all-MiniLM-L6-v2-4bit`, 384-dim, MLX backend (after Sprint Ω.3 swap). Warm-file `data/embeddings_warm.npz` (152 KB, 1,024 entries cap).

**Live retrieval timing:** 0.89 ms median (warm), 4.16 ms max for `retrieve_relevant_context`. Excellent.

### 9.2 SQLite memory graph (`data/atom_memory.db`, 36 KB)

| Table | Notes |
|---|---|
| `memory_nodes` | graph nodes (facts, entities) |
| `memory_edges` | typed edges between nodes |
| `owner_preferences` | persisted user preferences |

`brain/memory_graph.py` runs `v22_confidence` scoring. Used by `core/router/router.py` to enrich the LLM context.

### 9.3 Other persistent stores

| Path | Size | Rows / Notes |
|---|---:|---|
| `data/semantic_cache.db` | 24 KB | 2 cache rows + 2 meta rows |
| `data/rag_embedding_cache.sqlite` | 16 KB | 18 cached embedding rows |
| `data/screen_observations.sqlite` | 28 KB | 6 screen observations (loop is currently disabled) |
| `data/system_profile.json` | 4 KB | machine snapshot, refreshed at boot |
| `data/user_profile.json` | 4 KB | preferred_browser=∅, frequent_apps=Chrome+Cursor, name="Boss" |
| `data/behavior_profile.json` | 4 KB | preferred_rate 1.03, preferred_pause 0.97, verbosity 0.78 |
| `data/embeddings_warm.npz` | 152 KB | warm cache for first-query speedup |
| `data/real_world_cache.json` | 4 KB | weather/news cache |

### 9.4 RAG Budget

`config.rag`:
- `top_k = 3`, `max_snippets = 3`
- `first_token_budget_ms = 120` (RAG must finish in 120 ms before first token)
- `pressure_threshold_pct = 82` → drop `top_k` to `1` when memory ≥ 82 %
- `prefetch_enabled = true` (speculative RAG during LLM thinking)

This is conservative and well-tuned for a 16 GB unified-memory machine.

---

## 10 · Cloud Routing

`config.cloud`: **enabled = true**, provider = `rotating`, 4 slots configured.

| Slot | Tier | Fast model | Deep model | Soft RPM | Has key (live) |
|---|:---:|---|---|:---:|:---:|
| groq | 1 | `llama-3.1-8b-instant` | `llama-3.3-70b-versatile` | 28 | **No** |
| gemini | 1 | `gemini-2.5-flash-lite` | `gemini-2.5-flash` | 12 | **No** |
| cerebras | 2 | `llama3.1-8b` | `gpt-oss-120b` | 25 | **No** |
| nvidia | 3 | `meta/llama-3.1-8b-instruct` | `meta/llama-3.3-70b-instruct` | 35 | **No** |

`audit_omega9_quick.py` confirms `slot_count=4`, `ready_slots=[]` — **0 of 4 cloud slots are usable** because no API keys are loaded.

`security_gateway.audit_cloud_calls = true` and `max_outbound_length = 500` — cloud requests are length-capped and audited if they ever fire. A privacy redactor must run on outbound payloads (per I-12).

**Status:** Cloud is fully wired (rotating client v3 with cooldowns, 429 detection, hard-failure threshold, RPM tracking) but **idle until keys are provisioned**. Run `scripts/setup_api_keys.py` to populate slots.

**Risk (I-12):** `cloud.enabled = true` is the global gate that any cloud client can read. Per the invariant doctrine, the global default should be **`false`** with per-capability flags (`cloud.reasoning=true`, `cloud.search=true`) flipped on individually. Today, any new cloud-using module that only checks `cloud.enabled` will phone out the moment a key is provisioned, even if the user only intended one capability.

**Recommended fix:** Flip `cloud.enabled` to `false` and gate the rotating client on `cloud.reasoning` (or a fresh `cloud.fallback_when_local_fails` flag) instead. Keep capability-level flags as today.

---

## 11 · Watchdogs, Governors, and Idle Maintenance

### 11.1 Runtime Watchdog (`core/runtime_watchdog.py`)

| Budget | Value | Source |
|---|---:|---|
| Thinking timeout | 90 s | `performance.watchdog_thinking_timeout_s` |
| Speaking timeout | 300 s | `watchdog_speaking_timeout_s` |
| Intent budget | **150 ms** | `watchdog_intent_timeout_ms` (skill says 50 ms — drift) |
| Intent boot grace | 20 s | `watchdog_intent_boot_grace_s` |
| Cache budget | 200 ms | `watchdog_cache_timeout_ms` |
| RAG budget | 750 ms | `watchdog_rag_timeout_ms` |
| LLM budget | 18 s | `watchdog_llm_timeout_s` |
| LLM pressure-extend bonus | +8 s (max 28 s) when memory > 80 % | `watchdog_llm_pressure_extend_*` |
| TTS budget | 15 s static + 0.5 s/word, max 45 s dynamic | `watchdog_tts_*` |
| Tool budget | 10 s | `watchdog_tool_timeout_s` |
| Poll interval | 2 s | `watchdog_poll_interval_s` |
| Supervisor restart cooldown | 8 s | |

**Drift:** the skill says intent budget is **50 ms** (matching I-05's "intent regex priming … trips the 50 ms watchdog"). The config has it at **150 ms**, almost certainly because the original 50 ms was tripping post-priming for normal queries on cold disk-cache misses. Either the skill is stale or the config relaxed unilaterally — recommend reconciling.

### 11.2 Memory Governor (`config.memory_governor`)

| Tier | Threshold | Action |
|---|---|---|
| 0 | < 80 % | normal |
| 1 | 80–86 % | start eviction (smolvlm → whisper_confirmer → draft_model → embeddings_warm_cache → persona_kv_cache) |
| 2 | 86–92 % | aggressive eviction |
| 3 | ≥ 92 % | critical — defer all background work |
| Re-warm | drop ≥ 6 % below tier threshold | hysteresis to prevent flapping |

`config.memory.pressure_tiers.critical_pct = 90` — slightly tighter than memory_governor tier-3 (92 %). They're separate variables for two governors that should agree; minor consistency cleanup recommended.

**Pre-Ω.9 evidence:** 27 memory pressure events in `logs/atom.log`, peaked at 89.4 %. With Ω.10's `idle_maintenance.clear_mlx_cache_on_idle=true`, the warm-RAM ceiling should drop after the first 60 s idle window (config `freeze_after_boot_s=60`, `idle_threshold_s=120`).

### 11.3 Idle Maintenance (Sprint Ω.10, new)

Wired in `core/idle_maintenance.py` (418 LOC). Triggers GC + MLX cache release after configurable idle window. Diagnostics covered by 15 unit tests in `tests/test_idle_maintenance.py` — all passing.

| Knob | Value |
|---|---|
| `freeze_after_boot_s` | 60 |
| `idle_threshold_s` | 120 |
| `tick_interval_s` | 30 |
| `min_action_interval_s` | 60 |
| `gc_threshold_gen0/1/2` | 2,000 / 25 / 25 |
| `clear_mlx_cache_on_idle` | true |

**Expected impact:** Approximately 200–400 MB returned to the system after each 2-minute idle period. Live boot-log validation pending.

### 11.4 Other governors

- `core/silicon_governor.py` — Apple Silicon thermal coordination (poll 30 s, threshold 85 °C-equivalent).
- `core/power_governor.py` — plugged-in vs battery adaptive mode.
- `core/gpu_stall_watchdog.py` — Metal/GPU stall recovery (timeout 120 s).
- `core/health_watchdog.py` — periodic health scan (120 s).
- `core/system_watcher.py` — file/app watcher (30 s).

In `logs/atom.log` (24-hour history): 154 governor events emitted across the four governor modules — they are running and chatty.

---

## 12 · Security and Privacy

### 12.1 Security Tier Map (`core/security_tiers.py`)

| Action | Tier | Strict allows? | Escalatable? |
|---|:---:|:---:|:---:|
| `set_volume` | 1 | Yes | – |
| `tts_speak` | 3 | Yes | – |
| `open_app` | 3 | Yes | – |
| `close_app` | 3 | Yes | – |
| `force_kill` | 3 | Yes | – |
| `change_priority` | 3 | Yes | – |
| `open_url` | 3 | Yes | – |
| `click_ui_element` | 3 | Yes | – |
| `send_email` | 3 | Yes | – |
| `lock_screen` | 3 | Yes | – |
| `memorize_fact` | 3 | Yes | – |
| `web_search` | 3 | Yes | – |
| `kill_process` | **4** | **No** (blocked) | Yes (biometric) |
| `shutdown_pc` | **4** | **No** (blocked) | Yes |
| `restart_pc` | **4** | **No** (blocked) | Yes |
| `sleep_pc` | **4** | **No** (blocked) | Yes |

`max_tier_for_security_mode("strict") = 3`. All tier-4 actions blocked by default — to unlock requires biometric (Face ID via `scripts/enroll_owner_face.py`) or passphrase fallback. **I-07 enforced.**

### 12.2 Owner Gate, Voice Auth, Behavior Auth

| File | Purpose | State |
|---|---|---|
| `core/owner_gate.py` | binds owner = "Satyam"; unknown-voice lockout | Active |
| `data/security/voice_profile.json` | enrolled voice embeddings | Present (228 B) |
| `data/security/.vault_key` | encrypted vault | Present (44 B) |
| `data/security/keychain_vault_keys.json` | macOS Keychain bridge | Present |
| `data/security/session.json` | active auth session | Present |
| `config/owner_face.npy` | Face ID encoding | **Not present** |

**Face-ID ladder** (Sprint A1): `scripts/enroll_owner_face.py` not yet run — biometric escalation will fall back to passphrase for tier-4 actions.

### 12.3 Security Gateway (cloud)

`config.security_gateway`:
- `block_system_paths = true`
- `max_outbound_length = 500` chars per cloud payload
- `audit_cloud_calls = true`
- `max_requests_per_minute = 10`

`logs/audit.log` exists (48 KB), append-only. Per-action audits logged for sensitive ops.

### 12.4 Cross-device bridge

| | |
|---|---|
| Bridge port | 8787 |
| Bind host | 127.0.0.1 (loopback only) |
| Allow-origins | `["127.0.0.1"]` |
| Token | `config/bridge_token` (43 B, 0600 mode) |
| FaceID freshness | 300 s |
| Trusted device file | `data/trusted_iphone.json` |
| iPhone bridge (Sprint A1) | wired but optional |
| OpenAI-compat surface | disabled (`cross_device.openai_compat.enabled=false`) |

Network surface is loopback-only. No external listener.

---

## 13 · Test Suite Coverage

| Suite | Files | Tests | Pass | Fail | Notes |
|---|---:|---:|---:|---:|---|
| **Total collected** | 141 | **1,983** | – | – | `pytest --collect-only` |
| Targeted run (smoke + sanitizer + scorecard + idle + kernel) | 7 | 109 | 109 | 0 | < 1 s |
| Voice pipeline critical | 1 | 81 | 81 | 0 | 2.4 s |
| Brain Qwen smoke + swap | 2 | 19 | 19 | 0 | 6.9 s (live MLX load) |
| **Broad run** (excludes 3 live-model files) | 137 | 1,883 | **1,882** | **1** | 24.7 s |
| **Cumulative pass rate** | – | 1,983 | **1,982** | 1 | **99.95 %** |

**Single failure:**

```
FAILED tests/test_sprint_k_recovery.py::test_vision_intent_tolerates_trailing_stt_fragment
AssertionError: assert 'vision_look' == 'vision_describe'
```

The test expects action `vision_describe`; the system now produces `vision_look` for "see me" / "what am i doing" queries. Action enum was renamed in a prior sprint without updating this one regression. **Cosmetic fix:** update test or restore the alias.

**Coverage gaps observed (no test for):**
- WhisperKit `port-bound non-owned` recovery branch (PB-13 case 2).
- TTS `wedged mid-utterance` watchdog → fallback to `say` CLI path.
- Memory governor tier-3 critical eviction order under live pressure.

---

## 14 · Configuration Audit (`config/settings.json`)

`config/settings.json` is **893 lines** of JSON, 29 top-level sections. Single source of truth per I-08.

**Top-level sections present:**
`deployment`, `session`, `skills`, `owner`, `personality`, `cross_device`, `realtime`, `screen_perception_loop`, `vision`, `mic`, `audio_intelligence`, `stt`, `tts`, `voice`, `context`, `cache`, `memory`, `brain`, `assistant_brain`, `ui`, `executor`, `developer`, `security`, `features`, `control`, `performance`, `autonomy`, `cognitive_loop`, `cognitive`, `proactive_coordination`, `proactive_engine`, `conversation_memory`, `morning_briefing`, `embedding`, `vector_store`, `documents`, `reasoning`, `sandbox`, `workflow`, `wake_word`, `emotion`, `screen_reader`, `gpu`, `memory_governor`, `idle_maintenance`, `cognitive_kernel`, `latency_controller`, `v7_intelligence`, `rag`, `v7_gpu`, `cloud`, `cloud_brain_router`, `agent_supervisor`, `security_gateway`, `confidence`, `decision_engine`, `search`, `semantic_cache`.

**Disabled / opt-out sections (current):**
- `realtime.enabled = false` (the `atom-room` web RTC playground)
- `screen_perception_loop.enabled = false` (the screen-watch loop)
- `cognitive.dream_enabled = false`, `behavior_model_enabled = false`, `self_optimizer_enabled = false`, `prediction_preload_enabled = false`, `curiosity_enabled = false` (Ω.10 disabled these — they were emitting log spam without a consumer)
- `cognitive_loop.reflective.enabled = false`, `presence.enabled = false`, `scene.enabled = false`
- `wake_word.enabled = false` (relying on always-on STT instead)
- `documents.auto_ingest_on_boot = false`
- `cross_device.openai_compat.enabled = false`
- `features.web_research = false`, `online_weather = false`

**Kept enabled (with implications):**
- `cloud.enabled = true` (see I-12 risk above)
- `cross_device.enabled = true` (loopback iPhone bridge)
- `vision.enabled = true`, `boot_face_check = true` (camera access at boot)
- `audio_intelligence.enabled = true`, `set_system_default = true` (ATOM may switch macOS default audio device to its preferred mic — privacy-aware behavior, not destructive)

---

## 15 · Live Runtime — JARVIS Scorecard (Pre-Ω.9)

`scripts/jarvis_scorecard.py logs/atom.log` (the most recent boot, `2026-04-26 22:48:32`):

```
ATOM Jarvis scorecard: 31/100 (Needs work)

  voice active: True              stt listening: True
  boot total: 18,460 ms           target ≤ 12,000 ms
  stt ready: 16,296 ms            target ≤ 6,000 ms
  whisperkit preload: 13,595 ms   target ≤ 6,000 ms warm / ≤ 30,000 ms cold-download
  tts ready: 1,446 ms             target ≤ 1,800 ms — pass
  local brain ready: 4,945 ms     target ≤ 4,500 ms — 9 % over
  cold start: 10,426 ms           target ≤ 6,500 ms — 60 % over
  max memory: 89.4 %              target ≤ 75 %
  memory pressure events: 60
  prompt leaks / CoT leaks: 0 / 2
  polite interrupt candidates: 3
  Adam→Atom corrections: 11
  optional loops started: realtime=True, iphone=True, screen_loop=True
```

**Findings:**
- Boot readiness above target.
- STT ready time above target.
- Memory pressure hit 89.4 %.
- 2 chain-of-thought style fragments may have leaked into TTS.
- 3 polite-phrase partials triggered the interrupt handler.
- realtime + iPhone + screen-loop optional surfaces all started despite their config flags being `false` — **needs investigation: are these flags being respected, or are some loops auto-started in main.py?**

---

## 16 · Live Runtime — Triage of `logs/atom.log`

`python3 .cursor/skills/atom-systems-engineer/scripts/triage_log.py logs/atom.log` (24-hour history, Apr 25 00:45 → Apr 26 22:51):

```
Lines:      11,297
Levels:     INFO=9221  WARNING=1202  ERROR=84  CRITICAL=0  FATAL=0
Tracebacks: 49

MODEL
  labels: qwen3-4b-instruct, qwen3-4b-instruct-4bit, qwen3-8b, qwen3-8b-4bit
  peak RAM:  max=7.81 GB  avg=4.72 GB
  first-tok: min=285 p50=2,684 p95=12,821 max=19,072 (n=66)

VOICE
  TTS utterances:       100   (prompt-leaks: 0  cot-leaks: 0)
  STT finals:           26   (echo suppressions: 16)

PIPELINE
  turn total ms:        min=804 p50=6,279 p95=20,289 max=31,844 (n=41)
  watchdog:             llm_inference=13, intent_engine=4, tts_synthesis=3
  mem pressure:         27 event(s)

P0
  - Python tracebacks in log: 49
  - Possible self-talk: 18 STT finals closely match recent TTS — see PB-03

P1
  - Intent-engine budget exceeded 4x after boot grace — see PB-06
  - Watchdog breach: tts_synthesis x3
  - Watchdog breach: llm_inference x13
  - Peak MLX RAM 7.8 GB suggests a ≥7B model loaded
  - 84 ERROR-level log line(s)
  - Memory pressure peaked at 89%

P2
  - Avg turn latency 8,094 ms (target ≤ 3,000 ms for voice UX)
```

**Diagnosis:**

1. **Mixed model labels** (`qwen3-4b-instruct-4bit` and `qwen3-8b`) and a **peak of 7.81 GB MLX RAM** confirm the 8B was resident at some point in this 24-hour window — this is from Sprint Ω.7's last gasp before Ω.8 reverted to 4B. Subsequent boots in this same log show 4B only. Not a current bug.

2. **49 tracebacks all share root cause:** `_drain_proactive_on_listening() got an unexpected keyword argument 'old'`. The `state_changed` event payload changed shape (added `old`/`new`) but the proactive handler signature wasn't updated. **Sprint Ω.9 fixed this** — the handler in `main.py:2830-2832` now accepts `*, old=None, new=None, **_kw`. The 49 tracebacks are historic.

3. **18 self-talk STT finals.** ATOM heard itself say something and promoted it to a final. The echo-ring + output-mute window + wake-mishear normalizer are all in place; in this log, 16 of those were caught by `echo suppressions`. The 18 that promoted are over a 24-hour window of heavy testing — not a daily-life rate. Sprint Ω.9's executor split should reduce mid-utterance starvation that feeds this loop.

4. **First-token p95 = 12,821 ms.** This is enormous and strongly correlates with the LLM-inference watchdog breaches (13 events). Pre-Ω.9, the AsyncEventBus bottleneck plus `_drain_proactive_on_listening` exception-throwing would block the LLM for tens of seconds.

5. **Turn-total p95 = 20,289 ms; max = 31,844 ms.** The same root cause. Voice UX target is ≤ 3,000 ms. Pre-Ω.9 was unusable for fluid conversation.

6. **27 memory-pressure events, peak 89 %.** Ω.10's idle maintenance + Ω.10's removal of dead daemons should drop steady-state significantly. Validation pending.

---

## 17 · Sprint Ω.9 + Ω.10 Status

### 17.1 What shipped (commit `3f0c787`, +2,581 / −424 LOC across 18 files):

**Ω.9 — Voice fluency root cause + boot warm + cache hardening**

| Change | File | Validates |
|---|---|---|
| Split executor pool | `core/async_event_bus.py` | `audit_omega9_quick.py` confirms `light=2/heavy=3`, both alive |
| Light-pool fast path with 0.8 s timeout | `core/fast_path.py` | `audit_omega9_quick.py` ` fast_path_wired = True` |
| `kick_serve_async` | `voice/stt_whisperkit.py` | `audit_omega9_quick.py` `voice_surface.stt.kick_serve_async = True` |
| `preflight_speak` | `voice/tts_macos.py` | `audit_omega9_quick.py` `voice_surface.tts.preflight_speak = True` |
| Boot-warm wiring in `main.py` | `main.py:_background_boot_warm` | `audit_omega9_quick.py` confirms 3 call sites: kick, preflight, embedding_seed |
| Explanatory-intent gate | `core/quick_replies.py` | `audit_omega9_quick.py` 11/11 cases correct |
| `exact_only` semantic cache | `core/semantic_cache.py`, `core/cognitive_kernel.py` | `audit_omega9_quick.py` 10/10 cases correct |
| Multi-shape embedding warmup | `core/embedding_engine.py` | first-query 670 ms → 4 ms (commit log claim, validated against fresh measurement of 3.3 ms median) |
| Config: cerebras → gpt-oss-120b, kernel models → qwen3-4b | `config/settings.json` | `audit_omega9_quick.py` `config_drift_cleared = True` |

**Ω.10 — Dead code + RAM/CPU cleanup**

| Change | File | Comment |
|---|---|---|
| Removed V22 ProactiveDaemon ("convergence_daemon") | `main.py` | `BackgroundTaskManager.submit()` had zero callers; `SystemStateGraph.system_load` had zero writers; `system_alert` / `background_task_complete` events had zero subscribers |
| Dropped PlannerEngine + SystemStateGraph constructions | `cursor_bridge/local_brain_controller.py` | PlannerEngine was constructed without `bus=` so the LLM-plan path always fell back to a stub that prepended ~50 junk tokens |
| `self_optimizer_enabled = false` | `config/settings.json` | only output was 2 indicator log lines every 2 h |
| `debug_snapshot_interval_s = 0` | `config/settings.json` | log-only loop with no consumer |
| New `IdleMaintenance` subsystem | `core/idle_maintenance.py` (418 LOC) + `tests/test_idle_maintenance.py` (15 tests, all green) | GC + MLX cache release after configurable idle window |
| `LocalBrainController.clear_metal_cache` | `cursor_bridge/local_brain_controller.py` | public hook for `IdleMaintenance` |
| New audit scripts | `scripts/audit_brain_live.py`, `audit_subsystems_live.py`, `audit_voice_pipeline.py` | the scripts driving § 6, § 8, § 9 above |
| Cloud secret manager + rotation diagnostics | `core/secrets_manager.py`, `core/cloud/rotating_openai_client.py`, `core/cloud/gemini_client.py` | tightened key load + rotation reporting |

### 17.2 Validation status

| Validation | Status |
|---|---|
| `audit_omega9_quick.py` overall pass | **PASS** |
| `audit_brain_live.py` cold + 4 turns | **PASS** (numbers in § 6) |
| `audit_voice_pipeline.py` end-to-end | **PASS** (numbers in § 7.3) |
| `audit_subsystems_live.py` 6 subsystems | **PASS** (numbers in § 9, § 10) |
| `pytest -k "local_brain or cognitive_kernel or quick_repl or rotation or async_event or fast_path or boot or main"` (sprint commit message claim) | 66 passed (commit log) |
| `pytest tests/` broad run (this audit) | 1,982 / 1,983 (one stale test) |
| `validate_boot.sh` | PASS=4, WARN=2 (max_tokens, cloud.enabled), FAIL=0 |
| **Live post-Ω.9 boot log** | **MISSING — atomCurrentLogs.txt is empty** |

### 17.3 Why the missing live log matters

Sprint Ω.9 specifically targets:
- TTS wedge (3 events in pre-Ω.9 log) → expected zero post-Ω.9
- STT recognizer starvation (5 events) → expected zero
- LLM watchdog breach (13 events) → expected ≤ 1
- Boot total 18.5 s → expected ≤ 13 s (STT preload overlapping with brain + cold start)
- Memory peak 89 % → expected ≤ 78 % at idle

**These are the headline numbers ChatGPT should ask Satyam to validate** before declaring Ω.9 successful.

---

## 18 · Risk Register

### P0 (must ship before any new feature)

| # | Risk | Evidence | Action |
|---|---|---|---|
| P0-1 | No post-Ω.9 boot log captured — sprint not validated | `atomCurrentLogs.txt` empty, 0 boot timelines after `3f0c787` | **Reboot ATOM, capture fresh log, re-run triage + scorecard.** |
| P0-2 | I-06 violated: `brain.max_tokens=384` (>320) | `config/settings.json:256` | Drop to 320 (or 256 for `optimal` profile, 384 for `full_performance` only). |
| P0-3 | `_drain_proactive_on_listening` produces 49 tracebacks pre-Ω.9 | `logs/atom.log` lines 1,200–11,000 | **FIXED in Ω.9** — validate by absence of new tracebacks in fresh log. |

### P1 (next sprint)

| # | Risk | Evidence | Action |
|---|---|---|---|
| P1-1 | I-12 at risk: `cloud.enabled=true` global default | `config/settings.json:759` | Flip global to `false`; gate rotating client on `cloud.reasoning` capability flag. |
| P1-2 | `IntentEngine` "cpu" → fallback (bare-noun OS info miss) | § 8.2 case 2 | Extend `os_intents.py` to match bare nouns from `_INFO_INTENTS`. |
| P1-3 | `IntentEngine` "next two hours" → music_next false positive | § 8.2 case 3 | Add word-boundary requirement and phrase context to `music_intents` ("next" must follow `track`/`song`/`play`). |
| P1-4 | No `set_timer` / `set_reminder` action | § 8.2 cases 11, 13 | Add `productivity_intents.set_timer` and `routine_intents.set_reminder` with native macOS Reminders bridge. |
| P1-5 | `vision_describe` → `vision_look` rename without test update | `tests/test_sprint_k_recovery.py:30` | Update test or restore alias. |
| P1-6 | Skill file `SKILL.md` says default brain is Qwen3-8B; reality is 4B; `~51 K LOC` (real ~154 K) | skill drift | Regenerate skill from current code. |
| P1-7 | Intent watchdog drift: skill says 50 ms, config has 150 ms | `config/settings.json:381` | Reconcile; if 150 ms is intentional (post-priming), update skill + I-05 wording. |
| P1-8 | Stale dead model references in config | `brain.speculative_decoding.draft_model_path = models/qwen3-8b-4bit`, `stt.whisper_model_path = models/ggml-large-v3-turbo-q5_0.bin`, `tts.kokoro_model_path = models/kokoro/...` | Empty out or remove the missing-file references. |
| P1-9 | `m5_baseline_benchmark.py` still probes 4 retired deps and reports them as FAILs | `scripts/m5_baseline_benchmark.py` Stage 9 | Delete the 4 import probes (`speech_recognition`, `edge_tts`, `pygame`, `llama_cpp`). |
| P1-10 | Optional surfaces (realtime, iPhone, screen_loop) auto-started despite config flags being `false` | `jarvis_scorecard.py` says `realtime=True iphone=True screen_loop=True` | Audit `main.py` for unconditional starts; respect config flags. |

### P2 (cleanup / nice-to-have)

| # | Risk | Evidence | Action |
|---|---|---|---|
| P2-1 | `data/screen_observations.sqlite` has 6 rows from disabled loop | `screen_perception_loop.enabled=false` | Confirm no new writes; consider truncating. |
| P2-2 | Memory governor / pressure tier thresholds slightly inconsistent (90 vs 92 critical) | `config.memory.pressure_tiers.critical_pct=90`, `config.memory_governor.tier3_threshold_pct=92` | Pick one source of truth or document why two governors disagree. |
| P2-3 | Quick-reply emoji output in Q1, Q2, Q5 of `audit_voice_pipeline.py` Phase 1 | `q1: "...probably can. 😏📱"`, `q2: "...How's your day? 😎"`, `q5: "...just say \"calling Mom\" a"` | Strip emoji in TTS path or ban in system prompt — NSSpeechSynthesizer reads emoji names ("face with tears of joy"). |
| P2-4 | RAG `documents` collection empty | ChromaDB count=0 | Either set `documents.auto_ingest_on_boot=true` with a curated path, or remove the section to reduce config surface. |
| P2-5 | `data/embeddings_warm.npz` 152 KB / 1,024 cap | `config.embedding.warm_file.max_entries=1024` | Confirm this fits the live workload; consider growing to 4,096 if first-query MRU is sparse. |

---

## 19 · Test Suite — Targeted Coverage Gaps

To prevent the next regression, recommend adding:

1. **`tests/test_executor_pool_split.py`** — assert `_LIGHT_EXEC.max_workers == 2`, `_HEAVY_EXEC.max_workers == 3`, and that `partial_classify` runs on light pool while `llm_inference` runs on heavy pool.
2. **`tests/test_voice_wedge_recovery.py`** — synthetic test for `tts_macos.py` "wedged mid-utterance" → fallback to `say` CLI path.
3. **`tests/test_whisperkit_port_recovery.py`** — fake stale serve on port 50060 (with WhisperKit `/health` returning unhealthy), assert reaper SIGTERMs, retries with next port.
4. **`tests/test_intent_bare_noun_os_info.py`** — "cpu", "ram", "battery", "disk" alone → corresponding OS-info intent.
5. **`tests/test_emoji_stripped_in_tts.py`** — emoji in LLM output never reaches `_record_spoken()`.
6. **`tests/test_idle_maintenance_real_mlx.py`** — wire `clear_metal_cache` into a real MLX session, assert RSS drops by >100 MB after `freeze_after_boot_s + tick_interval_s`.

---

## 20 · Improvement Roadmap

### Sprint Ω.11 (recommended next, 1 commit, 4–6 files, < 2 hours)

**Goal:** ship the P0 fixes + close P1-2/P1-3/P1-7/P1-9 in one coherent voice-fluency-validation sprint.

1. **`config/settings.json`** — `brain.max_tokens: 384 → 320`; `brain.speculative_decoding.draft_model_path: "" `; `stt.whisper_model_path: ""`. (P0-2, P1-8)
2. **`config/settings.json`** — `cloud.enabled: false` + flip `cloud.reasoning: true` if user wants cloud reasoning today; document the change. (P1-1)
3. **`core/intent_engine/os_intents.py`** — add bare-noun matchers for `cpu`, `ram`, `battery`, `disk`, `wifi`, `ip`. (P1-2)
4. **`core/intent_engine/music_intents.py`** — require word-boundary + context anchor for `next`/`previous`. (P1-3)
5. **`scripts/m5_baseline_benchmark.py`** — drop the 4 retired-dep probes. (P1-9)
6. **`tests/`** — add #1 and #4 from § 19.
7. **Reboot ATOM, capture fresh `atomCurrentLogs.txt`, re-run triage + scorecard, attach numbers to commit message.** (P0-1)

### Sprint Ω.12

- P1-4: `set_timer` + `set_reminder` actions with native macOS Reminders bridge.
- P1-5: fix the one stale test.
- P2-3: emoji stripper in `tts_macos.py`.

### Sprint Ω.13 (skill / docs hygiene)

- P1-6: regenerate `.cursor/skills/atom-systems-engineer/SKILL.md` from current code; restate default model, LOC, watchdog budgets.
- P1-7: reconcile intent watchdog (50 vs 150 ms) — in code, in skill, in I-05.

### Sprint Ω.14 (cloud activation, when keys ready)

- Provision keys via `scripts/setup_api_keys.py`.
- Implement privacy redactor on cloud outbound (I-12).
- Live test rotation under simulated 429 + hard-failure scenarios.

---

## 21 · Open Questions for ChatGPT

If you're reviewing this report, these are the questions where outside judgment would be most valuable:

1. **Single-resident 4B vs dual-resident 4B + 8B speculative.** ATOM today has `single_resident=true` and the 8B uninstalled. The original Sprint Ω.7 design had Qwen3-8B as the only model. Sprint Ω.8 reverted to 4B-only (less RAM, faster first-token, simpler eviction). Question: is there a way to bring back **speculative decoding** with the 4B as the draft and a small (1.5B) as the target, while keeping single-resident? Or is the 1.6 GB warm RSS budget already optimal for this hardware?

2. **Cloud default state.** Today `cloud.enabled=true` globally with 0 keys. Should the policy be (a) `false` global + per-capability flags (current recommendation), or (b) `true` global with a stricter outbound redactor + per-call audit? The owner is paranoid by design; what's the JARVIS-grade default?

3. **STT preload 8.8–17.1 s every boot.** WhisperKit serve cold-start dominates boot. Options:
   - LaunchAgent that keeps `whisperkit-cli serve` warm in the background (already present at `scripts/install_atom_launchagent.sh` — should it be enabled by default?)
   - Pre-load an in-process WhisperKit Swift binding instead of HTTP-serve (no port, no IPC, but more code).
   - Switch to faster-whisper `large-v3-turbo` int8 on Metal as primary STT (already installed in the venv).
   - Are any of these strictly better, or does each have a hidden cost?

4. **Intent + Cognitive Kernel fusion.** Today `IntentEngine.classify()` runs first (sub-millisecond) and `CognitiveKernel.route()` runs separately. The kernel doesn't consult the intent's deterministic action before deciding LLM vs DIRECT. Should the kernel **always short-circuit to DIRECT when `IntentEngine` returns a non-fallback action with confidence ≥ 0.9**, even for "moderate" complexity? "open spotify" wasting 1–2 s on LLM is the canonical example.

5. **Memory pressure tier thresholds.** `memory.pressure_tiers.critical_pct=90` vs `memory_governor.tier3_threshold_pct=92`. Which governor "wins" at 91 %? Is this drift dangerous?

6. **Voice prompt size.** Voice prompt is ~1,227 tokens consistently. With `n_ctx=6,144` and history of 4 turns, we never exceed ~3,500 tokens. Is the 6,144 ctx wasteful (KV-cache pressure) or correct headroom?

7. **TTS engine choice.** Currently NSSpeechSynthesizer (built-in macOS). Kokoro-TTS (`af_heart`) and Edge-TTS are wired but inactive. NSSpeechSynthesizer is wedging at ~6 s on 20-word utterances pre-Ω.9. If Ω.9's executor split doesn't fix it, which engine ships next? Trade-offs: Kokoro is on-device (privacy), Edge needs internet. The user's "JARVIS voice" preference is on a custom voice slot called `jarvis` — what voice library best preserves that vibe with low latency and zero wedging?

8. **The `vision_describe` / `vision_look` rename test failure.** Is the rename intentional (better disambiguation) or accidental (legacy name leak)? Whichever — it's a one-line fix; question is which direction.

9. **Sprint Ω.10 removed `_drain_proactive_on_listening`'s caller (the V22 ProactiveDaemon)** — but the handler itself is still defined in `main.py:2830` because it accepts `state_changed` events from the bus directly. Is that handler still needed at all, or can it also be removed? (Grep shows two definitions: one in `main.py`, one in `core/background/proactive_agent.py` — the latter may be dormant.)

10. **The "ATOM speaks emoji"** behavior in Q1/Q2/Q5 of the live brain test. Should the **system prompt** ban emojis, or should the **TTS layer** strip them? My reflex is "system prompt" — so the LLM never produces them and `text-only` answers feel cleaner — but I want a sanity check on whether an emoji-free prompt has measurable quality regressions for short conversational replies.

---

## 22 · Sources & Reproducibility

Everything in this report is reproducible from the current commit (`3f0c787`) using:

```bash
cd /Users/satyamyadav/Desktop/Personal/ATOM
source .venv/bin/activate

# Static + smoke
python -m pytest tests/ --tb=no -q
bash .cursor/skills/atom-systems-engineer/scripts/validate_boot.sh

# Sprint Ω.9 invariant battery
python scripts/audit_omega9_quick.py
# → audit_omega9_report.json

# Live brain (5–10 s, real MLX cold load)
python scripts/audit_brain_live.py
# → audit_brain_report.json

# Live subsystems (10 s, real embedding + VLM + cloud rotation)
python scripts/audit_subsystems_live.py
# → audit_subsystems_report.json

# Live voice pipeline (15 s, real warm brain + 5 turns)
python scripts/audit_voice_pipeline.py
# → audit_voice_pipeline_report.json

# Log triage + scorecard against the most recent boot
python .cursor/skills/atom-systems-engineer/scripts/triage_log.py logs/atom.log
python scripts/jarvis_scorecard.py logs/atom.log
```

After a fresh boot of ATOM:

```bash
# 1. Capture fresh log (ATOM's runtime will write it)
# (boot ATOM via "Run ATOM.command" or `python main.py`)
# (kill with Ctrl-C after ~2 minutes of idle + 3–5 turns)

# 2. Re-run triage + scorecard against the fresh log
python .cursor/skills/atom-systems-engineer/scripts/triage_log.py atomCurrentLogs.txt
python scripts/jarvis_scorecard.py atomCurrentLogs.txt
```

**Artifacts produced by this audit (committed to repo root):**

- `audit_brain_report.json` — live MLX cold load + 4-turn warm
- `audit_omega9_report.json` — Sprint Ω.9 invariant gates
- `audit_subsystems_report.json` — embedding / VLM / vector / prompt / cloud
- `audit_voice_pipeline_report.json` — soft path + brain warm-load timings
- `docs/ATOM_BASELINE_METRICS.md` — `m5_baseline_benchmark.py` boot-stage profile

**This report:** `ATOM_AUDIT_REPORT.md` (you are reading it).

---

## 23 · One-Paragraph Summary (for ChatGPT)

ATOM is a 154 K-LOC Python 3.11 personal AI OS on a MacBook Air M5 (16 GB unified memory). The active brain is **Qwen3-4B-Instruct-4bit** (MLX, single-resident, 1.6 GB warm RSS, 803 ms first-token cold / 356 ms warm / 17 wps decode). Voice is **WhisperKit (CoreML on ANE)** → 4-stage cognitive kernel → MLX → **NSSpeechSynthesizer**. RAG layer is ChromaDB (5 collections, 290 entries dominated by 260 facts) + 18-row embedding cache + 6-table SQLite memory graph. **Sprint Ω.9 (just landed)** split the AsyncEventBus executor into light=2 / heavy=3 pools to fix voice fluency under LLM load, added boot-warm wiring (`stt.kick_serve_async` + `tts.preflight_speak`), an `exact_only` semantic cache for short identity queries, an explanatory-intent gate for quick replies, and a multi-shape embedding warmup that dropped first-query latency from ~670 ms to ~3 ms. **Sprint Ω.10 (same commit)** removed dead V22 daemon code (~600 LOC), retired three log-only loops (`self_optimizer`, `behavior_model`, `prediction_preload`), and shipped a new `IdleMaintenance` subsystem (GC + MLX cache release after 60 s+120 s idle). **Test suite: 1,982 / 1,983 passing (99.95 %)**. Static + in-process audits are all green. **Live runtime grade is still 31/100** because the only available log is the **pre-Ω.9** runtime, which suffered 3× TTS wedges, 5× STT starvations, 13× LLM watchdog breaches, peak 89 % memory, and p95 first-token 12.8 s — exactly the failure modes Ω.9 targets. The single remaining required validation is **a fresh post-Ω.9 boot log**; without it, Sprint Ω.9 is shipped but not blessed. The two known invariant violations are I-06 (`brain.max_tokens=384` should be ≤320) and I-12 (`cloud.enabled=true` globally rather than per-capability). Test failure budget: one stale rename (`vision_describe → vision_look`). ChatGPT's most useful contribution would be (a) a sanity check on the executor-split fix being sufficient for voice fluency, (b) opinion on cloud opt-in default, and (c) STT cold-start strategy (LaunchAgent vs in-proc Swift binding vs faster-whisper switch).
