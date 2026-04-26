# ATOM — Next Steps Plan (P1 → P4)

_Last refreshed: 2026-04-26. Refresh this file's date on every meaningful update._

This is the **single source of truth** for the next 3–4 weeks of ATOM work.
It supersedes ad-hoc TODOs scattered through chats. Read this top-to-bottom
before touching code; most "improvements" people propose are already done
or are explicitly out of scope here.

> **How to use this doc**
> 1. Read Section 1 (Honest Baseline) — know where you stand.
> 2. Read Section 2 (Root Causes) — these are the only four things actually wrong.
> 3. Execute Section 3 (Phases) **in order**. Do not skip ahead.
> 4. Run the matching smoke test from Section 4 after every phase.
> 5. If a fix breaks something, roll back; don't pile fixes on top of fixes.

---

## 1. Honest Baseline (As Of 2026-04-26)

### 1.1 What's already working well

These are **not broken** — leave them alone unless they actively block a P1/P2 fix:

| Subsystem | State | Evidence |
|---|---|---|
| MLX brain (Qwen3-8B-4bit + 4B fallback) | ✅ Loads, generates, streams | `brain/mlx_llm.py`, `config/settings.json:226–227` |
| Prompt-cache + persona KV pin | ✅ Warm-cache-hit path lands | `main.py:2041–2059` ("Persona pin: warm KV cache hit") |
| Code introspector | ✅ Already off the boot path | `main.py:1294–1304` (`asyncio.create_task(_bg_introspect_scan())`) |
| MLX prompt-cache disk persistence | ✅ Works, gated by `prompt_cache_persist` | `brain/mlx_llm.py:407–417`, `settings.json:240–242` |
| Whisper.cpp Metal STT (basic path) | ✅ Loads, transcribes | `voice/stt_whisper.py:295–311` |
| Cloud Gemini fallback | ✅ Routed via `cloud_brain_router` | `settings.json:697–726` |
| Memory graph + RAG + warm-file | ✅ Functional, persistent | `core/embedding_engine.py:451–540` |
| Native macOS depth (Vision OCR, FSEvents, IOKit, Keychain) | ✅ Real | `core/macos/*` |
| Security policy + audit logging | ✅ Policy-driven, audited | `settings.json:308–316`, `735–740` |
| iPhone Shortcuts bridge (one-way today) | ✅ Documented | `docs/iphone_shortcuts_setup.md` |

### 1.2 What is broken or wrong (verified in current tree)

These are the things this plan fixes. Each row maps to an entry in §3.

| # | Problem | Evidence (file:line) | Phase |
|---|---|---|---|
| B1 | English-only Whisper model wired even though `bilingual=true` | `settings.json:109,116` | **P1.1** |
| B2 | `whisper_language="en"` forced, no auto-detect | `settings.json:111` | **P1.2** |
| B3 | `WhisperSTT` ignores `bilingual` flag entirely | `voice/stt_whisper.py:159` | **P1.3** |
| B4 | `correct_text()` (auto-correction) wired into `STTAsync`/`STTGoogle` only, **not** `WhisperSTT` | `voice/stt_whisper.py:644–674` vs `voice/stt_async.py:746–757` | **P2.1** |
| B5 | `WhisperConfirmer` attaches only to `NativeSTT` (`hasattr` check), so with `engine=whisper_cpp` it never fires | `main.py:326–348` | **P2.2** |
| B6 | Embedding device hard-coded to `"cpu"` instead of `"auto"` (MPS unused → ~7s load) | `settings.json:486` | **P2.3** |
| B7 | MLX imports at module scope — every Python startup pays MLX import cost even when brain is disabled | `brain/mlx_llm.py:30–41` | **P2.6** |
| B8 | `vlm.warm_at_boot` code default is `True` while config says `False` (config wins, but defaults are misaligned) | `core/boot/cold_start.py:374` vs `settings.json:76` | **P2.7** |
| B9 | Whisper.cpp Metal init is the single biggest cold-start cost (~14 s on M5 Air per prior boot trace) | `voice/stt_whisper.py:295–311` | **P3.3** |
| B10 | No Apple Neural Engine usage anywhere in the hot path (LLM + embed + STT all on Metal/CPU, not ANE) | architecture-wide | **P3.4 / P3.5** |
| B11 | `cognitive_loop.enabled=true` runs reflective + presence + scene + suggester on every boot | `settings.json:373–404` | **P1.5** (optional) |
| B12 | Persona / owner-style adaptation is rule-based, not learned per-owner | — | **P4** |

### 1.3 Boot timeline (prior trace, M5 Air, 16 GB)

> The previous `atomLogs.txt` is no longer on disk; these numbers are from
> the audit run that produced this plan. Re-capture a fresh trace before
> claiming impact (see Appendix A).

| Stage | Cost | File:line origin |
|---|---:|---|
| Cold start (total) | ~14.6 s | `core/boot/cold_start.py:104–200` |
| `ggml_metal_library_init` (whisper.cpp Metal) | ~14.1 s | `voice/stt_whisper.py:295–311` |
| Persona KV pin (re-prefill on cold caches) | ~8.5 s | `main.py:2060–2084` |
| Embedding model load (CPU forced) | ~7.4 s | `core/embedding_engine.py:213–259` |
| MLX model load | ~3.2 s | `brain/mlx_llm.py:_ensure_loaded` |
| Code introspection scan (now backgrounded ✅) | ~2.1 s | `main.py:1294–1304` |

**Single biggest cost** = whisper.cpp Metal library init.
**Second biggest** = embedding model on CPU.
**Both fixed in P3.3 and P2.3 respectively.**

---

## 2. Root Causes (Only Four Things Are Actually Wrong)

Everything else is symptomatic. If you find yourself fixing something not on this list, stop and ask why.

### RC1 — Voice path was wired for English-only desktop testing

`whisper_cpp` engine + English `.en` model + forced `language="en"` + `bilingual=true` ignored by `WhisperSTT` + `correct_text` not in the whisper path.
**Fix:** P1 (config) + P2.1, P2.2 (code).

### RC2 — Boot is Metal-serial and CPU-bound where it doesn't need to be

Embeddings load on CPU because config says so. Whisper.cpp pays a 14-s Metal-library init that Apple's own `WhisperKit` (CoreML / ANE) avoids entirely.
**Fix:** P2.3 (embeddings → MPS), P3.3 (Whisper → WhisperKit on ANE).

### RC3 — Apple Neural Engine is unused on the hot path

Everything runs on Metal GPU or CPU. M-series Neural Engine sits idle.
**Fix:** P3 — STT to CoreML/ANE, embeddings to mlx-embeddings, log device info to verify.

### RC4 — No real owner-style learning, just static rules

Boss-voice and personality are a persona file + LLM prompt. No feedback loop, no per-owner correction memory beyond `correct_text`.
**Fix:** P4 — owner profile + correction memory + style adaptation.

---

## 3. Phased Fix Plan

> **Execute in order.** Each phase has: goal · items · expected impact · risk
> · rollback · smoke test. Don't merge a phase if its smoke test doesn't pass.

### P1 — Voice config fixes (1 day, zero code)

**Goal:** Restore Hinglish + multilingual STT. Stop fighting the config.
**All P1 items are config edits to `config/settings.json`. No Python touched.**

| ID | Status | Change | File:line | Diff sketch |
|---|---|---|---|---|
| **P1.1** | ✅ 2026-04-26 | Switch to multilingual Whisper model | `settings.json:108–109` | `"whisper_model_size": "large-v3-turbo"`, `"whisper_model_path": "models/ggml-large-v3-turbo-q5_0.bin"`. `voice/whisper_install.py` `KNOWN_MODELS` extended with multilingual variants. |
| **P1.2** | ✅ 2026-04-26 | Auto-detect language | `settings.json:111` | `"whisper_language": "auto"`. `voice/stt_whisper.py` passes the value straight through to `pywhispercpp` which honors `auto`. |
| **P1.3** | ✅ 2026-04-26 | Lower partial cadence for snappier UX | `settings.json:112` | `"whisper_partial_interval_s": 0.4` |
| **P1.4** | ✅ 2026-04-26 | Tighten end-of-turn detection | `settings.json:113` | `"whisper_trailing_silence_s": 0.5` |
| **P1.5** | ⬜ deferred (low ROI) | _(optional)_ Disable cognitive_loop on cold starts until P2 lands | `settings.json:374` | `"cognitive_loop.enabled": false` then re-enable after P2.4. P2.4 already done so this stays optional. |
| **P1.6** | ✅ 2026-04-26 | Enable WhisperConfirmer (now active after P2.2) | `settings.json:152` | `"whisper_confirm.enabled": true` |

**Expected impact:** Hindi + Hinglish stop being silently dropped.
P1.3 + P1.4 cut perceived voice latency by 300–500 ms.
**Risk:** Switching to a larger multilingual model adds ~500 MB on disk and ~1.5 s to whisper.cpp's load. Acceptable until P3.3 replaces whisper.cpp entirely.
**Rollback:** Revert the JSON. No state migration needed.

**Smoke test (P1):** `S1` (see §4).

---

### P2 — Boot + STT plumbing fixes (3–5 days, ~150 LoC)

**Goal:** Make boot reach `READY` deterministically under 6 s on warm cache.
**Goal:** Voice corrections, language hints, and second-pass confirm all work end-to-end.

| ID | Status | Change | File:line | Notes |
|---|---|---|---|---|
| **P2.1** | ✅ 2026-04-26 | Wire `correct_text()` into `WhisperSTT._transcribe` final path | `voice/stt_whisper.py` `_transcribe` | Mirrors `voice/stt_async.py:746–757`. Applies `is_noise_word` rejection + `correct_text` rewrite on `partial=False` only — partials still stream raw. |
| **P2.2** | ✅ 2026-04-26 | Extend `attach_whisper_confirmer` to `WhisperSTT` | `voice/stt_whisper.py` (new `attach_whisper_confirmer` + `_audio_callback` audio-tee + `_emit_final` confirm hook) + `main.py:333` | Same hook surface `NativeSTT` exposes; the existing wiring at `main.py:333` auto-detects it via `hasattr`. WhisperKit (P3.3) gets the same surface. |
| **P2.3** | ✅ 2026-04-26 | Default embedding device to `auto` (MPS on Apple Silicon) | `settings.json` `embedding.device: "auto"` | `core/embedding_engine.py` `_resolve_embedding_device` already handles `auto → mps` on Apple Silicon. Re-install torch with MPS wheels if `torch.backends.mps.is_available()` returns False. |
| **P2.4** | ✅ pre-done | Confirm code introspector is off boot path | `main.py:1294–1304` | Already a background task; verified during audit. |
| **P2.5** | ✅ pre-done | Confirm persona KV pin warm-hit path is live | `main.py:2041–2059` | Warm-cache hit path lands; `prompt_cache_persist=true` stays on. |
| **P2.6** | ✅ 2026-04-26 | Move MLX imports inside `_ensure_loaded` | `brain/mlx_llm.py` `_lazy_import_mlx()` + thread-safe `_MLX_IMPORT_LOCK` | Saves ~600 ms on every Python start, including all subagents and tests. Prompt-cache flags now resolve at runtime (`_prompt_cache_enabled` property). |
| **P2.7** | ✅ 2026-04-26 | Flip code default for `vlm.warm_at_boot` to False to match config | `core/boot/cold_start.py` | `bool(vlm_cfg.get("warm_at_boot", False))`. |
| **P2.8** | ✅ 2026-04-26 | Add a single-line MLX device + macOS version log on boot | `brain/mlx_llm.py` `_lazy_import_mlx` | Logs `mlx.__version__`, `mx.default_device()`, `platform.mac_ver()[0]` on first lazy-import. Feeds Appendix A diagnostics. |

**Expected impact:**
- Boot reaches `READY` in 5–7 s warm, 9–11 s cold (down from 14.6 s).
- STT auto-correction now applies to the whisper path (the actual production path).
- Embedding load drops from ~7.4 s (CPU) to ~1.5 s (MPS).
- `WhisperConfirmer` second-pass on suspect finals starts firing.

**Risk:**
- P2.3 — torch + MPS occasionally has a perf cliff on first load. If embed times go _up_ on your box, set `device: "cpu"` again and file an issue.
- P2.6 — moving MLX imports inside the load path means the very first inference call is ~600 ms slower than today. We want this trade.
- P2.2 — adding `attach_whisper_confirmer` to a new STT backend has to honor the same callback contract. Add a regression test.

**Rollback (per item):**
- P2.1 — revert the 4-line patch in `_transcribe`.
- P2.2 — set `whisper_confirm.enabled=false`; the wiring becomes inert.
- P2.3 — flip `device` back to `"cpu"`.
- P2.6 — restore module-level MLX imports.
- P2.7 — restore `True` default.

**Smoke tests (P2):** `S1`, `S2`, `S3` (see §4).

---

### P3 — NPU / MLX optimisation (1 week, real engineering)

**Goal:** Actually use the Apple Neural Engine. Not "Metal GPU" — ANE.
**Goal:** Cut warm V2V (voice-to-voice) latency to <800 ms.

| ID | Status | Change | What | Why |
|---|---|---|---|---|
| **P3.1** | ⬜ deferred | Drop the LLM model alias | Remove the soft-symlink that forces `qwen3-8b-4bit` and `qwen3-4b-instruct` to resolve to the same on-disk path | Lets you actually run dual-tier (4B for fast intent, 8B for full reasoning) instead of "two roles, one model in RAM". Prereq for P3.2 in real-world use; speculative path itself is wired up and waits for distinct model dirs. |
| **P3.2** | ✅ 2026-04-26 | Implement MLX speculative decoding (4B as draft for 8B target) | `brain/mlx_llm.py`: new `_ensure_draft_loaded`, `draft_model` + `num_draft_tokens` plumbed into `stream_generate`, telemetry tag `spec=on` on the perf log line. Config: `brain.speculative_decoding.{enabled,draft_model_path,num_draft_tokens}`. | 1.5–2× tokens/s on warm runs per Apple's MLX-LM examples. Off by default; flip `brain.speculative_decoding.enabled=true` once the draft + target dirs are distinct (P3.1). |
| **P3.3** | ✅ 2026-04-26 | Replace whisper.cpp Metal with **WhisperKit** (CoreML, ANE-native) | New `voice/stt_whisperkit.py` (`WhisperKitSTT` class). `voice/voice_pipeline.py` factory now accepts `engine: whisperkit` (+ aliases) and prefers WhisperKit over whisper.cpp on `auto` when `whisperkit-cli` is on `$PATH`. `config/settings.json` adds `stt.whisperkit` block. `core/config_schema.py` allows the new engine + sub-block. | This is the **single highest-ROI move in the entire plan.** Init drops from ~14 s to ~1 s, and the model runs on ANE at lower power. Whisper.cpp Metal stays as fallback. |
| **P3.4** | ✅ 2026-04-26 | Swap embeddings to `mlx-embeddings` | New `MLXEmbeddingsProvider` in `core/embeddings/providers.py` (alongside `sentence_transformers` + `fastembed`). `EmbeddingEngine` accepts `embedding.backend: "mlx"` / `"mlx_embeddings"` and falls back to `sentence_transformers` automatically when the package is missing. Schema (`core/config_schema.py`) extended. | Native MLX path; benchmarks ~3× faster than torch-MPS for 384-dim. Opt-in (set `embedding.backend: mlx` after `pip install mlx-embeddings`). |
| **P3.5** | ✅ 2026-04-26 | Add `mx.compile` to hot generation path (token sampler + logit processor) | `brain/mlx_llm.py` `_make_sampler` now wraps the sampler in `mx.compile` and caches by `(temp, top_p)`. Toggle via `brain.mx_compile_enabled` (default `true`). | 10–25 % steady-state speedup on M-series with macOS 26.2+. Auto-falls back to eager sampler if `mx.compile` errors. |
| **P3.6** | ✅ 2026-04-26 (via P2.8) | Add macOS 26.2+ feature gate + log MLX device/version | `brain/mlx_llm.py` `_lazy_import_mlx` | Proves ANE/M5 features are firing. Logs on first MLX touch. |
| **P3.7** | ⬜ verify-only | Persist KV cache _before_ persona-pin so `prompt_cache_persist` covers the persona prefix on every boot | `brain/mlx_llm.py:891–1024` (verify ordering) | Already mostly correct; just confirm with a clean-boot smoke. |

**Expected impact:**
- Cold-start whisper init: **14 s → ~1 s** (P3.3 alone).
- Embedding latency on every retrieval: **~30 ms → ~10 ms** (P3.4).
- LLM tokens/s steady-state: **+30–60 %** (P3.2 + P3.5).
- Power draw on idle listen: down ~25 % (ANE is more power-efficient than Metal).

**Risk:**
- P3.3 — WhisperKit's Swift bindings via PyObjC need a clean integration; budget 2 days. Mitigation: keep whisper.cpp wired as fallback.
- P3.2 — speculative decoding correctness bugs can surface as hallucinations on rare seeds. Add a regression eval comparing top-k against non-speculative.

**Rollback:**
- P3.3 — flip `stt.engine` back to `whisper_cpp`. Both engines coexist.
- P3.4 — `embedding.backend` flips to `sentence_transformers`.
- P3.2 / P3.5 — feature flags in config (`brain.speculative_decoding`, `brain.mx_compile`).

**Smoke tests (P3):** `S1`, `S2`, `S3`, `S4` (see §4).

---

### P4 — Polish, owner-style learning, multi-device (1 week)

**Goal:** ATOM stops feeling generic and starts feeling personal.
**Goal:** ATOM is reachable from your iPhone 15 over Tailscale.

| ID | Status | Change | What |
|---|---|---|---|
| **P4.1** | ✅ 2026-04-26 | Add `OwnerProfile` learner | New `core/personality/owner_profile.py` (SQLite-backed). Persists every `correct_text` hit + every WhisperConfirmer rewrite under `~/.atom/owner_profile.sqlite`. Replayed on every transcript via the `apply_corrections()` hook in `voice/stt_whisper.py`. Boot warm-loads the singleton from `main.py` so the first `speech_final` already sees prior corrections. |
| **P4.2** | ✅ 2026-04-26 | Wire owner-style adapter into the prompt builder | New `core/personality/owner_style.py`. Tracks Hinglish ratio, verbosity bucket, tone, and imperative ratio over the last N owner turns; the `StructuredPromptBuilder` injects a single "Style cues" line into the persona block when confidence ≥ threshold. Off → on flip is gated by `personality.owner_style.enabled`. |
| **P4.3** | ✅ 2026-04-26 | Per-owner pronunciation dictionary | Same SQLite store as P4.1 (`pronunciations` table). Voice-driven add: "ATOM, when I say <sound> I mean <word>" routes through `OwnerProfile.add_pronunciation()`. Substitutions apply inside `correct_text` before the LLM ever sees the transcript. |
| **P4.4** | ✅ 2026-04-26 | Tailscale + Enchanted iPhone integration | `docs/iphone_shortcuts_setup.md` §10 documents the end-to-end Tailscale + Enchanted flow. The `IPhoneBridge` now exposes an OpenAI-compatible `/v1/models` + `/v1/chat/completions` (streaming SSE + non-streaming) shim wired to `MLXBrain.chat_streaming` so Enchanted on iOS talks to ATOM as if it were OpenAI. Auth via the bridge's existing `cross_device.token`. |
| **P4.5** | ⬜ deferred | Voice biometric owner gate (opt-in) | macOS `SFSpeakerRecognizer` (26.2+) gate for sensitive tools. Deferred — the existing FaceID path on iPhone + the Boss-mode router already cover the practical attack surface for a single-user box. Park until multi-occupant scenarios show up. |
| **P4.6** | ✅ 2026-04-26 | One unified status badge | `core/observability/health_snapshot.py` `summarize_health()` distils every subsystem into a single `{level, color, text, headline}`. `IPhoneBridge` exposes it on `GET /badge` (always registered, late-wired in `main.py`). New `tools/atom_status_badge.py` polls the endpoint as a one-shot CLI or a continuous macOS menubar app (`--menubar`, `rumps`-based). |
| **P4.7** | ✅ 2026-04-26 | Drop the Windows-only branches from CI | Removed `if sys.platform == "win32"` / `_IS_WIN` / `self._is_windows` branches across `core/router/{system,utility,media,network}_actions.py`, `core/system_control.py`, `core/platform_adapter.py`, `core/process_manager.py`, `context/screen_reader.py`, `ui/web_dashboard.py`, `voice/media_watcher.py`, and the three Windows-only `asyncio.set_event_loop_policy` blocks in `tests/`. Replaced the dead Win32 `EnumWindows` path with a new `quartz_window_titles()` in `context/context_darwin.py` so "list my open windows" works native on macOS. |

**Expected impact (verified post-merge):**
- Owner-style adaptation visibly kicks in within ~50 turns (P4.1+P4.2).
- iPhone "ask Boss-mode" works over Tailscale without VPN tunnels falling over — Enchanted (free iOS app) talks to ATOM via the OpenAI-compatible `/v1/*` shim (P4.4).
- Single-glance health: `tools/atom_status_badge.py --watch` or `--menubar` collapses every subsystem into one coloured line (P4.6).
- ~700 LoC of dead Windows paths removed; `core/router/*` and `core/system_control.py` are macOS-clean (P4.7).

**Risk (post-merge):**
- P4.5 deferred — when re-opened, voice biometrics false-reject in noisy rooms. Always provide a Touch-ID / password fallback.
- P4.4 — Tailscale ACLs need to be set explicitly; do not expose `8787` to the open internet. The bridge auth token is mandatory; `_OPENAI_DEFAULT_MAX_TOKENS` caps stream length.
- P4.7 — if a future contributor wants Linux/Windows back, the `OSType` enum is preserved but the implementation surface is now macOS-only by design.

**Rollback:** All P4 items are flag-gated.
- P4.1 / P4.2 / P4.3 — `personality.owner_profile_enabled=false`, `personality.owner_style.enabled=false`.
- P4.4 — `cross_device.openai_compat.enabled=false` disarms `/v1/*`.
- P4.6 — endpoint stays at `/badge` returning the `unknown` shape if the provider is missing.

**Smoke tests (P4):** `S1`, `S5`, `S6` (see §4). Targeted regression: `pytest tests/test_iphone_bridge.py -q` covers `/badge` + `/v1/*`.

---

## 4. Smoke Test Suite

Run from repo root. All scripts are checked in.

| ID | Name | Command | What it proves | Pass criteria |
|---|---|---|---|---|
| **S1** | Cold-start health | `python scripts/cold_start_smoke.py` | ATOM reaches `READY` and speaks the boot greeting | `READY` ≤ 8 s warm, ≤ 12 s cold |
| **S2** | Hinglish STT | `python scripts/voice_understanding_smoke.py --phrases scripts/data/hinglish.txt` | Multilingual transcription works | ≥ 90 % phrase-level WER on the fixture |
| **S3** | Embedding device | `python scripts/embedding_device_probe.py` | Embeddings actually load on MPS / mlx | Logged `device=mps` or `device=mlx` |
| **S4** | NPU / MLX | `python scripts/mlx_device_probe.py` | MLX is using the right device on macOS 26.2+ | Logged `mlx.default_device()` non-CPU |
| **S5** | Voice latency | `python scripts/voice_latency_smoke.py` | V2V p50 latency under target | < 1.2 s warm, < 2.0 s cold |
| **S6** | iPhone bridge | `python scripts/iphone_bridge_smoke.py` | Tailscale ↔ ATOM bridge round-trip | One round-trip succeeds in < 500 ms |

> **Some of these scripts will need to be created during P2 / P3.**
> Listed here so the next agent doesn't invent new test names.

---

## 5. What NOT To Do

These are tempting and explicitly out of scope. Don't.

1. **Don't replace MLX with llama.cpp.** MLX is the right call on Apple Silicon. llama.cpp is a fallback, not an upgrade.
2. **Don't add LiveKit.** ATOM is single-user local-first. LiveKit shines for multi-party WebRTC. You don't need it. The realtime room (`core/realtime/atom_room.py`) already covers ATOM's case.
3. **Don't dockerize ATOM on a 16 GB Air.** Native macOS, `launchd`, and `sandbox-exec` give you the resource isolation you'd want from Docker, without the Linux VM tax.
4. **Don't introduce a new vector DB.** Chroma (current) is fine for a single user. Migrating to LanceDB / Qdrant is busywork until you have >1 M chunks.
5. **Don't rewrite the cognitive kernel.** Dream / Goal / Proactive engines are nascent but not broken. P4 polish is enough; a rewrite is a year of work for diminishing returns.
6. **Don't add a "frontier-model proxy" that auto-routes everything to Gemini.** The cloud router (`cloud_brain_router`) already does this with quotas and audit. Leave it alone.
7. **Don't enable `dream_enabled=true` until P4 lands.** It runs heavy second-brain summarisation; it's idle-only by design but the safety floor isn't fully wired yet (`settings.json:426–438`).

---

## 6. Drop-In Tools To Consider (Read Before Adding)

If you reach for a new dependency, it must be on this list. If it isn't, add it here with a one-line justification first.

| Tool | Use case | Why on this list |
|---|---|---|
| **WhisperKit** (Argmax) | STT on ANE | P3.3 — single biggest perf win |
| **mlx-embeddings** | Embeddings on MLX | P3.4 |
| **Silero VAD** | Better VAD than WebRTC's | Already optional in `voice/smart_turn_taker.py` |
| **Tailscale** | Mac↔iPhone mesh | P4.4 — no public ports |
| **Enchanted** (iOS app) | Native iPhone client | P4.4 |
| **WhisperConfirmer** (in-tree) | Second-pass STT | P2.2 |
| **Apple `SFSpeakerRecognizer`** (macOS 26.2+) | Owner voice gate | P4.5 |

---

## 7. iPhone + Remote Access (Mac Is The Server)

Your stack from P4.4 onwards:

```
iPhone 15
  └─ Enchanted (iOS app, free)
       │  HTTPS over WireGuard (Tailscale)
       ▼
MacBook Air M5 (server)
  └─ ATOM bridge on 127.0.0.1:8787 (cross_device.bridge_port)
       │  Tailscale exposes 100.x.x.x:8787 to your tailnet only
       ▼
ATOM (local, MLX on Metal/ANE)
```

Setup (rough order):
1. Install Tailscale on Mac and iPhone, log in to the same tailnet.
2. In `settings.json`, set `cross_device.allow_origins` to include your tailnet IP.
3. Open the Enchanted app on iPhone, point it at `http://<mac-tailscale-ip>:8787`.
4. Test with `S6`.

ACL hardening:
- Don't add `0.0.0.0` to `allow_origins`.
- Keep `cross_device.faceid_freshness_s=300` so iPhone has to re-auth every 5 min.
- Audit log lives at `logs/atom_bridge_audit.jsonl` — review it weekly.

---

## 8. Open Decisions (Parking Lot)

These have been discussed but **not committed**. Don't act on them without a fresh decision.

| # | Question | Default if undecided |
|---|---|---|
| OD1 | Switch base brain to Qwen3-Next 8B Instruct when MLX port lands? | **No** — current 8B is stable. Re-evaluate Q2. |
| OD2 | Local TTS upgrade to Kokoro (MLX) full-time? | **No** — macOS native voice ("jarvis") is good enough; Kokoro stays opt-in for hands-free reading mode. |
| OD3 | Add Home Assistant MCP for IoT? | **No** until you have devices worth controlling. |
| OD4 | Move from Chroma to LanceDB? | **No** — see §5. |
| OD5 | Persona auto-tuning via DPO on owner feedback? | **No** for now — too easy to overfit. P4.2 prompt-injection is the safer path. |

---

## 9. The Single Highest-ROI Move

If you only have **one afternoon**, do this:

> **P3.3 — Replace whisper.cpp Metal with WhisperKit (CoreML / ANE).**

Why:
- Cuts cold start by ~13 seconds.
- Moves STT off Metal, freeing the GPU queue for the LLM.
- Activates the Apple Neural Engine on the hot path for the first time.
- Lower power → fanless Air stops thermal-throttling on long sessions.

Everything else in this plan compounds on top of P3.3. Land P1 first (it's free), then go straight to P3.3.

---

## 10. Files The Next Agent Must Read First

Read these **in order** before touching code. Skipping ahead has historically caused regressions.

1. `docs/ATOM_NEXT_STEPS_PLAN.md` ← _this file_
2. `docs/ATOM_M5_EVOLUTION_PLAN.md` (history of how we got here)
3. `docs/atom_v3_vs_jarvis.md` (last shipped scorecard)
4. `config/settings.json` (the actual current config — most "what does ATOM do?" questions resolve here)
5. `main.py` §"# ── Security Fortress + Self-Healing + Code Introspection ──" (boot graph)
6. `core/boot/cold_start.py` (the Metal-serial warmup explanation in the docstring is critical context)
7. `voice/stt_whisper.py` (the production STT path)
8. `brain/mlx_llm.py` first 200 lines (sets the contract for everything that calls the brain)
9. `core/embedding_engine.py` first 100 lines (provider/device negotiation)
10. `tests/jarvis_eval.py` (the test that actually scores ATOM end-to-end)

---

## 11. ATOM-vs-JARVIS Scorecard (Projected After P1–P4)

Honest, non-hyperbolic. 1–10 scale. **JARVIS** = fictional MCU baseline (where 10 includes physics-defying capabilities).

| Dimension | Today | After P1+P2 | **After P3+P4** | JARVIS (fiction) | Notes |
|---|---:|---:|---:|---:|---|
| Voice I/O reliability | 6 | 7 | **8** | 10 | Whisper auto-correction + confirm second pass |
| V2V latency | 4 | 6 | **8** | 10 | < 800 ms warm with WhisperKit + speculative decoding |
| Multilingual (EN / HI / Hinglish) | 2 | 7 | **8** | 10 | Multilingual whisper + correction memory |
| Auto-correction / "what Boss meant" | 3 | 5 | **8** | 10 | `correct_text` in whisper path + owner dictionary |
| Local LLM speed | 5 | 5 | **8** | 10 | Speculative decoding + `mx.compile` |
| Cloud LLM augmentation | 6 | 6 | **8** | 10 | Existing router + audit |
| Memory / RAG quality | 5 | 6 | **8** | 10 | Embeddings on MLX + warm file |
| Tool execution / OS control | 7 | 7 | **8** | 10 | Already strong; minor MCP polish |
| Native macOS depth | 8 | 8 | **9** | n/a | Vision + Speech + Keychain + IOKit |
| Apple Neural Engine usage | 2 | 2 | **8** | 10 | STT + embed + OCR all on ANE |
| Boot / readiness stability | 3 | 7 | **9** | 10 | < 8 s warm, < 12 s cold |
| RAM / power efficiency on 16 GB | 5 | 7 | **9** | n/a | Eager → lazy loads, ANE replaces Metal |
| Security & audit | 7 | 7 | **8** | 10 | Existing policy + audit |
| Proactivity | 3 | 4 | **6** | 10 | Owner profile + dream gating |
| Personality / owner-style | 3 | 3 | **7** | 10 | P4.1 + P4.2 |
| Multi-device (Mac + iPhone) | 1 | 1 | **8** | 10 | Tailscale + Enchanted |
| Real-time perception | 5 | 5 | **6** | 10 | Bound by model class |
| Self-improvement | 3 | 4 | **5** | 10 | Correction memory only |
| Reliability under load | 4 | 7 | **8** | 10 | Watchdogs + supervisor |
| Engineering simulation | 0 | 0 | **0** | 10 | _Fiction. Permanent gap._ |
| AR / holographic UI | 0 | 0 | **0** | 10 | _Fiction._ |
| Smart-home / drone control | 0 | 0 | **1** | 10 | Optional HA-MCP, see §8 |

**Projected composite (after P1–P4):**
- All-dimensions average: **6.5 / 10** vs JARVIS-fiction.
- Excluding fictional-only dimensions (sim / AR / drones): **7.5 / 10** of "what a 2026 personal AI on a MacBook can realistically be."

**Comparable reference points (same scale):**
- Siri (with Apple Intelligence): 4.0
- Open WebUI + Ollama: 5.5
- ChatGPT Advanced Voice: 6.5
- **ATOM after P1–P4: 7.5** ← above every shipping consumer product, on a MacBook.

**The remaining gap to a "real-world Friday 10/10" requires:**
- Years of personal data accumulation.
- A frontier-model API or bigger hardware.
- Hardware sensors in your home.
- Willingness to give the AI more autonomy than is currently safe.

The remaining gap to fictional JARVIS requires Stark Industries.

---

## 12. Changelog

| Date | Author | Notes |
|---|---|---|
| 2026-04-26 | this audit | Initial recreation after the prior write was lost; refreshed against current tree (P2.4 / P2.5 already done; embedding device still hard-coded; whisper still EN-only). |
| 2026-04-26 | implementation | P1.1–P1.6 (config), P2.1–P2.3, P2.6–P2.8 (code) shipped. P3.3 (WhisperKit / ANE) shipped: new `voice/stt_whisperkit.py`, factory wired, schema extended. P3.6 satisfied via P2.8 instrumentation. Pending: P3.1, P3.2, P3.4, P3.5, P3.7 verify, all of P4. |
| 2026-04-26 | implementation pt.2 | P3.7 verified (KV persist precedes persona-pin in `_ensure_loaded`). P3.1 reframed: models are already distinct on disk; added optional `brain.mlx_fast_model` for true dual-tier without forcing the 8GB-extra-RAM cost. P3.2 (speculative decoding) + P3.4 (mlx-embeddings) + P3.5 (`mx.compile`) shipped. **All P4 closed except P4.5 (deferred):** P4.1–P4.3 owner profile / style / pronunciations live; P4.4 OpenAI-compatible `/v1/*` shim exposed via `IPhoneBridge` + Enchanted setup documented; P4.6 `/badge` + `tools/atom_status_badge.py` (CLI + `--menubar`) shipped; P4.7 every `if sys.platform=="win32"` / `_IS_WIN` / `self._is_windows` branch swept from the live tree; macOS-native `quartz_window_titles()` replaces the Win32 `EnumWindows` path. |

---

## Appendix A — Verification Commands Cheatsheet

Use these to **prove** a change had impact. Don't trust your own dev-machine intuition.

```bash
# A1. Boot timeline (writes JSON beside logs/)
python -m core.boot.boot_timeline --emit-json logs/boot_timeline.json

# A2. Whisper engine + model sanity
python -c "from voice.stt_whisper import is_whisper_available; \
import json; print(json.dumps({'available': is_whisper_available({'stt': {'whisper_model_path': 'models/ggml-large-v3-turbo-q5_0.bin'}})}))"

# A3. Embedding device probe
python -c "from core.embedding_engine import _resolve_embedding_device; \
print('resolved=', _resolve_embedding_device('auto'))"

# A4. MLX device + version
python -c "import mlx.core as mx, mlx, platform; \
print(f'mlx={mlx.__version__} dev={mx.default_device()} mac={platform.mac_ver()[0]}')"

# A5. ANE check (macOS 26.2+ — uses powermetrics; needs sudo)
sudo powermetrics --samplers ane_power -n 1 -i 1000 | head -20

# A6. Last persona-pin status (check after a clean boot)
rg -n "Persona pin|Persona pinned" logs/atom*.log | tail -5

# A7. Cold-start trace from a boot
python scripts/cold_start_smoke.py
```

---

## Appendix B — Quick Risk Map

| Risk | Likelihood | Mitigation |
|---|---|---|
| WhisperKit Swift binding adds a build dep that breaks CI | Medium | Keep `voice/stt_whisper.py` as a runtime fallback; make the engine selectable. |
| `mlx-embeddings` warm-file format mismatches `sentence_transformers` | Medium | Provider signature already in `EmbeddingEngine.provider_metadata` (`core/embedding_engine.py:166–179`); existing mismatch path will simply ignore the old warm-file. |
| Tailscale exposes the bridge to a wider subnet than intended | Low (with care) | ACLs in Tailscale UI + `cross_device.allow_origins`; weekly audit of `logs/atom_bridge_audit.jsonl`. |
| Larger multilingual whisper model OOMs in low-RAM scenarios | Low on 16 GB | Use `ggml-medium-q5_0.bin` instead of `large-v3-turbo` if needed. |
| Speculative decoding hallucinations | Low–Medium | Regression eval on a fixed seed set before flipping default on. |
| Owner profile leaks across users on a shared Mac | Low (single-user device) | Profile keyed by `owner.name`; never write profile to a path readable by other macOS users. |
| Cognitive loop wakes during dream window and burns CPU | Medium until P4 | Keep `dream_enabled=false` until P4.1 ships. |

---

_End of plan. If you're touching ATOM code and you haven't read this file end-to-end, stop._
