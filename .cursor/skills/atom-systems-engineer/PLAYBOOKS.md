# ATOM Playbooks — Symptom → Root Cause → Fix

Every recipe here has shipped. Each one cost several sessions to pin down. Use them *before* freestyling.

Format per playbook:
- **Symptom** — what you will see in the log or hear from the user
- **Root cause** — the actual bug
- **Fix sites** — every file that must change (omit one and the bug comes back)
- **Verification** — the shortest thing to check

---

## PB-01 · Prompt text leaks verbatim into TTS

**Symptom**
```
TTS stream slice: 'the final answer only.'
TTS stream slice: 'if the question is a simple, short, or info query, give one short sentence when possible, two short '
```
Model is parroting its own system-prompt rules.

**Root cause** System prompt contained imperative, quotable sentences. The LLM learns to echo them when instruction-following is strong (Phi-3.5 / Qwen-8B both do this with long contract-style prompts).

**Fix sites**
1. `cursor_bridge/structured_prompt_builder.py` — rewrite imperative rules as **negative noun phrases** (e.g. "no preface, no rules quoted, no speaker labels") instead of "Give only the final answer. One short line."
2. `cursor_bridge/structured_prompt_builder.py` — remove per-turn rule text from the query layer. Rules belong only in the system layer, and only in a shape the model cannot plausibly quote.
3. `brain/mlx_llm.py` — keep `_PROMPT_LEAK_FINGERPRINT_RE` catching the old phrases as a safety net.
4. `cursor_bridge/local_brain_controller.py` — same fingerprint check in `_sanitize_response()`.
5. `voice/tts_macos.py` — final defense: fingerprint match → drop utterance + log warning.

**Verification** Grep the latest `atomlogs.txt` for `the final answer only` and `if the question is a simple`. Zero hits = shipped.

---

## PB-02 · Chain-of-thought preface leaks

**Symptom**
```
TTS: "Okay, let's see. The user is asking what time it is..."
TTS: "So, the user said they want..."
```

**Root cause** Reasoning models emit a CoT preamble before the final answer. Our sanitizer must strip it before streaming to TTS.

**Fix sites**
1. `brain/mlx_llm.py` — `_COT_PREFACE_RE` and `_COT_PREFACE_STRIP_RE` must match:
   - `Okay,? (let's|let me) (see|think)...`
   - `So,? the user is (asking|saying)...`
   - `Hmm,? (let me|I should)...`
   - `Alright,? (let's|I will)...`
2. `cursor_bridge/local_brain_controller.py` — `_INSTRUCTION_ECHO_RE` catches quoted user-text prefixes (`"the user is asking"` anywhere in response).
3. Strip special tokens: `<|endoftext|>`, `<|im_end|>`, `Human:`, `Assistant:`, `User:`, `System:`.

**Verification** Ask ATOM a trivial question ("what time is it?"). TTS output should be exactly the answer — no preface. If you see any "Okay" or "So" at the start of any response, a regex is missing.

---

## PB-03 · ATOM speaks by itself / self-echo feedback loop

**Symptom**
```
STT: partial stable for 1.9s — promoting to final: 'What do you mean there boss what's up'
Perception: emotion=frustrated ... 'You are not taking my input correctly'
```
ATOM hears its own voice, transcribes it, routes it as a user query, responds to itself, and the loop repeats — often with rising frustration detected because the "user" keeps complaining.

**Root cause** The echo guard was not consulted by every finalization path. SFSpeechRecognizer promotes stable partials whether or not ATOM is speaking.

**Fix sites**
1. `voice/tts_macos.py` — `_record_spoken()` pushes every utterance into a ring buffer with a timestamp. `is_echo(text, now)` returns True if `text` overlaps any ring entry within N seconds.
2. `voice/stt_macos.py` — **all three** finalization paths must call `self._is_self_echo()` before promoting:
   - `on_stable_partial_promote()`
   - `on_final()`
   - `on_interrupt_predicted()` (this one was the regression)
3. `voice/interrupt_handler.py` — `handle_partial()` checks `tts.is_echo(partial)` before triggering barge-in.
4. `voice/wiring.py` — subscribes the controller + echo buffer to `tts_start` and `tts_chunk_spoken` events.
5. `core/router/router.py` — last line of defense: `attach_tts_echo_guard(tts)` drops incoming finals matching recent TTS.

**Verification** In the log, every TTS utterance should be followed by `STT: self-echo detected on stable partial — promotion suppressed` entries. Zero `promoting to final` should match a recent TTS utterance within 5 s.

---

## PB-04 · STT mishears wake word as "adam" / drops utterances silently

**Symptom** User speaks "Atom, what's the weather" — no response. Log shows:
```
STT partial: 'Adam, what's the weather'
(nothing routed — utterance dropped silently)
```

**Root cause** STT locale `en-IN` (India) consistently hears "Atom" as "Adam". Wake word filter rejected anything not exactly matching "atom".

**Fix sites**
1. `config/settings.json` → `stt.locale: "en-US"` (best pan-accent baseline).
2. `voice/listening_modes.py` → `WakeWordFilter.WAKE_PHRASES` includes common mishearings: `adam`, `atum`, `atom`, `adham`, `atomic`. Use **word-boundary regex** (`\b(atom|adam|atum)\b`) not substring.
3. `voice/listening_modes.py` → add **direct-address phrases** (`are you there`, `hey boss`, `you listening`) that count as wake events.
4. `voice/listening_modes.py` → wake-hint diagnostic: if ≥ 3 finals are suppressed in 30 s, emit a log warning so we catch regressions early.

**Verification** With `voice.activation_mode: "always_on"`, wake phrases are bypassed entirely — most common way to ship this is to enable always-on and let the router filter non-wake traffic by intent. Confirm with log line `Voice loop: Jarvis always-on mode`.

---

## PB-05 · Guardrail cascade → "I lost that answer, Boss."

**Symptom** User asks a real question, ATOM answers "I lost that answer, Boss." or similar clarifier. Log shows:
```
router: guardrail rewrite triggered
local_brain: response rejected as low-quality (length=0)
llm_error: strict recovery empty
```

**Root cause** Three independent guards fire in sequence and compound the failure:
1. Router rewrites "I will check that..." to a clarifier.
2. Controller rejects the rewrite as low-quality because it's short.
3. Strict recovery returns empty because the slot was already consumed.

**Fix sites**
1. `core/router/router.py` — **decouple guardrail rewrite from quality rejection**. A rewritten response is still a valid response. Don't re-gate it through quality.
2. `cursor_bridge/local_brain_controller.py` — try lighter repair before strict recovery. Preserve `pending_reprompt` slot so the original user utterance survives.
3. `cursor_bridge/local_brain_controller.py` — never emit `llm_error` on empty strict-recovery retry. Fall through to the cognitive-kernel quick_reply path instead.

**Verification** Ask ATOM an ambiguous question ("tell me about it"). Should get a natural clarifier like "Tell me what, Boss?" — never "I lost that answer".

---

## PB-06 · Intent engine 50 ms budget violation at boot

**Symptom**
```
WARNING | Runtime budget exceeded: intent_engine (limit=0.050s) | query=hey what time is it
```
First query after boot takes 60–120 ms in the classifier, trips the watchdog, falls back to LLM even for trivial intents.

**Root cause** Python regex compilation happens lazily on first match. Cold start → every intent class compiles its patterns on the first query.

**Fix sites**
1. `core/boot/cold_start.py` — `_prime_intent_engine_regexes()` walks every `core/intent_engine/*.py` module, forces pattern compilation during boot.
2. `core/runtime_watchdog.py` — `intent_engine` gets a **boot grace window** of ~30 s where budget violations are logged as INFO not WARNING.
3. Keep `post_tts_cooldown_ms` at 600 — shorter and PortAudio starves the second mic open.

**Verification** No `intent_engine` WARNING in the log after the first second of boot. If you see one, priming is broken.

---

## PB-07 · TTS doesn't stop after watchdog timeout

**Symptom**
```
WARNING | Runtime budget exceeded: tts (limit=15s)
(TTS keeps speaking for another 20s)
```

**Root cause** Watchdog reset its own timer but never told the synthesizer to stop.

**Fix sites**
1. `core/runtime_watchdog.py` — `attach_tts(tts_instance)` stores a reference. On `tts` budget breach, call `tts_instance.stop()`.
2. `voice/tts_macos.py` — `stop()` calls `NSSpeechSynthesizer.stopSpeaking()` **and** clears the pending stream queue, **and** emits `tts_stopped` event so STT comes back online.

**Verification** Force a long response via a prompt that makes the LLM ramble. At 15 s, TTS must go silent within 200 ms. No dangling speech.

---

## PB-08 · Mic records useless partials during SPEAKING/THINKING

**Symptom**
```
stt_macos: partial: 'the'
stt_macos: partial: 'the final'
stt_macos: partial: 'the final answer'
... (hundreds of these while ATOM is speaking)
```

**Root cause** SFSpeechRecognizer continues emitting partials even when audio feeding is paused. These pollute the log and cost CPU.

**Fix sites**
1. `voice/stt_macos.py` — **state-gate partial emission**: if `state != LISTENING`, drop partials silently. This also prevents accidental barge-in triggers.
2. `voice/stt_macos.py` — **state-gate trivial-final guard**: refuse to promote a final that is non-alphanumeric, single-char, or stale (from a previous listening window).
3. Bump SFSpeechRecognizer recreation thresholds so we don't tear down mid-answer.

**Verification** During a 5-turn conversation, total partial log lines should be ≤ 2× the number of user turns. Hundreds of partials = regression.

---

## PB-09 · Wake word dropped at boot (first turn missed)

**Symptom** User greets ATOM during the startup TTS. ATOM finishes its greeting but never processes the user's reply. Log shows finals being suppressed during the boot TTS.

**Root cause** STT was initialized *during* boot TTS and consumed its own boot audio.

**Fix sites**
1. `voice/voice_pipeline.py` — **defer STT init until boot TTS completes**. Subscribe to `tts_done(source=='boot')` and only then call `stt.start_listening()`.
2. `voice/voice_pipeline.py` — in `always_on` mode, explicitly **bypass passive wake-word gating** — every final is a candidate query.

**Verification** First user utterance after boot greeting should route cleanly. Log shows `STT ready` *after* the boot TTS `stream done`.

---

## PB-10 · System profile missing from LLM prompt

**Symptom** User asks "what's my CPU at?" — ATOM says "I don't have access to your system." Obviously false; we have a whole `SystemScanner`.

**Root cause** `SystemScanner` populated `SystemProfile` but nothing injected that into the prompt.

**Fix sites**
1. `core/system_profile.py` — `get_compact_context()` returns a one-line snapshot: `[MACHINE] macOS 15.0 | Apple M5 | RAM 6.2/16GB | Disk 128/460GB free | health 82/100`.
2. `cursor_bridge/structured_prompt_builder.py` — `set_system_profile_provider(provider)` plumbs the callable into the system layer.
3. `core/router/router.py` — `attach_system_profile_provider(provider)` on init wires scanner → prompt builder.
4. `config/settings.json` → `brain.inject_system_context: true`.

**Verification** In the log, `LLM system prompt (first 120ch)` should include `[MACHINE]`. Without it, ATOM will keep claiming ignorance.

---

## PB-11 · "I don't know what boss means" / context loss across turns

**Symptom** ATOM forgets who "Boss" is mid-conversation, or loses multi-turn context (refers to earlier turn as if it didn't happen).

**Root cause** Either (a) session context layer not attached, (b) RAG failing silently, or (c) `ConversationMemory` not persisting.

**Fix sites**
1. `core/router/router.py` — confirm `Context layer attached to Router (state=True session=True user=True)` in boot log.
2. `brain/memory_graph.py` — confirm `v22 confidence: 0.X for '<query>' (<N> chars)` is logged per turn; if not, memory store write failed.
3. `data/atom_memory.db` — sanity-check it's writable and not corrupted (`sqlite3 data/atom_memory.db ".tables"` returns tables).
4. `core/conversation/conversation_memory.py` — buffered writes must flush on SIGTERM.

**Verification** Two-turn test: "My favorite color is green" → "What did I just tell you?". Second response must contain "green".

---

## PB-12 · Model loaded is not the model in settings.json

**Symptom**
```
INFO | CognitiveKernel: quick=qwen3-8b, full=qwen3-8b
INFO | MLX [optimal/fast]: 3549ms, 9 tokens, peak 5.22GB
```
…but `settings.json` says `mlx_primary_model: models/phi-3.5-mini-mlx-4bit`.

**Root cause** `core/cognitive_kernel.py` (or similar orchestrator) has **hardcoded labels** or reads a stale config copy. Peak RAM tells the truth: Phi-3.5-4bit ≈ 2.5 GB, Qwen-8B-4bit ≈ 5–6 GB.

**Fix sites**
1. Grep for any hardcoded `qwen3-8b` / `qwen3-4b` strings: `rg "qwen3-" --type py`.
2. Ensure every model label derives from `settings['brain']['mlx_primary_model']` (or `mlx_fast_model` for the fast role).
3. Also check the legacy `brain.model_path` key — some llama.cpp loaders still read it. Keep it in sync or explicitly disabled.

**Verification** Boot log's model announcement line must match `settings.json`. Peak RAM must match the chosen model's expected footprint.

---

## PB-13 · WhisperKit port bound but `/health` unhealthy → STT never ready

**Symptom**
```
WARNING | WhisperKit: port 127.0.0.1:50060 is bound but /health is unhealthy; reaping stale whisperkit-cli before launch
WARNING | WhisperKitSTT preload failed: WhisperKit port 127.0.0.1:50060 is bound by an unhealthy non-owned process
RuntimeError: WhisperKit port 127.0.0.1:50060 is bound by an unhealthy non-owned process
WARNING | STT preload did not become ready (8789ms); voice input remains unavailable until restart succeeds
WARNING | STT not ready after preload; ATOM running without reliable voice input
```

**Root cause** A previous ATOM (or another `whisperkit-cli serve` invocation) crashed without releasing its listening socket, or a non-WhisperKit process is squatting on `50060`. `_reap_stale_serve_on_port()` refuses to kill processes whose `ps -o command=` does not contain `whisperkit-cli`, so the preload raises and STT never arms.

**Fix sites**
1. `voice/stt_whisperkit.py` — `_maybe_start_serve()` must:
   - Reap stale `whisperkit-cli` listeners (already wired).
   - When the squatter is **not** WhisperKit, fall back to a free port from a configured range (`stt.whisperkit.port_fallback: [50061, 50062, 50063]`) instead of raising. Update `self._serve_port` and proceed with launch.
2. `voice/stt_whisperkit.py` — on every successful preload, write the bound port + PID to a runtime sidecar (`data/runtime/whisperkit.pid`) so a clean shutdown / next boot can SIGTERM the previous owner before binding.
3. `main.py` — STT preload must surface the failure mode in the boot timeline log, not just "preload did not become ready". The owner needs to see "WhisperKit port squatted by non-owned process" to act.
4. Owner-side recovery: `lsof -nP -iTCP:50060 -sTCP:LISTEN` → identify squatter → `kill -TERM <pid>` (only if it is WhisperKit or a known dev process). Never auto-kill unknown listeners.

**Verification** Boot log shows `WhisperKitSTT preloaded (model=…, serve=127.0.0.1:50060, vad=3)` within ~6 s. `STT ready -- ATOM fully operational` appears before the boot greeting completes. Scorecard `stt ready` < 6 000 ms.

---

## PB-14 · Ambient noise (AC, drops, keyboard, distant voices) routed as input

**Symptom** ATOM responds without the owner speaking. STT finalizes 1–3 word "ghost" phrases (`"you"`, `"uh huh"`, `"okay"`, `"thank you"`) during quiet periods or while ATOM itself is speaking.

**Root cause** Without a hard RMS gate, low-energy ambient frames flow into `webrtcvad` and the recognizer's internal turn-taker. Once enough sub-floor frames accumulate, WhisperKit/SFSpeechRecognizer promote them to a near-empty hypothesis and `_emit_final` routes them to the LLM.

**Fix sites**
1. `voice/stt_whisperkit.py` — `_audio_callback` computes RMS dBFS and calls `_noise_gate_blocks(rms_db)` before pushing the frame into the VAD ring buffer. Hysteresis-free: `noise_gate_consecutive` consecutive frames below `noise_floor_dbfs` close the gate; the first supra-floor frame reopens it.
2. `voice/stt_macos.py` — same `_noise_gate_blocks()` is wired into both the AVAudioEngine tap and the sounddevice callback path.
3. `config/settings.json` — sane room defaults: `stt.noise_floor_dbfs: -45.0`, `stt.noise_gate_consecutive: 3`, `stt.speech_candidate_floor_dbfs: -42.0`, `stt.min_audio_duration_s: 0.55`, `stt.whisper_vad_aggressiveness: 3`. Owners on noisier mics may go to `-40.0`; quiet booth users may relax to `-55.0`.
4. `voice/stt_whisperkit.py` — `_min_utterance_ms` honors `stt.min_audio_duration_s`. Anything shorter is treated as a fragment and dropped.
5. macOS native voice processing (`stt.native_voice_processing: true`) must remain enabled — it provides AEC + noise suppression at the CoreAudio layer, complementing our gate.

**Verification** `WhisperKitSTT.get_diagnostics()` reports `noise_gate_dropped_total > 0` after a quiet session. No `speech_final` events while the owner is silent. Tests: `tests/test_stt_whisperkit_launch.py::test_noise_gate_blocks_quiet_whisperkit_frames` and the macOS counterpart in `tests/test_stt_noise_gate.py` must pass.

---

## Adding a new playbook

When you solve a novel symptom, append here with the same 5 fields. Next session saves 2+ hours of re-discovery. Commit message: `docs(skill): add playbook PB-NN for <symptom>`.
