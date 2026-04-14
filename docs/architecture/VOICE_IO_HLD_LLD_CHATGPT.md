# ATOM — Voice input & output: HLD + LLD (for external review)

This document describes **how ATOM handles voice input (STT) and spoken output (TTS)** as implemented in the current codebase. It is suitable to paste into ChatGPT or another assistant for architecture discussion. Paths are relative to the ATOM repository root.

---

## 1. High-level design (HLD)

### 1.1 Purpose

- **Input:** Convert user speech to text, then route it through the **Intent Engine** and **Router** (local commands, tools, or LLM).
- **Output:** Convert assistant replies to speech via **native macOS TTS**, with optional **token streaming** from the LLM path.
- **Coordination:** A central **`AtomState`** finite-state machine and an **async event bus** synchronize mic, recognition, reasoning, and playback so the mic is not left hot during TTS and recovery paths stay consistent.

### 1.2 Logical layers

| Layer | Responsibility |
|--------|----------------|
| **Launch / OS** | `ATOM.app` supplies `Info.plist` usage strings (`NSSpeechRecognitionUsageDescription`, `NSMicrophoneUsageDescription`). The bundle launcher sets `ATOM_APP_BUNDLE`, `ATOM_LAUNCH_MODE=bundle`, and runs `.venv/bin/python` so dependencies match the venv. |
| **STT backends** | Prefer **Apple `SFSpeechRecognizer` + `AVAudioEngine`** on macOS (`voice/stt_macos.py`). Fallbacks: **Faster-Whisper** (`voice/stt_async.py`), **Google** (`voice/stt_google.py`), or **DisabledSTT** (`main.py`). |
| **Intent + routing** | `core/intent_engine/*` classifies text; `core/router/router.py` executes actions or LLM fallback. |
| **Orchestration** | `core/boot/wiring.py` binds bus events: `speech_final` → Router; `response_ready` / `partial_response` → TTS. |
| **TTS** | `voice/tts_macos.py`: **`NSSpeechSynthesizer`**, streaming/chunk coalescing, barge-in hooks. |
| **UI** | `ui/web_dashboard.py`: browser dashboard; optional browser mic fallback (separate from native STT). |

### 1.3 End-to-end data flow (happy path)

```mermaid
flowchart LR
  Mic[Mic via AVAudioEngine]
  STT[SFSpeechRecognizer]
  Bus1[Event bus: speech_final]
  R[Router + IntentEngine]
  Brain[Cache / Memory / LLM]
  Bus2[Event bus: response_ready / partial_response]
  TTS[NSSpeechSynthesizer]
  Spk[Speakers]

  Mic --> STT --> Bus1 --> R --> Brain --> Bus2 --> TTS --> Spk
```

### 1.4 STT selection policy (macOS)

Configured via `config/settings.json` → `stt.engine` (e.g. `auto`, `macos_native`).

1. **Native** — `native_stt_launch_supported()` in `voice/stt_macos.py` must pass (PyObjC Speech/AVFoundation present; usage strings visible — including reading `ATOM_APP_BUNDLE/Contents/Info.plist` when the process image is venv Python after `exec`).
2. **Faster-Whisper** — if native unavailable and Python deps exist (`faster-whisper`, `SpeechRecognition`, `pyaudio`).
3. **Google** — if configured chain reaches it and deps exist.
4. **Disabled** — explicit stub with logging (no listening).

### 1.5 TTS policy (macOS)

- Primary: **`NSSpeechSynthesizer`** in `voice/tts_macos.py`.
- Voice selection: `tts.macos_voice` (e.g. `system` / Spoken Content–aligned resolution in code).
- **Streaming replies:** `partial_response` events drive chunked synthesis with **coalescing** and deduplication to avoid word-by-word audio.
- **Exit / sleep:** `response_ready` may carry `is_exit=True` or `is_sleep=True`; TTS completes then emits `tts_complete` / `enter_sleep_mode` / `shutdown_requested` as wired.

### 1.6 State machine (voice-relevant)

`AtomState` drives when STT listens vs when TTS plays. **Native STT** starts/stops its listen loop from **`stt.on_state_changed`** (see `docs/architecture/14_VOICE_PIPELINE.md`). Typical loop: **LISTENING** → **THINKING** → **SPEAKING** → **LISTENING**, with **`post_tts_cooldown_ms`** after TTS before reopening the mic.

---

## 2. Low-level design (LLD)

### 2.1 Key modules

| Module | Role |
|--------|------|
| `main.py` | Builds STT/TTS instances from config; registers `DisabledSTT` when no backend works; wires `WebDashboard` / native UI. |
| `voice/stt_macos.py` | `NativeSTT`: `SFSpeechRecognizer`, authorization, `_effective_bundle_info()` for plist + `NSBundle`, emits `speech_partial` / `speech_final`, `voice.partial` / `voice.final` for dashboard. |
| `voice/stt_async.py` | Faster-Whisper path; same bus contract where applicable. |
| `voice/tts_macos.py` | Subscribes to `response_ready`, `partial_response`; `on_response`, `on_partial_response`; `tts_complete` after speak; barge-in via `on_speech_partial`. |
| `core/boot/wiring.py` | **Central wiring:** `speech_final` → `router.on_speech` (optionally via `priority_scheduler`); `response_ready` → `tts.on_response`; `partial_response` → `tts.on_partial_response`; `tts_complete` → `state.on_tts_complete`; `shutdown_requested` → sets `shutdown_event`. |
| `core/router/router.py` | `_route()`: intent classification, `exit` vs `go_silent`, tool execution, LLM fallback; emits `response_ready`. |
| `core/intent_engine/meta_intents.py` | Meta intents: **`bye` / `goodbye` → `go_silent`**, hard quit phrases → `exit` (so casual “bye” does not shut down the app). |
| `core/async_event_bus.py` | Async pub/sub for all cross-module events. |
| `scripts/atom_app_bundle_launcher.sh` | macOS `.app` executable: `exec` venv Python with env vars; filters Finder `-psn_*` args. |

### 2.2 Environment variables (voice / bundle)

| Variable | Meaning |
|----------|---------|
| `ATOM_APP_BUNDLE` | Absolute path to `ATOM.app`; used to load `Contents/Info.plist` for usage-string checks when `mainBundle` is not the app. |
| `ATOM_LAUNCH_MODE` | `bundle` vs `venv` — native STT policy in `stt_macos.py` (`venv` may skip native bundle expectations). |
| `ATOM_LAUNCHED_FROM_APP` | Set by launcher scripts when appropriate. |
| `VOICE_DEBUG` / `VOICE_INPUT` | Diagnostics for mic/STT (see logs). |

### 2.3 Config keys (representative)

| Key | Area |
|-----|------|
| `stt.engine`, `stt.locale`, `stt.post_tts_cooldown_ms`, `stt.dev_prefer`, whisper/VAD-related keys | STT |
| `tts.macos_voice`, streaming-related behavior | TTS |
| `ui.mode`, `ui.web_port`, `ui.auto_open_browser` | Dashboard vs native UI |

Exact schema: `core/config_schema.py` and `config/settings.json`.

### 2.4 Event bus contracts (voice)

| Event | Producer | Main consumers |
|-------|-----------|----------------|
| `speech_partial` | STT | Indicator “hearing”, voice interrupt |
| `speech_final` | STT | Router (`on_speech`), consolidated handler (metrics/UI) |
| `voice.partial` / `voice.final` | Wiring (from STT) | `AtomState` bridge / dashboard |
| `response_ready` | Router, brain, wiring helpers | TTS `on_response` |
| `partial_response` | LLM streaming path | TTS `on_partial_response` |
| `tts_complete` | TTS | `StateManager.on_tts_complete` |
| `shutdown_requested` | TTS (after `is_exit` speak), other | Full shutdown in wiring |
| `enter_sleep_mode` | TTS after `is_sleep` | STT stop + sleep state |
| `restart_listening` | Recovery | STT `stop()` + restart listen loop |

### 2.5 Priority scheduling (optional)

When `priority_scheduler` is enabled, **`speech_final`** is submitted as **`PRIORITY_VOICE`** so voice work is ordered ahead of background LLM jobs where configured (`core/boot/wiring.py`).

### 2.6 Intent notes (product logic)

- **`exit`:** Spoken farewell, then **`shutdown_requested`** (full app shutdown).
- **`go_silent`:** Spoken line, then **`enter_sleep_mode`** (STT stopped until user resumes via hotkey/dashboard).
- Casual **“bye”** is **`go_silent`**, not **`exit`**, to avoid spurious STT matches closing the app.

---

## 3. Operational checklist

1. **Native STT + TCC:** Launch via **`ATOM.app`** or **`Run ATOM.command`** so `ATOM_APP_BUNDLE` / bundle mode are set; grant **Speech Recognition** and **Microphone** in **System Settings → Privacy & Security**.
2. **Raw `python main.py`:** May not satisfy native bundle checks; expect disabled native STT or install offline STT deps.
3. **Logs:** `logs/atom.log` — search for `STT:`, `TTS:`, `VOICE_DEBUG`, `Web dashboard running at`.

### 3.1 Launch validation (how to confirm what macOS sees)

On every boot, ATOM logs one line:

- **`VOICE_LAUNCH_DIAG: ATOM_LAUNCH_MODE=… ATOM_APP_BUNDLE=… label=…`**

Interpretation:

| `ATOM_LAUNCH_MODE` | Typical launch | Native STT |
|--------------------|----------------|------------|
| `bundle` | `ATOM.app` or `Run ATOM.command` when the bundle launcher self-test passes | Eligible (still needs TCC + plist merge for venv `exec`) |
| `venv` | `Run ATOM.command` fallback path, or Cursor/Terminal `python main.py` | **Refused by policy** in `voice/stt_macos.py` unless you use `stt.dev_prefer` (see below) |
| *(empty)* | Unusual | Treat like venv unless you set env vars yourself |

**Last lines of `logs/atom.log`:** search for `Voice input unavailable`, `native unavailable`, `Offline STT dependencies`, `STT backend selected`, `VOICE_LAUNCH_DIAG`.

### 3.2 Dev ergonomics: `stt.dev_prefer` (venv / IDE)

When `ATOM_LAUNCH_MODE=venv`, native macOS STT is skipped **by design**. To get **real** voice input without the `.app` process, set in `config/settings.json` under `stt`:

- `"dev_prefer": "faster_whisper"` — try Faster-Whisper first, then Google, then disabled.
- `"dev_prefer": "google_online"` — try Google first, then Faster-Whisper, then disabled.

Install the matching dependencies (`faster-whisper` / `SpeechRecognition` / `pyaudio` as needed). This does **not** bypass `SFSpeechRecognizer` security with a fake “native OK” flag; it selects an alternate engine.

### 3.3 Dashboard: voice runtime truth

The web dashboard voice strip shows **`STT: … [launch_mode]`**, permissions, **error line**, **fallback trace** (from `voice.fallback_chain`), and **`bundle:`** path when `app_bundle` is set — so “mic dead” is diagnosable without SSH.

### 3.4 Bluetooth and default input (native vs MicManager)

- **Native macOS STT** (`SFSpeechRecognizer` + `AVAudioEngine` in [`voice/stt_macos.py`](../../voice/stt_macos.py)) records from whatever macOS treats as the **current default input device**. Set **System Settings → Sound → Input** to your Bluetooth headset or earbuds before testing; ATOM does not pass PyAudio’s device index into `AVAudioEngine`.
- **MicManager** (PyAudio profiling in [`main.py`](../../main.py) background preload) selects a **best** device for logging and for **Faster-Whisper / Google** STT paths. That selection may **differ** from the label shown for native STT if defaults differ.
- **`mic.device_name`** in `config/settings.json` is reserved for future explicit PyAudio routing; it does **not** override native STT’s input today (see [`core/config_schema.py`](../../core/config_schema.py)).

**Operator checklist:** [`docs/operations/BLUETOOTH_VOICE_TEST.md`](../operations/BLUETOOTH_VOICE_TEST.md).

---

## 4. Related internal docs

- `docs/architecture/14_VOICE_PIPELINE.md` — state/listen ownership, mermaid diagram.
- `docs/architecture/09_STATE_MACHINE.md`, `08_EVENT_BUS.md` — cross-reference if present.

---

*Generated for sharing with ChatGPT / external reviewers; keep repo paths and behavior in sync when refactoring.*
