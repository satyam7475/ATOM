# ATOM OS — Full System Review & Architecture Document

> **Performance, voice pipeline timings, buddy/learning scenarios, ChatGPT-ready summary:**  
> See **[ATOM_End_to_End_Performance_Report.md](./ATOM_End_to_End_Performance_Report.md)** and **[ATOM_System_Diagram.svg](./ATOM_System_Diagram.svg)**.  
> **v15 offline build:** cloud Gemini/Groq and `brain_selector` are removed; inference is **local GGUF only** when `brain.enabled=true`. Align with `config/settings.json`.

**Owner:** Satyam  
**Version:** v15 (Cognitive + local LLM brain; this doc’s body still describes v14-era architecture in places)  
**Platform:** Windows 10/11 (corporate laptop, i7-1185G7, 4C/8T, 32 GB RAM)  
**Runtime:** Python 3.11+, single-process, async event-driven  
**Total codebase:** ~17,300 lines of Python across 76 files + 913-line HTML dashboard  
**Last updated:** March 2026  

---

## 1. What is ATOM?

ATOM is a **personal AI Operating System layer** that runs on top of Windows. It is a voice-controlled, always-listening assistant (similar to JARVIS from Iron Man) that can:

- Understand and execute voice commands locally in <5ms via regex-based Intent Engine
- Use the **local LLM** (llama.cpp) for open-ended questions when `brain.enabled=true`
- Control the desktop (scroll, click, type, open/close apps, manage files)
- Monitor system health (CPU, RAM, battery, network, Bluetooth) with a CPU governor
- Schedule reminders, research the web, and self-diagnose all subsystems on command
- Enforce corporate security policies (blocked commands, safe app allowlists, audit logging)
- Adapt its own resource usage with 4-tier performance modes (full / lite / ultra_lite / auto) and a closed-loop CPU governor that throttles background tasks and TTS when system load is high

ATOM is designed as a **brain that can be embedded** — it has a single `run_atom(config_overrides)` entry point, config-driven feature flags, and a lock mode for restricted operation.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          ATOM OS Layer                              │
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌────────────────────────────────┐  │
│  │  Voice    │   │ Security │   │         Router                 │  │
│  │  Input    │──▶│  Gate    │──▶│  (3-layer intelligence)        │  │
│  │(Vosk STT)│   │          │   │                                 │  │
│  └──────────┘   └──────────┘   │  Layer 1: Intent Engine (<5ms) │  │
│                                 │  Layer 2: Cache + Memory        │  │
│       ┌─────────────────────── │  Layer 3: Smart Brain Selector  │  │
│       │                        │           (Groq / Gemini)        │  │
│       ▼                        └───────────┬─────────────────────┘  │
│  ┌──────────┐                              │                        │
│  │  Voice   │◀─────────────────────────────┘                        │
│  │  Output  │   ┌─────────────────────────────────────────────┐     │
│  │(Edge TTS)│   │  OS Services                                │     │
│  └──────────┘   │  • Task Scheduler    • Process Manager      │     │
│                 │  • System Watcher    • Desktop Control       │     │
│  ┌──────────┐   │  • Web Researcher    • Self-Evolution (*)   │     │
│  │    UI    │   │  • Behavior Tracker(*)• Screen Analyzer      │     │
│  │  (Web    │   │  • Health Monitor    • Context Engine        │     │
│  │Dashboard)│   │         (*) = on-demand only                │     │
│  └──────────┘   └─────────────────────────────────────────────┘     │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Infrastructure                                              │    │
│  │  • AsyncEventBus (pub/sub)   • StateManager (6 states)      │    │
│  │  • MetricsCollector          • SecurityPolicy (audit)        │    │
│  │  • PipelineTimer             • Config Schema Validator       │    │
│  │  • CPU Governor              • Graceful Restart Loop         │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### Why this architecture?

1. **Event-driven (pub/sub):** Every module communicates through `AsyncEventBus`. No module directly calls another — they emit events and subscribe to events. This gives loose coupling, easy testing, and the ability to add/remove features without changing other code.

2. **3-layer routing:** 80-90% of commands are handled locally by the Intent Engine (regex) in <5ms. Only open-ended questions reach the cloud LLM. This keeps ATOM fast and reduces API costs.

3. **Single-process async:** Everything runs in one Python process with `asyncio`. Blocking operations (STT, LLM HTTP calls) run in a `ThreadPoolExecutor(2)`. This avoids IPC complexity while staying responsive.

4. **Config-driven:** All behavior is controlled by `config/settings.json`. Security policies, feature flags, lock modes, TTS voice, LLM models, performance modes — everything is configurable without code changes.

5. **Self-adaptive:** The CPU governor + auto performance mode form a closed-loop control system. ATOM monitors its own resource impact and backs off when the laptop is under heavy load.

---

## 3. State Machine

ATOM has 6 states managed by `StateManager`:

```
         ┌──────┐
    ┌───▶│SLEEP │◀──── "go silent" command
    │    └──┬───┘
    │       │ Ctrl+Alt+A hotkey
    │       ▼
    │    ┌──────┐       ┌─────────┐
    │    │ IDLE │◀─────▶│LISTENING│◀───────────┐
    │    └──────┘       └────┬────┘            │
    │                        │ speech detected  │
    │                        ▼                  │
    │                   ┌─────────┐             │
    │                   │THINKING │             │
    │                   └────┬────┘             │
    │                        │ response ready   │
    │                        ▼                  │
    │                   ┌─────────┐             │
    │                   │SPEAKING │─────────────┘
    │                   └────┬────┘   tts_complete
    │                        │
    │                        ▼
    │               ┌───────────────┐
    └───────────────│ERROR_RECOVERY │
                    └───────────────┘
```

**Why a state machine?** Prevents race conditions (e.g., STT listening while TTS is speaking), ensures clean transitions, and makes debugging simple — you always know what ATOM is doing. The health monitor auto-recovers from stuck states (THINKING/SPEAKING >120s, ERROR_RECOVERY >180s).

**`always_listen` mode:** After speaking, ATOM immediately returns to LISTENING (no IDLE gap). This is the "JARVIS feel" — always ready.

---

## 4. Module-by-Module Review

### 4.1 Voice Input — `voice/stt_async.py` (837 lines)

**What it does:** Captures audio from the microphone and transcribes speech. Supports multiple STT engines via config.

**STT Engines (config-driven):**

| Engine | Latency | Network | Notes |
|--------|---------|---------|-------|
| **Vosk** (default) | 100-200ms | Offline | Small English model (~50 MB), runs on CPU |
| Google Web Speech | 300-700ms | Required | Free, no API key, higher accuracy |
| vosk_with_fallback | 100-700ms | Optional | Vosk first, Google on low confidence |

**Why Vosk as default?**
- 100-200ms latency vs 300-700ms for Google — this alone makes ATOM feel 2-3× faster
- Fully offline — no network dependency, no corporate firewall issues
- Low CPU usage with the small model
- Deterministic — same audio always gives same text

**Key design decisions:**
- **Bluetooth mic priority:** Detects BT headsets and prefers them over the built-in Intel mic (less background noise)
- **Post-TTS cooldown (800ms):** After ATOM speaks, STT waits 800ms before accepting new audio — prevents hearing its own voice
- **Noise flood handling:** After 2+ consecutive "could not understand" errors, raises the energy threshold by 50%
- **Periodic recalibration:** Every 2 minutes without successful speech, re-calibrates the noise threshold
- **Text corrections:** Fixes common STT mishearings ("he adam" → "hey atom", "adam" → "atom")
- **BT-specific tuning:** Minimum threshold 3500 for BT mics, dynamic energy disabled (prevents threshold drift from HVAC/keyboard noise)
- **Minimum audio duration (0.6s):** Rejects clicks, pops, and accidental noises

**Dependencies:** `SpeechRecognition`, `PyAudio`, `vosk` (optional)

**Events emitted:** `speech_partial`, `speech_final`, `silence_timeout`, `stt_did_not_catch`, `stt_too_noisy`, `mic_changed`

---

### 4.2 Voice Output — `voice/tts_edge.py` (775 lines)

**What it does:** Synthesizes speech using Microsoft Edge Neural TTS (en-GB-RyanNeural — warm British male voice, like JARVIS).

**Why Edge TTS?**
- Free, no API key
- High quality neural voices (much better than Windows SAPI)
- Works over HTTPS (corporate-safe)
- Alternative: ElevenLabs (paid, higher quality but needs API key)

**Key design decisions:**
- **Sentence-level streaming:** Generates sentence N+1 while playing sentence N. This overlaps synthesis and playback for minimal gaps.
- **Smart buffering:** Accumulates all LLM response chunks, then speaks the first ~45 words as audio and shows the rest on screen. This balances audio length with speed.
- **SSML emotion profiles:** Different rate/pitch/volume for different contexts (neutral, friendly, urgent, calm, ack, error). Makes responses feel natural.
- **Pre-cached ack phrases:** 25+ common phrases ("Yes, Boss?", "On it.", "Working on it.") are pre-generated at startup for instant playback (<50ms).
- **Audio post-processing (configurable):** RMS normalization + tanh soft limiter for consistent volume. Disabled by default on CPU-constrained laptops (`edge_postprocess: false`). The CPU governor dynamically disables/restores this based on system load.
- **Barge-in:** User can interrupt ATOM mid-speech (Ctrl+Alt+A or new speech). Sets `_cancel_requested` and stops playback immediately.
- **Graceful degradation:** If Edge TTS fails (network error), falls back to showing the text on screen and emits `text_display`.
- **`_speak_lock`:** `asyncio.Lock` ensures only one piece of audio plays at a time — prevents ack overlapping with response.
- **Governor-controlled:** `set_postprocess(bool)` and `restore_postprocess()` methods let the CPU governor disable post-processing under heavy load and restore it when load drops.

**Dependencies:** `edge-tts`, `pygame`, `numpy`

**Events emitted:** `tts_complete`, `text_display`, `enter_sleep_mode`

---

### 4.3 Intent Engine — `core/intent_engine/` (11 modules, ~1,350 lines total)

**What it does:** Classifies user speech into intents using regex and keyword matching. Handles 80-90% of commands without needing an LLM.

**Why regex instead of ML?**
- **Speed:** <5ms classification. An ML model would take 50-200ms.
- **Reliability:** Deterministic — same input always gives same output. No model drift.
- **Offline:** Works without internet. The LLM is only for open-ended questions.
- **Maintainability:** Adding a new command = adding a regex pattern in the relevant sub-module. No retraining.

**Modular structure (split from a monolithic 1,625-line file):**

| Module | Lines | Responsibility |
|--------|-------|---------------|
| `__init__.py` | 99 | Package entry, `IntentEngine` class, classify chain |
| `base.py` | 126 | `IntentResult` dataclass, grammar words, reply templates |
| `meta_intents.py` | 95 | Greeting, thanks, status, confirm/deny, exit, sleep |
| `info_intents.py` | 197 | Time, date, CPU, RAM, battery, disk, IP, WiFi, uptime |
| `app_intents.py` | 161 | open_app, close_app, list_apps (APP_MAP for name resolution) |
| `media_intents.py` | 95 | play_youtube, stop_music, volume, mute/unmute |
| `system_intents.py` | 90 | Lock screen, screenshot, shutdown, restart, sleep PC |
| `desktop_intents.py` | 141 | Scroll, click, press_key, hotkey, type_text, go_back |
| `file_intents.py` | 52 | create_folder, move_path, copy_path |
| `network_intents.py` | 29 | search, open_url, weather, wifi_status |
| `os_intents.py` | 252 | self_check, self_diagnostic, set_performance_mode, behavior_report, research, reminders, kill_process |

**Why the split?** The original single file was 1,625 lines — hard to navigate, slow to iterate, and regex conflicts were hard to detect. Now each category is isolated: adding a media command only touches `media_intents.py`.

**Supported intents (35+):**
- **System info:** time, date, CPU, RAM, battery, disk, IP, WiFi, uptime
- **Apps:** open_app, close_app, list_apps
- **Media:** play_youtube, stop_music, set_volume, mute/unmute
- **Desktop:** scroll_down/up, click_screen, press_key, hotkey_combo, type_text, go_back
- **Files:** create_folder, move_path, copy_path
- **Network:** search, open_url, weather, wifi_status
- **System control:** lock_screen, screenshot, shutdown_pc, restart_pc, sleep_pc
- **AI OS:** set_reminder, show_reminders, cancel_reminders, kill_process, research_topic, self_check, set_performance_mode
- **Meta:** greeting, thanks, status, exit, go_silent, confirm/deny
- **Fallback:** anything not matched → sent to LLM

**Key design decisions:**
- **Priority ordering:** `meta_intents` → `os_intents.check_self_check` → `info_intents` → `system_intents` → `media_intents` → `desktop_intents` → `file_intents` → `network_intents` → `os_intents` → `app_intents`. More specific patterns checked first.
- **Entity extraction:** Extracts app names, queries, numbers from matched text
- **Calculator:** Safe `eval()` with a strict character whitelist (digits, operators, parentheses only)
- **Grammar words:** Exports a vocabulary list for Vosk integration

---

### 4.4 Router — `core/router/router.py` (1,063 lines)

**What it does:** The central brain. Takes classified intents, checks security, and dispatches actions to the appropriate handler modules.

**3-layer intelligence pipeline:**
1. **Intent Engine:** Instant regex match → direct action
2. **Cache + Memory:** If the query was asked before, serve cached answer
3. **Smart Brain Selector:** If no local answer, pick the best LLM (Groq or Gemini) and stream the response

**Why 3 layers?**
- Layer 1 handles 80-90% instantly (no network)
- Layer 2 avoids redundant LLM calls (saves money, saves time)
- Layer 3 is only for novel questions

**Key design decisions:**
- **Security gate:** Every action goes through `SecurityPolicy.allow_action()` before dispatch. Blocked actions are audit-logged and rejected.
- **Input sanitization:** Raw speech is capped at 2000 chars and stripped of shell-injection characters before processing.
- **Conversational continuity:** Resolves pronouns ("it", "that") to the last mentioned entity. Tracks 5-turn conversation history for context.
- **Clipboard injection:** If the user says "that error" or "this code", automatically injects clipboard content into the LLM prompt.
- **Repeat query detection:** If the same question is asked twice within 60s, bypasses cache and adds a "provide a different answer" hint to the LLM.
- **Confirmation flow:** Destructive actions (shutdown, delete, close app) require voice confirmation ("Yes" / "No") before execution. Times out after 25s.
- **Intent chaining:** After certain actions, suggests a follow-up (e.g., after opening VS Code → "Want me to check your git status?").
- **Smart acknowledgments:** Context-aware thinking messages based on query keywords ("Let me look into that issue" for error-related queries).
- **Self-check command:** Reports status of all ATOM subsystems (STT, TTS, LLMs, CPU, RAM, governor) via voice and screen.
- **Performance mode switching:** `set_performance_mode` action emits an event that triggers config update + graceful restart.

**Action dispatch table (`_ACTION_DISPATCH`):** Maps 40+ action names to handler methods. Sub-modules: `app_actions`, `file_actions`, `media_actions`, `network_actions`, `system_actions`, `utility_actions`.

---

### 4.5 Local LLM path (v15 offline)

**Removed:** `core/brain_selector.py`, Gemini (`cursor_controller`), Groq (`groq_controller`), and Gemini Vision (`screen_analyzer`). ATOM no longer opens cloud LLM connections.

**Current path:** `cursor_bridge/local_brain_controller.py` → `brain/mini_llm.py` (llama.cpp GGUF). Optional warm-up at startup; fake streaming via sentence chunks for responsive TTS.

---

### 4.6 Prompt Builder — `cursor_bridge/structured_prompt_builder.py` (190 lines)

**What it does:** Builds structured prompts for LLMs with personality, context, and privacy.

**Prompt structure:**
```
You are ATOM, a personal AI OS built by Satyam. Address the owner as "Boss".
[Active window context]
[Clipboard content if referenced]
[2-turn conversation history]
[Memory context]
[Query type hint (debug/architecture/how-to)]
[Time-of-day personality hint]

User query: ...
```

**Privacy:** All prompts pass through `privacy_filter.redact()` before sending to remove API keys, tokens, passwords, emails, connection strings.

---

### 4.7 Security — `core/security_policy.py` (297 lines)

**What it does:** The single security gate for all ATOM actions. Every sensitive operation goes through this module.

**Why security-first?**
- ATOM runs on a **corporate laptop** — it must never execute dangerous commands
- Voice input is inherently risky — users might accidentally trigger destructive actions
- Audit trail is required for accountability

**Security layers:**
1. **Action gate (`allow_action`):** Called by router before every action. Checks:
   - Lock mode (`off`, `safe_only`, `owner_only`)
   - Feature flags (desktop_control, file_ops, llm, system_analyze)
   - Executable allowlist for `open_app`
   - Process allowlist for `close_app`
   - Power action blocking in strict mode
2. **Shell command blocklist:** Blocks `format`, `del /s`, `reg delete`, `net user`, etc.
3. **Hotkey safety tiers:** `safe` (no confirmation), `confirm` (needs user OK), `block` (never allowed)
4. **Path allowlist:** File operations only allowed within user home or CWD — never in System32, Program Files, etc.
5. **Input sanitization:** Caps input at 2000 chars, strips `; & | $ \`` and `<script` patterns
6. **Audit log:** Every sensitive and blocked action logged to `logs/audit.log` with timestamp

**Config-driven (`settings.json`):**
```json
"security": {
    "mode": "strict",
    "audit_to_file": true,
    "require_confirmation_for": ["shutdown_pc", "restart_pc", "close_app", ...]
},
"features": {
    "desktop_control": true,
    "file_ops": true,
    "llm": true,
    "system_analyze": true
},
"control": {
    "lock_mode": "off"
}
```

---

### 4.8 AsyncEventBus — `core/async_event_bus.py` (173 lines)

**What it does:** Pub/sub backbone for all inter-module communication.

**Why pub/sub instead of direct calls?**
- **Loose coupling:** STT doesn't know about TTS, Router doesn't know about the Dashboard. They just emit events.
- **Easy testing:** Mock the bus, subscribe to events, verify behavior.
- **Extensibility:** Adding a new module = subscribing to existing events. No changes to existing code.

**Three emission modes:**
- `emit(event, **kwargs)` — Standard (10s timeout per handler)
- `emit_fast(event, **kwargs)` — Non-critical (5s timeout, errors suppressed)
- `emit_long(event, **kwargs)` — Long-running (60s timeout, for LLM calls)

**Safety features:**
- Per-handler try/except (one handler crash doesn't kill others)
- Slow handler warnings (>5s)
- `WeakSet` for active task tracking
- `pending_count` property for health monitoring

---

### 4.9 Cache Engine — `core/cache_engine.py` (169 lines)

**What it does:** TTL-aware LRU cache for LLM responses.

**Why not just a dict?**
- **TTL:** Responses expire (default 300s) so stale data isn't served forever
- **LRU eviction:** When cache is full (128 entries), least-recently-used entries are evicted
- **Jaccard similarity:** If exact match misses, scans top 32 entries for similar queries (≥0.75 similarity). "What is Docker?" and "Tell me about Docker" hit the same cache entry.
- **Self-tuning TTL:** The maintenance loop adjusts cache TTL based on hit rate — if >60% hits, TTL increases (up to 600s); if <20% hits, TTL decreases (down to 120s).

**Why Jaccard instead of embeddings?**
- No model download, no GPU, no latency
- Good enough for voice queries (which are naturally short and similar)
- Embeddings would add 50-100ms per query and require a model file

---

### 4.10 Memory Engine — `core/memory_engine.py` (110 lines)

**What it does:** Stores Q&A pairs for long-term recall. When a new query comes in, retrieves relevant past answers by keyword overlap.

**Why not a vector database?**
- ATOM handles ~50-200 queries per session. Keyword overlap works fine at this scale.
- No model download, no GPU, no extra process
- JSON storage is simple and debuggable

**Storage:** `logs/memory.json`, persisted on shutdown, max 500 entries.

---

### 4.11 Health Monitor — `core/health_monitor.py` (319 lines)

**What it does:** Background watchdog + CPU governor. Runs periodically and checks all subsystems.

**Checks:**
| Component | What it checks | Auto-recovery |
|-----------|---------------|---------------|
| Event bus | Pending tasks > 50 | Warning only |
| State machine | Stuck in THINKING/SPEAKING > 120s | Force → LISTENING |
| State machine | ERROR_RECOVERY > 180s | Force → LISTENING |
| CPU | Usage > 95% | Warning only |
| RAM | Usage > 90% | Warning only |
| STT | No microphone detected | Warning only |
| TTS | Mixer not initialized, 3+ failures | Warning only |
| Bluetooth | Device connect/disconnect | Auto-switch mic/output |

**CPU Governor (closed-loop control — NOT just a dashboard indicator):**

When `cpu_governor: true`, the HealthMonitor watches system-wide CPU usage. This is a **real behavioral change**, not just a visual indicator:

1. **Throttle** (CPU > `cpu_governor_threshold` for 2 consecutive checks):
   - HealthMonitor: check interval × 2.5 (e.g. 120s → 300s)
   - SystemWatcher: poll interval × 3.0 (e.g. 30s → 90s)
   - TTS: post-processing disabled (saves CPU on audio normalization)
   - Emits `governor_throttle` event for any future listeners
2. **Restore** (CPU below threshold for 3 consecutive checks):
   - All intervals restored to base values
   - TTS: post-processing restored to config value
   - Emits `governor_normal` event

**Notification:** After 3 consecutive warning cycles, speaks a warning to the user: "Boss, I'm having trouble with: [component list]. Check logs."

---

### 4.12 System Watcher — `core/system_watcher.py` (312 lines)

**What it does:** Background daemon that monitors Windows system events and emits them to the event bus.

**Monitors:**
- **App switching:** Detects foreground app changes via Win32 `GetForegroundWindow`
- **Network:** Checks connectivity to 8.8.8.8 / 1.1.1.1 every poll cycle
- **Battery:** Plug/unplug events, critical level (<20%) warnings
- **Bluetooth:** Audio device connect/disconnect (via PyAudio device enumeration)
- **Resources:** CPU > 85% or RAM > 85% (throttled to once per 5 min)

**Governor-aware:** Subscribes to `governor_throttle` / `governor_normal` events and adjusts its own poll interval (base × 3.0 when throttled).

---

### 4.13 Desktop Control — `core/desktop_control.py` (180 lines)

**What it does:** Voice-controlled desktop automation via `pyautogui`.

**Capabilities:** scroll, click, double-click, press key, hotkey combo, type text, move mouse, screenshot.

**Security:** All actions go through `SecurityPolicy` — `is_safe_key`, `is_safe_hotkey`, audit logging. Dangerous hotkeys (Win+R, Ctrl+Alt+Delete) are blocked. FAILSAFE is enabled (moving mouse to corner aborts).

---

### 4.14 Process Manager — `core/process_manager.py` (270 lines)

**What it does:** OS-level process and resource management via `psutil`.

**Capabilities:**
- Top processes by CPU/memory
- Kill process by name (with security check)
- Resource snapshots and trend analysis (rolling history)
- App switch history tracking
- Open windows listing (via Win32 `EnumWindows`)
- Full system report (for self-check command)

---

### 4.15 Task Scheduler — `core/task_scheduler.py` (221 lines)

**What it does:** Persistent reminders and scheduled tasks.

**How it works:**
- User says "remind me to check email in 10 minutes"
- Scheduler creates a `ScheduledTask` with due time
- Background poll every 30s checks for due tasks
- When due, emits `reminder_due` → TTS speaks the reminder
- Tasks persist to `logs/tasks.json` (survive restart)
- Supports recurring tasks (reschedules after firing)

---

### 4.16 Web Researcher — `core/web_researcher.py` (158 lines)

**What it does:** Searches the web using DuckDuckGo APIs and returns concise results.

**Why DuckDuckGo?**
- Free, no API key
- Instant Answer API for quick facts
- Lite search for URL results
- HTTPS only (corporate-safe)
- No tracking

---

### 4.17 Self-Evolution Engine — `core/self_evolution.py` (228 lines)

**What it does:** Analyzes ATOM's own performance metrics and suggests improvements.

**How it works:**
- Reads MetricsCollector snapshots (latency, cache hit rate, LLM calls)
- Calculates a health score (1-10)
- Identifies issues (high latency, low cache hit rate, too many LLM calls)
- Generates actionable suggestions
- Tracks evolution history in JSON

**On-demand only:** Does not run on a timer. Only invoked when the user explicitly asks for a behavior/performance report. This saves CPU and RAM.

---

### 4.18 Behavior Tracker — `core/behavior_tracker.py` (136 lines)

**What it does:** Logs user actions and detects patterns for proactive suggestions.

**Example:** If the user opens Outlook every day at 9 AM, ATOM might suggest "Want me to open Outlook?" at 9 AM.

**Storage:** `logs/behavior.json`, max 2000 entries, needs 3+ occurrences for a suggestion.

**On-demand only:** Logs intent classifications but only runs pattern analysis when explicitly requested. Does not consume CPU in the background.

---

### 4.19 Context Engine — `context/context_engine.py` (130 lines)

**What it does:** Gathers environment context for LLM prompts.

**Context bundle:**
- Active window title and app name (via Win32 `GetWindowTextW`)
- Clipboard content (via Win32 `GetClipboardData`)
- Current working directory
- Timestamp

**Privacy:** All context is passed through `privacy_filter.redact()` before being sent to LLMs.

---

### 4.20 Privacy Filter — `context/privacy_filter.py` (87 lines)

**What it does:** Redacts sensitive patterns from text before it reaches LLMs or logs.

**Patterns redacted:**
- API keys (GEMINI_API_KEY=xxx, Bearer tokens, x-api-key headers)
- PEM certificates
- GitHub PATs
- JDBC connection strings
- MongoDB URIs
- Passwords in URLs
- Email addresses

---

### 4.21 UI — Web Dashboard

**`ui/web_dashboard.py` (362 lines):** aiohttp HTTP server + WebSocket. Pushes state updates, logs, system stats to the browser. Serves as a **real-time control plane** — not just display.

**`ui/dashboard/index.html` (913 lines):** Single-file dashboard with:
- Three.js 3D orb (glows based on state — blue=listening, green=speaking, orange=thinking)
- Live conversation bubbles (heard, speaking, info)
- System stats (CPU, RAM, battery, disk) updated every 5s via WebSocket
- **Performance mode control panel:** 4 buttons (FULL, LITE, ULTRA, AUTO) with mode-specific accent colors (cyan, green, purple, amber). Clicking a mode button triggers config update + graceful restart.
- **Governor status indicator:** Shows "GOV: NORMAL" or "GOV: THROTTLED" in real-time
- Owner name, mic name, last query/intent/latency
- Auto-reconnect on WebSocket drop (2s retry)
- Futuristic fonts (Orbitron, Rajdhani, Share Tech Mono)
- 3D orb color theme changes per performance mode

**WebSocket protocol (bidirectional):**
- **Server → Client:** `state_changed`, `log`, `system_stats`, `perf_mode`, `governor`, `restarting`, `init`
- **Client → Server:** `change_mode` (triggers config update + restart)

**Security:** CSP headers, X-Frame-Options, WebSocket origin restricted to localhost.

**Alternative:** `ui/floating_indicator.py` (782 lines) — Tkinter floating orb for minimal UI without a browser.

---

## 5. Event Flow — A Complete Query Lifecycle

```
User speaks: "Open Chrome"
       │
       ▼
[STT] Captures audio → Vosk offline model → "open chrome"
       │
       ▼
[EventBus] emit("speech_final", text="open chrome")
       │
       ▼
[Router._route()]
  1. sanitize_input("open chrome") → "open chrome" (clean)
  2. compress_query() → remove fillers
  3. IntentEngine.classify("open chrome")
     → IntentResult(intent="open_app", action="open_app",
                    action_args={"exe": "chrome", "name": "chrome"})
  4. _execute_action()
     a. SecurityPolicy.allow_action("open_app", {"name": "chrome"})
        → (True, "ok") — chrome is in SAFE_EXECUTABLES
     b. audit_log("open_app", "args={'exe': 'chrome'}")
     c. _dispatch_action("open_app", args)
        → app_actions.open_app("chrome")
        → subprocess.Popen(["chrome"])
  5. emit("response_ready", text="Done, Boss. Chrome is open.")
       │
       ▼
[TTS] on_response("Done, Boss. Chrome is open.")
  1. Check ack cache → not cached
  2. Edge-TTS generate audio → play
  3. emit("tts_complete")
       │
       ▼
[StateManager] SPEAKING → LISTENING (always-listen mode)
```

---

## 6. Performance Mode Lifecycle

```
User clicks "ULTRA" on dashboard
       │
       ▼
[WebSocket] Client sends: { action: "change_mode", mode: "ultra_lite" }
       │
       ▼
[WebDashboard._ws_handler] → calls mode_change_callback("ultra_lite")
       │
       ▼
[main._execute_mode_switch("ultra_lite")]
  1. Update config/settings.json: performance.mode = "ultra_lite"
  2. TTS speaks: "Switching to ultra lite mode, Boss.
                  Entering low resource mode. Restarting now."
  3. Wait 3 seconds
  4. Set _restart_requested = True
  5. Set shutdown_event → triggers graceful shutdown
       │
       ▼
[run_atom() crash-guard loop]
  1. Detects _restart_requested = True
  2. Clears flags, sleeps 2s
  3. Calls asyncio.run(main()) again
       │
       ▼
[main() startup with new config]
  1. Reads settings.json → mode = "ultra_lite"
  2. Applies ultra_lite intervals (health=300s, watcher=60s, maint=300s)
  3. TTS speaks: "I am ATOM, your personal AI operating system.
                  Currently running in ultra lite mode.
                  Entering low resource mode.
                  All systems are online and I'm ready for you, Boss."
```

**Voice command equivalent:** User says "switch to ultra lite mode" → Intent Engine detects `set_performance_mode` → same flow.

**Auto mode:** When `mode: "auto"`, a background loop samples CPU every 60s and automatically switches between full/lite/ultra_lite based on configurable thresholds (default: ≥70% → ultra_lite, ≥40% → lite, else full).

---

## 7. Data Flow Diagram

```
                    ┌─────────────┐
                    │  Microphone  │
                    └──────┬──────┘
                           │ audio
                           ▼
                    ┌─────────────┐
                    │  STT Async  │ ←── Vosk (offline) / Google (cloud)
                    └──────┬──────┘
                           │ text
                           ▼
              ┌────────────────────────┐
              │     AsyncEventBus      │
              │  (speech_final event)  │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  SecurityPolicy        │
              │  (sanitize_input)      │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Intent Engine (pkg)   │
              │  (11 regex modules)    │
              └────────┬───────┬───────┘
                       │       │
              matched  │       │ fallback
                       ▼       ▼
              ┌────────┐ ┌──────────────┐
              │ Action │ │  Cache/Memory │
              │Dispatch│ └──────┬───────┘
              └────┬───┘        │ miss
                   │            ▼
                   │   ┌─────────────────┐
                   │   │ Brain Selector   │
                   │   │ (Groq / Gemini)  │
                   │   └────────┬────────┘
                   │            │ streaming
                   ▼            ▼
              ┌─────────────────────────┐
              │     response_ready      │
              │  (AsyncEventBus event)  │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │      Edge TTS           │
              │  (speak + display)      │
              └────────────┬────────────┘
                           │
                    ┌──────┴──────┐
                    │   Speaker   │
                    └─────────────┘
```

---

## 8. Governor & Auto-Throttle Architecture

```
                    ┌──────────────────────────┐
                    │    HealthMonitor          │
                    │    (periodic check)       │
                    │                           │
                    │    CPU > threshold?        │
                    │    ┌─────┴─────┐          │
                    │    YES         NO         │
                    │    │           │           │
                    │    ▼           ▼           │
                    │  THROTTLE   NORMAL        │
                    └────┬──────────┬───────────┘
                         │          │
              ┌──────────┼──────────┼──────────┐
              ▼          ▼          ▼          ▼
        ┌──────────┐ ┌──────────┐ ┌─────────┐ ┌────────┐
        │HealthMon │ │SysWatcher│ │  TTS    │ │  UI    │
        │ ×2.5     │ │ ×3.0     │ │postproc │ │"THROT" │
        │ interval │ │ interval │ │ off/on  │ │ badge  │
        └──────────┘ └──────────┘ └─────────┘ └────────┘
```

**What the governor actually changes (not just visual):**

| Component | Normal | Throttled |
|-----------|--------|-----------|
| HealthMonitor check interval | base (e.g. 120s) | base × 2.5 (300s) |
| SystemWatcher poll interval | base (e.g. 30s) | base × 3.0 (90s) |
| TTS post-processing | config value | disabled (saves CPU) |
| Dashboard indicator | "GOV: NORMAL" | "GOV: THROTTLED" |

---

## 9. Technology Choices & Rationale

| Technology | Purpose | Why chosen | Alternatives considered |
|-----------|---------|------------|------------------------|
| **Python 3.11+** | Runtime | Async support, rich ecosystem, fast development | Node.js (weaker audio libs), Go (less flexible) |
| **asyncio** | Concurrency | Native Python async, no extra dependency | threading (GIL issues), multiprocessing (IPC overhead) |
| **AsyncEventBus** | IPC | Custom, lightweight, <200 lines, fits exactly | Redis pub/sub (overkill), RxPY (heavy) |
| **Vosk** | STT (default) | Offline, 100-200ms, no API key, low CPU | Whisper (heavy, 1GB+), Azure Speech (paid) |
| **Google Web Speech** | STT (fallback) | Free, no API key, higher accuracy | DeepSpeech (discontinued), Bing Speech (paid) |
| **Edge TTS** | TTS | Free, neural quality, HTTPS-only | ElevenLabs (paid), SAPI (low quality) |
| **pygame** | Audio playback | Cross-platform, supports Bluetooth | sounddevice (less reliable on Windows) |
| **Gemini Flash** | Cloud LLM | Free tier, streaming SSE, good quality | GPT-4 (paid), Claude (paid) |
| **Groq** | Cloud LLM | Extremely fast (200-500ms), free tier | Mistral (slower), OpenAI (paid) |
| **psutil** | System monitoring | Cross-platform, comprehensive | WMI (Windows-only, slow) |
| **pyautogui** | Desktop control | Simple API, screenshot support | win32api (lower-level, more code) |
| **aiohttp** | Web server | Async, WebSocket support, lightweight | FastAPI (heavier), Flask (sync) |
| **DuckDuckGo** | Web search | Free, no API key, no tracking | Google (needs API key), Bing (needs key) |
| **JSON files** | Persistence | Simple, debuggable, no DB server | SQLite (overkill), Redis (extra process) |
| **regex** | Intent classification | <5ms, deterministic, offline | spaCy NER (50-200ms, model download) |
| **numpy** | Audio processing | Fast array ops for normalization | scipy (heavier) |

---

## 10. Configuration System

### `config/settings.json` — Central Configuration

```json
{
    "owner": { "name": "Satyam", "title": "Boss" },
    "mic": { "device_name": null, "prefer_bluetooth": true },
    "stt": {
        "engine": "vosk",
        "vosk_model_path": "models/vosk-model-small-en-us-0.15",
        "sample_rate": 16000, "chunk_size": 2048,
        "post_tts_cooldown_ms": 800, "preload": true,
        "calibration_delay_s": 2.0, "min_energy_threshold": 600
    },
    "tts": {
        "engine": "edge", "edge_voice": "en-GB-RyanNeural",
        "edge_rate": "+0%", "edge_postprocess": false,
        "edge_ack_cache": true, "max_lines": 4
    },
    "ai": { "enabled": true, "model": "gemini-2.5-flash-lite", ... },
    "groq": { "model": "llama-3.1-8b-instant", ... },
    "security": {
        "mode": "strict", "audit_to_file": true,
        "require_confirmation_for": ["shutdown_pc", "restart_pc", ...]
    },
    "features": { "desktop_control": true, "file_ops": true, "llm": true, "system_analyze": true },
    "control": { "lock_mode": "off" },
    "performance": {
        "mode": "lite",
        "auto_threshold_high": 70,
        "auto_threshold_mid": 40,
        "health_check_interval_s": 120,
        "system_watcher_interval_s": 30,
        "maintenance_interval_s": 180,
        "proactive_alerts": false,
        "idle_reminder": false,
        "cpu_governor": true,
        "cpu_governor_threshold": 75
    }
}
```

### `config/commands.json` — Command Registry

Defines action metadata (category, confirmation requirement, description). Used by the command registry for runtime lookup.

### API keys (not in repo)

Do **not** keep secrets in a project `.env` file (avoids OneDrive/git accidents). Options:

- Windows **User** environment variables: `GEMINI_API_KEY`, `GROQ_API_KEY`
- Private file: `%USERPROFILE%\.atom\env` (KEY=value lines)
- Or `ATOM_ENV_FILE` pointing to a secrets file

See **`.env.example`** in the repo. Optional dev-only: `ATOM_ALLOW_DOTENV=1` enables loading `./.env` locally (never commit).

### Schema Validation

`core/config_schema.py` validates `settings.json` at startup using `jsonschema`. If `jsonschema` is not installed, falls back to basic type checks. Warns if API keys are embedded in settings instead of OS env / private env file.

---

## 11. File Structure (Complete)

```
ATOM/
├── main.py                          # Entry point, run_atom(), event wiring (1,037 lines)
├── requirements.txt                 # Python dependencies
├── .env.example                     # Template for where to put keys (no secrets)
├── .gitignore
├── README.md
│
├── config/
│   ├── settings.json                # All configuration
│   └── commands.json                # Command registry metadata
│
├── core/
│   ├── __init__.py                  # Re-exports
│   ├── async_event_bus.py           # Pub/sub backbone (173 lines)
│   ├── state_manager.py             # State machine (159 lines)
│   ├── cache_engine.py              # TTL LRU + Jaccard (169 lines)
│   ├── memory_engine.py             # Keyword memory (110 lines)
│   ├── security_policy.py           # Security gate (297 lines)
│   ├── health_monitor.py            # Watchdog + CPU governor (319 lines)
│   ├── metrics.py                   # Counters/latency (141 lines)
│   ├── pipeline_timer.py            # E2E latency (115 lines)
│   ├── personality.py               # Response templates (539 lines)
│   ├── config_schema.py             # JSON schema validation (552 lines)
│   ├── logging_setup.py             # Logging + privacy (90 lines)
│   ├── process_manager.py           # psutil process control (270 lines)
│   ├── task_scheduler.py            # Reminders (221 lines)
│   ├── system_watcher.py            # OS event monitor (312 lines)
│   ├── behavior_tracker.py          # Usage patterns, on-demand (136 lines)
│   ├── self_evolution.py            # Self-improvement, on-demand (228 lines)
│   ├── web_researcher.py            # DuckDuckGo search (158 lines)
│   ├── desktop_control.py           # pyautogui automation (180 lines)
│   ├── command_cache.py             # Intent LRU cache (89 lines)
│   ├── command_filter.py            # STT output filter (89 lines)
│   ├── command_registry.py          # Command metadata (128 lines)
│   ├── rate_limiter.py              # Token bucket (51 lines)
│   │
│   ├── intent_engine/               # Modular intent classifier (~1,350 lines)
│   │   ├── __init__.py              # IntentEngine class + classify chain (99 lines)
│   │   ├── base.py                  # IntentResult, grammar, replies (126 lines)
│   │   ├── meta_intents.py          # Greeting, thanks, status, exit (95 lines)
│   │   ├── info_intents.py          # Time, CPU, RAM, battery, etc. (197 lines)
│   │   ├── app_intents.py           # Open/close/list apps (161 lines)
│   │   ├── media_intents.py         # YouTube, volume, music (95 lines)
│   │   ├── system_intents.py        # Lock, screenshot, shutdown (90 lines)
│   │   ├── desktop_intents.py       # Scroll, click, type, hotkey (141 lines)
│   │   ├── file_intents.py          # Folder create, move, copy (52 lines)
│   │   ├── network_intents.py       # Search, URL, weather (29 lines)
│   │   └── os_intents.py            # Self-check, perf mode, reminders (252 lines)
│   │
│   └── router/
│       ├── __init__.py
│       ├── router.py                # 3-layer routing (1,063 lines)
│       ├── app_actions.py           # App open/close/list (81 lines)
│       ├── file_actions.py          # File create/move/copy (66 lines)
│       ├── media_actions.py         # Volume/play/YouTube (75 lines)
│       ├── network_actions.py       # URL/search/weather (70 lines)
│       ├── system_actions.py        # Lock/screenshot/shutdown (154 lines)
│       └── utility_actions.py       # Window/clipboard/timer (68 lines)
│
├── voice/
│   ├── __init__.py
│   ├── stt_async.py                 # Vosk/Google STT (837 lines)
│   ├── tts_edge.py                  # Edge Neural TTS (775 lines)
│   ├── tts_async.py                 # SAPI TTS fallback (251 lines)
│   ├── mic_manager.py               # Mic ownership (89 lines)
│   ├── mic_selector.py              # Auto mic selection (373 lines)
│   ├── audio_pipeline.py            # Audio processing (143 lines)
│   ├── speech_detector.py           # Text corrections (104 lines)
│   └── voice_profiles.py            # SSML emotion (76 lines)
│
├── cursor_bridge/
│   ├── __init__.py
│   ├── local_brain_controller.py    # Offline LLM adapter (event bus)
│   └── structured_prompt_builder.py # Prompt builder (190 lines)
│
├── context/
│   ├── __init__.py
│   ├── context_engine.py            # Window/clipboard context (130 lines)
│   └── privacy_filter.py            # Redaction patterns (87 lines)
│
├── brain/
│   ├── __init__.py
│   ├── mini_llm.py                  # Local GGUF via llama.cpp (Llama 3.2 3B, etc.)
│   └── prompt_engine.py             # ChatML prompts (legacy TinyLlama format)
│
├── vision/
│   ├── __init__.py
│   └── face_recognizer.py           # Owner detection (231 lines)
│
├── ui/
│   ├── __init__.py
│   ├── web_dashboard.py             # aiohttp server + WebSocket (362 lines)
│   ├── floating_indicator.py        # Tkinter orb (782 lines)
│   └── dashboard/
│       └── index.html               # Web UI + 3D orb + perf controls (913 lines)
│
├── models/
│   └── vosk-model-small-en-us-0.15/ # Vosk offline speech model (~50 MB)
│
├── scripts/
│   └── enroll_owner_face.py         # Face enrollment (83 lines)
│
├── tests/
│   ├── test_all_components.py       # Unit tests (490 lines)
│   ├── test_heavy_deployment.py     # Stress tests (1,568 lines)
│   ├── test_state_machine.py        # State transitions (255 lines)
│   ├── test_context_engine.py       # Context tests (132 lines)
│   └── test_mic_manager.py          # Mic tests (176 lines)
│
├── docs/
│   ├── README.md
│   ├── ATOM_End_to_End_Performance_Report.md  # Voice pipeline + timings (canonical)
│   ├── ATOM_System_Diagram.svg
│   ├── ATOM_Benchmark_Snapshot.json
│   └── ATOM_Full_System_Review.md   # This document
├── tools/
│   └── e2e_benchmark.py             # Reproducible module timing benchmark
│
└── logs/
    ├── atom.log                     # Application log (rotating)
    ├── atom_metrics.log             # Metrics log
    ├── audit.log                    # Security audit trail
    ├── behavior.json                # User patterns
    ├── memory.json                  # Q&A memory
    └── tasks.json                   # Scheduled tasks
```

**Total:** ~17,300 lines of Python across 76 files + 913 lines HTML.

---

## 12. Dependencies

### Core (required)
| Package | Version | Purpose |
|---------|---------|---------|
| SpeechRecognition | ≥3.10 | Microphone capture + Google Web Speech fallback |
| PyAudio | ≥0.2.14 | Microphone capture |
| vosk | ≥0.3.45 | Offline speech recognition (default STT engine) |
| numpy | ≥1.24 | Audio post-processing |
| edge-tts | ≥6.1 | Microsoft Neural TTS |
| pygame | ≥2.5 | Audio playback |
| psutil | ≥5.9 | System monitoring + CPU governor |
| keyboard | ≥0.13.5 | Global hotkey (Ctrl+Alt+A) |
| aiohttp | ≥3.9 | Web dashboard server |
| jsonschema | ≥4.17 | Config validation |
| truststore | ≥0.9 | Corporate SSL certificates |
| mss | ≥9.0 | Screen capture |
| Pillow | ≥10.0 | Image processing |
| comtypes | ≥1.4 | SAPI TTS fallback |
| cffi | ≥1.0 | PyAudio dependency |

### Optional
| Package | Purpose |
|---------|---------|
| soxr | Better audio resampling |
| opencv-python | Face detection |
| face_recognition | Owner identification |

---

## 13. Security Model

```
                     ┌──────────────────────────────────┐
                     │     SecurityPolicy                │
                     │                                    │
Voice input ─────▶  │  1. sanitize_input()               │
                     │     - Cap 2000 chars               │
                     │     - Strip ; & | $ ` <script      │
                     │                                    │
Action dispatch ──▶  │  2. allow_action()                 │
                     │     - Check lock_mode               │
                     │     - Check feature flags           │
                     │     - Check executable list         │
                     │     - Check close-target list       │
                     │     - Block power in strict         │
                     │                                    │
App open ─────────▶  │  3. is_safe_executable()           │
                     │     - 35+ allowed apps              │
                     │                                    │
App close ────────▶  │  4. is_safe_close_target()         │
                     │     - 20+ allowed processes         │
                     │                                    │
Shell command ────▶  │  5. is_safe_command()               │
                     │     - 20+ blocked patterns          │
                     │                                    │
Hotkey ───────────▶  │  6. is_safe_hotkey()                │
                     │     - safe / confirm / block        │
                     │                                    │
File path ────────▶  │  7. path_allowed()                  │
                     │     - Only user home / CWD          │
                     │                                    │
All actions ──────▶  │  8. audit_log()                     │
                     │     - logs/audit.log                │
                     └──────────────────────────────────┘
```

---

## 14. Embedding ATOM as a Brain

ATOM provides a single programmatic entry point:

```python
from main import run_atom

# Minimal — runs with default settings.json
run_atom()

# Restricted — disable desktop control, lock to safe-only intents
run_atom({
    "features": {"desktop_control": False, "file_ops": False},
    "control": {"lock_mode": "safe_only"},
})

# Custom voice and model
run_atom({
    "tts": {"edge_voice": "en-US-GuyNeural"},
    "ai": {"model": "gemini-2.0-flash"},
})
```

Config overrides are merged on top of `config/settings.json`. The crash-guard (exponential backoff, max 5 retries) and graceful restart (on mode change) are included.

---

## 15. Performance Modes & CPU Governor

### Problem
The target hardware is a corporate laptop (i7-1185G7, 4C/8T, 32 GB RAM) that
is already under moderate load (~34-50% CPU) from Chrome, Cursor IDE, corporate
security tools, and ~296 background processes. Running ATOM at full capacity
pushes CPU to 70-95%, causing lag, thermal throttling, and fan noise.

### Solution: 4-Tier Performance Modes

ATOM ships with a `performance` section in `config/settings.json`:

```json
"performance": {
    "mode": "lite",
    "auto_threshold_high": 70,
    "auto_threshold_mid": 40,
    "health_check_interval_s": 120,
    "system_watcher_interval_s": 30,
    "maintenance_interval_s": 180,
    "proactive_alerts": false,
    "idle_reminder": false,
    "cpu_governor": true,
    "cpu_governor_threshold": 75
}
```

| Setting | full | lite (default) | ultra_lite | auto |
|---------|------|-------|------------|------|
| Health watchdog interval | 60s | 120s | 300s | CPU-driven |
| System watcher poll | 10s | 30s | 60s | CPU-driven |
| Maintenance cycle | 120s | 180s | 300s | CPU-driven |
| Proactive voice alerts | Yes | No | No | No |
| Idle reminders | Yes | No | No | No |
| Health logging | Every cycle | Every cycle | Disabled | Per active mode |
| CPU Governor | Configurable | Configurable | Configurable | Configurable |
| Mode selection | Fixed | Fixed | Fixed | CPU thresholds |

**Choosing a mode:**
- **full** — Use when ATOM is the primary task (Cursor/Chrome closed).
- **lite** — Default. Best for running alongside Chrome + Cursor on 4-core laptops.
- **ultra_lite** — Extreme CPU saving. Only voice-in / voice-out, minimal background.
- **auto** — ATOM monitors CPU and switches between full/lite/ultra_lite automatically.

**How to switch modes:**
1. **Dashboard UI:** Click FULL / LITE / ULTRA / AUTO button → ATOM saves config, speaks confirmation with personality phrase, restarts.
2. **Voice command:** "Switch to ultra lite mode" → same flow.
3. **Direct config edit:** Change `performance.mode` in `settings.json`, restart ATOM.

**Personality phrases per mode:**
| Mode | Phrase |
|------|--------|
| Full | "All systems at maximum performance." |
| Lite | "Optimizing for efficiency." |
| Ultra Lite | "Entering low resource mode." |
| Auto | "Adapting to system load." |

### CPU Governor (Closed-Loop Auto-Throttle)

When `cpu_governor: true`, the HealthMonitor watches system-wide CPU usage:

1. **Throttle**: If CPU > `cpu_governor_threshold` for 2 consecutive checks:
   - HealthMonitor check interval → base × 2.5
   - SystemWatcher poll interval → base × 3.0
   - TTS post-processing → disabled (saves CPU on audio normalization)
   - Dashboard → "GOV: THROTTLED" indicator
   - Emits `governor_throttle` event

2. **Restore**: When CPU drops below threshold for 3 consecutive checks:
   - All intervals restored to base values
   - TTS post-processing → restored to config value
   - Dashboard → "GOV: NORMAL" indicator
   - Emits `governor_normal` event

This means ATOM automatically backs off when you're compiling, running builds,
or doing other CPU-heavy work — and speeds back up when CPU is free.

### Auto Performance Mode

When `performance.mode` is set to `"auto"`, ATOM runs a background loop that
samples system CPU every 60 seconds and applies thresholds:

- CPU ≥ `auto_threshold_high` (default 70) → switch to **ultra_lite**
- CPU ≥ `auto_threshold_mid` (default 40) → switch to **lite**
- Otherwise → **full**

On a mode change, ATOM saves the new mode to config, speaks the personality
phrase (e.g. "System load is high. Switching to ultra lite mode."), and
performs a graceful restart into that mode.

### Graceful Restart Mechanism

Mode changes use a clean restart lifecycle:

```
1. _execute_mode_switch() sets _restart_requested = True
2. shutdown_event.set() triggers orderly cleanup:
   - Stop scheduler, watcher, health monitor
   - Cancel maintenance task
   - Persist behavior + evolution data
   - Unhook keyboard hotkeys
   - Clear event bus
   - Shutdown STT, TTS, LLM connections
   - Shutdown UI
3. run_atom() detects _restart_requested = True
4. Clear flags, sleep 2s
5. asyncio.run(main()) → fresh start with new config
```

No process leaks, no partial state corruption, no zombie processes.

### Impact

| Without Lite Mode | With Lite Mode |
|------------------|----------------|
| Health check every 60s | Every 120s (50% fewer psutil calls) |
| System watcher every 10s | Every 30s (67% fewer Win32 API calls) |
| Proactive alerts (battery, idle) every cycle | Disabled (zero overhead) |
| ~15-20% CPU from background tasks | ~5-8% CPU from background tasks |

---

## 16. Current Limitations & Future Work

| Area | Limitation | Potential improvement |
|------|-----------|---------------------|
| **STT** | Vosk small model; accuracy lower than Whisper for complex speech | Whisper (GPU needed) or Vosk large model |
| **TTS** | Edge TTS is cloud-dependent; occasional network failures | Local Piper TTS or Coqui TTS |
| **LLM** | Free tier rate limits (25-30 req/min) | Self-hosted Ollama with Llama 3 |
| **Memory** | Keyword-based, no semantic search | Add sentence embeddings (all-MiniLM-L6-v2) |
| **Vision** | Disabled, requires opencv + face_recognition | Enable for owner authentication |
| **Multi-modal** | Text-only LLM interaction | Add image/document understanding |
| **Calendar** | No integration | Windows Calendar API or Google Calendar |
| **Email** | No integration | Outlook COM automation |
| **Multi-language** | English only | Multi-language STT + TTS |
| **Authentication** | No voice/face auth for owner_only mode | Add speaker verification or face recognition |
| **Persistence** | JSON files, no query history | SQLite for structured history |
| **Testing** | Manual testing mostly | CI/CD with pytest, mock event bus |
| **Mode switching** | Restart-based (5-15s friction) | Hot-reload config without restart |

---

## 17. Consolidation Opportunities

These are known areas where shared logic could be extracted to reduce duplication:

| Area | Files involved | Suggestion |
|------|---------------|------------|
| (removed) | Cloud SSE clients | Replaced by local LLM only |
| URL opening | network_actions, media_actions | `open_url_in_browser()` helper |
| Key simulation | media_actions, utility_actions | `send_key_press(vk)` helper |

---

## 18. Summary for ChatGPT

**ATOM is a ~17,300-line Python AI OS layer** running on Windows. It is:

- **Event-driven** — AsyncEventBus pub/sub, 6-state state machine, ~25 distinct event types
- **3-layer intelligent** — Regex intent (<5ms) → Cache/memory → **Local LLM** (GGUF via llama.cpp when `brain.enabled=true`)
- **Voice-controlled** — Vosk offline STT → TTS: SAPI offline by default, or Edge (optional, needs network)
- **Security-first** — Single security gate, allowlists, audit logging, config-driven policies, input sanitization
- **Desktop-aware** — pyautogui control, psutil monitoring, Win32 APIs for context, Bluetooth mic/speaker handling
- **Self-adaptive** — 4-tier performance modes (full/lite/ultra_lite/auto) + closed-loop CPU governor that throttles HealthMonitor (×2.5), SystemWatcher (×3.0), and TTS post-processing
- **Self-diagnosing** — Voice-triggered system check reports status of all subsystems (STT, TTS, LLMs, CPU, RAM, governor)
- **UI-controlled** — WebSocket control plane with mode selector buttons (FULL/LITE/ULTRA/AUTO), governor indicator, 3D orb, live conversation, and system stats
- **Embeddable** — `run_atom(config_overrides)` entry point with feature flags, lock modes, and crash-guard + graceful restart
- **On-demand extras** — Self-evolution engine, behavior tracker, web researcher (all off by default, only run when asked)

The codebase is modular (76 Python files, 12 directories), uses no frameworks, and every module communicates through the event bus. It runs in a single async Python process with a 2-thread executor for blocking operations.

**Owner:** Satyam  
**Status:** Production-grade for personal use on a corporate laptop.
