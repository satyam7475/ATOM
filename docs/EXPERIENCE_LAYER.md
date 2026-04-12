# ATOM — Experience layer (voice, personality, dashboard)

**Scope:** How the operator *feels* the system: speech I/O, assistant stance, and the web HUD. Complements stability (`ATOM_IMPLEMENTATION_PLAN.md` Phase 9).

## Voice pipeline

| Setting | Location | Notes |
|---------|----------|--------|
| STT engine | `config/settings.json` → `stt.engine` | e.g. `macos_native`, Faster-Whisper options per `voice/stt_async.py`. |
| TTS engine | `tts.engine` | e.g. `macos_native`; Edge voices use `edge_voice`, `edge_rate`. |
| Mic | `mic.device_name`, `mic.prefer_bluetooth` | Resolved at startup; shown on dashboard **Connections** pod. |

The web dashboard init payload includes **`voice_mode`** / **`voice_note`** (browser is text-first unless dev fallback). Production voice path is the bundled app path described in-dashboard.

## Personality & assistant stance

| Setting | Location |
|---------|----------|
| Assistant mode | `control.assistant_mode` + runtime **`AssistantModeManager`** — `command_only`, `hybrid`, `conversational`. |
| Brain profile | `BrainModeManager` profiles (OPT / FULL) — performance vs depth. |
| Lock / security | `control.lock_mode`, `security.mode` — see [`SECURITY_SETUP.md`](./SECURITY_SETUP.md). |

Runtime pod buttons send WebSocket commands; **`execution_state`** pushes live **`assistant_mode`** plus STT/TTS engine labels for the **Runtime Truth** experience line.

## Dashboard (web)

- **`ui/dashboard/index.html`** — HUD, orb, cockpit cards, **Runtime Truth** (execution + cache + voice/assistant summary).
- **`ui/web_dashboard.py`** — WebSocket feed, `set_execution_state_provider` from `main.py`.

Polish for Phase 9: page title, input placeholder, and experience line fed from `main._execution_state_payload()`.
