# ATOM — Experience layer (voice, personality, dashboard)

**Scope:** How the operator *feels* the system: speech I/O, assistant stance, and the web HUD. Complements stability (`ATOM_IMPLEMENTATION_PLAN.md` Phase 9).

## Voice pipeline

| Setting | Location | Notes |
|---------|----------|--------|
| STT engine | `config/settings.json` → `stt.engine` | e.g. `macos_native`, Faster-Whisper options per `voice/stt_async.py`. |
| TTS engine | `tts.engine` | e.g. `macos_native`; Edge voices use `edge_voice`, `edge_rate`. |
| Mic | `mic.device_name`, `mic.prefer_bluetooth` | Resolved at startup; shown on dashboard **Connections** pod. |

The web dashboard init payload includes **`voice_mode`** / **`voice_note`** (browser SpeechRecognition is the reliable path when **`ATOM_LAUNCH_MODE=venv`**). **Native** `SFSpeechRecognizer` needs the process to be **`ATOM.app`**’s executable (`atom_python`) with usage strings in **`Info.plist`** — use **`Run ATOM.command`** when the bundle launcher self-test passes, or rebuild/sign with **`scripts/build_atom_app_launcher.sh`**.

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

## Operator checklist (launch & reliability)

| Step | Action |
|------|--------|
| 1 | Create/use **`.venv`** and install deps: `pip install -r requirements-desktop.txt` (see repo `README.md`). |
| 2 | **Recommended:** double-click **`Run ATOM.command`** or run it from Terminal — uses venv if the bundle launcher fails; logs to **`logs/atom_run_command.log`**. |
| 3 | **Background agent (optional):** `bash scripts/install_atom_launchagent.sh` — uses **`scripts/atom_run.sh`** + venv (stable); stdout/stderr under **`logs/launchagent.*.log`**. |
| 4 | **Double-click `ATOM.app`:** runs **`Contents/MacOS/atom_python`** only. If it misbehaves, use **`Run ATOM.command`** or rebuild: `bash scripts/build_atom_app_launcher.sh` (prefer a repo copy **outside iCloud Desktop** if **codesign** complains about metadata). |
| 5 | **Native macOS STT:** requires bundle process + plist usage keys; **venv** launch sets **`ATOM_LAUNCH_MODE=venv`** and falls back messaging in **`voice/stt_macos.py`**. |
