# ATOM OS — Company laptop full test session report

**Profile:** Full feature testing on corporate hardware with **`security.mode: strict`**.  
**Report generated:** 2026-03-20 (session bootstrap from automated checks + startup log).  
**Owner:** Satyam (Boss)

---

## 1. Configuration applied

| Area | Value | Notes |
|------|--------|--------|
| `security.mode` | **`strict`** | Power actions (`shutdown_pc`, `restart_pc`, `logoff`, `sleep_pc`) blocked at policy gate unless your code path overrides; confirm list active for destructive ops. |
| `security.audit_to_file` | `true` | Review `logs/audit.log` after sensitive tests. |
| `features.desktop_control` | **`true`** | Full desktop automation surface enabled. |
| `features.file_ops` | **`true`** | Create/move/copy paths allowed per policy + paths. |
| `features.llm` | **`true`** | LLM routing enabled (cloud off — see below). |
| `features.system_analyze` | **`true`** | System analysis intents enabled. |
| Cloud AI | Not used | ATOM is fully offline — no cloud API keys. |
| `brain.enabled` | `true` | Local **Llama-3.2-3B-Instruct-Q4_K_M** GGUF, **CPU** (`n_gpu_layers: 0`). |
| `control.assistant_mode` | `hybrid` | Commands + open queries can hit **local** brain. |
| `control.lock_mode` | `off` | Runtime brain/assistant switches allowed (dashboard/voice). |
| `performance.mode` | `lite` | Company laptop profile. |
| `autonomy.enabled` | `true` | `auto_execute_threshold: 0.95`, `suggest_threshold: 0.72`. |
| `cognitive.*` | goals / predictions / behavior / optimizer **on** | Full cognitive layer. |
| `vision.enabled` | `false` | Camera off (typical for office). |
| `developer.company_laptop_full_test` | **`true`** | Marker that this `settings.json` is aligned for your full laptop test. |

**Schema validation:** `validate_config(settings.json)` → **pass** (no errors).

---

## 2. Startup verification (this session)

ATOM was started with `py -3.11 main.py` from the project root. Log excerpts:

| Checkpoint | Result |
|-------------|--------|
| Configuration validated | OK |
| SecurityPolicy | `mode=strict`, `lock=off` |
| Cloud AI | Disabled (`ai.enabled=false`) |
| Local brain | Enabled; model loaded **~875ms** (balanced profile, ctx 1536) |
| TTS | Edge neural `en-GB-RyanNeural` ready |
| STT | Vosk small-en-us; mic detected (Intel Smart Array) |
| Web dashboard | **http://127.0.0.1:8765** — client connected |
| Autonomy | Started `auto>=0.95`, `suggest>=0.72` |
| Cognitive | Goal engine, behavior model, prediction, self-optimizer, personality modes started |
| Hotkey | `Ctrl+Alt+A` registered |

**Operational URL:** http://127.0.0.1:8765  

---

## 3. What `strict` means for your manual tests

- **Blocked without extra policy:** `shutdown_pc`, `restart_pc`, `logoff`, `sleep_pc` (see `SecurityPolicy.allow_action` in `core/security_policy.py`).
- **Confirmation-gated** (per `require_confirmation_for`): includes `empty_recycle_bin`, `kill_process`, `close_app`, plus power actions when not in strict block path — exercise voice/UI and confirm prompts behave as expected.
- **Allowlists still apply:** `open_app` / `close_app` only for configured safe executables/process names.

---

## 4. Recommended manual test checklist (fill after your session)

Use this table during testing; append notes and pass/fail.

| # | Test | Expected | Pass? | Notes |
|---|------|----------|-------|-------|
| 1 | Open dashboard | Orb + panels load, WS connects | | |
| 2 | Voice: “what time is it” | Intent → spoken response | | |
| 3 | Voice: “open notepad” | Notepad launches if allowlisted | | |
| 4 | Hybrid chat | Non-command phrase → local brain reply (may take seconds on CPU) | | |
| 5 | Dashboard RUNTIME | Brain profile + assistant mode change (if not locked) | | |
| 6 | `Ctrl+Alt+A` | Listening toggles / activates | | |
| 7 | Strict: “shutdown” / restart | Blocked or no unsafe execution | | |
| 8 | Clipboard context | Only if comfortable — verify redaction if cloud ever enabled later | | |
| 9 | `logs/audit.log` | Entries on blocked/sensitive attempts | | |
| 10 | Cognitive | “Set a goal …” — goal flow without cloud | | |

---

## 5. Static analysis summary (background)

| Layer | Role |
|--------|------|
| `main.py` | Validates config, wires bus, router, brain, UI, cognitive. |
| `core/security_policy.py` | Single gate: `allow_action`, path/shell/hotkey helpers, audit. |
| `core/router/router.py` | Intent → action dispatch with security checks. |
| `ui/web_dashboard.py` | Localhost-only bind, CSP headers, WS origin check. |
| Local brain | `cursor_bridge/local_brain_controller.py` + GGUF via configured `model_path`. |

---

## 6. Post-session: your notes

*(Add observations, latency, crashes, IT policy issues, or feature gaps here.)*

- Session date/time:
- Build / commit:
- Issues:
- Next fixes:

---

*This report is a living document; copy a row from §4 into §6 or attach screenshots/log snippets as needed.*
