# ATOM -- Personal Cognitive AI Operating System

**Owner:** Satyam ("Boss")
**Platform:** macOS (Apple Silicon, M-series)
**What it is:** A local-first, JARVIS-style voice OS that perceives (voice, system state), reasons (intent, cache, memory, RAG, local LLM), acts (security-gated tools), and learns (behavior, habits, feedback metrics). Not a chatbot -- a continuously running AI OS.

**Docs:** [`docs/ATOM_M5_EVOLUTION_PLAN.md`](docs/ATOM_M5_EVOLUTION_PLAN.md) -- current active roadmap · [`docs/ATOM_VS_JARVIS_SCORECARD.md`](docs/ATOM_VS_JARVIS_SCORECARD.md) — module rating vs “Jarvis” reference
**Architecture:** [`docs/architecture/INDEX.md`](docs/architecture/INDEX.md) -- modular architecture modules
**Implementation tracking:** [`docs/ATOM_IMPLEMENTATION_PLAN.md`](docs/ATOM_IMPLEMENTATION_PLAN.md) — phased work, **ACT** protocol, and change log. Say **“Next step ACT”** or **“ACT step `runtime-truth`”** (any step ID from that doc) so the assistant implements that work and updates the tracker.

---

## Features

| Area | Description |
|------|-------------|
| **Owner-first** | Configured in `config/settings.json` (`owner.name`, `owner.title`). |
| **Instant path** | Intent engine + cache for sub-millisecond command routing. |
| **Local brain** | MLX local brain (`Qwen3 8B MLX 4-bit`) with shared fast/primary roles on Apple Silicon. |
| **Cognitive kernel** | Central routing across DIRECT / CACHE / QUICK / FULL / DEEP execution paths. |
| **Security** | Fernet-encrypted credentials, SecurityPolicy gating on all actions, audit logging. |
| **UI** | aiohttp web dashboard + WebSocket on localhost with token auth. |

---

## Requirements

- **macOS** on Apple Silicon (M1/M2/M3/M4/M5)
- **Python 3.11+**
- **Microphone** for voice

---

## Install

```bash
cd ATOM
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements-desktop.txt
```

---

## Configure

1. Edit `config/settings.json`.
2. Set `brain.enabled: true` and configure model paths for MLX models.
3. Run `python scripts/setup_api_keys.py` to set up encrypted API credentials.
4. See `core/config_schema.py` for the full schema reference.

### Product tiers (local vs cloud)

| `deployment.product_tier` | Intent | Typical settings |
|---------------------------|--------|------------------|
| `local_only` | Privacy-first; no cloud calls | `cloud.enabled: false`, `semantic_cache.enabled: false` optional |
| `balanced` | Default; local brain, cloud off until you opt in | `cloud.enabled: false` or true with low quotas |
| `cloud_augmented` | Hard questions / buddy tone via Gemini when needed | `cloud.enabled: true` |

**Source of truth:** `cloud.enabled` (and related keys under `cloud` in settings) actually gates Gemini and cognitive escalation. `product_tier` is a **label** for you and the dashboard—keep it aligned when you change posture.

---

## Run

```bash
source .venv/bin/activate   # if not already active
python main.py
```

- **Finder / double-click:** use **`Run ATOM.command`** in the repo root (venv-first; optional bundle launcher when `ATOM.app/Contents/MacOS/atom_python` passes self-test). Logs: `logs/atom_run_command.log`.
- **Always-on (optional):** `bash scripts/install_atom_launchagent.sh` installs a **launchd** agent that runs **`scripts/atom_run.sh`**, which now prefers the bundle launcher for native macOS STT and falls back to `.venv/bin/python`.
- **Double-click `ATOM.app`:** launches the embedded **`atom_python`** binary only; if that fails, prefer **`Run ATOM.command`** or rebuild with `bash scripts/build_atom_app_launcher.sh`.

- Dashboard: `http://127.0.0.1:<port>/` (port from `ui.web_port` in settings).
- Health endpoint: `GET /v7/health` on the same port.

### Voice (macOS)

- **Prefer `Run ATOM.command` when the bundle self-test passes** — `ATOM.app`’s `atom_python` is the process that carries **Speech Recognition + Microphone** usage strings for Apple’s on-device STT. Plain `venv` Python often falls back to **Faster-Whisper** (still works; different stack).
- **Current default voice mode:** `voice.activation_mode` is `always_on`, so ATOM keeps the command path hot instead of waiting for a wake phrase. Set it back to `wake_word` if you want passive gating again.
- **Production STT preflight:** when `stt.engine` is `whisper_cpp`, boot now refuses to start unless `pywhispercpp`, `sounddevice`, `webrtcvad`, `numpy`, and the configured GGML model are present. Install the model with `python scripts/install_whisper_model.py`.
- **Permissions:** System Settings → Privacy & Security → **Microphone** and **Speech Recognition** — enable for the app you use to launch ATOM (Terminal, Cursor, or the embedded `atom_python`).
- **Duplex / interruptibility:** `stt.barge_in_during_speak` is enabled by default, so ATOM can hear you during TTS for real interruption. Headphones are still the cleanest setup to avoid echo.
- **After startup speech:** the engine waits until state is **LISTENING** (not during TTS) and applies **`stt.post_tts_cooldown_ms`** before reopening the mic so the greeting does not fight the capture path.
- **Voice health strip** under the top bar shows STT engine, permissions, lifecycle state, tier, and last error (WebSocket `state_diff`).

### Testing (golden path)

```bash
python3 tests/golden_path_e2e.py
```

Optional macOS live-mic smoke (may **SKIP** under venv): `python3 tests/golden_path_e2e.py --live-mic`

---

## Project layout

```
ATOM/
├── main.py                 # Entry point: async main loop
├── config/                 # Settings and example configs
├── core/                   # Runtime: router, state, scheduler, security, cognition, RAG
├── brain/                  # MLX LLM, memory graph
├── cursor_bridge/          # Agentic loop: LocalBrainController, prompt builder
├── context/                # Perception: screen reader, privacy filter
├── voice/                  # STT, TTS, wake word, mic
├── ui/                     # Web dashboard (aiohttp + WebSocket)
├── tests/                  # Test suite
├── scripts/                # Setup and utility scripts
├── docs/                   # Architecture docs and evolution plan
├── models/                 # MLX model weights (gitignored)
├── requirements.txt
└── requirements-desktop.txt
```

---

## Documentation

| Document | Content |
|----------|---------|
| [`docs/ATOM_M5_EVOLUTION_PLAN.md`](docs/ATOM_M5_EVOLUTION_PLAN.md) | Active roadmap and evolution plan |
| [`docs/ATOM_M5_AIR_HLD_LLD_WHITEPAPER.md`](docs/ATOM_M5_AIR_HLD_LLD_WHITEPAPER.md) | HLD/LLD technical whitepaper |
| [`docs/architecture/INDEX.md`](docs/architecture/INDEX.md) | Module index mapping code to architecture |

---

## License / ownership

ATOM is Satyam's personal cognitive OS project.
