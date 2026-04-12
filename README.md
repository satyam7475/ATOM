# ATOM -- Personal Cognitive AI Operating System

**Owner:** Satyam ("Boss")
**Platform:** macOS (Apple Silicon, M-series)
**What it is:** A local-first, JARVIS-style voice OS that perceives (voice, system state), reasons (intent, cache, memory, RAG, local LLM), acts (security-gated tools), and learns (behavior, habits, feedback metrics). Not a chatbot -- a continuously running AI OS.

**Docs:** [`docs/ATOM_M5_EVOLUTION_PLAN.md`](docs/ATOM_M5_EVOLUTION_PLAN.md) -- current active roadmap
**Architecture:** [`docs/architecture/INDEX.md`](docs/architecture/INDEX.md) -- modular architecture modules
**Implementation tracking:** [`docs/ATOM_IMPLEMENTATION_PLAN.md`](docs/ATOM_IMPLEMENTATION_PLAN.md) — phased work, **ACT** protocol, and change log. Say **“Next step ACT”** or **“ACT step `runtime-truth`”** (any step ID from that doc) so the assistant implements that work and updates the tracker.

---

## Features

| Area | Description |
|------|-------------|
| **Owner-first** | Configured in `config/settings.json` (`owner.name`, `owner.title`). |
| **Instant path** | Intent engine + cache for sub-millisecond command routing. |
| **Local brain** | MLX dual-model (Qwen3-4B primary + Qwen3-1.7B fast) on Apple Silicon. |
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

---

## Run

```bash
python main.py
```

- Dashboard: `http://127.0.0.1:<port>/` (port from `ui.web_port` in settings).
- Health endpoint: `GET /v7/health` on the same port.

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
