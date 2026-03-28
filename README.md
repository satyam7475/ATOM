# ATOM — Personal Cognitive AI Operating System

**Owner:** Satyam (“Boss”)  
**What it is:** A local-first, JARVIS-style voice OS — not a thin chatbot. It perceives (voice, system state), reasons (intent → cache/memory → optional RAG/graph → local LLM), acts (security-gated tools), and learns (behavior, habits, V7 feedback metrics).

**Docs:** Deep architecture → [`ATOM_ARCHITECTURE_BLUEPRINT.md`](ATOM_ARCHITECTURE_BLUEPRINT.md)  
**Code review & desktop plan:** [`docs/ATOM_CODE_REVIEW_AND_DESKTOP_PLAN.md`](docs/ATOM_CODE_REVIEW_AND_DESKTOP_PLAN.md)

---

## Features (summary)

| Area | Description |
|------|-------------|
| **Owner-first** | Configured in `config/settings.json` (`owner.name`, `owner.title`). |
| **Instant path** | Intent engine + cache for sub-millisecond-class command routing where applicable. |
| **Offline brain** | Local GGUF via `llama-cpp-python` when `brain.enabled` and model path set. |
| **V7 intelligence** | Runtime modes (FAST/SMART/DEEP/SECURE), `V7RuntimeContext`, feedback engine, mode stability, bounded prefetch, graph-first RAG with validation, `/v7/health` observability. |
| **Security** | `SecurityPolicy` + `allow_action` on routed actions; treat policy changes as high-risk. |
| **UI** | aiohttp web dashboard (default) + WebSocket; optional floating indicator mode. |

---

## Requirements

- **Python 3.11+** (64-bit recommended)
- **Microphone** for voice
- **No cloud API keys** for core offline operation (optional Edge TTS uses network if selected)

---

## Install (personal desktop)

Use a dedicated virtual environment.

```bash
cd ATOM
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-desktop.txt
```

- **Canonical pinned set:** [`requirements-desktop.txt`](requirements-desktop.txt) (keep in sync with [`requirements.txt`](requirements.txt) when bumping versions).
- **Optional dev tools (tests):** [`requirements-dev.txt`](requirements-dev.txt)

---

## Configure

1. Copy or edit **`config/settings.json`**.
2. For a desktop-oriented baseline, see **`config/settings.desktop.example.json`**.
3. For corporate-style restrictions, see **`config/settings.corporate.example.json`** and [`docs/ATOM_Deployment_Profiles.md`](docs/ATOM_Deployment_Profiles.md).
4. Set **`brain.model_path`** to your GGUF file when using the local LLM.
5. **`v7_intelligence`** — thresholds for health, prefetch, mode stability, observability (see schema in `core/config_schema.py`).

---

## Run

```bash
python main.py
```

- Dashboard (default): `http://127.0.0.1:<port>/` — port from `ui.web_port` in settings (often **8765**).
- **V7 health (JSON):** `GET http://127.0.0.1:<port>/v7/health` when the dashboard is running.
- **Hotkey:** Ctrl+Alt+A toggles listening / resume; use dashboard **UNSTICK** if the state machine hangs.

If the browser does not open automatically, use Cursor Simple Browser or open the URL manually — see [`docs/CURSOR_DASHBOARD.md`](docs/CURSOR_DASHBOARD.md) if present.

---

## Project layout (abbreviated)

```
ATOM/
├── main.py                 # Entry: run_atom() / asyncio main
├── config/
│   ├── settings.json       # Active config
│   └── *.example.json      # Desktop / corporate templates
├── core/                   # Router, state, cognition, RAG, GPU, observability
├── brain/                  # Mini LLM, memory graph, pipelines
├── cursor_bridge/          # Local brain controller, prompts
├── voice/                  # STT, TTS, mic
├── ui/                     # Web dashboard (aiohttp)
├── docs/                   # Reports, deployment, benchmarks
├── requirements.txt
├── requirements-desktop.txt
└── ATOM_ARCHITECTURE_BLUEPRINT.md
```

---

## V7 observability (quick reference)

| Signal | Where |
|--------|--------|
| Health + metrics + warnings | `GET /v7/health` |
| Periodic snapshot | Log tag `v7_debug_snapshot` |
| Mode decisions | `v7_mode_selected`, `v7_mode_switch` |
| RAG / graph | `v7_rag_retrieval`, `v7_graph_*`, `v7_rag_fallback` |
| Prefetch | `v7_prefetch_*` |

---

## Security & privacy (short)

- Clipboard/context redaction: `context/privacy_filter.py`
- Sensitive actions can be tied to optional vision / owner recognition — see settings and deployment docs
- **Do not** weaken `SecurityPolicy` or authentication paths without a full review

---

## More documentation

| Document | Content |
|----------|---------|
| [`docs/ATOM_CODE_REVIEW_AND_DESKTOP_PLAN.md`](docs/ATOM_CODE_REVIEW_AND_DESKTOP_PLAN.md) | Review summary, V7 map, desktop migration, static validation |
| [`docs/README.md`](docs/README.md) | Index of benchmarks and reports |
| [`docs/ATOM_Deployment_Profiles.md`](docs/ATOM_Deployment_Profiles.md) | Corporate vs home hardware |
| [`ATOM_ARCHITECTURE_BLUEPRINT.md`](ATOM_ARCHITECTURE_BLUEPRINT.md) | Full system blueprint |

---

## License / ownership

ATOM is Satyam’s personal cognitive OS project; use and deployment are subject to your environment’s policies.
