# ATOM v13 — Your JARVIS-Level AI Assistant

Enterprise-grade voice assistant owned by **Satyam**. ATOM helps with everything on your company laptop: system control, apps, files, queries, and more. Optional camera-based owner recognition keeps it secure and personal.

## Features

- **Owner-first**: Configured for Satyam; ATOM addresses you by name and title (Boss).
- **Instant Brain**: Local Intent Engine handles most commands in &lt;5 ms (open/close apps, volume, lock, screenshot, timer, etc.).
- **Offline brain**: Local GGUF LLM (`brain.*` in settings) — no Gemini/Groq; JARVIS-style personality and system awareness.
- **Faster**: Cache, preload, layered routing (intent → cache/memory → local LLM).
- **Secured**: Privacy filter for clipboard/context; optional “only when I see you” for sensitive actions.
- **Vision (optional)**: Local-only camera face recognition to detect you (Satyam). No images leave the machine.

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Microphone
- **No cloud LLM keys required** — ATOM v15 is offline-first. Optional: `%USERPROFILE%\.atom\env` for future integrations (see `.env.example`).

### 2. Install

```bash
cd ATOM
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 3. Configure

Edit `config/settings.json`:

- **Owner**: `"owner": { "name": "Satyam", "title": "Boss" }`
- **Vision** (camera): Set `"vision": { "enabled": true }` when you want face detection. Run enrollment once (see below).

### 4. Run

```bash
python main.py
```

Dashboard opens at http://127.0.0.1:8765. ATOM uses **always-on listening** by default; use **Ctrl+Alt+A** to toggle idle / resume listening, or **UNSTICK** on the dashboard if the state machine hangs.

**If the browser tab does not open:** use Cursor **Simple Browser** or tasks — see **[docs/CURSOR_DASHBOARD.md](docs/CURSOR_DASHBOARD.md)**.

## Owner & Vision (Camera)

- **Owner** is set in `config/settings.json` under `owner.name` and `owner.title`. ATOM uses this in greetings and responses.
- **Face recognition** is optional and runs entirely on your machine:
  1. Install: `pip install opencv-python face_recognition numpy`
  2. Enroll once: `python scripts/enroll_owner_face.py` — capture your face with SPACE, quit with Q. This creates `config/owner_face.npy`.
  3. In `settings.json` set `"vision": { "enabled": true }`. Optionally set `"require_owner_for_sensitive": true` so lock/sleep/shutdown/close app etc. only run when the camera recognizes you.

The dashboard shows an **OWNER** panel (e.g. “Satyam” / “Recognized ✓” or “Unknown” / “Camera off”).

## Deployment profiles (company laptop vs future PC)

See **[docs/ATOM_Deployment_Profiles.md](docs/ATOM_Deployment_Profiles.md)** for:

- How to align ATOM for **corporate laptop testing** (security, CPU-only brain, performance).
- A **~₹1 lakh** India PC parts list oriented toward **local LLM + ATOM** when you build your own machine.

**Staged evolution (corporate → incredible):** **[docs/ATOM_Corporate_Evolution.md](docs/ATOM_Corporate_Evolution.md)** — trust, performance, context, skills, safe proactivity. Copy **`config/settings.corporate.example.json`** as a baseline for work machines.

**Rating & full review:** **[docs/ATOM_OS_Review.md](docs/ATOM_OS_Review.md)** — scores by dimension + **recent enhancements**. Production-style metrics: **[docs/ATOM_Production_Readiness_Scorecard.md](docs/ATOM_Production_Readiness_Scorecard.md)**.

**Verbal shortcuts:** edit **`config/skills.json`** — phrases expand before intent classification (e.g. “atom health check” → “self check”).

## Security

- **Secrets**: Offline build needs no API keys. If you add integrations later, keep secrets out of the repo (see `.env.example`).
- **Privacy**: Clipboard and context are redacted before prompts (`context/privacy_filter.py`).
- **Vision**: Camera frames are never stored or sent; only a local face encoding is used for recognition.
- **Sensitive actions**: With `require_owner_for_sensitive: true`, actions like lock screen, shutdown, restart, logoff, sleep, empty recycle bin, close app, move/copy path require owner detection.

## Configuration (main keys)

| Key | Purpose |
|-----|--------|
| `owner.name` | Your name (e.g. Satyam). |
| `owner.title` | How ATOM addresses you (e.g. Boss). |
| `vision.enabled` | Turn on camera-based owner detection. |
| `vision.require_owner_for_sensitive` | Require face recognition for sensitive commands. |
| `brain.enabled` / `brain.model_path` | Local GGUF model (llama.cpp). |
| `tts.engine` | `sapi` (offline, default) or `edge` (neural, needs network). |
| `features.web_research` / `online_weather` | `false` by default for offline use. |

## Project Structure

```
ATOM/
├── main.py                    # Entry point, wires bus & modules
├── config/settings.json       # Owner, vision, STT/TTS/AI, etc.
├── core/                      # State, router, intent engine, cache, memory
├── voice/                     # STT, TTS, mic
├── brain/                     # Mini LLM prompt engine
├── cursor_bridge/             # Prompt builder + local brain controller
├── context/                   # Context engine, privacy filter
├── vision/                    # Face recognizer (local-only camera)
├── ui/                        # Web dashboard (JARVIS-style)
├── scripts/enroll_owner_face.py  # One-time face enrollment
└── logs/
```

## Usage

| Action | Example |
|--------|--------|
| Activate | “Hey Atom” or Ctrl+Alt+A |
| Greeting | “Hi”, “Good morning” |
| System | “Check CPU”, “Battery”, “Lock screen”, “Screenshot” |
| Apps | “Open Notepad”, “Close Chrome” |
| Media | “Set volume to 50”, “Play X on YouTube”, “Mute” |
| Power | “Shutdown”, “Restart”, “Sleep” (with optional face check) |
| Exit ATOM | “Bye”, “Shutdown”, “Go silent” |

ATOM is your dedicated assistant: smarter, faster, and secured for your company laptop.
