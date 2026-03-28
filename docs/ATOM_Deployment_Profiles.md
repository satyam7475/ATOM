# ATOM OS — Deployment profiles (company laptop vs future PC)

Use this doc to keep **ATOM aligned** with where you run it today (corporate testing) and where you want to go (dedicated home/workstation PC).

---

## 1. Company laptop (testing now)

**Goals:** Respect employer policy, reduce data exfil risk, stay light on CPU/RAM, avoid needing admin/GPU drivers you can’t install.

### Policy & hygiene (non-config)

- Follow your org’s rules on **local AI**, **microphone/camera**, and **code on the laptop** (MDM, DLP, acceptable use).
- Keep **secrets out of `config/settings.json`**. ATOM is fully offline and requires no cloud API keys.
- Store the ATOM repo in an **approved location** (e.g. user profile, not synced public folders if policy forbids).
- Treat the web dashboard as **localhost-only** (already bound to `127.0.0.1`) — still only as private as the Windows user session.

### `deployment` block (v15+)

| Key | Suggestion |
|-----|------------|
| `deployment.profile` | `"corporate_laptop"` while on a work machine — enables startup **alignment warnings** in logs (cloud STT/TTS, camera, web features, etc.). |
| `deployment.dashboard_badge` | `true` to show **CORPORATE** on the localhost dashboard; `false` for a cleaner top bar in demos. |

See **[ATOM_Corporate_Evolution.md](ATOM_Corporate_Evolution.md)** for the full staged roadmap.

### Recommended `settings.json` alignment (corporate testing)

| Area | Suggestion | Why |
|------|------------|-----|
| Cloud AI | Not used | ATOM is fully offline — no cloud API keys needed. |
| `brain.enabled` | `true` with **small** GGUF (e.g. 3B Q4) | Local-only reasoning; fits CPU testing. |
| `brain.n_gpu_layers` | `0` | No CUDA dependency on locked-down laptops. |
| `brain.n_threads` | Match **physical** cores − 1 | Leaves headroom for OS + Teams/Edge. |
| `vision.enabled` | `false` unless HR/IT approves camera | Avoids workplace camera concerns. |
| `performance.mode` | `lite` or `ultra_lite` | Less background load during work hours. |
| `autonomy` | **`enabled: true`** + **`auto_execute_threshold: 0.95`** + slightly higher **`suggest_threshold`** | See **§1b** — best balance for *testing* without surprise actions. |
| `control.assistant_mode` | **`hybrid`** for full tests; **`command_only`** when you need zero “open chat” LLM | See **§1a**. |
| `security.mode` | `strict` | Blocks destructive power actions by default. |
| `features.desktop_control` / `file_ops` | **`true`** if you want to **exercise the full OS layer** (recommended for real ATOM testing). Turn off only for minimal surface experiments. |

### §1a — `control.assistant_mode` (company laptop)

| Mode | What you get |
|------|----------------|
| **`command_only`** | Voice/UI still run **intents** (time, cpu, open app, etc.). **No** “open conversation” path to the LLM for vague questions — least risk of accidental dictation or sensitive context going to a model. |
| **`hybrid`** | Commands stay fast; **unmatched / chatty** queries go to the **local brain**. **Use this to test ATOM properly** on a laptop — you exercise routing + local LLM. |
| **`conversational`** | More chat-oriented behavior (where implemented). |

**For you (proper testing on company laptop):** keep **`hybrid`** to test full behavior using the **local** brain.

### §1b — `autonomy`: off vs high threshold (company laptop)

- **`enabled: false`** — Autonomy **never** runs: no background habit loop, no auto/suggest from that engine. **Safest and quietest**, but you **don’t test** autonomy at all.
- **`enabled: true` + `auto_execute_threshold: 0.95`** (and e.g. `suggest_threshold: 0.72`) — Engine **still runs** (logs, learning, suggestions in the high band), but **almost nothing** auto-executes unless confidence is extremely high. Destructive actions are **never** auto-run per `autonomy_engine` rules; everything still passes **`SecurityPolicy`**.

**Better for you while testing ATOM properly:** **`enabled: true`** with **`auto_execute_threshold: 0.95`** — you keep the **real subsystem active** for a realistic test, with **minimal** risk of surprise automation during meetings or IT observation. Use **`false`** only if your org asks for **zero** background decision-making.

### Full ATOM testing on company laptop (default recommendation)

- **`features.desktop_control`**: `true`  
- **`features.file_ops`**: `true`  
- **`features.llm`**: `true` (local brain path)  
- **`control.assistant_mode`**: `hybrid`  
- **`autonomy`**: `enabled: true`, `auto_execute_threshold: 0.95`, `suggest_threshold: 0.72`  
- **`security.mode`**: `strict`  
- **Secrets**: ATOM is fully offline — no cloud API keys needed.

### Optional: two copies of config

- Keep **`config/settings.json`** as your daily driver.
- Maintain **`config/settings.home-workstation.json`** (or similar) as a **reference** for the future PC so you don’t forget which knobs change (GPU layers, model path, performance mode). Swap or merge when you migrate.

---

## 2. Future dedicated PC (“full” ATOM workstation)

**Goals:** Faster local brain, optional larger models, GPU offload, more RAM for context and multitasking.

### Config direction (when hardware is ready)

| Area | Direction |
|------|-----------|
| `brain.model_path` | Move up to **7B–8B** (or larger) quantized GGUF as VRAM/RAM allows. |
| `brain.n_gpu_layers` | Increase until latency stabilizes (e.g. start with **20–35**, tune per model). |
| `brain.n_ctx` | Can increase if you have **RAM headroom** (32GB+ recommended for comfort). |
| `performance.mode` | `full` or `auto` when thermals allow. |
| `vision` | Enable if **only you** use the machine and you want owner gating. |
| `assistant_brain.profiles` | Widen `brain` profile `max_tokens` / `n_ctx` for deeper answers. |

Install **NVIDIA drivers** + a CUDA-capable **`llama-cpp-python`** build (or your project’s documented install) so `n_gpu_layers` actually uses the GPU.

---

## 3. Gaming + ATOM @ ~₹1,00,000 (India)

**Soft target:** **₹95k–₹1.05L** after GST; prices move weekly — check **PrimeABGB, Vedant, MDComputers, The IT Gear**, local shops, and festival sales.

### What this budget does well

| Use | Expectation |
|-----|-------------|
| **Gaming** | **1080p** high/ultra in most titles at strong FPS (**RTX 4060** or **4060 Ti 8GB** class). |
| **ATOM local brain** | **3B–7B** GGUF with **many layers on GPU** (8GB VRAM). **32GB RAM** keeps Windows + game + ATOM + browser comfortable. |
| **Stretch** | **16GB VRAM** = headroom for **7B–8B** and nicer quants — usually **₹10k–₹20k** above a strict ₹1L list. |

**Trade-off:** At exactly ₹1L you choose between **4060 Ti 8GB** (better gaming + LLM speed) vs **4060 8GB** (saves cash for PSU/SSD/case). For **gaming + ATOM on Windows**, prefer **NVIDIA** (CUDA + `llama-cpp-python`).

### Recommended parts list (balanced gaming + ATOM)

| Part | Pick | Notes |
|------|------|--------|
| **CPU** | **Ryzen 5 7500F** or **7600** | 7600 adds **iGPU** (useful if GPU RMA). Both great for 1080p. |
| **Motherboard** | **B650M** DDR5 (e.g. MSI **Pro B650M-B** / **B650M-P**) | Skip paid RGB; check RAM QVL if you want max EXPO stability. |
| **RAM** | **32GB (2×16) DDR5-5600** | **Do not** build with 16GB for this combo. |
| **GPU** | **RTX 4060 Ti 8GB** if total ~₹1L; else **RTX 4060 8GB** | Ti = faster gaming + faster inference; 4060 = easier to stay under cap. |
| **SSD** | **1TB NVMe Gen4** | Games + GGUF models; add 2TB later if needed. |
| **PSU** | **650W 80+ Gold** | Worth it for GPU spikes + long LLM loads. |
| **Case** | Mesh front + **2–3 fans** | LLM keeps GPU busy; airflow = quieter, stable clocks. |
| **Cooler** | **Tower air** (e.g. **AK400** class) or good stock | 7500F/7600 are fine with airflow. |

### Hard cap under ₹1.0L

- **4060 8GB** + **32GB** + **650W Gold** + **1TB** — still excellent **1080p gaming**; ATOM happy with **3B–7B** on GPU.

### Best extra spend for ATOM

- **+₹10k–₹20k → 16GB VRAM** (e.g. **4060 Ti 16GB** or current-gen equivalent on sale) = biggest upgrade for **local LLM**, still solid gaming.

### AMD Radeon?

- Great for games only; **ATOM + llama.cpp on Windows** is **much easier on GeForce** unless you run **Linux + ROCm**.

### After build — `settings.json`

- Raise `brain.n_gpu_layers` in steps until stable (watch VRAM/temps).
- On **8GB** VRAM, stay around **7B Q4** or smaller; **16GB** unlocks bigger models / context.

---

## 4. Quick migration checklist (laptop → new PC)

- [ ] Copy repo + `models/` (or re-download GGUFs).
- [ ] Reinstall Python venv + `requirements.txt` (+ GPU build of `llama-cpp-python` if used).
- [ ] Update `brain.model_path`, `n_gpu_layers`, `n_threads`.
- [ ] Run `python -c "from core.config_schema import validate_config; ..."` on your merged `settings.json`.
- [ ] Re-enroll vision (`owner_face.npy`) only if you enable camera on the new machine.

---

*Owner: Satyam — align profiles with your employer’s policy first; this doc is technical guidance, not legal/compliance advice.*
