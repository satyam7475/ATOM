# ATOM — Corporate / laptop evolution plan

This is the **staged roadmap** for making ATOM stronger on a **managed company laptop** first, then scaling up on a future workstation. Each stage builds on the previous one; skip ahead only when the prior stage is stable in daily use.

**Owner:** Satyam  
**Related:** [ATOM_Deployment_Profiles.md](ATOM_Deployment_Profiles.md), `config/settings.corporate.example.json`, `core/deployment_profile.py`

---

## What shipped in this wave (baseline)

| Piece | Purpose |
|-------|---------|
| `deployment.profile` | `corporate_laptop` \| `personal` \| `workstation` — drives startup hints and optional UI badge. |
| `deployment.dashboard_badge` | Shows **CORPORATE** / **WORKSTATION** / **PERSONAL** on the web dashboard when enabled. |
| Startup audit | Logs **warnings** if config conflicts with typical corporate norms (cloud STT/TTS, web features, camera, non-strict security). |
| Tighter laptop defaults | Smaller clipboard snippet, `cognitive.auto_mode_switching: false`, extended `security.require_confirmation_for`. Mic Bluetooth is **user preference** (`mic.prefer_bluetooth`). |
| Session + skills | `SessionContext` prior-turn line in local LLM prompts; `config/skills.json` phrase expansions via `SkillsRegistry`. |
| Clearer confirm prompts | Voice confirmations for URLs, lock, screenshot, kill process, file ops. |

---

## Stage 1 — Trust & predictability *(you are here)*

**Goal:** ATOM never surprises you in front of colleagues or IT.

- Keep `security.mode: strict`, `features.web_research` / `online_weather` off unless approved.
- Use **`hybrid`** + local brain for real testing; switch to **`command_only`** in sensitive meetings.
- Rely on **`require_confirmation_for`** for anything that touches the network, screen, files, or session.
- Watch **`logs/audit.log`** after each test day.

**Exit criteria:** One week without accidental actions; audit log is understandable.

---

## Stage 2 — Performance discipline on shared CPU

**Goal:** ATOM stays a “background copilot,” not a resource hog.

- Keep `performance.mode` at **`lite`** or **`ultra_lite`** on the laptop; use **`auto`** only after benchmarking.
- Keep `brain.n_gpu_layers: 0` until you have an approved GPU stack.
- Tune `brain.n_threads` to **physical cores − 1** on the laptop.
- Use **`atom`** brain profile in meetings if latency matters.

**Exit criteria:** Teams/Edge stay smooth while ATOM listens; no thermal throttling during long sessions.

---

## Stage 3 — Unified session context *(partially shipped)*

**Goal:** One coherent “what we’re doing now” across voice, dashboard, and memory.

- **Shipped:** `SessionContext` + `session_summary` in the **local LLM** prompt (`settings.json` → `session`).
- **Next:** Richer memory summaries after multi-step tasks; optional dashboard line for last intent.

**Exit criteria:** Follow-up commands work without repeating the full context.

---

## Stage 4 — Skill library (procedures) *(foundation shipped)*

**Goal:** Reusable workflows (“start my standup stack,” “run self-check”) as **named skills**, not ad-hoc chat.

- **Shipped:** `config/skills.json` + `SkillsRegistry` — triggers expand to utterances the intent engine already knows (`settings.json` → `skills`).
- **Next:** Multi-step skill sequences with explicit allowlists (safer than free-form expansion).

**Exit criteria:** Three daily skills you actually use; each has a one-line description and a test or manual checklist.

---

## Stage 5 — Safe proactivity

**Goal:** FRIDAY-style *hints*, not autonomous chaos.

- Enable only **local** signals (battery, idle, stuck-state recovery you already have).
- Add new proactive features behind **`features.*`** toggles and **`SecurityPolicy`**.

**Exit criteria:** Every proactive message is explainable and dismissible; nothing runs without policy.

---

## Config quick reference

```json
"deployment": {
  "profile": "corporate_laptop",
  "dashboard_badge": true
}
```

- **`workstation`:** set when you move to your own PC; see [ATOM_Deployment_Profiles.md](ATOM_Deployment_Profiles.md) §2 for GPU/RAM knobs.
- **`dashboard_badge: false`:** discreet mode (no pill in the top bar).

---

## Compliance note

This document is **technical guidance**, not legal advice. Align microphone, camera, local AI, and code storage with **your employer’s** policies and MDM/DLP rules.
