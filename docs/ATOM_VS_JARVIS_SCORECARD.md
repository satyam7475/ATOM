# ATOM vs "Jarvis" — module scorecard

**Purpose:** Honest, versionable comparison between ATOM (real software) and
the fictional JARVIS / FRIDAY reference from film. Update this file when major
milestones land (voice reliability, cloud, tools). Scores are **1–5** with
short criteria — not scientific benchmarks.

**Legend:** **5** = best-in-class for a consumer/local AI OS;
**3** = usable; **1** = missing or unreliable.

## Snapshot — Apr 26 2026 (post P1–P4 + boot-stall fix)

**Aggregate:** ATOM **8.3 / 10** vs an aspirational Jarvis/Friday **10 / 10**.
First post-P1–P4 measurement. Local-only mode (cloud disabled by default —
`cloud.enabled = false`, `cloud_brain_router.enabled = false`), so the cloud
column reflects the gate, not capability.

| Module | Criteria (what "5" means) | ATOM (Apr 26 2026) | Jarvis (fiction) | Notes |
|--------|----------------------------|---------------------|------------------|-------|
| **Voice I/O** | Reliable STT/TTS, barge-in, permissions | **4** | **5** | whisper.cpp Metal STT (large-v3-turbo) + WhisperConfirmer second-pass on suspect finals; macOS NSSpeechSynthesizer TTS with stream chunker; barge-in via `voice_interrupt` + post-TTS cooldown. P3.3 WhisperKit ANE swap is the next +1. |
| **State / lifecycle** | Clear IDLE / LISTENING / THINKING / SPEAKING | **5** | **5** | Explicit FSM (`core/state_manager.py`) + audited transitions; voice loop now boots cleanly to "Cognitive loop ready" → "Realtime room ready" → "iPhone bridge listening" without the post-AdaptiveEngine stall (root cause: missing `WhisperSTT.on_state_changed`, fixed Apr 26 2026). |
| **Reasoning (local)** | MLX local model, routing, latency | **4–5** | **5** | Qwen3-8B-4bit primary + Qwen3-4B-Instruct-4bit fast/draft on MLX with KV-quant warm-up, `mx.compile`'d sampler, persona prompt cache pinned at boot. Speculative decoding (4B drafts 8B) wired and **enabled** Apr 26 2026 — expect 1.5–2× tokens/s on warm runs. |
| **Cloud / Gemini** | Optional escalation, quotas, safety | **3** (gated, off by default) | **5** | `cloud.enabled = false` and `cloud_brain_router.enabled = false` ship by default — ATOM is **100 % offline-first**. Flip both true + set `GEMINI_API_KEY` (and `pip install google-generativeai`) to escalate hard queries. Daily quota + cooldown wiring is real. |
| **Memory / RAG** | Graph + vectors, recall quality | **4** | **5** | Memory engine + vector store (Chroma) + graph-first retrieval with project boost; `mlx-embeddings` backend (Apr 26 2026) on the Apple Neural Engine, ~3× faster than torch-MPS for the 384-dim MiniLM-L6 model. Owner-priority + recency half-life smart-scoring is live. |
| **Tools / OS control** | Security-gated actions, confirmations | **4** | **5** | 40+ tools, ReAct loop, code sandbox, security gateway with system-path block, `pending_tool_confirmation` flow, autonomy thresholds (auto-execute 0.95 / suggest 0.72), runtime watchdog. Real safety constraints vs plot armour. |
| **UI / dashboard** | Health at a glance, WebSocket state | **4** | **5** | AtomRuntimeStateBridge → realtime room (`http://127.0.0.1:8770/play/`) with voice health strip + orb. Unified `/badge` endpoint + standalone menubar polling daemon (`tools/atom_status_badge.py`) — see *How to enable the menubar status badge* below. |
| **Proactivity** | Reminders, hints, habits | **4** | **5** | Phase G cognitive loop is **all on**: ReflectiveLoop, PresenceSampler, SceneContext, MoodInference, JarvisSuggester, AwarenessLoop. Quiet-hours + per-category cooldowns + `suppress_moods` keep nudges respectful. |

## Local model inventory (no cloud calls)

| Role | Model | Quant | On-disk |
|------|-------|-------|---------|
| LLM primary | `qwen3-8b-4bit` | 4-bit | yes (`models/qwen3-8b-4bit`) |
| LLM fast / spec-decoding draft | `qwen3-4b-instruct-4bit` | 4-bit | yes (`models/qwen3-4b-instruct-4bit`) |
| STT primary | `large-v3-turbo` | q5_0 | yes (`models/whisper.cpp/ggml-large-v3-turbo-q5_0.bin`) |
| STT confirmer (second-pass) | `tiny` | q5_1 | yes (`models/whisper.cpp/ggml-tiny-q5_1.bin`) |
| VLM (vision) | `SmolVLM-Instruct-4bit` | 4-bit | yes (`models/smolvlm-instruct-4bit`) — lazy-loaded on first describe call |
| Embeddings | `mlx-community/all-MiniLM-L6-v2-4bit` | 4-bit | HF cache (auto-downloaded by `mlx-embeddings`) |

That's **3 LLM-shaped models on disk** (counting the 4B once even though it
serves both *fast role* and *speculative draft*) plus 4 supporting models for
STT / VLM / embeddings — **6 distinct artefacts, 7 model-roles**.

## Investment order (recommended)

1. **WhisperKit ANE STT swap** (Plan §P3.3) — single biggest perceived-Jarvis
   win; moves STT from CPU/Metal to the Neural Engine.
2. **Latency** — speculative decoding is on; next is `mx.compile` on the
   prefill path and tighter prompt-token budget.
3. **Tool success rate** — router + executor + confirmations; track
   `tool_success_rate` metric over a week.
4. **Cloud** (when you want it back on) — set `GEMINI_API_KEY` + flip
   `cloud.enabled` + `cloud_brain_router.enabled` back to `true` after `pip
   install google-generativeai`. Daily quota + smart-route keywords are
   already wired.
5. **Personality / proactivity polish** — after the above are stable.

## How to enable the menubar status badge

The badge is *opt-in*: ATOM never auto-spawns it (so no Dock icon and no
extra event-loop on cold boot). To run it:

```bash
# one-shot (default port resolution: logs/atom_bridge.port)
python tools/atom_status_badge.py

# continuous menubar (needs `pip install rumps` once)
python tools/atom_status_badge.py --menubar

# JSON for scripts / dashboards
python tools/atom_status_badge.py --json
```

The badge polls the iPhone bridge's `/badge` endpoint (HTTP, localhost-only
by default) so it works against a running ATOM with zero extra wiring.

## Version

- **2026-04 (initial):** Scorecard + voice pipeline doc reference.
- **2026-04-26 (post P1–P4 + boot-stall fix):** Aggregate **8.3 / 10**.
  Speculative decoding on, MLX embeddings on, cloud disabled-by-default,
  boot reaches "Cognitive loop ready" in ≈ 15 s after the
  `WhisperSTT.on_state_changed` parity fix landed (boot was silently exiting
  mid-init via an unhandled `AttributeError` and `asyncio.run()` was
  deadlocking on an un-cancellable iPhone-bridge BG task during teardown).
