# ATOM vs “Jarvis” — module scorecard

**Purpose:** Honest, versionable comparison between ATOM (real software) and the fictional JARVIS reference from film. Update this file when major milestones land (voice reliability, cloud, tools). Scores are **1–5** with short criteria — not scientific benchmarks.

**Legend:** **5** = best-in-class for a consumer/local AI OS; **3** = usable; **1** = missing or unreliable.

| Module | Criteria (what “5” means) | ATOM (typ.) | Jarvis (fiction) | Notes |
|--------|-----------------------------|-------------|------------------|--------|
| **Voice I/O** | Reliable STT/TTS, barge-in, permissions | **3–4** | **5** | Native STT needs ATOM.app + TCC; venv → Whisper fallback |
| **State / lifecycle** | Clear IDLE/LISTENING/THINKING/SPEAKING | **4–5** | **5** | Explicit FSM + [`14_VOICE_PIPELINE.md`](architecture/14_VOICE_PIPELINE.md) |
| **Reasoning (local)** | MLX local model, routing, latency | **4** | **5** | On-device bounded by silicon and prompt size |
| **Cloud / Gemini** | Optional escalation, quotas, safety | **3–4** when enabled | **5** | `cloud.enabled` is the gate; rate limits real |
| **Memory / RAG** | Graph + vectors, recall quality | **3–4** | **5** | Depends on ingestion and embedding stack |
| **Tools / OS control** | Security-gated actions, confirmations | **3–4** | **5** | Real safety constraints vs plot armor |
| **UI / dashboard** | Health at a glance, WebSocket state | **3–4** | **5** | Voice health strip + orb; room for polish |
| **Proactivity** | Reminders, hints, habits | **2–3** | **5** | Emerging; lower priority until voice stable |

## Investment order (recommended)

1. **Voice reliability** — highest perceived “Jarvis” feel.  
2. **Latency** — MLX + prompt token budget.  
3. **Tool success rate** — router + executor + confirmations.  
4. **Cloud** — Gemini for hard queries when `cloud.enabled` is on.  
5. **Personality / proactivity** — after the above are stable.

## Version

- **2026-04:** Initial scorecard + voice pipeline doc reference.
