# ATOM Module 12: Evolution Roadmap

> Read this when planning new features or upgrades.

## Phase 1: Perception Evolution

| Upgrade | Current | Target | Effort |
|---------|---------|--------|--------|
| Wake Word Detection | Always listening | "Hey ATOM" (Porcupine/OpenWakeWord) | Medium |
| Emotion Detection | None | Voice tone analysis (pitch, speed) | Medium |
| Multi-Language STT | English only | Hindi + English code-switching | Medium |
| Ambient Sound Awareness | Noise filter only | Classify music/talking/silence | Hard |
| Screen Understanding | Screenshot only | Local vision model for OCR | Hard |

## Phase 2: Intelligence Evolution

| Upgrade | Current | Target | Effort |
|---------|---------|--------|--------|
| RAG Pipeline | Keyword memory | Vector embeddings + FAISS/ChromaDB | Medium |
| Multi-Turn Reasoning | 5-turn window | Chain-of-thought with scratchpad | Medium |
| Tool Use (Function Calling) | Dispatch table | LLM chooses and chains tools | Hard |
| Code Execution Sandbox | None | Safe Python/JS eval | Medium |
| Document Ingestion | None | PDF/DOCX → knowledge base | Medium |

## Phase 3: Autonomy Evolution

| Upgrade | Current | Target | Effort |
|---------|---------|--------|--------|
| Workflow Automation | Single actions | Multi-step recorded workflows | Hard |
| Calendar Integration | None | Outlook/Google Calendar read/write | Medium |
| Email Triage | None | Summarize + prioritize inbox | Medium |
| Proactive Research | Disabled | Background web research | Medium |
| Cross-Device Sync | Single machine | State sync across devices | Hard |

## Phase 4: Expression Evolution

| Upgrade | Current | Target | Effort |
|---------|---------|--------|--------|
| Voice Cloning | Fixed TTS voice | Custom ATOM voice (XTTS/Bark) | Hard |
| 3D Avatar | Three.js orb | Animated face/character | Hard |
| Spatial Audio | Mono output | Directional audio | Hard |
| Multi-Modal Response | Text + voice | Voice + images + code blocks | Medium |

## Phase 5: Meta-Cognitive Evolution

| Upgrade | Current | Target | Effort |
|---------|---------|--------|--------|
| Dream Mode | None | Offline consolidation from day's interactions | Hard |
| Epistemic Humility | None | Confidence scoring on every answer | Medium |
| Curiosity Engine | None | ATOM asks questions to learn about you | Medium |
| Personality Evolution | 4 fixed modes | Personality drift from long-term behavior | Hard |
| Attention Economy | Process all | Priority-weighted attention | Hard |

## Phase 6: Distributed ATOM

| Upgrade | Current | Target | Effort |
|---------|---------|--------|--------|
| ATOM Mesh | Single instance | Multiple ATOMs coordinating | Very Hard |
| Plugin Architecture | Hardcoded | Dynamic plugin loading | Hard |
| Agent Protocol | None | MCP/A2A server + client | Hard |
| Hardware Brain | CPU-only | Dedicated NPU/GPU card | Hardware |

## Build Priority

1. **Tier 1 (Foundation):** Remove kill switch, run on target, test all commands
2. **Tier 2 (Quick Wins):** Add intent patterns, quick replies, tune STT
3. **Tier 3 (Core):** RAG, wake word, calendar, code sandbox
4. **Tier 4 (Advanced):** Plugin arch, multi-language, tool use, voice cloning
