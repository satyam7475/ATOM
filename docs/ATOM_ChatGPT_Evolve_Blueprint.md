# ATOM: Personal Cognitive AI Operating System
## Architecture & Design Blueprint for Evolve Process

**Target Environment:** Ryzen 7 CPU, RTX 40/50 Series GPU (12-16GB VRAM), 16GB RAM, 1TB SSD, Windows 11.
**Core Philosophy:** Fully offline, zero-latency, context-aware, buddy-level interaction (JARVIS standard).

---

### 1. System Overview

ATOM is not a simple voice assistant; it is a **Personal Cognitive AI Operating System**. It runs entirely locally, integrating deep system awareness, emotional intelligence, and real-time environmental context into a single, unified prompt for an agentic LLM.

The system is built on an **Async Event Bus** architecture, allowing decoupled modules (Perception, Cognition, Action, Output) to communicate instantly without blocking the main event loop.

---

### 2. Core Architecture Layers

#### A. The Perception Layer (Input & Filtering)
*   **AudioPreprocessor:** Processes raw PCM audio using `numpy`. Applies DC offset removal, pre-emphasis, and a spectral noise gate that actively learns ambient room noise (fans, AC).
*   **MicManager:** Profiles all connected microphones, scoring them on latency, sample rate, and failure history to auto-select the optimal input device.
*   **STT Engine (`faster-whisper large-v3-turbo`):** Runs on GPU. Achieves ~1.9% WER in English and ~7.2% in Hindi. Features auto-language detection and whisper hallucination filtering.

#### B. The Intelligence & Context Layer (The "Thalamus")
Before the LLM sees a prompt, ATOM fuses multiple intelligence streams into a single `FusedContext` object:
*   **SystemIndexer:** Background thread that maps all installed apps, running processes (PIDs), and recently modified files (Desktop, Documents, Downloads). Provides instant, zero-latency system knowledge.
*   **MediaWatcher:** Uses `winsdk` to monitor Windows Media Transport Controls. Knows exactly what song/video is playing and on what app (Spotify, Chrome).
*   **RealWorldIntelligence:** Offline-first module tracking time, date, seasonal context, and approximate sunrise/sunset.
*   **OwnerUnderstanding:** Tracks the user's emotional trajectory (stressed, happy, focused) and adjusts ATOM's verbosity and tone accordingly.

#### C. The Cognition Layer (The Brain)
*   **Model:** `Qwen3.5-4B-Q5_K_M` (GGUF via `llama-cpp-python`).
*   **Why this model?** It achieves 97.5% tool-calling accuracy, processes at ~48 tokens/sec, and uses only ~3.5GB VRAM. It features native dual thinking modes (`/think` for complex tasks, `/no_think` for instant chat).
*   **Structured Prompt Builder:** A 9-layer prompt architecture that dynamically injects the `FusedContext` (System, Media, Emotion, History) and a formal `ToolRegistry` schema.

#### D. The Action Layer (Execution & Security)
*   **ToolRegistry:** 40+ formal tools (e.g., `open_app`, `kill_process`, `search_web`) with JSON schemas.
*   **ActionExecutor:** The bridge between the LLM's JSON output and system execution.
*   **SecurityFortress:** A strict policy gate.
*   **Smart Confirmations:** Replaces robotic "Are you sure?" prompts with buddy-like, context-aware advice (e.g., *"Killing Chrome might cause instability, Boss. Are you absolutely sure?"*).

#### E. The Output Layer (Speech)
*   **TTS Engine (`Kokoro TTS`):** A neural, fully offline TTS model.
*   **Performance:** Generates audio chunks in ~80ms (100x faster than real-time) and streams them directly to `sounddevice` for zero-latency playback.

---

### 3. VRAM Budget & Performance Optimization

To ensure ATOM never crashes or stutters the host OS, VRAM is strictly budgeted:
1.  **STT (`large-v3-turbo` fp16):** ~2.5 GB
2.  **Brain (`Qwen3.5-4B-Q5_K_M`):** ~3.5 GB
3.  **TTS (`Kokoro`):** ~1.5 GB
*   **Total VRAM Footprint:** ~7.5 GB.
*   *(Leaves 4.5GB - 8.5GB free for the user's games/IDEs on a 12GB/16GB GPU).*

**Latency Target:** Hear → Transcribe → Think → Speak = **< 0.8 seconds.**

---

### 4. The "Evolve" Process Request for ChatGPT

**Prompt for ChatGPT:**

> "I am building ATOM, a Personal Cognitive AI Operating System designed to run locally on a Ryzen 7 / RTX 40-50 Series desktop. My goal is to achieve a JARVIS-level, zero-latency, buddy-like interaction.
> 
> Please review the attached Architecture & Design Blueprint, along with the High-Level Design (HLD) and Low-Level Design (LLD) concepts provided.
> 
> I want you to execute an 'Evolve Process' on this design. Analyze it deeply as a Senior Principal AI Architect and provide:
> 1. **Critical Vulnerabilities:** Are there any bottlenecks, race conditions, or memory leaks likely to occur in this async/multi-model pipeline (specifically looking at the EventBus and ThreadPoolExecutor interactions)?
> 2. **Context Fusion Improvements:** How can I make the `SystemIndexer` and `MediaWatcher` even more proactive without polling the OS too aggressively?
> 3. **LLM Optimization:** Is there a better way to handle the prompt context window (32K) so the model doesn't suffer from 'lost in the middle' syndrome during long sessions?
> 4. **Buddy Persona:** How can I improve the `OwnerUnderstanding` module to make ATOM's emotional intelligence feel more genuine and less programmatic?
> 5. **Next-Gen Features:** What is one cutting-edge local AI feature (feasible in 2026) that is missing from this architecture?

*(Note to User: When pasting this to ChatGPT, you can optionally append the contents of `ATOM_HLD.md` and `ATOM_LLD.md` to give it the absolute maximum technical depth for the review).*