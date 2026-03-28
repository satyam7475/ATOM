# ATOM: High-Level Design (HLD)

## 1. Introduction
ATOM is a Personal Cognitive AI Operating System designed to run entirely locally on consumer hardware (Ryzen 7, RTX 40/50 series). It acts as a JARVIS-like companion, providing zero-latency voice interaction, deep system awareness, and autonomous execution capabilities.

## 2. System Architecture Overview
ATOM follows an event-driven, microservices-inspired architecture running within a single Python process. Components communicate asynchronously via an `AsyncEventBus`, ensuring that heavy tasks (like LLM inference or system scanning) do not block the main interaction loop.

### 2.1 Core Architectural Layers

1.  **Perception Layer (Input)**
    *   **Audio Capture:** `pyaudio` and `speech_recognition` handle raw mic input.
    *   **AudioPreprocessor:** Cleans audio (noise gating, normalization) using `numpy` before STT.
    *   **STT Engine:** `faster-whisper` (large-v3-turbo) transcribes audio to text, handling bilingual (EN/HI) input.

2.  **Cognition & Context Layer (The Brain)**
    *   **ContextFusionEngine:** Aggregates data from `SystemIndexer`, `MediaWatcher`, `RealWorldIntel`, `OwnerUnderstanding`, and `L1Cache` into a single `FusedContext`.
    *   **Memory Architecture:**
        *   **L1 Cache:** Zero-latency, in-memory LRU/LFU cache for instant recall of highly relevant/recent facts.
        *   **SecondBrain:** Vector-embedded long-term semantic memory (disk-backed).
    *   **StructuredPromptBuilder:** Constructs a 9-layer prompt injecting the fused context, conversation history, and available tools.
    *   **LocalBrainController:** Manages the LLM (`Qwen3.5-4B-Q5_K_M` via `llama.cpp`). It handles the ReAct (Reason+Act) loop, parsing tool calls and generating responses.

3.  **Action Layer (Execution)**
    *   **ToolRegistry:** Defines available capabilities (e.g., `open_app`, `search_web`) using JSON schemas.
    *   **SecurityFortress:** Validates all tool calls against a strict policy.
    *   **ActionExecutor:** Executes the validated tool calls, handling smart confirmations for dangerous actions.

4.  **Output Layer (Speech & UI)**
    *   **TTS Engine:** `Kokoro TTS` generates neural voice audio in streaming chunks for ultra-low latency.
    *   **WebDashboard:** A local WebSocket-based UI for visual feedback and debugging.

## 3. Data Flow Architecture

### 3.1 The Main Interaction Loop (Hear -> Think -> Act -> Speak)
1.  **User Speaks:** Mic captures audio.
2.  **Preprocess & STT:** Audio is cleaned and transcribed. `speech_final` event is emitted.
3.  **Context Gathering:** `ContextFusionEngine` pulls the latest system state, media state, and emotional trajectory.
4.  **Prompt Generation:** `StructuredPromptBuilder` creates the prompt.
5.  **LLM Inference:** `LocalBrainController` streams the prompt to the LLM.
6.  **Tool Parsing:** If the LLM outputs a tool call (e.g., `{"name": "play_music"}`), the `ToolParser` intercepts it.
7.  **Execution & Feedback:** `ActionExecutor` runs the tool. The result is fed back into the LLM context.
8.  **Speech Synthesis:** The final text response is sent to `Kokoro TTS`, which streams audio to the speakers.

### 3.2 Background Indexing Flow
1.  **SystemIndexer:** Runs every 5 minutes. Scans registry for apps, lists running PIDs, and checks recent files. Updates internal dictionaries.
2.  **MediaWatcher:** Polls Windows Media Controls every 2 seconds. Updates `current_media` state.
3.  **RealWorldIntel:** Polls weather/news APIs every 30-60 minutes. Updates cache.

## 4. Hardware & Resource Budgeting
*   **Target Hardware:** Ryzen 7, 16GB RAM, RTX 4060/4070 (12-16GB VRAM).
*   **VRAM Allocation Strategy:**
    *   STT (Whisper): ~2.5GB
    *   LLM (Qwen3.5-4B): ~3.5GB
    *   TTS (Kokoro): ~1.5GB
    *   **Total Reserved:** ~7.5GB (Leaves ample headroom for OS and user applications).

## 5. Security Architecture
*   **Air-Gapped Core:** Core LLM and STT/TTS run locally. No audio or text leaves the machine unless explicitly requested (e.g., `search_web` tool).
*   **Tiered Tool Safety:** Tools are categorized as `safe`, `moderate`, `dangerous`, or `blocked`.
*   **Smart Confirmations:** `dangerous` tools (like `delete_file`) trigger a conversational confirmation prompt before execution.