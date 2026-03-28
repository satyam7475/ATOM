# ATOM V3: Hybrid Multi-Process Architecture Plan

## 1. The Problem with V2
Currently, ATOM runs entirely in a single Python process. While `asyncio` and `ThreadPoolExecutor` provide concurrency, they are bound by the Python Global Interpreter Lock (GIL). 
*   **Audio Processing (STT):** CPU-bound math (`numpy` operations, VAD) blocks the event loop.
*   **LLM Inference:** Even with GPU offload, the Python bindings and token streaming callbacks consume main thread time.
*   **TTS Generation:** High-frequency audio chunk generation competes with the LLM for CPU time.
*   **Result:** Unpredictable latency spikes and potential VRAM fragmentation over long sessions.

## 2. The V3 Solution: Isolated Micro-Services
ATOM V3 will move to a multi-process architecture using ZeroMQ (ZMQ) or gRPC for inter-process communication (IPC).

### 2.1 The Core Controller (Process 1)
*   **Role:** The orchestrator. Runs the `PriorityEventBus`, `ActionExecutor`, `ContextFusion`, and `SystemIndexer`.
*   **Resource:** Very low CPU, minimal RAM. Purely I/O bound.

### 2.2 The Perception Engine (Process 2)
*   **Role:** Dedicated to listening. Runs `MicManager`, `AudioPreprocessor`, and `faster-whisper`.
*   **Resource:** High CPU (for noise gating) and ~2.5GB VRAM.
*   **IPC:** Streams `speech_partial` and `speech_final` events to the Core Controller.

### 2.3 The Cognition Engine (Process 3)
*   **Role:** The Brain. Runs `llama-cpp-python` and the `StructuredPromptBuilder`.
*   **Resource:** ~3.5GB VRAM.
*   **IPC:** Receives prompts from Core, streams tokens back to Core.
*   **Advantage:** Can be independently restarted ("soft reset") every few hours to clear CUDA fragmentation without dropping the user session.

### 2.4 The Speech Engine (Process 4)
*   **Role:** The Voice. Runs `Kokoro TTS`.
*   **Resource:** ~1.5GB VRAM.
*   **IPC:** Receives text chunks from Core, plays audio directly to `sounddevice`.

## 3. Implementation Roadmap
1.  **Phase 1: IPC Backbone:** Replace `AsyncEventBus` with a ZMQ-backed distributed bus.
2.  **Phase 2: Extract STT:** Move the STT pipeline to a standalone script that publishes to the ZMQ bus.
3.  **Phase 3: Extract LLM:** Wrap `llama-cpp-python` in a lightweight FastAPI or ZMQ server.
4.  **Phase 4: Extract TTS:** Move Kokoro to a standalone worker.

## 4. Benefits
*   **Crash Isolation:** If the LLM runs out of memory, the Core Controller stays alive, says "Give me a second, Boss", restarts the LLM process, and resumes.
*   **Zero GIL Contention:** STT math will never delay a TTS audio chunk.
*   **Hot Reloading:** Models can be swapped or updated without restarting the entire ATOM system.