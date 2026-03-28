# ATOM: Low-Level Design (LLD)

## 1. Introduction
This document provides the technical implementation details, class structures, and specific algorithms used in the ATOM Personal Cognitive AI Operating System.

## 2. Component Specifications

### 2.1 Event Bus (`core/async_event_bus.py`)
*   **Class:** `AsyncEventBus`
*   **Mechanism:** Uses `asyncio.Queue` and a background worker task to dispatch events.
*   **Methods:**
    *   `on(event_name, callback)`: Registers an async callback.
    *   `emit(event_name, **kwargs)`: Places an event on the queue.
*   **Error Handling:** Wraps callback execution in `try/except` blocks to prevent one failing handler from crashing the bus.

### 2.2 Audio Preprocessing (`voice/audio_preprocessor.py`)
*   **Class:** `AudioPreprocessor`
*   **Dependencies:** `numpy`
*   **Algorithms:**
    *   **DC Offset Removal:** Subtracts the mean of the audio array.
    *   **Pre-emphasis:** Applies a high-pass filter (`y[t] = x[t] - α * x[t-1]`, where α=0.97) to boost high frequencies.
    *   **Spectral Noise Gate:** Maintains a rolling average of ambient noise energy. Audio segments below a dynamic threshold (calculated from the noise profile) are zeroed out.
    *   **Peak Normalization:** Scales the audio array so the maximum absolute value hits a target level (e.g., 0.9).

### 2.3 System Indexer (`core/system_indexer.py`)
*   **Class:** `SystemIndexer`
*   **Data Structures:**
    *   `_apps_index`: `dict[str, AppIndexEntry]` (maps lowercase app names/aliases to paths).
    *   `_process_index`: `dict[str, int]` (maps process names to PIDs).
*   **Implementation:**
    *   Runs in a `ThreadPoolExecutor` to avoid blocking the `asyncio` event loop.
    *   Uses `winreg` to scan `SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall` for installed applications.
    *   Uses `psutil` (via `PlatformAdapter`) to snapshot running processes.

### 2.4 Media Watcher (`voice/media_watcher.py`)
*   **Class:** `MediaWatcher`
*   **Dependencies:** `winsdk.windows.media.control`
*   **Implementation:**
    *   Uses `GlobalSystemMediaTransportControlsSessionManager.request_async()` to get the current media session.
    *   Polls `session.try_get_media_properties_async()` every 2 seconds.
    *   Stores state in a `MediaInfo` dataclass (title, artist, app_name, is_playing).

### 2.5 Context Fusion (`core/context_fusion.py`)
*   **Class:** `ContextFusionEngine`
*   **Output:** `FusedContext` dataclass.
*   **Algorithm:**
    *   On `get_fused_context()`, it aggregates data from `SystemScanner`, `SystemIndexer`, `MediaWatcher`, `L1Cache`, and `OwnerUnderstanding`.
    *   Generates a formatted string via `get_llm_context_block()` that looks like:
        ```text
        [SITUATION] Morning, weekday | Session: 45min | System: 98/100
        [OWNER] Emotion: focused | Energy: normal
        [SYSTEM] Running apps: chrome, vscode | Recent files: project.py
        [MEDIA] Playing: 'Starboy' by The Weeknd (on Spotify)
        [FAST MEMORY] Boss's favorite drink: Coffee | Current project: ATOM
        ```

### 2.5b L1 Cache (`core/l1_cache.py`)
*   **Class:** `L1Cache`
*   **Data Structures:**
    *   `_cache`: `OrderedDict[str, CacheEntry]` (LRU implementation).
    *   `_sticky_cache`: `dict[str, CacheEntry]` (LFU/Permanent implementation).
*   **Algorithm:**
    *   Provides O(1) dictionary lookup for instant memory retrieval.
    *   Automatically evicts least recently used items when `max_size` is reached.
    *   `search_values` performs a fast substring scan across both caches.

### 2.6 LLM Integration (`brain/mini_llm.py` & `cursor_bridge/local_brain_controller.py`)
*   **Class:** `MiniLLM`
*   **Library:** `llama-cpp-python`
*   **Configuration:**
    *   `n_gpu_layers = -1` (Full offload to RTX GPU).
    *   `n_ctx = 32768` (Context window for Qwen3.5-4B).
    *   `n_batch = 512`
*   **Streaming:** Uses a generator to yield tokens as they are produced, enabling the TTS engine to start speaking before the full response is generated.

### 2.7 Tool Parsing & Execution (`core/reasoning/tool_parser.py` & `action_executor.py`)
*   **Parsing:** Uses Regex to extract JSON tool calls from the LLM output. Supports multiple formats (ATOM native `<tool_call>`, Qwen `✿FUNCTION✿`).
*   **Execution Flow:**
    1.  `ToolParser` yields a `ToolCall` object.
    2.  `ActionExecutor.execute()` validates the tool against `ToolRegistry`.
    3.  Checks `SecurityPolicy`.
    4.  If `requires_confirmation` is true, it calls `personality.confirmation_prompt()` to generate a buddy-like prompt and returns a `needs_confirmation=True` state.
    5.  Otherwise, it invokes the registered handler function.

### 2.8 TTS Engine (`voice/tts_kokoro.py`)
*   **Class:** `KokoroTTSAsync`
*   **Dependencies:** `kokoro-tts`, `sounddevice`
*   **Implementation:**
    *   Runs generation in a `ThreadPoolExecutor`.
    *   Uses `model.create_stream(text)` which yields `(audio_array, sample_rate)` chunks.
    *   Plays chunks immediately using `sounddevice.play()`, achieving ~80ms Time-To-First-Audio (TTFA).

## 3. Error Handling & Resilience
*   **Graceful Degradation:** If `winsdk` fails to load, `MediaWatcher` disables itself silently without crashing the system.
*   **Timeout Isolation:** `ActionExecutor` enforces timeouts on tool execution to prevent hanging the ReAct loop.
*   **Noise Floods:** `STTAsync` tracks consecutive failed audio captures. If it detects a "noise flood," it dynamically raises the energy threshold and forces a recalibration.