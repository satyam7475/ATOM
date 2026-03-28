# ATOM Module 10: Brain / LLM System

> Read this before changing: `brain/mini_llm.py`, `cursor_bridge/`, `core/llm_inference_queue.py`, `core/brain_mode_manager.py`

## Modules

| Module | File | Purpose |
|--------|------|---------|
| **MiniLLM** | `brain/mini_llm.py` | Offline GGUF model wrapper (llama.cpp) |
| **LocalBrainController** | `cursor_bridge/local_brain_controller.py` | Event bus interface for LLM |
| **StructuredPromptBuilder** | `cursor_bridge/structured_prompt_builder.py` | ATOM personality prompt construction |
| **LLMInferenceQueue** | `core/llm_inference_queue.py` | Serial queue with request coalescing |
| **BrainModeManager** | `core/brain_mode_manager.py` | Profile switching (atom/balanced/brain) |
| **AssistantModeManager** | `core/assistant_mode_manager.py` | Mode switching (hybrid/command_only) |

## Dual-Model Routing

```
Query complexity heuristic:
  - >20 words → 3B model
  - Contains complex keywords (explain, compare, analyze, debug) → 3B
  - ≤8 words + simple keywords (what is, define, who is) → 1B
  - ≤6 words → 1B
  - Default (7-20 words) → 3B if >12 words, else 1B
```

## Brain Contract (any LLM replacement MUST implement)

```python
class BrainContract:
    available: bool
    is_loaded: bool
    def request_preempt() -> None
    async warm_up() -> None
    async on_query(text, memory_context, context, history) -> None
    def close() -> None
    # MUST emit: partial_response, cursor_response, metrics_latency, llm_error
```

## Fake Streaming

LLM generates full response, then splits into sentence chunks:
```
"Hello Boss. Let me explain. Python uses..." 
→ ["Hello Boss.", "Let me explain.", "Python uses..."]
→ partial_response(is_first=True) ... partial_response(is_last=True)
```
Each chunk sent with 50ms delay — feels 2x faster to user.

## Brain Profiles

| Profile | max_tokens | n_ctx | n_threads | timeout |
|---------|-----------|-------|-----------|---------|
| **atom** | 80 | 1024 | 8 | 45s |
| **balanced** | 80 | 1536 | 8 | 90s |
| **brain** | 100 | 2048 | 8 | 120s |

Switchable at runtime via voice or dashboard.

## Priority Scheduling

```
PriorityScheduler (single worker):
  Priority 0 (VOICE):      speech processing — never delayed
  Priority 1 (LLM):        LLM inference — after voice
  Priority 2 (BACKGROUND): autonomy, maintenance — lowest
```

## Configuration

```json
{
  "brain": {
    "enabled": true,
    "model_path": "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    "model_path_1b": "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    "dual_model": true,
    "n_ctx": 2048,
    "n_threads": 8,
    "n_gpu_layers": 0,
    "max_tokens": 80,
    "temperature": 0.4,
    "timeout_seconds": 90
  }
}
```
