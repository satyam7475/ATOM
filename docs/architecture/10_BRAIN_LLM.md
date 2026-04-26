# ATOM Module 10: Brain / LLM System

> Read this before changing: `brain/mlx_llm.py`, `cursor_bridge/`, `core/llm_inference_queue.py`, `core/brain_mode_manager.py`

## Modules

| Module | File | Purpose |
|--------|------|---------|
| **MLXBrain** | `brain/mlx_llm.py` | Production Apple Silicon local LLM wrapper (Qwen3-8B-4bit, single-resident) |
| **MiniLLM** | `brain/mini_llm.py` | Legacy GGUF wrapper for benchmarks/tooling only |
| **LocalBrainController** | `cursor_bridge/local_brain_controller.py` | Event bus interface for LLM |
| **StructuredPromptBuilder** | `cursor_bridge/structured_prompt_builder.py` | ATOM personality prompt construction |
| **LLMInferenceQueue** | `core/llm_inference_queue.py` | Serial queue with request coalescing |
| **BrainModeManager** | `core/brain_mode_manager.py` | Profile switching (atom/balanced/brain) |
| **AssistantModeManager** | `core/assistant_mode_manager.py` | Mode switching (hybrid/command_only) |

## Role-Based Routing

```
Current production setup:
  - DIRECT / CACHE → skip LLM
  - QUICK → fast role on the shared local model
  - FULL → primary role on the shared local model
  - DEEP → primary role + larger budget / RAG when allowed
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
    "mlx_model": "models/qwen3-8b-4bit",
    "single_resident": true,
    "n_ctx": 4096,
    "n_threads": 4,
    "n_gpu_layers": -1,
    "n_batch": 256,
    "max_tokens": 320,
    "temperature": 0.6,
    "timeout_seconds": 28,
    "speculative_decoding": {
      "enabled": false,
      "draft_model_path": "models/qwen3-8b-4bit",
      "num_draft_tokens": 3
    }
  }
}
```
