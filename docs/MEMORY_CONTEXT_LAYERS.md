# ATOM — Memory & context layers

**Purpose:** Clarify **short-term**, **episodic timeline**, **long-term store**, and **RAG** so prompts and routing stay consistent.

---

## Layers

| Layer | Canonical module | Role |
|--------|-------------------|------|
| **Short-term (session)** | `core/conversation_memory.py` | Turn count, active topics, `recent_summary()` for the current dialogue. |
| **Episodic timeline** | `core/memory/timeline_memory.py` | Rolling window of user queries, actions, files — `summary_for_prompt()`. |
| **Long-term Q&A store** | `core/memory_engine.py` | Persisted Q&A; **async** `retrieve()` / `retrieve_with_scores()` with hybrid vector + keyword **top-k** and blended scores. |
| **RAG / documents** | `core/rag/rag_engine.py` | Chunk retrieval under time budget (`retrieve_with_time_budget`) — used inside `LocalBrainController`. |
| **L1 fast cache** | `core/l1_cache.py` | Ultra-hot facts — surfaced as `[FAST MEMORY]` in fusion. |

---

## Where they appear in prompts

1. **`ContextFusionEngine.get_llm_context_block()`** (`core/context_fusion.py`) injects labeled blocks:
   - `[SHORT_TERM / SESSION]` — conversation summary  
   - `[EPISODIC_TIMELINE]` — timeline window (config `memory.timeline_window_sec`, default 600s)  
   - `[LONG_TERM_STORE]` — **hint only** (entry count + vector/keyword mode); **not** a full async retrieve  
   - `[FAST MEMORY]` — L1 cache line  

2. **`StructuredPromptBuilder`** merges this fusion block into the structured prompt (existing behavior).

3. **Agent path** — `cursor_bridge/local_brain_controller.py` runs **async** RAG + router-provided `memory_context`; scored long-term lines can be formatted with **`MemoryEngine.format_scored_for_prompt()`** when using `retrieve_with_scores()`.

---

## Configuration

| Key | Meaning |
|-----|---------|
| `memory.timeline_window_sec` | Timeline text window for fusion (seconds). |
| `memory.top_k` | Default top-k for `MemoryEngine.retrieve` (capped by `max_vector_results`). |

---

## Ground rule

**Synchronous** fusion must not call async `MemoryEngine.retrieve()` (would block or deadlock the event loop). Full **top-k scored** retrieval stays on the **async** brain/Router path; fusion exposes timeline + session text + store **metadata** for orientation.
