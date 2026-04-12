# ATOM — Canonical vs legacy modules

**Purpose:** Single place to see which code paths are authoritative for the live runtime (`main.py` → Router → local brain) versus experimental or historical duplicates under `brain/`.

**Platform:** macOS (Apple Silicon). This doc is maintained with the ACT **consolidation** phase.

---

## Interrupt & control (user wins)

| Mechanism | Role |
|-----------|------|
| `VoiceInterruptHandler` | Preempts TTS, moves state to LISTENING, calls `LocalBrainController.request_preempt()`, **`LLMInferenceQueue.clear_pending()`** (drops coalesced next job), emits `user_interrupt`. |
| `brain/mlx_llm.MLXBrain` | `request_abort_preempt()` bumps a generation token so streaming exits; `is_generating()` reflects active MLX work. |
| `PriorityScheduler` | Voice jobs ahead of LLM; optional `JobHandle.cancel()`. |
| Web dashboard | WebSocket `execution_state` (every ~5s with system push): queue / MLX / scheduler depth. |

---

## Live execution & agent loop

| Concern | Canonical location | Notes |
|--------|---------------------|--------|
| User input routing, intents, cache, security | `core/router/` | `Router` + `ActionExecutor` dispatch |
| MLX inference, streaming, tool-use / ReAct | `cursor_bridge/local_brain_controller.py`, `brain/mlx_llm.py` | `LocalBrainController` owns the agent loop |
| Inline LLM “planner” hints inside a query | `core/cognition/planner.py` (`PlannerEngine`) | Used by `LocalBrainController` where configured |
| Tool definitions & confirmation policy | `core/reasoning/tool_registry.py`, `core/reasoning/action_executor.py` | Registry applied at startup |
| Cognitive routing (DIRECT / CACHE / QUICK / …) | `core/cognitive_kernel.py` | Central path selection |
| Memory graph + RAG hooks | `brain/memory_graph.py`, `core/rag/` | Graph DB + RAG engines wired from `main.py` |

---

## Reasoning & multi-step planning

| Module | Role |
|--------|------|
| **`core/reasoning/planner.py` (`ReasoningPlanner`)** | Multi-step templates, `needs_planning()`, `timeline_hint()` — **library module**; wire from Router or `LocalBrainController` when multi-step orchestration should drive the live path. |
| **`brain/planning_engine.py`** | Legacy plan graph used by `brain/plan_evaluator.py` / simulation stack — **deprecated for new features**; see module docstring. |

Do not add new call sites to `brain/planning_engine` unless extending the legacy evaluator pipeline.

---

## Goals, behavior, intent (duplicate namespaces)

| Legacy (`brain/`) | Runtime (`core/`) |
|-------------------|-------------------|
| `brain/goal_engine.py` | `core/cognitive/goal_engine.py` |
| `brain/behavior_model.py` | `core/cognitive/behavior_model.py` |
| `brain/intent_engine.py` | `core/intent_engine/` (package) |

The **`brain/`** copies exist for `local_cognitive_pipeline` and older experiments. **New code should import from `core/`.**

---

## `brain/` — what stays first-class

| Module | Status |
|--------|--------|
| `brain/mlx_llm.py` | **Canonical** MLX brain |
| `brain/memory_graph.py` | **Canonical** structured memory graph (also used by RAG) |
| Other `brain/*.py` | Legacy or supporting the simulation / plan-evaluator subgraph — treat as **deprecated** unless explicitly listed elsewhere |

---

## Memory & context layers

See [`MEMORY_CONTEXT_LAYERS.md`](./MEMORY_CONTEXT_LAYERS.md): short-term conversation, episodic timeline, long-term `MemoryEngine`, RAG, and L1 cache — and how `ContextFusionEngine` labels them in prompts.

---

## Security (policy gate + tiers)

| Piece | Role |
|-------|------|
| `core/security_policy.py` | **`allow_action()`** — rate limit, owner gate, degradation, lock mode, **permission tier** vs `security.mode`, feature flags, executables, confirmations, audit to `logs/audit.log`. |
| `core/security_tiers.py` | Maps intents to **tiers 1–4**; **`strict`** caps at tier **3** (blocks power / high-impact tier-4 intents). |
| `scripts/setup_api_keys.py` | **Credential setup** — encrypted storage; see [`SECURITY_SETUP.md`](./SECURITY_SETUP.md). |

---

## Proactive intelligence (goals + bounded nudges)

| Piece | Role |
|-------|------|
| `core/jarvis_core.py` | Periodic proactive loop; **`wire_intelligence`** from `main.py` attaches fusion, behavior, prediction, memory, **GoalEngine**; `jarvis_insight` emissions respect **`ProactiveInsightQuota`**. |
| `core/cognitive/proactive_engine.py` | Slower scan loop (workflows, M5, conversation); emits **`jarvis_insight`** through the same quota. |
| `core/proactive_quota.py` | Rolling hourly cap + **`logs/proactive_insights.log`** audit; low priorities (≤ `critical_priority_max`) bypass the cap. |
| `core/autonomy_engine.py` | Habits / rules → **`habit_suggestion`** / **`autonomous_action`** (not `jarvis_insight`); audited in **`logs/autonomy.log`**. |

---

## Experience layer (voice + HUD)

See [`EXPERIENCE_LAYER.md`](./EXPERIENCE_LAYER.md): STT/TTS settings, assistant/brain modes, and web dashboard wiring (`execution_state` carries live voice + mode hints).

---

## Performance observability

| Signal | Where |
|--------|--------|
| End-to-end pipeline | `core/pipeline_timer.py` — speech → intent → action → TTS; plus **`pipeline_llm_first_partial`** (cursor_query → first streamed partial). |
| Response cache budget | `core/cache_engine.py` — `max_size` / TTL from config; **`cache_evictions`** when LRU drops; **`stats()`** for entries. |
| Cognitive budgets | `core/cognitive_kernel.py` — path budgets + `LatencyController` (unchanged; measured elsewhere). |

---

## Document hierarchy

- **Roadmap / vision:** `ATOM_M5_EVOLUTION_PLAN.md`
- **Execution tracker (ACT):** `ATOM_IMPLEMENTATION_PLAN.md`
- **This file:** canonical vs duplicate modules (consolidation)

When you introduce a new subsystem, add one row to the relevant table above in the same PR.
