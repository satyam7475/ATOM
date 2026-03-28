# ATOM Architecture Module Index

> Master index of all architecture docs. Each module is self-contained and covers one area of ATOM.

## Modules

| # | Module | File | When to Read |
|---|--------|------|-------------|
| 00 | [System Identity](00_SYSTEM_IDENTITY.md) | `00_SYSTEM_IDENTITY.md` | Before ANY change to ATOM |
| 01 | [Perception Layer](01_PERCEPTION_LAYER.md) | `01_PERCEPTION_LAYER.md` | Changing voice/, STT, mic, context |
| 02 | [Understanding Layer](02_UNDERSTANDING_LAYER.md) | `02_UNDERSTANDING_LAYER.md` | Changing intent_engine/, skills, commands |
| 03 | [Decision Layer](03_DECISION_LAYER.md) | `03_DECISION_LAYER.md` | Changing router, cache, memory |
| 04 | [Execution Layer](04_EXECUTION_LAYER.md) | `04_EXECUTION_LAYER.md` | Changing *_actions.py, desktop_control |
| 05 | [Expression Layer](05_EXPRESSION_LAYER.md) | `05_EXPRESSION_LAYER.md` | Changing TTS, UI, personality |
| 06 | [Cognition Layer](06_COGNITION_LAYER.md) | `06_COGNITION_LAYER.md` | Changing cognitive/ modules |
| 07 | [Autonomy Layer](07_AUTONOMY_LAYER.md) | `07_AUTONOMY_LAYER.md` | Changing autonomy, security, health |
| 08 | [Event Bus](08_EVENT_BUS.md) | `08_EVENT_BUS.md` | Adding/consuming ANY event |
| 09 | [State Machine](09_STATE_MACHINE.md) | `09_STATE_MACHINE.md` | Changing state_manager.py |
| 10 | [Brain / LLM](10_BRAIN_LLM.md) | `10_BRAIN_LLM.md` | Changing brain/, cursor_bridge/, LLM |
| 11 | [Performance](11_PERFORMANCE.md) | `11_PERFORMANCE.md` | Changing performance, metrics, scheduler |
| 12 | [Evolution Roadmap](12_EVOLUTION_ROADMAP.md) | `12_EVOLUTION_ROADMAP.md` | Planning new features |
| 13 | [Upgrade Playbook](13_UPGRADE_PLAYBOOK.md) | `13_UPGRADE_PLAYBOOK.md` | Replacing or adding modules |

## File → Module Mapping

| Code Path | Read Module(s) |
|-----------|---------------|
| `main.py` | 00, 08, 09 |
| `voice/*` | 01 (Perception) |
| `core/intent_engine/*` | 02 (Understanding) |
| `core/router/router.py` | 03 (Decision) |
| `core/router/*_actions.py` | 04 (Execution) |
| `core/desktop_control.py` | 04 (Execution) |
| `voice/tts_*.py` | 05 (Expression) |
| `ui/*` | 05 (Expression) |
| `core/personality*.py` | 05 (Expression) |
| `core/cognitive/*` | 06 (Cognition) |
| `core/autonomy_engine.py` | 07 (Autonomy) |
| `core/security_policy.py` | 07 (Autonomy) |
| `core/health_monitor.py` | 07 (Autonomy) |
| `core/behavior_tracker.py` | 07 (Autonomy) |
| `core/self_evolution.py` | 07 (Autonomy) |
| `core/async_event_bus.py` | 08 (Event Bus) |
| `core/state_manager.py` | 09 (State Machine) |
| `brain/*` | 10 (Brain/LLM) |
| `cursor_bridge/*` | 10 (Brain/LLM) |
| `core/llm_inference_queue.py` | 10 (Brain/LLM) |
| `core/brain_mode_manager.py` | 10 (Brain/LLM) |
| `core/priority_scheduler.py` | 11 (Performance) |
| `core/metrics.py` | 11 (Performance) |
| `core/pipeline_timer.py` | 11 (Performance) |
| `core/cache_engine.py` | 03 (Decision) |
| `core/memory_engine.py` | 03 (Decision) |
| `core/skills_registry.py` | 02 (Understanding) |
| `core/command_cache.py` | 02 (Understanding) |
| `core/command_registry.py` | 04 (Execution) |
| `core/task_scheduler.py` | 07 (Autonomy) |
| `core/process_manager.py` | 04 (Execution) |
| `context/*` | 01 (Perception) |
| `config/*` | 00 (System Identity) |
