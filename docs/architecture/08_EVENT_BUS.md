# ATOM Module 08: Event Bus — The Nervous System

> Read this before changing: `core/async_event_bus.py`, or when adding/consuming ANY event.

## Architecture

The `AsyncEventBus` is ATOM's backbone. EVERY module communicates through it. Zero direct coupling between modules.

### Emission Tiers

| Tier | Method | Timeout | Use Case |
|------|--------|---------|----------|
| **fast** | `emit_fast()` | None | Metrics, logging, UI state (<1ms) |
| **normal** | `emit()` | 10s, warns >5s | Standard handlers |
| **long** | `emit_long()` | 60s | TTS playback, LLM inference |

### Safety Features

- Each handler runs as independent `asyncio.Task`
- Emitter never blocks
- Failing handler cannot crash others
- Hung handler cancelled after timeout
- Active tasks tracked via `WeakSet`
- Slow handler warnings (>5s)

## Complete Event Catalog

### Core Pipeline

| Event | Payload | Emitter → Consumer |
|-------|---------|-------------------|
| `speech_partial` | `{text}` | STT → Dashboard |
| `speech_final` | `{text}` | STT → Router, Metrics |
| `intent_classified` | `{intent, ms, text, action_args}` | Router → Metrics, Autonomy, Behavior |
| `thinking_ack` | `{text}` | Router → TTS |
| `cursor_query` | `{text, memory_context, context, history}` | Router → LocalBrain |
| `cursor_response` | `{query, response}` | LocalBrain → Cache, Memory, Router |
| `response_ready` | `{text}` | Router/Autonomy → TTS, Dashboard |
| `partial_response` | `{text, is_first, is_last, source}` | LocalBrain → TTS, Dashboard |
| `text_display` | `{text}` | Router → Dashboard |
| `tts_complete` | `{}` | TTS → StateManager |

### State

| Event | Payload | Emitter → Consumer |
|-------|---------|-------------------|
| `state_changed` | `{old, new}` | StateManager → STT, Dashboard, Health, Watchdog |
| `resume_listening` | `{}` | Hotkey/Dashboard → StateManager, STT |
| `enter_sleep_mode` | `{}` | Router → STT, StateManager |
| `restart_listening` | `{}` | StateManager → STT |
| `silence_timeout` | `{}` | STT → StateManager |

### System

| Event | Payload | Emitter → Consumer |
|-------|---------|-------------------|
| `system_event` | `{kind, app, message, level, device}` | SystemWatcher → main.py |
| `media_started` | `{}` | Router → STT |
| `mic_changed` | `{name}` | HealthMonitor → Dashboard |
| `llm_error` | `{source}` | LocalBrain → StateManager |
| `shutdown_requested` | `{}` | UI → main.py |

### Autonomy + Cognitive

| Event | Payload | Emitter → Consumer |
|-------|---------|-------------------|
| `context_snapshot` | `{time_of_day, hour, cpu, ram, idle_minutes, active_app}` | Health → Behavior, Autonomy |
| `habit_suggestion` | `{text, habit_id, confidence}` | Autonomy → TTS |
| `autonomous_action` | `{action, target, habit_id, confidence}` | Autonomy → Router |
| `user_feedback` | `{habit_id, accepted}` | main.py → Autonomy |
| `goal_update` | `{goal_id, action, title}` | GoalEngine → — |
| `goal_briefing` | `{text}` | GoalEngine → TTS |
| `prediction_ready` | `{predictions: []}` | Prediction → Dashboard |
| `user_energy_state` | `{energy, idle_minutes, active_app}` | BehaviorModel → PersonalityModes |
| `mode_changed` | `{mode, old_mode, voice_rate_adj, ...}` | PersonalityModes → TTS, Dashboard |
| `optimization_suggestions` | `{suggestions: []}` | SelfOptimizer → Dashboard |
| `idle_detected` | `{idle_minutes}` | Health → — |

### Performance

| Event | Payload | Emitter → Consumer |
|-------|---------|-------------------|
| `governor_throttle` | `{cpu}` | Health → TTS, Dashboard, Autonomy |
| `governor_normal` | `{cpu}` | Health → TTS, Dashboard, Autonomy |
| `set_performance_mode` | `{mode}` | Router/Dashboard → main.py |
| `runtime_settings_changed` | `{brain_profile, assistant_mode}` | Router → Dashboard |
| `metrics_event` | `{counter}` | various → MetricsCollector |
| `metrics_latency` | `{name, ms}` | various → MetricsCollector |
| `reminder_due` | `{label, task_id}` | TaskScheduler → TTS |
| `intent_chain_suggestion` | `{suggestion, action}` | Router → Dashboard |

## Rules for Adding New Events

1. Choose the right tier (fast/normal/long)
2. Document payload shape in this file
3. Use `**_kw` in handlers to be forward-compatible with new payload fields
4. Never block in `emit_fast` handlers
5. Always handle exceptions in handlers (bus isolates, but log clearly)
