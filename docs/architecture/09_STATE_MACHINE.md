# ATOM Module 09: State Machine — The Heartbeat

> Read this before changing: `core/state_manager.py`

## States

| State | Description | CPU |
|-------|-------------|-----|
| `SLEEP` | Fully shut down, no audio processing | ~0% |
| `IDLE` | Resting, minimal background work | <0.5% |
| `LISTENING` | STT active, processing microphone | 2-5% |
| `THINKING` | LLM query in flight or action processing | 5-100% |
| `SPEAKING` | TTS playing audio output | 1-3% |
| `ERROR_RECOVERY` | Transient error, auto-recovers to IDLE | <0.5% |

## Valid Transitions

```
SLEEP        → IDLE, LISTENING
IDLE         → LISTENING, SLEEP
LISTENING    → THINKING, SPEAKING, IDLE, ERROR_RECOVERY, SLEEP
THINKING     → SPEAKING, LISTENING, IDLE, ERROR_RECOVERY, SLEEP
SPEAKING     → IDLE, LISTENING, ERROR_RECOVERY, SLEEP
ERROR_RECOVERY → IDLE, SLEEP
```

Any transition NOT in this table is **blocked** with a warning log.

## Key Behaviors

- **Thread-safe:** transitions guarded by `asyncio.Lock`
- **No-op safe:** same-state transitions are silently ignored
- **Always-listen mode:** `SPEAKING → tts_complete → LISTENING` (not IDLE)
- **Auto-recovery:** `ERROR_RECOVERY → IDLE` happens automatically
- **Stuck detection:** HealthMonitor forces recovery if THINKING/SPEAKING >75s

## Events

- `state_changed(old, new)` — emitted on every successful transition
- Consumed by: STT, Dashboard, HealthMonitor, RuntimeWatchdog

## Design Rules

1. Never add a state without updating `VALID_TRANSITIONS`
2. Always transition through `state.transition()` — never set `_state` directly
3. The lock ensures atomic transitions even under concurrent access
4. Read-only `state.current` is lock-free (CPython GIL guarantee)
