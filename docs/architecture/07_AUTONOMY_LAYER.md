# ATOM Module 07: Autonomy + Self-Awareness Layer (Ring 7)

> Read this before changing: `core/autonomy_engine.py`, `core/self_evolution.py`, `core/health_monitor.py`, `core/security_policy.py`, `core/behavior_tracker.py`, `core/runtime_watchdog.py`

## Modules

| Module | File | Purpose |
|--------|------|---------|
| **AutonomyEngine** | `core/autonomy_engine.py` | Autonomous habit detection + auto-execution |
| **SelfEvolutionEngine** | `core/self_evolution.py` | Performance diagnostics + improvement suggestions |
| **HealthMonitor** | `core/health_monitor.py` | CPU governor + stuck-state recovery + context snapshots |
| **RuntimeWatchdog** | `core/runtime_watchdog.py` | Thinking/Speaking timeout enforcement |
| **BehaviorTracker** | `core/behavior_tracker.py` | Habit formation from action patterns |
| **SecurityPolicy** | `core/security_policy.py` | Central security gate for ALL actions |

## AutonomyEngine — Decision Loop

Runs every ~90s (adaptive based on CPU load):

```
1. Apply habit decay (confidence erodes over time)
2. Check rule-based decisions (high CPU, idle detection)
3. Check habits from BehaviorTracker:
   - confidence ≥ 0.95 + NOT destructive → AUTO-EXECUTE
   - confidence ≥ 0.72 → SUGGEST to user
   - user confirms → confidence +0.1
   - user denies → confidence -0.15
```

### NEVER Auto-Execute (hardcoded safety)

```
shutdown_pc, restart_pc, logoff, sleep_pc, close_app, kill_process,
empty_recycle_bin, create_folder, move_path, copy_path,
type_text, hotkey_combo, press_key
```

## SecurityPolicy — 7 Security Gates

| Gate | What It Does |
|------|-------------|
| 1. Input Sanitization | Max 2000 chars, strip shell injection chars |
| 2. Action-Level Gate | Lock mode, feature flags, allowlists |
| 3. Confirmation Flow | Configurable per-action, 25s timeout |
| 4. Shell Blocklist | Blocks format, del, reg delete, diskpart, etc. |
| 5. Hotkey Safety Tiers | safe / confirm / block per combo |
| 6. Path Allowlist | Blocks System32, ProgramData; allows home/cwd |
| 7. Privacy Filter | PII redaction, audit logging (chmod 600) |

## HealthMonitor — System Watchdog

Checks every 60-300s (based on performance mode):
- Event bus pending tasks (threshold: 50)
- State machine stuck detection (threshold: 75s)
- System resources (CPU >95%, RAM >90%)
- Microphone connectivity
- TTS mixer + consecutive failures
- Bluetooth device changes (every 4th cycle)

### CPU Governor

```
CPU > 75% for 2 cycles → governor_throttle
  → health interval × 2.5
  → TTS postprocessing disabled

CPU < 75% for 3 cycles → governor_normal
  → intervals restored
```

### Context Snapshots (every health cycle)

Emits `context_snapshot` with: `{time_of_day, hour, cpu, ram, idle_minutes, active_app, is_weekday, weekday}`
Consumed by: BehaviorModel, AutonomyEngine, PredictionEngine.

## SelfEvolutionEngine — Diagnostics

Analyzes on-demand (`self diagnostic` voice command):
- Latency analysis (perceived_avg >3s = issue)
- Cache efficiency (hit rate <15% with >20 queries = issue)
- LLM dependency (<50% local = heavy LLM dependency)
- Health score: 1-10 scale

## Events

| Event | Emitter | Consumer |
|-------|---------|----------|
| `context_snapshot` | HealthMonitor | BehaviorModel, Autonomy, Prediction |
| `governor_throttle` | HealthMonitor | TTS, Dashboard, Autonomy |
| `governor_normal` | HealthMonitor | TTS, Dashboard, Autonomy |
| `habit_suggestion` | AutonomyEngine | main.py → TTS |
| `autonomous_action` | AutonomyEngine | main.py → Router |
| `user_feedback` | main.py | AutonomyEngine |
| `autonomy_decision_log` | AutonomyEngine | Dashboard |
| `idle_detected` | HealthMonitor | — |

## Configuration

```json
{
  "autonomy": {
    "enabled": true,
    "auto_execute_threshold": 0.95,
    "suggest_threshold": 0.72,
    "idle_timeout_minutes": 10,
    "habit_decay_days": 7,
    "check_interval_s": 90
  },
  "performance": {
    "cpu_governor": true,
    "cpu_governor_threshold": 75,
    "stuck_state_threshold_s": 75,
    "watchdog_thinking_timeout_s": 120,
    "watchdog_speaking_timeout_s": 300
  },
  "security": {
    "mode": "strict",
    "audit_to_file": true,
    "require_confirmation_for": [...]
  }
}
```

## Log Files

| File | Content |
|------|---------|
| `logs/autonomy.log` | Every autonomous decision (timestamped) |
| `logs/audit.log` | Every security gate pass/block (timestamped) |
| `logs/evolution.json` | Diagnostic history across sessions |
