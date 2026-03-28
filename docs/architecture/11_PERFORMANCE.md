# ATOM Module 11: Performance Architecture

> Read this before changing: `core/priority_scheduler.py`, `core/pipeline_timer.py`, `core/metrics.py`, performance config

## Performance Modes

| Mode | Health Check | System Watch | Maintenance | Use Case |
|------|-------------|-------------|-------------|----------|
| `full` | 60s | 10s | 120s | Dedicated PC |
| `lite` | 120s | 30s | 180s | Corporate laptop |
| `ultra_lite` | 300s | 60s | 300s | Low-resource |
| `auto` | Adaptive | Adaptive | Adaptive | Latency-driven |

## Auto Mode Logic

Every 45s (skips during THINKING/SPEAKING):
```
latency > 25s → ultra_lite
latency > 12s → lite
latency < 8s AND cpu < 50% → full
Cooldown: 120s between switches
```

Also auto-switches brain profile: full→brain, lite→balanced, ultra_lite→atom.

## Latency Budget

| Stage | Target |
|-------|--------|
| STT (Whisper) | <500ms |
| Intent classify | <5ms |
| Cache lookup | <1ms |
| Memory retrieval | <5ms |
| Action dispatch | <10ms |
| LLM 1B | <5s |
| LLM 3B | <15s |
| TTS start | <100ms |
| **End-to-end (local)** | **<200ms** |
| **End-to-end (LLM)** | **<10s** |

## Metrics Collected

- `queries_total` — total queries processed
- `cache_hits` / `cache_misses` — cache performance
- `local_routed_queries` — handled by intent engine
- `llm_routed_queries` — sent to LLM
- `llm_calls` — actual LLM invocations
- `perceived` latency — speech_final → first TTS audio
- `ttfa` — time to first acknowledgment
- `llm` latency — LLM inference time
- `scheduler_queue_depth` — priority scheduler backlog

## Configuration

```json
{
  "performance": {
    "mode": "auto",
    "health_check_interval_s": 120,
    "system_watcher_interval_s": 30,
    "maintenance_interval_s": 180,
    "stuck_state_threshold_s": 75,
    "cpu_governor": true,
    "cpu_governor_threshold": 75,
    "watchdog_thinking_timeout_s": 120,
    "watchdog_speaking_timeout_s": 300,
    "use_priority_scheduler": true
  }
}
```
