# ATOM Module 05: Expression Layer (Ring 5)

> Read this before changing: `voice/tts_*.py`, `ui/`, `core/personality*.py`

## Modules

| Module | File | Purpose |
|--------|------|---------|
| **TTSAsync** | `voice/tts_async.py` | Windows SAPI TTS (offline fallback) |
| **EdgeTTSAsync** | `voice/tts_edge.py` | Edge Neural TTS (primary, online) |
| **WebDashboard** | `ui/web_dashboard.py` | JARVIS-style web UI + WebSocket |
| **FloatingIndicator** | `ui/floating_indicator.py` | Tkinter fallback UI |
| **Personality** | `core/personality.py` | Response tone/style generator |
| **PersonalityModes** | `core/personality_modes.py` | work/focus/chill/sleep modes |

## TTS Contract (any replacement MUST implement)

```python
class TTSContract:
    async init_voice() -> None
    async on_response(text: str) -> None
    async on_partial_response(text, is_first, is_last) -> None
    async speak_ack(text: str) -> None
    async stop() -> None              # Barge-in / interrupt
    async shutdown() -> None
    # MUST emit: tts_complete
```

## TTS Engine Selection

```
config.tts.engine == "edge"  →  EdgeTTSAsync (neural, network required)
config.tts.engine == "sapi"  →  TTSAsync (offline, Windows only)
Edge import fails             →  fallback to TTSAsync
```

## UI Contract (any replacement MUST implement)

```python
class IndicatorContract:
    def start() -> None
    def shutdown() -> None
    def on_state_changed(old, new) -> None
    def add_log(category: str, text: str) -> None
    def show_hearing(text: str) -> None
    def clear_hearing() -> None
    def set_mic_name(name: str) -> None
    def set_shutdown_callback(cb) -> None
    def set_mode_change_callback(cb) -> None
```

## Personality Modes

| Mode | Voice Rate | Suggestions | Interruptions | Verbosity |
|------|-----------|-------------|---------------|-----------|
| **work** | +2 | Yes | Yes | full |
| **focus** | +5 | No | Urgent only | minimal |
| **chill** | -3 | Yes | Yes | full |
| **sleep** | -5 | No | Urgent only | silent |

Urgent events that bypass focus/sleep: `battery_critical`, `shutdown_requested`, `critical_reminder`

## WebDashboard Features

- Three.js animated orb (state-reactive)
- Real-time system status panels
- Conversation log (heard/action/info/warning)
- Performance mode switcher
- Brain profile / assistant mode toggles
- Goal tracking panel
- Prediction display
- Behavior profile
- UNSTICK button (recovers from stuck states)
- Text input (type instead of speak)

## Events Consumed

| Event | Handler |
|-------|---------|
| `response_ready` | Speak full response |
| `partial_response` | Speak sentence chunk (fake streaming) |
| `thinking_ack` | Quick acknowledgment |
| `tts_complete` | → StateManager (transition to IDLE/LISTENING) |
| `state_changed` | Update dashboard orb + status |
| `mode_changed` | Update dashboard mode display |

## Configuration

```json
{
  "tts": {
    "engine": "sapi",
    "max_lines": 4,
    "rate": 2,
    "edge_voice": "en-GB-RyanNeural",
    "edge_rate": "+0%",
    "edge_postprocess": false,
    "edge_ack_cache": true
  },
  "ui": {
    "mode": "web",
    "web_port": 8765,
    "auto_open_browser": true
  }
}
```
