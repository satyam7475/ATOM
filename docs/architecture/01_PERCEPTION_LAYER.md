# ATOM Module 01: Perception Layer (Ring 1)

> Read this before changing: `voice/`, `core/system_watcher.py`, `context/`

## Modules

| Module | File | Purpose |
|--------|------|---------|
| **STTAsync** | `voice/stt_async.py` | Multi-engine speech-to-text orchestrator |
| **MicManager** | `voice/mic_manager.py` | Mic device detection, BT preference, hot-swap |
| **AudioPreprocessor** | `voice/audio_preprocessor.py` | Audio conditioning (noise gate, normalization) |
| **SpeechDetector** | `voice/speech_detector.py` | Noise filtering, text corrections |
| **VoiceProfiles** | `voice/voice_profiles.py` | Voice configuration profiles |
| **SystemWatcher** | `core/system_watcher.py` | Network, power, Bluetooth event detection |
| **ContextEngine** | `context/context_engine.py` | Active window + clipboard context bundle |
| **PrivacyFilter** | `context/privacy_filter.py` | Redacts PII/secrets before storage or prompts |

## STT Pipeline

```
Microphone (PyAudio)
  → sr.Recognizer.listen()
  → AudioPreprocessor → faster-whisper (bilingual en+hi)
  → text corrections (speech_detector)
  → noise word filter (is_noise_word)
  → IntentEngine.quick_match() [early exit for known commands]
  → speech_final event (on bus)
```

## STT Contract (any replacement MUST implement)

```python
class STTContract:
    mic_name: str
    async preload() -> None
    async start_listening() -> None
    def stop() -> None
    def shutdown() -> None
    def on_state_changed(old, new) -> None
    def on_media_started() -> None
    def refresh_mic() -> bool
```

## Events Emitted

| Event | Payload | When |
|-------|---------|------|
| `speech_final` | `{text}` | Complete utterance recognized |
| `speech_partial` | `{text}` | Partial recognition (live preview) |
| `stt_did_not_catch` | `{}` | Recognition returned empty/gibberish |
| `stt_too_noisy` | `{}` | Background noise too high |
| `silence_timeout` | `{}` | No speech detected within timeout |
| `system_event` | `{kind, app, message, level, device}` | Network/power/BT change |

## Events Consumed

| Event | Handler |
|-------|---------|
| `state_changed` | STT starts/stops listening based on state |
| `media_started` | STT pauses during media playback |
| `restart_listening` | STT restarts after timeout |

## Noise Hardening (Corporate Office)

- BT mic minimum threshold: 1800
- Dynamic energy disabled for BT (prevents drift)
- Minimum audio duration: 0.5s (rejects clicks/pops)
- Noise flood: escalate after 2 fails, +50% threshold
- Recalibration every 90s without successful speech
- Post-TTS cooldown 600ms (absorbs earbuds echo)

## Configuration

```json
{
  "mic": { "device_name": null, "prefer_bluetooth": true },
  "stt": {
    "engine": "faster_whisper",
    "whisper_model_size": "base.en",
    "sample_rate": 16000,
    "chunk_size": 2048,
    "post_tts_cooldown_ms": 800,
    "preload": true,
    "calibration_delay_s": 2.0,
    "min_energy_threshold": 400
  },
  "context": {
    "enable_clipboard": true,
    "enable_active_window": true,
    "clipboard_max_chars": 400
  }
}
```

## Upgrade Targets

| Upgrade | Effort | Impact |
|---------|--------|--------|
| Wake word ("Hey ATOM") | Medium | Natural activation, saves CPU |
| Emotion detection (pitch/speed) | Medium | Empathetic responses |
| Multi-language (Hindi+English) | Medium | Natural for Indian users |
| Ambient sound classification | Hard | Context-aware behavior |
