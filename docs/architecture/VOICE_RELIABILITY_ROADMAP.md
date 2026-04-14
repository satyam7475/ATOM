# Voice reliability roadmap (JARVIS-grade)

Strategic backlog for voice I/O. Items are ordered by **impact / risk** tradeoff. Prefer tightening existing paths (`event bus`, `AtomState`, `metrics`) over new subsystems unless a row explicitly calls for one.

| Priority | Theme | Goal | Notes / code anchors |
|----------|--------|------|----------------------|
| P0 | **Voice health & recovery** | Detect STT failure, mic loss, permission changes, repeated errors; surface in UI; optional auto `restart_listening` or user prompt | Extend [`core/health_monitor.py`](../../core/health_monitor.py), `stt._last_error`, [`restart_listening`](../../core/boot/wiring.py); avoid static “pick STT once at boot” as the only story |
| P0 | **Duplex barge-in** | User can interrupt TTS with speech | Requires mic active during `SPEAKING` when enabled: [`stt.barge_in_during_speak`](../../voice/stt_macos.py); TTS stops on real partials: [`tts_macos.on_speech_partial`](../../voice/tts_macos.py). Echo risk — Apple Voice Processing I/O helps; test with headphones first |
| P1 | **Dynamic STT switching** | Optional runtime switch (e.g. offline when net drops) | New config e.g. `stt.dynamic_switch` + small supervisor; must not fight [`main.py`](../../main.py) boot selection without clear rules |
| P1 | **Observability** | STT/TTS/perceived latency and last intent visible | [`_last_perceived_ms`](../../core/boot/wiring.py) → dashboard; extend [`atom_state`](../../core/state/atom_state.py) `voice` section |
| P2 | **Cold start** | Speak-ready ASAP | Already: STT preload async; consider deferring non-voice heavy work after first `LISTENING` |
| P2 | **TTS prosody** | Rate/pause by intent/severity | [`tts_macos`](../../voice/tts_macos.py) + router metadata |
| P2 | **Voice pre-processing** | Filler strip, normalize before [`IntentEngine`](../../core/intent_engine/) | Shared text hygiene module; keep router `clean_text` single owner |
| P3 | **Voice identity** | Speaker verification gates | [`SecurityFortress`](../../core/security_fortress.py) / voice auth — policy-heavy |
| P3 | **Proactive voice** | Speak-first nudges | [`ProactiveIntelligenceEngine`](../../core/cognitive/proactive_engine.py), existing TTS |

## What not to do

- Fake `native_stt_launch_supported()` in venv — use [`stt.dev_prefer`](../../docs/architecture/VOICE_IO_HLD_LLD_CHATGPT.md) or bundle launch.
- Large “capability layer” refactor until P0/P1 items are measured in production.

## Implemented in-tree (track)

- Launch diagnostics: `VOICE_LAUNCH_DIAG` in logs; dashboard fallback trace + bundle path.
- `stt.dev_prefer` for venv explicit fallback engines.
- `stt.barge_in_during_speak` (optional): allow native mic during `SPEAKING` for interrupt-driven UX.

See also: [`14_VOICE_PIPELINE.md`](14_VOICE_PIPELINE.md), [`VOICE_IO_HLD_LLD_CHATGPT.md`](VOICE_IO_HLD_LLD_CHATGPT.md), [`docs/operations/BLUETOOTH_VOICE_TEST.md`](../operations/BLUETOOTH_VOICE_TEST.md).
