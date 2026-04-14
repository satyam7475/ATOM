# Voice pipeline fix — test report (2026-04-14)

## Summary

Changes target three themes: **(1)** reducing perceived **first/last word loss** on STT (native + Whisper), **(2)** reducing **skipped spoken words** during streaming TTS, and **(3)** a **female, polite, FRIDAY-like** spoken persona (voice + prompt + ack phrases).

## Configuration snapshot (`config/settings.json`)

| Area | Setting | Value | Rationale |
|------|---------|-------|-----------|
| STT | `locale` | `en-GB` | Align native recognition with UK assistant tone and user preference |
| STT | `post_tts_cooldown_ms` | `550` | Slightly shorter gap after TTS before mic reopens (balance vs echo) |
| STT | `whisper_vad` | `speech_pad_ms` 400, `min_silence_duration_ms` 220, `threshold` 0.32 | More padding at speech edges; slightly less aggressive silence cut (Whisper path) |
| STT | `min_audio_duration_s` | `0.42` | Allow shorter valid utterances without rejecting as “click” noise |
| STT | `audio_buffer_frames` | `2048` | Larger AVAudioEngine tap buffer for stabler capture |
| STT | `native_stop_audio_delay_ms` | `120` | Brief pause before `endAudio()` so the recognizer can finalize trailing phonemes |
| TTS | `macos_voice` | `Flo` | UK female eloquence voice (`com.apple.eloquence.en-GB.Flo` on this machine) |
| TTS | `macos_rate` | `188` | Calm, clear speech rate |
| TTS | `edge_voice` | `en-GB-SoniaNeural` | Female UK neural fallback if Edge path is used |

## Code changes (high level)

1. **`voice/stt_async.py`** — Whisper VAD and min-duration are **config-driven** (`stt.whisper_vad`, `min_audio_duration_s`, `whisper_no_speech_threshold`).
2. **`voice/stt_macos.py`** — Configurable **tap buffer size**; **trailing delay** before ending audio on stop.
3. **`voice/tts_macos.py`** — **Female-first** `_PREFERRED_VOICES` (Martha premium if installed, else Flo UK); **streaming dedupe** only skips **consecutive identical** chunks (avoids false “duplicate” skips that sounded like missing words).
4. **`cursor_bridge/structured_prompt_builder.py`** — **SPOKEN / VOICE PERSONA (Friday-style)** instructions for LLM text that will be read aloud.
5. **`main.py`** — Default macOS voice label default `Flo` (matches config).
6. **`core/config_schema.py`** — Schema entries for new keys.

## Local verification (non-interactive)

```
python3 -m compileall voice/stt_async.py voice/stt_macos.py voice/tts_macos.py cursor_bridge/structured_prompt_builder.py -q
# OK

python3 -c "from voice.tts_macos import _pick_best_voice; print(_pick_best_voice('Flo'))"
# picked voice id: 'com.apple.eloquence.en-GB.Flo'

python3 -c "from core.config_schema import validate_config; import json; ... "
# errors: []
```

Full pytest was not run in this environment (`pytest` module not installed on system Python). Run in your venv:

```bash
cd /path/to/ATOM && pytest tests/test_stt_macos_bundle_guard.py tests/test_atom_state_contract.py -q
```

## What to watch in logs (live ATOM.app)

Filter for:

- `atom.stt_macos` — `Native STT listening started (on-device, buffer_frames=..., locale=...)` and `STT final: '...'`
- `atom.stt` — Whisper path: `Captured X.Xs of audio` and VAD is implicit in model config
- `atom.tts_macos` — `macOS TTS ready` with voice name; avoid repeated `TTS stream duplicate chunk skipped` unless the model truly repeats a line

## Residual risks / tuning

- **Echo after TTS**: If mic picks up speaker audio, shorten utterances or raise `post_tts_cooldown_ms` again (e.g. 650–800).
- **Native STT**: If **delay before `endAudio`** feels sluggish at stop, lower `native_stop_audio_delay_ms` (e.g. 80).
- **Whisper-only users**: If hallucinations increase, raise `whisper_vad.threshold` slightly (e.g. 0.38).

## Persona note

FRIDAY tone is reinforced by: **Flo** (or Martha when downloaded), **structured prompt** spoken persona block, and **ack phrase** list in `voice/tts_macos.py`. Restart the app after changes so the prompt cache rebuilds on a fresh `StructuredPromptBuilder` instance.
