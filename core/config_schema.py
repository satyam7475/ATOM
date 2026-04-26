"""
ATOM -- Configuration schema validation.

Validates config/settings.json at startup using jsonschema.
Falls back gracefully if jsonschema is not installed.

Validates:
    - chunk_size (int, range)
    - silence timeouts (numeric)
    - gate multipliers (numeric)
    - gain caps (numeric)
    - mic sample rate (int, range)
    - tts / cache / memory / brain parameters
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("atom.config")

CONFIG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "mic": {
            "type": "object",
            "properties": {
                "device_name": {
                    "type": ["string", "null"],
                    "description": "Reserved for future PyAudio device binding; native macOS STT uses AVAudioEngine and follows the system default input (set in System Settings).",
                },
                "prefer_bluetooth": {
                    "type": "boolean",
                    "description": "If false, use system default mic only (avoids noisy BT HFP).",
                },
            },
            "additionalProperties": False,
        },
        "audio_intelligence": {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "description": "Enable the Audio Intelligence Engine for automatic device discovery, testing, and selection.",
                },
                "auto_select": {
                    "type": "boolean",
                    "description": "Automatically select the best audio device at boot.",
                },
                "active_test_duration_s": {
                    "type": "number",
                    "minimum": 0.5,
                    "maximum": 10.0,
                    "description": "Duration in seconds to record from each device during active testing.",
                },
                "allow_bluetooth": {
                    "type": "boolean",
                    "description": "Allow Bluetooth devices to be selected (penalised but not excluded).",
                },
                "bluetooth_penalty": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Score penalty applied to Bluetooth devices (0.0-1.0).",
                },
                "min_rms_threshold_db": {
                    "type": "number",
                    "minimum": -120,
                    "maximum": 0,
                    "description": "Minimum RMS in dBFS; devices below this are rejected as dead/muted.",
                },
                "monitoring_interval_s": {
                    "type": "number",
                    "minimum": 5,
                    "maximum": 120,
                    "description": "Seconds between watchdog health checks.",
                },
                "degradation_checks_before_switch": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Consecutive failed health checks before triggering a device switch.",
                },
                "voice_feedback": {
                    "type": "boolean",
                    "description": "Speak device status changes via TTS (Jarvis personality).",
                },
                "prefer_device": {
                    "type": ["string", "null"],
                    "description": "Preferred device name; gets a +0.1 score bonus when matched.",
                },
                "vad_aggressiveness": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 3,
                    "description": "WebRTC VAD aggressiveness (0=least aggressive, 3=most).",
                },
                "set_system_default": {
                    "type": "boolean",
                    "description": "Programmatically set macOS system default input via CoreAudio.",
                },
                "debug_logging": {
                    "type": "boolean",
                    "description": "Enable verbose audio intelligence debug logs.",
                },
                "context_aware": {
                    "type": "boolean",
                    "description": "Enable context-aware device switching based on active app, time, and activity.",
                },
                "device_learning": {
                    "type": "boolean",
                    "description": "Enable persistent device history to learn which devices work best over time.",
                },
                "predictive_switching": {
                    "type": "boolean",
                    "description": "Enable predictive switching that detects quality degradation trends before failure.",
                },
                "predictive_rms_slope_threshold": {
                    "type": "number",
                    "minimum": -5.0,
                    "maximum": 0.0,
                    "description": "RMS slope threshold (dB/check) below which predictive pre-warming triggers.",
                },
                "predictive_snr_slope_threshold": {
                    "type": "number",
                    "minimum": -5.0,
                    "maximum": 0.0,
                    "description": "SNR slope threshold (dB/check) below which predictive pre-warming triggers.",
                },
                "night_suppress_feedback": {
                    "type": "boolean",
                    "description": "Suppress voice feedback during night hours (22:00-07:00).",
                },
                "low_confidence_ask_user": {
                    "type": "boolean",
                    "description": "Emit audio_confirm_needed event when switch confidence is low instead of switching silently.",
                },
            },
            "additionalProperties": False,
        },
        "stt": {
            "type": "object",
            "properties": {
                "engine": {
                    "type": "string",
                    "enum": [
                        "auto",
                        "macos_native",
                        "whisper_cpp",
                        "whispercpp",
                        "whisper",
                        "whisper.cpp",
                        "whisperkit",
                        "whisper_kit",
                        "whisper-kit",
                        "wk",
                        "faster_whisper",
                        "google_online",
                        "google",
                    ],
                    "description": "STT: whisperkit uses Argmax's CoreML-on-ANE WhisperKit (Sprint P3.3, highest-ROI on Apple Silicon). whisper_cpp uses the Metal-accelerated whisper.cpp backend (Sprint B). On macOS, auto prefers WhisperKit when its CLI is present, then whisper.cpp, then macos_native (SFSpeechRecognizer). faster_whisper/google_* are for non-macOS / legacy configs.",
                },
                "whisper_model_path": {
                    "type": "string",
                    "description": "Sprint B: ggml model path consumed by voice/stt_whisper.py. Relative names resolve under ./models/. Default: ggml-small.en-q5_1.bin (small.en-q5_0 was removed upstream Apr 2026 and is auto-redirected).",
                },
                "whisper_n_threads": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 16,
                    "description": "whisper.cpp inference threads (Metal parallelism). 4 is a good default on M-series.",
                },
                "whisper_language": {
                    "type": "string",
                    "description": "BCP-47 language code (e.g. 'en'). Used by the whisper.cpp transcribe call.",
                },
                "whisper_partial_interval_s": {
                    "type": "number",
                    "minimum": 0.25,
                    "maximum": 5.0,
                    "description": "Cadence at which whisper.cpp emits partial transcripts.",
                },
                "whisper_trailing_silence_s": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 3.0,
                    "description": "Trailing silence (seconds) after speech that triggers a final transcript.",
                },
                "whisper_max_utterance_s": {
                    "type": "number",
                    "minimum": 1.0,
                    "maximum": 60.0,
                    "description": "Hard cap on a single utterance length before forcing a final.",
                },
                "whisper_vad_aggressiveness": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 3,
                    "description": "WebRTC VAD aggressiveness (0=permissive, 3=strict).",
                },
                "whisper_model_size": {
                    "type": "string",
                    "enum": [
                        "tiny",
                        "base",
                        "small",
                        "medium",
                        "large-v3",
                        "large-v3-turbo",
                    ],
                    "description": (
                        "Whisper model size. P1.1 (Apr 26 2026) added "
                        "'large-v3-turbo' for the q5_0 multilingual GGML "
                        "weights checked into models/ggml-large-v3-turbo-q5_0.bin."
                    ),
                },
                "bilingual": {
                    "type": "boolean",
                    "description": "Enable bilingual STT hints / handling.",
                },
                "sample_rate": {
                    "type": "integer",
                    "minimum": 8000,
                    "maximum": 48000,
                    "description": "Mic sample rate in Hz",
                },
                "chunk_size": {
                    "type": "integer",
                    "minimum": 256,
                    "maximum": 16384,
                    "description": "Audio buffer chunk size in samples",
                },
                "post_tts_cooldown_ms": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 2000,
                },
                "preload": {
                    "type": "boolean",
                },
                "calibration_delay_s": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 10,
                    "description": "Seconds to wait before first calibration to avoid TTS.",
                },
                "min_energy_threshold": {
                    "type": "number",
                    "minimum": 50,
                    "maximum": 2000,
                    "description": "Minimum speech energy (default 100; lower = more sensitive).",
                },
                "locale": {
                    "type": "string",
                    "description": "BCP-47 locale for native macOS STT (e.g. en-US, en-GB). Use \"auto\" to follow the Mac primary locale (same idea as Siri language).",
                },
                "whisper_vad": {
                    "type": "object",
                    "description": "faster-whisper VAD tuning (reduces clipped first/last words).",
                    "properties": {
                        "min_silence_duration_ms": {"type": "integer", "minimum": 50, "maximum": 2000},
                        "speech_pad_ms": {"type": "integer", "minimum": 0, "maximum": 1000},
                        "threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                    "additionalProperties": False,
                },
                "whisper_no_speech_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "min_audio_duration_s": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 3.0,
                    "description": "Reject shorter captures as noise (Whisper path).",
                },
                "audio_buffer_frames": {
                    "type": "integer",
                    "minimum": 256,
                    "maximum": 8192,
                    "description": "AVAudioEngine tap buffer size (native STT).",
                },
                "native_stop_audio_delay_ms": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 500,
                    "description": "Brief delay before endAudio so the last word can finalize.",
                },
                "voice_debug": {
                    "type": "boolean",
                    "description": "Log VOICE_DEBUG / VOICE_INPUT lines for mic, permissions, and AtomState.",
                },
                "voice_debug_interval_s": {
                    "type": "number",
                    "minimum": 5,
                    "maximum": 120,
                    "description": "Seconds between VOICE_DEBUG heartbeat lines.",
                },
                "dev_prefer": {
                    "type": "string",
                    "enum": [
                        "",
                        "faster_whisper",
                        "whisper",
                        "offline",
                        "google_online",
                        "google",
                    ],
                    "description": "Legacy / no-op on macOS (native-only STT). Reserved for non-macOS tooling.",
                },
                "barge_in_during_speak": {
                    "type": "boolean",
                    "description": "If true, native STT may open the mic during SPEAKING so user speech can interrupt TTS (test with headphones; echo can false-trigger).",
                },
                "native_requires_on_device": {
                    "type": "boolean",
                    "description": "If true, SFSpeechAudioBufferRecognitionRequest uses on-device only; if false, Apple may use server-assisted recognition (network).",
                },
                "native_voice_processing": {
                    "type": "boolean",
                    "description": "If true, enable AVAudioEngine Voice Processing I/O; if false, often better for Bluetooth headsets.",
                },
                "native_tap_sample_rate": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 48000,
                    "description": "Force mono tap at this sample rate for Speech (e.g. 48000); 0 = match device format (default).",
                },
                "noise_floor_dbfs": {
                    "type": "number",
                    "minimum": -96.0,
                    "maximum": 0.0,
                    "description": "Phase E3 RMS noise gate. Frames below this dBFS for noise_gate_consecutive in a row are dropped before SFSpeechRecognizer. Set to -96 (or below) to disable.",
                },
                "noise_gate_consecutive": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 60,
                    "description": "Phase E3: number of consecutive sub-floor frames required to close the noise gate. A single supra-floor frame reopens it.",
                },
                "promotion_min_confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Phase E4: absolute SFSpeechRecognizer confidence floor for any promotion. Hypotheses below this never reach the LLM, even with wake context.",
                },
                "promotion_min_confidence_no_wake": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Phase E4: stricter confidence floor for cold openers (no recent wake context). Lets a high-confidence sentence land but blocks medium-confidence noise during silence.",
                },
                "passive_revert_delay_s": {
                    "type": "number",
                    "minimum": 1.0,
                    "maximum": 120.0,
                    "description": "Seconds of silence after TTS completion before the dual-channel listening mode reverts to PASSIVE (wake-word gated). Reset on every TTS reply and user utterance.",
                },
                "whisper_confirm": {
                    "type": "object",
                    "description": (
                        "v3 Phase 4 second-pass STT. Re-decodes suspect "
                        "streaming finals with faster-whisper for materially "
                        "lower WER on short, low-confidence utterances. "
                        "Sprint Ω.7 (2026-04-26): kept disabled in shipped "
                        "settings — when last enabled in production, the "
                        "synchronous faster-whisper transcribe call (~1-3s) "
                        "blocked the asyncio loop from inside _emit_final, "
                        "producing 5-7s 'Slow handler' bus warnings across "
                        "every speech_partial / speech_final / fs_event "
                        "subscriber. Before flipping enabled=true again, "
                        "refactor voice/stt_*.py::_consume_once / _emit_final "
                        "to dispatch the WhisperConfirmer.confirm() call via "
                        "loop.run_in_executor (or convert _emit_final to "
                        "async); otherwise the demo-stable bus will stall "
                        "exactly as logged on 2026-04-26 13:22 (lines "
                        "6883-6893 of logs/atom.log)."
                    ),
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "model_size": {
                            "type": "string",
                            "enum": ["tiny", "tiny.en", "base", "base.en", "small", "small.en"],
                        },
                        "ring_seconds": {"type": "number", "minimum": 1.0, "maximum": 15.0},
                        "decode_seconds": {"type": "number", "minimum": 1.0, "maximum": 10.0},
                        "min_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "max_confirm_ms": {"type": "number", "minimum": 50.0, "maximum": 2000.0},
                        "language": {"type": ["string", "null"]},
                        "min_text_chars": {"type": "integer", "minimum": 0, "maximum": 20},
                    },
                    "additionalProperties": False,
                },
                "whisperkit": {
                    "type": "object",
                    "description": "Sprint P3.3 (Apr 26 2026). Argmax WhisperKit (CoreML on Apple Neural Engine). Spawns `whisperkit-cli serve` as a long-running subprocess and POSTs PCM utterances over HTTP. Replaces whisper.cpp Metal as the highest-throughput STT path on Apple Silicon when the CLI is installed (`brew install whisperkit-cli`).",
                    "properties": {
                        "model": {
                            "type": "string",
                            "description": "WhisperKit model identifier. Default: openai_whisper-large-v3-v20240930. The CLI will download on first use when auto_download=true.",
                        },
                        "host": {
                            "type": "string",
                            "description": "Bind host for the local serve subprocess. Use 127.0.0.1 unless you know what you're doing.",
                        },
                        "port": {
                            "type": "integer",
                            "minimum": 1024,
                            "maximum": 65535,
                            "description": "Bind port for the serve subprocess. Default 50060.",
                        },
                        "auto_download": {
                            "type": "boolean",
                            "description": "If true, pass `--download` to `whisperkit-cli serve` so the model is fetched on first use.",
                        },
                        "startup_timeout_s": {
                            "type": "number",
                            "minimum": 1.0,
                            "maximum": 300.0,
                            "description": "Seconds to wait for whisperkit-cli serve to bind its port before declaring failure.",
                        },
                        "model_dir": {
                            "type": ["string", "null"],
                            "description": "Optional override for the directory the CLI uses to store downloaded models. None = inherit $WHISPERKIT_HOME.",
                        },
                    },
                    "additionalProperties": False,
                },
                "smart_turn_taker": {
                    "type": "object",
                    "description": "Sprint Ω9 -- adaptive end-of-turn detector built on Silero VAD. When enabled the trailing-silence ceiling becomes a soft floor; final cuts can drop to ~min_silence_s and pauses extend up to max_silence_s when the model judges the speaker is mid-thought.",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "sample_rate": {"type": "integer", "minimum": 8000, "maximum": 48000},
                        "decision_window_s": {"type": "number", "minimum": 0.2, "maximum": 4.0},
                        "min_silence_s": {"type": "number", "minimum": 0.05, "maximum": 1.0},
                        "max_silence_s": {"type": "number", "minimum": 0.3, "maximum": 4.0},
                        "eot_probability_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "midthought_lockout_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "min_eval_interval_ms": {"type": "number", "minimum": 10.0, "maximum": 500.0},
                    },
                    "additionalProperties": False,
                },
            },
            "additionalProperties": False,
        },
        "tts": {
            "type": "object",
            "properties": {
                "engine": {
                    "type": "string",
                    "enum": ["sapi", "edge", "kokoro", "macos_native"],
                },
                "kokoro_voice": {
                    "type": "string",
                    "description": "Voice profile for Kokoro TTS (e.g., af_heart, am_adam)",
                },
                "max_lines": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                },
                "rate": {
                    "type": "integer",
                    "minimum": -10,
                    "maximum": 10,
                },
                "edge_voice": {
                    "type": "string",
                },
                "edge_rate": {
                    "type": "string",
                },
                "edge_postprocess": {
                    "type": "boolean",
                },
                "edge_ack_cache": {
                    "type": "boolean",
                },
                "macos_voice": {
                    "type": "string",
                    "description": "NSSpeechSynthesizer voice: \"system\" (default, same family as macOS Spoken Content / Siri on-device TTS), or a name substring (e.g. Flo, Martha).",
                },
                "macos_rate": {
                    "type": "integer",
                    "minimum": 50,
                    "maximum": 500,
                    "description": "Words per minute for native macOS TTS.",
                },
                "macos_first_word_warmup_ms": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 1000,
                    "description": "Phase E1: pre-roll silence (ms) before the first NSSpeechSynthesizer.startSpeakingString_ so Bluetooth / USB-C dongles don't latch onto the first phoneme. Skipped during continuous speech.",
                },
                "macos_tail_drain_ms": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 1000,
                    "description": "Phase E2: post-speech sleep (ms) after isSpeaking() flips to False so CoreAudio's render buffer fully flushes the last sample on wired outputs.",
                },
                "macos_tail_drain_bluetooth_ms": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 1000,
                    "description": "Phase E2: post-speech sleep (ms) for Bluetooth outputs (extra ~80ms hardware latency). Set automatically via audio_intelligence output-change subscription.",
                },
                "macos_warmup_skip_window_ms": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 5000,
                    "description": "Phase E1: window (ms) within which a follow-up speak skips the warmup pre-roll because the audio device is still hot.",
                },
                "kokoro_speed": {
                    "type": "number",
                    "minimum": 0.5,
                    "maximum": 2.0,
                    "description": "Sprint Ω2 -- Kokoro neural TTS playback speed multiplier. 1.0 = native model rate.",
                },
                "kokoro_language": {
                    "type": "string",
                    "description": "Sprint Ω2 -- Kokoro language tag passed to the phonemizer (e.g. 'en-us', 'en-gb').",
                },
                "kokoro_model_path": {
                    "type": "string",
                    "description": "Sprint Ω2 -- path to the Kokoro ONNX model file (default models/kokoro/kokoro-v1.0.onnx).",
                },
                "kokoro_voices_path": {
                    "type": "string",
                    "description": "Sprint Ω2 -- path to the Kokoro voices.bin file (default models/kokoro/voices-v1.0.bin).",
                },
            },
            "additionalProperties": False,
        },
        "voice": {
            "type": "object",
            "properties": {
                "activation_mode": {
                    "type": "string",
                    "enum": ["always_on", "wake_word", "jarvis"],
                    "description": "Voice command activation policy. 'always_on' keeps STT routed continuously; 'wake_word' preserves passive/active gating. 'jarvis' is accepted as an alias for always_on.",
                },
                "earcons": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "volume": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                        "heartbeat_enabled": {"type": "boolean"},
                        "heartbeat_interval_s": {
                            "type": "number",
                            "minimum": 1.0,
                            "maximum": 3600.0,
                        },
                    },
                    "additionalProperties": False,
                },
            },
            "additionalProperties": False,
        },
        "wake_word": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "sensitivity": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "model": {
                    "type": "string",
                    "description": "Primary OpenWakeWord model name (e.g. hey_jarvis).",
                },
                "models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "Optional list of OpenWakeWord model names to load together.",
                },
            },
            "additionalProperties": False,
        },
        "context": {
            "type": "object",
            "properties": {
                "enable_clipboard": {"type": "boolean"},
                "enable_active_window": {"type": "boolean"},
                "clipboard_max_chars": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 5000,
                },
            },
            "additionalProperties": False,
        },
        "cache": {
            "type": "object",
            "properties": {
                "max_size": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10000,
                },
                "ttl_seconds": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 86400,
                },
            },
            "additionalProperties": False,
        },
        "semantic_cache": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "max_size": {
                    "type": "integer",
                    "minimum": 16,
                    "maximum": 4096,
                    "description": "In-memory hot-set size (LRU).",
                },
                "ttl_seconds": {
                    "type": "number",
                    "minimum": 10,
                    "maximum": 86400,
                    "description": "Session-level TTL for cached answers.",
                },
                "threshold": {
                    "type": "number",
                    "minimum": 0.5,
                    "maximum": 1.0,
                    "description": "Minimum cosine similarity for a semantic hit.",
                },
                "min_jaccard_overlap": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Phase D4: minimum token-overlap (Jaccard, stopwords excluded) required alongside the cosine similarity before returning a cached answer. Blocks topically-irrelevant cache hits that pass embedding-only similarity.",
                },
                "persistent": {
                    "type": "boolean",
                    "description": "Persist cache to SQLite so it survives restarts.",
                },
                "persistent_max": {
                    "type": "integer",
                    "minimum": 100,
                    "maximum": 1000000,
                    "description": "Max entries in the on-disk store (LRU-evicted).",
                },
                "persistent_ttl_seconds": {
                    "type": "number",
                    "minimum": 60,
                    "maximum": 30 * 24 * 3600,
                    "description": "TTL for on-disk entries (default 7 days).",
                },
                "db_path": {
                    "type": "string",
                    "description": "SQLite file for the durable cache.",
                },
            },
            "additionalProperties": False,
        },
        "embedding": {
            "type": "object",
            "properties": {
                "backend": {
                    "type": "string",
                    "enum": [
                        "sentence_transformers",
                        "legacy",
                        "fastembed",
                        "onnx",
                        "mlx",
                        "mlx_embeddings",
                    ],
                    "description": "Sprint P3.4 (Apr 26 2026): mlx / mlx_embeddings runs Apple's mlx-embeddings package on the Apple Neural Engine -- materially faster than torch-MPS on Apple Silicon. Falls back to sentence_transformers automatically if the package is missing.",
                },
                "provider_version": {"type": "string"},
                "shadow_compare": {"type": "boolean"},
                "shadow_phrases": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "model": {"type": "string"},
                "dimension": {"type": "integer", "minimum": 1, "maximum": 4096},
                "device": {"type": "string"},
                "warm_file": {
                    "oneOf": [
                        {"type": "string"},
                        {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "path": {"type": "string"},
                                "max_entries": {"type": "integer", "minimum": 1},
                            },
                            "additionalProperties": False,
                        },
                    ],
                },
            },
            "additionalProperties": False,
        },
        "memory": {
            "type": "object",
            "properties": {
                "graph_db_path": {
                    "type": "string",
                    "description": "SQLite path for MemoryGraph (V7 timeline + RAG graph hints).",
                },
                "vector_path": {
                    "type": "string",
                    "description": "Chroma persistence path for MemoryGraph vectors; defaults to vector_store.path.",
                },
                "max_entries": {
                    "type": "integer",
                    "minimum": 10,
                    "maximum": 10000,
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                },
                "max_vector_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Hard cap for semantic/vector memory candidates under normal load.",
                },
                "max_graph_nodes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100000,
                    "description": "Maximum retained MemoryGraph nodes before pruning old/low-value nodes.",
                },
                "pressure_threshold_pct": {
                    "type": "number",
                    "minimum": 50,
                    "maximum": 99,
                    "description": "Unified-memory usage percent that enables memory pressure mode.",
                },
                "pressure_relief_pct": {
                    "type": "number",
                    "minimum": 40,
                    "maximum": 95,
                    "description": "Unified-memory usage percent below which memory pressure mode clears.",
                },
                "pressure_tiers": {
                    "type": "object",
                    "description": "Per-tier pressure thresholds used by the main orchestrator (warn -> active -> critical).",
                    "properties": {
                        "warn_pct": {
                            "type": "number",
                            "minimum": 40,
                            "maximum": 99,
                            "description": "Memory usage that triggers tier 1 (drop MLX prompt-prefix KV cache).",
                        },
                        "warn_relief_pct": {
                            "type": "number",
                            "minimum": 30,
                            "maximum": 95,
                            "description": "Memory usage that clears tier 1.",
                        },
                        "critical_pct": {
                            "type": "number",
                            "minimum": 70,
                            "maximum": 99,
                            "description": "Memory usage that triggers tier 3 (unload sentence-transformer weights, fall back to warm-file/keyword search).",
                        },
                        "critical_relief_pct": {
                            "type": "number",
                            "minimum": 40,
                            "maximum": 95,
                            "description": "Memory usage that drops the pressure tier back below 3.",
                        },
                    },
                    "additionalProperties": False,
                },
                "pressure_top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Retrieval top_k used while memory pressure mode is active.",
                },
                "pressure_query_limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Maximum MemoryGraph query result limit while memory pressure mode is active.",
                },
                "semantic_weight": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "v7_scoring": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "recency_weight": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "importance_weight": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "success_rate_weight": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "similarity_weight": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                    "additionalProperties": False,
                },
            },
            "additionalProperties": False,
        },
        "brain": {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "description": "Enable local offline LLM brain.",
                },
                "model_path": {
                    "type": "string",
                    "description": "Legacy GGUF fallback model path (not used by the default MLX runtime).",
                },
                "mlx_model": {
                    "type": "string",
                    "description": (
                        "Path to the single MLX model directory ATOM "
                        "loads at boot. ATOM v3.2 runs one local model; "
                        "legacy mlx_primary_model / mlx_fast_model / "
                        "mlx_deep_model / mlx_default_role / model_path "
                        "are still accepted by the brain loader for "
                        "backwards compatibility but should be removed."
                    ),
                },
                "mlx_primary_model": {
                    "type": "string",
                    "description": "DEPRECATED: legacy alias for mlx_model.",
                },
                "mlx_fast_model": {
                    "type": "string",
                    "description": (
                        "Sprint P3.1 (Apr 26 2026) -- optional dual-tier "
                        "brain. When set to a *separate, smaller* model "
                        "directory (e.g. models/qwen3-4b-instruct-4bit) "
                        "the brain loads it for the 'fast' role while "
                        "'primary' keeps using ``mlx_model``. Roughly "
                        "doubles RAM, so leave unset on 16 GB Macs unless "
                        "you have profiled and want the headroom trade. "
                        "Missing directories silently fall back to "
                        "single-tier."
                    ),
                },
                "mlx_deep_model": {
                    "type": "string",
                    "description": (
                        "DEPRECATED: the v3.2 brain is single-model; "
                        "deep reasoning routes to cloud (Gemini) via the "
                        "cognitive kernel. Ignored at load time."
                    ),
                },
                "mlx_default_role": {
                    "type": "string",
                    "enum": ["primary", "fast", "deep"],
                    "description": "DEPRECATED: the brain has a single role slot.",
                },
                "n_ctx": {
                    "type": "integer",
                    "minimum": 64,
                    "maximum": 131072,
                    "description": "Context window size in tokens.",
                },
                "n_threads": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 32,
                    "description": "CPU threads for inference.",
                },
                "n_gpu_layers": {
                    "type": "integer",
                    "minimum": -1,
                    "maximum": 100,
                    "description": "GPU layers to offload (-1 = all layers, 0 = CPU only).",
                },
                "n_batch": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 8192,
                    "description": "Batch size for prompt processing / inference.",
                },
                "max_tokens": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 4096,
                    "description": "Maximum tokens to generate per response.",
                },
                "temperature": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 2.0,
                    "description": "Sampling temperature (lower = more focused).",
                },
                "top_p": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Top-p nucleus sampling cutoff.",
                },
                "repeat_penalty": {
                    "type": "number",
                    "minimum": 0.5,
                    "maximum": 3.0,
                    "description": "Penalty for repeated tokens.",
                },
                "timeout_seconds": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 300,
                    "description": "Timeout for local LLM inference.",
                },
                "prompt_cache_enabled": {
                    "type": "boolean",
                    "description": "Reuse the KV state of the constant system prefix across turns for lower first-token latency.",
                },
                "prompt_cache_max_size": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 64,
                    "description": "Max number of turns kept in the LRU prompt-prefix KV cache.",
                },
                "prompt_cache_max_mb": {
                    "type": "integer",
                    "minimum": 64,
                    "maximum": 8192,
                    "description": "Upper bound on memory used by the prompt-prefix KV cache (per MLX role).",
                },
                "prompt_cache_persist": {
                    "type": "boolean",
                    "description": "Persist the warm system-prompt KV state to disk on first turn and restore on next boot to skip the ~7s cold prefill.",
                },
                "prompt_cache_persist_path": {
                    "type": "string",
                    "description": "Filesystem path for the persisted prompt cache (per-role suffixes are appended automatically).",
                },
                "prompt_cache_persist_min_tokens": {
                    "type": "integer",
                    "minimum": 32,
                    "maximum": 8192,
                    "description": "Don't persist cache snapshots smaller than this — saves disk on degenerate first turns.",
                },
                "kv_bits": {
                    "type": "integer",
                    "enum": [0, 4, 8],
                    "description": "Sprint C5: KV cache quantisation bits (mlx-lm 0.22+). 0 disables; 8 halves KV memory and frees ~10-15% throughput on long prompts; 4 is even tighter but quality drops on the FAST role.",
                },
                "kv_group_size": {
                    "type": "integer",
                    "minimum": 16,
                    "maximum": 256,
                    "description": "KV quant group size (mlx-lm default 64). Smaller = better quality, larger = faster.",
                },
                "kv_quant_warmup_tokens": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 8192,
                    "description": "Tokens to keep at full precision before switching to quantised KV cache. Higher = better quality on the first sentence; lower = bigger memory savings sooner.",
                },
                "mlx_model_fallback": {
                    "type": "string",
                    "description": "Sprint Ω1c -- secondary MLX model directory used when ``mlx_model`` cannot be loaded (e.g. half-downloaded weights, OOM under thermal pressure). Sprint Ω.7 (Apr 26 2026): ATOM ships with the same path as ``mlx_model`` so the fallback is a no-op; the dual-model story (4B-as-fallback) was retired to keep RAM constant under any condition. Set to a separate model directory only if you have profiled and want a degraded brain to keep speaking when the primary fails to load.",
                },
                "mx_compile_enabled": {
                    "type": "boolean",
                    "description": "Sprint P3.5 (Apr 26 2026): wrap the per-token sampler in ``mx.compile`` to fuse temp/top_p ops into one kernel. 10-25% steady-state speedup on M-series with macOS 26.2+. Falls back to eager sampler if compile fails.",
                },
                "speculative_decoding": {
                    "type": "object",
                    "description": "Sprint P3.2 (Apr 26 2026): MLX speculative decoding. Use a small draft model (e.g. Qwen3-4B-Instruct-4bit) to predict candidate tokens that the target model verifies in parallel. 1.5-2x tokens/s on warm runs per Apple's MLX-LM examples. Off by default; enable after profiling. Sprint Ω.7 (Apr 26 2026): structurally incompatible with ``single_resident=true`` (target+draft are co-resident by definition); the brain refuses the draft load when both flags are set.",
                    "properties": {
                        "enabled": {
                            "type": "boolean",
                            "description": "Master switch.",
                        },
                        "draft_model_path": {
                            "type": "string",
                            "description": "Directory containing the smaller draft model. MUST share a tokenizer with the target model.",
                        },
                        "num_draft_tokens": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 16,
                            "description": "Number of tokens the draft model proposes per verification step. Higher = more potential speedup but more rejected proposals on uncertain prefixes. 3-5 is a typical sweet spot.",
                        },
                    },
                    "additionalProperties": False,
                },
                "single_resident": {
                    "type": "boolean",
                    "description": "Sprint Ω.7 (Apr 26 2026): exactly one MLX chat model in RAM at a time. When true, ``_ensure_loaded`` evicts any sibling role whose loaded path differs from the requested one before bringing the new weights in, and the speculative-decoding draft load is refused. Default true on 16 GB Apple Silicon (M1-M5) where co-resident 4B+8B causes thermal throttle. Set false only on machines with >=24 GB unified memory that explicitly trade RAM for throughput.",
                },
                "role_switch_min_interval_s": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 600.0,
                    "description": "Sprint Ω.7 (Apr 26 2026): hysteresis floor (seconds) between two single-resident model swaps. Prevents the cognitive kernel's `primary` <-> `fast` decision from ping-ponging on borderline system-context readings (e.g. memory_pct hovering around the threshold) and paying full model-load cost every turn.",
                },
            },
            "additionalProperties": False,
        },
        "ui": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["web", "tkinter", "native"],
                },
                "web_port": {
                    "type": "integer",
                    "minimum": 1024,
                    "maximum": 65535,
                },
                "auto_open_browser": {"type": "boolean"},
                "password_auth_enabled": {
                    "type": "boolean",
                    "description": "If false, dashboard token gate is disabled (no password-style token auth).",
                },
                "dashboard_access_token": {
                    "type": "string",
                    "description": "If set, WebSocket /ws requires ?token= or X-ATOM-Token header.",
                },
                "voice_only_input": {
                    "type": "boolean",
                    "description": "If true, dashboard WebSocket text_input is ignored; speech comes from STT only.",
                },
                "jarvis_boot_splash": {
                    "type": "boolean",
                    "description": "If true, dashboard shows a brief boot splash until the first state sync.",
                },
            },
            "additionalProperties": False,
        },
        "executor": {
            "type": "object",
            "properties": {
                "max_workers": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 16,
                },
            },
            "additionalProperties": False,
        },
        "developer": {
            "type": "object",
            "properties": {
                "role": {"type": "string"},
                "focus": {"type": "string"},
                "project_name": {"type": "string"},
            },
            "additionalProperties": True,
        },
        "auth": {
            "type": "object",
            "properties": {
                "sessions_enabled": {"type": "boolean"},
                "session_ttl_s": {"type": "number", "minimum": 60},
                "session_max_idle_s": {"type": "number", "minimum": 60},
                "privilege_default": {"type": "string"},
                "persist_sessions": {"type": "boolean"},
                "session_db_path": {"type": "string"},
                "revoke_on_ws_close": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
        "owner": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "title": {"type": "string"},
                "registered_device_fingerprint": {
                    "type": "string",
                    "description": "SHA256 prefix from device_binding.get_device_id(); enforced in paranoid mode.",
                },
            },
            "additionalProperties": True,
        },
        "cross_device": {
            "type": "object",
            "description": (
                "Phase 1 iPhone Shortcuts bridge. When enabled, a local HTTP "
                "listener accepts POST /faceid, /presence, /trigger from an "
                "iPhone's Shortcuts app. No Xcode, no Apple developer account."
            ),
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "description": "Master switch. Off by default so a fresh install does not open a port.",
                },
                "bridge_port": {
                    "type": "integer",
                    "minimum": 1024,
                    "maximum": 65535,
                    "description": "Preferred listener port. On bind failure the bridge tries port+1 and port+2.",
                },
                "bind_host": {
                    "type": "string",
                    "description": "Bind address. 127.0.0.1 for localhost-only (recommended); LAN IP to accept Wi-Fi POSTs from iPhone.",
                },
                "faceid_freshness_s": {
                    "type": "number",
                    "minimum": 30,
                    "maximum": 3600,
                    "description": "Seconds a Face ID verification stays 'fresh' for tier-3 gate. 300 (5 min) is the default.",
                },
                "allow_origins": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Allow-listed source IPs (reserved for Phase 1.5; enforcement lives in bridge_auth).",
                },
                "token_path": {
                    "type": "string",
                    "description": "Path to the pre-shared bridge token file. Auto-minted at first boot.",
                },
                "audit_log_path": {
                    "type": "string",
                    "description": "Append-only JSONL log of auth failures + 409 rejections.",
                },
                "trusted_device_path": {
                    "type": "string",
                    "description": "Single-iPhone UDID-hash state file. Reset by deleting or via `python -m core.cross_device.trusted_device reset`.",
                },
                "port_file_path": {
                    "type": "string",
                    "description": "Path written with the actual bound port so Shortcuts can pick it up after fallback.",
                },
                "openai_compat": {
                    "type": "object",
                    "description": (
                        "Sprint P4.4 (Apr 26 2026): OpenAI-compatible "
                        "/v1/models + /v1/chat/completions shim, used by "
                        "Enchanted on iOS over Tailscale. The shim only "
                        "registers when the bridge is wired with a "
                        "``chat_stream`` callable, so flipping ``enabled`` "
                        "off in config is a hard, audit-friendly disarm."
                    ),
                    "properties": {
                        "enabled": {
                            "type": "boolean",
                            "description": (
                                "Master switch. Default true; flip false "
                                "to fully disarm /v1/* even if the bridge "
                                "is wired."
                            ),
                        },
                        "model_id": {
                            "type": "string",
                            "description": (
                                "Public model id returned by /v1/models. "
                                "'atom-local' by default; rename if you "
                                "want clients to recognise a custom name."
                            ),
                        },
                        "default_max_tokens": {
                            "type": "integer",
                            "minimum": 16,
                            "maximum": 8192,
                            "description": (
                                "Per-request token cap if the client "
                                "doesn't supply max_tokens. 256 is a sane "
                                "default for chat-style replies."
                            ),
                        },
                        "rate_window_s": {
                            "type": "number",
                            "minimum": 1.0,
                            "description": (
                                "Sliding rate-limit window for /v1/* "
                                "requests in seconds. Inherits the bridge "
                                "auth contract on top of this."
                            ),
                        },
                    },
                    "additionalProperties": True,
                },
            },
            "additionalProperties": True,
        },
        "vision": {
            "type": "object",
            "description": (
                "Camera + Apple Vision (Neural Engine) settings. ATOM is "
                "MacBook-camera only as of Apr 26 2026 (built-in FaceTime HD "
                "webcam on the laptop) — iPhone Continuity Camera is "
                "intentionally not used as a vision source. Detection runs "
                "on-device via VNDetectFaceRectangles; no LLM/VLM is loaded "
                "by this subsystem."
            ),
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "description": (
                        "Master switch. False keeps AVFoundation untouched and "
                        "the camera lights stay off."
                    ),
                },
                "preferred_camera": {
                    "type": "string",
                    "enum": ["builtin", "auto", "continuity"],
                    "description": (
                        "builtin (default) = the laptop's built-in FaceTime "
                        "HD webcam. auto = alias of builtin after the "
                        "iPhone-camera removal. continuity = deprecated "
                        "(legacy configs only); aliased to builtin and "
                        "logged once at startup."
                    ),
                },
                "explicit_camera_uid": {
                    "type": ["string", "null"],
                    "description": (
                        "AVCaptureDevice uniqueID; if set, wins over "
                        "preferred_camera. Use to pin a specific external "
                        "(non-iPhone) USB rig."
                    ),
                },
                "boot_face_check": {
                    "type": "boolean",
                    "description": (
                        "If true, ATOM captures one frame at end of boot to "
                        "detect a face and log 'I see you' / 'no face yet'. "
                        "Off by default — opt in once you trust the camera."
                    ),
                },
                "boot_face_check_announce": {
                    "type": "boolean",
                    "description": (
                        "If true *and* boot_face_check is true *and* a face is "
                        "detected, ATOM speaks one short greeting confirming "
                        "it can see the user. False = log only."
                    ),
                },
                "capture_timeout_s": {
                    "type": "number",
                    "minimum": 0.5,
                    "maximum": 10.0,
                    "description": (
                        "Max wall time to wait for a single frame from the "
                        "AVCaptureSession. The built-in FaceTime HD typically "
                        "wakes in <500 ms; the higher end of this range "
                        "accommodates first-call cold start after lid-open."
                    ),
                },
                "min_gap_s": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 60.0,
                    "description": (
                        "Minimum spacing between consecutive captures from the "
                        "engine. Throttle for tool-call retry loops."
                    ),
                },
                "audit_log_path": {
                    "type": "string",
                    "description": (
                        "JSONL audit log path (one line per capture). 0o600 on "
                        "first write."
                    ),
                },
                "camera_index": {
                    "type": "integer",
                    "minimum": 0,
                    "description": (
                        "Legacy opencv index (used only by "
                        "scripts/enroll_owner_face.py)."
                    ),
                },
                "check_interval_seconds": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 60,
                    "description": "Legacy poll cadence (unused by VisionEngine).",
                },
                "require_owner_for_sensitive": {"type": "boolean"},
                "owner_encoding_path": {
                    "type": "string",
                    "description": (
                        "Legacy face_recognition encoding (only "
                        "scripts/enroll_owner_face.py writes this)."
                    ),
                },
                "describe_on_wake": {
                    "type": "boolean",
                    "description": (
                        "When true AND vision.vlm.enabled is true, ATOM "
                        "captures one frame each time the wake phrase "
                        "fires and runs the VLM captioner. The caption "
                        "is stashed on the vision engine so the next "
                        "LLM turn gets a ``visual_context`` entry in "
                        "its context_bundle. This is the single flag "
                        "that makes ATOM feel always-watching — off by "
                        "default to respect privacy + keep boot clean."
                    ),
                },
                "caption_max_age_s": {
                    "type": "number",
                    "minimum": 1.0,
                    "maximum": 600.0,
                    "description": (
                        "How long a VLM caption stays 'fresh' for "
                        "visual_context injection. After this many "
                        "seconds the caption is treated as stale and "
                        "not attached to new LLM turns."
                    ),
                },
                "vlm": {
                    "type": "object",
                    "description": (
                        "Visual Language Model settings (SmolVLM-Instruct-"
                        "4bit via mlx-vlm; ~1.2 GB on disk). Opt-in via "
                        "the ``enabled`` flag because the weights are a "
                        "separate download."
                    ),
                    "properties": {
                        "enabled": {
                            "type": "boolean",
                            "description": (
                                "Master switch for the VLM captioner. "
                                "When false, ``vision_describe`` falls "
                                "back to ``vision_look`` and "
                                "``describe_on_wake`` is a no-op."
                            ),
                        },
                        "model_path": {
                            "type": "string",
                            "description": (
                                "Local directory containing the mlx-vlm "
                                "weights. Default: "
                                "models/smolvlm-instruct-4bit. Fetch "
                                "with: hf download "
                                "mlx-community/SmolVLM-Instruct-4bit "
                                "--local-dir models/smolvlm-instruct-4bit"
                            ),
                        },
                        "model_repo": {
                            "type": "string",
                            "description": (
                                "Optional Hugging Face repo id used as "
                                "fallback when ``model_path`` is missing "
                                "on disk. mlx-vlm + huggingface_hub will "
                                "fetch on first use. Leave empty for "
                                "strict offline operation."
                            ),
                        },
                        "prompt": {
                            "type": "string",
                            "description": (
                                "Default prompt for short scene captions. "
                                "Tool calls may override with their own."
                            ),
                        },
                        "max_tokens": {
                            "type": "integer",
                            "minimum": 4,
                            "maximum": 256,
                            "description": (
                                "Hard cap on caption length. 48 keeps "
                                "output at one natural sentence."
                            ),
                        },
                        "temperature": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 2.0,
                            "description": (
                                "Sampling temperature. 0.0 = greedy, "
                                "deterministic, best for consistency."
                            ),
                        },
                    },
                    "additionalProperties": True,
                },
            },
            "additionalProperties": True,
        },
        "security": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["strict", "relaxed"],
                    "description": "strict = corporate rules; relaxed = allow more executables.",
                },
                "audit_to_file": {"type": "boolean"},
                "require_confirmation_for": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Action names that always require voice confirmation.",
                },
                "rate_limit_window_s": {"type": "number"},
                "rate_limit_max_actions": {"type": "integer"},
                "action_signing_secret": {
                    "type": "string",
                    "description": "HMAC secret; override with ATOM_ACTION_SECRET env.",
                },
                "paranoid_require_session_even_when_local_trust": {"type": "boolean"},
                "paranoid_signing_disabled": {
                    "type": "boolean",
                    "description": "If true, paranoid mode skips HMAC verification (not recommended).",
                },
                "behavior_monitor": {"type": "object", "additionalProperties": True},
            },
            "additionalProperties": True,
        },
        "features": {
            "type": "object",
            "properties": {
                "desktop_control": {"type": "boolean"},
                "file_ops": {"type": "boolean"},
                "llm": {"type": "boolean"},
                "system_analyze": {"type": "boolean"},
                "web_research": {
                    "type": "boolean",
                    "description": "DuckDuckGo web research (requires internet).",
                },
                "online_weather": {
                    "type": "boolean",
                    "description": "wttr.in weather (requires internet).",
                },
                "proactive_awareness": {
                    "type": "boolean",
                    "description": "Time-of-day greetings, app-context tips, idle hints.",
                },
            },
            "additionalProperties": False,
        },
        "control": {
            "type": "object",
            "properties": {
                "lock_mode": {
                    "type": "string",
                    "enum": [
                        "off",
                        "safe_only",
                        "owner_only",
                        "open",
                        "restricted",
                        "secure",
                        "paranoid",
                    ],
                    "description": "Canonical: open|restricted|secure|paranoid. Legacy: off→open, safe_only→restricted, owner_only→secure.",
                },
                "executor_mode": {
                    "type": "string",
                    "enum": ["in_process", "isolated"],
                    "description": "in_process = ActionExecutor in main; isolated = subprocess IPC worker.",
                },
                "assistant_mode": {
                    "type": "string",
                    "enum": ["command_only", "hybrid", "conversational"],
                    "description": "command_only = no LLM on fallback; hybrid/conversational = allow inference.",
                },
                "allow_runtime_mode_switch": {
                    "type": "boolean",
                    "description": "If false, voice/dashboard cannot change brain profile or assistant mode.",
                },
                "persist_assistant_mode": {"type": "boolean"},
                "restore_persisted_assistant_mode": {"type": "boolean"},
                "audit_assistant_mode_changes": {"type": "boolean"},
                "command_only_message": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "assistant_brain": {
            "type": "object",
            "description": "Local LLM profiles (optimal/full_performance with legacy aliases) and static quick replies.",
            "additionalProperties": True,
        },
        "performance": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["optimal", "full_performance", "auto", "full", "lite", "ultra_lite"],
                    "description": "optimal/full_performance = requested M5 modes; auto = promote/demote from telemetry. Legacy full|lite|ultra_lite are accepted as aliases.",
                },
                "auto_threshold_high": {
                    "type": "integer",
                    "minimum": 50,
                    "maximum": 95,
                    "description": "CPU percent above which the auto tuner forces Optimal mode.",
                },
                "auto_threshold_mid": {
                    "type": "integer",
                    "minimum": 20,
                    "maximum": 70,
                    "description": "CPU percent below which the auto tuner may promote to Full Performance when other telemetry is healthy.",
                },
                "health_check_interval_s": {
                    "type": "number",
                    "minimum": 30,
                    "maximum": 600,
                    "description": "Seconds between health watchdog checks.",
                },
                "stuck_state_threshold_s": {
                    "type": "number",
                    "minimum": 30,
                    "maximum": 600,
                    "description": "THINKING/SPEAKING with no state change for this long -> auto LISTENING.",
                },
                "system_watcher_interval_s": {
                    "type": "number",
                    "minimum": 5,
                    "maximum": 300,
                    "description": "Seconds between system watcher polls.",
                },
                "maintenance_interval_s": {
                    "type": "number",
                    "minimum": 60,
                    "maximum": 600,
                    "description": "Seconds between periodic maintenance cycles.",
                },
                "proactive_alerts": {
                    "type": "boolean",
                    "description": "Enable battery / idle proactive voice alerts.",
                },
                "idle_reminder": {
                    "type": "boolean",
                    "description": "Enable 'I'm here whenever you need me' idle reminders.",
                },
                "cpu_governor": {
                    "type": "boolean",
                    "description": "Auto-throttle ATOM background work when system CPU > threshold.",
                },
                "cpu_governor_threshold": {
                    "type": "integer",
                    "minimum": 30,
                    "maximum": 95,
                    "description": "CPU percent above which governor throttles background tasks.",
                },
                "watchdog_thinking_timeout_s": {
                    "type": "number",
                    "minimum": 30,
                    "maximum": 600,
                    "description": "RuntimeWatchdog: THINKING dwell before auto recovery.",
                },
                "watchdog_speaking_timeout_s": {
                    "type": "number",
                    "minimum": 60,
                    "maximum": 3600,
                    "description": "RuntimeWatchdog: SPEAKING dwell before auto recovery.",
                },
                "watchdog_intent_timeout_ms": {
                    "type": "number",
                    "minimum": 10,
                    "maximum": 500,
                    "description": "RuntimeWatchdog: Intent engine budget before forced fallback.",
                },
                "watchdog_intent_boot_grace_s": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 120,
                    "description": (
                        "Disables the intent_engine budget for the first N "
                        "seconds after RuntimeWatchdog construction so the "
                        "first turn after a cold boot isn't killed while "
                        "JIT-compiling regex paths and warming caches."
                    ),
                },
                "watchdog_cache_timeout_ms": {
                    "type": "number",
                    "minimum": 10,
                    "maximum": 2000,
                    "description": "RuntimeWatchdog: Cache lookup budget before skipping cache.",
                },
                "watchdog_rag_timeout_ms": {
                    "type": "number",
                    "minimum": 50,
                    "maximum": 5000,
                    "description": "RuntimeWatchdog: Hard cap for RAG retrieval budget.",
                },
                "watchdog_llm_timeout_s": {
                    "type": "number",
                    "minimum": 5,
                    "maximum": 300,
                    "description": "RuntimeWatchdog: LLM inference timeout before preempt + reset.",
                },
                "watchdog_tts_timeout_s": {
                    "type": "number",
                    "minimum": 3,
                    "maximum": 120,
                    "description": "RuntimeWatchdog: TTS synthesis static-floor timeout before audio skip.",
                },
                "watchdog_tts_per_word_s": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 5.0,
                    "description": "RuntimeWatchdog: Per-word TTS budget multiplier; effective budget = max(floor, words * this).",
                },
                "watchdog_tts_max_dynamic_s": {
                    "type": "number",
                    "minimum": 5.0,
                    "maximum": 600.0,
                    "description": "RuntimeWatchdog: Hard cap for the dynamic per-utterance TTS budget.",
                },
                "watchdog_tool_timeout_s": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 120,
                    "description": "RuntimeWatchdog: Tool execution timeout before aborting the step.",
                },
                "supervisor_restart_cooldown_s": {
                    "type": "number",
                    "minimum": 2,
                    "maximum": 120,
                    "description": "Minimum seconds between watchdog recovery bursts.",
                },
                "watchdog_poll_interval_s": {
                    "type": "number",
                    "minimum": 0.5,
                    "maximum": 30,
                    "description": "Seconds between stuck-state polls.",
                },
                "error_recovery_hold_s": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 10,
                    "description": "Seconds to remain in ERROR_RECOVERY before returning to IDLE.",
                },
                "use_priority_scheduler": {
                    "type": "boolean",
                    "description": "Route speech_final and cursor_query through PriorityScheduler (voice > LLM > background).",
                },
            },
            "additionalProperties": False,
        },
        "cognitive_kernel": {
            "type": "object",
            "properties": {
                "quick_model": {"type": "string"},
                "full_model": {"type": "string"},
                "simple_query_max_chars": {"type": "integer", "minimum": 8, "maximum": 200},
                "deep_query_min_chars": {"type": "integer", "minimum": 40, "maximum": 2000},
                "battery_degrade": {"type": "boolean"},
                "thermal_degrade": {"type": "boolean"},
                "memory_pressure_threshold": {
                    "type": "number",
                    "minimum": 40,
                    "maximum": 98,
                },
                "rag_complexity_threshold": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "memory_governor": {
            "type": "object",
            "description": "Per-role tunable eviction order on top of SiliconGovernor "
            "(Sprint \u03a9.4.C, Apr 26 2026). Roles are evicted in declared order "
            "as unified-memory pressure escalates through three tiers; the last "
            "role is sacred and only released at tier 3.",
            "properties": {
                "enabled": {"type": "boolean"},
                "tier1_threshold_pct": {
                    "type": "number",
                    "minimum": 40,
                    "maximum": 95,
                    "description": "Memory % that triggers first eviction wave.",
                },
                "tier2_threshold_pct": {
                    "type": "number",
                    "minimum": 50,
                    "maximum": 97,
                    "description": "Memory % that triggers second eviction wave.",
                },
                "tier3_threshold_pct": {
                    "type": "number",
                    "minimum": 60,
                    "maximum": 99,
                    "description": "Memory % that triggers final eviction wave "
                    "(everything except the sacred last role).",
                },
                "rewarm_hysteresis_pct": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 30,
                    "description": "Drop-below offset before a tier is considered "
                    "relaxed; prevents flapping when pressure dances at a threshold.",
                },
                "eviction_order": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "Roles in declared eviction priority. The last "
                    "entry is treated as sacred and only evicted at tier 3.",
                },
            },
            "additionalProperties": False,
        },
        "latency_controller": {
            "type": "object",
            "properties": {
                "direct_budget_ms": {"type": "number", "minimum": 10, "maximum": 1000},
                "cache_budget_ms": {"type": "number", "minimum": 10, "maximum": 2000},
                "quick_budget_ms": {"type": "number", "minimum": 100, "maximum": 5000},
                "full_budget_ms": {"type": "number", "minimum": 500, "maximum": 15000},
                "deep_budget_ms": {"type": "number", "minimum": 1000, "maximum": 30000},
                "simple_query_max_chars": {"type": "integer", "minimum": 8, "maximum": 200},
                "memory_pressure_pct": {"type": "number", "minimum": 40, "maximum": 98},
                "low_battery_pct": {"type": "integer", "minimum": 1, "maximum": 50},
                "cpu_busy_pct": {"type": "number", "minimum": 40, "maximum": 100},
                "battery_scale": {"type": "number", "minimum": 0.2, "maximum": 2.0},
                "low_battery_scale": {"type": "number", "minimum": 0.2, "maximum": 2.0},
                "thermal_scale": {"type": "number", "minimum": 0.2, "maximum": 2.0},
                "memory_scale": {"type": "number", "minimum": 0.2, "maximum": 2.0},
                "cpu_scale": {"type": "number", "minimum": 0.2, "maximum": 2.0},
                "simple_scale": {"type": "number", "minimum": 0.2, "maximum": 2.0},
                "deep_scale": {"type": "number", "minimum": 0.2, "maximum": 3.0},
                "rag_fraction_full": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "rag_fraction_deep": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "rag_min_ms": {"type": "number", "minimum": 0, "maximum": 5000},
                "rag_max_ms": {"type": "number", "minimum": 0, "maximum": 10000},
            },
            "additionalProperties": False,
        },
        "autonomy": {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "description": "Enable/disable the autonomy engine entirely.",
                },
                "auto_execute_threshold": {
                    "type": "number",
                    "minimum": 0.5,
                    "maximum": 1.0,
                    "description": "Habit confidence at or above which auto-execution triggers.",
                },
                "suggest_threshold": {
                    "type": "number",
                    "minimum": 0.2,
                    "maximum": 1.0,
                    "description": "Habit confidence at or above which a suggestion is offered.",
                },
                "idle_timeout_minutes": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 120,
                    "description": "Minutes of inactivity before idle_detected event fires.",
                },
                "habit_decay_days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 90	,
                    "description": "Days since last occurrence before confidence starts decaying.",
                },
                "habit_decay_rate": {
                    "type": "number",
                    "minimum": 0.001,
                    "maximum": 0.5,
                    "description": "in the per-decay-cycle confidence reduction for stale habits.",
                },
                "max_habits": {
                    "type": "integer",
                    "minimum": 5,
                    "maximum": 500,
                    "description": "Maximum number of tracked habits (weakest evicted first).",
                },
                "check_interval_s": {
                    "type": "number",
                    "minimum": 15,
                    "maximum": 600,
                    "description": "Base seconds between autonomy decision cycles.",
                },
                "log_all_decisions": {
                    "type": "boolean",
                    "description": "Write every autonomous decision to logs/autonomy.log.",
                },
            },
            "additionalProperties": False,
        },
        "conversation_memory": {
            "type": "object",
            "description": "Short-term conversation memory with topic tracking and prior-turn session context.",
            "properties": {
                "max_turns": {
                    "type": "integer",
                    "minimum": 2,
                    "maximum": 30,
                },
            },
            "additionalProperties": False,
        },
        "morning_briefing": {
            "type": "object",
            "description": "First-wake-of-the-day briefing (battery + weather + calendar + news).",
            "properties": {
                "enabled": {"type": "boolean"},
                "wake_hour_start": {"type": "integer", "minimum": 0, "maximum": 23},
                "wake_hour_end": {"type": "integer", "minimum": 0, "maximum": 23},
                "include_battery": {"type": "boolean"},
                "include_weather": {"type": "boolean"},
                "include_calendar": {"type": "boolean"},
                "include_news": {"type": "boolean"},
                "news_count": {"type": "integer", "minimum": 1, "maximum": 10},
                "calendar_timeout_s": {"type": "number", "minimum": 0.5, "maximum": 15.0},
                "state_path": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "session": {
            "type": "object",
            "description": "Prior-turn summary config (read by ConversationMemory).",
            "properties": {
                "enabled": {"type": "boolean"},
                "max_query_snippet_chars": {
                    "type": "integer",
                    "minimum": 40,
                    "maximum": 500,
                },
            },
            "additionalProperties": False,
        },
        "skills": {
            "type": "object",
            "description": "Named phrase expansions (config/skills.json).",
            "properties": {
                "enabled": {"type": "boolean"},
                "path": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "deployment": {
            "type": "object",
            "description": "Where ATOM runs: corporate_laptop, personal, or workstation.",
            "properties": {
                "profile": {
                    "type": "string",
                    "enum": [
                        "corporate_laptop",
                        "personal",
                        "personal_desktop",
                        "workstation",
                        "unset",
                    ],
                },
                "dashboard_badge": {
                    "type": "boolean",
                    "description": "Show profile pill on web dashboard top bar.",
                },
                "product_tier": {
                    "type": "string",
                    "enum": ["local_only", "balanced", "cloud_augmented"],
                    "description": "Operator label: privacy-first vs cloud help. Maps to cloud.enabled / semantic_cache in docs; single source of truth remains cloud.enabled.",
                },
            },
            "additionalProperties": False,
        },
        "cognitive_loop": {
            "type": "object",
            "description": "Phase G runtime cognitive loop (reflective/presence/scene/mood/suggester).",
            "properties": {
                "enabled": {"type": "boolean"},
                "reflective": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "cooldown_s": {"type": "number"},
                        "min_user_chars": {"type": "integer"},
                        "max_tokens": {"type": "integer"},
                    },
                    "additionalProperties": False,
                },
                "presence": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "interval_s": {"type": "number"},
                        "min_interval_s": {"type": "number"},
                    },
                    "additionalProperties": False,
                },
                "scene": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "cooldown_s": {"type": "number"},
                        "significance_min_seconds": {"type": "number"},
                    },
                    "additionalProperties": False,
                },
                "mood": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "min_consecutive": {"type": "integer"},
                    },
                    "additionalProperties": False,
                },
                "suggester": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "cooldown_s": {"type": "number"},
                        "category_cooldown_s": {"type": "number"},
                        "daily_cap": {"type": "integer"},
                        "relevance_threshold": {"type": "number"},
                        "quiet_hours": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                        "suppress_moods": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "additionalProperties": False,
                },
            },
            "additionalProperties": False,
        },
        "cognitive": {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "description": "Master switch for the cognitive layer.",
                },
                "goals_enabled": {
                    "type": "boolean",
                    "description": "Enable goal-based intelligence.",
                },
                "predictions_enabled": {
                    "type": "boolean",
                    "description": "Enable predictive action engine.",
                },
                "behavior_model_enabled": {
                    "type": "boolean",
                    "description": "Enable personal behavior model.",
                },
                "self_optimizer_enabled": {
                    "type": "boolean",
                    "description": "Enable self-optimization engine.",
                },
                "prediction_check_interval_s": {
                    "type": "number",
                    "minimum": 30,
                    "maximum": 600,
                    "description": "Seconds between prediction checks.",
                },
                "behavior_update_interval_s": {
                    "type": "number",
                    "minimum": 120,
                    "maximum": 3600,
                    "description": "Seconds between full profile updates.",
                },
                "goal_evaluation_interval_s": {
                    "type": "number",
                    "minimum": 300,
                    "maximum": 86400,
                    "description": "Seconds between goal evaluations.",
                },
                "prediction_min_confidence": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 1.0,
                    "description": "Minimum confidence for predictions.",
                },
                "prediction_preload_enabled": {
                    "type": "boolean",
                    "description": "Warm lightweight resources for high-confidence predictions.",
                },
                "prediction_preload_min_confidence": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 1.0,
                    "description": "Minimum confidence before prediction preloading runs.",
                },
                "prediction_preload_max_items": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Maximum predicted items to preload per cycle.",
                },
                "prediction_preload_cooldown_s": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 3600,
                    "description": "Cooldown before preloading the same prediction again.",
                },
                "prediction_preload_timeout_s": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 30,
                    "description": "Timeout for a single prediction preload action.",
                },
                "auto_mode_switching": {
                    "type": "boolean",
                    "description": "Allow auto personality mode switching.",
                },
                "default_mode": {
                    "type": "string",
                    "enum": ["work", "focus", "chill", "sleep"],
                    "description": "Default personality mode.",
                },
                "max_goals": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Maximum number of goals.",
                },
                "max_predictions": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Max predictions to show.",
                },
                "optimizer_check_interval_s": {
                    "type": "number",
                    "minimum": 300,
                    "maximum": 7200,
                    "description": "Seconds between optimizer checks.",
                },
                "energy_inference_interval_s": {
                    "type": "number",
                    "minimum": 30,
                    "maximum": 600,
                    "description": "Seconds between energy state updates.",
                },
                "dream_enabled": {
                    "type": "boolean",
                    "description": "Enable dream/idle consolidation mode.",
                },
                "dream_idle_minutes": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 1440,
                    "description": "Idle minutes before dream mode can activate.",
                },
                "dream_interval_hours": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 168,
                    "description": "Hours between dream/consolidation passes.",
                },
                "curiosity_enabled": {
                    "type": "boolean",
                    "description": "Enable curiosity-driven proactive exploration.",
                },
                "curiosity_max_per_hour": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 60,
                    "description": "Maximum curiosity events per hour.",
                },
                "curiosity_cooldown_minutes": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1440,
                    "description": "Cooldown between curiosity events.",
                },
                "goal_tool_auto_complete": {
                    "type": "boolean",
                    "description": "When true, completing a ReAct tool run that matches a goal step's suggested_tool auto-marks that step done.",
                },
                "goal_tool_match_strict": {
                    "type": "boolean",
                    "description": "When true, suggested_args must overlap executed tool arguments before auto-complete.",
                },
                "dream_require_idle_signal": {
                    "type": "boolean",
                    "description": "When true, dream cycles only run after HealthMonitor idle_detected exceeds dream_idle_minutes.",
                },
                "dream_min_interactions": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Minimum session interactions buffered before a dream cycle runs.",
                },
                "dream_prune_second_brain": {
                    "type": "boolean",
                    "description": "When true, dream cycle prunes very old low-importance SecondBrain facts.",
                },
                "dream_prewarm_embeddings": {
                    "type": "boolean",
                    "description": "When true, dream cycle warms embedding cache via lightweight SecondBrain.retrieve probes.",
                },
                "dream_prewarm_retrieve_topics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Topics passed to SecondBrain.retrieve during embedding prewarm.",
                },
            },
            "additionalProperties": False,
        },
        "proactive_engine": {
            "type": "object",
            "description": "Background proactive intelligence scan (workflow, M5 context triggers).",
            "properties": {
                "check_interval_s": {
                    "type": "number",
                    "minimum": 60,
                    "maximum": 3600,
                },
                "m5_triggers": {
                    "type": "object",
                    "additionalProperties": True,
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "battery_low_pct": {"type": "number", "minimum": 1, "maximum": 50},
                        "memory_high_pct": {"type": "number", "minimum": 50, "maximum": 100},
                        "disk_free_gb_warn": {"type": "number", "minimum": 1, "maximum": 128},
                        "project_stale_days": {"type": "number", "minimum": 1, "maximum": 90},
                        "idle_goal_nudge_minutes": {
                            "type": "number",
                            "minimum": 5,
                            "maximum": 240,
                        },
                        "morning_briefing_hours": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 0, "maximum": 23},
                        },
                    },
                },
            },
            "additionalProperties": False,
        },
        "v7_intelligence": {
            "type": "object",
            "description": "V7 intelligence layer: modes, timeline, prediction prefetch.",
            "properties": {
                "default_mode": {
                    "type": "string",
                    "enum": ["FAST", "SMART", "DEEP", "SECURE"],
                },
                "auto_mode": {"type": "boolean"},
                "mode_stability_enabled": {"type": "boolean"},
                "simple_query_max_chars": {"type": "integer", "minimum": 8, "maximum": 200},
                "timeline_max_events": {"type": "integer", "minimum": 50, "maximum": 10000},
                "max_timeline_size": {"type": "integer", "minimum": 50, "maximum": 10000},
                "timeline_summarize_on_prune": {"type": "boolean"},
                "prediction_prefetch_enabled": {"type": "boolean"},
                "gpu_util_fast_threshold": {
                    "type": "number",
                    "minimum": 50,
                    "maximum": 100,
                },
                "deep_query_min_chars": {"type": "integer", "minimum": 40, "maximum": 2000},
                "prefer_secure_when_paranoid_ui": {"type": "boolean"},
                "secure_rag_budget_factor": {
                    "type": "number",
                    "minimum": 0.2,
                    "maximum": 1.0,
                },
                "cpu_force_fast_above": {
                    "type": "number",
                    "minimum": 50,
                    "maximum": 100,
                },
                "cpu_idle_deep_below": {
                    "type": "number",
                    "minimum": 5,
                    "maximum": 80,
                },
                "low_prediction_accuracy_deep_threshold": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "mode_stability": {
                    "type": "object",
                    "properties": {
                        "cooldown_turns": {"type": "integer", "minimum": 0, "maximum": 20},
                        "significant_cpu_delta": {"type": "number"},
                        "significant_gpu_delta": {"type": "number"},
                    },
                    "additionalProperties": False,
                },
                "observability": {
                    "type": "object",
                    "properties": {
                        "debug_snapshot_interval_s": {"type": "number", "minimum": 0, "maximum": 3600},
                        "debug_snapshot_cache_ttl_s": {"type": "number", "minimum": 0, "maximum": 60},
                    },
                    "additionalProperties": False,
                },
                "prefetch": {
                    "type": "object",
                    "properties": {
                        "max_prefetch_candidates": {"type": "integer", "minimum": 1, "maximum": 64},
                        "hard_abort_gpu_above": {"type": "number"},
                        "soft_scale_gpu_above": {"type": "number"},
                        "soft_scale_factor": {"type": "number"},
                        "soft_delay_s": {"type": "number"},
                        "gpu_soft_extra_delay_s": {"type": "number"},
                        "min_prediction_confidence": {"type": "number"},
                        "low_conf_extra_delay_s": {"type": "number"},
                    },
                    "additionalProperties": False,
                },
                "feedback": {
                    "type": "object",
                    "properties": {
                        "max_records": {"type": "integer"},
                        "learn_rate": {"type": "number"},
                        "rolling_window_50": {"type": "integer"},
                        "rolling_window_100": {"type": "integer"},
                        "trend_flat_epsilon": {"type": "number"},
                        "min_query_chars": {"type": "integer"},
                        "learn_confidence_threshold": {"type": "number"},
                    },
                    "additionalProperties": True,
                },
                "health": {
                    "type": "object",
                    "properties": {
                        "prediction_good_above": {"type": "number"},
                        "prediction_poor_below": {"type": "number"},
                        "prediction_unstable_low": {"type": "number"},
                        "prediction_unstable_high": {"type": "number"},
                        "prefetch_good_above": {"type": "number"},
                        "prefetch_poor_below": {"type": "number"},
                        "memory_relevance_good_above": {"type": "number"},
                        "memory_relevance_poor_below": {"type": "number"},
                        "system_load_cpu_low_below": {"type": "number"},
                        "system_load_cpu_high_above": {"type": "number"},
                        "system_load_ram_high_above": {"type": "number"},
                    },
                    "additionalProperties": False,
                },
                "warnings": {
                    "type": "object",
                    "properties": {
                        "warn_on_degrading_prediction": {"type": "boolean"},
                        "graph_miss_rate_above": {"type": "number"},
                        "prefetch_waste_above": {"type": "number"},
                    },
                    "additionalProperties": False,
                },
                "preemption": {
                    "type": "object",
                    "properties": {
                        "restart_cost": {"type": "number"},
                        "relevance_scale": {"type": "number"},
                        "context_scale": {"type": "number"},
                        "min_improvement_score": {"type": "number"},
                        "max_preemptions_per_query": {"type": "integer", "minimum": 0, "maximum": 10},
                    },
                    "additionalProperties": False,
                },
            },
            "additionalProperties": False,
        },
        "v7_gpu": {
            "type": "object",
            "description": "ATOM V7 GPU resource manager, power modes, and degradation.",
            "properties": {
                "enabled": {"type": "boolean"},
                "strict_control": {
                    "type": "boolean",
                    "description": "Require load grants before GPU model loads (single authority).",
                },
                "deployment_mode": {
                    "type": "string",
                    "enum": ["fused", "distributed"],
                    "description": "fused = single-process voice+LLM; distributed = ZMQ workers.",
                },
                "simulation_mode": {
                    "type": "string",
                    "enum": ["heuristic", "hybrid", "memory_weighted"],
                },
                "vram_reserve_mb": {"type": "integer", "minimum": 0, "maximum": 8192},
                "model_slots_mb": {
                    "type": "object",
                    "additionalProperties": {"type": "integer", "minimum": 0},
                },
                "eviction_order": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "idle_unload_stt_s": {"type": "number", "minimum": 10, "maximum": 3600},
                "idle_unload_llm_s": {"type": "number", "minimum": 30, "maximum": 7200},
                "idle_sleep_s": {"type": "number", "minimum": 60, "maximum": 86400},
                "fused_gpu_worker": {"type": "boolean"},
                "gpu_stall_timeout_s": {"type": "number", "minimum": 10, "maximum": 600},
                "allow_cuda_reset": {"type": "boolean"},
                "degradation_default": {
                    "type": "string",
                    "enum": ["full", "limited", "safe"],
                },
                "event_replay_max": {"type": "integer", "minimum": 8, "maximum": 256},
                "speculative_response": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
        "gpu_execution": {
            "type": "object",
            "description": "Hardware-aware GPU coordinator (soft deferral, NVML).",
            "properties": {
                "vram_reserve_mb": {"type": "number", "minimum": 0},
                "high_gpu_util_defer_background": {"type": "number", "minimum": 0, "maximum": 100},
                "fragmentation_defer_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                "defer_backoff_s": {"type": "number", "minimum": 0.01},
                "max_defer_cycles": {"type": "integer", "minimum": 1},
                "embed_light_max_mb": {"type": "number", "minimum": 0},
                "overlap_max_gpu_util": {"type": "number", "minimum": 0, "maximum": 100},
                "gpu_state_ttl_s": {"type": "number", "minimum": 0.05},
                "feedback_ewma_alpha": {"type": "number", "minimum": 0.01, "maximum": 1},
                "exec_log_max": {"type": "integer", "minimum": 16},
            },
            "additionalProperties": False,
        },
        "rag": {
            "type": "object",
            "description": "GPU-aware RAG: hybrid retrieval, Qdrant optional, bounded wait before LLM.",
            "properties": {
                "enabled": {"type": "boolean"},
                "backend": {"type": "string", "enum": ["chroma", "qdrant"]},
                "first_token_budget_ms": {"type": "number", "minimum": 0, "maximum": 2000},
                "collections": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "top_k": {"type": "integer", "minimum": 1, "maximum": 32},
                "max_snippets": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 16,
                    "description": "Hard cap for RAG snippets merged into prompt context.",
                },
                "vector_weight": {"type": "number", "minimum": 0, "maximum": 1},
                "keyword_weight": {"type": "number", "minimum": 0, "maximum": 1},
                "recency_weight": {"type": "number", "minimum": 0, "maximum": 1},
                "skip_embed_gpu_util_above": {"type": "number", "minimum": 0, "maximum": 100},
                "embed_vram_mb": {"type": "number", "minimum": 0},
                "batch_embed_min": {"type": "integer", "minimum": 1},
                "pressure_threshold_pct": {
                    "type": "number",
                    "minimum": 50,
                    "maximum": 99,
                    "description": "Unified-memory usage percent that forces low-memory RAG mode.",
                },
                "pressure_relief_pct": {
                    "type": "number",
                    "minimum": 40,
                    "maximum": 95,
                    "description": "Unified-memory usage percent below which normal RAG mode resumes.",
                },
                "pressure_top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 8,
                    "description": "Snippet budget while memory pressure mode is active.",
                },
                "fast_mode": {
                    "type": "boolean",
                    "description": "Graph + memory cache only; skip vector embed for minimum latency.",
                },
                "persistent_embed_cache": {"type": "boolean"},
                "embed_cache_path": {"type": "string"},
                "embed_cache_bucket_fallback": {"type": "boolean"},
                "prefetch_enabled": {"type": "boolean"},
                "late_restart_confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Late RAG preempt+restart when confidence exceeds this.",
                },
                "graph_first": {
                    "type": "object",
                    "description": "Prefer MemoryGraph when confidence is high; skip vector RAG.",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "min_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "min_snippets": {"type": "integer", "minimum": 1, "maximum": 20},
                        "project_boost": {"type": "number", "minimum": 0, "maximum": 0.5},
                        "relevance_validation_min": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "additionalProperties": False,
                },
                "smart_scoring": {
                    "type": "object",
                    "description": "Heuristics for recency, owner-priority, retrieval-frequency, and stale-result handling.",
                    "properties": {
                        "recency_half_life_hours": {"type": "number", "minimum": 1},
                        "owner_priority_multiplier": {"type": "number", "minimum": 1, "maximum": 4},
                        "owner_priority_sources": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "usage_boost_max": {"type": "number", "minimum": 0, "maximum": 1},
                        "usage_history_size": {"type": "integer", "minimum": 64, "maximum": 32768},
                        "stale_after_hours": {"type": "number", "minimum": 1},
                        "time_sensitive_stale_after_hours": {"type": "number", "minimum": 1},
                        "stale_penalty": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "additionalProperties": False,
                },
                "adaptive": {
                    "type": "object",
                    "properties": {
                        "budget_min_ms": {"type": "number"},
                        "budget_max_ms": {"type": "number"},
                    },
                    "additionalProperties": False,
                },
                "qdrant_path": {"type": "string"},
                "qdrant_collection": {"type": "string"},
                "cache": {
                    "type": "object",
                    "properties": {
                        "embed_ttl_s": {"type": "number"},
                        "retrieval_ttl_s": {"type": "number"},
                        "max_entries": {"type": "integer"},
                    },
                    "additionalProperties": False,
                },
            },
            "additionalProperties": False,
        },
        # Sprint P4 (Apr 26 2026): owner personality + style learning.
        # Fully optional; ATOM works without this block. Used by
        # core/personality/* and the structured prompt builder.
        "personality": {
            "type": "object",
            "properties": {
                "persona_file": {
                    "type": "string",
                    "description": (
                        "Path to the Boss-authored persona markdown file "
                        "pinned into the LLM's KV prefix."
                    ),
                },
                "owner_profile_db": {
                    "type": "string",
                    "description": (
                        "SQLite path for OwnerProfile (corrections + "
                        "pronunciation dictionary). Defaults to "
                        "data/owner_profile.sqlite3."
                    ),
                },
                "owner_style": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "window_size": {
                            "type": "integer", "minimum": 8, "maximum": 2048,
                        },
                        "min_observations": {
                            "type": "integer", "minimum": 1, "maximum": 256,
                        },
                    },
                    "additionalProperties": False,
                },
            },
            "additionalProperties": True,
        },
    },
    "additionalProperties": True,
}

# Audio pipeline constants validated at runtime
AUDIO_CONSTRAINTS = {
    "noise_floor_init": (10.0, 500.0),
    "noise_floor_alpha": (0.01, 0.5),
    "speech_gate_mult": (1.0, 5.0),
    "gain_cap": (1.0, 20.0),
    "rms_smooth_alpha": (0.05, 0.5),
}


def validate_config(config: dict) -> list[str]:
    """Validate the config dict against the schema.

    Returns a list of error messages (empty = valid).
    Falls back to basic type checks if jsonschema is not installed.
    """
    errors: list[str] = []
    if not isinstance(config, dict):
        return ["(root): config must be an object"]
    if not config:
        return ["(root): config is empty; config/settings.json was not loaded"]

    try:
        import jsonschema
        validator = jsonschema.Draft7Validator(CONFIG_SCHEMA)
        for err in sorted(validator.iter_errors(config),
                          key=lambda e: list(e.absolute_path)):
            path = ".".join(str(p) for p in err.absolute_path)
            errors.append(f"  {path or '(root)'}: {err.message}")
        return errors
    except ImportError:
        pass

    errors.extend(_basic_validation(config))
    return errors


def _basic_validation(config: dict) -> list[str]:
    """Minimal validation without jsonschema dependency."""
    errors: list[str] = []

    stt = config.get("stt", {})
    if not isinstance(stt, dict):
        errors.append("stt: must be an object")
    else:
        engine = stt.get("engine")
        if engine is not None and engine not in (
            "auto",
            "macos_native",
            "whisper_cpp",
            "whispercpp",
            "whisper",
            "whisper.cpp",
            "faster_whisper",
            "google_online",
            "google",
        ):
            errors.append(
                "stt.engine: must be auto/macos_native/whisper_cpp/"
                "faster_whisper/google_online/google, "
                f"got {engine}",
            )

        chunk = stt.get("chunk_size")
        if chunk is not None and (not isinstance(chunk, int)
                                  or chunk < 256 or chunk > 16384):
            errors.append(f"stt.chunk_size: must be int 256-16384, got {chunk}")

        rate = stt.get("sample_rate")
        if rate is not None and (not isinstance(rate, int)
                                 or rate < 8000 or rate > 48000):
            errors.append(f"stt.sample_rate: must be int 8000-48000, got {rate}")

        cooldown = stt.get("post_tts_cooldown_ms")
        if cooldown is not None and (not isinstance(cooldown, (int, float))
                                     or cooldown < 0):
            errors.append(f"stt.post_tts_cooldown_ms: must be >= 0, "
                          f"got {cooldown}")

    cache = config.get("cache", {})
    if isinstance(cache, dict):
        max_size = cache.get("max_size")
        if max_size is not None and (not isinstance(max_size, int)
                                     or max_size < 1):
            errors.append(f"cache.max_size: must be int >= 1, got {max_size}")

    perf = config.get("performance", {})
    if isinstance(perf, dict):
        mode = perf.get("mode")
        if mode is not None and mode not in (
            "optimal",
            "full_performance",
            "auto",
            "full",
            "lite",
            "ultra_lite",
        ):
            errors.append(
                "performance.mode: must be optimal|full_performance|auto "
                f"(legacy: full|lite|ultra_lite), got {mode}"
            )

    auto = config.get("autonomy", {})
    if isinstance(auto, dict):
        at = auto.get("auto_execute_threshold")
        if at is not None and (not isinstance(at, (int, float)) or at < 0.5 or at > 1.0):
            errors.append(f"autonomy.auto_execute_threshold: must be 0.5-1.0, got {at}")
        st = auto.get("suggest_threshold")
        if st is not None and (not isinstance(st, (int, float)) or st < 0.2 or st > 1.0):
            errors.append(f"autonomy.suggest_threshold: must be 0.2-1.0, got {st}")

    return errors


def _check_embedded_secrets(config: dict) -> list[str]:
    """Reserved for future secret checks (ATOM offline build uses no cloud API keys)."""
    return []


def validate_and_log(config: dict) -> bool:
    """Validate config and log any errors. Returns True if valid."""
    errors = validate_config(config)
    secret_warnings = _check_embedded_secrets(config)
    for w in secret_warnings:
        logger.warning(w)
    if not errors:
        logger.info("Configuration validated successfully")
        return True
    logger.warning("Configuration validation errors:")
    for err in errors:
        logger.warning("    %s", err)
    logger.error("Configuration validation failed; refusing to boot.")
    return False
