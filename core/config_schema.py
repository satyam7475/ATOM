"""
ATOM v14 -- Configuration schema validation.

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
                },
                "prefer_bluetooth": {
                    "type": "boolean",
                    "description": "If false, use system default mic only (avoids noisy BT HFP).",
                },
            },
            "additionalProperties": False,
        },
        "stt": {
            "type": "object",
            "properties": {
                "engine": {
                    "type": "string",
                    "enum": ["vosk", "google", "vosk_with_fallback"],
                    "description": "STT engine: vosk (offline), google (cloud), vosk_with_fallback",
                },
                "vosk_model_path": {
                    "type": "string",
                    "description": "Path to Vosk model directory (relative to ATOM root)",
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
            },
            "additionalProperties": False,
        },
        "tts": {
            "type": "object",
            "properties": {
                "engine": {
                    "type": "string",
                    "enum": ["sapi", "edge"],
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
        "memory": {
            "type": "object",
            "properties": {
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
                    "description": "Path to GGUF model file.",
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
                    "minimum": 0,
                    "maximum": 100,
                    "description": "GPU layers to offload (0 = CPU only).",
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
                "timeout_seconds": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 300,
                    "description": "Timeout for local LLM inference.",
                },
            },
            "additionalProperties": False,
        },
        "ui": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["web", "tkinter"],
                },
                "web_port": {
                    "type": "integer",
                    "minimum": 1024,
                    "maximum": 65535,
                },
                "auto_open_browser": {"type": "boolean"},
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
        "owner": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "title": {"type": "string"},
            },
            "additionalProperties": True,
        },
        "vision": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "camera_index": {"type": "integer", "minimum": 0},
                "check_interval_seconds": {"type": "number", "minimum": 1, "maximum": 60},
                "require_owner_for_sensitive": {"type": "boolean"},
                "owner_encoding_path": {"type": "string"},
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
            },
            "additionalProperties": False,
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
                    "enum": ["off", "safe_only", "owner_only"],
                    "description": "off = normal; safe_only = only safe intents; owner_only = placeholder for auth.",
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
            "description": "Local LLM profiles (atom/balanced/brain) and static quick replies.",
            "additionalProperties": True,
        },
        "performance": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["full", "lite", "ultra_lite", "auto"],
                    "description": "full|lite|ultra_lite = fixed; auto = CPU-based (thresholds).",
                },
                "auto_threshold_high": {
                    "type": "integer",
                    "minimum": 50,
                    "maximum": 95,
                    "description": "CPU percent above which auto mode switches to ultra_lite.",
                },
                "auto_threshold_mid": {
                    "type": "integer",
                    "minimum": 20,
                    "maximum": 70,
                    "description": "CPU percent above which auto mode switches to lite (below = full).",
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
                "use_priority_scheduler": {
                    "type": "boolean",
                    "description": "Route speech_final and cursor_query through PriorityScheduler (voice > LLM > background).",
                },
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
                        "workstation",
                        "unset",
                    ],
                },
                "dashboard_badge": {
                    "type": "boolean",
                    "description": "Show profile pill on web dashboard top bar.",
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
            },
            "additionalProperties": False,
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
        if engine is not None and engine not in ("vosk", "google", "vosk_with_fallback"):
            errors.append(f"stt.engine: must be vosk|google|vosk_with_fallback, got {engine}")

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
        if mode is not None and mode not in ("full", "lite", "ultra_lite", "auto"):
            errors.append(f"performance.mode: must be full|lite|ultra_lite|auto, got {mode}")

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
        logger.warning("  %s", err)
    return False
