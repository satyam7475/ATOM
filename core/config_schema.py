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
                    "enum": ["faster_whisper"],
                    "description": "STT engine: faster_whisper (offline, GPU-accelerated)",
                },
                "whisper_model_size": {
                    "type": "string",
                    "enum": ["tiny", "base", "small", "medium", "large-v3"],
                    "description": "Whisper model size (recommended: small for bilingual)",
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
            },
            "additionalProperties": False,
        },
        "tts": {
            "type": "object",
            "properties": {
                "engine": {
                    "type": "string",
                    "enum": ["sapi", "edge", "kokoro"],
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
                "graph_db_path": {
                    "type": "string",
                    "description": "SQLite path for MemoryGraph (V7 timeline + RAG graph hints).",
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
                    "description": "Path to GGUF model file.",
                },
                "mlx_primary_model": {
                    "type": "string",
                    "description": "Path to the primary MLX model directory.",
                },
                "mlx_fast_model": {
                    "type": "string",
                    "description": "Path to the fast MLX model directory.",
                },
                "mlx_default_role": {
                    "type": "string",
                    "enum": ["primary", "fast"],
                    "description": "Default MLX role to load first.",
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
                "password_auth_enabled": {
                    "type": "boolean",
                    "description": "If false, dashboard token gate is disabled (no password-style token auth).",
                },
                "dashboard_access_token": {
                    "type": "string",
                    "description": "If set, WebSocket /ws requires ?token= or X-ATOM-Token header.",
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
                "watchdog_intent_timeout_ms": {
                    "type": "number",
                    "minimum": 10,
                    "maximum": 500,
                    "description": "RuntimeWatchdog: Intent engine budget before forced fallback.",
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
                    "description": "RuntimeWatchdog: TTS synthesis timeout before audio skip.",
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
        "cognitive_loop": {
            "type": "object",
            "description": "Jarvis-style observe/predict/decide/act interval.",
            "properties": {
                "interval_s": {"type": "number", "minimum": 5, "maximum": 3600},
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
        if engine is not None and engine != "faster_whisper":
            errors.append(f"stt.engine: must be faster_whisper, got {engine}")

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
        logger.warning("    %s", err)
    logger.warning("Continuing despite validation warnings (non-fatal).")
    return True
