"""ATOM -- WhisperKit STT backend (Sprint P3.3, Apr 26 2026).

Drop-in replacement for ``voice.stt_whisper.WhisperSTT`` that runs Apple
Silicon's **CoreML-on-ANE** Whisper via Argmax's WhisperKit instead of
the whisper.cpp Metal path. This is the single highest-ROI move in the
P3 plan: it cuts the ~14-s Metal-library init down to ~1 s and pushes
STT off the GPU queue so the LLM gets it.

Architecture::

    Microphone (sounddevice 16-kHz mono PCM)
        v
    WebRTC VAD (30-ms frames, energy gate)
        v
    Rolling 30-s ring buffer
        v
    HTTP POST multipart/form-data ->
       http://127.0.0.1:50060/v1/audio/transcriptions
        |
        +-- whisperkit-cli serve (CoreML / ANE), OpenAI-compatible API
        v
    AsyncEventBus -- speech_partial / speech_final / voice.partial /
                     voice.final  (same shape as WhisperSTT/NativeSTT)

Public surface mirrors :class:`voice.stt_whisper.WhisperSTT` (and through
it :class:`voice.stt_macos.NativeSTT`) so the voice_pipeline factory can
swap engines without touching callers.

Setup
-----

    # one-time, on Apple Silicon Macs
    brew install whisperkit-cli
    # Model auto-downloads from HuggingFace argmaxinc/whisperkit-coreml on
    # first ``serve`` call. The CLI prepends ``openai_`` to the model name
    # via ``--model-prefix openai`` (default), so pass model names WITHOUT
    # that prefix -- e.g. ``whisper-large-v3-v20240930_turbo_632MB``
    # resolves to ``openai_whisper-large-v3-v20240930_turbo_632MB`` in the
    # repo.

The factory in ``voice/voice_pipeline.py`` falls back to whisper.cpp
when ``whisperkit-cli`` is not on $PATH.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

logger = logging.getLogger("atom.stt_whisperkit")

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.state_manager import StateManager

# ── Optional native deps ──────────────────────────────────────────
try:
    import sounddevice as _sd  # type: ignore[import-untyped]
except Exception:  # pragma: no cover -- environment-specific
    _sd = None  # type: ignore[assignment]

try:
    import webrtcvad as _webrtcvad  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    _webrtcvad = None  # type: ignore[assignment]

try:
    import numpy as _np  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    _np = None  # type: ignore[assignment]

# Sprint P2.1 (Apr 26 2026): unify auto-correction with the rest of the
# STT family.
try:
    from voice.speech_detector import correct_text as _correct_text
    from voice.speech_detector import is_noise_word as _is_noise_word
except Exception:  # pragma: no cover
    _correct_text = None  # type: ignore[assignment]
    _is_noise_word = None  # type: ignore[assignment]

# Sprint P4.1+P4.3 (Apr 26 2026): owner-aware learning hooks.
try:
    from core.personality import get_owner_profile as _get_owner_profile
    from core.personality import get_owner_style as _get_owner_style
except Exception:  # pragma: no cover - defensive
    _get_owner_profile = None  # type: ignore[assignment]
    _get_owner_style = None  # type: ignore[assignment]


# ── Constants ─────────────────────────────────────────────────────

_SAMPLE_RATE = 16000
_FRAME_MS = 30
_FRAME_SAMPLES = (_SAMPLE_RATE * _FRAME_MS) // 1000  # 480
_RING_SECONDS = 30.0
_PARTIAL_INTERVAL_S = 0.4
_TRAILING_SILENCE_S = 0.5
_MAX_UTTERANCE_S = 20.0
_MIN_UTTERANCE_MS = 250

# Default WhisperKit model -- multilingual large-v3-turbo, quantized to
# ~632 MB for ANE. The CLI auto-prepends ``openai_`` (see module
# docstring), so this string is the suffix only. Resolves to
# ``openai_whisper-large-v3-v20240930_turbo_632MB`` in
# ``argmaxinc/whisperkit-coreml`` on HuggingFace.
_DEFAULT_WHISPERKIT_MODEL = "whisper-large-v3-v20240930_turbo_632MB"
_DEFAULT_SERVE_PORT = 50060
_DEFAULT_SERVE_HOST = "127.0.0.1"

_ATOM_DIAGNOSTIC_AUDIO_RE = re.compile(
    r"\b("
    r"rag|rack|memory|embedding|snippet|boot|diagnostic|watchdog|"
    r"pressure|pipeline|governor"
    r")\b.*\b("
    r"pressure mode|snippet budget|engine shut ?down|boot diagnostic|"
    r"budget reduced|memory pressure|watchdog|pipeline"
    r")\b",
    re.I,
)
_ATOM_SELF_SPEECH_RE = re.compile(
    r"^\s*(?:"
    r"what\s+do\s+you\s+need|"
    r"one\s+moment|"
    r"working\s+on\s+it|"
    r"right\s+away|"
    r"on\s+it(?:\s+boss)?|"
    r"let\s+me\s+check|"
    r"give\s+me\s+a\s+sec|"
    r"i'?m\s+(?:good|here|it)\s*,?\s*boss(?:\s+ready\s+for\s+you)?|"
    r"system\s+is\s+degraded\s*,?\s+boss.*|"
    r"atom\.?\s*local\s*brain|"
    r"atom\.?\s*localbrain|"
    r"\[?\s*system\s*\]?\s+initiating\s+system\s+diagnostics|"
    r"here\s+boss.*active\s+goals.*what\s+do\s+you\s+need"
    r")[\s.?!]*$",
    re.I,
)
_ATOM_WAKE_MISHEAR_RE = re.compile(
    r"\b(?:adam|adom|adtan|adton|atan|atum|autumn|atam|attom|adum)\b",
    re.I,
)
_ATOM_TRAILING_ACK_RE = re.compile(
    r"\s+(?:okay|ok|mm-?|mm\s*hmm|uh\s*huh)[\s.?!]*$",
    re.I,
)


def _looks_like_atom_self_speech(text: str) -> bool:
    text = (text or "").strip()
    return bool(
        text
        and (
            _ATOM_SELF_SPEECH_RE.search(text)
            or _ATOM_DIAGNOSTIC_AUDIO_RE.search(text)
        )
    )


def _normalize_atom_final_text(text: str) -> str:
    """Remove ATOM speaker bleed while preserving a real owner suffix."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return ""

    text = _ATOM_WAKE_MISHEAR_RE.sub("atom", text)
    if _looks_like_atom_self_speech(text):
        return ""

    parts = [
        p.strip()
        for p in re.split(r"(?<=[.!?])\s+", text)
        if p.strip()
    ]
    if len(parts) > 1:
        kept = [p for p in parts if not _looks_like_atom_self_speech(p)]
        if kept != parts:
            text = " ".join(kept).strip()

    if len(text.split()) > 3:
        text = _ATOM_TRAILING_ACK_RE.sub("", text).strip()
    if _looks_like_atom_self_speech(text):
        return ""
    return text


# ── Sprint Ω.8: STT garbage filter ───────────────────────────────────
# Whisper / WhisperKit are notorious for hallucinating filler text
# during silence, music, or breath noise. atomCurrentLogs.txt L371,
# L467, L509 show the exact failure modes we now reject:
#   - "Продолжение следует..." (Russian "to be continued" — common
#     youtube-trained hallucination)
#   - "*cough*", "*spray the door*", "[Music]" (literal stage-direction
#     markup leaking through as "speech")
#   - "Thank you for watching" / "Subscribe to my channel" (youtube
#     ad/end-card hallucinations)
#   - Doubled wake words: "hey atom hey atom hey atom"
# Anything that's mostly-asterisks or mostly-bracketed is rejected
# unconditionally; any pure non-Latin transcription is rejected when
# the configured STT language is English (the ATOM default).

_HALLUCINATION_PHRASES: tuple[re.Pattern, ...] = (
    re.compile(r"продолжен", re.I),
    re.compile(r"^thank you for watching\b", re.I),
    re.compile(r"^subscribe (?:to (?:my|the))? channel\b", re.I),
    re.compile(r"^like and subscribe\b", re.I),
    re.compile(r"^thanks for watching\b", re.I),
    re.compile(r"\bMBE\.com\b", re.I),
    re.compile(r"\bw{3}\.\S+\.com\b", re.I),
)

_STAGE_DIRECTION_RE = re.compile(r"^[\*\[\(<].+?[\*\]\)>]\s*$")
_ALL_PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)
_NON_LATIN_RE = re.compile(r"[^\x00-\x7F]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_DOUBLED_WAKE_RE = re.compile(
    r"^(?:hey\s+|hi\s+)?(?:atom|adam|atim|item)"
    r"(?:[\s,.!?]+(?:hey\s+|hi\s+)?(?:atom|adam|atim|item)){2,}"
    r"[\s,.!?]*$",
    re.I,
)


def _is_stt_garbage(text: str, *, language: str = "en") -> str | None:
    """Return a reason string if ``text`` is a known STT garbage
    pattern, else None. Called from :meth:`_emit_final` before any
    routing so hallucinations never reach the brain.

    Sprint Ω.8 (Apr 26 2026): centralised garbage filter.
    """
    if not text:
        return None
    stripped = text.strip()
    if not stripped:
        return "empty"

    # Pure stage direction: "*cough*", "[Music]", "(applause)"
    if _STAGE_DIRECTION_RE.match(stripped):
        return f"stage_direction:{stripped[:40]}"

    # Almost no real letters — punctuation / asterisks / brackets only
    if _ALL_PUNCT_RE.match(stripped):
        return "punct_only"

    # Doubled wake words: "hey atom hey atom hey atom"
    if _DOUBLED_WAKE_RE.match(stripped):
        return f"doubled_wake:{stripped[:40]}"

    # Whisper hallucination corpus
    for pat in _HALLUCINATION_PHRASES:
        if pat.search(stripped):
            return f"hallucination:{pat.pattern[:30]}"

    # Pure non-Latin transcription on an English-configured pipeline.
    # A Hindi / Hinglish phrase typed in Devanagari is rare and almost
    # always a Whisper noise hallucination — Boss types Hinglish in
    # Latin script. We allow short non-Latin tokens mixed with Latin
    # words; pure-CJK / pure-Cyrillic finals are rejected.
    if (language or "en").lower().startswith("en"):
        non_latin = _NON_LATIN_RE.findall(stripped)
        latin_words = _LATIN_WORD_RE.findall(stripped)
        if non_latin and not latin_words:
            return f"non_latin_only:{stripped[:40]}"

    return None


def _whisperkit_cli_path() -> str | None:
    """Return the path to ``whisperkit-cli`` if installed, else ``None``."""
    cli = shutil.which("whisperkit-cli")
    if cli:
        return cli
    # Common Homebrew install locations (Apple Silicon + Intel).
    for candidate in (
        "/opt/homebrew/bin/whisperkit-cli",
        "/usr/local/bin/whisperkit-cli",
    ):
        if Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def is_whisperkit_available(_config: dict | None = None) -> bool:
    """Best-effort check used by the factory.

    Returns True when:
    - The host is Apple Silicon (or Intel Mac with WhisperKit installed,
      though Argmax docs say Intel Macs are unsupported).
    - ``whisperkit-cli`` is on $PATH.
    - ``sounddevice`` + ``webrtcvad`` + ``numpy`` are importable.

    The model itself is NOT required at probe time -- the first ``preload``
    call will trigger ``whisperkit-cli serve`` which lazy-downloads from
    HuggingFace ``argmaxinc/whisperkit-coreml`` on first run.
    """
    if _whisperkit_cli_path() is None:
        return False
    if _sd is None or _webrtcvad is None or _np is None:
        return False
    return True


def _port_is_open(host: str, port: int, *, timeout_s: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


# ── Backend ───────────────────────────────────────────────────────


class WhisperKitSTT:
    """WhisperKit (CoreML / ANE) STT backend.

    Mirrors :class:`voice.stt_whisper.WhisperSTT` exactly so the voice
    pipeline factory can swap engines without touching callers. The
    primary differences are:

    1. Transcription happens via a long-running ``whisperkit-cli serve``
       HTTP server we spawn as a subprocess, rather than an in-process
       whisper.cpp Model.
    2. CoreML / ANE replaces Metal / GPU.

    The HTTP server is owned by this class. It is started in
    :meth:`preload` and shut down in :meth:`shutdown`. The server stays
    warm across utterances so per-call latency is dominated by the ANE
    decode (~80 ms for 4 s of audio at large-v3-turbo on M5).
    """

    def __init__(
        self,
        bus: "AsyncEventBus",
        state: "StateManager",
        config: dict | None = None,
        mic_manager: Any = None,
        intent_engine: Any = None,
    ) -> None:
        self._bus = bus
        self._state = state
        full_cfg = config or {}
        self._config = full_cfg.get("stt", {})
        wk_cfg = self._config.get("whisperkit", {}) or {}
        self._mic_manager = mic_manager
        self._intent_engine = intent_engine

        self._cli_path: str | None = _whisperkit_cli_path()
        self._model: str = str(
            wk_cfg.get("model", _DEFAULT_WHISPERKIT_MODEL),
        )
        self._serve_host: str = str(wk_cfg.get("host", _DEFAULT_SERVE_HOST))
        self._serve_port: int = int(wk_cfg.get("port", _DEFAULT_SERVE_PORT))
        # Where ``whisperkit-cli serve`` lazy-downloads model bundles.
        # The CLI respects $WHISPERKIT_HOME, so we just inherit the env
        # default unless the user has overridden it via ``model_dir``.
        self._model_dir: str | None = wk_cfg.get("model_dir")
        # Boot-time download is async-friendly: the first `preload` call
        # will block on the download. We let the user gate this.
        self._auto_download: bool = bool(wk_cfg.get("auto_download", True))
        # How long to wait for the serve subprocess to come up.
        self._serve_startup_timeout_s: float = float(
            wk_cfg.get("startup_timeout_s", 30.0),
        )

        self._n_threads: int = int(self._config.get("whisper_n_threads", 4))
        self._language: str = str(self._config.get("whisper_language", "auto"))
        self._partial_interval_s: float = float(
            self._config.get("whisper_partial_interval_s", _PARTIAL_INTERVAL_S),
        )
        self._trailing_silence_s: float = float(
            self._config.get("whisper_trailing_silence_s", _TRAILING_SILENCE_S),
        )
        self._max_utterance_s: float = float(
            self._config.get("whisper_max_utterance_s", _MAX_UTTERANCE_S),
        )
        self._vad_aggressiveness: int = max(
            0, min(3, int(self._config.get("whisper_vad_aggressiveness", 2))),
        )
        self._noise_floor_dbfs: float = float(
            self._config.get("noise_floor_dbfs", -55.0),
        )
        self._noise_gate_consecutive: int = max(
            1, int(self._config.get("noise_gate_consecutive", 5)),
        )
        self._noise_gate_below_count: int = 0
        self._noise_gate_dropped_total: int = 0
        self._speech_candidate_floor_dbfs: float = float(
            self._config.get(
                "speech_candidate_floor_dbfs",
                max(-42.0, self._noise_floor_dbfs + 3.0),
            ),
        )
        self._min_utterance_ms: int = max(
            _MIN_UTTERANCE_MS,
            int(float(self._config.get("min_audio_duration_s", 0.42)) * 1000),
        )
        self.mic_name: str = "sounddevice (PortAudio/CoreAudio)"

        # State
        self._serve_proc: subprocess.Popen | None = None
        self._stream: Any = None
        self._available: bool = False
        self._listening: bool = False
        self._running_async: bool = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._on_final: Callable[[str], None] | None = None
        self._on_partial: Callable[[str], None] | None = None
        self._last_partial: str = ""
        self._last_final: str = ""
        self._last_confidence: float = 0.97
        self._last_error: str | None = None
        self._last_speech_time: float = 0.0
        self._tap_buffer_count: int = 0
        self._last_audio_rms_db: float = -96.0
        self._last_speech_candidate_time: float = 0.0
        self._permanently_disabled: bool = False
        self._http_session: Any = None  # urllib3 / requests if available
        self._output_muted_until: float = 0.0

        # Audio + VAD ring
        self._ring_lock = threading.Lock()
        self._ring: collections.deque[bytes] = collections.deque(
            maxlen=int(_RING_SECONDS * _SAMPLE_RATE * 2 / _FRAME_SAMPLES),
        )
        self._vad: Any = None
        self._utterance_frames: list[bytes] = []
        self._silence_frames: int = 0
        self._utterance_started_at: float = 0.0
        self._last_partial_emit_at: float = 0.0
        self._worker_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

        # Echo guard + WhisperConfirmer (parity with WhisperSTT)
        self._echo_guard: Callable[[str], bool] | None = None
        self._whisper_confirmer: Any = None

    def _clear_utterance_buffers(self) -> None:
        self._utterance_frames.clear()
        self._silence_frames = 0
        self._utterance_started_at = 0.0
        self._last_partial_emit_at = 0.0
        self._last_partial = ""

    def _is_output_muted(self) -> bool:
        try:
            from core.state_manager import AtomState
            if self._state.current is AtomState.SPEAKING:
                return True
        except Exception:
            pass
        return time.monotonic() < self._output_muted_until

    # ── public properties ───────────────────────────────────────

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def is_listening(self) -> bool:
        return self._listening

    @property
    def backend_name(self) -> str:
        if self._available:
            return f"WhisperKit (CoreML/ANE, {self._model})"
        return "WhisperKit (unavailable)"

    # ── lifecycle ───────────────────────────────────────────────

    def preload(self) -> bool:
        """Launch ``whisperkit-cli serve`` and verify the model loads.

        Idempotent. Returns True if the serve subprocess is up and the
        VAD + sounddevice deps are present.
        """
        if self._available:
            return True
        if self._cli_path is None:
            self._last_error = (
                "whisperkit-cli not on $PATH; "
                "run `brew install whisperkit-cli`"
            )
            logger.warning("WhisperKitSTT: %s", self._last_error)
            return False
        if _sd is None or _webrtcvad is None or _np is None:
            missing = [
                name for name, mod in (
                    ("sounddevice", _sd),
                    ("webrtcvad", _webrtcvad),
                    ("numpy", _np),
                ) if mod is None
            ]
            self._last_error = (
                f"WhisperKitSTT missing native deps: {missing}; "
                "`pip install sounddevice webrtcvad numpy`"
            )
            logger.warning(self._last_error)
            return False

        try:
            t0 = time.perf_counter()
            self._maybe_start_serve()
            self._wait_for_serve_ready()
            self._vad = _webrtcvad.Vad(self._vad_aggressiveness)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._available = True
            self._last_error = None
            logger.info(
                "WhisperKitSTT preloaded (model=%s, serve=%s:%d, vad=%d) "
                "in %.0f ms",
                self._model,
                self._serve_host,
                self._serve_port,
                self._vad_aggressiveness,
                elapsed_ms,
            )
            return True
        except Exception as exc:
            self._available = False
            self._last_error = f"WhisperKitSTT preload failed: {exc}"
            logger.warning(
                "WhisperKitSTT: %s", self._last_error, exc_info=True,
            )
            return False

    async def async_preload(self) -> bool:
        loop = asyncio.get_running_loop()
        return bool(await loop.run_in_executor(None, self.preload))

    # ── serve lifecycle ─────────────────────────────────────────

    def _maybe_start_serve(self) -> None:
        """Launch ``whisperkit-cli serve`` if not already running."""
        if _port_is_open(self._serve_host, self._serve_port):
            if self._serve_health_ok(timeout_s=0.5):
                logger.info(
                    "WhisperKit: healthy serve already up on %s:%d -- attaching",
                    self._serve_host, self._serve_port,
                )
                return
            logger.warning(
                "WhisperKit: port %s:%d is bound but /health is unhealthy; "
                "reaping stale whisperkit-cli before launch",
                self._serve_host, self._serve_port,
            )
            if not self._reap_stale_serve_on_port():
                raise RuntimeError(
                    f"WhisperKit port {self._serve_host}:{self._serve_port} "
                    "is bound by an unhealthy non-owned process"
                )
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                if not _port_is_open(self._serve_host, self._serve_port):
                    break
                time.sleep(0.1)
            if _port_is_open(self._serve_host, self._serve_port):
                raise RuntimeError(
                    f"WhisperKit stale serve on {self._serve_host}:"
                    f"{self._serve_port} did not release its port"
                )
        if self._cli_path is None:
            raise RuntimeError("whisperkit-cli not available")

        cmd = [
            self._cli_path,
            "serve",
            "--host", self._serve_host,
            "--port", str(self._serve_port),
            "--model", self._model,
        ]
        # Sprint Ω.6.C (Apr 26 2026): whisperkit-cli 0.18.0 dropped the
        # bare ``--download`` flag (Argmax now treats ``--model NAME`` as
        # implicit "download if not cached, else use cache"). Passing
        # ``--download`` makes Swift ArgumentParser exit immediately with
        # ``Error: Unknown option '--download'`` and the serve subprocess
        # dies in <100 ms, leaving ATOM to wait the full
        # ``startup_timeout_s`` before raising. ``self._auto_download``
        # is now a no-op kept for back-compat with older config files.
        if self._model_dir:
            # ``--model-prefix`` only accepts the variant tag
            # ("openai" | "distil"); a custom directory belongs in
            # ``--download-model-path`` so the cache is rooted there.
            cmd.extend(["--download-model-path", self._model_dir])

        logger.info("WhisperKit: launching `%s`", " ".join(cmd))
        try:
            self._serve_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"whisperkit-cli not executable at {self._cli_path}",
            ) from exc

    def _serve_health_ok(self, *, timeout_s: float = 1.0) -> bool:
        import urllib.request

        health_url = f"http://{self._serve_host}:{self._serve_port}/health"
        try:
            with urllib.request.urlopen(health_url, timeout=timeout_s) as resp:
                return bool(resp.status == 200)
        except Exception:
            return False

    def _reap_stale_serve_on_port(self) -> bool:
        """Terminate stale whisperkit-cli listeners that block our port."""
        try:
            proc = subprocess.run(
                [
                    "lsof",
                    "-ti",
                    f"TCP:{self._serve_port}",
                    "-sTCP:LISTEN",
                ],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
        except Exception:
            logger.debug("WhisperKit: lsof port lookup failed", exc_info=True)
            return False

        pids = [p.strip() for p in proc.stdout.splitlines() if p.strip()]
        if not pids:
            return False

        reaped = False
        for pid_s in pids:
            try:
                pid = int(pid_s)
            except ValueError:
                continue
            try:
                ps = subprocess.run(
                    ["ps", "-p", str(pid), "-o", "command="],
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                    check=False,
                )
                command = ps.stdout.strip()
            except Exception:
                logger.debug("WhisperKit: ps lookup failed for pid=%s", pid)
                continue
            if "whisperkit-cli" not in command:
                logger.warning(
                    "WhisperKit: refusing to kill non-WhisperKit listener pid=%s cmd=%r",
                    pid, command[:160],
                )
                continue
            try:
                os.kill(pid, signal.SIGTERM)
                reaped = True
                logger.warning("WhisperKit: terminated stale serve pid=%s", pid)
            except ProcessLookupError:
                reaped = True
            except Exception:
                logger.debug("WhisperKit: SIGTERM failed for pid=%s", pid, exc_info=True)
        return reaped

    def _wait_for_serve_ready(self) -> None:
        # Sprint Ω.6.A (Apr 26 2026): two-stage readiness check.
        #   1. Port-bound  -> Vapor accepted the listen()
        #   2. /health 200 -> the model is loaded AND the route table is
        #      registered. With large turbo + first-load ANE compile,
        #      stage 1 can fire several seconds before stage 2; hitting
        #      transcribe between the two stages returns 404. The health
        #      check closes that race.
        # Sprint Ω.6.C (Apr 26 2026): early-death short-circuit. If the
        # serve subprocess dies before binding (e.g. bad CLI flag, model
        # download failure, port conflict) we capture its exit + last
        # bytes of stdout/stderr instead of waiting the full
        # ``startup_timeout_s``. Cuts the failure-mode latency from 60 s
        # to <1 s and surfaces the actual error.
        import urllib.request

        deadline = time.monotonic() + self._serve_startup_timeout_s
        port_ok = False
        health_url = f"http://{self._serve_host}:{self._serve_port}/health"
        while time.monotonic() < deadline:
            self._raise_if_serve_died()
            if not port_ok:
                if _port_is_open(self._serve_host, self._serve_port):
                    port_ok = True
                else:
                    time.sleep(0.25)
                    continue
            try:
                with urllib.request.urlopen(health_url, timeout=1.0) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                pass
            time.sleep(0.25)
        raise TimeoutError(
            f"whisperkit-cli serve did not become healthy at "
            f"{self._serve_host}:{self._serve_port}/health within "
            f"{self._serve_startup_timeout_s:.1f}s "
            f"(port_bound={port_ok})",
        )

    def _raise_if_serve_died(self) -> None:
        proc = self._serve_proc
        if proc is None:
            return
        rc = proc.poll()
        if rc is None:
            return
        # Subprocess exited before we observed a healthy port. Drain
        # whatever it printed before dying so the error is actionable.
        tail = ""
        if proc.stdout is not None:
            try:
                raw = proc.stdout.read() or b""
            except Exception:
                raw = b""
            tail = raw.decode("utf-8", errors="replace").strip()
            # Cap so we don't spam the log with megabyte tracebacks.
            if len(tail) > 800:
                tail = tail[:400] + " …<truncated>… " + tail[-400:]
        msg = (
            f"whisperkit-cli serve exited early "
            f"(returncode={rc}, host={self._serve_host}:"
            f"{self._serve_port})"
        )
        if tail:
            msg = f"{msg}: {tail}"
        raise RuntimeError(msg)

    def _stop_serve(self) -> None:
        proc = self._serve_proc
        self._serve_proc = None
        if proc is None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            logger.debug("whisperkit-cli serve teardown raised", exc_info=True)

    # ── start/stop listening (mirrors WhisperSTT) ───────────────

    def start_listening(
        self,
        loop: asyncio.AbstractEventLoop | None = None,
        on_final: Callable[[str], None] | None = None,
        on_partial: Callable[[str], None] | None = None,
    ) -> bool:
        if not self._available:
            if self.preload() is False:
                return False
        if self._listening:
            return True

        self._loop = loop
        self._on_final = on_final
        self._on_partial = on_partial
        self._stop_event = asyncio.Event()
        self._reset_utterance_state()

        try:
            self._stream = _sd.RawInputStream(
                samplerate=_SAMPLE_RATE,
                blocksize=_FRAME_SAMPLES,
                dtype="int16",
                channels=1,
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as exc:
            self._last_error = f"sounddevice open failed: {exc}"
            logger.warning(
                "WhisperKitSTT: %s", self._last_error, exc_info=True,
            )
            self._stream = None
            return False

        self._listening = True
        if loop is not None:
            self._worker_task = loop.create_task(self._consume_loop())
        logger.info(
            "WhisperKitSTT listening (%d Hz, %d-ms VAD, partial @%.1fs, "
            "trail %.2fs)",
            _SAMPLE_RATE, _FRAME_MS,
            self._partial_interval_s, self._trailing_silence_s,
        )
        return True

    def stop_listening(self) -> str:
        if not self._listening:
            return self._last_final
        self._listening = False
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception:
            logger.debug("WhisperKitSTT: stream close raised", exc_info=True)
        self._stream = None

        text = self._flush_utterance(force=True)
        if text:
            self._emit_final(text)
            self._last_final = text

        try:
            self._stop_event.set()
        except Exception:
            logger.debug("stop_event.set raised", exc_info=True)
        if self._worker_task is not None:
            try:
                self._worker_task.cancel()
            except Exception:
                logger.debug(
                    "worker_task cancel raised", exc_info=True,
                )
            self._worker_task = None

        logger.info("WhisperKitSTT stopped")
        return self._last_final

    def shutdown(self) -> None:
        self.stop_listening()
        self._stop_serve()
        self._vad = None
        self._available = False
        logger.info("WhisperKitSTT shut down")

    # ── async-compatible wrappers ───────────────────────────────

    async def async_start_listening(self, **_kw: Any) -> None:
        if self._permanently_disabled:
            logger.info(
                "WhisperKitSTT permanently disabled — voice input unavailable",
            )
            return
        loop = asyncio.get_running_loop()
        self._loop = loop
        self._running_async = True

        def _on_final(text: str) -> None:
            if not text:
                return
            loop.call_soon_threadsafe(
                lambda t=text: (
                    self._bus.emit_fast(
                        "voice.final",
                        text=t,
                        language=self._language,
                        confidence=float(self._last_confidence),
                        engine=self.backend_name,
                        mic=self.mic_name,
                    ),
                ),
            )

        def _on_partial(text: str) -> None:
            if not text:
                return
            loop.call_soon_threadsafe(
                lambda t=text: (
                    self._bus.emit_fast(
                        "voice.partial",
                        text=t,
                        confidence=float(self._last_confidence),
                        engine=self.backend_name,
                        mic=self.mic_name,
                    ),
                ),
            )

        if not self.start_listening(
            loop=loop, on_final=_on_final, on_partial=_on_partial,
        ):
            logger.warning(
                "WhisperKitSTT async_start_listening: start failed (%s)",
                self._last_error or "unknown",
            )
            return

        try:
            while self._running_async and self._listening:
                await asyncio.sleep(0.5)
        finally:
            self._running_async = False

    # ── audio path ──────────────────────────────────────────────

    def _audio_callback(
        self, indata: bytes, frames: int, time_info, status,
    ) -> None:
        if status:
            logger.debug("WhisperKitSTT audio status: %s", status)
        try:
            self._tap_buffer_count += 1
            data = bytes(indata)
            if not data:
                return
            samples = _np.frombuffer(data, dtype=_np.int16) if _np else None
            rms_db: float | None = None
            if samples is not None and samples.size:
                rms_db = self._rms_db_from_int16(samples)
                self._last_audio_rms_db = rms_db
            if self._noise_gate_blocks(rms_db):
                return
            with self._ring_lock:
                self._ring.append(data)
            wc = self._whisper_confirmer
            if (
                wc is not None
                and getattr(wc, "is_enabled", lambda: False)()
                and samples is not None
                and samples.size
            ):
                try:
                    f32 = (samples.astype(_np.float32) / 32768.0)
                    wc.feed_audio(f32.tobytes())
                except Exception:
                    if not getattr(self, "_logged_wc_feed_fail", False):
                        logger.debug(
                            "WhisperKitSTT: confirmer feed_audio raised",
                            exc_info=True,
                        )
                        self._logged_wc_feed_fail = True
        except Exception:
            logger.debug("WhisperKitSTT audio_callback raised", exc_info=True)

    async def _consume_loop(self) -> None:
        try:
            while self._listening and not self._stop_event.is_set():
                await asyncio.sleep(_FRAME_MS / 1000.0)
                self._consume_once()
        except asyncio.CancelledError:  # pragma: no cover
            pass
        except Exception:
            logger.exception("WhisperKitSTT consume loop crashed")

    def _consume_once(self) -> None:
        with self._ring_lock:
            frames = list(self._ring)
            self._ring.clear()
        if not frames:
            return

        for frame in frames:
            try:
                is_speech = self._vad.is_speech(frame, _SAMPLE_RATE)
            except Exception:
                is_speech = False
            now = time.monotonic()
            if is_speech:
                if not self._utterance_frames:
                    self._utterance_started_at = now
                self._utterance_frames.append(frame)
                self._silence_frames = 0
                self._last_speech_time = now
                if self._last_audio_rms_db >= self._speech_candidate_floor_dbfs:
                    self._last_speech_candidate_time = now
            elif self._utterance_frames:
                self._utterance_frames.append(frame)
                self._silence_frames += 1

        if not self._utterance_frames:
            return

        utterance_duration_s = (
            len(self._utterance_frames) * _FRAME_MS / 1000.0
        )
        silence_duration_s = self._silence_frames * _FRAME_MS / 1000.0
        if (
            silence_duration_s >= self._trailing_silence_s
            or utterance_duration_s >= self._max_utterance_s
        ):
            text = self._flush_utterance(force=True)
            if text:
                self._emit_final(text)
                self._last_final = text
            return

        now = time.monotonic()
        since_partial = now - self._last_partial_emit_at
        if (
            utterance_duration_s >= self._partial_interval_s
            and since_partial >= self._partial_interval_s
        ):
            text = self._transcribe(self._utterance_frames, partial=True)
            if text and text != self._last_partial:
                self._last_partial = text
                self._emit_partial(text)
                self._last_partial_emit_at = now

    def _flush_utterance(self, *, force: bool) -> str:
        if not self._utterance_frames:
            return ""
        duration_ms = len(self._utterance_frames) * _FRAME_MS
        if duration_ms < self._min_utterance_ms:
            return ""
        text = self._transcribe(self._utterance_frames, partial=False)
        self._reset_utterance_state()
        return text

    def _noise_gate_blocks(self, rms_db: float | None) -> bool:
        if self._noise_floor_dbfs <= -96.0:
            return False
        if rms_db is None or rms_db >= self._noise_floor_dbfs:
            self._noise_gate_below_count = 0
            return False
        self._noise_gate_below_count += 1
        if self._noise_gate_below_count >= self._noise_gate_consecutive:
            self._noise_gate_dropped_total += 1
            return True
        return False

    @staticmethod
    def _rms_db_from_int16(samples: Any) -> float:
        try:
            arr = samples.astype(_np.float32) if _np is not None else None
            if arr is None or not getattr(arr, "size", 0):
                return -96.0
            rms = float(_np.sqrt(_np.mean(arr ** 2)))
            if rms < 1.0:
                return -96.0
            return 20.0 * float(_np.log10(rms / 32768.0))
        except Exception:
            return -96.0

    def _transcribe(self, frames: list[bytes], *, partial: bool) -> str:
        """POST PCM bytes to whisperkit-cli serve and return text."""
        if _np is None:
            return ""
        try:
            audio_bytes = b"".join(frames)
            audio = (
                _np.frombuffer(audio_bytes, dtype=_np.int16)
                .astype(_np.float32) / 32768.0
            )
            t0 = time.perf_counter()
            text = self._http_transcribe(audio, partial=partial)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.debug(
                "WhisperKitSTT %s: %d-ms audio -> %d chars in %.0f ms",
                "partial" if partial else "final",
                len(frames) * _FRAME_MS,
                len(text or ""),
                elapsed_ms,
            )
            text = (text or "").strip()

            # Sprint P2.1 (Apr 26 2026): mirror WhisperSTT's correction +
            # noise-word filter on FINAL only. Partials stream raw to keep
            # the UI flicker-free.
            if not partial and text:
                if _is_noise_word is not None:
                    try:
                        if _is_noise_word(text):
                            logger.info(
                                "WhisperKitSTT: rejected noise word: '%s'",
                                text,
                            )
                            return ""
                    except Exception:
                        logger.debug(
                            "WhisperKitSTT noise-word check raised",
                            exc_info=True,
                        )
                pre_correct = text
                if _correct_text is not None:
                    try:
                        original = text
                        text = _correct_text(text)
                        if text and text != original:
                            logger.info(
                                "WhisperKitSTT correction: '%s' -> '%s'",
                                original, text,
                            )
                    except Exception:
                        logger.debug(
                            "WhisperKitSTT correct_text raised",
                            exc_info=True,
                        )

                # Sprint P4.1 + P4.3 (Apr 26 2026): owner-aware rewrites.
                # Pronunciations + replayed corrections AFTER correct_text;
                # see voice/stt_whisper.py for the full rationale.
                if _get_owner_profile is not None:
                    try:
                        profile = _get_owner_profile()
                        if profile is not None:
                            text = profile.apply_pronunciations(text)
                            text = profile.replay_corrections(text)
                            if pre_correct and text and text != pre_correct:
                                profile.record_correction(
                                    pre_correct, text,
                                    source="whisperkit_correct_text",
                                )
                    except Exception:
                        logger.debug(
                            "OwnerProfile post-correct hook raised",
                            exc_info=True,
                        )
                text = _normalize_atom_final_text(text)
                if not text:
                    return ""
                if _ATOM_DIAGNOSTIC_AUDIO_RE.search(text):
                    logger.warning(
                        "WhisperKitSTT: rejected ATOM diagnostic self-input: '%s'",
                        text[:120],
                    )
                    return ""
                if _ATOM_SELF_SPEECH_RE.search(text):
                    logger.info(
                        "WhisperKitSTT: rejected ATOM self-speech final: '%s'",
                        text[:120],
                    )
                    return ""
            return text
        except Exception:
            logger.exception("WhisperKitSTT transcribe raised")
            return ""

    def _http_transcribe(
        self, audio_f32: Any, *, partial: bool,
    ) -> str:
        """POST a float32 mono PCM array to whisperkit-cli serve.

        WhisperKit's HTTP server accepts WAV / raw PCM via multipart and
        returns JSON ``{"text": ...}``. We use stdlib ``urllib`` so we
        don't add a hard ``requests`` / ``httpx`` dep.
        """
        import io
        import urllib.error
        import urllib.request
        import wave

        # Encode the float32 PCM as a 16-kHz mono WAV in memory. WhisperKit
        # accepts WAV / FLAC / M4A; WAV is the cheapest to produce and
        # decode on both sides.
        buf = io.BytesIO()
        try:
            pcm16 = (
                _np.clip(audio_f32, -1.0, 1.0) * 32767.0
            ).astype(_np.int16).tobytes()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(_SAMPLE_RATE)
                wf.writeframes(pcm16)
        except Exception:
            logger.debug(
                "WhisperKitSTT: WAV serialise failed", exc_info=True,
            )
            return ""

        # Sprint Ω.6.A (Apr 26 2026): whisperkit-cli 0.18.0 serves an
        # OpenAI-compatible API at /v1/audio/transcriptions, NOT the older
        # /transcribe path the original driver was written against.
        # Multipart fields:
        #   - file:    audio bytes (we send 16 kHz mono WAV)
        #   - model:   model name (whisperkit ignores the value but
        #              Vapor's OpenAI parser requires the field)
        #   - language: BCP-47 / Whisper short code or "auto"
        # WhisperKit's "language" auto-detect uses an empty string, not the
        # literal "auto" token (the OpenAI spec doesn't define "auto").
        # If the configured language is "auto" or empty we omit the field
        # entirely so WhisperKit picks the language from the audio.
        boundary = "----ATOMWhisperKitBoundary"
        body = io.BytesIO()
        body.write(f"--{boundary}\r\n".encode())
        body.write(
            b'Content-Disposition: form-data; name="file"; '
            b'filename="utterance.wav"\r\n'
        )
        body.write(b"Content-Type: audio/wav\r\n\r\n")
        body.write(buf.getvalue())
        body.write(f"\r\n--{boundary}\r\n".encode())
        body.write(b'Content-Disposition: form-data; name="model"\r\n\r\n')
        body.write(b"whisper-1")
        lang = (self._language or "").strip().lower()
        if lang and lang != "auto":
            body.write(f"\r\n--{boundary}\r\n".encode())
            body.write(
                b'Content-Disposition: form-data; name="language"\r\n\r\n',
            )
            body.write(lang.encode())
        body.write(f"\r\n--{boundary}--\r\n".encode())

        url = (
            f"http://{self._serve_host}:{self._serve_port}"
            "/v1/audio/transcriptions"
        )
        req = urllib.request.Request(
            url,
            data=body.getvalue(),
            method="POST",
            headers={
                "Content-Type": (
                    f"multipart/form-data; boundary={boundary}"
                ),
            },
        )
        try:
            with urllib.request.urlopen(
                req, timeout=10.0 if partial else 30.0,
            ) as resp:
                data = resp.read()
            payload = json.loads(data.decode("utf-8", errors="ignore"))
            text = (
                payload.get("text")
                or payload.get("transcript")
                or payload.get("result")
                or ""
            )
            return str(text)
        except urllib.error.URLError as exc:
            logger.warning(
                "WhisperKitSTT: HTTP transcribe failed (%s) -- "
                "is `whisperkit-cli serve` still running?",
                exc,
            )
            return ""
        except Exception:
            logger.debug(
                "WhisperKitSTT: HTTP transcribe parse failed", exc_info=True,
            )
            return ""

    def _reset_utterance_state(self) -> None:
        self._utterance_frames = []
        self._silence_frames = 0
        self._utterance_started_at = 0.0
        self._last_partial_emit_at = 0.0
        self._last_partial = ""

    # ── emit helpers (duck-compatible with NativeSTT/WhisperSTT) ─

    def _emit_partial(self, text: str) -> None:
        if self._is_output_muted():
            return
        if _ATOM_SELF_SPEECH_RE.search(text or ""):
            logger.debug(
                "WhisperKitSTT: suppressed self-speech partial: '%s'",
                (text or "")[:120],
            )
            return
        if _ATOM_DIAGNOSTIC_AUDIO_RE.search(text or ""):
            logger.debug(
                "WhisperKitSTT: suppressed diagnostic partial: '%s'",
                (text or "")[:120],
            )
            return
        if self._echo_guard is not None:
            try:
                if self._echo_guard(text):
                    return
            except Exception:
                logger.debug("echo_guard raised", exc_info=True)
        cb = self._on_partial
        if cb is not None:
            try:
                cb(text)
            except Exception:
                logger.debug("on_partial raised", exc_info=True)
        try:
            self._bus.emit("speech_partial", text=text)
        except Exception:
            logger.debug("emit speech_partial raised", exc_info=True)

    def _emit_final(self, text: str) -> None:
        text = _normalize_atom_final_text(text)
        if not text:
            return
        # Sprint Ω.8 R6: drop common Whisper hallucination + stage-
        # direction / non-Latin garbage finals before they hit the
        # brain. atomCurrentLogs.txt L371, L467, L509 are textbook
        # examples ("*spray the door*", "Продолжение следует...",
        # bare asterisks). Cheap; runs only on finals.
        garbage = _is_stt_garbage(
            text, language=getattr(self, "_language", "en"),
        )
        if garbage is not None:
            logger.info(
                "WhisperKitSTT: dropped garbage final (%s): '%s'",
                garbage, (text or "")[:120],
            )
            try:
                self._bus.emit_fast(
                    "metrics_event", counter="stt_garbage_dropped",
                )
            except Exception:
                pass
            return
        if self._is_output_muted():
            logger.info(
                "WhisperKitSTT: dropped final during TTS tail mute: '%s'",
                (text or "")[:120],
            )
            return
        if _ATOM_SELF_SPEECH_RE.search(text or ""):
            logger.info(
                "WhisperKitSTT: dropped self-speech final before routing: '%s'",
                (text or "")[:120],
            )
            return
        if _ATOM_DIAGNOSTIC_AUDIO_RE.search(text or ""):
            logger.warning(
                "WhisperKitSTT: dropped diagnostic final before routing: '%s'",
                (text or "")[:120],
            )
            return
        if self._echo_guard is not None:
            try:
                if self._echo_guard(text):
                    return
            except Exception:
                logger.debug("echo_guard raised", exc_info=True)

        wc = self._whisper_confirmer
        confirmer_input = text
        confirmer_corrected = False
        if wc is not None and getattr(wc, "is_enabled", lambda: False)():
            try:
                result = wc.confirm(
                    text or "", float(self._last_confidence),
                )
                new_text = (result.text or "").strip()
                if result.used_whisper:
                    if not new_text:
                        logger.debug(
                            "WhisperConfirmer collapsed noise final '%s' "
                            "to empty",
                            (text or "")[:60],
                        )
                        return
                    confirmer_corrected = (new_text != confirmer_input)
                    text = new_text
            except Exception:
                logger.debug(
                    "WhisperConfirmer.confirm raised; using streaming text",
                    exc_info=True,
                )

        # Sprint P4.1 + P4.2 + P4.3 (Apr 26 2026): owner-aware learning.
        if _get_owner_profile is not None:
            try:
                profile = _get_owner_profile()
                if profile is not None:
                    if confirmer_corrected and confirmer_input and text:
                        profile.record_correction(
                            confirmer_input, text,
                            source="whisperkit_confirmer",
                        )
                    learn = profile.parse_learn_command(text or "")
                    if learn is not None:
                        pattern, replacement = learn
                        if profile.add_pronunciation(pattern, replacement):
                            logger.info(
                                "OwnerProfile: learned pronunciation %r -> %r",
                                pattern, replacement,
                            )
            except Exception:
                logger.debug(
                    "OwnerProfile final hook raised", exc_info=True,
                )
        if _get_owner_style is not None:
            try:
                style = _get_owner_style()
                if style is not None and text:
                    style.observe(text)
            except Exception:
                logger.debug(
                    "OwnerStyle observe raised", exc_info=True,
                )

        cb = self._on_final
        if cb is not None:
            try:
                cb(text)
            except Exception:
                logger.debug("on_final raised", exc_info=True)
        try:
            self._bus.emit(
                "speech_final", text=text, language=self._language,
            )
        except Exception:
            logger.debug("emit speech_final raised", exc_info=True)

    # ── public attach surface (parity with NativeSTT/WhisperSTT) ─

    async def on_state_changed(self, old: Any = None, new: Any = None, **_kw: Any) -> None:
        try:
            new_value = getattr(new, "value", str(new)).lower()
            old_value = getattr(old, "value", str(old)).lower()
            if new_value == "speaking":
                self._clear_utterance_buffers()
                return
            if old_value == "speaking":
                self._clear_utterance_buffers()
                self._output_muted_until = max(
                    self._output_muted_until,
                    time.monotonic() + 1.2,
                )
        except Exception:
            logger.debug("WhisperKitSTT.on_state_changed raised", exc_info=True)

    async def on_tts_complete(self, **_kw: Any) -> None:
        self._clear_utterance_buffers()
        self._output_muted_until = max(
            self._output_muted_until,
            time.monotonic() + 1.2,
        )

    def attach_echo_guard(
        self, guard: Callable[[str], bool] | None,
    ) -> None:
        self._echo_guard = guard

    def attach_whisper_confirmer(self, confirmer: Any) -> None:
        self._whisper_confirmer = confirmer
        if confirmer is not None:
            setter = getattr(confirmer, "set_sample_rate", None)
            if callable(setter):
                try:
                    setter(int(_SAMPLE_RATE))
                except Exception:
                    logger.debug(
                        "WhisperConfirmer.set_sample_rate raised",
                        exc_info=True,
                    )

    # ── diagnostics (mirrors NativeSTT.get_diagnostics shape) ──

    def get_diagnostics(self) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "engine": self.backend_name,
            "available": self._available,
            "listening": self._listening,
            "running_async": self._running_async,
            "model": self._model,
            "serve_host": self._serve_host,
            "serve_port": self._serve_port,
            "serve_pid": (
                self._serve_proc.pid if self._serve_proc else None
            ),
            "tap_buffer_count": self._tap_buffer_count,
            "last_audio_rms_db": round(self._last_audio_rms_db, 1),
            "noise_floor_dbfs": round(self._noise_floor_dbfs, 1),
            "noise_gate_dropped_total": self._noise_gate_dropped_total,
            "since_last_speech_s": (
                round(now - self._last_speech_time, 1)
                if self._last_speech_time else None
            ),
            "last_error": self._last_error,
        }

    # ── compatibility shims ─────────────────────────────────────

    def _restart_recognition_chain(self) -> None:
        """Watchdog soft-restart. Bounce the sounddevice stream so the
        same recovery flow that works for WhisperSTT also works here."""
        if not self._listening:
            return
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception:
            logger.debug(
                "WhisperKitSTT: chain-restart stream-close raised",
                exc_info=True,
            )
        self._stream = None
        try:
            self._stream = _sd.RawInputStream(
                samplerate=_SAMPLE_RATE,
                blocksize=_FRAME_SAMPLES,
                dtype="int16",
                channels=1,
                callback=self._audio_callback,
            )
            self._stream.start()
            logger.info("WhisperKitSTT: sounddevice stream re-bound")
        except Exception:
            logger.warning(
                "WhisperKitSTT: stream rebind failed", exc_info=True,
            )


__all__ = ("WhisperKitSTT", "is_whisperkit_available")
