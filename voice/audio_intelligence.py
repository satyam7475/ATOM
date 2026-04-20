"""
ATOM -- Audio Intelligence Engine (JARVIS-Level Device Selection).

Central brain for all audio I/O decisions. Replaces passive metadata-only
device scoring with active audio testing, multi-signal quality assessment,
and continuous runtime monitoring.

Boot sequence:
    1. Discovery  -- enumerate all input/output devices via sounddevice + CoreAudio
    2. Testing    -- record 2s from each input, compute RMS / VAD / SNR
    3. Scoring    -- weighted multi-signal quality score per device
    4. Selection  -- pick best input, match output, set macOS system default
    5. Monitoring -- background watchdog for quality degradation
    6. Switching  -- seamless device swap without pipeline crash
    7. Feedback   -- voice announcements for device events

Owner: Satyam
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.state_manager import StateManager
    from voice.mic_manager import MicManager

logger = logging.getLogger("atom.audio_intel")

_DEVICE_HISTORY_PATH = Path("logs/audio_device_history.json")

# ── Optional dependency probes ─────────────────────────────────────────────

_HAS_SD = False
try:
    import sounddevice as _sd
    _HAS_SD = True
except ImportError:
    _sd = None  # type: ignore[assignment]

_HAS_VAD = False
try:
    import webrtcvad as _webrtcvad
    _HAS_VAD = True
except ImportError:
    _webrtcvad = None  # type: ignore[assignment]

# ── CoreAudio Bridge (macOS only) ─────────────────────────────────────────

_HAS_COREAUDIO = False

if sys.platform == "darwin":
    try:
        import ctypes
        import ctypes.util
        from ctypes import POINTER, Structure, byref, c_int, c_long, c_uint32, c_void_p

        class _AudioObjPropAddr(Structure):
            _fields_ = [
                ("mSelector", c_uint32),
                ("mScope", c_uint32),
                ("mElement", c_uint32),
            ]

        def _fourcc(s: str) -> int:
            return struct.unpack(">I", s.encode("ascii"))[0]

        _PROP_DEVICES     = _fourcc("dev#")
        _PROP_DEFAULT_IN  = _fourcc("dIn ")
        _PROP_DEFAULT_OUT = _fourcc("dOut")
        _PROP_NAME        = _fourcc("lnam")
        _SCOPE_GLOBAL     = _fourcc("glob")
        _ELEM_MAIN        = 0
        _SYS_OBJ          = 1
        _CF_UTF8           = 0x08000100

        _ca_lib = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/CoreAudio.framework/CoreAudio"
        )
        _cf_lib = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
        )

        _ca_lib.AudioObjectGetPropertyDataSize.argtypes = [
            c_uint32, POINTER(_AudioObjPropAddr), c_uint32, c_void_p, POINTER(c_uint32),
        ]
        _ca_lib.AudioObjectGetPropertyDataSize.restype = c_int

        _ca_lib.AudioObjectGetPropertyData.argtypes = [
            c_uint32, POINTER(_AudioObjPropAddr), c_uint32, c_void_p,
            POINTER(c_uint32), c_void_p,
        ]
        _ca_lib.AudioObjectGetPropertyData.restype = c_int

        _ca_lib.AudioObjectSetPropertyData.argtypes = [
            c_uint32, POINTER(_AudioObjPropAddr), c_uint32, c_void_p,
            c_uint32, c_void_p,
        ]
        _ca_lib.AudioObjectSetPropertyData.restype = c_int

        _cf_lib.CFStringGetLength.argtypes = [c_void_p]
        _cf_lib.CFStringGetLength.restype = c_long
        _cf_lib.CFStringGetCString.argtypes = [
            c_void_p, ctypes.c_char_p, c_long, c_uint32,
        ]
        _cf_lib.CFStringGetCString.restype = ctypes.c_bool
        _cf_lib.CFRelease.argtypes = [c_void_p]
        _cf_lib.CFRelease.restype = None

        _HAS_COREAUDIO = True
    except (OSError, AttributeError):
        pass


class _CoreAudioBridge:
    """Thin ctypes bridge to macOS CoreAudio for get/set default devices."""

    @staticmethod
    def available() -> bool:
        return _HAS_COREAUDIO

    @staticmethod
    def _cfstring_to_str(cfstr: Any) -> str:
        if not cfstr:
            return ""
        length = _cf_lib.CFStringGetLength(cfstr)
        buf = ctypes.create_string_buffer(length * 4 + 1)
        ok = _cf_lib.CFStringGetCString(cfstr, buf, c_long(len(buf)), c_uint32(_CF_UTF8))
        _cf_lib.CFRelease(cfstr)
        return buf.value.decode("utf-8") if ok else ""

    @staticmethod
    def get_all_device_ids() -> list[int]:
        if not _HAS_COREAUDIO:
            return []
        addr = _AudioObjPropAddr(_PROP_DEVICES, _SCOPE_GLOBAL, _ELEM_MAIN)
        size = c_uint32(0)
        if _ca_lib.AudioObjectGetPropertyDataSize(
            c_uint32(_SYS_OBJ), byref(addr), c_uint32(0), None, byref(size),
        ) != 0:
            return []
        count = size.value // 4
        ids = (c_uint32 * count)()
        if _ca_lib.AudioObjectGetPropertyData(
            c_uint32(_SYS_OBJ), byref(addr), c_uint32(0), None, byref(size), ids,
        ) != 0:
            return []
        return [ids[i] for i in range(count)]

    @staticmethod
    def get_device_name(device_id: int) -> str:
        if not _HAS_COREAUDIO:
            return ""
        addr = _AudioObjPropAddr(_PROP_NAME, _SCOPE_GLOBAL, _ELEM_MAIN)
        cfstr = c_void_p(0)
        size = c_uint32(ctypes.sizeof(c_void_p))
        if _ca_lib.AudioObjectGetPropertyData(
            c_uint32(device_id), byref(addr), c_uint32(0), None, byref(size), byref(cfstr),
        ) != 0 or not cfstr.value:
            return ""
        return _CoreAudioBridge._cfstring_to_str(cfstr)

    @staticmethod
    def _get_default_device(prop: int) -> int | None:
        if not _HAS_COREAUDIO:
            return None
        addr = _AudioObjPropAddr(prop, _SCOPE_GLOBAL, _ELEM_MAIN)
        dev_id = c_uint32(0)
        size = c_uint32(4)
        if _ca_lib.AudioObjectGetPropertyData(
            c_uint32(_SYS_OBJ), byref(addr), c_uint32(0), None, byref(size), byref(dev_id),
        ) != 0:
            return None
        return dev_id.value

    @staticmethod
    def get_default_input_id() -> int | None:
        return _CoreAudioBridge._get_default_device(_PROP_DEFAULT_IN)

    @staticmethod
    def get_default_output_id() -> int | None:
        return _CoreAudioBridge._get_default_device(_PROP_DEFAULT_OUT)

    @staticmethod
    def _set_default_device(prop: int, device_id: int) -> bool:
        if not _HAS_COREAUDIO:
            return False
        addr = _AudioObjPropAddr(prop, _SCOPE_GLOBAL, _ELEM_MAIN)
        val = c_uint32(device_id)
        return _ca_lib.AudioObjectSetPropertyData(
            c_uint32(_SYS_OBJ), byref(addr), c_uint32(0), None, c_uint32(4), byref(val),
        ) == 0

    @staticmethod
    def set_default_input(device_id: int) -> bool:
        return _CoreAudioBridge._set_default_device(_PROP_DEFAULT_IN, device_id)

    @staticmethod
    def set_default_output(device_id: int) -> bool:
        return _CoreAudioBridge._set_default_device(_PROP_DEFAULT_OUT, device_id)

    @staticmethod
    def build_name_to_id_map() -> dict[str, int]:
        """Map device name -> CoreAudio AudioObjectID for all devices."""
        mapping: dict[str, int] = {}
        for dev_id in _CoreAudioBridge.get_all_device_ids():
            name = _CoreAudioBridge.get_device_name(dev_id)
            if name:
                mapping[name] = dev_id
        return mapping


# ── Data Classes ───────────────────────────────────────────────────────────

_BT_KEYWORDS = frozenset({
    "headset", "hands-free", "bluetooth", "bt", "buds",
    "airpods", "earbuds", "jbl", "bose", "sony", "mivi",
    "oneplus", "realme", "yealink", "blaupunkt", "jabra",
    "beats", "galaxy", "nord", "wireless",
})
_USB_KEYWORDS = frozenset({
    "usb", "rode", "blue yeti", "snowball", "samson",
    "audio-technica", "fifine", "hyperx", "elgato",
})
_VIRTUAL_KEYWORDS = frozenset({
    "virtual", "voicemeeter", "cable", "vb-audio",
    "stereo mix", "wave link", "loopback", "blackhole",
    "soundflower", "zoom",
})


def _classify_device_type(name: str) -> str:
    lower = name.lower()
    if any(kw in lower for kw in _BT_KEYWORDS):
        return "bluetooth"
    if any(kw in lower for kw in _USB_KEYWORDS):
        return "usb"
    if any(kw in lower for kw in _VIRTUAL_KEYWORDS):
        return "virtual"
    if "hdmi" in lower or "display" in lower:
        return "hdmi"
    if "macbook" in lower or "built-in" in lower or "internal" in lower or "realtek" in lower:
        return "builtin"
    return "unknown"


@dataclass
class AudioDeviceProfile:
    """Full profile of an audio device with active test results."""

    index: int = -1
    name: str = ""
    device_type: str = "unknown"
    sample_rate: float = 0.0
    channels: int = 0
    input_latency_ms: float = 0.0
    is_input: bool = False
    is_output: bool = False
    is_default_input: bool = False
    is_default_output: bool = False
    host_api: str = ""
    core_audio_id: int = 0

    rms_db: float = -100.0
    snr_db: float = 0.0
    variance: float = 0.0
    speech_detected: bool = False
    speech_ratio: float = 0.0
    capture_latency_ms: float = 0.0
    test_passed: bool = False
    test_error: str = ""
    quality_score: float = 0.0

    failure_count: int = 0
    last_failure_time: float = 0.0
    consecutive_low_rms: int = 0
    rejection_reason: str = ""


# ── Scoring Helpers ────────────────────────────────────────────────────────

def _normalize_rms(rms_db: float) -> float:
    """Map RMS dBFS [-80, 0] -> [0.0, 1.0]. Anything below -80 is 0."""
    if rms_db <= -80.0:
        return 0.0
    if rms_db >= 0.0:
        return 1.0
    return (rms_db + 80.0) / 80.0


def _normalize_snr(snr_db: float) -> float:
    """Map SNR [0, 40] -> [0.0, 1.0]."""
    if snr_db <= 0.0:
        return 0.0
    if snr_db >= 40.0:
        return 1.0
    return snr_db / 40.0


def _latency_score(latency_ms: float) -> float:
    """Lower latency is better. [0, 50ms] -> [1.0, 0.0]."""
    if latency_ms <= 5.0:
        return 1.0
    if latency_ms >= 50.0:
        return 0.0
    return 1.0 - (latency_ms - 5.0) / 45.0


def _type_score(device_type: str) -> float:
    return {
        "usb": 1.0,
        "builtin": 0.7,
        "bluetooth": 0.4,
        "unknown": 0.3,
        "virtual": 0.1,
        "hdmi": 0.0,
    }.get(device_type, 0.3)


def _stability_score(failure_count: int) -> float:
    if failure_count == 0:
        return 1.0
    if failure_count >= 5:
        return 0.0
    return 1.0 - (failure_count * 0.2)


# ── VAD Helpers ────────────────────────────────────────────────────────────

def _webrtc_vad(audio_int16: np.ndarray, sample_rate: int, aggressiveness: int) -> tuple[bool, float]:
    """Run WebRTC VAD. Returns (speech_detected, speech_frame_ratio)."""
    vad = _webrtcvad.Vad(aggressiveness)
    frame_ms = 30
    frame_samples = int(sample_rate * frame_ms / 1000)
    speech_count = 0
    total = 0
    for i in range(0, len(audio_int16) - frame_samples, frame_samples):
        frame_bytes = audio_int16[i : i + frame_samples].tobytes()
        if vad.is_speech(frame_bytes, sample_rate):
            speech_count += 1
        total += 1
    ratio = speech_count / max(total, 1)
    return ratio > 0.08, ratio


def _energy_vad(audio_int16: np.ndarray, sample_rate: int) -> tuple[bool, float]:
    """Energy-based VAD fallback when webrtcvad is not installed."""
    frame_samples = int(sample_rate * 0.03)
    threshold = 500.0
    speech_count = 0
    total = 0
    for i in range(0, len(audio_int16) - frame_samples, frame_samples):
        frame = audio_int16[i : i + frame_samples].astype(np.float64)
        rms = np.sqrt(np.mean(frame ** 2))
        if rms > threshold:
            speech_count += 1
        total += 1
    ratio = speech_count / max(total, 1)
    return ratio > 0.05, ratio


def _run_vad(audio_int16: np.ndarray, sample_rate: int, aggressiveness: int = 2) -> tuple[bool, float]:
    if _HAS_VAD:
        try:
            return _webrtc_vad(audio_int16, sample_rate, aggressiveness)
        except Exception:
            logger.debug("WebRTC VAD failed, falling back to energy VAD", exc_info=True)
    return _energy_vad(audio_int16, sample_rate)


# ── Audio Analysis ─────────────────────────────────────────────────────────

def _compute_rms_db(audio_float: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(audio_float ** 2)))
    if rms < 1e-10:
        return -100.0
    return float(20.0 * np.log10(rms))


def _estimate_snr(audio_float: np.ndarray, sample_rate: int) -> float:
    """Estimate SNR by comparing overall energy to the quietest 20% of frames."""
    frame_size = int(sample_rate * 0.03)
    if len(audio_float) < frame_size * 5:
        return 0.0
    frame_energies = []
    for i in range(0, len(audio_float) - frame_size, frame_size):
        frame = audio_float[i : i + frame_size]
        energy = float(np.mean(frame ** 2))
        frame_energies.append(energy)
    if not frame_energies:
        return 0.0
    frame_energies.sort()
    n_noise = max(1, len(frame_energies) // 5)
    noise_power = np.mean(frame_energies[:n_noise])
    signal_power = np.mean(frame_energies)
    if noise_power < 1e-14:
        return 40.0
    ratio = signal_power / noise_power
    return float(min(40.0, 10.0 * np.log10(max(ratio, 1e-10))))


# ── Context Policy ─────────────────────────────────────────────────────────

_MEETING_APPS = frozenset({
    "zoom", "facetime", "google meet", "microsoft teams", "teams",
    "discord", "slack", "webex", "skype", "whereby",
})


class _ContextPolicy:
    """Adjusts device scoring bias based on runtime context signals."""

    __slots__ = ("_activity", "_active_app", "_hour", "_idle_min")

    def __init__(self) -> None:
        self._activity: str = "idle"
        self._active_app: str = ""
        self._hour: int = -1
        self._idle_min: float = 0.0

    def update(self, *, activity: str = "", app: str = "", hour: int = -1,
               idle_min: float = 0.0) -> None:
        if activity:
            self._activity = activity
        if app:
            self._active_app = app
        if hour >= 0:
            self._hour = hour
        self._idle_min = idle_min

    @property
    def is_night(self) -> bool:
        return self._hour >= 22 or 0 <= self._hour < 7

    @property
    def in_meeting(self) -> bool:
        return (
            self._activity == "meeting"
            or any(m in self._active_app.lower() for m in _MEETING_APPS)
        )

    def get_bias(self, device_type: str) -> float:
        if self.in_meeting:
            return {"bluetooth": 0.15, "usb": 0.10}.get(device_type, -0.05)
        if self._activity in ("coding", "browsing"):
            return {"builtin": 0.05, "usb": 0.05}.get(device_type, 0.0)
        if self._idle_min > 5:
            return {"builtin": 0.05}.get(device_type, 0.0)
        return 0.0

    def pick_feedback_variant(self, event: str) -> str:
        if event == "switch":
            if self.in_meeting:
                return "meeting"
            return "default"
        if event == "lost":
            return "bt_disconnect"
        return "default"


# ── Device Learning Memory ─────────────────────────────────────────────────

@dataclass
class DeviceHistoryRecord:
    """Persistent per-device performance record."""
    device_name: str = ""
    device_type: str = "unknown"
    total_sessions: int = 0
    successful_sessions: int = 0
    total_failures: int = 0
    avg_rms_db: float = -100.0
    avg_snr_db: float = 0.0
    avg_quality_score: float = 0.0
    last_used: float = 0.0
    last_success: float = 0.0
    cumulative_listen_s: float = 0.0


class DeviceMemory:
    """Persistent device history with JSON file backing."""

    __slots__ = ("_records", "_path", "_dirty", "_last_persist")

    def __init__(self, path: Path = _DEVICE_HISTORY_PATH) -> None:
        self._records: dict[str, DeviceHistoryRecord] = {}
        self._path = path
        self._dirty = False
        self._last_persist = 0.0
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            for name, rec in data.items():
                self._records[name] = DeviceHistoryRecord(**rec)
            logger.debug("Loaded %d device history records", len(self._records))
        except Exception:
            logger.debug("Could not load device history", exc_info=True)

    def persist(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {name: asdict(rec) for name, rec in self._records.items()}
            self._path.write_text(json.dumps(data, indent=2))
            self._dirty = False
            self._last_persist = time.monotonic()
        except Exception:
            logger.debug("Could not persist device history", exc_info=True)

    def maybe_persist(self, interval: float = 60.0) -> None:
        if self._dirty and time.monotonic() - self._last_persist > interval:
            self.persist()

    def get(self, device_name: str) -> DeviceHistoryRecord | None:
        return self._records.get(device_name)

    def _ensure(self, name: str, dtype: str = "unknown") -> DeviceHistoryRecord:
        if name not in self._records:
            self._records[name] = DeviceHistoryRecord(device_name=name, device_type=dtype)
        return self._records[name]

    def record_session_start(self, name: str, dtype: str = "unknown") -> None:
        rec = self._ensure(name, dtype)
        rec.total_sessions += 1
        rec.last_used = time.time()
        self._dirty = True

    def record_success(self, name: str, rms_db: float = 0.0,
                       snr_db: float = 0.0, score: float = 0.0) -> None:
        rec = self._ensure(name)
        rec.successful_sessions += 1
        rec.last_success = time.time()
        n = rec.successful_sessions
        rec.avg_rms_db = rec.avg_rms_db + (rms_db - rec.avg_rms_db) / n
        rec.avg_snr_db = rec.avg_snr_db + (snr_db - rec.avg_snr_db) / n
        rec.avg_quality_score = rec.avg_quality_score + (score - rec.avg_quality_score) / n
        self._dirty = True

    def record_failure(self, name: str) -> None:
        rec = self._ensure(name)
        rec.total_failures += 1
        self._dirty = True

    def success_rate(self, name: str) -> float:
        rec = self._records.get(name)
        if not rec or rec.total_sessions < 5:
            return 0.5
        return rec.successful_sessions / max(rec.total_sessions, 1)

    def most_reliable_device(self) -> str | None:
        best_name, best_rate = None, -1.0
        for name, rec in self._records.items():
            if rec.total_sessions < 3:
                continue
            rate = rec.successful_sessions / max(rec.total_sessions, 1)
            if rate > best_rate:
                best_rate = rate
                best_name = name
        return best_name


# ── Voice Presence Tracker ─────────────────────────────────────────────────

class VoicePresenceTracker:
    """Tracks speech density to classify conversation vs ambient mode."""

    __slots__ = ("_bus", "_event_times", "_mode", "_last_event", "_window_s")

    def __init__(self, bus: Any, *, window_s: float = 60.0) -> None:
        self._bus = bus
        self._event_times: deque[float] = deque(maxlen=200)
        self._mode: str = "ambient"
        self._last_event: float = 0.0
        self._window_s = window_s

    def wire(self) -> None:
        self._bus.on("speech_partial", self._on_speech)
        self._bus.on("speech_final", self._on_speech)

    async def _on_speech(self, *_a: Any, **_kw: Any) -> None:
        now = time.monotonic()
        self._event_times.append(now)
        self._last_event = now
        self._update_mode()

    def _update_mode(self) -> None:
        now = time.monotonic()
        cutoff = now - self._window_s
        while self._event_times and self._event_times[0] < cutoff:
            self._event_times.popleft()
        density = len(self._event_times) / (self._window_s / 60.0)
        new_mode = "conversation" if density > 4 else "ambient"
        if new_mode != self._mode:
            old = self._mode
            self._mode = new_mode
            logger.info("Voice presence: %s -> %s (density=%.1f/min)", old, new_mode, density)
            try:
                self._bus.emit(
                    "voice_presence_changed",
                    mode=new_mode, density=round(density, 1),
                )
            except Exception:
                logger.debug('Event bus emit failed', exc_info=True)

    @property
    def mode(self) -> str:
        self._update_mode()
        return self._mode

    @property
    def silence_duration(self) -> float:
        if self._last_event <= 0:
            return float("inf")
        return time.monotonic() - self._last_event

    @property
    def speech_density(self) -> float:
        now = time.monotonic()
        cutoff = now - self._window_s
        while self._event_times and self._event_times[0] < cutoff:
            self._event_times.popleft()
        return len(self._event_times) / (self._window_s / 60.0)


# ── Audio Intelligence Engine ─────────────────────────────────────────────

class AudioIntelligenceEngine:
    """Central brain for all audio I/O decisions.

    Lifecycle::

        engine = AudioIntelligenceEngine(bus, state, config, mic_manager=mm)
        best = await engine.boot()            # discover + test + select
        engine.configure(stt=stt, tts=tts)    # after pipeline build
        await engine.start_watchdog()          # after voice loop starts
        ...
        engine.shutdown()
    """

    __slots__ = (
        "_bus", "_state", "_config", "_cfg", "_mic_manager",
        "_stt", "_tts",
        "_devices", "_input_devices", "_output_devices",
        "_selected_input", "_selected_output",
        "_ca_name_map", "_watchdog", "_boot_time_ms",
        "_context_policy", "_device_memory", "_voice_presence",
        "_last_switch_time", "_stt_restart_times",
        "_switch_cooldown_s", "_switch_in_progress",
        "_last_stt_recovery_t",
    )

    def __init__(
        self,
        bus: AsyncEventBus,
        state: StateManager,
        config: dict,
        *,
        mic_manager: MicManager | None = None,
    ) -> None:
        self._bus = bus
        self._state = state
        self._config = config
        self._cfg: dict = config.get("audio_intelligence", {})
        self._mic_manager = mic_manager
        self._stt: Any = None
        self._tts: Any = None
        self._devices: list[AudioDeviceProfile] = []
        self._input_devices: list[AudioDeviceProfile] = []
        self._output_devices: list[AudioDeviceProfile] = []
        self._selected_input: AudioDeviceProfile | None = None
        self._selected_output: AudioDeviceProfile | None = None
        self._ca_name_map: dict[str, int] = {}
        self._watchdog: AudioWatchdog | None = None
        self._boot_time_ms: float = 0.0
        self._context_policy = _ContextPolicy()
        self._device_memory = DeviceMemory()
        self._voice_presence: VoicePresenceTracker | None = None
        self._last_switch_time: float = 0.0
        self._stt_restart_times: deque[float] = deque(maxlen=10)
        self._switch_cooldown_s: float = 8.0
        self._switch_in_progress: bool = False
        # Tracks the last monotonic time _smart_stt_recovery actually ran
        # so we can debounce log spam during normal chain rotations.
        self._last_stt_recovery_t: float = 0.0

    def configure(self, *, stt: Any = None, tts: Any = None) -> None:
        """Wire STT/TTS after pipeline construction (needed for switching + feedback)."""
        if stt is not None:
            self._stt = stt
        if tts is not None:
            self._tts = tts

    # ── Context Wiring ─────────────────────────────────────────────────

    def wire_context(self) -> None:
        """Subscribe to bus events for context-aware behaviour."""
        self._bus.on("context_snapshot", self._on_context_snapshot)
        self._bus.on("system_event", self._on_system_event)
        self._bus.on("stt_watchdog_restart", self._on_stt_stuck)
        self._bus.on("speech_final", self._on_speech_final)
        self._bus.on("audio_device_switched", self._on_device_switched)

        self._voice_presence = VoicePresenceTracker(self._bus)
        self._voice_presence.wire()
        logger.info("Audio intelligence context wired")

    def _can_switch(self) -> bool:
        """Cooldown guard to prevent event-storm rapid flip-flop."""
        if self._switch_in_progress:
            return False
        return time.monotonic() - self._last_switch_time > self._switch_cooldown_s

    async def _on_context_snapshot(self, **kw: Any) -> None:
        self._context_policy.update(
            activity=kw.get("activity_type", ""),
            app=kw.get("active_app", ""),
            hour=kw.get("hour", -1),
            idle_min=kw.get("idle_minutes", 0.0),
        )

    async def _on_system_event(self, **kw: Any) -> None:
        kind = kw.get("kind", "")
        if kind == "app_switch":
            self._context_policy.update(app=kw.get("app", ""))
        elif kind == "bt_connected":
            logger.info("BT connected (%s) -- triggering immediate rescan", kw.get("device", ""))
            asyncio.create_task(self._bt_rescan())
        elif kind == "bt_disconnected":
            logger.info("BT disconnected (%s) -- falling back to best non-BT", kw.get("device", ""))
            asyncio.create_task(self._bt_fallback())

    async def _bt_rescan(self) -> None:
        if not self._can_switch():
            logger.debug("BT rescan skipped -- switch cooldown active")
            return
        loop = asyncio.get_running_loop()
        new_best = await loop.run_in_executor(None, self._full_rescan)
        if new_best and (not self._selected_input or new_best.index != self._selected_input.index):
            confidence = self._compute_switch_confidence(new_best)
            await self.seamless_switch(new_best, confidence=confidence, reason="bt_connect")

    async def _bt_fallback(self) -> None:
        if not self._can_switch():
            logger.debug("BT fallback skipped -- switch cooldown active")
            return
        loop = asyncio.get_running_loop()

        def _find_non_bt() -> AudioDeviceProfile | None:
            self.discover_devices()
            non_bt = [d for d in self._input_devices if d.device_type != "bluetooth"]
            if not non_bt:
                return None
            for d in non_bt:
                self.test_device(d)
                self.score_device(d)
            non_bt.sort(key=lambda d: d.quality_score, reverse=True)
            return non_bt[0] if non_bt[0].quality_score > 0 else None

        fallback = await loop.run_in_executor(None, _find_non_bt)
        if fallback:
            await self.seamless_switch(
                fallback, confidence=0.95, reason="bt_disconnect",
                old_device_name=self._selected_input.name if self._selected_input else "",
            )

    def _full_rescan(self) -> AudioDeviceProfile | None:
        self.discover_devices()
        if not self._input_devices:
            return None
        self.test_all_input_devices()
        self.score_all_devices()
        return self.select_best_input()

    # ── Smart STT Recovery ─────────────────────────────────────────────

    async def _on_stt_stuck(self, **kw: Any) -> None:
        reason = str(kw.get("reason") or "")
        # Benign restart reasons that are NOT a device problem. The STT
        # engine rotates chains every few seconds on idle (kLSRErrorDomain
        # 301), restarts after a normal silence timeout, or recovers from
        # a recognition-callback starvation — none of those mean the mic
        # itself is broken, so don't log-spam "switching device" on them.
        _BENIGN_REASONS = {
            "recognition_starved",
            "reactive_klsr_301",
            "klsr_301_timeout",
            "no_speech_timeout",
            "silent_timeout",
            "chain_rotation",
            "idle_restart",
        }
        if reason in _BENIGN_REASONS:
            logger.debug(
                "STT watchdog restart reason=%s -- benign rotation, skipping device switch",
                reason,
            )
            return

        now = time.monotonic()
        self._stt_restart_times.append(now)
        # Debounce: only run smart recovery when we haven't done one in
        # the last 15s. Prevents every normal chain rotation from
        # re-scanning the device list.
        last_recovery = getattr(self, "_last_stt_recovery_t", 0.0)
        if (now - last_recovery) < 15.0:
            logger.debug(
                "STT stuck event (reason=%s) within 15s of last recovery -- skipping",
                reason,
            )
            return
        self._last_stt_recovery_t = now

        if self._selected_input:
            self._device_memory.record_failure(self._selected_input.name)

        asyncio.create_task(self._smart_stt_recovery())

    async def _smart_stt_recovery(self) -> None:
        sel = self._selected_input
        if not sel:
            return

        if not self._can_switch():
            logger.debug("STT stuck but device switched recently -- allowing normal restart")
            return

        now = time.monotonic()
        recent = sum(1 for t in self._stt_restart_times if now - t < 60)

        if recent >= 3:
            reliable = self._device_memory.most_reliable_device()
            if reliable and reliable != sel.name:
                candidate = next(
                    (d for d in self._input_devices if d.name == reliable), None
                )
                if candidate:
                    logger.warning(
                        "Repeated STT failures (%d in 60s) -- falling back to historically reliable '%s'",
                        recent, reliable,
                    )
                    await self.seamless_switch(candidate, confidence=0.7, reason="repeated_failure")
                    return

        if sel.rms_db > -60:
            logger.info("STT stuck but audio flowing (RMS=%.1fdB) -- switching device", sel.rms_db)
            loop = asyncio.get_running_loop()
            new_best = await loop.run_in_executor(None, self._full_rescan)
            if new_best and new_best.index != sel.index:
                await self.seamless_switch(new_best, confidence=0.6, reason="stt_stuck_audio_flowing")
        elif sel.rms_db < -85:
            logger.debug("STT stuck in silence (RMS=%.1fdB) -- suppressing restart", sel.rms_db)
        else:
            logger.debug("STT stuck at moderate level -- allowing normal restart")

    # ── Device Memory Event Handlers ───────────────────────────────────

    async def _on_speech_final(self, **kw: Any) -> None:
        if self._selected_input:
            self._device_memory.record_success(
                self._selected_input.name,
                rms_db=self._selected_input.rms_db,
                snr_db=self._selected_input.snr_db,
                score=self._selected_input.quality_score,
            )
        self._device_memory.maybe_persist()

    async def _on_device_switched(self, **kw: Any) -> None:
        new_name = kw.get("new", "")
        if new_name:
            self._device_memory.record_session_start(
                new_name,
                self._selected_input.device_type if self._selected_input else "unknown",
            )

    # ── Phase 1: Device Discovery ──────────────────────────────────────

    def discover_devices(self) -> list[AudioDeviceProfile]:
        """Enumerate all audio devices via sounddevice + CoreAudio metadata."""
        if not _HAS_SD:
            logger.warning("sounddevice not installed -- audio intelligence unavailable")
            return []

        if _HAS_COREAUDIO:
            self._ca_name_map = _CoreAudioBridge.build_name_to_id_map()
            logger.debug("CoreAudio device map: %d devices", len(self._ca_name_map))

        devices: list[AudioDeviceProfile] = []
        try:
            all_devs = _sd.query_devices()
            try:
                dev_list: list[dict] = list(all_devs)
                if dev_list and not isinstance(dev_list[0], dict):
                    dev_list = [dict(all_devs)]
            except TypeError:
                dev_list = [dict(all_devs)]

            try:
                default_in = _sd.default.device[0]
                default_out = _sd.default.device[1]
            except (IndexError, TypeError):
                default_in, default_out = -1, -1

            for i, info in enumerate(dev_list):
                name = info.get("name", "Unknown")
                max_in = info.get("max_input_channels", 0)
                max_out = info.get("max_output_channels", 0)
                if max_in <= 0 and max_out <= 0:
                    continue

                rate = info.get("default_samplerate", 0.0)
                latency = info.get("default_low_input_latency", 0.0) * 1000
                host_api_idx = info.get("hostapi", 0)
                try:
                    host_api_name = _sd.query_hostapis(host_api_idx).get("name", "")
                except Exception:
                    host_api_name = ""

                profile = AudioDeviceProfile(
                    index=i,
                    name=name,
                    device_type=_classify_device_type(name),
                    sample_rate=rate,
                    channels=max_in if max_in > 0 else max_out,
                    input_latency_ms=round(latency, 1),
                    is_input=max_in > 0,
                    is_output=max_out > 0,
                    is_default_input=(i == default_in),
                    is_default_output=(i == default_out),
                    host_api=host_api_name,
                    core_audio_id=self._ca_name_map.get(name, 0),
                )
                devices.append(profile)

        except Exception:
            logger.exception("Device discovery failed")

        self._devices = devices
        self._input_devices = [d for d in devices if d.is_input]
        self._output_devices = [d for d in devices if d.is_output]

        logger.info(
            "Discovered %d devices (%d input, %d output)",
            len(devices), len(self._input_devices), len(self._output_devices),
        )
        return devices

    # ── Phase 2: Active Audio Testing ──────────────────────────────────

    def test_device(self, device: AudioDeviceProfile) -> AudioDeviceProfile:
        """Record a short clip from the device and analyse RMS/VAD/SNR."""
        if not _HAS_SD or not device.is_input:
            device.test_error = "sounddevice unavailable or not an input device"
            return device

        duration = self._cfg.get("active_test_duration_s", 2.0)
        test_rate = 16000
        vad_aggressiveness = self._cfg.get("vad_aggressiveness", 2)

        try:
            t0 = time.perf_counter()
            audio = _sd.rec(
                int(duration * test_rate),
                samplerate=test_rate,
                channels=1,
                dtype="int16",
                device=device.index,
                blocking=True,
            )
            capture_ms = (time.perf_counter() - t0) * 1000
            device.capture_latency_ms = round(capture_ms - duration * 1000, 1)

            if audio is None or len(audio) == 0:
                device.test_error = "empty capture"
                return device

            samples_int16 = audio.flatten()
            samples_float = samples_int16.astype(np.float32) / 32768.0

            device.rms_db = _compute_rms_db(samples_float)
            device.variance = float(np.var(samples_float))
            device.snr_db = _estimate_snr(samples_float, test_rate)

            detected, ratio = _run_vad(samples_int16, test_rate, vad_aggressiveness)
            device.speech_detected = detected
            device.speech_ratio = ratio

            device.test_passed = True
            logger.debug(
                "  Tested [%d] '%s': RMS=%.1fdB  SNR=%.1fdB  VAD=%s(%.2f)  var=%.2e",
                device.index, device.name, device.rms_db, device.snr_db,
                device.speech_detected, device.speech_ratio, device.variance,
            )

        except Exception as exc:
            device.test_error = str(exc)
            logger.debug("  Test failed [%d] '%s': %s", device.index, device.name, exc)

        return device

    def test_all_input_devices(self) -> list[AudioDeviceProfile]:
        """Sequentially test every input device."""
        for dev in self._input_devices:
            self.test_device(dev)
        return self._input_devices

    # ── Phase 3: Multi-Signal Scoring ──────────────────────────────────

    def score_device(self, device: AudioDeviceProfile) -> float:
        """Compute weighted quality score with context bias and device history."""
        min_rms = self._cfg.get("min_rms_threshold_db", -80)

        if not device.test_passed:
            device.rejection_reason = f"test failed: {device.test_error}"
            device.quality_score = 0.0
            return 0.0
        if device.rms_db < min_rms:
            device.rejection_reason = f"RMS {device.rms_db:.1f}dB below threshold {min_rms}dB"
            device.quality_score = 0.0
            return 0.0
        if device.variance < 1e-10:
            device.rejection_reason = "digital silence (variance ~0)"
            device.quality_score = 0.0
            return 0.0

        bt_penalty = self._cfg.get("bluetooth_penalty", 0.1) if device.device_type == "bluetooth" else 0.0
        prefer_name = self._cfg.get("prefer_device")
        preference_bonus = 0.1 if prefer_name and prefer_name.lower() in device.name.lower() else 0.0

        W_HISTORY = 0.15
        history_score = 0.5
        hist = self._device_memory.get(device.name)
        if hist and hist.total_sessions >= 5:
            history_score = hist.successful_sessions / max(hist.total_sessions, 1)

        ctx_bias = self._context_policy.get_bias(device.device_type)
        bonus = min(ctx_bias + preference_bonus, 0.15)

        score = (
            0.13 * _normalize_rms(device.rms_db)
            + 0.30 * (1.0 if device.speech_detected else 0.3)
            + 0.17 * _normalize_snr(device.snr_db)
            + 0.08 * _latency_score(device.capture_latency_ms)
            + 0.09 * _type_score(device.device_type)
            + 0.08 * _stability_score(device.failure_count)
            + W_HISTORY * history_score
            + bonus
            - bt_penalty
        )
        device.quality_score = round(max(0.0, min(1.0, score)), 3)
        return device.quality_score

    def score_all_devices(self) -> list[AudioDeviceProfile]:
        """Score every tested input device and sort by quality."""
        for dev in self._input_devices:
            self.score_device(dev)
        self._input_devices.sort(key=lambda d: d.quality_score, reverse=True)
        return self._input_devices

    # ── Phase 4: Auto-Selection ────────────────────────────────────────

    def select_best_input(self) -> AudioDeviceProfile | None:
        """Pick the highest-scoring non-rejected input device."""
        candidates = [d for d in self._input_devices if d.quality_score > 0]
        if not candidates:
            logger.warning("No usable input device found after scoring")
            return None

        best = candidates[0]
        self._selected_input = best
        logger.info(
            "Selected input: [%d] '%s' (%s, score=%.3f)",
            best.index, best.name, best.device_type, best.quality_score,
        )
        return best

    def match_output_device(self, input_dev: AudioDeviceProfile) -> AudioDeviceProfile | None:
        """Match an output device to the selected input.

        Rules:
            - Bluetooth input  -> same-name bluetooth output (if exists)
            - Otherwise        -> system default output
        """
        if input_dev.device_type == "bluetooth":
            bt_stem = input_dev.name.split("(")[0].strip().lower()
            for out in self._output_devices:
                if bt_stem in out.name.lower() and out.is_output:
                    self._selected_output = out
                    logger.info("Matched BT output: '%s'", out.name)
                    return out

        default_out = next((d for d in self._output_devices if d.is_default_output), None)
        self._selected_output = default_out
        if default_out:
            logger.info("Using default output: '%s'", default_out.name)
        return default_out

    def apply_system_default(self, device: AudioDeviceProfile) -> bool:
        """Set macOS system default input to the selected device via CoreAudio."""
        if not self._cfg.get("set_system_default", True):
            logger.info("set_system_default disabled in config -- skipping")
            return False

        if device.is_default_input:
            logger.info("'%s' is already the system default input", device.name)
            return True

        if not _HAS_COREAUDIO or not device.core_audio_id:
            logger.warning(
                "Cannot set system default: CoreAudio %s, device_id=%d. "
                "Recommendation: manually switch macOS Sound Input to '%s'.",
                "available" if _HAS_COREAUDIO else "unavailable",
                device.core_audio_id, device.name,
            )
            return False

        ok = _CoreAudioBridge.set_default_input(device.core_audio_id)
        if ok:
            logger.info("System default input changed to '%s' (CoreAudio ID %d)", device.name, device.core_audio_id)
        else:
            logger.warning(
                "Failed to change system default input to '%s'. "
                "Manually switch in System Settings > Sound > Input.",
                device.name,
            )
        return ok

    def apply_output_default(self, device: AudioDeviceProfile) -> bool:
        """Set macOS system default output if needed."""
        if device.is_default_output:
            return True
        if not _HAS_COREAUDIO or not device.core_audio_id:
            return False
        return _CoreAudioBridge.set_default_output(device.core_audio_id)

    # ── Full Boot Sequence ─────────────────────────────────────────────

    async def boot(self) -> AudioDeviceProfile | None:
        """Run the complete boot sequence: discover -> test -> score -> select.

        Heavy I/O (recording) runs in an executor to keep the event loop free.
        """
        if not self._cfg.get("enabled", True):
            logger.info("Audio Intelligence disabled in config")
            return None

        t0 = time.monotonic()
        loop = asyncio.get_running_loop()

        def _blocking_boot() -> AudioDeviceProfile | None:
            self.discover_devices()
            if not self._input_devices:
                logger.warning("No input devices found")
                return None
            self.test_all_input_devices()
            self.score_all_devices()
            best = self.select_best_input()
            if best is None:
                return None
            self.match_output_device(best)
            was_already_default = best.is_default_input
            self.apply_system_default(best)
            if self._selected_output:
                self.apply_output_default(self._selected_output)
            if not was_already_default:
                time.sleep(0.8)
                logger.debug("CoreAudio default switch: waited 0.8s for HAL propagation")
            return best

        best = await loop.run_in_executor(None, _blocking_boot)
        self._boot_time_ms = (time.monotonic() - t0) * 1000

        if self._mic_manager and best:
            from voice.mic_manager import MicDeviceProfile
            mic_profile = MicDeviceProfile(
                index=best.index,
                name=best.name,
                host_api=best.host_api,
                max_input_channels=best.channels,
                default_sample_rate=int(best.sample_rate),
                input_latency_ms=best.input_latency_ms,
                device_type=best.device_type,
                quality_score=int(best.quality_score * 100),
                is_default=best.is_default_input,
                supports_16khz=(8000 <= best.sample_rate <= 48000),
                supports_44khz=(best.sample_rate >= 44100),
            )
            self._mic_manager.active_device = mic_profile

        if best:
            self._device_memory.record_session_start(best.name, best.device_type)

        report = self.diagnostic_report()
        logger.info("\n%s", report)

        try:
            self._bus.emit(
                "audio_intelligence_boot",
                selected_input=best.name if best else None,
                selected_output=self._selected_output.name if self._selected_output else None,
                devices_tested=len(self._input_devices),
                boot_time_ms=round(self._boot_time_ms),
            )
        except Exception:
            logger.debug('Event bus emit failed', exc_info=True)

        self._emit_state_diff()
        return best

    @property
    def input_output_hardware_mismatch(self) -> bool:
        """True when selected input and output are on different hardware types.

        Voice Processing I/O (echo cancellation) fails when, e.g., a built-in
        mic is paired with Bluetooth output.  Callers should disable VPIO in
        this case.
        """
        si, so = self._selected_input, self._selected_output
        if si is None or so is None:
            return False
        return si.device_type != so.device_type

    # ── Phase 5 + 6: Watchdog + Seamless Switching ─────────────────────

    async def start_watchdog(self) -> None:
        """Start the background audio health watchdog."""
        if not self._cfg.get("enabled", True):
            return
        interval = self._cfg.get("monitoring_interval_s", 10)
        self._watchdog = AudioWatchdog(self, interval=interval)
        await self._watchdog.start()
        logger.info("AudioWatchdog started (interval=%ds)", interval)

    async def seamless_switch(
        self,
        new_device: AudioDeviceProfile,
        *,
        confidence: float = 0.5,
        reason: str = "quality",
        old_device_name: str = "",
    ) -> bool:
        """Pause STT, switch device, resume STT without crashing the pipeline."""
        if not self._can_switch():
            logger.info("Seamless switch blocked by cooldown (reason=%s, target=%s)", reason, new_device.name)
            return False

        from voice.recovery_lock import voice_recovery_lock, stream_drain_delay

        old_name = old_device_name or (self._selected_input.name if self._selected_input else "none")
        logger.info("Seamless switch: '%s' -> '%s' (reason=%s, conf=%.2f)", old_name, new_device.name, reason, confidence)

        async with voice_recovery_lock(
            f"audio_intelligence:{reason}",
            max_wait_s=2.0,
        ) as got_lock:
            if not got_lock:
                logger.info(
                    "Seamless switch deferred — STT watchdog is restarting (reason=%s)",
                    reason,
                )
                return False

            self._switch_in_progress = True

            try:
                if self._stt and hasattr(self._stt, "stop"):
                    self._stt.stop()
                elif self._stt and hasattr(self._stt, "stop_listening"):
                    self._stt.stop_listening()

                # Give CoreAudio time to fully release the old input stream
                # before we re-bind. Without this, the subsequent start_listening
                # sees PaMacCore (AUHAL) err=-50 because the device is still
                # owned by the previous PortAudio stream.
                await stream_drain_delay(400)

                ok = self.apply_system_default(new_device)
                if not ok:
                    logger.warning("System default switch failed for '%s'", new_device.name)

                await asyncio.sleep(0.3)

                if self._stt and hasattr(self._stt, "start_listening"):
                    self._stt.start_listening()
                elif self._stt and hasattr(self._stt, "async_start_listening"):
                    asyncio.ensure_future(self._stt.async_start_listening())

                self._selected_input = new_device
                self._last_switch_time = time.monotonic()
                self.match_output_device(new_device)
                if self._selected_output:
                    self.apply_output_default(self._selected_output)

                if self._mic_manager:
                    from voice.mic_manager import MicDeviceProfile
                    self._mic_manager.active_device = MicDeviceProfile(
                        index=new_device.index,
                        name=new_device.name,
                        host_api=new_device.host_api,
                        max_input_channels=new_device.channels,
                        default_sample_rate=int(new_device.sample_rate),
                        input_latency_ms=new_device.input_latency_ms,
                        device_type=new_device.device_type,
                        quality_score=int(new_device.quality_score * 100),
                        is_default=True,
                        supports_16khz=(8000 <= new_device.sample_rate <= 48000),
                        supports_44khz=(new_device.sample_rate >= 44100),
                    )

                try:
                    self._bus.emit(
                        "audio_device_switched",
                        old=old_name,
                        new=new_device.name,
                        score=new_device.quality_score,
                        reason=reason,
                    )
                except Exception:
                    logger.debug('Async sleep step failed', exc_info=True)

                self._emit_state_diff()

                await self.voice_feedback(
                    "switch" if reason != "bt_disconnect" else "lost",
                    confidence=confidence,
                    device_name=new_device.name,
                    old_device=old_name,
                    reason=reason,
                )
                return True

            except Exception:
                logger.exception("Seamless switch failed")
                await self.voice_feedback("switch_failed")
                return False
            finally:
                self._switch_in_progress = False

    # ── Phase 7: Voice Feedback (context-aware + confidence gating) ───

    _FEEDBACK_MESSAGES: dict[str, dict[str, str]] = {
        "switch": {
            "meeting": "Switching to headset for your call.",
            "bt_disconnect": "Your {old_device} disconnected. Using {device_name} now.",
            "noisy": "Switching to a closer microphone for clarity.",
            "default": "Switching to a better microphone.",
        },
        "lost": {
            "bt_disconnect": "Your {old_device} disconnected. Falling back to {device_name}.",
            "default": "Mic disconnected. Falling back to built-in.",
        },
        "boot": {
            "default": "Audio calibrated. Using {device_name}.",
        },
        "degraded": {
            "default": "Audio quality dropping. Checking alternatives.",
        },
        "no_device": {
            "default": "No suitable microphone detected. Please check your audio devices.",
        },
        "switch_failed": {
            "default": "Microphone switch failed. Continuing with current device.",
        },
    }

    def _compute_switch_confidence(self, target: AudioDeviceProfile) -> float:
        """Derive confidence [0-1] from score gap, history, and context."""
        if not self._selected_input:
            return 0.8
        gap = target.quality_score - self._selected_input.quality_score
        conf = 0.5 + gap * 2.0
        hist = self._device_memory.get(target.name)
        if hist and hist.total_sessions >= 5:
            sr = hist.successful_sessions / max(hist.total_sessions, 1)
            conf += (sr - 0.5) * 0.3
        if self._context_policy.in_meeting and target.device_type in ("bluetooth", "usb"):
            conf += 0.15
        return max(0.0, min(1.0, conf))

    async def voice_feedback(self, event: str, *, confidence: float = 0.5, **kwargs: Any) -> None:
        """Speak a context-aware status message, gated by confidence."""
        if not self._cfg.get("voice_feedback", True):
            return
        if self._context_policy.is_night:
            return

        if confidence > 0.9:
            logger.debug("Silent switch (confidence=%.2f, event=%s)", confidence, event)
            return
        if confidence < 0.4:
            try:
                self._bus.emit(
                    "audio_confirm_needed",
                    event=event, confidence=confidence, **kwargs,
                )
            except Exception:
                logger.debug('Event bus emit failed', exc_info=True)
            return

        if self._tts is None:
            return

        variants = self._FEEDBACK_MESSAGES.get(event, {})
        variant_key = self._context_policy.pick_feedback_variant(event)
        reason = kwargs.get("reason", "")
        if reason == "bt_disconnect":
            variant_key = "bt_disconnect"
        template = variants.get(variant_key) or variants.get("default", "")
        if not template:
            return
        try:
            msg = template.format(**kwargs)
        except KeyError:
            msg = variants.get("default", "")
        try:
            if hasattr(self._tts, "speak"):
                result = self._tts.speak(msg)
                if asyncio.iscoroutine(result):
                    await result
        except Exception:
            logger.debug("Voice feedback failed for event '%s'", event, exc_info=True)

    # ── Phase 9: Diagnostics ──────────────────────────────────────────

    def diagnostic_report(self) -> str:
        """Generate a human-readable boot diagnostic report."""
        lines = ["=== ATOM Audio Intelligence Report ==="]
        lines.append(f"Devices found: {len(self._input_devices)} input, {len(self._output_devices)} output")

        for dev in self._input_devices:
            status = ""
            if self._selected_input and dev.index == self._selected_input.index:
                status = " [SELECTED]"
            elif dev.rejection_reason:
                status = f" [REJECTED: {dev.rejection_reason}]"
            hist = self._device_memory.get(dev.name)
            hist_tag = ""
            if hist and hist.total_sessions >= 3:
                sr = hist.successful_sessions / max(hist.total_sessions, 1)
                hist_tag = f", HistSR: {sr:.0%}"
            lines.append(
                f"  [{dev.index}] {dev.name} ({dev.device_type}) -- "
                f"RMS: {dev.rms_db:.1f}dB, SNR: {dev.snr_db:.1f}dB, "
                f"VAD: {dev.speech_detected}({dev.speech_ratio:.2f}), "
                f"Score: {dev.quality_score:.3f}{hist_tag}{status}"
            )

        default_in = next((d for d in self._input_devices if d.is_default_input), None)
        lines.append(f"System default input: {default_in.name if default_in else 'unknown'}")
        lines.append(f"Selected input: {self._selected_input.name if self._selected_input else 'none'}")
        lines.append(f"Selected output: {self._selected_output.name if self._selected_output else 'system default'}")
        lines.append(f"CoreAudio bridge: {'active' if _HAS_COREAUDIO else 'unavailable'}")
        lines.append(f"WebRTC VAD: {'active' if _HAS_VAD else 'fallback (energy-based)'}")
        lines.append(f"Boot time: {self._boot_time_ms:.0f}ms")

        vp = self._voice_presence
        lines.append(f"Voice presence: {vp.mode if vp else 'not wired'}")
        ctx = self._context_policy
        lines.append(f"Context: activity={ctx._activity}, app={ctx._active_app or '(none)'}, night={ctx.is_night}")
        lines.append(f"Device memory: {len(self._device_memory._records)} records")
        lines.append("=" * 42)
        return "\n".join(lines)

    def get_diagnostics(self) -> dict[str, Any]:
        """Structured diagnostics for dashboard/LLM/router."""
        vp = self._voice_presence
        wd = self._watchdog
        return {
            "enabled": self._cfg.get("enabled", True),
            "input_devices": len(self._input_devices),
            "output_devices": len(self._output_devices),
            "selected_input": self._selected_input.name if self._selected_input else None,
            "selected_output": self._selected_output.name if self._selected_output else None,
            "selected_score": self._selected_input.quality_score if self._selected_input else 0,
            "coreaudio": _HAS_COREAUDIO,
            "webrtc_vad": _HAS_VAD,
            "boot_time_ms": round(self._boot_time_ms),
            "watchdog_running": wd is not None and wd.running,
            "voice_presence": vp.mode if vp else "unknown",
            "speech_density": round(vp.speech_density, 1) if vp else 0,
            "context": {
                "activity": self._context_policy._activity,
                "active_app": self._context_policy._active_app,
                "in_meeting": self._context_policy.in_meeting,
                "is_night": self._context_policy.is_night,
            },
            "device_memory_records": len(self._device_memory._records),
            "switch_history": wd.switch_history if wd else [],
            "devices": [
                {
                    "name": d.name,
                    "type": d.device_type,
                    "rms_db": d.rms_db,
                    "snr_db": d.snr_db,
                    "vad": d.speech_detected,
                    "score": d.quality_score,
                    "selected": self._selected_input is not None and d.index == self._selected_input.index,
                    "rejected": d.rejection_reason or None,
                    "history_sr": round(self._device_memory.success_rate(d.name), 2),
                }
                for d in self._input_devices
            ],
        }

    def get_status_for_llm(self) -> str:
        """Compact string for LLM context injection."""
        sel = self._selected_input
        if not sel:
            return "[AUDIO] No active input device"
        wd_status = "monitoring" if self._watchdog and self._watchdog.running else "off"
        vp_mode = self._voice_presence.mode if self._voice_presence else "unknown"
        ctx_act = self._context_policy._activity
        return (
            f"[AUDIO] input={sel.name} ({sel.device_type}), "
            f"score={sel.quality_score:.2f}, watchdog={wd_status}, "
            f"presence={vp_mode}, activity={ctx_act}"
        )

    # ── Dashboard State Emission ─────────────────────────────────────

    def _emit_state_diff(self) -> None:
        """Push audio state to the dashboard via the runtime state bridge."""
        sel = self._selected_input
        vp = self._voice_presence
        wd = self._watchdog
        try:
            self._bus.emit_fast(
                "state.diff",
                diff={
                    "audio": {
                        "selected_input": sel.name if sel else None,
                        "selected_output": self._selected_output.name if self._selected_output else None,
                        "input_score": sel.quality_score if sel else 0,
                        "input_rms_db": round(sel.rms_db, 1) if sel else -100,
                        "input_snr_db": round(sel.snr_db, 1) if sel else 0,
                        "devices": [
                            {
                                "name": d.name,
                                "type": d.device_type,
                                "score": d.quality_score,
                                "selected": sel is not None and d.index == sel.index,
                            }
                            for d in self._input_devices
                        ],
                        "voice_presence": vp.mode if vp else "unknown",
                        "watchdog_status": "monitoring" if wd and wd.running else "off",
                        "updated_at": time.time(),
                    }
                },
                source="audio_intel",
            )
        except Exception:
            logger.debug("State diff emission failed", exc_info=True)

    # ── Lifecycle ──────────────────────────────────────────────────────

    @property
    def selected_input(self) -> AudioDeviceProfile | None:
        return self._selected_input

    @property
    def selected_output(self) -> AudioDeviceProfile | None:
        return self._selected_output

    @property
    def input_devices(self) -> list[AudioDeviceProfile]:
        return self._input_devices

    def record_failure(self, device_name: str | None = None) -> None:
        """Track a failure for the active (or named) device."""
        target = self._selected_input
        if device_name:
            target = next((d for d in self._input_devices if d.name == device_name), target)
        if target:
            target.failure_count += 1
            target.last_failure_time = time.monotonic()
            self.score_device(target)
            logger.info("Device '%s' failure #%d recorded", target.name, target.failure_count)

    @property
    def voice_presence(self) -> VoicePresenceTracker | None:
        return self._voice_presence

    @property
    def device_memory(self) -> DeviceMemory:
        return self._device_memory

    @property
    def context_policy(self) -> _ContextPolicy:
        return self._context_policy

    def shutdown(self) -> None:
        if self._watchdog:
            self._watchdog.stop()
            self._watchdog = None
        self._device_memory.persist()
        logger.info("AudioIntelligenceEngine shutdown")


# ── Audio Watchdog ─────────────────────────────────────────────────────────

class AudioWatchdog:
    """Background monitor with trend tracking and predictive switching."""

    __slots__ = (
        "_engine", "_interval", "_base_interval", "_task", "_running",
        "_degradation_count", "_max_degradation",
        "_rms_history", "_snr_history", "_pre_warm_candidate",
        "_switch_history",
    )

    def __init__(self, engine: AudioIntelligenceEngine, *, interval: float = 10.0) -> None:
        self._engine = engine
        self._interval = interval
        self._base_interval = interval
        self._task: asyncio.Task | None = None
        self._running = False
        self._degradation_count = 0
        cfg = engine._cfg
        self._max_degradation = cfg.get("degradation_checks_before_switch", 3)
        self._rms_history: deque[tuple[float, float]] = deque(maxlen=30)
        self._snr_history: deque[tuple[float, float]] = deque(maxlen=30)
        self._pre_warm_candidate: AudioDeviceProfile | None = None
        self._switch_history: deque[dict[str, Any]] = deque(maxlen=20)

    @property
    def running(self) -> bool:
        return self._running

    @property
    def switch_history(self) -> list[dict[str, Any]]:
        return list(self._switch_history)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None

    def _adapt_interval(self) -> None:
        """Adjust monitoring frequency based on voice presence mode."""
        vp = self._engine._voice_presence
        if vp and vp.mode == "conversation":
            self._interval = max(3.0, self._base_interval * 0.5)
        else:
            self._interval = self._base_interval

    async def _monitor_loop(self) -> None:
        while self._running:
            try:
                self._adapt_interval()
                await asyncio.sleep(self._interval)
                if not self._running:
                    break
                await self._check_health()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.debug("Watchdog check error", exc_info=True)

    async def _check_health(self) -> None:
        selected = self._engine._selected_input
        if selected is None:
            return

        loop = asyncio.get_running_loop()

        def _quick_test() -> AudioDeviceProfile:
            return self._engine.test_device(selected)

        try:
            result = await loop.run_in_executor(None, _quick_test)
        except Exception:
            logger.debug("Watchdog quick test failed", exc_info=True)
            return

        now = time.monotonic()
        self._rms_history.append((now, result.rms_db))
        self._snr_history.append((now, result.snr_db))

        if result.rms_db < self._engine._cfg.get("min_rms_threshold_db", -80):
            self._degradation_count += 1
            selected.consecutive_low_rms += 1
            logger.warning(
                "Audio degraded: '%s' RMS=%.1fdB (%d/%d)",
                selected.name, result.rms_db,
                self._degradation_count, self._max_degradation,
            )
        else:
            self._degradation_count = 0
            selected.consecutive_low_rms = 0

        await self._check_trends()

        if self._degradation_count >= self._max_degradation:
            logger.warning("Degradation threshold reached -- triggering re-evaluation")
            self._degradation_count = 0
            await self._try_switch(reason="degradation")

        connected = await self._check_device_presence()
        if not connected:
            logger.warning("Selected device '%s' disappeared -- triggering re-scan", selected.name)
            self._engine.record_failure(selected.name)
            await self._try_switch(reason="disconnected")

        self._engine._emit_state_diff()

    async def _check_trends(self) -> None:
        """Detect quality degradation trends and pre-warm alternatives."""
        if len(self._rms_history) < 10:
            return

        rms_slope = self._compute_slope(self._rms_history)
        snr_slope = self._compute_slope(self._snr_history)

        if rms_slope < -0.5 or snr_slope < -0.3:
            if self._pre_warm_candidate is None:
                logger.info(
                    "Quality trend degrading (rms_slope=%.2f, snr_slope=%.2f) -- pre-warming alternative",
                    rms_slope, snr_slope,
                )
                loop = asyncio.get_running_loop()
                self._pre_warm_candidate = await loop.run_in_executor(None, self._find_alternative)

            min_rms = self._engine._cfg.get("min_rms_threshold_db", -80)
            current_rms = self._rms_history[-1][1] if self._rms_history else -100
            if current_rms < min_rms + 10 and self._pre_warm_candidate:
                logger.warning("Predictive switch: quality near threshold -- using pre-warmed candidate")
                candidate = self._pre_warm_candidate
                self._pre_warm_candidate = None
                await self._engine.seamless_switch(
                    candidate, confidence=0.85, reason="predictive",
                )
                self._switch_history.append({
                    "time": time.time(), "reason": "predictive",
                    "from": self._engine._selected_input.name if self._engine._selected_input else "",
                    "to": candidate.name,
                })
        else:
            self._pre_warm_candidate = None

    @staticmethod
    def _compute_slope(history: deque[tuple[float, float]]) -> float:
        """Simple linear regression slope over the history window."""
        if len(history) < 5:
            return 0.0
        n = len(history)
        t0 = history[0][0]
        xs = [h[0] - t0 for h in history]
        ys = [h[1] for h in history]
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        den = sum((x - mean_x) ** 2 for x in xs)
        if den < 1e-10:
            return 0.0
        return num / den

    def _find_alternative(self) -> AudioDeviceProfile | None:
        """Discover and test devices to find a pre-warmed alternative."""
        current = self._engine._selected_input
        self._engine.discover_devices()
        candidates = [d for d in self._engine._input_devices
                       if not current or d.index != current.index]
        if not candidates:
            return None
        for d in candidates:
            self._engine.test_device(d)
            self._engine.score_device(d)
        candidates.sort(key=lambda d: d.quality_score, reverse=True)
        return candidates[0] if candidates[0].quality_score > 0 else None

    async def _check_device_presence(self) -> bool:
        selected = self._engine._selected_input
        if not selected or not _HAS_SD:
            return True
        try:
            loop = asyncio.get_running_loop()
            devs = await loop.run_in_executor(None, _sd.query_devices)
            dev_list = list(devs) if not isinstance(devs, list) else devs
            return any(
                (d.get("name") if isinstance(d, dict) else "") == selected.name
                for d in dev_list
            )
        except Exception:
            return True

    async def _try_switch(self, *, reason: str = "degradation") -> None:
        await self._engine.voice_feedback("degraded")

        loop = asyncio.get_running_loop()

        def _rescan() -> AudioDeviceProfile | None:
            self._engine.discover_devices()
            if not self._engine._input_devices:
                return None
            self._engine.test_all_input_devices()
            self._engine.score_all_devices()
            return self._engine.select_best_input()

        new_best = await loop.run_in_executor(None, _rescan)
        if new_best is None:
            await self._engine.voice_feedback("no_device")
            return

        current = self._engine._selected_input
        if current and new_best.index == current.index:
            logger.info("Re-scan: same device is still best -- no switch")
            return

        confidence = self._engine._compute_switch_confidence(new_best)
        old_name = current.name if current else ""
        await self._engine.seamless_switch(new_best, confidence=confidence, reason=reason)
        self._switch_history.append({
            "time": time.time(), "reason": reason,
            "from": old_name, "to": new_best.name,
        })
