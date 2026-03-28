"""
ATOM -- Microphone Ownership Manager (JARVIS-Level Audio Device Intelligence).

Ensures only ONE component can own the microphone at a time, and provides
deep device profiling, quality assessment, and input optimization.

Ownership protocol:
    1. Call acquire(owner_name) before opening a PyAudio stream
    2. Call release(owner_name) after closing the stream
    3. acquire() blocks until the current owner releases

Device Intelligence:
    - Enumerates all audio input devices with full metadata
    - Profiles device capabilities (sample rates, channels, latency)
    - Detects Bluetooth vs wired vs USB vs virtual
    - Quality scoring per device (0-100)
    - Auto-selects best available device
    - Tracks device health (failures, reconnects)

This prevents PortAudioError on Windows where exclusive-mode audio
devices reject concurrent stream opens.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("atom.mic")


@dataclass
class MicDeviceProfile:
    """Full profile of an audio input device."""
    index: int = -1
    name: str = ""
    host_api: str = ""
    max_input_channels: int = 0
    default_sample_rate: int = 0
    input_latency_ms: float = 0.0
    device_type: str = "unknown"  # bluetooth, usb, builtin, virtual, hdmi
    quality_score: int = 0
    is_default: bool = False
    supports_16khz: bool = False
    supports_44khz: bool = False
    failure_count: int = 0
    last_failure_time: float = 0.0


class MicManager:
    """Central microphone lock with device profiling and auto-selection.

    Thread-safe: uses threading.Condition internally. Safe to call from
    asyncio executor threads and any background audio consumers.
    """

    __slots__ = (
        "_lock", "_condition", "_owner", "_devices",
        "_active_device", "_pyaudio", "_profiled",
    )

    _BT_KEYWORDS = (
        "headset", "hands-free", "bluetooth", "bt", "buds",
        "airpods", "earbuds", "jbl", "bose", "sony", "mivi",
        "oneplus", "realme", "yealink", "blaupunkt", "jabra",
    )
    _USB_KEYWORDS = ("usb", "rode", "blue yeti", "snowball", "samson",
                     "audio-technica", "fifine", "hyperx", "elgato")
    _VIRTUAL_KEYWORDS = ("virtual", "voicemeeter", "cable", "vb-audio",
                         "stereo mix", "wave link", "loopback")
    _DRIVER_BLACKLIST = (
        "@system32\\drivers", "\\drivers\\", ".sys,",
        ".sys)", "bthhfenum", "bthenum",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._owner: str | None = None
        self._devices: list[MicDeviceProfile] = []
        self._active_device: MicDeviceProfile | None = None
        self._pyaudio: Any = None
        self._profiled = False

    def profile_devices(self, pyaudio_instance: Any = None) -> list[MicDeviceProfile]:
        """Enumerate and profile all audio input devices.

        Scores each device on quality (0-100) considering:
            - Device type (USB > BT > builtin > virtual)
            - Sample rate support (16kHz preferred for STT)
            - Channel count
            - Input latency
            - Failure history
        """
        pa = pyaudio_instance
        if pa is None:
            try:
                import pyaudio
                pa = pyaudio.PyAudio()
                self._pyaudio = pa
            except ImportError:
                logger.warning("pyaudio not available for device profiling")
                return []

        devices: list[MicDeviceProfile] = []
        try:
            default_idx = -1
            try:
                default_info = pa.get_default_input_device_info()
                default_idx = int(default_info.get("index", -1))
            except Exception:
                pass

            for i in range(pa.get_device_count()):
                try:
                    info = pa.get_device_info_by_index(i)
                except Exception:
                    continue

                if info.get("maxInputChannels", 0) <= 0:
                    continue

                name = info.get("name", "Unknown")
                lower_name = name.lower()

                if any(blk in lower_name for blk in self._DRIVER_BLACKLIST):
                    continue

                try:
                    host_api_info = pa.get_host_api_info_by_index(
                        info.get("hostApi", 0))
                    host_api = host_api_info.get("name", "Unknown")
                except Exception:
                    host_api = "Unknown"

                rate = int(info.get("defaultSampleRate", 0))
                latency = float(info.get("defaultLowInputLatency", 0)) * 1000

                device_type = self._classify_device_type(lower_name)

                profile = MicDeviceProfile(
                    index=i,
                    name=name,
                    host_api=host_api,
                    max_input_channels=int(info.get("maxInputChannels", 0)),
                    default_sample_rate=rate,
                    input_latency_ms=round(latency, 1),
                    device_type=device_type,
                    is_default=(i == default_idx),
                    supports_16khz=(8000 <= rate <= 48000),
                    supports_44khz=(rate >= 44100),
                )

                existing = next(
                    (d for d in self._devices if d.index == i), None)
                if existing:
                    profile.failure_count = existing.failure_count
                    profile.last_failure_time = existing.last_failure_time

                profile.quality_score = self._score_device(profile)
                devices.append(profile)

        except Exception:
            logger.exception("Device profiling error")

        self._devices = sorted(devices, key=lambda d: d.quality_score, reverse=True)
        self._profiled = True

        if self._devices:
            best = self._devices[0]
            logger.info(
                "Profiled %d input devices. Best: [%d] '%s' (%s, score=%d, rate=%d)",
                len(self._devices), best.index, best.name,
                best.device_type, best.quality_score, best.default_sample_rate,
            )
            for d in self._devices:
                logger.debug(
                    "  Device [%d] '%s': type=%s, score=%d, rate=%d, latency=%.1fms, api=%s",
                    d.index, d.name, d.device_type, d.quality_score,
                    d.default_sample_rate, d.input_latency_ms, d.host_api,
                )
        else:
            logger.warning("No audio input devices found!")

        return self._devices

    def get_best_device(self, prefer_bluetooth: bool = True) -> MicDeviceProfile | None:
        """Get the highest-quality device, optionally preferring Bluetooth."""
        if not self._devices:
            return None

        if prefer_bluetooth:
            bt_devices = [d for d in self._devices if d.device_type == "bluetooth"]
            usable_bt = [d for d in bt_devices if d.failure_count < 3]
            if usable_bt:
                return usable_bt[0]

        usable = [d for d in self._devices if d.failure_count < 5]
        return usable[0] if usable else self._devices[0]

    def record_failure(self, device_index: int) -> None:
        """Track a device failure for quality scoring."""
        for d in self._devices:
            if d.index == device_index:
                d.failure_count += 1
                d.last_failure_time = time.monotonic()
                d.quality_score = self._score_device(d)
                logger.info(
                    "Device [%d] '%s' failure #%d recorded",
                    d.index, d.name, d.failure_count,
                )
                break

    def get_device_summary(self) -> str:
        """Human-readable summary of audio devices for voice output."""
        if not self._devices:
            return "No audio input devices detected."
        lines = [f"Found {len(self._devices)} input device{'s' if len(self._devices) > 1 else ''}:"]
        for d in self._devices[:5]:
            status = "ACTIVE" if self._active_device and d.index == self._active_device.index else "available"
            lines.append(
                f"  {d.name} ({d.device_type}, quality {d.quality_score}/100, {status})"
            )
        return "\n".join(lines)

    def get_device_intelligence_for_llm(self) -> str:
        """Compact device info for LLM context."""
        if not self._devices:
            return "[MIC] No input devices"
        active = self._active_device
        active_str = f"{active.name} ({active.device_type})" if active else "none"
        return (
            f"[MIC] {len(self._devices)} devices, active={active_str}, "
            f"best_score={self._devices[0].quality_score}/100"
        )

    # ── Device Classification ────────────────────────────────────────

    def _classify_device_type(self, lower_name: str) -> str:
        if any(kw in lower_name for kw in self._BT_KEYWORDS):
            return "bluetooth"
        if any(kw in lower_name for kw in self._USB_KEYWORDS):
            return "usb"
        if any(kw in lower_name for kw in self._VIRTUAL_KEYWORDS):
            return "virtual"
        if "hdmi" in lower_name or "display" in lower_name:
            return "hdmi"
        if "realtek" in lower_name or "integrated" in lower_name:
            return "builtin"
        return "unknown"

    @staticmethod
    def _score_device(d: MicDeviceProfile) -> int:
        """Score a device 0-100 for STT suitability."""
        score = 50

        type_scores = {
            "usb": 25, "bluetooth": 20, "builtin": 10,
            "unknown": 5, "virtual": -10, "hdmi": -20,
        }
        score += type_scores.get(d.device_type, 0)

        if d.default_sample_rate == 16000:
            score += 15
        elif 16000 < d.default_sample_rate <= 48000:
            score += 10
        elif d.default_sample_rate == 8000:
            score -= 5

        if d.input_latency_ms < 10:
            score += 5
        elif d.input_latency_ms > 50:
            score -= 5

        if d.is_default:
            score += 5

        score -= d.failure_count * 10

        return max(0, min(100, score))

    # ── Ownership Protocol ───────────────────────────────────────────

    @property
    def active_device(self) -> MicDeviceProfile | None:
        return self._active_device

    @active_device.setter
    def active_device(self, device: MicDeviceProfile | None) -> None:
        self._active_device = device

    @property
    def devices(self) -> list[MicDeviceProfile]:
        return self._devices

    @property
    def is_profiled(self) -> bool:
        return self._profiled

    def acquire(self, owner: str, timeout: float = 5.0) -> bool:
        """Acquire microphone ownership, blocking until it's free.

        Returns True if acquired, False if timed out.
        The caller MUST call release() when done with the mic.
        """
        deadline = time.monotonic() + timeout

        with self._condition:
            while self._owner is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning(
                        "Mic acquire timed out for '%s' (held by '%s')",
                        owner, self._owner,
                    )
                    return False
                self._condition.wait(timeout=remaining)

            self._owner = owner
            return True

    def release(self, owner: str) -> None:
        """Release microphone ownership. Only the current owner can release.

        Wakes up any thread waiting in acquire().
        """
        with self._condition:
            if self._owner == owner:
                self._owner = None
                self._condition.notify_all()
            elif self._owner is None:
                pass
            else:
                logger.warning(
                    "Mic release rejected: '%s' tried to release but '%s' owns it",
                    owner, self._owner,
                )

    @property
    def owner(self) -> str | None:
        """Current mic owner name, or None if free. Lock-free read (GIL safe)."""
        return self._owner

    @property
    def is_free(self) -> bool:
        return self._owner is None
