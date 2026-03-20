"""
ATOM v14 -- Smart microphone auto-detection with runtime hot-swap.

Returns ``(device_index, native_sample_rate, max_channels)`` so callers
can open multi-channel streams for proper mono mixing.

Selection strategy:
    1. Last-known good device (cached, zero-scan fast path)
    2. Configured ``device_name`` match
    3. Best-scored device (mic array > USB > HFP BT > A2DP BT)
    4. Windows default input device
    5. Every available input device (last resort)

Runtime features:
    - Periodic device re-scan detects hot-plugged devices
    - Auto-switch when a better device appears (e.g. plug in USB headset)
    - RMS signal quality test filters muted or dead microphones
    - Score-based ranking logged at startup for easy debugging

Bluetooth note:
    Windows exposes TWO endpoints per Bluetooth headset:
      - A2DP (44100Hz) -- stereo media, NOT a real speech mic (near-zero RMS)
      - HFP  (8000-16000Hz) -- actual hands-free speech mic
    This module detects and prefers HFP endpoints automatically.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("atom.mic_selector")

VOSK_RATE = 16_000

_devices_logged = False
_last_good: tuple[int, int] | None = None

_BT_KEYWORDS = ("headset", "hands-free", "bluetooth", "bt", "buds",
                "airpods", "earbuds", "jbl", "bose", "sony", "mivi",
                "oneplus", "realme", "yealink", "blaupunkt", "jabra")

RESCAN_INTERVAL_S = 15.0
_last_scan_time: float = 0.0
_last_device_count: int = -1


def _is_bluetooth(name: str) -> bool:
    low = name.lower()
    return any(kw in low for kw in _BT_KEYWORDS)


def _score_device(idx: int, info: dict) -> int:
    """Score a device for speech recognition suitability (higher = better).

    Scoring priorities:
        100  Built-in microphone array (Intel/Realtek, best audio quality)
         90  USB / analog mic (Realtek line-in)
         70  Bluetooth HFP at 16kHz (usable but noisy)
         40  Generic input device with channels
         30  Bluetooth HFP at 8kHz (very poor for speech recognition)
         10  Bluetooth A2DP endpoint (44100Hz, NOT a real speech input)
          5  Sound Mapper / loopback / stereo mix

    Bonuses:
        +50  Windows system default input (respects OS audio routing)
        +10  sample rate >= 16kHz (better quality)
        +5   mono device (no phase cancellation risk)
    """
    name = info.get("name", "").lower()
    rate = int(info.get("defaultSampleRate", 0))
    ch = int(info.get("maxInputChannels", 0))

    if "sound mapper" in name or "stereo mix" in name or "pc speaker" in name:
        return 5

    if _is_bluetooth(name) or "bthhfenum" in name:
        if rate > 16000:
            return 10
        if rate <= 8000:
            return 30
        return 70

    base = 20
    if "microphone array" in name or "smart sound" in name:
        base = 100
    elif "microphone" in name or "mic" in name:
        base = 90
    elif "usb" in name:
        base = 85
    elif "headset" in name:
        base = 80
    elif ch > 0:
        base = 40

    if rate >= 16000:
        base += 10
    if ch == 1:
        base += 5

    return base


def _list_input_devices(pa: Any) -> list[tuple[int, dict]]:
    devices = []
    for i in range(pa.get_device_count()):
        try:
            info = pa.get_device_info_by_index(i)
        except Exception:
            continue
        if info.get("maxInputChannels", 0) > 0:
            devices.append((i, info))
    return devices


def _log_all_devices(devices: list[tuple[int, dict]]) -> None:
    global _devices_logged
    if _devices_logged:
        return
    _devices_logged = True
    if not devices:
        logger.warning("No input devices found on this system")
        return
    logger.info("=== Microphone scan: %d input devices ===", len(devices))
    for idx, info in devices:
        rate = int(info.get("defaultSampleRate", 0))
        name = info.get("name", "?")
        ch = int(info.get("maxInputChannels", 0))
        score = _score_device(idx, info)
        logger.info("  [%2d] %-45s rate=%5d  ch=%d  score=%3d",
                     idx, name, rate, ch, score)


def _channels(info: dict) -> int:
    return min(int(info.get("maxInputChannels", 1)), 4)


# ── RMS signal quality test ─────────────────────────────────────────

def test_device_rms(pa: Any, device_idx: int, rate: int,
                    channels: int, duration_s: float = 0.3) -> float:
    """Record a brief sample from a device and measure RMS signal level.

    Returns the RMS value (0.0 if device fails to open or is silent).
    Used to detect muted microphones or dead audio endpoints.
    """
    import numpy as np

    chunk = 2048
    frames_needed = max(2, int(duration_s * rate / chunk))

    try:
        stream = pa.open(
            format=8,  # paInt16
            channels=min(channels, 2),
            rate=rate,
            input=True,
            input_device_index=device_idx,
            frames_per_buffer=chunk,
            start=True,
        )
    except Exception as exc:
        logger.debug("RMS test: device [%d] failed to open: %s",
                     device_idx, exc)
        return 0.0

    rms_values: list[float] = []
    try:
        for _ in range(frames_needed):
            raw = stream.read(chunk, exception_on_overflow=False)
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            if len(samples) > 0:
                rms_values.append(
                    float(np.sqrt(np.mean(samples * samples)))
                )
    except Exception as exc:
        logger.debug("RMS test: device [%d] read error: %s", device_idx, exc)
    finally:
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass

    if not rms_values:
        return 0.0

    avg_rms = sum(rms_values) / len(rms_values)
    logger.debug("RMS test: device [%d] avg_rms=%.1f (%d frames)",
                 device_idx, avg_rms, len(rms_values))
    return avg_rms


# ── Cache management ────────────────────────────────────────────────

def mark_good(device_index: int, rate: int) -> None:
    global _last_good
    _last_good = (device_index, rate)


def clear_good() -> None:
    global _last_good
    _last_good = None


# ── Primary device selection ────────────────────────────────────────

def find_device(
    pa: Any,
    device_name: str | None = None,
    exclude_bluetooth: bool = False,
) -> tuple[int | None, int, int]:
    """Return ``(device_index, native_sample_rate, channels)``.

    When exclude_bluetooth is True, only non-Bluetooth devices are considered
    (useful when the system default is a BT headset with low/no usable level).
    """
    global _last_device_count, _last_scan_time

    all_devices = _list_input_devices(pa)
    if exclude_bluetooth:
        all_devices = [(i, d) for i, d in all_devices
                       if not _is_bluetooth(d.get("name", ""))]
        if not all_devices:
            logger.warning("No non-Bluetooth input devices found -- using full list")
            all_devices = _list_input_devices(pa)
    _log_all_devices(all_devices)
    _last_device_count = len(all_devices)
    _last_scan_time = time.monotonic()

    if _last_good is not None:
        idx, rate = _last_good
        for i, info in all_devices:
            if i == idx:
                logger.info("Reusing last-good mic [%d]: %s (rate=%d)",
                             idx, info.get("name", "?"), rate)
                return idx, rate, _channels(info)
        logger.info("Last-good device [%d] disappeared -- rescanning", idx)
        clear_good()

    if device_name:
        needle = device_name.lower()
        matches = [(i, d) for i, d in all_devices
                    if needle in d.get("name", "").lower()]
        if matches:
            scored = [(i, d, _score_device(i, d)) for i, d in matches]
            scored.sort(key=lambda x: x[2], reverse=True)
            best_idx, best_info, best_score = scored[0]
            native_rate = int(best_info.get("defaultSampleRate", VOSK_RATE))
            logger.info("Selected mic [%d]: %s (rate=%d, ch=%d, score=%d)",
                         best_idx, best_info["name"], native_rate,
                         _channels(best_info), best_score)
            return best_idx, native_rate, _channels(best_info)
        logger.warning("No device matching '%s'", device_name)

    if all_devices:
        default_idx = -1
        default_name = "?"
        try:
            di = pa.get_default_input_device_info()
            default_name = di.get("name", "?")
            default_idx = int(di["index"])
        except Exception:
            pass

        scored = []
        for idx, info in all_devices:
            s = _score_device(idx, info)
            if idx == default_idx:
                s += 50
            scored.append((idx, info, s))
        scored.sort(key=lambda x: x[2], reverse=True)
        best_idx, best_info, best_score = scored[0]
        native_rate = int(best_info.get("defaultSampleRate", VOSK_RATE))
        ch = _channels(best_info)

        if best_idx == default_idx:
            logger.info("Using system default mic [%d]: %s "
                        "(rate=%d, ch=%d, score=%d)",
                        best_idx, best_info.get("name", "?"),
                        native_rate, ch, best_score)
        else:
            logger.info("Selected mic [%d] '%s' (score=%d) over default "
                        "[%d] '%s' (rate=%d, ch=%d)",
                        best_idx, best_info.get("name", "?"), best_score,
                        default_idx, default_name, native_rate, ch)
        return best_idx, native_rate, ch

    logger.error("No input devices available at all")
    return None, VOSK_RATE, 1


# ── Runtime hot-swap detection ──────────────────────────────────────

def check_for_better_device(
    pa: Any,
    current_idx: int | None,
) -> tuple[int, dict, int] | None:
    """Re-scan devices and return a better one if available.

    Returns ``(device_index, device_info, score)`` if a higher-scored
    device is found that isn't the current one. Returns None otherwise.

    Throttled to once per RESCAN_INTERVAL_S.
    """
    global _last_scan_time, _last_device_count

    now = time.monotonic()
    if (now - _last_scan_time) < RESCAN_INTERVAL_S:
        return None
    _last_scan_time = now

    all_devices = _list_input_devices(pa)
    new_count = len(all_devices)

    if new_count == _last_device_count and current_idx is not None:
        still_exists = any(i == current_idx for i, _ in all_devices)
        if still_exists:
            return None

    _last_device_count = new_count

    if not all_devices:
        return None

    default_idx = -1
    try:
        di = pa.get_default_input_device_info()
        default_idx = int(di["index"])
    except Exception:
        pass

    current_score = 0
    if current_idx is not None:
        for i, info in all_devices:
            if i == current_idx:
                current_score = _score_device(i, info)
                if i == default_idx:
                    current_score += 50
                break

    scored = []
    for idx, info in all_devices:
        s = _score_device(idx, info)
        if idx == default_idx:
            s += 50
        scored.append((idx, info, s))
    scored.sort(key=lambda x: x[2], reverse=True)
    best_idx, best_info, best_score = scored[0]

    if best_idx == current_idx:
        return None

    if current_idx is not None and not any(
        i == current_idx for i, _ in all_devices
    ):
        logger.warning(
            "Current mic [%s] disconnected! Switching to [%d] '%s' (score=%d)",
            current_idx, best_idx, best_info.get("name", "?"), best_score,
        )
        clear_good()
        return best_idx, best_info, best_score

    if best_score > current_score + 15:
        logger.info(
            "Better mic detected: [%d] '%s' (score=%d) > current [%s] (score=%d)",
            best_idx, best_info.get("name", "?"), best_score,
            current_idx, current_score,
        )
        return best_idx, best_info, best_score

    return None
