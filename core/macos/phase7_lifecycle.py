"""Phase 7.2 — small, testable macOS lifecycle helpers (audio route text parse, boot time).

Full AirPods / sleep-wake / ``memorypressure`` trials are manual; this module supports
automated smoke that still reads real system state where cheap (no sudo).
"""

from __future__ import annotations

import re
import subprocess
from typing import Any, Optional


def parse_default_system_output_device(spaudio_text: str) -> Optional[str]:
    """Parse ``system_profiler SPAudioDataType`` stdout for the default system output device.

    Walks upward from the line ``Default System Output Device: Yes`` to the nearest
    device block title (``Some Name:``).
    """
    lines = spaudio_text.replace("\r\n", "\n").splitlines()
    skip_titles = {"devices", "audio", "default input device", "default output device"}

    for i, raw in enumerate(lines):
        if "Default System Output Device:" in raw and "Yes" in raw:
            for j in range(i - 1, max(-1, i - 24), -1):
                t = lines[j].strip()
                if not t.endswith(":") or len(t) < 2:
                    continue
                name = t[:-1].strip()
                low = name.lower()
                if low in skip_titles:
                    continue
                if name:
                    return name
    return None


def fetch_default_audio_output_label(timeout_s: float = 25.0) -> Optional[str]:
    try:
        r = subprocess.run(
            ["system_profiler", "SPAudioDataType"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if r.returncode != 0 or not (r.stdout or "").strip():
            return None
        return parse_default_system_output_device(r.stdout)
    except (OSError, subprocess.TimeoutExpired):
        return None


def read_kern_boottime_dict(timeout_s: float = 3.0) -> Optional[dict[str, Any]]:
    """Parse ``sysctl kern.boottime`` into ``{"sec": int, "usec": int}`` if present."""
    try:
        r = subprocess.run(
            ["sysctl", "-n", "kern.boottime"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if r.returncode != 0:
            return None
        m = re.search(r"sec\s*=\s*(\d+),\s*usec\s*=\s*(\d+)", r.stdout or "")
        if not m:
            return None
        return {"sec": int(m.group(1)), "usec": int(m.group(2))}
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


__all__ = [
    "parse_default_system_output_device",
    "fetch_default_audio_output_label",
    "read_kern_boottime_dict",
]
