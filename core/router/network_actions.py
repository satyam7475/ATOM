"""ATOM -- Network and web action handlers (macOS-only).

Handles: open_url, weather, search, wifi status.

Sprint P4.7 (Apr 26 2026): Windows ``netsh`` / ``cmd start`` branches
removed. The Linux ``xdg-open`` fallback is retained because it's
trivial and lets headless CI / dev boxes open weather URLs in
default browsers without crashing. See
``docs/ATOM_NEXT_STEPS_PLAN.md`` § P4.7.
"""

from __future__ import annotations

import logging
import subprocess
import sys

logger = logging.getLogger("atom.router.network")

_IS_MAC = sys.platform == "darwin"


def _open_url_platform(url: str) -> None:
    """Open a URL in the default browser. macOS uses ``open``; the
    Linux ``xdg-open`` fallback keeps headless CI / dev boxes happy."""
    if _IS_MAC:
        subprocess.Popen(["open", url],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.Popen(["xdg-open", url],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def open_url(url: str) -> None:
    _open_url_platform(url)
    logger.info("Opened URL: %s", url)


def web_search(url: str) -> None:
    _open_url_platform(url)
    logger.info("Web search: %s", url)


def get_weather() -> str | None:
    """Fetch weather from wttr.in. Returns text or None on failure."""
    try:
        import urllib.request
        req = urllib.request.urlopen("https://wttr.in/?format=3", timeout=3)
        return req.read().decode("utf-8").strip()
    except Exception:
        return None


def open_weather_fallback() -> None:
    _open_url_platform("https://www.google.com/search?q=weather")


def get_wifi_status() -> str:
    """Get WiFi connection info via ``networksetup`` (macOS-native)."""
    if _IS_MAC:
        return _get_wifi_macos()
    return "WiFi status not supported on this platform."


def _get_wifi_macos() -> str:
    try:
        proc = subprocess.run(
            ["networksetup", "-getairportnetwork", "en0"],
            capture_output=True, text=True, timeout=3,
        )
        line = proc.stdout.strip()
        if "You are not associated" in line:
            return "WiFi is not connected."
        if "Current Wi-Fi Network:" in line:
            ssid = line.split(":", 1)[-1].strip()
            return f"Connected to {ssid}."
        return line or "Couldn't determine WiFi status."
    except Exception:
        return "Couldn't check WiFi status."
