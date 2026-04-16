"""
ATOM -- Network and web action handlers (cross-platform).

Handles: open_url, weather, search, wifi status
"""

from __future__ import annotations

import logging
import subprocess
import sys

logger = logging.getLogger("atom.router.network")

_IS_MAC = sys.platform == "darwin"
_IS_WIN = sys.platform == "win32"


def _open_url_platform(url: str) -> None:
    """Open a URL in the default browser, platform-aware."""
    if _IS_MAC:
        subprocess.Popen(["open", url],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif _IS_WIN:
        subprocess.Popen(["cmd", "/c", "start", url],
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
    """Get WiFi connection info. macOS uses networksetup, Windows uses netsh."""
    if _IS_MAC:
        return _get_wifi_macos()
    elif _IS_WIN:
        return _get_wifi_windows()
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


def _get_wifi_windows() -> str:
    try:
        proc = subprocess.run(
            ["netsh", "wlan", "show", "interfaces"],
            capture_output=True, text=True, timeout=3,
        )
        output = proc.stdout
        ssid = ""
        signal = ""
        state = ""
        for raw_line in output.splitlines():
            line = raw_line.strip()
            if line.startswith("SSID") and "BSSID" not in line:
                ssid = line.split(":", 1)[-1].strip()
            elif line.startswith("Signal"):
                signal = line.split(":", 1)[-1].strip()
            elif line.startswith("State"):
                state = line.split(":", 1)[-1].strip()
        if ssid:
            return f"Connected to {ssid}, signal strength {signal}."
        if state:
            return f"WiFi state: {state}."
        return "No WiFi connection detected."
    except Exception:
        return "Couldn't check WiFi status."
