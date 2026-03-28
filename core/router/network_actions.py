"""
ATOM -- Network and web action handlers.

Handles: open_url, weather, search
"""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger("atom.router.network")


def open_url(url: str) -> None:
    subprocess.Popen(["cmd", "/c", "start", url],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logger.info("Opened URL: %s", url)


def web_search(url: str) -> None:
    subprocess.Popen(["cmd", "/c", "start", url],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
    subprocess.Popen(
        ["cmd", "/c", "start",
         "https://www.google.com/search?q=weather"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def get_wifi_status() -> str:
    """Get WiFi connection info via netsh. Returns human-readable string."""
    try:
        proc = subprocess.run(
            ["netsh", "wlan", "show", "interfaces"],
            capture_output=True, text=True, timeout=3,
        )
        output = proc.stdout
        ssid = ""
        signal = ""
        state = ""
        for line in output.splitlines():
            line = line.strip()
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
