"""
ATOM Intent Engine -- App intents (open_app, close_app, list_apps, search).
"""

from __future__ import annotations

import re
import urllib.parse

from core import personality
from .base import IntentResult

_OPEN_APP = re.compile(
    r"\b(open|launch|start|run|khol|kholo|chalu\s+karo)\s+(.+)", re.I)

_CLOSE_APP = re.compile(
    r"\b(close|kill|stop|end|quit|band\s+karo|band\s+kar)\s+(.+)", re.I)

_LIST_APPS = re.compile(
    r"\b(list|show)\s+(all\s+)?(apps|applications|installed\s+apps)\b", re.I)

_SEARCH_WEB = re.compile(
    r"\b(search|google|look\s+up|find\s+online)\s+(.+)", re.I)

APP_MAP: dict[str, dict] = {
    "chrome": {"exe": "cmd", "args": ["/c", "start", "chrome"]},
    "google chrome": {"exe": "cmd", "args": ["/c", "start", "chrome"]},
    "browser": {"exe": "cmd", "args": ["/c", "start", "https://www.google.com"]},
    "edge": {"exe": "cmd", "args": ["/c", "start", "msedge"]},
    "microsoft edge": {"exe": "cmd", "args": ["/c", "start", "msedge"]},
    "firefox": {"exe": "cmd", "args": ["/c", "start", "firefox"]},
    "brave": {"exe": "cmd", "args": ["/c", "start", "brave"]},
    "notepad": {"exe": "notepad.exe", "args": []},
    "notepad++": {"exe": "cmd", "args": ["/c", "start", "notepad++"]},
    "calculator": {"exe": "calc.exe", "args": []},
    "calc": {"exe": "calc.exe", "args": []},
    "explorer": {"exe": "explorer.exe", "args": []},
    "file explorer": {"exe": "explorer.exe", "args": []},
    "files": {"exe": "explorer.exe", "args": []},
    "downloads": {"exe": "explorer.exe", "args": ["shell:Downloads"]},
    "documents": {"exe": "explorer.exe", "args": ["shell:Personal"]},
    "desktop": {"exe": "explorer.exe", "args": ["shell:Desktop"]},
    "terminal": {"exe": "wt.exe", "args": []},
    "cmd": {"exe": "cmd", "args": ["/c", "start", "cmd"]},
    "command prompt": {"exe": "cmd", "args": ["/c", "start", "cmd"]},
    "powershell": {"exe": "cmd", "args": ["/c", "start", "powershell"]},
    "task manager": {"exe": "taskmgr.exe", "args": []},
    "settings": {"exe": "cmd", "args": ["/c", "start", "ms-settings:"]},
    "control panel": {"exe": "control.exe", "args": []},
    "device manager": {"exe": "devmgmt.msc", "args": []},
    "disk management": {"exe": "diskmgmt.msc", "args": []},
    "event viewer": {"exe": "eventvwr.msc", "args": []},
    "registry editor": {"exe": "regedit.exe", "args": []},
    "outlook": {"exe": "cmd", "args": ["/c", "start", "outlook"]},
    "teams": {"exe": "cmd", "args": ["/c", "start", "msteams:"]},
    "microsoft teams": {"exe": "cmd", "args": ["/c", "start", "msteams:"]},
    "word": {"exe": "cmd", "args": ["/c", "start", "winword"]},
    "excel": {"exe": "cmd", "args": ["/c", "start", "excel"]},
    "powerpoint": {"exe": "cmd", "args": ["/c", "start", "powerpnt"]},
    "onenote": {"exe": "cmd", "args": ["/c", "start", "onenote:"]},
    "vscode": {"exe": "cmd", "args": ["/c", "start", "code"]},
    "vs code": {"exe": "cmd", "args": ["/c", "start", "code"]},
    "visual studio code": {"exe": "cmd", "args": ["/c", "start", "code"]},
    "cursor": {"exe": "cmd", "args": ["/c", "start", "cursor"]},
    "snipping tool": {"exe": "snippingtool.exe", "args": []},
    "screenshot": {"exe": "snippingtool.exe", "args": []},
    "paint": {"exe": "mspaint.exe", "args": []},
    "spotify": {"exe": "cmd", "args": ["/c", "start", "spotify:"]},
    "slack": {"exe": "cmd", "args": ["/c", "start", "slack:"]},
    "discord": {"exe": "cmd", "args": ["/c", "start", "discord:"]},
    "zoom": {"exe": "cmd", "args": ["/c", "start", "zoommtg:"]},
    "postman": {"exe": "cmd", "args": ["/c", "start", "postman"]},
    "intellij": {"exe": "cmd", "args": ["/c", "start", "idea64"]},
    "idea": {"exe": "cmd", "args": ["/c", "start", "idea64"]},
    "git bash": {"exe": "cmd", "args": ["/c", "start", "git-bash"]},
    "docker": {"exe": "cmd", "args": ["/c", "start", "docker"]},
    "whatsapp": {"exe": "cmd", "args": ["/c", "start", "whatsapp:"]},
    "telegram": {"exe": "cmd", "args": ["/c", "start", "tg:"]},
}

CLOSE_MAP: dict[str, str] = {
    "chrome": "chrome.exe",
    "google chrome": "chrome.exe",
    "edge": "msedge.exe",
    "microsoft edge": "msedge.exe",
    "firefox": "firefox.exe",
    "brave": "brave.exe",
    "notepad": "notepad.exe",
    "notepad++": "notepad++.exe",
    "calculator": "CalculatorApp.exe",
    "calc": "CalculatorApp.exe",
    "explorer": "explorer.exe",
    "outlook": "OUTLOOK.EXE",
    "teams": "ms-teams.exe",
    "word": "WINWORD.EXE",
    "excel": "EXCEL.EXE",
    "powerpoint": "POWERPNT.EXE",
    "spotify": "Spotify.exe",
    "slack": "slack.exe",
    "discord": "Discord.exe",
    "zoom": "Zoom.exe",
    "postman": "Postman.exe",
    "vscode": "Code.exe",
    "vs code": "Code.exe",
    "cursor": "Cursor.exe",
    "docker": "Docker Desktop.exe",
    "whatsapp": "WhatsApp.exe",
    "telegram": "Telegram.exe",
}


def check(text: str) -> IntentResult | None:
    m = _OPEN_APP.search(text)
    if m:
        app_name = re.sub(r"[\s!.]+$", "", m.group(2).strip().lower())
        spec = APP_MAP.get(app_name)
        if spec:
            args = {**spec, "name": app_name}
            return IntentResult("open_app",
                                response=personality.action_done("open_app", app_name),
                                action="open_app", action_args=args)

    m = _CLOSE_APP.search(text)
    if m:
        app_name = re.sub(r"[\s!.]+$", "", m.group(2).strip().lower())
        proc = CLOSE_MAP.get(app_name)
        if proc:
            return IntentResult("close_app",
                                response=f"Closing {app_name}.",
                                action="close_app",
                                action_args={"process": proc, "name": app_name})

    if _LIST_APPS.search(text):
        return IntentResult("list_apps", response="Listing installed apps.",
                            action="list_apps", action_args={})

    m = _SEARCH_WEB.search(text)
    if m:
        query = m.group(2).strip()
        query = re.sub(r"^(for|about|on|up)\s+", "", query, flags=re.I).strip()
        if query:
            url_query = urllib.parse.quote_plus(query)
            return IntentResult("search", response=f"Searching for {query}.",
                                action="search",
                                action_args={"url": f"https://www.google.com/search?q={url_query}"})
    return None


def quick_match(text: str) -> str | None:
    """Fast check for app intents used by STT early-exit."""
    m = _OPEN_APP.search(text)
    if m:
        app = re.sub(r"[\s!.]+$", "", m.group(2).strip().lower())
        if app in APP_MAP:
            return "open_app"
    m = _CLOSE_APP.search(text)
    if m:
        app = re.sub(r"[\s!.]+$", "", m.group(2).strip().lower())
        if app in CLOSE_MAP:
            return "close_app"
    return None
