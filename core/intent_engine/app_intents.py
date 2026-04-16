"""
ATOM Intent Engine -- App intents (open_app, close_app, list_apps, search).
"""

from __future__ import annotations

import re
import urllib.parse

from core import adaptive_personality as personality
from .base import IntentResult

_OPEN_APP = re.compile(
    r"\b(open|launch|start|run|khol|kholo|chalu\s+karo)\s+(.+)", re.I)

_CLOSE_APP = re.compile(
    r"\b(close|kill|stop|end|quit|band\s+karo|band\s+kar)\s+(.+)", re.I)

_LIST_APPS = re.compile(
    r"\b(list|show)\s+(all\s+)?(apps|applications|installed\s+apps)\b", re.I)

_SEARCH_WEB = re.compile(
    r"\b(search|google|look\s+up|find\s+online)\s+(.+)", re.I)

import platform as _platform

_IS_MACOS = _platform.system() == "Darwin"

def _app(win_exe: str, win_args: list[str], mac_app: str = "") -> dict:
    """Return platform-appropriate app spec."""
    if _IS_MACOS and mac_app:
        return {"exe": "open", "args": ["-a", mac_app]}
    return {"exe": win_exe, "args": win_args}

APP_MAP: dict[str, dict] = {
    "chrome": _app("cmd", ["/c", "start", "chrome"], "Google Chrome"),
    "google chrome": _app("cmd", ["/c", "start", "chrome"], "Google Chrome"),
    "browser": _app("cmd", ["/c", "start", "https://www.google.com"], "Safari"),
    "safari": _app("cmd", ["/c", "start", "https://www.google.com"], "Safari"),
    "edge": _app("cmd", ["/c", "start", "msedge"], "Microsoft Edge"),
    "microsoft edge": _app("cmd", ["/c", "start", "msedge"], "Microsoft Edge"),
    "firefox": _app("cmd", ["/c", "start", "firefox"], "Firefox"),
    "brave": _app("cmd", ["/c", "start", "brave"], "Brave Browser"),
    "notepad": _app("notepad.exe", [], "TextEdit"),
    "notes": _app("notepad.exe", [], "Notes"),
    "calculator": _app("calc.exe", [], "Calculator"),
    "calc": _app("calc.exe", [], "Calculator"),
    "explorer": _app("explorer.exe", [], "Finder"),
    "file explorer": _app("explorer.exe", [], "Finder"),
    "finder": _app("explorer.exe", [], "Finder"),
    "files": _app("explorer.exe", [], "Finder"),
    "downloads": _app("explorer.exe", ["shell:Downloads"], "Finder"),
    "documents": _app("explorer.exe", ["shell:Personal"], "Finder"),
    "desktop": _app("explorer.exe", ["shell:Desktop"], "Finder"),
    "terminal": _app("wt.exe", [], "Terminal"),
    "cmd": _app("cmd", ["/c", "start", "cmd"], "Terminal"),
    "command prompt": _app("cmd", ["/c", "start", "cmd"], "Terminal"),
    "powershell": _app("cmd", ["/c", "start", "powershell"], "Terminal"),
    "task manager": _app("taskmgr.exe", [], "Activity Monitor"),
    "activity monitor": _app("taskmgr.exe", [], "Activity Monitor"),
    "settings": _app("cmd", ["/c", "start", "ms-settings:"], "System Preferences"),
    "system preferences": _app("cmd", ["/c", "start", "ms-settings:"], "System Preferences"),
    "system settings": _app("cmd", ["/c", "start", "ms-settings:"], "System Preferences"),
    "mail": _app("cmd", ["/c", "start", "outlook"], "Mail"),
    "email": _app("cmd", ["/c", "start", "outlook"], "Mail"),
    "outlook": _app("cmd", ["/c", "start", "outlook"], "Microsoft Outlook"),
    "messages": _app("cmd", ["/c", "start", "cmd"], "Messages"),
    "facetime": _app("cmd", ["/c", "start", "cmd"], "FaceTime"),
    "teams": _app("cmd", ["/c", "start", "msteams:"], "Microsoft Teams"),
    "microsoft teams": _app("cmd", ["/c", "start", "msteams:"], "Microsoft Teams"),
    "word": _app("cmd", ["/c", "start", "winword"], "Microsoft Word"),
    "excel": _app("cmd", ["/c", "start", "excel"], "Microsoft Excel"),
    "powerpoint": _app("cmd", ["/c", "start", "powerpnt"], "Microsoft PowerPoint"),
    "onenote": _app("cmd", ["/c", "start", "onenote:"], "Microsoft OneNote"),
    "pages": _app("cmd", ["/c", "start", "cmd"], "Pages"),
    "numbers": _app("cmd", ["/c", "start", "cmd"], "Numbers"),
    "keynote": _app("cmd", ["/c", "start", "cmd"], "Keynote"),
    "vscode": _app("cmd", ["/c", "start", "code"], "Visual Studio Code"),
    "vs code": _app("cmd", ["/c", "start", "code"], "Visual Studio Code"),
    "visual studio code": _app("cmd", ["/c", "start", "code"], "Visual Studio Code"),
    "cursor": _app("cmd", ["/c", "start", "cursor"], "Cursor"),
    "xcode": _app("cmd", ["/c", "start", "cmd"], "Xcode"),
    "spotify": _app("cmd", ["/c", "start", "spotify:"], "Spotify"),
    "slack": _app("cmd", ["/c", "start", "slack:"], "Slack"),
    "discord": _app("cmd", ["/c", "start", "discord:"], "Discord"),
    "zoom": _app("cmd", ["/c", "start", "zoommtg:"], "zoom.us"),
    "postman": _app("cmd", ["/c", "start", "postman"], "Postman"),
    "intellij": _app("cmd", ["/c", "start", "idea64"], "IntelliJ IDEA"),
    "idea": _app("cmd", ["/c", "start", "idea64"], "IntelliJ IDEA"),
    "docker": _app("cmd", ["/c", "start", "docker"], "Docker"),
    "whatsapp": _app("cmd", ["/c", "start", "whatsapp:"], "WhatsApp"),
    "telegram": _app("cmd", ["/c", "start", "tg:"], "Telegram"),
    "photos": _app("cmd", ["/c", "start", "cmd"], "Photos"),
    "music": _app("cmd", ["/c", "start", "cmd"], "Music"),
    "preview": _app("cmd", ["/c", "start", "cmd"], "Preview"),
    "calendar": _app("cmd", ["/c", "start", "cmd"], "Calendar"),
    "reminders": _app("cmd", ["/c", "start", "cmd"], "Reminders"),
    "maps": _app("cmd", ["/c", "start", "cmd"], "Maps"),
}

def _close(win_proc: str, mac_app: str = "") -> str:
    """Return platform-appropriate process/app name for closing."""
    if _IS_MACOS and mac_app:
        return mac_app
    return win_proc

CLOSE_MAP: dict[str, str] = {
    "chrome": _close("chrome.exe", "Google Chrome"),
    "google chrome": _close("chrome.exe", "Google Chrome"),
    "safari": _close("", "Safari"),
    "edge": _close("msedge.exe", "Microsoft Edge"),
    "microsoft edge": _close("msedge.exe", "Microsoft Edge"),
    "firefox": _close("firefox.exe", "Firefox"),
    "brave": _close("brave.exe", "Brave Browser"),
    "notepad": _close("notepad.exe", "TextEdit"),
    "notes": _close("", "Notes"),
    "calculator": _close("CalculatorApp.exe", "Calculator"),
    "calc": _close("CalculatorApp.exe", "Calculator"),
    "finder": _close("explorer.exe", "Finder"),
    "explorer": _close("explorer.exe", "Finder"),
    "mail": _close("", "Mail"),
    "email": _close("", "Mail"),
    "outlook": _close("OUTLOOK.EXE", "Microsoft Outlook"),
    "teams": _close("ms-teams.exe", "Microsoft Teams"),
    "word": _close("WINWORD.EXE", "Microsoft Word"),
    "excel": _close("EXCEL.EXE", "Microsoft Excel"),
    "powerpoint": _close("POWERPNT.EXE", "Microsoft PowerPoint"),
    "spotify": _close("Spotify.exe", "Spotify"),
    "slack": _close("slack.exe", "Slack"),
    "discord": _close("Discord.exe", "Discord"),
    "zoom": _close("Zoom.exe", "zoom.us"),
    "postman": _close("Postman.exe", "Postman"),
    "vscode": _close("Code.exe", "Visual Studio Code"),
    "vs code": _close("Code.exe", "Visual Studio Code"),
    "cursor": _close("Cursor.exe", "Cursor"),
    "docker": _close("Docker Desktop.exe", "Docker"),
    "whatsapp": _close("WhatsApp.exe", "WhatsApp"),
    "telegram": _close("Telegram.exe", "Telegram"),
}


def check(text: str) -> IntentResult | None:
    m = _OPEN_APP.search(text)
    if m:
        app_name = re.sub(r"[\s!.]+$", "", m.group(2).strip().lower())
        app_name = re.sub(r"^(my|the|a)\s+", "", app_name)
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
