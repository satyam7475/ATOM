"""
ATOM Intent Engine -- Base types and shared utilities.
"""

from __future__ import annotations

import json
import re


class IntentResult:
    __slots__ = ("intent", "response", "action", "action_args")

    def __init__(self, intent: str, response: str | None = None,
                 action: str | None = None, action_args: dict | None = None):
        self.intent = intent
        self.response = response
        self.action = action
        self.action_args = action_args


def clean_slot(value: str | None) -> str:
    """Sanitise a regex group capture into a clean string."""
    if not value:
        return ""
    v = value.strip()
    v = re.sub(r"^[\"']|[\"']$", "", v)
    v = re.sub(r"[\s.!,]+$", "", v)
    return v




# ── Command vocabulary for intent matching ────────────────────────────

GRAMMAR_WORDS: list[str] = sorted(set([
    "atom", "hey", "boss", "buddy",
    "open", "close", "launch", "start", "run", "stop", "kill", "play",
    "pause", "search", "check", "create", "make", "move", "copy",
    "list", "show", "set", "increase", "decrease", "find", "tell",
    "lock", "mute", "unmute", "duplicate", "shift", "take", "read",
    "minimize", "maximize", "switch", "restart", "reboot", "sleep",
    "timer", "remind", "calculate", "solve", "type", "write",
    "empty", "clear", "flush", "navigate", "visit", "dim",
    "hi", "hello", "howdy", "bye", "goodbye", "shutdown", "quit", "exit",
    "silent", "quiet", "shh", "hush", "enough", "rest",
    "thanks", "thank", "you", "namaste", "good", "morning", "evening",
    "afternoon", "night", "great", "nice", "well", "done", "job",
    "sup", "yo", "hola",
    "what", "what's", "how", "is", "the", "it", "my", "me", "a", "an",
    "are", "can", "do", "to", "on", "in", "at", "of", "for", "up",
    "down", "about", "there", "here", "this", "that", "much",
    "time", "date", "day", "today", "current", "now", "please",
    "cpu", "ram", "memory", "battery", "disk", "storage", "space",
    "system", "status", "usage", "info", "health", "load", "report",
    "ip", "address", "network", "charging", "power", "processor",
    "level", "percent", "percentage", "free", "computer", "laptop",
    "temperature", "temp", "uptime", "process", "running", "wifi",
    "connected", "internet", "speed", "weather", "forecast",
    "brightness", "brighter", "screen", "capture", "clipboard",
    "recycle", "bin", "trash", "dns",
    "volume", "sound", "full", "half", "maximum", "max", "minimum",
    "music", "song", "audio", "youtube", "silent",
    "chrome", "notepad", "teams", "outlook", "edge", "firefox",
    "calculator", "calc", "explorer", "terminal", "settings",
    "word", "excel", "powerpoint", "vscode", "cursor", "paint",
    "spotify", "slack", "postman", "browser", "intellij",
    "task", "manager", "control", "panel", "snipping", "tool",
    "screenshot", "discord", "zoom", "docker", "whatsapp",
    "telegram", "brave", "downloads", "documents", "desktop",
    "device", "registry", "event", "viewer",
    "folder", "file", "named", "called", "apps", "applications",
    "installed", "all", "new",
    "yes", "no", "cancel", "confirm",
    "second", "seconds", "minute", "minutes", "hour", "hours",
    "plus", "minus", "times", "divided",
    "kholo", "khol", "band", "karo", "kar", "chalao", "bajao",
    "haan", "ha", "han", "nahi", "mat", "awaaz", "chalu",
    "chup", "ruk", "ja", "bas", "do", "pe", "par", "le", "aao",
    "shukriya", "dhanyavaad", "tala", "lagao", "chhota", "bada",
    "neeche", "dusri", "badha", "kam", "mausam",
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "fifteen", "twenty", "twenty five",
    "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
    "ninety", "hundred",
    "google", "microsoft", "visual", "studio", "code",
    "command", "prompt", "shell",
    "log", "off", "sign", "out", "hibernate", "reboot",
    "remind", "reminder", "reminders", "pending", "cancel",
    "research", "investigate", "diagnostic", "diagnostics", "diagnose",
    "evolve", "evolution", "improve", "health",
    "function", "functions", "module", "modules", "test", "yourself",
    "working", "fine", "okay", "sab",
    "resource", "trend", "report", "summary", "detailed",
    "history", "force", "terminate", "end",
    "listening", "ready", "alive", "awake", "hear",
    "online", "offline", "window",
    "[unk]",
]))


def get_grammar_json() -> str:
    """Return JSON string of recognized command vocabulary."""
    return json.dumps(GRAMMAR_WORDS)
