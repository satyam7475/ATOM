"""
ATOM v14 -- Command Probability Filter.

Scores STT output to reject garbage speech before it reaches the Intent Engine.
Combines Vosk confidence with command keyword detection and length heuristics.

Pipeline position:
    Vosk -> Text Corrections -> **Command Filter** -> Intent Engine -> Router
"""

from __future__ import annotations

import logging

logger = logging.getLogger("atom.cmd_filter")

COMMAND_KEYWORDS = frozenset({
    # Action verbs
    "open", "close", "launch", "start", "run", "stop", "kill",
    "shutdown", "restart", "check", "show", "tell", "play",
    "pause", "search", "create", "make", "move", "copy",
    "list", "set", "increase", "decrease", "find", "mute", "unmute",
    "lock", "duplicate", "take", "screenshot", "capture",
    "minimize", "maximize", "switch", "timer", "remind",
    "calculate", "solve", "read", "clipboard", "empty", "flush",
    "navigate", "visit", "sleep", "reboot", "logoff", "hibernate",
    "brightness", "dim", "brighter",
    # App names
    "chrome", "cursor", "teams", "outlook", "notepad", "edge",
    "firefox", "calculator", "calc", "explorer", "terminal",
    "vscode", "excel", "word", "powerpoint", "paint", "spotify",
    "slack", "postman", "intellij", "settings", "discord",
    "zoom", "docker", "whatsapp", "telegram", "brave",
    # System info
    "cpu", "memory", "ram", "battery", "disk", "storage",
    "system", "status", "usage", "info", "health", "temperature",
    "ip", "address", "network", "wifi", "internet", "connected",
    "uptime", "process", "running", "weather", "forecast",
    "recycle", "bin", "dns", "screen",
    # Media / volume
    "volume", "sound", "music", "song", "youtube", "silent",
    # Time / date
    "time", "date", "day", "today",
    # Greetings / control
    "hello", "hi", "hey", "atom", "good", "morning", "evening",
    "afternoon", "bye", "goodbye", "quit", "exit",
    # Confirm / deny
    "yes", "no", "cancel", "confirm",
    # File operations
    "folder", "file", "apps", "applications",
})

MIN_SCORE = 0.35


def command_probability(text: str, confidence: float) -> float:
    """Score a phrase for command likelihood (0.0 to 1.0).

    Higher scores mean more likely to be a real command.
    """
    words = text.lower().split()
    word_count = len(words)

    if word_count == 0:
        return 0.0

    score = confidence

    if word_count == 1:
        score *= 0.6
    elif word_count > 5:
        score *= 0.5

    keyword_hits = sum(1 for w in words if w in COMMAND_KEYWORDS)
    score += keyword_hits * 0.15

    return min(score, 1.0)


def is_valid_command(text: str, confidence: float) -> bool:
    """Return True if the phrase passes the command probability threshold."""
    score = command_probability(text, confidence)
    if score < MIN_SCORE:
        logger.info("Command filter REJECTED: '%.40s' (score=%.2f, conf=%.2f)",
                     text, score, confidence)
        return False
    logger.debug("Command filter PASSED: '%.40s' (score=%.2f, conf=%.2f)",
                  text, score, confidence)
    return True
