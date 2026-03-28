"""
ATOM -- Command Probability Filter (Bilingual: English + Hindi).

Scores STT output to reject garbage speech before it reaches the Intent Engine.
Combines STT confidence with command keyword detection and length heuristics.
Supports both English and Hindi command keywords.

Pipeline position:
    AudioPreprocessor -> Whisper STT -> Text Corrections
    -> **Command Filter** -> Intent Engine -> Router
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("atom.cmd_filter")

COMMAND_KEYWORDS_EN = frozenset({
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

COMMAND_KEYWORDS_HI = frozenset({
    # Action verbs (Hindi)
    "kholo", "band", "karo", "chalu", "shuru", "bando", "dikhao",
    "batao", "chalao", "bajao", "ruko", "dhoondo", "banao",
    "hatao", "bhejo", "padho", "likho", "suno", "dekho",
    "nikalo", "daalo", "badlo", "ghatao", "badhao",
    # System (Hindi)
    "system", "battery", "screen", "awaaz", "volume",
    "time", "samay", "waqt", "tareekh", "din",
    # Greetings (Hindi)
    "namaste", "namaskar", "shukriya", "dhanyavaad", "alvida",
    "suprabhat", "shubh", "ratri",
    # ATOM-related
    "atom", "boss",
    # Confirm/deny (Hindi)
    "haan", "nahi", "theek", "sahi", "galat", "ruko", "chalo",
    # Queries (Hindi)
    "kya", "kaise", "kab", "kahan", "kaun", "kitna", "kyun",
    "mausam", "khabar", "news",
})

COMMAND_KEYWORDS = COMMAND_KEYWORDS_EN | COMMAND_KEYWORDS_HI

_HINDI_SCRIPT_PATTERN = re.compile(r'[\u0900-\u097F]')

MIN_SCORE = 0.35


def contains_hindi(text: str) -> bool:
    """Check if text contains Devanagari characters (native Hindi script)."""
    return bool(_HINDI_SCRIPT_PATTERN.search(text))


def detect_language_heuristic(text: str) -> str:
    """Quick heuristic language detection from text content.

    Returns 'hi' for Hindi, 'en' for English. Useful as a secondary
    signal alongside whisper's language detection.
    """
    if contains_hindi(text):
        return "hi"
    words = text.lower().split()
    hi_hits = sum(1 for w in words if w in COMMAND_KEYWORDS_HI)
    en_hits = sum(1 for w in words if w in COMMAND_KEYWORDS_EN)
    if hi_hits > en_hits and hi_hits >= 2:
        return "hi"
    return "en"


def command_probability(text: str, confidence: float) -> float:
    """Score a phrase for command likelihood (0.0 to 1.0).

    Checks both English and Hindi keywords. Higher scores mean
    more likely to be a real command.
    """
    words = text.lower().split()
    word_count = len(words)

    if word_count == 0:
        return 0.0

    score = confidence

    if word_count == 1:
        score *= 0.6
    elif word_count > 8:
        score *= 0.5

    keyword_hits = sum(1 for w in words if w in COMMAND_KEYWORDS)
    score += keyword_hits * 0.15

    if contains_hindi(text):
        score += 0.1

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
