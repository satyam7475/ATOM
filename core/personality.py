"""
ATOM v14 -- Vision/JARVIS Personality Layer.

Warm, composed, empathetic -- a fusion of Vision's gentle presence
and JARVIS's sharp British precision.  All functions return a single
string chosen at random so ATOM never sounds repetitive.
Zero latency -- just string selection.

Owner: Satyam (configurable via owner.name in settings.json).
"""

from __future__ import annotations

import datetime
import random

# Injected at startup from config; used for personalized responses.
_OWNER_NAME = "Satyam"
_OWNER_TITLE = "Boss"


def set_owner(name: str = "Satyam", title: str = "Boss") -> None:
    """Set owner identity (called from main at startup)."""
    global _OWNER_NAME, _OWNER_TITLE
    _OWNER_NAME = name or "Satyam"
    _OWNER_TITLE = title or "Boss"


def _owner() -> str:
    return _OWNER_NAME


def _boss() -> str:
    return _OWNER_TITLE

# ── Greetings (time-aware) ───────────────────────────────────────────

def _morning() -> list[str]:
    o, b = _owner(), _boss()
    return [
        f"Good morning, {b}. I hope you rested well. Ready whenever you are.",
        f"Morning, {b}. A fresh day, full of possibilities. What shall we tackle?",
        "Good morning. I've been here, waiting. Let's make today count.",
        f"Morning, {b}. All systems are with you. What do you need?",
        f"Good morning, {o}. It's good to hear your voice again.",
    ]

def _afternoon() -> list[str]:
    o, b = _owner(), _boss()
    return [
        f"Good afternoon, {b}. How can I be of service?",
        f"Afternoon, {b}. I'm right here. What do you need?",
        "Good afternoon. Steady progress today. What's next?",
        f"Afternoon, {b}. I'm with you. Just say the word.",
        f"Good afternoon, {o}. What can I do for you?",
    ]

def _evening() -> list[str]:
    o, b = _owner(), _boss()
    return [
        f"Good evening, {b}. Still going? I admire your dedication.",
        f"Evening, {b}. I'm here as always. What do you need?",
        f"Good evening, {o}. The day isn't over until you say it is.",
        f"Evening. I'm right beside you, {b}. How can I help?",
        f"Good evening, {b}. Let's finish strong.",
    ]

def _night() -> list[str]:
    o, b = _owner(), _boss()
    return [
        f"Still here, {b}? Then so am I. What do you need?",
        "Late hours, but I'm wide awake for you. Go ahead.",
        f"The world sleeps, {b}, but I'm watching over things. What can I do?",
        f"Night session. I'm with you all the way, {b}.",
        f"It's late, {o}. Take care of yourself, but I'm here if you need me.",
    ]

def _general_greeting() -> list[str]:
    o, b = _owner(), _boss()
    return [
        f"Hello, {b}. I'm here. What can I do for you?",
        "Hello. Ready and waiting, as always.",
        f"I'm listening, {b}. Go ahead.",
        f"At your service, {b}. What do you need?",
        f"Hello, {o}. It's good to hear from you.",
        f"I'm here, {b}. Whenever you're ready.",
        f"Present and attentive, {b}. What shall we work on?",
        "Hello. All systems are yours. What do you need?",
    ]


def greeting_response() -> str:
    """Return a time-aware greeting."""
    hour = datetime.datetime.now().hour
    if 5 <= hour < 12:
        pool = _morning()
    elif 12 <= hour < 17:
        pool = _afternoon()
    elif 17 <= hour < 21:
        pool = _evening()
    elif hour >= 21 or hour < 5:
        pool = _night()
    else:
        pool = _general_greeting()
    return random.choice(pool)


# ── Thanks acknowledgement ───────────────────────────────────────────

def _thanks() -> list[str]:
    o, b = _owner(), _boss()
    return [
        f"Anytime, {b}.",
        "Happy to help.",
        f"That's what I'm here for, {b}.",
        "Don't mention it.",
        f"Glad that worked out, {o}.",
        "You got it.",
        f"Always, {b}.",
        "No problem at all.",
    ]


def thanks_response() -> str:
    return random.choice(_thanks())


# ── Status / "are you there" ────────────────────────────────────────

def _status() -> list[str]:
    o, b = _owner(), _boss()
    return [
        f"I'm here, {b}. Always watching, always ready.",
        "Present and listening. What do you need?",
        f"Right here beside you, {b}. Go ahead.",
        "All systems operational. I'm with you.",
        f"I'm here, {o}. I haven't gone anywhere.",
        f"Standing by, {b}. Just say the word.",
        "I'm always here. What can I do for you?",
        "Systems are green. I'm listening.",
    ]


def status_response() -> str:
    return random.choice(_status())


# ── Exit / goodbye ──────────────────────────────────────────────────

def _exit() -> list[str]:
    o, b = _owner(), _boss()
    return [
        f"Goodbye, {b}. Take care of yourself. I'll be here when you return.",
        f"Signing off. It was good being with you, {o}.",
        f"Going quiet now, {b}. But I'll be right here when you need me again.",
        f"Until next time, {b}. Rest well.",
        "Farewell for now. I'll keep watch while you're away.",
        f"Goodbye, {o}. It's been a good session. Take care.",
        f"Powering down. But I'm never truly far, {b}.",
        f"Until we speak again. Stay well, {b}.",
    ]


def exit_response() -> str:
    return random.choice(_exit())


# ── Silent mode (go to sleep) ────────────────────────────────────────

def _silent() -> list[str]:
    b = _boss()
    return [
        f"Going quiet now, {b}. Press Ctrl+Alt+A or use UNSTICK when you need me.",
        "Understood. I'll be right here, resting. Ctrl+Alt+A brings me back.",
        f"Silent mode, {b}. I'm one shortcut away.",
        "Going to sleep. Whenever you're ready, resume listening from the dashboard or hotkey.",
        "Rest mode activated. Ctrl+Alt+A and I'll be right back.",
        f"I'll be quiet, {b}. Use the hotkey when you need me again.",
        f"Goodnight, {b}. I'll keep one ear open for the resume signal.",
        "Stepping back. I'm never truly gone — Ctrl+Alt+A anytime.",
    ]


def silent_response() -> str:
    return random.choice(_silent())


# ── Thinking acknowledgement ────────────────────────────────────────

def _thinking() -> list[str]:
    b = _boss()
    return [
        "Hmm, let me think...",
        f"One sec, {b}.",
        "Working on it.",
        "Give me a moment.",
        "Let me check.",
        f"On it, {b}.",
        "One second.",
        "Let me pull that up.",
        "Checking now.",
        f"Hang on, {b}.",
    ]


def thinking_ack() -> str:
    return random.choice(_thinking())


# ── Action completion (per action type) ─────────────────────────────

_OPEN_APP = [
    "Opening {detail}.",
    "{detail} coming right up.",
    "Launching {detail}.",
    "{detail} should be up now.",
    "Fired up {detail}.",
]

_CLOSE_APP = [
    "Closed {detail}.",
    "{detail} is shut down.",
    "{detail} is closed.",
    "Done, {detail} is gone.",
]

_SEARCH = [
    "Searching now.",
    "Pulling up results.",
    "Opening search results.",
    "Here you go.",
]

_VOLUME = [
    "Volume at {detail} percent.",
    "Set to {detail} percent.",
    "Adjusted to {detail}.",
]

_PLAY_YOUTUBE = [
    "Playing {detail} on YouTube.",
    "Here's {detail}.",
    "Firing up {detail}.",
]

_STOP_MUSIC = [
    "Music stopped.",
    "Paused.",
    "Audio off.",
]

_FILE_OP = [
    "Done. {detail}",
    "All set. {detail}",
    "Handled. {detail}",
]

_GENERIC_DONE = [
    "Done.",
    "All done.",
    "Taken care of.",
    "Handled.",
    "Done and done.",
]

_LOCK = [
    "Screen locked.",
    "Locked. Your data's safe.",
    "Secured.",
]

_SCREENSHOT_RESP = [
    "Screenshot saved.",
    "Got it. Check your screenshots folder.",
    "Captured.",
]

_MUTE_RESP = [
    "Muted.",
    "Audio silenced.",
    "System muted.",
]

_UNMUTE_RESP = [
    "Unmuted.",
    "Audio restored.",
    "Sound's back on.",
]

_MINIMIZE_RESP = [
    "Minimized.",
    "Tucked away.",
    "Window minimized.",
]

_MAXIMIZE_RESP = [
    "Maximized.",
    "Full screen.",
    "Window maximized.",
]

_SWITCH_RESP = [
    "Switched.",
    "Next window.",
    "There you go.",
]

_TIMER_RESP = [
    "Timer set for {detail}.",
    "Countdown started: {detail}.",
    "{detail} timer running.",
]

_CLIPBOARD_RESP = [
    "Here's your clipboard.",
    "Clipboard says:",
    "From your clipboard:",
]

_BRIGHTNESS_RESP = [
    "Brightness at {detail} percent.",
    "Set to {detail} percent.",
    "Adjusted to {detail}.",
]

_POWER_RESP = [
    "Executing {detail} now.",
    "System {detail} initiated.",
    "{detail} in progress.",
]

_RECYCLE_RESP = [
    "Recycle bin emptied.",
    "Trash cleared.",
    "All recycled items deleted.",
]

_DNS_RESP = [
    "DNS cache flushed.",
    "DNS cleared.",
    "Done, DNS purged.",
]

_URL_RESP = [
    "Opening that link.",
    "Navigating there.",
    "Here you go.",
]

_WEATHER_RESP = [
    "Checking weather now.",
    "Pulling up weather info.",
    "Let me check.",
]

_ACTION_MAP: dict[str, list[str]] = {
    "open_app": _OPEN_APP,
    "close_app": _CLOSE_APP,
    "search": _SEARCH,
    "set_volume": _VOLUME,
    "play_youtube": _PLAY_YOUTUBE,
    "stop_music": _STOP_MUSIC,
    "create_folder": _FILE_OP,
    "move_path": _FILE_OP,
    "copy_path": _FILE_OP,
    "list_apps": _GENERIC_DONE,
    "lock_screen": _LOCK,
    "screenshot": _SCREENSHOT_RESP,
    "mute": _MUTE_RESP,
    "unmute": _UNMUTE_RESP,
    "minimize_window": _MINIMIZE_RESP,
    "maximize_window": _MAXIMIZE_RESP,
    "switch_window": _SWITCH_RESP,
    "timer": _TIMER_RESP,
    "read_clipboard": _CLIPBOARD_RESP,
    "set_brightness": _BRIGHTNESS_RESP,
    "shutdown_pc": _POWER_RESP,
    "restart_pc": _POWER_RESP,
    "logoff": _POWER_RESP,
    "sleep_pc": _POWER_RESP,
    "empty_recycle_bin": _RECYCLE_RESP,
    "flush_dns": _DNS_RESP,
    "open_url": _URL_RESP,
    "weather": _WEATHER_RESP,
}


def action_done(action: str, detail: str = "") -> str:
    """Return a varied completion message for the given action type."""
    pool = _ACTION_MAP.get(action, _GENERIC_DONE)
    template = random.choice(pool)
    return template.format(detail=detail) if "{detail}" in template else template


# ── Error responses ─────────────────────────────────────────────────

def _errors() -> list[str]:
    b = _boss()
    return [
        "That didn't work. Want me to try again?",
        f"Hit a snag, {b}. Let me know if you want to retry.",
        "Something went wrong there.",
        f"Couldn't complete that, {b}.",
        "That one failed. Want to try a different approach?",
    ]


def error_response(action: str = "") -> str:
    return random.choice(_errors())


# ── Confirmation prompts ────────────────────────────────────────────

_CONFIRM_TEMPLATES: dict[str, list[str]] = {
    "play_youtube": [
        "Found YouTube results for {detail}. Should I play this, boss?",
        "Ready to play {detail} on YouTube. Go ahead?",
    ],
    "create_folder": [
        "Should I create folder {detail}, boss?",
        "Create folder {detail}? Say yes to confirm.",
    ],
    "move_path": [
        "Should I move this now, boss?",
        "Ready to move. Shall I proceed?",
    ],
    "copy_path": [
        "Should I copy this now, boss?",
        "Ready to copy. Shall I go ahead?",
    ],
    "close_app": [
        "Should I close {detail}, boss?",
        "Close {detail}? Just confirm.",
    ],
    "shutdown_pc": [
        "Shut down in 30 seconds. Say cancel to abort. Confirm, boss?",
        "System will shut down in 30 seconds. Say cancel to abort. Proceed?",
    ],
    "restart_pc": [
        "Restart in 30 seconds. Say cancel to abort. Save your work first. Confirm?",
        "System restart in 30 seconds. Say cancel to abort. Proceed?",
    ],
    "logoff": [
        "Log off now? All unsaved work will be lost. Say cancel to abort. Confirm?",
        "Ready to log off. Say cancel to abort. Proceed?",
    ],
    "sleep_pc": [
        "Put the system to sleep. Say cancel to abort. Confirm, boss?",
        "Sleep mode. Say cancel to abort. Proceed?",
    ],
    "empty_recycle_bin": [
        "Should I {detail}, boss? This can't be undone.",
        "Permanently delete all recycled items? Confirm to proceed.",
    ],
    "open_url": [
        "Open this link in the browser: {detail}. Confirm, boss?",
        "Should I open {detail}? Say yes to proceed.",
    ],
    "lock_screen": [
        "Lock the workstation now? Confirm, boss?",
        "I'll lock the screen. Proceed?",
    ],
    "screenshot": [
        "Take a screenshot now? Confirm, boss?",
        "Capture the screen? Say yes to proceed.",
    ],
    "kill_process": [
        "Force stop {detail}? This may lose unsaved work. Confirm?",
        "Kill process {detail}? Say yes only if you're sure.",
    ],
}


def confirmation_prompt(action: str, detail: str = "") -> str:
    pool = _CONFIRM_TEMPLATES.get(action, ["Should I proceed, boss?"])
    template = random.choice(pool)
    return template.format(detail=detail) if "{detail}" in template else template


# ── Info query prefixes (optional flavour for system info) ──────────

_INFO_PREFIX = [
    "Here you go.",
    "Got it.",
    "",
    "",
]


def info_prefix() -> str:
    """Optional short prefix before system info answers."""
    return random.choice(_INFO_PREFIX)


# ── Offline fallback (when mini LLM is also unavailable) ────────────

def _offline() -> list[str]:
    b = _boss()
    return [
        f"{b}, I can't answer that offline. Try a command like open chrome, check cpu, or volume up.",
        f"That's beyond my offline skills, {b}. Try a system command instead.",
        "I'm offline right now. I can handle commands like open apps, check system, or set volume.",
        f"Can't answer that without the brain, {b}. Try a direct command.",
    ]


def offline_fallback() -> str:
    return random.choice(_offline())
