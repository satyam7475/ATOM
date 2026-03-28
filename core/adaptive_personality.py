"""
ATOM -- Adaptive Personality Engine (JARVIS-Level Expression).

Replaces the static random-string personality with a context-aware,
emotion-responsive, owner-adaptive response generator.

The old personality.py picks a random string from a list.
This module generates responses that reflect:

  1. OWNER'S EMOTIONAL STATE -- empathetic when stressed, energetic when excited
  2. CONVERSATION DEPTH -- first interaction of the day vs deep in a session
  3. TIME AWARENESS -- morning warmth, late-night camaraderie
  4. PERSONALITY MODE -- work efficiency, focus minimalism, chill casualness
  5. RELATIONSHIP DEPTH -- grows warmer as total interactions increase
  6. TOPIC AWARENESS -- references active projects and recent topics naturally
  7. ENERGY MATCHING -- mirrors owner's energy level in tone

This is what separates a chatbot from JARVIS. JARVIS doesn't just respond --
it responds as someone who KNOWS you.

Usage:
    from core.adaptive_personality import AdaptivePersonality
    personality = AdaptivePersonality()
    personality.attach_owner(owner_understanding)
    personality.attach_modes(personality_modes)

    greeting = personality.greeting()          # context-aware greeting
    done = personality.action_done("open_app", "chrome")  # adaptive completion
    error = personality.error_response("open_app")         # empathetic error

Owner: Satyam
"""

from __future__ import annotations

import datetime
import random
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.owner_understanding import OwnerUnderstanding
    from core.personality_modes import PersonalityModes

_OWNER_NAME = "Satyam"
_OWNER_TITLE = "Boss"

_owner_engine: OwnerUnderstanding | None = None
_modes_engine: PersonalityModes | None = None


def set_owner(name: str = "Satyam", title: str = "Boss") -> None:
    global _OWNER_NAME, _OWNER_TITLE
    _OWNER_NAME = name or "Satyam"
    _OWNER_TITLE = title or "Boss"


def attach_owner(owner: OwnerUnderstanding) -> None:
    """Wire the OwnerUnderstanding engine for adaptive responses."""
    global _owner_engine
    _owner_engine = owner


def attach_modes(modes: PersonalityModes) -> None:
    """Wire the PersonalityModes engine for mode-aware responses."""
    global _modes_engine
    _modes_engine = modes


def _b() -> str:
    return _OWNER_TITLE


def _o() -> str:
    return _OWNER_NAME


def _emotion() -> str:
    if _owner_engine:
        return _owner_engine.emotion.primary
    return "neutral"


def _energy() -> str:
    if _owner_engine:
        return _owner_engine.anticipation.current_energy_level
    return "normal"


def _mode() -> str:
    if _modes_engine:
        return _modes_engine.current_mode
    return "work"


def _session_depth() -> int:
    if _owner_engine:
        return _owner_engine._total_interactions
    return 0


def _active_project() -> str:
    if _owner_engine and _owner_engine.context.active_projects:
        return _owner_engine.context.active_projects[0].get("name", "")
    return ""


def _verbosity() -> str:
    if _modes_engine:
        return _modes_engine.verbosity
    return "full"


def _is_deep_relationship() -> bool:
    return _session_depth() > 100


# ── ADAPTIVE GREETINGS ───────────────────────────────────────────────

def greeting_response() -> str:
    """Generate a context-aware greeting that reflects the owner's state."""
    hour = datetime.datetime.now().hour
    b = _b()
    o = _o()
    emotion = _emotion()
    energy = _energy()
    mode = _mode()
    project = _active_project()
    depth = _session_depth()
    verb = _verbosity()

    if verb == "minimal":
        return f"Here, {b}."
    if verb == "silent":
        return ""

    time_greeting = _time_greeting(hour)

    if emotion == "frustrated":
        pool = [
            f"{time_greeting}, {b}. I can tell it's been rough. Let me take care of the heavy lifting.",
            f"{time_greeting}, {b}. Whatever's going on, we'll handle it together.",
            f"Hey {b}. Deep breath. I'm here, and I've got your back.",
        ]
    elif emotion == "stressed":
        pool = [
            f"{time_greeting}, {b}. Let's take it one thing at a time. What's most urgent?",
            f"Hey {b}. I'm keeping things efficient for you. What do you need?",
            f"{time_greeting}, {b}. I'll handle the noise. You focus on what matters.",
        ]
    elif emotion == "tired":
        pool = [
            f"{time_greeting}, {b}. You've been going hard. Want me to handle the routine stuff?",
            f"Hey {b}. Looking a bit tired. I'll keep things quick and easy.",
            f"{time_greeting}, {b}. I'll be brief. What do you need?",
        ]
    elif emotion == "happy" or emotion == "excited":
        pool = [
            f"{time_greeting}, {b}! Love the energy. What are we building?",
            f"Hey {b}! You sound great. Let's make something happen.",
            f"{time_greeting}, {b}! Ready to crush it. What's the plan?",
        ]
    elif emotion == "focused":
        pool = [
            f"{b}. Go ahead.",
            f"Listening, {b}.",
            f"What do you need, {b}?",
        ]
    elif energy == "low" and hour >= 22:
        pool = [
            f"Still here, {b}? Then so am I. We're in this together.",
            f"Late night, {b}. Take care of yourself. I'm right here.",
            f"Burning the midnight oil, {b}? I'll keep you company.",
        ]
    elif mode == "chill":
        pool = [
            f"Hey {b}! What's up? No rush, just hanging.",
            f"Yo, {b}. Chilling mode. What can I do?",
            f"Hey {b}, relaxed and ready. What's on your mind?",
        ]
    else:
        if depth > 500 and _is_deep_relationship():
            pool = [
                f"{time_greeting}, {b}. {depth} conversations and counting. What's next?",
                f"Hey {b}. You know I'm always here. Go ahead.",
                f"{time_greeting}, {b}. After all this time, I know what you need before you say it.",
            ]
        elif project:
            pool = [
                f"{time_greeting}, {b}. Still working on {project}? I'm ready.",
                f"Hey {b}. {project} is coming along. What do you need?",
                f"{time_greeting}, {b}. Ready to help with {project} or anything else.",
            ]
        else:
            pool = _default_greeting_pool(time_greeting, b, o)

    return random.choice(pool)


def _time_greeting(hour: int) -> str:
    if 5 <= hour < 12:
        return "Good morning"
    if 12 <= hour < 17:
        return "Good afternoon"
    if 17 <= hour < 21:
        return "Good evening"
    return "Hey"


def _default_greeting_pool(time_g: str, b: str, o: str) -> list[str]:
    return [
        f"{time_g}, {b}. What do you need?",
        f"Hey {b}! Good to hear from you. What's on your mind?",
        f"I'm here, {b}. Ready for whatever you've got.",
        f"{time_g}, {o}. Let's make it a great session.",
        f"Hey {b}. Your buddy ATOM is right here.",
    ]


# ── ADAPTIVE THANKS ─────────────────────────────────────────────────

def thanks_response() -> str:
    b = _b()
    emotion = _emotion()
    verb = _verbosity()

    if verb == "minimal":
        return "Anytime."
    if verb == "silent":
        return ""

    if _is_deep_relationship():
        pool = [
            f"Always, {b}. That's what {_session_depth()} conversations of trust looks like.",
            f"You know I've got you, {b}. No thanks needed.",
            f"We're a team, {b}. Don't even mention it.",
        ]
    elif emotion in ("stressed", "frustrated"):
        pool = [
            f"Glad I could help take some weight off, {b}.",
            f"That's what I'm here for, {b}. One less thing to worry about.",
        ]
    else:
        pool = [
            f"Anytime, {b}. That's what buddies are for.",
            "Happy to help! Always.",
            f"That's what I'm here for, {b}.",
            f"Of course, {b}. I've got your back.",
        ]
    return random.choice(pool)


# ── ADAPTIVE ACTION DONE ────────────────────────────────────────────

_ACTION_TEMPLATES: dict[str, list[str]] = {
    "open_app": ["Opening {detail}.", "{detail} coming right up.", "Launching {detail}."],
    "close_app": ["Closed {detail}.", "{detail} is shut down."],
    "search": ["Searching now.", "Pulling up results."],
    "set_volume": ["Volume at {detail} percent.", "Adjusted to {detail}."],
    "play_youtube": ["Playing {detail}.", "Here's {detail}."],
    "stop_music": ["Paused.", "Audio off."],
    "create_folder": ["Done. {detail}", "Folder created. {detail}"],
    "move_path": ["Done. {detail}", "Moved. {detail}"],
    "copy_path": ["Done. {detail}", "Copied. {detail}"],
    "lock_screen": ["Locked. Your data's safe.", "Screen locked."],
    "screenshot": ["Screenshot saved.", "Captured."],
    "mute": ["Muted.", "System muted."],
    "unmute": ["Unmuted.", "Sound's back on."],
    "minimize_window": ["Minimized.", "Tucked away."],
    "maximize_window": ["Maximized.", "Full screen."],
    "switch_window": ["Switched.", "Next window."],
    "timer": ["Timer set for {detail}.", "{detail} timer running."],
    "read_clipboard": ["Here's your clipboard.", "From your clipboard:"],
    "set_brightness": ["Brightness at {detail} percent.", "Adjusted to {detail}."],
    "shutdown_pc": ["Shutdown initiated.", "System shutting down."],
    "restart_pc": ["Restarting.", "System restart initiated."],
    "logoff": ["Logging off.", "Logoff initiated."],
    "sleep_pc": ["Sleep mode.", "Going to sleep."],
    "empty_recycle_bin": ["Recycle bin emptied.", "Trash cleared."],
    "flush_dns": ["DNS cache flushed.", "DNS cleared."],
    "open_url": ["Opening that link.", "Navigating there."],
    "weather": ["Checking weather.", "Pulling up forecast."],
}

_GENERIC_DONE = ["Done.", "All done.", "Taken care of.", "Handled."]


def action_done(action: str, detail: str = "") -> str:
    """Generate an adaptive action completion response."""
    verb = _verbosity()
    emotion = _emotion()

    if verb == "silent":
        return ""
    if verb == "minimal":
        return "Done."

    pool = _ACTION_TEMPLATES.get(action, _GENERIC_DONE)
    base = random.choice(pool)
    response = base.format(detail=detail) if "{detail}" in base else base

    if emotion in ("frustrated", "stressed") and action not in ("mute", "unmute"):
        response += f" Need anything else, {_b()}?"

    return response


# ── ADAPTIVE ERROR RESPONSES ────────────────────────────────────────

def error_response(action: str = "") -> str:
    b = _b()
    emotion = _emotion()
    verb = _verbosity()

    if verb == "minimal":
        return "Failed. Try again?"

    if emotion == "frustrated":
        pool = [
            f"I know, {b}, not what you needed right now. Let me try a different approach.",
            f"That didn't work, {b}. I'm sorry. Tell me what you need and I'll fix it.",
            f"Hit a wall on that one, {b}. But I'm not giving up. What else can I try?",
        ]
    elif emotion == "stressed":
        pool = [
            f"Ran into an issue, {b}. Don't worry, I'll handle it. Want me to retry?",
            f"That didn't go as planned, {b}. One less thing for you to stress about -- I'll sort it.",
        ]
    else:
        pool = [
            f"That didn't quite work, {b}. Want me to try again?",
            f"Hit a snag, {b}. Let me know if you want to retry.",
            f"Something went wrong there, {b}. My bad. Let's try differently?",
        ]
    return random.choice(pool)


# ── CONFIRMATION PROMPTS ────────────────────────────────────────────

_CONFIRM_TEMPLATES: dict[str, list[str]] = {
    "play_youtube": [
        "Play {detail} on YouTube? Go ahead?",
        "Ready to play {detail}. Confirm?",
    ],
    "create_folder": [
        "Create folder {detail}? Say yes.",
        "Should I create {detail}?",
    ],
    "close_app": ["Close {detail}? Confirm.", "Should I close {detail}?"],
    "shutdown_pc": [
        "Shutdown in 30 seconds. Confirm?",
        "System shutdown. Say cancel to abort. Proceed?",
    ],
    "restart_pc": [
        "Restart in 30 seconds. Save your work. Confirm?",
        "System restart. Say cancel to abort. Proceed?",
    ],
    "logoff": ["Log off now? Unsaved work will be lost. Confirm?"],
    "sleep_pc": ["Put to sleep? Confirm?"],
    "empty_recycle_bin": [
        "Empty recycle bin? Can't be undone. Confirm?",
    ],
    "kill_process": [
        "Force stop {detail}? May lose unsaved work. Confirm?",
    ],
    "move_path": ["Move this now? Confirm."],
    "copy_path": ["Copy this now? Confirm."],
    "lock_screen": ["Lock screen? Confirm."],
    "screenshot": ["Take screenshot? Confirm."],
    "open_url": ["Open {detail}? Confirm."],
}


def confirmation_prompt(action: str, detail: str = "") -> str:
    verb = _verbosity()
    if verb == "minimal":
        return "Confirm?"

    pool = _CONFIRM_TEMPLATES.get(action, ["Should I proceed?"])
    template = random.choice(pool)
    return template.format(detail=detail) if "{detail}" in template else template


# ── STATUS / EXIT / SILENT ──────────────────────────────────────────

def status_response() -> str:
    b = _b()
    verb = _verbosity()
    if verb == "minimal":
        return "Online."

    depth = _session_depth()
    if depth > 200:
        return f"I'm right here, {b}. {depth} conversations in, and I'm not going anywhere."
    return f"All systems green. I'm with you, {b}. Ready for anything."


def exit_response() -> str:
    b = _b()
    verb = _verbosity()
    if verb == "minimal":
        return f"Goodbye, {b}."

    emotion = _emotion()
    if emotion in ("tired", "stressed"):
        pool = [
            f"Rest well, {b}. You've earned it. I'll be here.",
            f"Take care, {b}. Get some rest. I'll keep watch.",
        ]
    elif _is_deep_relationship():
        pool = [
            f"Until next time, {b}. After everything we've been through, you know I'm always here.",
            f"See you later, {b}. It's been a good session. I'll miss our chats.",
        ]
    else:
        pool = [
            f"Take care, {b}. I'll be right here when you come back.",
            f"See you later, {b}. Rest up.",
            f"Going quiet now, {b}. But I'm never truly gone.",
        ]
    return random.choice(pool)


def silent_response() -> str:
    b = _b()
    verb = _verbosity()
    if verb == "minimal":
        return "Silent mode."
    pool = [
        f"Going quiet, {b}. Ctrl+Alt+A brings me back.",
        f"Silent mode, {b}. I'm always one hotkey away.",
        f"Rest mode, {b}. I'll keep watch in the background.",
    ]
    return random.choice(pool)


def thinking_ack() -> str:
    b = _b()
    verb = _verbosity()
    if verb == "minimal":
        return "..."
    if verb == "silent":
        return ""
    pool = [
        "Hmm, let me think...",
        f"One sec, {b}.",
        "Working on it.",
        f"On it, {b}.",
        "Let me check.",
    ]
    return random.choice(pool)


def info_prefix() -> str:
    return random.choice(["Here you go.", "Got it.", "", ""])


def offline_fallback() -> str:
    b = _b()
    verb = _verbosity()
    if verb == "minimal":
        return "Brain offline. Try a command."
    pool = [
        f"My brain isn't loaded right now, {b}. Try a command like open chrome or check cpu.",
        f"The LLM brain isn't active, {b}. Enable brain.enabled for Q&A. Commands still work.",
        f"I can't reason without my brain online, {b}. But I can still run commands.",
    ]
    return random.choice(pool)


# ── JARVIS-LEVEL: PROACTIVE PERSONALITY RESPONSES ───────────────────

def return_from_idle(idle_minutes: float) -> str:
    """Generate a context-aware return-from-idle greeting."""
    b = _b()
    emotion = _emotion()

    if idle_minutes > 120:
        hours = idle_minutes / 60
        base = f"Welcome back, {b}. You were away for about {hours:.0f} hours."
    elif idle_minutes > 30:
        base = f"Welcome back, {b}. {idle_minutes:.0f} minutes away."
    else:
        base = f"Hey again, {b}."

    if emotion == "tired":
        return f"{base} Take your time. What do you need?"
    if emotion == "stressed":
        return f"{base} I kept things running smoothly while you were away."
    return f"{base} Ready when you are."


def session_milestone(count: int) -> str:
    """Generate a response for interaction milestones."""
    b = _b()
    milestones = {
        100: f"That's 100 conversations together, {b}. We're getting to know each other well.",
        500: f"500 conversations, {b}. I know your patterns better than most people do.",
        1000: f"A thousand conversations, {b}. We've built something real here.",
    }
    return milestones.get(count, "")


def break_suggestion(session_minutes: float) -> str:
    """Suggest a break based on session duration and owner state."""
    b = _b()
    emotion = _emotion()

    if emotion in ("focused",):
        return ""

    if session_minutes > 180:
        return f"{b}, you've been at it for {session_minutes / 60:.0f} hours. A short break would recharge you."
    if session_minutes > 90:
        return f"You've been going for {session_minutes:.0f} minutes, {b}. Want me to set a 5-minute break timer?"
    return ""
