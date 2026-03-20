"""
ATOM Intent Engine -- OS-level intents (reminder, kill_process, resource_report/trend,
app_history, research, self_check, self_diagnostic, system_analyze, screen_analyze,
timer, read_clipboard, calculate, behavior_report).
"""

from __future__ import annotations

import math
import re

from .base import IntentResult, clean_slot

_REMINDER = re.compile(
    r"\b(?:remind\s+me|set\s+(?:a\s+)?reminder)\s+"
    r"(?:in\s+|for\s+|after\s+)?"
    r"(?P<amount>\d+)\s*"
    r"(?P<unit>seconds?|secs?|minutes?|mins?|hours?|hrs?)\s*"
    r"(?:to\s+|for\s+)?"
    r"(?P<task>.+)", re.I)

_REMINDER_REV = re.compile(
    r"\bremind\s+me\s+to\s+(?P<task>.+?)\s+in\s+(?P<amount>\d+)\s*"
    r"(?P<unit>seconds?|secs?|minutes?|mins?|hours?|hrs?)\b", re.I)

_SHOW_REMINDERS = re.compile(
    r"\b(show\s+(my\s+)?reminders|list\s+reminders|pending\s+reminders|"
    r"what\s+reminders|any\s+reminders|my\s+reminders)\b", re.I)

_CANCEL_REMINDERS = re.compile(
    r"\b(cancel\s+(?:all\s+)?reminders?|clear\s+(?:all\s+)?reminders?|"
    r"remove\s+(?:all\s+)?reminders?|delete\s+(?:all\s+)?reminders?)\b", re.I)

_KILL_PROCESS = re.compile(
    r"\b(?:kill\s+(?:the\s+)?process|force\s+(?:close|quit|stop)|"
    r"end\s+task|terminate)\s+(?P<name>.+?)[\s.!]*$", re.I)

_RESOURCE_REPORT = re.compile(
    r"\b(resource\s+(?:report|summary)|full\s+(?:system\s+)?report|"
    r"detailed\s+(?:system\s+)?status|system\s+resource|"
    r"resource\s+usage|resource\s+check)\b", re.I)

_RESOURCE_TREND = re.compile(
    r"\b(resource\s+trend|system\s+trend|performance\s+trend|"
    r"how\s+(?:is|are)\s+(?:my\s+)?resources?\s+trending)\b", re.I)

_APP_HISTORY = re.compile(
    r"\b(app\s+history|app\s+switches|what\s+(?:apps?|windows?)\s+"
    r"(?:was|were|have)\s+I\s+(?:using|on|in)|"
    r"recent\s+apps|show\s+app\s+history)\b", re.I)

_RESEARCH = re.compile(
    r"\b(?:research|investigate|deep\s+search)\s+(?P<topic>.+)", re.I)

_SELF_DIAGNOSTIC = re.compile(
    r"\b(?:self\s+diagnos(?:e|tic)|atom\s+diagnos(?:e|tic)|"
    r"evolution\s+(?:report|summary|status)|"
    r"evolve\s+(?:yourself|atom)|self\s+improv|improve\s+(?:yourself|atom)|"
    r"atom\s+(?:evolution)|diagnose\s+(?:yourself|atom)|"
    r"rate\s+(?:yourself|your\s+performance)|how\s+well\s+are\s+you\s+performing|"
    r"self\s+(?:analysis|review))\b", re.I)

_BEHAVIOR_REPORT = re.compile(
    r"\b(?:(?:show|tell)\s+(?:me\s+)?(?:my\s+)?(?:pattern|behavior|behaviour|habit|usage)|"
    r"behavior\s+report|behaviour\s+report|what\s+do\s+i\s+usually\s+do|"
    r"my\s+(?:pattern|habit|usage|routine)s?|predict\s+(?:my\s+)?(?:action|behavior))\b", re.I)

_SELF_CHECK = re.compile(
    r"\b(?:(?:atom\s+)?(?:self|system)\s*check(?:\s+karo)?|"
    r"check\s+(?:all|every)\s*(?:the\s+)?(?:functions?|systems?|things?|modules?)|"
    r"(?:atom\s+)?(?:run\s+)?(?:full\s+)?diagnostics|"
    r"(?:atom\s+)?health\s+(?:check|report)|"
    r"self\s+(?:check|test|report)|"
    r"check\s+(?:yourself|atom)|atom\s+check|"
    r"(?:run\s+)?system\s+(?:test|diagnostic)|"
    r"(?:sab\s+)?check\s+karo|status\s+report|"
    r"atom\s+(?:status|health)\s+(?:check|report)|"
    r"are\s+(?:all\s+)?systems?\s+(?:working|ok(?:ay)?|fine|good))\b", re.I)

_SCREEN_ANALYZE = re.compile(
    r"\b(analy[sz]e|look\s+at|read|check|see|describe|explain|review|scan|inspect)"
    r"\s+(my\s+|the\s+|this\s+)?"
    r"(screen|display|monitor|code|page|window|tab|error|bug)"
    r"(\s+.+)?$|"
    r"\bwhat('?s|\s+is)\s+(on\s+)?(my\s+|the\s+)?screen\b|"
    r"\bscreen\s+(pe|par|ko)\s+(kya|dekho|padho|check)\b|"
    r"\bscreen\s+dekho\b|"
    r"\bwhat\s+(do|can)\s+you\s+see\b", re.I)

_SYSTEM_ANALYZE = re.compile(
    r"\b(?:analyze\s+(?:my\s+)?system|system\s+analys(?:is|e)|"
    r"what(?:'s|\s+is)\s+(?:running|happening|going\s+on)(?:\s+on\s+(?:my\s+)?(?:system|computer|pc|laptop))?|"
    r"system\s+(?:status|overview|report|check)|"
    r"show\s+(?:me\s+)?(?:all\s+)?(?:running|open)\s+(?:apps?|programs?|windows?)|"
    r"what\s+apps?\s+(?:are\s+)?(?:running|open)|"
    r"full\s+(?:system\s+)?report)\b", re.I)

_TIMER = re.compile(
    r"\b(set\s+(?:a\s+)?timer|timer\s+(?:for|of)\s+|remind\s+me\s+in\s+|"
    r"countdown)\s*(?P<amount>\d+)\s*(?P<unit>second|seconds|sec|secs|minute|minutes|min|mins)\b",
    re.I)

_READ_CLIPBOARD = re.compile(
    r"\b(read\s+(my\s+)?clipboard|what('?s|\s+is)\s+(on\s+)?(my\s+)?clipboard|"
    r"clipboard\s+(content|text|read)|paste\s+text|show\s+clipboard)\b", re.I)

_CALCULATE = re.compile(
    r"\b(calculate|what('?s|\s+is)\s+\d|compute|solve|math)\s+(?P<expr>.+)", re.I)

_CALCULATE_SIMPLE = re.compile(r"^[\d\s\+\-\*\/\.\(\)\^%]+$")

_TYPING_SPEED = re.compile(r"\b(type|write)\s+(?P<text>.+)", re.I)

_SET_PERF_MODE = re.compile(
    r"\b(?:switch|set|change|go)\s+(?:to\s+)?(?:performance\s+)?(?:mode\s+)?"
    r"(?P<mode>full(?:\s+performance)?|lite|ultra\s*lite)"
    r"(?:\s+mode)?\b", re.I)


def _parse_duration(amount_str: str, unit_str: str) -> int:
    amount = int(amount_str)
    u = unit_str.lower()
    if u.startswith("h"):
        return amount * 3600
    if u.startswith("m"):
        return amount * 60
    return amount


def check_self_check(text: str) -> IntentResult | None:
    """Priority check -- must run before info_intents to avoid system_info overlap."""
    if _SELF_CHECK.search(text):
        return IntentResult("self_check", action="self_check", action_args={})
    return None


def check(text: str) -> IntentResult | None:
    if _SELF_CHECK.search(text):
        return IntentResult("self_check", action="self_check", action_args={})

    m = _SET_PERF_MODE.search(text)
    if m:
        raw = m.group("mode").strip().lower()
        mode = "ultra_lite" if "ultra" in raw else ("full" if "full" in raw else "lite")
        return IntentResult("set_performance_mode", action="set_performance_mode",
                            action_args={"mode": mode})

    m = _SCREEN_ANALYZE.search(text)
    if m:
        return IntentResult("screen_analyze", action="screen_analyze",
                            action_args={"question": text.strip()})

    m = _REMINDER_REV.search(text)
    if m:
        task = clean_slot(m.group("task"))
        secs = _parse_duration(m.group("amount"), m.group("unit"))
        return IntentResult("set_reminder", action="set_reminder",
                            action_args={"label": task, "delay_seconds": secs})
    m = _REMINDER.search(text)
    if m:
        task = clean_slot(m.group("task"))
        secs = _parse_duration(m.group("amount"), m.group("unit"))
        return IntentResult("set_reminder", action="set_reminder",
                            action_args={"label": task, "delay_seconds": secs})

    if _SHOW_REMINDERS.search(text):
        return IntentResult("show_reminders", action="show_reminders", action_args={})
    if _CANCEL_REMINDERS.search(text):
        return IntentResult("cancel_reminders", action="cancel_reminders", action_args={})

    m = _KILL_PROCESS.search(text)
    if m:
        name = m.group("name").strip()
        if name:
            return IntentResult("kill_process", action="kill_process",
                                action_args={"name": name})

    if _RESOURCE_REPORT.search(text):
        return IntentResult("resource_report", action="resource_report", action_args={})
    if _RESOURCE_TREND.search(text):
        return IntentResult("resource_trend", action="resource_trend", action_args={})
    if _APP_HISTORY.search(text):
        return IntentResult("app_history", action="app_history", action_args={})

    m = _RESEARCH.search(text)
    if m:
        topic = m.group("topic").strip()
        if topic:
            return IntentResult("research", action="research_topic",
                                action_args={"topic": topic})

    if _BEHAVIOR_REPORT.search(text):
        return IntentResult("behavior_report", action="behavior_report", action_args={})
    if _SELF_DIAGNOSTIC.search(text):
        return IntentResult("self_diagnostic", action="self_diagnostic", action_args={})

    if _SYSTEM_ANALYZE.search(text):
        return IntentResult("system_analyze", action="system_analyze", action_args={})

    m = _TIMER.search(text)
    if m:
        amount = int(m.group("amount"))
        unit = m.group("unit").lower()
        if unit.startswith("min"):
            seconds = amount * 60
            label = f"{amount} minute{'s' if amount != 1 else ''}"
        else:
            seconds = amount
            label = f"{amount} second{'s' if amount != 1 else ''}"
        return IntentResult("timer", response=f"Timer set for {label}, boss.",
                            action="timer", action_args={"seconds": seconds, "label": label})

    if _READ_CLIPBOARD.search(text):
        return IntentResult("read_clipboard", action="read_clipboard", action_args={})

    m = _CALCULATE.search(text)
    if m:
        expr = m.group("expr").strip()
    elif _CALCULATE_SIMPLE.match(text.strip()):
        expr = text.strip()
    else:
        return None

    expr_clean = expr.replace("^", "**").replace("x", "*")
    _SAFE_NAMES = {
        "sqrt": math.sqrt, "pi": math.pi, "e": math.e,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "log": math.log, "abs": abs, "round": round,
    }
    if re.search(r'[a-zA-Z_]\s*\(', expr_clean):
        func_names = re.findall(r'([a-zA-Z_]+)\s*\(', expr_clean)
        if any(fn not in _SAFE_NAMES for fn in func_names):
            return None
    _BLOCKED = (
        "import", "exec", "eval", "compile", "open", "os.", "sys.",
        "__", "getattr", "setattr", "delattr", "globals", "locals",
        "vars", "dir", "type", "class", "lambda", "breakpoint",
    )
    if any(kw in expr_clean for kw in _BLOCKED):
        return None
    if len(expr_clean) > 100:
        return None
    try:
        result = eval(expr_clean, {"__builtins__": {}}, _SAFE_NAMES)  # noqa: S307
        if not isinstance(result, (int, float, complex)):
            return None
        if isinstance(result, float):
            result = round(result, 6)
        return IntentResult("calculate", response=f"The answer is {result}.")
    except Exception:
        return None
