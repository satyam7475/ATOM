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

# ── Self-Healing Intents ─────────────────────────────────────────────

_DIAGNOSE_FAILURE = re.compile(
    r"\b(?:diagnos(?:e|tic)\s+(?:the\s+)?(?:last\s+)?(?:failure|error|crash|problem|issue|bug)|"
    r"what\s+(?:went\s+wrong|failed|crashed|broke|is\s+broken)|"
    r"show\s+(?:me\s+)?(?:the\s+)?(?:last\s+)?(?:error|failure|crash)|"
    r"(?:any|show)\s+(?:errors?|failures?|crashes?|problems?|issues?|bugs?)|"
    r"failure\s+(?:report|log|status|analysis)|"
    r"error\s+(?:report|log|status)|"
    r"what\s+(?:errors?|problems?)\s+(?:do\s+you\s+have|are\s+there))\b", re.I)

_FIX_IT = re.compile(
    r"\b(?:fix\s+it|fix\s+(?:the\s+)?(?:error|failure|problem|issue|bug|crash)|"
    r"repair\s+(?:yourself|it|this|that)|"
    r"heal\s+(?:yourself|it)|auto\s*fix|self\s*fix|"
    r"fix\s+(?:yourself|atom)|apply\s+(?:the\s+)?fix|"
    r"fix\s+all(?:\s+errors|\s+issues|\s+failures)?)\b", re.I)

_FIX_ALL = re.compile(
    r"\b(?:fix\s+all|repair\s+all|heal\s+all|fix\s+everything|"
    r"auto\s*fix\s+all|self\s*heal)\b", re.I)

_MODULE_HEALTH = re.compile(
    r"\b(?:module\s+health|check\s+(?:all\s+)?modules?|"
    r"module\s+status|are\s+(?:all\s+)?modules?\s+(?:working|ok|healthy)|"
    r"which\s+modules?\s+(?:are\s+)?(?:broken|failing|down)|"
    r"health\s+check\s+modules?)\b", re.I)

# ── Code Introspection Intents ───────────────────────────────────────

_READ_OWN_CODE = re.compile(
    r"\b(?:read\s+(?:your\s+)?(?:own\s+)?(?:code|source|programming)|"
    r"explain\s+(?:your\s+)?(?:own\s+)?(?:code|source|design|architecture|system)|"
    r"(?:show|tell)\s+(?:me\s+)?(?:your\s+)?(?:architecture|design|structure|code\s+structure)|"
    r"how\s+(?:are\s+you|do\s+you)\s+(?:built|designed|structured|programmed|coded)|"
    r"analyze\s+(?:your\s+)?(?:own\s+)?(?:code|source|codebase)|"
    r"introspect(?:\s+(?:your|the)\s+code)?|"
    r"code\s+(?:health|quality|analysis)|"
    r"scan\s+(?:your\s+)?(?:own\s+)?(?:code|source|files))\b", re.I)

_EXPLAIN_MODULE = re.compile(
    r"\b(?:explain|describe|what\s+(?:is|does))\s+(?:the\s+)?(?:module\s+)?(?P<module>\S+\.py)\b", re.I)

_SEARCH_CODE = re.compile(
    r"\b(?:search|find|look\s+for|grep)\s+(?:in\s+)?(?:(?:your|the|my)\s+)?(?:code|source)\s+(?:for\s+)?(?P<query>.+)\b", re.I)

# ── Security Intents ─────────────────────────────────────────────────

_SECURITY_STATUS = re.compile(
    r"\b(?:security\s+(?:status|report|check|scan)|"
    r"(?:how\s+)?(?:is|am)\s+(?:I|my\s+system)\s+(?:secure|safe|protected)|"
    r"(?:check|show|tell)\s+(?:me\s+)?(?:the\s+)?security|"
    r"(?:are|is)\s+(?:we|atom|everything)\s+secure|"
    r"threat\s+(?:level|status|report)|"
    r"integrity\s+(?:check|verify|scan))\b", re.I)

_SECURITY_LOCKDOWN = re.compile(
    r"\b(?:lock\s*down|lockdown|enter\s+(?:secure|lockdown)\s+mode|"
    r"maximum\s+security|full\s+security)\b", re.I)

# ── Voice Authentication Intents ─────────────────────────────────────

_VOICE_ENROLL = re.compile(
    r"\b(?:enroll\s+(?:my\s+)?voice|register\s+(?:my\s+)?voice|"
    r"voice\s+(?:enrollment|registration|enroll|register)|"
    r"learn\s+my\s+voice|save\s+my\s+voice(?:\s*print)?|"
    r"record\s+(?:my\s+)?voice(?:\s*print)?|"
    r"create\s+(?:my\s+)?voice\s*(?:profile|print)|"
    r"set\s*up\s+voice\s+(?:auth|authentication|id|identity))\b", re.I)

_VOICE_VERIFY = re.compile(
    r"\b(?:verify\s+(?:my\s+)?(?:voice|identity)|"
    r"authenticate\s+(?:me|my\s+voice)|"
    r"voice\s+(?:verify|verification|authenticate|check)|"
    r"(?:am\s+)?I\s+(?:the\s+)?(?:owner|verified)|"
    r"who\s+am\s+I|confirm\s+(?:my\s+)?identity|"
    r"(?:check|prove)\s+(?:my\s+)?identity|"
    r"identify\s+me|voice\s+id)\b", re.I)

_VOICE_AUTH_STATUS = re.compile(
    r"\b(?:voice\s+(?:auth|authentication)\s+(?:status|info|report)|"
    r"(?:my\s+)?voice\s*(?:print)?\s+status|"
    r"(?:show|check)\s+voice\s+(?:profile|enrollment)|"
    r"voice\s+(?:enrollment|profile)\s+(?:status|info))\b", re.I)

_VOICE_RESET = re.compile(
    r"\b(?:reset\s+(?:my\s+)?voice\s*(?:print|enrollment|profile)?|"
    r"clear\s+(?:my\s+)?voice\s*(?:print|enrollment|profile)?|"
    r"delete\s+(?:my\s+)?voice\s*(?:print|enrollment|profile)?|"
    r"remove\s+voice\s+(?:auth|authentication|enrollment))\b", re.I)

_BEHAVIOR_AUTH_STATUS = re.compile(
    r"\b(?:behavior(?:al)?\s+(?:auth|authentication|trust|status|report)|"
    r"trust\s+(?:score|level|status)|"
    r"(?:am\s+I|is\s+(?:my\s+)?behavior)\s+(?:normal|trusted)|"
    r"anomaly\s+(?:score|report|status|check))\b", re.I)


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

    # ── Self-healing intents (high priority) ──
    if _FIX_ALL.search(text):
        return IntentResult("fix_all", action="fix_all", action_args={})

    if _FIX_IT.search(text):
        return IntentResult("fix_it", action="fix_it", action_args={})

    if _DIAGNOSE_FAILURE.search(text):
        return IntentResult("diagnose_failure", action="diagnose_failure", action_args={})

    if _MODULE_HEALTH.search(text):
        return IntentResult("module_health", action="module_health", action_args={})

    # ── Code introspection intents ──
    m = _EXPLAIN_MODULE.search(text)
    if m:
        module = m.group("module").strip()
        return IntentResult("explain_module", action="explain_module",
                            action_args={"module": module})

    m = _SEARCH_CODE.search(text)
    if m:
        query = m.group("query").strip()
        return IntentResult("search_code", action="search_code",
                            action_args={"query": query})

    if _READ_OWN_CODE.search(text):
        return IntentResult("read_own_code", action="read_own_code", action_args={})

    # ── Security intents ──
    if _SECURITY_LOCKDOWN.search(text):
        return IntentResult("security_lockdown", action="security_lockdown", action_args={})

    if _SECURITY_STATUS.search(text):
        return IntentResult("security_status", action="security_status", action_args={})

    # ── Voice authentication intents ──
    if _VOICE_ENROLL.search(text):
        return IntentResult("voice_enroll", action="voice_enroll", action_args={})

    if _VOICE_VERIFY.search(text):
        return IntentResult("voice_verify", action="voice_verify", action_args={})

    if _VOICE_RESET.search(text):
        return IntentResult("voice_reset", action="voice_reset", action_args={})

    if _VOICE_AUTH_STATUS.search(text):
        return IntentResult("voice_auth_status", action="voice_auth_status", action_args={})

    if _BEHAVIOR_AUTH_STATUS.search(text):
        return IntentResult("behavior_auth_status", action="behavior_auth_status",
                            action_args={})

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
