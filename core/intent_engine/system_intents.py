"""
ATOM Intent Engine -- System intents (lock, screenshot, brightness, shutdown, restart, logoff, sleep, recycle_bin, flush_dns).
"""

from __future__ import annotations

import re

from .base import IntentResult

_LOCK_SCREEN = re.compile(
    r"\b(lock\s+(screen|pc|computer|system|laptop)|screen\s+lock|"
    r"lock\s+it|tala\s+lagao|lock\s+karo)\b", re.I)

_SCREENSHOT = re.compile(
    r"\b(take\s+(a\s+)?screenshot|screenshot|screen\s+capture|capture\s+screen|"
    r"snap\s+screen|print\s+screen|ss\s+le(na)?)\b", re.I)

_SCREEN_BRIGHTNESS = re.compile(
    r"\b(brightness|set\s+brightness|screen\s+brightness)\s*(to\s+|at\s+|ko\s+)?(?P<pct>\d{1,3})\s*(percent|%)?",
    re.I,
)

_BRIGHTNESS_UP = re.compile(
    r"\b(increase\s+brightness|brightness\s+up|brighter|brightness\s+badha)\b", re.I)

_BRIGHTNESS_DOWN = re.compile(
    r"\b(decrease\s+brightness|brightness\s+down|dimmer|dim\s+screen|brightness\s+kam)\b", re.I)

_SHUTDOWN_PC = re.compile(
    r"\b(shut\s*down\s+(?:my\s+)?(?:pc|computer|system|laptop|windows|mac|macbook|this)|"
    r"power\s+off\s+(?:my\s+)?(?:pc|computer|system|mac|macbook|laptop)|"
    r"turn\s+off\s+(?:my\s+)?(?:pc|computer|system|mac|macbook|laptop))\b", re.I)

_RESTART_PC = re.compile(
    r"\b(restart\s+(?:my\s+)?(?:pc|computer|system|laptop|windows|mac|macbook|this)|reboot)\b", re.I)

_LOGOFF = re.compile(
    r"\b(log\s*off|sign\s*out|logout|log\s+out)\b", re.I)

_SLEEP_PC = re.compile(
    r"\b(sleep\s+(?:my\s+)?(?:pc|computer|system|laptop|mac|macbook)|"
    r"put\s+(?:my\s+)?(?:pc|computer|it|mac|macbook|laptop)\s+to\s+sleep|"
    r"hibernate)\b", re.I)

_EMPTY_RECYCLE_BIN = re.compile(
    r"\b(empty\s+(?:recycle\s+bin|trash|dustbin)|clear\s+(?:recycle\s+bin|trash)|"
    r"clean\s+trash|delete\s+recycle\s+bin|recycle\s+bin\s+(empty|clear)|"
    r"trash\s+(?:empty|clear|clean))\b", re.I)

_FLUSH_DNS = re.compile(
    r"\b(flush\s+dns|clear\s+dns|dns\s+flush|reset\s+dns)\b", re.I)

_TERMINAL_CMD = re.compile(
    r"^(?:run|execute|terminal|shell|command)\s*[:\-]?\s*(.+)",
    re.I,
)
_TERMINAL_EXPLICIT = re.compile(
    r"\b(?:run\s+(?:the\s+)?(?:command|terminal|shell)|"
    r"execute\s+(?:the\s+)?(?:command|terminal|shell)|"
    r"in\s+(?:the\s+)?terminal\s+(?:run|execute|type)|"
    r"(?:terminal|shell)\s+command)\b",
    re.I,
)

# ── System Control v1: storage, network discovery, atom optimization ─

_OPEN_PORTS = re.compile(
    r"\b(?:(?:show|list|scan|get|check)\s+(?:me\s+)?(?:the\s+)?(?:open|listening|active)\s+ports?|"
    r"what\s+ports\s+(?:are\s+)?(?:open|listening|in\s+use)|"
    r"port\s+scan|network\s+connections|listening\s+sockets|"
    r"who(?:'s|\s+is)\s+listening\s+on\s+(?:a\s+)?port)\b",
    re.I,
)

_WIFI_SCAN = re.compile(
    r"\b(?:(?:scan|show|list|get)\s+(?:me\s+)?(?:the\s+|nearby\s+|available\s+)?"
    r"(?:wifi|wi[-\s]?fi|wireless)\s+(?:networks?|ssids?|signals?)|"
    r"(?:wifi|wi[-\s]?fi)\s+scan|"
    r"(?:nearby|available|around\s+me)\s+wifi|"
    r"networks?\s+around\s+me)\b",
    re.I,
)

_FIND_LARGE_FILES = re.compile(
    r"\b(?:find\s+(?:me\s+)?(?:the\s+)?(?:biggest|largest|huge|large|big)\s+files?|"
    r"(?:what|which)\s+files?\s+(?:are\s+)?(?:taking|using|eating|hogging)\s+"
    r"(?:up\s+)?(?:my\s+|the\s+)?(?:space|storage|disk)|"
    r"(?:show|list)\s+(?:me\s+)?(?:the\s+)?(?:biggest|largest|large|huge|big)\s+files?|"
    r"what(?:'s|\s+is)\s+taking\s+up\s+(?:my\s+)?(?:space|storage|disk))\b",
    re.I,
)

_ANALYZE_TEMP = re.compile(
    r"\b(?:(?:analyze|scan|check)\s+(?:my\s+|the\s+)?(?:temp|temporary|junk|cache)\s+files?|"
    r"how\s+much\s+(?:temp|junk|cache)\s+(?:do\s+(?:i|we)\s+have|is\s+there)|"
    r"temp\s+files?\s+(?:report|analysis|scan|summary)|"
    r"junk\s+(?:scan|report)|"
    r"(?:can\s+i\s+|should\s+i\s+)?(?:clean|clear|free)\s+(?:up\s+)?(?:temp|junk|cache))\b",
    re.I,
)

_OPTIMIZE_FOR_ATOM = re.compile(
    r"\b(?:optimize\s+(?:for\s+)?(?:atom|yourself)|"
    r"free\s+(?:up\s+)?(?:some\s+)?(?:ram|memory|resources?)\s+for\s+(?:atom|yourself)|"
    r"(?:boost|tune|tune\s+up)\s+(?:atom|yourself)|"
    r"give\s+(?:atom|yourself)\s+(?:more|all)\s+(?:the\s+)?(?:ram|memory|power|resources?))\b",
    re.I,
)


def check(text: str) -> IntentResult | None:
    if _LOCK_SCREEN.search(text):
        return IntentResult("lock_screen", action="lock_screen", action_args={})
    if _SCREENSHOT.search(text):
        return IntentResult("screenshot", action="screenshot", action_args={})

    m = _SCREEN_BRIGHTNESS.search(text)
    if m:
        pct = max(0, min(100, int(m.group("pct"))))
        return IntentResult("set_brightness", action="set_brightness",
                            action_args={"percent": pct})
    if _BRIGHTNESS_UP.search(text):
        return IntentResult("set_brightness", action="set_brightness",
                            action_args={"delta": 20})
    if _BRIGHTNESS_DOWN.search(text):
        return IntentResult("set_brightness", action="set_brightness",
                            action_args={"delta": -20})

    if _SHUTDOWN_PC.search(text):
        return IntentResult("shutdown_pc", action="shutdown_pc", action_args={})
    if _RESTART_PC.search(text):
        return IntentResult("restart_pc", action="restart_pc", action_args={})
    if _LOGOFF.search(text):
        return IntentResult("logoff", action="logoff", action_args={})
    if _SLEEP_PC.search(text):
        return IntentResult("sleep_pc", action="sleep_pc", action_args={})
    if _EMPTY_RECYCLE_BIN.search(text):
        return IntentResult("empty_recycle_bin", action="empty_recycle_bin", action_args={})
    if _FLUSH_DNS.search(text):
        return IntentResult("flush_dns", action="flush_dns", action_args={})

    if _OPEN_PORTS.search(text):
        return IntentResult("get_open_ports", action="get_open_ports",
                            action_args={})
    if _WIFI_SCAN.search(text):
        return IntentResult("get_wifi_networks", action="get_wifi_networks",
                            action_args={})
    if _ANALYZE_TEMP.search(text):
        return IntentResult("analyze_temp_files", action="analyze_temp_files",
                            action_args={})
    if _OPTIMIZE_FOR_ATOM.search(text):
        return IntentResult("optimize_for_atom", action="optimize_for_atom",
                            action_args={})
    if _FIND_LARGE_FILES.search(text):
        args: dict = {}
        mnums = re.search(r"over\s+(\d+)\s*(gb|mb)?|bigger\s+than\s+(\d+)\s*(gb|mb)?",
                          text, re.I)
        if mnums:
            val = mnums.group(1) or mnums.group(3)
            unit = (mnums.group(2) or mnums.group(4) or "mb").lower()
            try:
                mb = int(val) * (1024 if unit == "gb" else 1)
                args["min_size_mb"] = mb
            except ValueError:
                pass
        return IntentResult("find_large_files", action="find_large_files",
                            action_args=args)

    m = _TERMINAL_CMD.search(text)
    if m:
        cmd = m.group(1).strip()
        return IntentResult("run_terminal_command", action="run_terminal_command",
                            action_args={"command": cmd})
    if _TERMINAL_EXPLICIT.search(text):
        return IntentResult("run_terminal_command", action="run_terminal_command",
                            action_args={"command": text})
    return None


def quick_match(text: str) -> str | None:
    if _LOCK_SCREEN.search(text):
        return "lock_screen"
    if _SCREENSHOT.search(text):
        return "screenshot"
    return None
