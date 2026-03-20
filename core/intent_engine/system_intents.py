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
    r"\b(shutdown\s+(pc|computer|system|laptop|windows)|"
    r"power\s+off\s+(pc|computer|system)|turn\s+off\s+(pc|computer|system))\b", re.I)

_RESTART_PC = re.compile(
    r"\b(restart\s+(pc|computer|system|laptop|windows)|reboot)\b", re.I)

_LOGOFF = re.compile(
    r"\b(log\s*off|sign\s*out|logout|log\s+out)\b", re.I)

_SLEEP_PC = re.compile(
    r"\b(sleep\s+(pc|computer|system|laptop)|put\s+(pc|computer|it)\s+to\s+sleep|"
    r"hibernate)\b", re.I)

_EMPTY_RECYCLE_BIN = re.compile(
    r"\b(empty\s+recycle\s+bin|clear\s+recycle\s+bin|clean\s+trash|"
    r"delete\s+recycle\s+bin|recycle\s+bin\s+(empty|clear))\b", re.I)

_FLUSH_DNS = re.compile(
    r"\b(flush\s+dns|clear\s+dns|dns\s+flush|reset\s+dns)\b", re.I)


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
    return None


def quick_match(text: str) -> str | None:
    if _LOCK_SCREEN.search(text):
        return "lock_screen"
    if _SCREENSHOT.search(text):
        return "screenshot"
    return None
