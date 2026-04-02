"""
ATOM Intent Engine -- Info intents (time, date, cpu, ram, battery, disk, system_info, ip, wifi, uptime, top_processes).
"""

from __future__ import annotations

import datetime
import re
import socket
import time

from core import adaptive_personality as personality
from .base import IntentResult
import psutil

_TIME = re.compile(
    r"\b(what('?s|\s+is)?\s+the\s+time|current\s+time|time\s+(now|please)|"
    r"tell\s+(me\s+)?the\s+time|what\s+time\s+is\s+it|kitna\s+baj)", re.I)

_DATE = re.compile(
    r"\b(what('?s|\s+is)?\s+the\s+date|today('?s)?\s+date|current\s+date|"
    r"what\s+day\s+is\s+(it|today)|aaj\s+kya\s+date)", re.I)

_CPU = re.compile(
    r"\b(check\s+cpu|cpu\s+(usage|status|load|percent)|"
    r"how('?s|\s+is)?\s+(the\s+)?cpu|system\s+load|processor\s+usage)", re.I)

_RAM = re.compile(
    r"\b(check\s+(ram|memory)|ram\s+(usage|status)|memory\s+(usage|status)|"
    r"how\s+much\s+(ram|memory))", re.I)

_BATTERY = re.compile(
    r"\b(check\s+battery|battery\s+(status|level|percent|life)|"
    r"how\s+much\s+battery|is\s+it\s+charging|power\s+status)", re.I)

_DISK = re.compile(
    r"\b(check\s+disk|disk\s+(space|usage|status)|storage\s+(space|status|left)|"
    r"how\s+much\s+(disk|storage|space))", re.I)

_SYSTEM_INFO = re.compile(
    r"\b(system\s+(info|status|health|report|check)|"
    r"check\s+(system|health|status)|how('?s|\s+is)?\s+(the\s+)?(system|computer|laptop|pc))", re.I)

_IP = re.compile(
    r"\b(my\s+ip|ip\s+address|what('?s|\s+is)?\s+my\s+ip|show\s+ip|network\s+info)", re.I)

_WIFI_STATUS = re.compile(
    r"\b(wifi\s+(status|info|speed|connected|name)|"
    r"am\s+i\s+(connected|online)|internet\s+(status|connected|speed)|"
    r"network\s+status|ssid)\b", re.I)

_UPTIME = re.compile(
    r"\b(uptime|how\s+long\s+(running|been\s+on|active)|system\s+uptime)\b", re.I)

_TOP_PROCESSES = re.compile(
    r"\b(top\s+process|running\s+process|task\s+list|what('?s|\s+is)\s+running|"
    r"process\s+list|heavy\s+process|resource\s+hog|cpu\s+hog|"
    r"show\s+process|list\s+process)\b", re.I)


def check(text: str) -> IntentResult | None:
    if _TIME.search(text):
        now = datetime.datetime.now()
        t = now.strftime("%I:%M %p")
        return IntentResult("time", response=f"It's {t}, boss.")
    if _DATE.search(text):
        now = datetime.datetime.now()
        d = now.strftime("%A, %B %d, %Y")
        return IntentResult("date", response=f"Today is {d}.")
    if _CPU.search(text):
        pfx = personality.info_prefix()
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            return IntentResult("cpu", response=f"{pfx} CPU usage is {cpu:.0f} percent.".strip())
        except Exception:
            return IntentResult("cpu", response="Couldn't check CPU right now.")
    if _RAM.search(text):
        pfx = personality.info_prefix()
        try:
            mem = psutil.virtual_memory()
            used_gb = mem.used / (1024 ** 3)
            total_gb = mem.total / (1024 ** 3)
            msg = (f"RAM usage is {mem.percent:.0f} percent. "
                   f"{used_gb:.1f} of {total_gb:.1f} GB in use.")
            return IntentResult("ram", response=f"{pfx} {msg}".strip())
        except Exception:
            return IntentResult("ram", response="Couldn't check RAM right now.")
    if _BATTERY.search(text):
        pfx = personality.info_prefix()
        try:
            bat = psutil.sensors_battery()
            if bat:
                plug = "and charging" if bat.power_plugged else "on battery"
                msg = f"Battery is at {bat.percent:.0f} percent, {plug}."
                return IntentResult("battery", response=f"{pfx} {msg}".strip())
            return IntentResult("battery", response="No battery detected.")
        except Exception:
            return IntentResult("battery", response="Couldn't check battery.")
    if _DISK.search(text):
        pfx = personality.info_prefix()
        try:
            d = psutil.disk_usage("C:\\")
            free_gb = d.free / (1024 ** 3)
            total_gb = d.total / (1024 ** 3)
            msg = (f"Disk C has {free_gb:.0f} GB free out of {total_gb:.0f} GB. "
                   f"{d.percent:.0f} percent used.")
            return IntentResult("disk", response=f"{pfx} {msg}".strip())
        except Exception:
            return IntentResult("disk", response="Couldn't check disk space.")
    if _SYSTEM_INFO.search(text):
        pfx = personality.info_prefix()
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            bat = psutil.sensors_battery()
            disk = psutil.disk_usage("C:\\")
            parts = [
                f"CPU at {cpu:.0f} percent.",
                f"RAM at {mem.percent:.0f} percent.",
                f"Disk C has {disk.free / (1024**3):.0f} GB free.",
            ]
            if bat:
                plug = "charging" if bat.power_plugged else "on battery"
                parts.append(f"Battery {bat.percent:.0f} percent, {plug}.")
            msg = " ".join(parts)
            return IntentResult("system_info", response=f"{pfx} {msg}".strip())
        except Exception:
            return IntentResult("system_info", response="Couldn't get system info.")
    if _IP.search(text):
        try:
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            return IntentResult("ip",
                response=f"Your hostname is {hostname}. Local IP is {ip}.")
        except Exception:
            return IntentResult("ip", response="Couldn't get network info.")
    if _WIFI_STATUS.search(text):
        return IntentResult("wifi", action="wifi_status", action_args={})
    if _UPTIME.search(text):
        try:
            boot = psutil.boot_time()
            uptime_secs = time.time() - boot
            hours = int(uptime_secs // 3600)
            minutes = int((uptime_secs % 3600) // 60)
            if hours > 0:
                msg = f"System has been up for {hours} hours and {minutes} minutes."
            else:
                msg = f"System has been up for {minutes} minutes."
            return IntentResult("uptime", response=msg)
        except Exception:
            return IntentResult("uptime", response="Couldn't get uptime.")
    if _TOP_PROCESSES.search(text):
        try:
            procs = []
            for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
                try:
                    procs.append(p.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            procs.sort(key=lambda x: x.get("cpu_percent", 0) or 0, reverse=True)
            top5 = procs[:5]
            parts = []
            for p in top5:
                name = p.get("name", "?")
                cpu = p.get("cpu_percent", 0) or 0
                mem = p.get("memory_percent", 0) or 0
                parts.append(f"{name} ({cpu:.0f}% CPU, {mem:.0f}% RAM)")
            msg = "Top processes: " + ", ".join(parts) + "."
            return IntentResult("top_processes", response=msg)
        except Exception:
            return IntentResult("top_processes", response="Couldn't list processes.")
    return None


def quick_match(text: str) -> str | None:
    """Fast check for info intents used by STT early-exit."""
    if _TIME.search(text):
        return "time"
    if _DATE.search(text):
        return "date"
    if _CPU.search(text):
        return "cpu"
    if _RAM.search(text):
        return "ram"
    if _BATTERY.search(text):
        return "battery"
    if _DISK.search(text):
        return "disk"
    if _SYSTEM_INFO.search(text):
        return "system_info"
    return None
