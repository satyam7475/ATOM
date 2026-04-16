"""ATOM system-level action handlers."""

from __future__ import annotations

import datetime
import logging
import re
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("atom.router.system")

if sys.platform == "win32":
    import ctypes
    import struct


def lock_screen() -> None:
    if sys.platform == "darwin":
        subprocess.Popen(
            [
                "osascript",
                "-e",
                'tell application "System Events" to keystroke "q" '
                "using {control down, command down}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("Screen locked")
    elif sys.platform == "win32":
        ctypes.windll.user32.LockWorkStation()
        logger.info("Screen locked")


def take_screenshot() -> None:
    """Capture screen to a timestamped image on Desktop."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    desktop = Path.home() / "Desktop"
    if sys.platform == "darwin":
        filepath = desktop / f"screenshot_{ts}.png"
        subprocess.run(
            ["screencapture", "-x", str(filepath)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        logger.info("Screenshot saved: %s", filepath)
    elif sys.platform == "win32":
        filepath = desktop / f"screenshot_{ts}.bmp"
        try:
            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()
            w = user32.GetSystemMetrics(0)
            h = user32.GetSystemMetrics(1)
            gdi32 = ctypes.windll.gdi32
            hdc_screen = user32.GetDC(0)
            hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
            hbmp = gdi32.CreateCompatibleBitmap(hdc_screen, w, h)
            gdi32.SelectObject(hdc_mem, hbmp)
            gdi32.BitBlt(hdc_mem, 0, 0, w, h, hdc_screen, 0, 0, 0x00CC0020)

            bmp_header_size = 14 + 40
            row_size = ((w * 3 + 3) // 4) * 4
            img_size = row_size * h
            bfh = struct.pack(
                "<2sIHHI", b"BM", bmp_header_size + img_size, 0, 0, bmp_header_size
            )
            bih = struct.pack(
                "<IiiHHIIiiII", 40, w, -h, 1, 24, 0, img_size, 0, 0, 0, 0
            )

            buf = ctypes.create_string_buffer(img_size)

            class BITMAPINFOHEADER(ctypes.Structure):
                _fields_ = [
                    ("biSize", ctypes.c_uint32),
                    ("biWidth", ctypes.c_int32),
                    ("biHeight", ctypes.c_int32),
                    ("biPlanes", ctypes.c_uint16),
                    ("biBitCount", ctypes.c_uint16),
                    ("biCompression", ctypes.c_uint32),
                    ("biSizeImage", ctypes.c_uint32),
                    ("biXPelsPerMeter", ctypes.c_int32),
                    ("biYPelsPerMeter", ctypes.c_int32),
                    ("biClrUsed", ctypes.c_uint32),
                    ("biClrImportant", ctypes.c_uint32),
                ]

            bi = BITMAPINFOHEADER()
            bi.biSize = 40
            bi.biWidth = w
            bi.biHeight = -h
            bi.biPlanes = 1
            bi.biBitCount = 24
            gdi32.GetDIBits(hdc_mem, hbmp, 0, h, buf, ctypes.byref(bi), 0)

            with open(str(filepath), "wb") as f:
                f.write(bfh + bih + buf.raw)

            gdi32.DeleteObject(hbmp)
            gdi32.DeleteDC(hdc_mem)
            user32.ReleaseDC(0, hdc_screen)
            logger.info("Screenshot saved: %s", filepath)
        except Exception:
            subprocess.Popen(
                ["snippingtool.exe"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("Opened Snipping Tool as screenshot fallback")


def _darwin_brightness_current_percent() -> int:
    try:
        proc = subprocess.run(
            ["brightness", "-l"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        m = re.search(r"brightness\s+([\d.]+)", proc.stdout, re.I)
        if not m:
            return 50
        v = float(m.group(1))
        if v <= 1.0:
            return int(round(v * 100))
        return int(round(max(0, min(100, v))))
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return 50


def set_brightness(
    percent: int | None = None, delta: int | None = None
) -> int:
    """Set or adjust screen brightness; returns target percent."""
    if percent is not None:
        target = max(0, min(100, percent))
    elif delta is not None:
        if sys.platform == "darwin":
            current = _darwin_brightness_current_percent()
        else:
            current = 50
            if sys.platform == "win32":
                try:
                    get_cmd = (
                        "(Get-WmiObject -Namespace root/WMI "
                        "-Class WmiMonitorBrightness).CurrentBrightness"
                    )
                    proc = subprocess.run(
                        ["powershell", "-NoProfile", "-Command", get_cmd],
                        capture_output=True,
                        text=True,
                        timeout=3,
                    )
                    current = (
                        int(proc.stdout.strip()) if proc.stdout.strip() else 50
                    )
                except Exception:
                    current = 50
        target = max(0, min(100, current + delta))
    else:
        target = 50

    if sys.platform == "darwin":
        try:
            subprocess.run(
                ["brightness", str(target / 100.0)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
                check=False,
            )
        except FileNotFoundError:
            pass
        return target

    if sys.platform == "win32":
        try:
            cmd_parts = [
                "powershell",
                "-NoProfile",
                "-Command",
                f"(Get-WmiObject -Namespace root/WMI "
                f"-Class WmiMonitorBrightnessMethods)"
                f".WmiSetBrightness(1,{target})",
            ]
            subprocess.Popen(
                cmd_parts, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return target
        except Exception:
            return 50

    return target


def shutdown_pc() -> None:
    if sys.platform == "darwin":
        subprocess.Popen(
            ["osascript", "-e", 'tell app "System Events" to shut down'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    elif sys.platform == "win32":
        subprocess.Popen(
            ["shutdown", "/s", "/t", "30"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def restart_pc() -> None:
    if sys.platform == "darwin":
        subprocess.Popen(
            ["osascript", "-e", 'tell app "System Events" to restart'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    elif sys.platform == "win32":
        subprocess.Popen(
            ["shutdown", "/r", "/t", "30"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def logoff() -> None:
    if sys.platform == "darwin":
        subprocess.Popen(
            ["osascript", "-e", 'tell app "System Events" to log out'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    elif sys.platform == "win32":
        subprocess.Popen(
            ["shutdown", "/l"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def sleep_pc() -> None:
    if sys.platform == "darwin":
        subprocess.Popen(
            ["pmset", "sleepnow"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    elif sys.platform == "win32":
        subprocess.Popen(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Add-Type -Assembly System.Windows.Forms;"
                "[System.Windows.Forms.Application]"
                "::SetSuspendState('Suspend', $false, $false)",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def empty_recycle_bin() -> None:
    if sys.platform == "darwin":
        subprocess.Popen(
            [
                "osascript",
                "-e",
                'tell application "Finder" to empty trash',
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    elif sys.platform == "win32":
        subprocess.Popen(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Clear-RecycleBin -Force -ErrorAction SilentlyContinue",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def flush_dns() -> None:
    if sys.platform == "darwin":
        subprocess.run(
            ["dscacheutil", "-flushcache"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    elif sys.platform == "win32":
        subprocess.Popen(
            ["ipconfig", "/flushdns"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
