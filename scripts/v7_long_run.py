#!/usr/bin/env python3
"""
ATOM V7 long-run monitor: periodic GPU/CPU snapshots for 8–12h sessions.

Usage (on target machine):
  ``python ATOM/scripts/v7_long_run.py --hours 8``

Emits JSON lines to stdout; redirect to a log file for analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=float, default=8.0)
    p.add_argument("--interval_s", type=float, default=60.0)
    args = p.parse_args()

    end = time.time() + args.hours * 3600.0
    while time.time() < end:
        row = {"t": time.time(), "kind": "v7_long_run_tick"}
        try:
            from core.gpu_resource_manager import get_nvml_vram_mb
            used, total = get_nvml_vram_mb()
            row["vram_used_mb"] = round(used, 1)
            row["vram_total_mb"] = round(total, 1)
        except Exception as e:
            row["vram_error"] = str(e)
        try:
            import psutil
            row["cpu_pct"] = psutil.cpu_percent(interval=0.5)
            row["ram_pct"] = psutil.virtual_memory().percent
        except Exception as e:
            row["psutil_error"] = str(e)
        print(json.dumps(row), flush=True)
        time.sleep(max(5.0, args.interval_s))


if __name__ == "__main__":
    main()
