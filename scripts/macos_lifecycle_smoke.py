#!/usr/bin/env python3
"""
Phase 7.2 — macOS lifecycle smoke (automated slice of MEMORY_BANK step 7.2).

Verifies without starting full ``main.py``:
  - Apple Silicon monitor + psutil battery snapshot
  - Default audio output device label (AirPods vs built-in shows up here)
  - Proactive M5 triggers for synthetic low battery + high RAM
  - MemoryGraph unified-memory pressure hook on/off
  - Optional: kernel boot time (sanity for post-sleep manual runs)

Manual checklist (not automated): sleep/wake full ATOM, AirPods connect/disconnect
during live STT, optional ``memorypressure`` while ATOM runs.

Run from repo root:
  python3 scripts/macos_lifecycle_smoke.py
"""

from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _require_darwin_arm64() -> None:
    if platform.system() != "Darwin":
        print("macos_lifecycle_smoke: skip — not macOS")
        sys.exit(0)
    if platform.machine() != "arm64":
        print("macos_lifecycle_smoke: skip — not Apple Silicon arm64")
        sys.exit(0)


def _load_config() -> dict:
    return json.loads((ROOT / "config" / "settings.json").read_text(encoding="utf-8"))


def _check_monitor() -> None:
    from core.apple_silicon_monitor import AppleSiliconMonitor

    s = AppleSiliconMonitor().get_stats()
    assert s.memory_pct >= 0.0
    assert 0.0 <= s.battery_pct <= 100.0
    print(
        "monitor:",
        f"mem_pct={s.memory_pct:.1f}",
        f"battery={s.battery_pct:.0f}%",
        f"on_battery={s.on_battery}",
        f"thermal={s.thermal_pressure}",
    )


def _check_audio_and_boot() -> None:
    from core.macos.phase7_lifecycle import fetch_default_audio_output_label, read_kern_boottime_dict

    out = fetch_default_audio_output_label()
    print(f"audio_default_output={out!r}")
    bt = read_kern_boottime_dict()
    print(f"kern_boottime={bt}")


def _check_proactive() -> None:
    from unittest.mock import MagicMock

    from core.cognitive.proactive_engine import ProactiveIntelligenceEngine

    bus = MagicMock()
    bus.on = MagicMock()
    bus.emit_long = MagicMock()
    eng = ProactiveIntelligenceEngine(bus=bus, config=_load_config())
    eng._last_scan = {
        "ram_percent": 90,
        "battery": {"percent": 11, "plugged": False},
    }
    t = time.time()
    insights = eng._scan_m5_context_triggers(t)
    cats = {x.get("category") for x in insights}
    assert "system_battery" in cats, cats
    assert "system_memory" in cats, cats
    print("proactive: system_battery + system_memory triggers OK")


def _check_memory_graph_pressure() -> None:
    from brain.memory_graph import MemoryGraph

    fd, path = tempfile.mkstemp(suffix="_mg_phase7.db")
    os.close(fd)
    try:
        g = MemoryGraph(
            path,
            {"memory": {"pressure_threshold_pct": 85.0, "pressure_relief_pct": 75.0}},
        )
        a = g.apply_memory_pressure(92.0)
        assert a["active"] is True
        b = g.apply_memory_pressure(74.0)
        assert b["active"] is False
        print("memory_graph: pressure on/off OK")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def main() -> None:
    _require_darwin_arm64()
    _check_monitor()
    _check_audio_and_boot()
    _check_proactive()
    _check_memory_graph_pressure()
    print(
        "manual_phase7_2:",
        "sleep/wake ATOM once;",
        "toggle AirPods and re-run to see audio_default_output change;",
        "optional: memorypressure while main.py runs.",
    )
    print("macos_lifecycle_smoke: OK")


if __name__ == "__main__":
    main()
