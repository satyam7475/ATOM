#!/usr/bin/env python3
"""Observability dashboard — evolution plan §7.2 (development artifact).

**Live ATOM:** start ``main.py`` and use ``GET /v7/health`` on the dashboard port
(see README). The JSON includes ``latency_board`` (per-module rolling latencies fed
from Router, LocalBrain, PipelineTimer TTS, etc.).

**This CLI:** runs a short **offline demo** of ``ObservabilityLatencyBoard`` so the
module stays import-tested without booting the full voice stack.

Usage (repo root):

  python3 tools/observability_dashboard.py
  python3 tools/observability_dashboard.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _demo_board() -> dict:
    from core.observability.per_module_latency import ObservabilityLatencyBoard

    b = ObservabilityLatencyBoard(state_snapshot=lambda: {"demo": True})
    b.record_module_call("router", 2.1, error=False)
    b.record_module_call("router", 4.8, error=False)
    b.record_module_call("llm_small", 120.0, error=False)
    b.record_module_call("llm_small", 400.0, error=True)
    b.log_event("demo", "synthetic samples")
    return b.get_dashboard_data()


def main() -> None:
    p = argparse.ArgumentParser(description="ATOM observability latency board demo")
    p.add_argument("--json", action="store_true", help="print JSON only")
    args = p.parse_args()
    data = _demo_board()
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print("ObservabilityLatencyBoard demo (offline)")
        print(json.dumps(data, indent=2))
        print("\nTip: with main.py running, use GET /v7/health for live metrics.")


if __name__ == "__main__":
    main()
