"""
ATOM CLI — visibility into goals, metrics, and runtime snapshot.

Usage (from repo):
  python ATOM/atom_cli.py status
  python ATOM/atom_cli.py metrics
  python ATOM/atom_cli.py goals
  python ATOM/atom_cli.py health

Does not start the full ATOM stack; reads persisted state and config files.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONFIG = ROOT / "config"
RUNTIME = CONFIG / "atom_runtime.json"
LEARNING = Path.cwd() / "atom_learning_state.json"
if not LEARNING.exists():
    LEARNING = ROOT.parent / "atom_learning_state.json"
PLAN_REG = CONFIG / "plan_registry.json"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def cmd_status() -> None:
    rt = _load_json(RUNTIME)
    snap = rt.get("snapshot") or {}
    print("=== ATOM status ===")
    print(f"active_goal_id:    {snap.get('active_goal_id')}")
    print(f"active_objective:  {snap.get('active_objective')}")
    print(f"plan_template:     {snap.get('plan_template')}")
    print(f"last_success:      {snap.get('last_success')}")
    w = snap.get("plan_score_weights") or {}
    if w:
        print("plan_score_weights:")
        for k, v in w.items():
            print(f"  {k}: {v:.4f}")


def cmd_metrics() -> None:
    rt = _load_json(RUNTIME)
    summ = rt.get("telemetry_summary") or {}
    agg = summ.get("aggregates") or {}
    print("=== ATOM metrics (last run / aggregates) ===")
    if not agg:
        print("(no telemetry yet — run brain orchestrator to populate)")
        return
    for name, stats in agg.items():
        print(f"{name}: {stats}")


def cmd_goals() -> None:
    rt = _load_json(RUNTIME)
    snap = rt.get("snapshot") or {}
    print("=== Goals / reflection ===")
    print(f"active_goal_id: {snap.get('active_goal_id')}")
    print(f"objective:      {snap.get('active_objective')}")
    insights = snap.get("reflection_insights") or []
    print("recent_reflection_insights:")
    for i in insights:
        print(f"  - {i}")
    print("top_failing_skills:")
    for s in snap.get("top_failing_skills") or []:
        print(f"  - {s}")


def cmd_registry() -> None:
    data = _load_json(PLAN_REG)
    print("=== plan_registry.json ===")
    for tid, entry in data.items():
        if not isinstance(entry, dict):
            continue
        print(f"{tid}: success_rate={entry.get('success_rate')} avg_time_s={entry.get('avg_time_s')} runs={entry.get('_runs')}")


def cmd_health() -> None:
    """V6.5 system health score (0–10) from telemetry + profiler snapshot."""
    try:
        from core.health_monitor import compute_v65_health_score
    except Exception:
        print("=== ATOM health (V6.5) ===")
        print("(health module unavailable — run from ATOM directory on PYTHONPATH)")
        return
    report = compute_v65_health_score()
    score = report.get("health_score_10", 0)
    print("=== ATOM health (V6.5) ===")
    print(f"ATOM Health: {score} / 10")
    print(json.dumps(report, indent=2))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="atom", description="ATOM cognitive system CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="Active goal, template, last success, score weights")
    sub.add_parser("metrics", help="Telemetry aggregates (execution time, goal success)")
    sub.add_parser("goals", help="Goals snapshot and reflection insights")
    sub.add_parser("registry", help="Plan template registry stats")
    sub.add_parser("health", help="V6.5 composite health score (0–10)")

    args = p.parse_args(argv)
    if args.cmd == "status":
        cmd_status()
    elif args.cmd == "metrics":
        cmd_metrics()
    elif args.cmd == "goals":
        cmd_goals()
    elif args.cmd == "registry":
        cmd_registry()
    elif args.cmd == "health":
        cmd_health()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
