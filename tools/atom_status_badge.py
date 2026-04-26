#!/usr/bin/env python3
"""ATOM unified status badge — terminal + macOS menubar renderer.

Sprint P4.6 (Apr 26 2026). One-shot CLI that polls ATOM's ``/badge``
endpoint and either prints a single coloured line (default) or runs as
a continuous menubar polling daemon.

Usage::

    # one-shot (default port resolution: logs/atom_bridge.port)
    python tools/atom_status_badge.py

    # continuous, refresh every 5 s, write to stdout
    python tools/atom_status_badge.py --watch --interval 5

    # JSON for scripts
    python tools/atom_status_badge.py --json

The endpoint shape is documented in
:py:meth:`core.cross_device.iphone_bridge.IPhoneBridge._handle_badge`.

This tool deliberately uses *only* the standard library so it can run
in a fresh shell, on a fresh checkout, without an active ATOM venv.
The macOS menubar mode (``--menubar``) needs ``rumps`` (``pip install
rumps``) and is opt-in.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


_COLOUR = {
    "ok":       "\033[32m",
    "warn":     "\033[33m",
    "critical": "\033[31m",
    "unknown":  "\033[90m",
}
_RESET = "\033[0m"
_GLYPH = {
    "ok":       "●",
    "warn":     "▲",
    "critical": "✗",
    "unknown":  "○",
}


def _resolve_default_port() -> int:
    """Read ``logs/atom_bridge.port`` if it exists, else fall back to 8787.

    The bridge persists its actually-bound port to that file at boot so
    we always hit the right one even after a +1/+2 port-fallback.
    """
    candidate = _REPO_ROOT / "logs" / "atom_bridge.port"
    if candidate.is_file():
        try:
            return int(candidate.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            pass
    return 8787


def fetch_badge(host: str, port: int, *, timeout: float = 2.0) -> dict:
    """Hit ``/badge`` and return the parsed JSON. Raises on transport
    errors so callers can decide whether to render an "offline" badge.
    """
    url = f"http://{host}:{port}/badge"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", "replace")
        return json.loads(body)


def _render_line(badge: dict) -> str:
    level = str(badge.get("level") or "unknown")
    text = str(badge.get("text") or "ATOM status unknown")
    headline = str(badge.get("headline") or "")
    glyph = _GLYPH.get(level, _GLYPH["unknown"])
    if not sys.stdout.isatty():
        rendered = f"{glyph} {text}"
    else:
        colour = _COLOUR.get(level, _COLOUR["unknown"])
        rendered = f"{colour}{glyph} {text}{_RESET}"
    if headline:
        rendered += f"  ({headline})"
    return rendered


def _render_offline(host: str, port: int, exc: Exception) -> dict:
    """Construct an "offline" badge so the watch loop never crashes."""
    return {
        "ok": False,
        "level": "critical",
        "color": "red",
        "text": f"ATOM is offline ({host}:{port})",
        "headline": f"{type(exc).__name__}: {exc}",
        "warnings": [],
        "criticals": [{"name": "bridge", "status": "down"}],
        "subsystems_total": 0,
        "uptime_s": None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render ATOM's unified status badge.",
    )
    parser.add_argument(
        "--host", default=os.environ.get("ATOM_BRIDGE_HOST", "127.0.0.1"),
        help="bridge host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help=(
            "bridge port (default: read from logs/atom_bridge.port, "
            "else 8787)"
        ),
    )
    parser.add_argument(
        "--json", action="store_true",
        help="emit the raw badge payload as JSON (one line per poll)",
    )
    parser.add_argument(
        "--watch", action="store_true",
        help="poll forever and re-render on each tick",
    )
    parser.add_argument(
        "--interval", type=float, default=5.0,
        help="seconds between polls when --watch is set",
    )
    parser.add_argument(
        "--menubar", action="store_true",
        help=(
            "run as a macOS menubar daemon (requires `pip install rumps`)"
        ),
    )
    parser.add_argument(
        "--timeout", type=float, default=2.0,
        help="HTTP timeout (seconds) per poll",
    )
    args = parser.parse_args(argv)

    port = args.port if args.port is not None else _resolve_default_port()

    if args.menubar:
        return _run_menubar(args.host, port, args.interval, args.timeout)

    def _emit_once() -> int:
        try:
            badge = fetch_badge(args.host, port, timeout=args.timeout)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            badge = _render_offline(args.host, port, exc)
        if args.json:
            print(json.dumps(badge, ensure_ascii=False))
        else:
            print(_render_line(badge))
        return 0 if str(badge.get("level")) == "ok" else 1

    if not args.watch:
        return _emit_once()

    try:
        while True:
            _emit_once()
            time.sleep(max(0.5, float(args.interval)))
    except KeyboardInterrupt:
        return 130


def _run_menubar(host: str, port: int, interval: float, timeout: float) -> int:
    try:
        import rumps  # type: ignore
    except ImportError:
        print(
            "macOS menubar mode needs `rumps`. Install with: "
            "pip install rumps",
            file=sys.stderr,
        )
        return 2

    class AtomBadgeApp(rumps.App):  # type: ignore[misc]
        def __init__(self_):  # noqa: N805
            super().__init__("ATOM ○", quit_button="Quit ATOM Badge")
            self_._timer = rumps.Timer(self_._tick, max(1.0, float(interval)))
            self_._timer.start()
            self_._tick(None)

        def _tick(self_, _):
            try:
                badge = fetch_badge(host, port, timeout=timeout)
            except Exception as exc:  # pragma: no cover - GUI runtime path
                badge = _render_offline(host, port, exc)
            level = str(badge.get("level") or "unknown")
            self_.title = f"ATOM {_GLYPH.get(level, '○')}"
            tooltip = badge.get("text") or "ATOM"
            headline = badge.get("headline") or ""
            self_.menu = [
                rumps.MenuItem(tooltip),
                rumps.MenuItem(headline) if headline else None,
                None,
                rumps.MenuItem(
                    f"Subsystems: {badge.get('subsystems_total', 0)}",
                ),
                rumps.MenuItem(f"Last polled: {time.strftime('%H:%M:%S')}"),
            ]

    AtomBadgeApp().run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
