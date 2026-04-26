#!/usr/bin/env python3
"""S4 -- MLX / Apple Neural Engine probe (Sprint P3.6 verification).

Loads ``mlx.core`` and reports the active device, the mlx-lm version,
and the macOS version. Exits non-zero when MLX is unavailable or stuck
on CPU on Apple Silicon.

Pass criteria (per docs/ATOM_NEXT_STEPS_PLAN.md §4):
    * ``mx.default_device()`` is non-CPU on Apple Silicon (Metal/MLX device).
    * macOS reports >= 26.2 (so ANE features in mlx-lm are available).

Exit codes:
    0  OK
    1  MLX import failed
    2  default device is CPU on Apple Silicon
    3  unsupported macOS version (only emitted with --strict)

Usage::

    python scripts/mlx_device_probe.py
    python scripts/mlx_device_probe.py --json
    python scripts/mlx_device_probe.py --strict   # fail < macOS 26.2
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from typing import Any


def _parse_mac_ver(s: str) -> tuple[int, int, int]:
    parts = (s or "0").split(".")
    out: list[int] = []
    for piece in parts[:3]:
        try:
            out.append(int(piece))
        except ValueError:
            out.append(0)
    while len(out) < 3:
        out.append(0)
    return out[0], out[1], out[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="ATOM S4 MLX probe")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--strict", action="store_true",
        help=(
            "exit non-zero when on Apple Silicon and default device is "
            "cpu, or macOS < 26.2"
        ),
    )
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "status": "ok",
        "machine": platform.machine(),
        "system": platform.system(),
        "macos": platform.mac_ver()[0] or "n/a",
    }
    is_apple_silicon = (
        platform.system() == "Darwin" and platform.machine() == "arm64"
    )
    payload["apple_silicon"] = is_apple_silicon

    try:
        import mlx
        import mlx.core as mx
    except Exception as exc:
        payload["status"] = "mlx_import_failed"
        payload["error"] = repr(exc)
        if args.json:
            print(json.dumps(payload))
        else:
            for k, v in payload.items():
                print(f"{k}: {v}")
        return 1

    try:
        device = str(mx.default_device())
    except Exception as exc:
        device = f"<error: {exc!r}>"
    payload["mlx_version"] = getattr(mlx, "__version__", "?")
    payload["mlx_default_device"] = device

    try:
        import mlx_lm
        payload["mlx_lm_version"] = getattr(mlx_lm, "__version__", "?")
    except Exception:
        payload["mlx_lm_version"] = "missing"

    macos = _parse_mac_ver(payload["macos"])
    payload["macos_supports_ane_compile"] = macos >= (26, 2, 0)

    rc = 0
    if args.strict:
        if is_apple_silicon and "cpu" in device.lower():
            payload["status"] = "mlx_on_cpu"
            rc = 2
        elif (
            is_apple_silicon
            and not payload["macos_supports_ane_compile"]
        ):
            payload["status"] = "macos_too_old_for_ane_compile"
            rc = 3

    if args.json:
        print(json.dumps(payload, default=str))
    else:
        for k, v in payload.items():
            print(f"{k}: {v}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
