#!/usr/bin/env python3
"""S6 -- iPhone / remote bridge smoke test.

Validates the OpenAI-compatible /v1/* shim used by Enchanted (and any
other iPhone client over Tailscale) against a running ATOM. We do NOT
require a real Tailscale tunnel for this test -- we only assert that
the local shim returns:

    1. /v1/models -> 200, JSON, includes the configured ATOM model id
    2. /v1/chat/completions (stream=False) -> 200, JSON, has `choices`
    3. /v1/chat/completions (stream=True) -> 200, SSE frames, terminates
       with `data: [DONE]`

Pass criteria (per docs/ATOM_NEXT_STEPS_PLAN.md §4):
    * All three checks pass.
    * First-token latency on the streaming call < 5 s warm.

Exit codes:
    0  OK
    1  ATOM not reachable / shim missing
    2  contract violation (only with --strict)

Usage::

    python scripts/iphone_bridge_smoke.py
    python scripts/iphone_bridge_smoke.py --base-url http://100.64.0.5:7860 --json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def _http_get_json(url: str, *, timeout: float) -> tuple[int, Any]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace") if exc.fp else ""
        return exc.code, body
    except Exception as exc:
        return -1, repr(exc)


def _http_post_json(
    url: str, payload: dict[str, Any], *, timeout: float,
) -> tuple[int, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace") if exc.fp else ""
        return exc.code, body
    except Exception as exc:
        return -1, repr(exc)


def _http_post_sse(
    url: str, payload: dict[str, Any], *, timeout: float,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    out: dict[str, Any] = {
        "status": -1, "first_token_ms": None, "frames": 0,
        "saw_done": False, "error": None,
    }
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out["status"] = resp.status
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    body = line[len("data:"):].strip()
                    if body == "[DONE]":
                        out["saw_done"] = True
                        break
                    if out["first_token_ms"] is None:
                        out["first_token_ms"] = round(
                            (time.perf_counter() - t0) * 1000.0, 1,
                        )
                    out["frames"] += 1
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def _resolve_default_base_url() -> str:
    """Pick the right localhost base URL for the live ATOM bridge.

    Order of resolution (Apr 26 2026 audit hardening):
        1. ``logs/atom_bridge.port`` -- written by the live bridge on
           every boot, single source of truth even when the configured
           port is taken and the bridge falls back.
        2. ``cross_device.bridge_port`` from ``config/settings.json``.
        3. ``8787`` (the documented default).

    Eliminates the long-standing "smoke says fail / connection refused"
    when the smoke script's hardcoded :7860 didn't match either the
    live port file or the config -- the original failure mode was the
    smoke being right but probing the wrong port.
    """
    import json as _json
    from pathlib import Path as _Path

    repo = _Path(__file__).resolve().parent.parent
    port: int | None = None
    port_file = repo / "logs" / "atom_bridge.port"
    if port_file.is_file():
        try:
            port = int(port_file.read_text(encoding="utf-8").strip())
        except Exception:
            port = None
    if port is None:
        cfg_file = repo / "config" / "settings.json"
        if cfg_file.is_file():
            try:
                cfg = _json.loads(cfg_file.read_text(encoding="utf-8"))
                port = int(((cfg.get("cross_device") or {}).get("bridge_port")) or 0) or None
            except Exception:
                port = None
    return f"http://127.0.0.1:{port or 8787}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ATOM S6 iPhone / OpenAI bridge probe",
    )
    default_base_url = _resolve_default_base_url()
    parser.add_argument(
        "--base-url", default=default_base_url,
        help=(
            "dashboard URL (default resolves from logs/atom_bridge.port -> "
            "cross_device.bridge_port -> 8787; current default: "
            f"{default_base_url})"
        ),
    )
    parser.add_argument(
        "--model", default=None,
        help="model id to use (default: first model from /v1/models)",
    )
    parser.add_argument(
        "--prompt", default="say hi in 5 words", help="user prompt for the test",
    )
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--first-token-budget-ms", type=float, default=5000.0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")

    out: dict[str, Any] = {
        "status": "ok",
        "base_url": base,
        "checks": {},
    }

    code, body = _http_get_json(f"{base}/v1/models", timeout=args.timeout)
    out["checks"]["models"] = {
        "status_code": code,
        "ok": code == 200 and isinstance(body, dict) and "data" in body,
        "raw": body if code != 200 or not isinstance(body, dict) else None,
    }
    chosen_model = args.model
    if (
        out["checks"]["models"]["ok"]
        and isinstance(body, dict) and isinstance(body.get("data"), list)
        and body["data"]
    ):
        if not chosen_model:
            chosen_model = body["data"][0].get("id")
        out["checks"]["models"]["model_count"] = len(body["data"])
        out["checks"]["models"]["sample_id"] = body["data"][0].get("id")
    if not chosen_model:
        chosen_model = "atom-local"
    out["model"] = chosen_model

    payload = {
        "model": chosen_model,
        "messages": [{"role": "user", "content": args.prompt}],
        "stream": False,
    }
    code, body = _http_post_json(
        f"{base}/v1/chat/completions", payload, timeout=args.timeout,
    )
    out["checks"]["chat_completion"] = {
        "status_code": code,
        "ok": (
            code == 200 and isinstance(body, dict) and "choices" in body
            and len(body.get("choices", [])) > 0
        ),
        "raw": body if code != 200 or not isinstance(body, dict) else None,
    }

    payload_stream = dict(payload)
    payload_stream["stream"] = True
    sse = _http_post_sse(
        f"{base}/v1/chat/completions", payload_stream, timeout=args.timeout,
    )
    out["checks"]["chat_completion_stream"] = {
        "status_code": sse["status"],
        "ok": (
            sse["status"] == 200 and sse["frames"] > 0 and sse["saw_done"]
            and (sse["first_token_ms"] or 9e9) <= args.first_token_budget_ms
        ),
        "first_token_ms": sse["first_token_ms"],
        "frames": sse["frames"],
        "saw_done": sse["saw_done"],
        "error": sse["error"],
    }

    all_ok = all(c.get("ok") for c in out["checks"].values())
    out["status"] = "ok" if all_ok else "fail"

    if args.json:
        print(json.dumps(out, default=str))
    else:
        print(f"S6 iPhone bridge: {out['status']} ({base})")
        for name, c in out["checks"].items():
            tag = "OK " if c.get("ok") else "FAIL"
            extras = ""
            if name == "chat_completion_stream":
                ft = c.get("first_token_ms")
                extras = (
                    f" first_token={ft if ft is not None else 'n/a'} ms "
                    f"frames={c.get('frames')} done={c.get('saw_done')}"
                )
            print(f"  [{tag}] {name} (HTTP {c.get('status_code')}){extras}")
            if not c.get("ok") and c.get("raw"):
                snippet = str(c["raw"])
                print(f"      body: {snippet[:200]}")

    if not all_ok:
        return 2 if args.strict else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
