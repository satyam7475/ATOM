#!/usr/bin/env python3
"""S5 -- Voice-to-voice (V2V) latency smoke test.

Exercises the dashboard's WebSocket as if a user typed a question, and
measures the time from "send" to "first TTS audio frame on the wire".
This is a strict superset of LLM latency (LLM + TTS warmup + audio
chunking) but a strict subset of V2V (it skips STT). Treat the result
as a *lower bound* on real V2V latency.

Pass criteria (per docs/ATOM_NEXT_STEPS_PLAN.md §4):
    * p50 latency < 1.2 s warm.
    * p50 latency < 2.0 s cold.

Exit codes:
    0  OK
    1  ATOM not reachable / dashboard error
    2  p50 above budget (only with --strict)

Usage::

    python scripts/voice_latency_smoke.py
    python scripts/voice_latency_smoke.py --runs 5 --json
    python scripts/voice_latency_smoke.py --base-url http://127.0.0.1:7860 --strict
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


DEFAULT_PROMPTS: list[str] = [
    "what time is it",
    "atom mujhe ek chota joke batao",
    "explain unified memory in one sentence",
    "open a new chrome tab",
    "good night atom",
]


def _extract_ws_url(base_url: str, *, timeout: float = 5.0) -> str:
    html = urlopen(base_url, timeout=timeout).read().decode("utf-8", "replace")
    match = re.search(r"/ws\?token=([^']+)'", html)
    if not match:
        raise RuntimeError(
            "could not extract dashboard websocket token from index html",
        )
    base = base_url.replace("http://", "ws://").rstrip("/")
    return f"{base}/ws?token={match.group(1)}"


async def _measure_one(ws_url: str, prompt: str, *, idx: int) -> dict[str, Any]:
    import aiohttp
    out: dict[str, Any] = {
        "idx": idx, "prompt": prompt, "send_to_first_audio_ms": None,
        "send_to_done_ms": None, "error": None,
    }
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.ws_connect(ws_url) as ws:
                # Drain the init burst.
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    msg = await ws.receive(timeout=1.0)
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    data = json.loads(msg.data)
                    if data.get("type") in ("init", "state"):
                        continue
                    break

                t0 = time.perf_counter()
                await ws.send_str(json.dumps({
                    "type": "text_input",
                    "text": prompt,
                }))

                first_audio_ms: float | None = None
                done_ms: float | None = None
                deadline = time.monotonic() + 30.0
                while time.monotonic() < deadline:
                    try:
                        msg = await ws.receive(timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    data = json.loads(msg.data)
                    kind = data.get("type") or data.get("event") or ""
                    if (
                        first_audio_ms is None
                        and kind in (
                            "tts_audio", "audio_chunk", "speech_chunk",
                            "tts.audio", "audio",
                        )
                    ):
                        first_audio_ms = (time.perf_counter() - t0) * 1000.0
                    if kind in ("tts_done", "speech_done", "reply_done", "done"):
                        done_ms = (time.perf_counter() - t0) * 1000.0
                        break
                out["send_to_first_audio_ms"] = (
                    None if first_audio_ms is None
                    else round(first_audio_ms, 1)
                )
                out["send_to_done_ms"] = (
                    None if done_ms is None else round(done_ms, 1)
                )
    except Exception as exc:
        out["error"] = repr(exc)
    return out


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        ws_url = _extract_ws_url(args.base_url)
    except (URLError, RuntimeError) as exc:
        return {
            "status": "atom_unreachable",
            "base_url": args.base_url,
            "error": repr(exc),
        }

    prompts: list[str] = []
    while len(prompts) < args.runs:
        prompts.extend(DEFAULT_PROMPTS)
    prompts = prompts[: args.runs]

    results: list[dict[str, Any]] = []
    for i, p in enumerate(prompts):
        results.append(await _measure_one(ws_url, p, idx=i))

    samples = [
        r["send_to_first_audio_ms"] for r in results
        if r["send_to_first_audio_ms"] is not None
    ]
    p50 = statistics.median(samples) if samples else None
    p95 = (
        statistics.quantiles(samples, n=20)[18] if len(samples) >= 5 else None
    )

    return {
        "status": "ok" if samples else "no_audio_observed",
        "ws_url": ws_url,
        "runs": len(results),
        "successful": len(samples),
        "first_audio_p50_ms": p50,
        "first_audio_p95_ms": p95,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ATOM S5 V2V latency probe")
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:7860",
        help="dashboard URL (default: http://127.0.0.1:7860)",
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--budget-warm-ms", type=float, default=1200.0,
        help="p50 budget when --cold is not set (default 1200 ms)",
    )
    parser.add_argument(
        "--budget-cold-ms", type=float, default=2000.0,
        help="p50 budget when --cold is set (default 2000 ms)",
    )
    parser.add_argument("--cold", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = asyncio.run(_run(args))

    if args.json:
        print(json.dumps(summary, default=str))
    else:
        print(f"S5 V2V latency: {summary['status']} (runs={summary.get('runs')})")
        if summary.get("status") == "ok":
            print(
                f"  first_audio p50: {summary['first_audio_p50_ms']:.0f} ms, "
                f"p95: "
                + (
                    f"{summary['first_audio_p95_ms']:.0f} ms"
                    if summary['first_audio_p95_ms'] is not None
                    else "n/a"
                ),
            )
        if summary.get("status") == "atom_unreachable":
            print(f"  base_url: {args.base_url}")
            print(f"  error:    {summary.get('error')}")

    if summary.get("status") != "ok":
        return 1
    if args.strict:
        budget = args.budget_cold_ms if args.cold else args.budget_warm_ms
        if (summary["first_audio_p50_ms"] or 9e9) > budget:
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
