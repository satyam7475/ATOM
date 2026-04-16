#!/usr/bin/env python3
from __future__ import annotations

import logging

logger = logging.getLogger('atom.scripts.atom_runtime_smoke')

import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.request import urlopen

import aiohttp

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]


DEFAULT_PROMPTS: list[str] = [
    "what time is it",
    "atom mujhe ek chota joke batao",
    "mujhe short me batao unified memroy kya hota hai",
    "hinglish me samjha do ki optimal aur full performance me farak kya hai",
    "compare safari and arc for coding on a macbook air",
    "atom me cpu spike kyu hota hai beech beech me",
    "explain properly what is docker",
]
_TRANSCRIPT_LABEL_RE = re.compile(r"\b(?:User|Boss|ATOM|Assistant):", re.I)


def _load_prompts(path: str | None) -> list[str]:
    if not path:
        return list(DEFAULT_PROMPTS)
    data = Path(path).read_text(encoding="utf-8")
    return [line.strip() for line in data.splitlines() if line.strip()]


def _extract_ws_url(base_url: str) -> str:
    html = urlopen(base_url).read().decode("utf-8", "replace")
    match = re.search(r"/ws\?token=([^']+)'", html)
    if not match:
        raise RuntimeError("Could not extract dashboard websocket token.")
    return base_url.replace("http://", "ws://").rstrip("/") + f"/ws?token={match.group(1)}"


async def _wait_until_listening(ws: aiohttp.ClientWebSocketResponse) -> None:
    deadline = time.monotonic() + 30.0
    state = ""
    while time.monotonic() < deadline:
        msg = await ws.receive(timeout=max(0.1, deadline - time.monotonic()))
        if msg.type != aiohttp.WSMsgType.TEXT:
            continue
        data = json.loads(msg.data)
        if data.get("type") == "init":
            state = str(data.get("state") or state)
        elif data.get("type") == "state":
            state = str(data.get("state") or state)
        if state == "listening":
            return
    raise TimeoutError("Dashboard never reached listening state.")


async def _monitor_process(samples: list[dict[str, float]], stop_event: asyncio.Event, pid: int | None) -> None:
    if pid is None or psutil is None:
        await stop_event.wait()
        return
    proc = psutil.Process(pid)
    proc.cpu_percent(None)
    while not stop_event.is_set():
        try:
            mem = proc.memory_info()
            samples.append(
                {
                    "cpu": proc.cpu_percent(None),
                    "rss_mb": round(mem.rss / (1024 * 1024), 1),
                },
            )
        except Exception:
            logger.debug('JSON state load failed', exc_info=True)
        await asyncio.sleep(0.25)


async def _run_prompt(
    ws: aiohttp.ClientWebSocketResponse,
    prompt: str,
    samples: list[dict[str, float]],
) -> dict[str, Any]:
    start = time.monotonic()
    sample_start = len(samples)
    await ws.send_json({"type": "text_input", "text": prompt})
    deadline = start + 70.0
    saw_work = False
    reply = ""
    screen = ""
    recovery = ""
    while time.monotonic() < deadline:
        msg = await ws.receive(timeout=max(0.2, deadline - time.monotonic()))
        if msg.type != aiohttp.WSMsgType.TEXT:
            continue
        data = json.loads(msg.data)
        if data.get("type") == "log":
            tag = str(data.get("tag") or "")
            message = str(data.get("message") or "").strip()
            if tag in {"speaking", "action"} and message:
                reply = message
            if tag == "info" and message.startswith("[screen]"):
                screen = message.removeprefix("[screen]").strip()
            if "recovery" in message.lower() or "ask it once more" in message.lower():
                recovery = message
        elif data.get("type") == "state":
            state = str(data.get("state") or "")
            if state in {"thinking", "speaking"}:
                saw_work = True
            elif saw_work and state == "listening":
                break

    segment = samples[sample_start:]
    return {
        "prompt": prompt,
        "elapsed_s": round(time.monotonic() - start, 2),
        "reply": reply[:260],
        "screen": screen[:220],
        "recovery": recovery[:220],
        "peak_cpu": round(max((s["cpu"] for s in segment), default=0.0), 1),
        "avg_cpu": round(mean([s["cpu"] for s in segment]), 1) if segment else 0.0,
        "peak_rss_mb": round(max((s["rss_mb"] for s in segment), default=0.0), 1),
        "passed": bool(reply) and not recovery and not _TRANSCRIPT_LABEL_RE.search(reply),
    }


async def _main(args: argparse.Namespace) -> int:
    prompts = _load_prompts(args.prompts_file)
    ws_url = _extract_ws_url(args.base_url)
    samples: list[dict[str, float]] = []
    stop_event = asyncio.Event()
    monitor_task = asyncio.create_task(_monitor_process(samples, stop_event, args.pid))
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
            async with session.ws_connect(ws_url, heartbeat=15.0) as ws:
                await _wait_until_listening(ws)
                results = []
                for prompt in prompts:
                    results.append(await _run_prompt(ws, prompt, samples))
                    await asyncio.sleep(0.3)
    finally:
        stop_event.set()
        await monitor_task

    summary = {
        "base_url": args.base_url,
        "prompt_count": len(results),
        "pass_count": sum(1 for item in results if item["passed"]),
        "fail_count": sum(1 for item in results if not item["passed"]),
        "overall_peak_cpu": round(max((s["cpu"] for s in samples), default=0.0), 1),
        "overall_peak_rss_mb": round(max((s["rss_mb"] for s in samples), default=0.0), 1),
        "results": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if summary["fail_count"] == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke-test ATOM via the web dashboard.")
    parser.add_argument("--base-url", required=True, help="Dashboard base URL, e.g. http://127.0.0.1:62397/")
    parser.add_argument("--pid", type=int, default=None, help="Optional ATOM child PID for CPU/RSS sampling.")
    parser.add_argument("--prompts-file", default=None, help="Optional newline-delimited prompt file.")
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    raise SystemExit(asyncio.run(_main(parser.parse_args())))
