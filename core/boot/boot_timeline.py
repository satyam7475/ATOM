"""Boot timeline tracker — Sprint Ω.2.

A tiny, dependency-free recorder that lets bootstrap code drop a
single timestamped event per stage and then emit one human-readable
summary line at the end of boot.

Why
---
Until now, "where did the 13 seconds go?" was a question we answered
by grepping a 200-line atomLogs.txt for INFO entries timestamped by
``%Y-%m-%d %H:%M:%S`` and doing arithmetic in our heads. With the
``BootTimeline`` singleton, every stage adds one cheap call
(``mark("stt_preload", ms=304)``) and we get one summary log line at
the end of bootstrap:

    Boot timeline: total=5380ms | tts_init=420ms stt_preload=2900ms
        embed_load=1850ms cold_start=4900ms persona_pin=80ms
        kv_restore=70ms

That makes regressions instantly obvious.

Notes
-----
- Stage names are free-form. Ordering in the summary follows insert
  order; if a stage is recorded twice we keep the first ``ms`` and
  log a debug warning.
- The timeline is intentionally process-global. Multiple boots in the
  same process (smoke tests, hot reloads) call :py:func:`reset_boot_timeline`
  to clear state.
- Total wall time is computed at ``log_summary`` from the configured
  start (``mark_boot_start``) to the time ``log_summary`` is called.
  Stage ms values are individual (per-stage durations), so sum of
  stages can exceed total when stages run concurrently.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger("atom.boot.timeline")


@dataclass(slots=True)
class _Stage:
    name: str
    ms: float
    parallel: bool = False


@dataclass(slots=True)
class _Timeline:
    start_ts: float = 0.0
    stages: list[_Stage] = field(default_factory=list)
    seen: set[str] = field(default_factory=set)


_TIMELINE = _Timeline()
_LOCK = threading.Lock()


def mark_boot_start() -> None:
    """Record the wall-clock start of boot. Call once near the top of
    ``main()`` so the final summary's ``total=`` is accurate."""
    with _LOCK:
        _TIMELINE.start_ts = time.monotonic()


def mark(name: str, ms: float, *, parallel: bool = False) -> None:
    """Record a single stage's elapsed wall time.

    ``parallel=True`` flags the stage as overlapping other work; the
    summary marks these with a ``∥`` prefix so the reader knows the
    stage doesn't add to wall time linearly. Common use: STT preload
    that runs concurrently with cold-start warmup.
    """
    if not name:
        return
    with _LOCK:
        if name in _TIMELINE.seen:
            logger.debug("BootTimeline: duplicate stage %r ignored", name)
            return
        _TIMELINE.seen.add(name)
        _TIMELINE.stages.append(_Stage(name=name, ms=float(ms), parallel=bool(parallel)))


def log_summary() -> str:
    """Emit the one-line summary and return the formatted string.

    Always logs at INFO so the timeline shows up in the standard boot
    log. Returns the same string in case the caller wants to render
    it elsewhere (status JSON, dashboard, etc).
    """
    with _LOCK:
        stages = list(_TIMELINE.stages)
        start = _TIMELINE.start_ts
    total_ms = (time.monotonic() - start) * 1000 if start else 0.0
    if not stages:
        line = f"Boot timeline: total={total_ms:.0f}ms (no stages recorded)"
        logger.info(line)
        return line
    parts = []
    for s in stages:
        prefix = "\u2225" if s.parallel else ""
        parts.append(f"{prefix}{s.name}={s.ms:.0f}ms")
    line = f"Boot timeline: total={total_ms:.0f}ms | " + " ".join(parts)
    logger.info(line)
    return line


def reset_boot_timeline() -> None:
    """Clear all recorded stages. Used by tests or hot-reload paths."""
    with _LOCK:
        _TIMELINE.start_ts = 0.0
        _TIMELINE.stages.clear()
        _TIMELINE.seen.clear()


def get_summary() -> dict:
    """Return the recorded timeline as a JSON-friendly dict (for
    dashboards / health endpoints)."""
    with _LOCK:
        stages = list(_TIMELINE.stages)
        start = _TIMELINE.start_ts
    total_ms = (time.monotonic() - start) * 1000 if start else 0.0
    return {
        "total_ms": round(total_ms, 1),
        "stages": [
            {"name": s.name, "ms": round(s.ms, 1), "parallel": s.parallel}
            for s in stages
        ],
    }
